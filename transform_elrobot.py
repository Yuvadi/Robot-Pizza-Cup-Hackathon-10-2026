#!/usr/bin/env python3
# Convert a folder of NormaCore robot recordings into a LeRobot v2 dataset.
#
# INPUT STRUCTURE
# The input folder is searched recursively for .parquet files, in sorted order.
# Each file contains synchronized samples with these columns:
#   episode_start_ns: uint64, episode identifier (UNIX nanoseconds).
#   global_frame_id: binary, little-endian Station frame identifier.
#   timestamp_ns_since_episode_start: uint64, elapsed nanoseconds in the episode.
#   joints: list<struct<range_min:uint32, range_max:uint32, position:uint32,
#       position_norm:float32, goal:uint32, goal_norm:float32,
#       current_ma:uint32, velocity:uint32>>. Joint list order is preserved.
#   images: list<struct<jpeg:binary>>, one RGB JPEG per camera in camera order.
#   task: string, the natural-language instruction for that sample.
# The joint and camera counts are inferred, not fixed. Each camera can have its
# own resolution. Counts, joint order, camera order, and each camera's resolution
# must remain consistent across all input files. No anatomical joint names or
# camera locations are inferred; default names are joint_N and camera_N.
# An episode is identified by (source file, episode_start_ns), so equal episode
# identifiers in different files remain separate. Files are not stitched into
# episodes; each file must contain complete episodes. Samples are time-sorted
# within each episode. All input files contribute to one output dataset.
#
# OUTPUT STRUCTURE (LeRobot v2.1 by default)
#   data/chunk-NNN/episode_NNNNNN.parquet: one file per episode, 1000 per chunk.
#     observation.state: float32 vector from joints[].position_norm.
#     action: float32 vector from joints[].goal_norm at the SAME source sample.
#     Normalized values are kept unchanged; no unit conversion or action shift.
#     timestamp: float32 elapsed seconds, frame_index: int64 (episode-local),
#     episode_index: int64, index: int64 (global), task_index: int64.
#   videos/chunk-NNN/observation.images.camera_N/episode_NNNNNN.mp4:
#     one H.264 video per camera per episode. Image pixels are not in the frame
#     Parquet files. LeRobot locates videos using meta/info.json's video_path
#     template and selects frames using each sample's timestamp.
#     Original image width/height are preserved: no resizing, cropping, padding,
#     or downscaling. Even dimensions use yuv420p; odd dimensions use yuv444p.
#     H.264 CRF 18 is lossy. --image-storage image instead stores original JPEG
#     bytes as Hugging Face Image columns and produces no MP4 files.
#   meta/info.json: version, FPS, counts, feature names/shapes, file templates.
#   meta/tasks.jsonl: {task_index, task} with dataset-wide task indices.
#   meta/episodes.jsonl: {episode_index, tasks, length} with unique episode indices.
#   meta/episodes_stats.jsonl: v2.1 per-episode min/max/mean/std/count statistics.
#     --format-version v2.0 instead writes aggregate statistics in meta/stats.json.
#     RGB statistics use [0,1], shape (3,1,1); vector statistics keep source units.
#   meta/conversion.json: source files, episode IDs, selected source row indices,
#     and timing policy, so every output frame can be traced to its input file.
#   meta/source_frames/source_NNNNNN.parquet: original non-image columns plus
#     source_row_index for each input file, retaining raw telemetry and timing.
#
# TIMING AND VALIDATION
# FPS is inferred from the median within-episode interval across all input files,
# or supplied with --fps. Each episode is sampled on a regular grid starting at
# its first timestamp. The nearest complete source row is selected (ties choose
# earlier), keeping state/action/task/images aligned. Resampling can repeat or
# skip rows; it never crosses episode/file boundaries. Statistics use all source
# RGB pixels from the selected frames, without downscaling the images.
# Input files are processed one at a time. Inconsistent dimensions or malformed
# samples fail with their file/row context instead of being silently altered.
# Existing output directories are refused. Output is built in a temporary sibling
# folder and renamed only after successful conversion and video validation.
#
# Setup: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Run:   .venv/bin/python convert_to_lerobot.py /path/to/parquet_folder /path/to/output
# Reference: https://github.com/huggingface/lerobot/tree/v0.3.3/src/lerobot/datasets

from __future__ import annotations

import argparse
import io
import json
import math
import re
import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, Image as HFImage, Sequence, Value
from PIL import Image


DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
REQUIRED = {
    "episode_start_ns", "timestamp_ns_since_episode_start", "global_frame_id",
    "joints", "images", "task",
}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def vector_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def image_stats(blobs: list[bytes]) -> dict:
    """Accumulate all pixels without retaining the decoded episode in memory."""
    total = np.zeros(3, dtype=np.float64)
    squares = total.copy()
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    pixels = 0
    for blob in blobs:
        with Image.open(io.BytesIO(blob)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255
        total += rgb.sum(axis=0)
        squares += np.square(rgb).sum(axis=0)
        low = np.minimum(low, rgb.min(axis=0))
        high = np.maximum(high, rgb.max(axis=0))
        pixels += len(rgb)
    mean = total / pixels
    return {
        key: value.reshape(3, 1, 1).tolist()
        for key, value in {
            "min": low, "max": high, "mean": mean,
            "std": np.sqrt(np.maximum(0, squares / pixels - mean**2)),
        }.items()
    } | {"count": [len(blobs)]}


def encode_video(blobs: list[bytes], path: Path, fps: int, shape: tuple) -> dict:
    """Encode explicit frame times, then check the file before committing output."""
    height, width, _ = shape
    pixel_format = "yuv444p" if height % 2 or width % 2 else "yuv420p"
    path.parent.mkdir(parents=True, exist_ok=True)
    time_base = Fraction(1, fps)
    with av.open(str(path), "w", options={"movflags": "+faststart"}) as container:
        stream = container.add_stream("libx264", rate=fps, options={"crf": "18", "preset": "medium"})
        stream.width, stream.height = width, height
        stream.pix_fmt = pixel_format
        stream.gop_size = 2
        stream.codec_context.max_b_frames = 0
        for index, blob in enumerate(blobs):
            with Image.open(io.BytesIO(blob)) as image:
                frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = index, time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        metadata = {
            "video.height": stream.height, "video.width": stream.width,
            "video.codec": stream.codec_context.name, "video.pix_fmt": stream.pix_fmt,
            "video.is_depth_map": False, "video.fps": round(float(stream.average_rate)),
            "video.channels": 3, "has_audio": False,
        }
        count = 0
        for index, frame in enumerate(container.decode(video=0)):
            if (frame.height, frame.width) != (height, width):
                raise ValueError(f"Encoded video changed the source dimensions: {path}")
            if frame.pts is None or abs(float(frame.pts * frame.time_base) - index / fps) > 1e-4:
                raise ValueError(f"Encoded video timestamps are out of sync: {path}")
            count += 1
        if count != len(blobs):
            raise ValueError(f"Encoded video has {count} frames; expected {len(blobs)}: {path}")
    return metadata


def aggregate_stats(episodes: list[dict]) -> dict:
    """Combine population moments; camera dimensions are constant throughout."""
    result = {}
    for key in episodes[0]:
        stats = [episode[key] for episode in episodes]
        weights = np.array([s["count"][0] for s in stats], dtype=np.float64)
        means = np.array([s["mean"] for s in stats])
        mean = np.average(means, axis=0, weights=weights)
        variance = np.average(
            np.array([s["std"] for s in stats]) ** 2 + (means - mean) ** 2,
            axis=0, weights=weights,
        )
        result[key] = {
            "min": np.min([s["min"] for s in stats], axis=0).tolist(),
            "max": np.max([s["max"] for s in stats], axis=0).tolist(),
            "mean": mean.tolist(), "std": np.sqrt(variance).tolist(),
            "count": [int(weights.sum())],
        }
    return result


def nearest_rows(timestamps_ns: list[int], fps: int) -> tuple[np.ndarray, np.ndarray]:
    elapsed = np.array([t - timestamps_ns[0] for t in timestamps_ns], dtype=np.int64)
    if len(elapsed) > 1 and np.any(np.diff(elapsed) <= 0):
        raise ValueError("Episode timestamps must be distinct and strictly increasing")
    times = np.arange(int(elapsed[-1]) * fps // 1_000_000_000 + 1) / fps
    grid_ns = np.rint(times * 1e9).astype(np.int64)
    right = np.searchsorted(elapsed, grid_ns).clip(0, len(elapsed) - 1)
    left = (right - 1).clip(0)
    selected = np.where(grid_ns - elapsed[left] <= elapsed[right] - grid_ns, left, right)
    return selected, times.astype(np.float32)


def validate_names(names: list[str], count: int, label: str) -> list[str]:
    if len(names) != count or len(set(names)) != count:
        raise ValueError(f"Provide exactly {count} unique {label} names")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
        raise ValueError(f"{label} names may contain only letters, digits, underscores, and hyphens")
    return names


def index_sources(source: Path) -> tuple[list[dict], list[int]]:
    """Read only episode/timestamp columns up front; avoid loading all JPEGs."""
    if not source.exists():
        raise FileNotFoundError(f"Input does not exist: {source}")
    paths = (
        sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() == ".parquet")
        if source.is_dir() else [source]
    )
    if not paths:
        raise ValueError(f"No .parquet files found in {source}")
    files, intervals = [], []
    for path in paths:
        parquet = pq.ParquetFile(path)
        missing = REQUIRED - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{path}: Missing source columns: {sorted(missing)}")
        if not parquet.metadata.num_rows:
            raise ValueError(f"{path}: Source contains no frames")
        times = parquet.read(columns=["episode_start_ns", "timestamp_ns_since_episode_start"])
        if any(column.null_count for column in times.columns):
            raise ValueError(f"{path}: Episode IDs and timestamps must not contain nulls")
        timestamps = times["timestamp_ns_since_episode_start"].to_pylist()
        groups: dict[int, list[int]] = {}
        for row_index, episode_id in enumerate(times["episode_start_ns"].to_pylist()):
            groups.setdefault(episode_id, []).append(row_index)
        for indices in groups.values():
            indices.sort(key=lambda index: timestamps[index])
            differences = [timestamps[b] - timestamps[a] for a, b in zip(indices, indices[1:])]
            if any(dt <= 0 for dt in differences):
                raise ValueError(f"{path}: Duplicate timestamps within a source episode")
            intervals.extend(differences)
        files.append({"path": path, "source_file": str(path.relative_to(source)) if source.is_dir() else path.name,
                      "num_rows": parquet.metadata.num_rows, "timestamps": timestamps, "groups": groups})
    return files, intervals


def convert(source: Path, output: Path, *, fps: int | None = None,
            format_version: str = "v2.1", robot_type: str | None = None,
            camera_names: list[str] | None = None, joint_names: list[str] | None = None,
            image_storage: str = "video") -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new output folder.")
    if format_version not in {"v2.0", "v2.1"}:
        raise ValueError("Supported format versions: v2.0 and v2.1")
    if image_storage not in {"video", "image"}:
        raise ValueError("Image storage must be video or image")
    if fps is not None and (isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0):
        raise ValueError("FPS must be a positive integer")
    files, intervals = index_sources(source)
    if fps is None:
        if not intervals:
            raise ValueError("Cannot infer FPS from single-frame episodes; supply --fps")
        fps = max(1, round(1e9 / float(np.median(intervals))))

    first = next(pq.ParquetFile(files[0]["path"]).iter_batches(batch_size=1)).to_pylist()[0]
    if not isinstance(first["joints"], list) or not isinstance(first["images"], list):
        raise ValueError(f"{files[0]['source_file']}: First row must contain joint and image lists")
    n_joints, n_cameras = len(first["joints"]), len(first["images"])
    if not n_joints:
        raise ValueError("Source has no joints")
    camera_names = validate_names(
        camera_names if camera_names is not None else [f"camera_{i}" for i in range(n_cameras)],
        n_cameras, "camera",
    )
    joint_names = validate_names(
        joint_names if joint_names is not None else [f"joint_{i}" for i in range(n_joints)],
        n_joints, "joint",
    )
    feature_info = {
        key: {"dtype": dtype, "shape": [1], "names": None}
        for key, dtype in {
            "timestamp": "float32", "frame_index": "int64", "episode_index": "int64",
            "index": "int64", "task_index": "int64",
        }.items()
    }
    for key in ("observation.state", "action"):
        feature_info[key] = {"dtype": "float32", "shape": [n_joints], "names": joint_names}
    hf_features = Features({
        key: Value(info["dtype"]) if info["shape"] == [1] and key not in {"observation.state", "action"}
        else Sequence(Value("float32"), length=n_joints)
        for key, info in feature_info.items()
    })
    camera_keys = [f"observation.images.{name}" for name in camera_names]
    image_shapes: dict[str, tuple] = {}
    tasks: dict[str, int] = {}
    episodes, episodes_stats, mappings = [], [], []
    global_index = 0
    episode_specs = [(file_index, episode_id, indices)
                     for file_index, file in enumerate(files)
                     for episode_id, indices in sorted(file["groups"].items())]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        root = Path(temporary) / "dataset"
        (root / "meta").mkdir(parents=True)
        current_file_index = None
        for episode_index, (file_index, episode_id, indices) in enumerate(episode_specs):
            file = files[file_index]
            source_times = file["timestamps"]
            if current_file_index != file_index:
                print(f"Reading {file['source_file']} ({file['num_rows']} rows)", flush=True)
                table = pq.ParquetFile(file["path"]).read()
                if any(table[name].null_count for name in REQUIRED):
                    raise ValueError(f"{file['source_file']}: Required source columns must not contain nulls")
                provenance = table.drop(["images"]).append_column(
                    "source_row_index", pa.array(range(table.num_rows), type=pa.int64()))
                provenance_path = root / f"meta/source_frames/source_{file_index:06d}.parquet"
                provenance_path.parent.mkdir(exist_ok=True)
                pq.write_table(provenance, provenance_path, compression="zstd")
                current_file_index = file_index
            rows = table.take(pa.array(indices, type=pa.int64())).to_pylist()
            # Validate every source sample, including rows resampling may skip.
            for source_index, row in zip(indices, rows):
                try:
                    if len(row["joints"]) != n_joints or len(row["images"]) != n_cameras:
                        raise ValueError("Joint/camera count changed")
                    values = [j[field] for j in row["joints"] for field in ("position_norm", "goal_norm")]
                    if any(value is None for value in values) or not np.isfinite(values).all():
                        raise ValueError("Joint state/action contains null or non-finite values")
                    if not isinstance(row["task"], str) or not row["task"].strip():
                        raise ValueError("Task must be a non-empty string")
                    for key, image in zip(camera_keys, row["images"]):
                        with Image.open(io.BytesIO(image["jpeg"])) as decoded:
                            if decoded.format != "JPEG":
                                raise ValueError("Expected JPEG camera bytes")
                            if decoded.mode != "RGB":
                                raise ValueError("Expected RGB camera JPEG (no implicit color conversion)")
                            decoded.load()
                            shape = (decoded.height, decoded.width, 3)
                        if key in image_shapes and image_shapes[key] != shape:
                            raise ValueError(f"Camera dimensions changed for {key}: expected {image_shapes[key]}, got {shape}. Split incompatible recordings into separate datasets; images are never resized.")
                        image_shapes[key] = shape
                except (KeyError, TypeError, ValueError, OSError) as error:
                    raise ValueError(f"Invalid source row {source_index} in {file['source_file']}: {error}") from error
            selected, timestamps = nearest_rows([row["timestamp_ns_since_episode_start"] for row in rows], fps)
            chosen = [rows[int(i)] for i in selected]
            length = len(chosen)
            for row in chosen:
                tasks.setdefault(row["task"], len(tasks))
            data = {
                "timestamp": timestamps,
                "frame_index": np.arange(length, dtype=np.int64),
                "episode_index": np.full(length, episode_index, dtype=np.int64),
                "index": np.arange(global_index, global_index + length, dtype=np.int64),
                "task_index": np.array([tasks[row["task"]] for row in chosen], dtype=np.int64),
                "observation.state": np.array([[j["position_norm"] for j in row["joints"]] for row in chosen], dtype=np.float32),
                "action": np.array([[j["goal_norm"] for j in row["joints"]] for row in chosen], dtype=np.float32),
            }
            print(f"Episode {episode_index}: computing statistics and encoding {length} frames at native resolution", flush=True)
            stats = {key: vector_stats(value) for key, value in data.items()}
            for camera_index, key in enumerate(camera_keys):
                blobs = [row["images"][camera_index]["jpeg"] for row in chosen]
                stats[key] = image_stats(blobs)
                feature_info[key] = {"dtype": image_storage, "shape": list(image_shapes[key]),
                                     "names": ["height", "width", "channels"]}
                if image_storage == "video":
                    video_path = root / VIDEO_PATH.format(
                        episode_chunk=episode_index // 1000, video_key=key, episode_index=episode_index)
                    feature_info[key]["info"] = encode_video(blobs, video_path, fps, image_shapes[key])
                else:
                    data[key] = [{"bytes": blob, "path": None} for blob in blobs]
                    hf_features[key] = HFImage()
            parquet_path = root / DATA_PATH.format(episode_chunk=episode_index // 1000, episode_index=episode_index)
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            Dataset.from_dict(data, features=hf_features).to_parquet(str(parquet_path))
            episodes.append({"episode_index": episode_index,
                             "tasks": list(dict.fromkeys(row["task"] for row in chosen)), "length": length})
            episodes_stats.append({"episode_index": episode_index, "stats": stats})
            mappings.append({"episode_index": episode_index, "source_episode_start_ns": episode_id,
                             "source_file": file["source_file"], "source_file_index": file_index,
                             "source_rows": len(rows), "output_frames": length,
                             "source_row_indices": [indices[int(i)] for i in selected],
                             "max_source_gap_seconds": max(
                                 (source_times[b] - source_times[a] for a, b in zip(indices, indices[1:])),
                                 default=0) / 1e9})
            global_index += length
            print(f"Episode {episode_index}: {len(rows)} source rows -> {length} frames at {fps} FPS")
        info = {
            "codebase_version": format_version, "robot_type": robot_type,
            "total_episodes": len(episodes), "total_frames": global_index,
            "total_tasks": len(tasks), "total_videos": len(episodes) * n_cameras if image_storage == "video" else 0,
            "total_chunks": math.ceil(len(episodes) / 1000), "chunks_size": 1000,
            "fps": fps, "splits": {"train": f"0:{len(episodes)}"},
            "data_path": DATA_PATH, "video_path": VIDEO_PATH if image_storage == "video" and n_cameras else None,
            "features": feature_info,
        }
        write_json(root / "meta/info.json", info)
        write_jsonl(root / "meta/tasks.jsonl", [{"task_index": i, "task": task} for task, i in tasks.items()])
        write_jsonl(root / "meta/episodes.jsonl", episodes)
        if format_version == "v2.1":
            write_jsonl(root / "meta/episodes_stats.jsonl", episodes_stats)
        else:
            write_json(root / "meta/stats.json", aggregate_stats([ep["stats"] for ep in episodes_stats]))
        write_json(root / "meta/conversion.json", {
            "source": str(source), "source_rows": sum(file["num_rows"] for file in files),
            "source_files": [{"source_file": file["source_file"], "source_rows": file["num_rows"],
                              "provenance_path": f"meta/source_frames/source_{i:06d}.parquet"}
                             for i, file in enumerate(files)],
            "image_storage": image_storage,
            "timing": "Nearest complete source row on a regular grid; ties choose earlier; origin is first source timestamp",
            "joint_mapping": {"observation.state": "joints[].position_norm", "action": "joints[].goal_norm"},
            "camera_mapping": {key: f"images[{i}].jpeg" for i, key in enumerate(camera_keys)},
            "episodes": mappings,
        })
        root.rename(output)
    print(f"Saved {format_version}: {len(episodes)} episodes, {global_index} frames to {output}")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NormaCore Parquet to LeRobot v2 with MP4 camera videos.")
    parser.add_argument("input_folder", type=Path, help="Folder containing source Parquet files (searched recursively)")
    parser.add_argument("output_folder", type=Path, help="New folder for the combined LeRobot dataset")
    parser.add_argument("--image-storage", choices=["video", "image"], default="video",
                        help="Camera storage: MP4 videos (default) or embedded JPEGs")
    parser.add_argument("--fps", type=int, help="Positive integer FPS; default: infer from median source interval")
    parser.add_argument("--format-version", choices=["v2.0", "v2.1"], default="v2.1")
    parser.add_argument("--robot-type", help="Robot identifier; default: unspecified (source has no robot model)")
    parser.add_argument("--camera-names", nargs="+", help="Names in source camera order; default: camera_0, camera_1, ...")
    parser.add_argument("--joint-names", nargs="+", help="Names in source joint order; default: joint_0, joint_1, ...")
    args = parser.parse_args()
    try:
        if not args.input_folder.is_dir():
            raise ValueError(f"Input must be a folder: {args.input_folder}")
        convert(args.input_folder, args.output_folder, fps=args.fps, format_version=args.format_version,
                robot_type=args.robot_type, camera_names=args.camera_names, joint_names=args.joint_names,
                image_storage=args.image_storage)
    except (ValueError, OSError, pa.ArrowException, av.error.FFmpegError) as error:
        parser.exit(1, f"Conversion failed: {error}\n")


if __name__ == "__main__":
    main()

