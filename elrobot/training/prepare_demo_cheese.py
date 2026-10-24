"""Convert the pinned n4chi cheese dataset to local LeRobot v2.1 videos."""

import argparse
import io
import json
from pathlib import Path

from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

SOURCE = "n4chi/7-robopizzeria-cheese-v1"
REVISION = "82b75acbeb6f302bda8bbb43e5c06ce25cc4505e"
CAMERAS = {"observation.images.top": "observation.images.camera_0",
           "observation.images.front": "observation.images.camera_1"}


def joint_values(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (8,) or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected eight finite joint values")
    if np.any(result < 0) or np.any(result > 1):
        raise ValueError(f"{name}: normalized values must be in [0, 1]")
    return result


def image_values(value):
    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError("Expected an image embedded in Parquet, not an external image path")
    with Image.open(io.BytesIO(value["bytes"])) as source:
        if source.size != (224, 224):
            raise ValueError(f"Unexpected image size: {source.size}")
        return np.array(source.convert("RGB"), copy=True)


def convert(root: Path):
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite {root}")
    source = Path(snapshot_download(SOURCE, repo_type="dataset", revision=REVISION))
    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != 10:
        raise ValueError("Unexpected source version or frame rate")
    tasks = {row["task_index"]: row["task"]
             for row in pq.read_table(source / "meta/tasks.parquet").to_pylist()}
    features = {
        name: info["features"][name] for name in ("observation.state", "action")
    }
    for target in CAMERAS.values():
        features[target] = {
            "dtype": "video", "shape": (224, 224, 3), "names": ["height", "width", "channels"]
        }
    dataset = LeRobotDataset.create(
        repo_id="local/demo-cheese-data", root=root, fps=10,
        robot_type="elrobot_station", features=features, use_videos=True, video_backend="pyav",
    )
    episode, frame_index, total = 0, 0, 0
    for path in sorted((source / "data").glob("*/*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                current = int(row["episode_index"])
                if current != episode:
                    if current != episode + 1 or frame_index == 0:
                        raise ValueError("Episodes must be contiguous, ascending and nonempty")
                    dataset.save_episode()
                    print(f"Saved episode {episode}: {frame_index} frames", flush=True)
                    episode, frame_index = current, 0
                if int(row["frame_index"]) != frame_index or int(row["index"]) != total:
                    raise ValueError("Source frame/global indices are not contiguous")
                if abs(float(row["timestamp"]) - frame_index / 10) > 1e-4:
                    raise ValueError("Source timestamps do not follow the expected 10 FPS timeline")
                frame = {name: joint_values(row[name], name) for name in features if name not in CAMERAS.values()}
                frame.update({target: image_values(row[name]) for name, target in CAMERAS.items()})
                dataset.add_frame(frame, task=tasks[int(row["task_index"])], timestamp=frame_index / 10)
                frame_index += 1
                total += 1
    if not frame_index:
        raise ValueError("Dataset has no frames")
    dataset.save_episode()
    if total != info["total_frames"] or episode + 1 != info["total_episodes"]:
        raise ValueError("Converted counts do not match source metadata")
    (root / "conversion_provenance.json").write_text(json.dumps({
        "source_repo": SOURCE, "source_revision": REVISION,
        "camera_mapping": CAMERAS, "total_frames": total, "total_episodes": episode + 1,
        "joint_mapping": "Original normalized slots preserved; physical motor IDs unverified.",
        "source_timestamps": "Original source.timestamp fields remain in the pinned Hub source.",
    }, indent=2) + "\n")
    print(f"Converted {episode + 1} episodes / {total} frames into {root}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local-datasets/demo-cheese-data"))
    convert(parser.parse_args().output.resolve())
