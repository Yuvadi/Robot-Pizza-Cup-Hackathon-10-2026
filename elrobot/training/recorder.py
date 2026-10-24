"""Local, read-only Station episode recorder. Run with the training WSL Python."""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import math
import secrets
import sys
import time
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from aiohttp import web
from PIL import Image

from safety import Joint, ensure_fresh, span, verify_station_config

ROOT = Path(__file__).resolve().parents[2]
TASK_CATEGORIES = ("pick_and_place_salami", "pick_and_place_cheese")
JOINT_FIELDS = ("range_min", "range_max", "position", "position_norm", "goal", "goal_norm", "current_ma", "velocity")
SCHEMA = pa.schema([
    ("episode_start_ns", pa.uint64()), ("global_frame_id", pa.binary()),
    ("timestamp_ns_since_episode_start", pa.uint64()),
    ("joints", pa.list_(pa.struct([
        (key, pa.float32() if key.endswith("_norm") else pa.uint32()) for key in JOINT_FIELDS
    ]))),
    ("images", pa.list_(pa.struct([("jpeg", pa.binary())]))), ("task", pa.string()),
])
LOG = logging.getLogger("recorder")


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def decode(frame, now_ns: int) -> tuple[int, dict]:
    joints, images = frame.get_joints(), frame.get_images()
    if len(joints) != 8 or len(images) != 2:
        raise ValueError("Need exactly 8 follower joints and 2 cameras.")
    stamp = frame.get_monotonic_stamp_ns()
    ensure_fresh(stamp, [v.get_monotonic_stamp_ns() for v in [*joints, *images]], now_ns, 1.0)
    values = [{key: getattr(j, f"get_{key}")() for key in JOINT_FIELDS} for j in joints]
    for value in values:
        span(Joint(value["position"], value["position_norm"], value["range_min"], value["range_max"]))
        if not math.isfinite(value["goal_norm"]) or not 0 <= value["goal_norm"] <= 1:
            raise ValueError("Recorded goal is outside [0, 1]; check calibration.")
    blobs = [bytes(image.get_jpeg()) for image in images]
    for blob in blobs:
        with Image.open(io.BytesIO(blob)) as image:
            if image.size != (224, 224):
                raise ValueError(f"Expected training images 224x224, got {image.size}.")
            image.verify()
    identity = bytes(frame.get_global_frame_id())
    if not identity:
        raise ValueError("Missing Station frame identity.")
    return stamp, {"global_frame_id": identity, "joints": values, "images": [{"jpeg": b} for b in blobs]}


class Recorder:
    def __init__(self, root: Path, args):
        self.root, self.args = root.resolve(), args
        for name in ("saved", "drafts", "trash", "exports"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.latest = None
        self.latest_stamp = 0
        self.connection = "Waiting for Station"
        self.error = ""
        self.active = None
        self.writer = None
        self.buffer = []
        self.busy = False
        self.job = None
        self.job_status = ""
        self.export = None
        self.token = secrets.token_urlsafe(32)

    def records(self, category: str) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted((self.root / category).glob("*/episode.json"))]

    def state(self) -> dict:
        return {
            "connection": self.connection, "fresh": self.fresh(), "error": self.error,
            "active": self.active, "saved": self.records("saved"), "drafts": self.records("drafts"),
            "busy": self.busy, "job": self.job_status, "export": str(self.export) if self.export else None,
            "task": self.args.task, "repo": self.args.repo, "public": self.args.public,
            "categories": list(dict.fromkeys((*TASK_CATEGORIES, self.args.task))),
            "elapsed": (time.monotonic() - self.active["clock"]) if self.active else 0,
        }

    def fresh(self) -> bool:
        return self.latest is not None and 0 <= (time.clock_gettime_ns(time.CLOCK_BOOTTIME) - self.latest_stamp) / 1e9 < 1

    def idle(self) -> None:
        if self.active or self.busy:
            raise ValueError("Finish the recording or background operation first.")

    def start(self, seconds: int) -> None:
        self.idle()
        if isinstance(seconds, bool) or not isinstance(seconds, int) or not 2 <= seconds <= 300:
            raise ValueError("Episode timer must be an integer from 2 to 300 seconds.")
        if not self.fresh():
            raise ValueError("Fresh synchronized cameras/joints required. Check Station and teleoperation.")
        calibration = [[j["range_min"], j["range_max"]] for j in self.latest["joints"]]
        for episode in self.records("saved"):
            if (episode["calibration"] != calibration
                    or episode["follower_serial"] != self.args.bus_serial):
                raise ValueError("Follower or calibration differs from saved episodes. Use a new --root.")
        identity = uuid.uuid4().hex
        folder = self.root / "drafts" / identity
        folder.mkdir()
        metadata = {
            "id": identity, "task": self.args.task, "started_ns": time.time_ns(),
            "timer": seconds, "frames": 0, "duration": 0, "status": "recording",
            "follower_serial": self.args.bus_serial, "motor_ids": list(range(1, 9)),
            "camera_order": [0, 1], "calibration": calibration,
        }
        self.writer = pq.ParquetWriter(folder / "frames.partial", SCHEMA, compression="zstd")
        atomic_json(folder / "episode.json", metadata)
        self.active = metadata | {"clock": time.monotonic(), "first_stamp": None, "last_stamp": None}
        self.buffer = []
        self.error = ""

    def ingest(self, stamp: int, row: dict) -> None:
        self.latest, self.latest_stamp = row, stamp
        if not self.active:
            return
        a = self.active
        if a["last_stamp"] is not None and stamp <= a["last_stamp"]:
            raise ValueError("Duplicate or reversed recording timestamp.")
        if a["last_stamp"] is not None and stamp - a["last_stamp"] > 500_000_000:
            raise ValueError("Recording gap exceeds 0.5 seconds; discard and retry after checking Station.")
        if [[j["range_min"], j["range_max"]] for j in row["joints"]] != a["calibration"]:
            raise ValueError("Calibration changed during recording.")
        if a["first_stamp"] is None:
            a["first_stamp"] = stamp
        a["last_stamp"] = stamp
        a["duration"] = (stamp - a["first_stamp"]) / 1e9
        a["frames"] += 1
        self.buffer.append(row | {
            "episode_start_ns": a["started_ns"], "task": a["task"],
            "timestamp_ns_since_episode_start": stamp - a["first_stamp"],
        })
        if len(self.buffer) >= 32:
            self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.writer.write_table(pa.Table.from_pylist(self.buffer, schema=SCHEMA))
            self.buffer = []

    def stop(self, failure: str = "") -> None:
        if not self.active:
            return
        a = self.active
        folder = self.root / "drafts" / a["id"]
        try:
            self.flush()
            self.writer.close()
            (folder / "frames.partial").replace(folder / "frames.parquet")
            a["status"] = "invalid" if failure or a["frames"] < 2 else "review"
            a["error"] = failure or ("Need at least two frames." if a["frames"] < 2 else "")
            atomic_json(folder / "episode.json", {k: v for k, v in a.items() if k not in ("clock", "first_stamp", "last_stamp")})
        finally:
            self.writer = None
            self.active = None
            self.buffer = []
        if failure:
            self.error = failure

    def manage(self, action: str, identity: str, task: str | None = None) -> None:
        self.idle()
        if not isinstance(identity, str) or len(identity) != 32 or any(c not in "0123456789abcdef" for c in identity):
            raise ValueError("Invalid episode identifier.")
        category = "saved" if action == "delete" else "drafts"
        source = self.root / category / identity
        metadata = json.loads((source / "episode.json").read_text())
        if action == "save":
            selected_task = self.args.task if task is None else task
            if selected_task not in (*TASK_CATEGORIES, self.args.task):
                raise ValueError("Choose a supported episode category.")
            if metadata["status"] not in ("review", "saved"):
                raise ValueError("Only completed, valid episodes can be saved. Discard interrupted recordings.")
            with pq.ParquetFile(source / "frames.parquet") as parquet:
                if parquet.metadata.num_rows != metadata["frames"]:
                    raise ValueError("Episode frame count does not match its metadata.")
            if selected_task != metadata["task"]:
                temporary = source / "frames.relabel.partial"
                try:
                    with pq.ParquetFile(source / "frames.parquet") as parquet, \
                            pq.ParquetWriter(temporary, SCHEMA, compression="zstd") as writer:
                        for batch in parquet.iter_batches(batch_size=32):
                            table = pa.Table.from_batches([batch])
                            table = table.set_column(
                                table.schema.get_field_index("task"), SCHEMA.field("task"),
                                pa.array([selected_task] * table.num_rows, type=pa.string()),
                            )
                            writer.write_table(table)
                    temporary.replace(source / "frames.parquet")
                finally:
                    temporary.unlink(missing_ok=True)
                metadata["task"] = selected_task
                atomic_json(source / "episode.json", metadata)
            # Windows/DrvFS refuses directory renames with an open Parquet reader.
            destination = self.root / "saved" / identity
            source.rename(destination)
            metadata["status"] = "saved"
            atomic_json(destination / "episode.json", metadata)
        elif action in ("discard", "delete"):
            source.rename(self.root / "trash" / identity)
        else:
            raise ValueError("Unknown episode action.")
        self.export = None

    async def build(self) -> Path:
        if not self.records("saved"):
            raise ValueError("Save at least one episode before conversion.")
        output = self.root / "exports" / uuid.uuid4().hex
        self.job_status = "Converting saved episodes to LeRobot v2.1..."
        process = await asyncio.create_subprocess_exec(
            sys.executable, str(ROOT / "transform_elrobot.py"), str(self.root / "saved"), str(output),
            "--fps", "10", "--robot-type", "elrobot",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        data, _ = await process.communicate()
        (self.root / "conversion.log").write_bytes(data)
        if process.returncode:
            raise RuntimeError("Conversion failed: " + data.decode(errors="replace")[-2000:])
        info = json.loads((output / "meta/info.json").read_text())
        if info["total_episodes"] != len(self.records("saved")):
            raise ValueError("Converted episode count does not match saved episodes.")
        self.export = output
        return output

    async def run_job(self, upload: bool) -> None:
        try:
            output = self.export or await self.build()
            if upload:
                self.job_status = f"Uploading to {self.args.repo}..."
                await asyncio.to_thread(self.upload, output)
                self.job_status = f"Uploaded: https://huggingface.co/datasets/{self.args.repo}"
            else:
                self.job_status = f"Saved LeRobot dataset: {output}"
        except Exception as exc:
            LOG.exception("Dataset operation failed")
            self.error = str(exc)
            self.job_status = "FAILED: " + str(exc)
        finally:
            self.busy = False

    def upload(self, output: Path) -> None:
        from huggingface_hub import HfApi

        api = HfApi()
        api.whoami()
        api.create_repo(self.args.repo, repo_type="dataset", private=not self.args.public, exist_ok=True)
        actual_private = api.repo_info(self.args.repo, repo_type="dataset").private
        if actual_private != (not self.args.public):
            raise ValueError("Existing repository visibility differs from UI settings. Change it on Hugging Face first.")
        api.upload_folder(
            repo_id=self.args.repo, repo_type="dataset", folder_path=output,
            path_in_repo="", delete_patterns=["data/**", "videos/**", "meta/**"],
            commit_message="Update recorded pizza episodes",
        )


async def station(recorder: Recorder) -> None:
    norma = ROOT / "elrobot/norma-core"
    sys.path[:0] = [str(norma), str(norma / "software/station/shared")]
    from station_py import new_station_client
    from target.gen_python.protobuf.drivers.inferences import normvla

    while True:
        client = None
        # The bundled client owns untracked connection tasks. Cancel only tasks
        # created by this connection, never HTTP handlers or conversion jobs.
        existing = asyncio.all_tasks()
        try:
            client = await asyncio.wait_for(new_station_client("localhost", LOG), 10)
            reader = client.read_from_tail("inference/normvla", offset=b"\x00", limit=0, step=1, buf_size=32)
            last = None
            while True:
                entry = await asyncio.wait_for(reader.data.get(), 2)
                if entry is None:
                    raise RuntimeError(f"Station stream ended: {reader.err}")
                frame = normvla.FrameReader(entry.Data)
                stamp, row = decode(frame, time.clock_gettime_ns(time.CLOCK_BOOTTIME))
                if stamp == last:
                    continue
                last = stamp
                recorder.ingest(stamp, row)
                recorder.connection = "Connected: synchronized follower + two cameras (read-only)"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOG.warning("Station recording/connection failed: %s", exc)
            recorder.connection = f"Station unavailable: {exc}"
            if recorder.active:
                recorder.stop(str(exc))
        finally:
            if client:
                client.close_connection()
            owned = [
                task for task in asyncio.all_tasks() - existing
                if task is not asyncio.current_task()
                and getattr(task.get_coro(), "cr_code", None) is not None
                and task.get_coro().cr_code.co_filename == str(norma / "software/station/shared/station_py/client.py")
            ]
            for task in owned:
                task.cancel()
            await asyncio.gather(*owned, return_exceptions=True)
        await asyncio.sleep(2)


async def timers(recorder: Recorder) -> None:
    while True:
        await asyncio.sleep(0.05)
        if recorder.active:
            if not recorder.fresh():
                recorder.stop("Synchronized feed went stale; recording cannot be saved.")
            elif time.monotonic() - recorder.active["clock"] >= recorder.active["timer"]:
                recorder.stop()


def create_app(recorder: Recorder) -> web.Application:
    @web.middleware
    async def guard(request, handler):
        if request.host not in (f"localhost:{recorder.args.port}", f"127.0.0.1:{recorder.args.port}"):
            raise web.HTTPForbidden(text="Loopback host required.")
        if request.method != "GET" and request.headers.get("X-Recorder-Token") != recorder.token:
            raise web.HTTPForbidden(text="Missing local recorder token.")
        try:
            return await handler(request)
        except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
            return web.json_response({"error": str(exc)}, status=400)
        except OSError as exc:
            LOG.exception("Local recording operation failed")
            recorder.error = str(exc)
            return web.json_response({"error": str(exc)}, status=500)

    app = web.Application(middlewares=[guard], client_max_size=4096)

    async def index(request):
        html = Path(__file__).with_name("recorder.html").read_text().replace("__TOKEN__", recorder.token)
        return web.Response(text=html, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def state(request):
        return web.json_response(recorder.state(), headers={"Cache-Control": "no-store"})

    async def image(request):
        index = int(request.match_info["index"])
        if index not in (0, 1) or not recorder.fresh():
            raise web.HTTPServiceUnavailable(text="No fresh synchronized camera frame")
        return web.Response(body=recorder.latest["images"][index]["jpeg"], content_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})

    async def action(request):
        data = await request.json()
        name = request.match_info["name"]
        if name == "start":
            recorder.start(data["seconds"])
        elif name == "stop":
            recorder.stop()
        elif name in ("save", "discard", "delete"):
            if name == "save" and not isinstance(data["task"], str):
                raise ValueError("Choose a supported episode category.")
            recorder.manage(name, data["id"], data["task"] if name == "save" else None)
        elif name in ("convert", "upload"):
            recorder.idle()
            if name == "upload" and data.get("confirm_repo") != recorder.args.repo:
                raise ValueError("Upload needs explicit repository confirmation.")
            recorder.busy = True
            recorder.error = ""
            recorder.job = asyncio.create_task(recorder.run_job(name == "upload"))
        else:
            raise ValueError("Unknown action.")
        return web.json_response(recorder.state())

    async def lifecycle(app):
        feed = asyncio.create_task(station(recorder))
        timer = asyncio.create_task(timers(recorder))
        yield
        for task in (feed, timer):
            task.cancel()
        await asyncio.gather(feed, timer, return_exceptions=True)
        recorder.stop("Server stopped during recording.")
        if recorder.job:
            await recorder.job

    app.router.add_get("/", index)
    app.router.add_get("/api/state", state)
    app.router.add_get("/camera/{index}", image)
    app.router.add_post("/api/{name}", action)
    app.cleanup_ctx.append(lifecycle)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/cheese-recordings")
    parser.add_argument("--task", default="pick_and_place_cheese",
                        help="Default task; also offered alongside salami/cheese when saving a draft.")
    parser.add_argument("--repo", default="yaditya/place-toppings-on-pizza")
    parser.add_argument("--public", action="store_true", help="Create public uploads; otherwise private.")
    parser.add_argument("--bus-serial", default="5B61032837")
    parser.add_argument("--port", type=int, default=8890)
    args = parser.parse_args()
    if not args.task.strip() or not 1 <= args.port <= 65535:
        parser.error("Non-empty task and valid port required.")
    from huggingface_hub.utils import validate_repo_id
    validate_repo_id(args.repo)
    config = yaml.safe_load((ROOT / "elrobot/norma-core/bin/station.yaml").read_text())
    verify_station_config(config, args.bus_serial, "inference/normvla")
    recorder = Recorder(args.root, args)
    import fcntl
    lock = (recorder.root / ".recorder.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another recorder is already using this data directory.")
    # Interrupted drafts are never eligible for Save.
    for metadata in recorder.records("drafts"):
        if metadata["status"] == "recording":
            metadata.update(status="invalid", error="Interrupted server session; discard this draft.")
            atomic_json(recorder.root / "drafts" / metadata["id"] / "episode.json", metadata)
        elif metadata["status"] == "saved":
            metadata.update(status="review", error="Previous Save did not complete; retry Save or Discard.")
            atomic_json(recorder.root / "drafts" / metadata["id"] / "episode.json", metadata)
    # Arrow's first scalar conversion loads pandas; do it before sampling live frames.
    pa.array([0], type=pa.uint64())
    logging.basicConfig(level=logging.WARNING)
    web.run_app(create_app(recorder), host="127.0.0.1", port=args.port)
    lock.close()


if __name__ == "__main__":
    main()
