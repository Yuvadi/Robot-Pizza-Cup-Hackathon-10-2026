"""Review paired LeRobot v2.1 videos and export episode destination conditioning."""

import argparse
import asyncio
import hashlib
import json
import logging
import secrets
import shutil
import tempfile
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from aiohttp import web

ROOT = Path(__file__).resolve().parents[2]
DESTINATIONS = tuple(f"put-in-q{i}" for i in range(1, 5))
CAMERAS = ("observation.images.camera_0", "observation.images.camera_1")
LOG = logging.getLogger("destination-labeler")


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def latest_export(collection):
    exports = list((collection / "exports").glob("*/meta/info.json"))
    if not exports:
        raise ValueError(f"No completed LeRobot export in {collection}. Build it in the recorder first.")
    return max(exports, key=lambda path: path.stat().st_mtime_ns).parent.parent


class Labeler:
    def __init__(self, source, work):
        self.source, self.work = source.resolve(), work.resolve()
        if self.work == self.source or self.source in self.work.parents:
            raise ValueError("Labeling workspace must be outside the immutable source export.")
        self.info = json.loads((self.source / "meta/info.json").read_text())
        if self.info["codebase_version"] != "v2.1" or not self.info.get("video_path"):
            raise ValueError("Expected a video-based LeRobot v2.1 export.")
        self.episodes = read_lines(self.source / "meta/episodes.jsonl")
        if not self.episodes:
            raise ValueError("Source dataset has no episodes to label.")
        indices = [episode["episode_index"] for episode in self.episodes]
        if indices != list(range(self.info["total_episodes"])):
            raise ValueError("Expected contiguous episode indices matching dataset metadata.")
        self.tasks = {row["task_index"]: row["task"] for row in
                      read_lines(self.source / "meta/tasks.jsonl")}
        for episode in self.episodes:
            if len(episode["tasks"]) != 1 or episode["tasks"][0] not in self.tasks.values():
                raise ValueError("Each source episode must have exactly one existing task label.")
            if episode["length"] < 1:
                raise ValueError("Empty source episode.")
            for camera in CAMERAS:
                if self.info["features"][camera]["dtype"] != "video":
                    raise ValueError("Both camera features must be videos.")
                if not self.path_for(episode["episode_index"], camera).is_file():
                    raise FileNotFoundError("Missing source episode video.")
        digest = hashlib.sha256()
        for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"):
            digest.update((self.source / "meta" / name).read_bytes())
        self.identity = {"source": str(self.source), "metadata_sha256": digest.hexdigest()}
        self.work.mkdir(parents=True, exist_ok=True)
        self.labels_path = self.work / "annotations.json"
        self.labels = {}
        if self.labels_path.exists():
            saved = json.loads(self.labels_path.read_text())
            if saved["dataset"] != self.identity:
                raise ValueError("Annotations belong to another export. Use a separate --work directory.")
            self.labels = saved["destinations"]
            if any(key not in {str(i) for i in indices} or value not in DESTINATIONS
                   for key, value in self.labels.items()):
                raise ValueError("Invalid saved destination annotations.")
        self.token = secrets.token_urlsafe(32)
        self.busy = False
        self.job = None
        self.error = ""
        self.output = None

    def path_for(self, index, camera=None, root=None):
        template = self.info["video_path"] if camera else self.info["data_path"]
        base = self.source if root is None else root
        path = (base / template.format(
            episode_index=index, episode_chunk=index // self.info["chunks_size"],
            video_key=camera,
        )).resolve()
        if base not in path.parents:
            raise ValueError("Dataset path escapes its root.")
        return path

    def state(self):
        return {
            "source": str(self.source), "destinations": DESTINATIONS,
            "episodes": [episode | {"destination": self.labels.get(str(episode["episode_index"]))}
                         for episode in self.episodes],
            "labeled": len(self.labels), "busy": self.busy, "error": self.error,
            "output": str(self.output) if self.output else None,
        }

    def save(self, index, destination):
        if self.busy:
            raise ValueError("Wait for the export to finish.")
        if type(index) is not int or not 0 <= index < len(self.episodes):
            raise ValueError("Invalid episode index.")
        if destination not in DESTINATIONS:
            raise ValueError("Choose put-in-q1, put-in-q2, put-in-q3 or put-in-q4.")
        labels = self.labels | {str(index): destination}
        atomic_json(self.labels_path, {"dataset": self.identity, "destinations": labels})
        self.labels = labels
        self.output = None

    def build(self):
        if len(self.labels) != len(self.episodes):
            raise ValueError("Label every episode before building a training export.")
        exports = self.work / "exports"
        exports.mkdir(exist_ok=True)
        output = exports / uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix=".building-", dir=exports) as temporary:
            staging = Path(temporary) / "dataset"
            shutil.copytree(self.source, staging)
            tasks, episodes, annotations = {}, [], []
            stats = read_lines(staging / "meta/episodes_stats.jsonl")
            if [row["episode_index"] for row in stats] != list(range(len(self.episodes))):
                raise ValueError("Episode statistics do not match episode metadata.")
            for episode in self.episodes:
                index = episode["episode_index"]
                original = episode["tasks"][0]
                destination = self.labels[str(index)]
                combined = f"{original} {destination}"
                task_index = tasks.setdefault(combined, len(tasks))
                path = self.path_for(index, root=staging)
                table = pq.read_table(path)
                if table.num_rows != episode["length"]:
                    raise ValueError(f"Episode {index}: frame count mismatch.")
                if set(table["episode_index"].to_pylist()) != {index}:
                    raise ValueError(f"Episode {index}: incorrect frame episode indices.")
                expected = {key for key, value in self.tasks.items() if value == original}
                if not set(table["task_index"].to_pylist()).issubset(expected):
                    raise ValueError(f"Episode {index}: frame tasks disagree with episode metadata.")
                field = table.schema.field("task_index")
                table = table.set_column(table.schema.get_field_index("task_index"), field,
                                         pa.array([task_index] * table.num_rows, type=field.type))
                pq.write_table(table, path)
                stats[index]["stats"]["task_index"] = {
                    "min": [task_index], "max": [task_index], "mean": [task_index],
                    "std": [0.], "count": [table.num_rows],
                }
                episodes.append(episode | {"tasks": [combined]})
                annotations.append({
                    "episode_index": index, "original_task": original,
                    "destination": destination, "task": combined,
                })
            write_lines(staging / "meta/tasks.jsonl",
                        [{"task_index": value, "task": key} for key, value in tasks.items()])
            write_lines(staging / "meta/episodes.jsonl", episodes)
            write_lines(staging / "meta/episodes_stats.jsonl", stats)
            write_lines(staging / "meta/episode_labels.jsonl", annotations)
            atomic_json(staging / "meta/info.json", self.info | {"total_tasks": len(tasks)})
            atomic_json(staging / "meta/destination_labeling.json", self.identity | {
                "task_format": "{original_task} {destination}",
                "labels": list(DESTINATIONS),
            })
            staging.rename(output)
        return output

    async def export(self):
        self.error = ""
        try:
            self.output = await asyncio.to_thread(self.build)
        except Exception as error:
            LOG.exception("Destination export failed")
            self.error = str(error)
        finally:
            self.busy = False


def create_app(labeler, port):
    @web.middleware
    async def guard(request, handler):
        if request.host not in (f"localhost:{port}", f"127.0.0.1:{port}"):
            raise web.HTTPForbidden(text="Loopback host required.")
        if request.method != "GET" and request.headers.get("X-Labeler-Token") != labeler.token:
            raise web.HTTPForbidden(text="Missing local labeler token.")
        try:
            return await handler(request)
        except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
            return web.json_response({"error": str(error)}, status=400)
        except OSError as error:
            LOG.exception("Labeling operation failed")
            return web.json_response({"error": str(error)}, status=500)

    app = web.Application(middlewares=[guard], client_max_size=4096)

    async def index(request):
        html = Path(__file__).with_suffix(".html").read_text().replace("__TOKEN__", labeler.token)
        return web.Response(text=html, content_type="text/html", headers={"Cache-Control": "no-store"})

    async def state(request):
        return web.json_response(labeler.state(), headers={"Cache-Control": "no-store"})

    async def video(request):
        index, camera = int(request.match_info["index"]), int(request.match_info["camera"])
        if not 0 <= index < len(labeler.episodes) or camera not in (0, 1):
            raise ValueError("Invalid episode or camera.")
        return web.FileResponse(labeler.path_for(index, CAMERAS[camera]))

    async def save(request):
        data = await request.json()
        labeler.save(data["episode_index"], data["destination"])
        return web.json_response(labeler.state())

    async def build(request):
        if labeler.busy or len(labeler.labels) != len(labeler.episodes):
            raise ValueError("Finish labeling all episodes and wait for any current export.")
        labeler.busy = True
        labeler.job = asyncio.create_task(labeler.export())
        return web.json_response(labeler.state())

    async def cleanup(app):
        if labeler.job:
            await labeler.job

    app.on_cleanup.append(cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/api/state", state)
    app.router.add_get("/video/{index}/{camera}", video)
    app.router.add_post("/api/label", save)
    app.router.add_post("/api/build", build)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=ROOT / "data/place-toppings-on-pizza")
    parser.add_argument("--source", type=Path, help="Explicit immutable LeRobot v2.1 export.")
    parser.add_argument("--work", type=Path, help="Persistent annotation/export workspace.")
    parser.add_argument("--port", type=int, default=8892)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port.")
    source = args.source or latest_export(args.collection)
    work = args.work or args.collection / "destination-labeling" / source.name
    work.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (work / ".labeler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another labeler owns this workspace.")
        labeler = Labeler(source, work)
        logging.basicConfig(level=logging.INFO)
        print(f"Reviewing {len(labeler.episodes)} episodes from {labeler.source}", flush=True)
        web.run_app(create_app(labeler, args.port), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
