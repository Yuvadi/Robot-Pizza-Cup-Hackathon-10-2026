import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from aiohttp.test_utils import TestClient, TestServer

from label_destinations import CAMERAS, Labeler, atomic_json, create_app, latest_export, read_lines, write_lines


def fixture(root):
    source = root / "source"
    (source / "meta").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1", "total_episodes": 2, "total_frames": 4,
        "total_tasks": 2, "chunks_size": 1000, "fps": 10,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {key: {"dtype": "video"} for key in CAMERAS},
    }
    atomic_json(source / "meta/info.json", info)
    tasks = ["pick-n-put-cheese", "pick-n-put-salami"]
    write_lines(source / "meta/tasks.jsonl", [
        {"task_index": i, "task": task} for i, task in enumerate(tasks)])
    write_lines(source / "meta/episodes.jsonl", [
        {"episode_index": i, "length": 2, "tasks": [task]} for i, task in enumerate(tasks)])
    write_lines(source / "meta/episodes_stats.jsonl", [
        {"episode_index": i, "stats": {"task_index": {}, "action": {"mean": [.5]}}}
        for i in range(2)])
    for i in range(2):
        path = source / info["data_path"].format(episode_chunk=0, episode_index=i)
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({
            "episode_index": pa.array([i, i], type=pa.int64()),
            "task_index": pa.array([i, i], type=pa.int64()),
            "action": [[.2, .3], [.4, .5]], "timestamp": [0., .1],
        }), path)
        for camera in CAMERAS:
            video = source / info["video_path"].format(
                episode_chunk=0, episode_index=i, video_key=camera)
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"video-{i}-{camera}".encode())
    return source


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = fixture(self.root)
        self.labeler = Labeler(self.source, self.root / "labels")

    def test_labels_persist_and_invalid_values_do_not_overwrite(self):
        self.labeler.save(0, "put-in-q4")
        for index, value in ((True, "put-in-q1"), (-1, "put-in-q1"), (2, "put-in-q1"),
                             (0, "q2"), (0, None)):
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                self.labeler.save(index, value)
        other = Labeler(self.source, self.root / "labels")
        self.assertEqual(other.labels, {"0": "put-in-q4"})
        other.save(0, "put-in-q1")
        self.assertEqual(other.labels, {"0": "put-in-q1"})

    def test_export_preserves_source_videos_and_nonlabel_columns(self):
        before = {str(p.relative_to(self.source)): p.read_bytes()
                  for p in self.source.rglob("*") if p.is_file()}
        self.labeler.save(0, "put-in-q2")
        with self.assertRaisesRegex(ValueError, "every episode"):
            self.labeler.build()
        self.labeler.save(1, "put-in-q4")
        output = self.labeler.build()
        tasks = read_lines(output / "meta/tasks.jsonl")
        self.assertEqual([r["task"] for r in tasks],
                         ["pick-n-put-cheese put-in-q2", "pick-n-put-salami put-in-q4"])
        self.assertEqual(json.loads((output / "meta/info.json").read_text())["total_tasks"], 2)
        labels = read_lines(output / "meta/episode_labels.jsonl")
        self.assertEqual(labels[0]["original_task"], "pick-n-put-cheese")
        self.assertEqual(labels[0]["destination"], "put-in-q2")
        for i in range(2):
            old = pq.read_table(self.labeler.path_for(i))
            new = pq.read_table(self.labeler.path_for(i, root=output))
            self.assertEqual(old.schema, new.schema)
            for column in old.column_names:
                if column != "task_index":
                    self.assertTrue(old[column].equals(new[column]))
            task_index = new["task_index"][0].as_py()
            self.assertEqual(tasks[task_index]["task"], labels[i]["task"])
            self.assertEqual(read_lines(output / "meta/episodes.jsonl")[i]["tasks"], [labels[i]["task"]])
            for camera in CAMERAS:
                self.assertEqual(self.labeler.path_for(i, camera).read_bytes(),
                                 self.labeler.path_for(i, camera, output).read_bytes())
        self.assertEqual(before, {str(p.relative_to(self.source)): p.read_bytes()
                                 for p in self.source.rglob("*") if p.is_file()})

    def test_export_reindexes_tasks_and_stats_for_shared_source_task(self):
        episodes = read_lines(self.source / "meta/episodes.jsonl")
        episodes[1]["tasks"] = episodes[0]["tasks"]
        write_lines(self.source / "meta/episodes.jsonl", episodes)
        path = self.labeler.path_for(1)
        table = pq.read_table(path)
        table = table.set_column(1, "task_index", pa.array([0, 0], type=pa.int64()))
        pq.write_table(table, path)
        self.labeler = Labeler(self.source, self.root / "other-labels")
        self.labeler.save(0, "put-in-q1")
        self.labeler.save(1, "put-in-q1")
        output = self.labeler.build()
        self.assertEqual(json.loads((output / "meta/info.json").read_text())["total_tasks"], 1)
        self.assertEqual(pq.read_table(self.labeler.path_for(1, root=output))["task_index"].to_pylist(), [0, 0])
        self.assertEqual(read_lines(output / "meta/episodes_stats.jsonl")[1]["stats"]["task_index"],
                         {"min": [0], "max": [0], "mean": [0], "std": [0.], "count": [2]})

    def test_wrong_source_and_workspace_are_rejected(self):
        self.labeler.save(0, "put-in-q1")
        atomic_json(self.source / "meta/info.json", self.labeler.info | {"fps": 15})
        with self.assertRaisesRegex(ValueError, "another export"):
            Labeler(self.source, self.root / "labels")
        with self.assertRaisesRegex(ValueError, "outside"):
            Labeler(self.source, self.source / "labels")

    def test_failed_copy_leaves_no_completed_export(self):
        for i in range(2):
            self.labeler.save(i, "put-in-q1")
        with patch("label_destinations.shutil.copytree", side_effect=OSError("disk full")), \
                self.assertRaisesRegex(OSError, "disk full"):
            self.labeler.build()
        self.assertEqual(list((self.labeler.work / "exports").iterdir()), [])
        self.assertEqual(len(self.labeler.labels), 2)

    def test_latest_export_requires_completed_metadata(self):
        with self.assertRaisesRegex(ValueError, "No completed"):
            latest_export(self.root)
        meta = self.root / "exports" / "latest" / "meta"
        meta.mkdir(parents=True)
        (meta / "info.json").write_text("{}")
        self.assertEqual(latest_export(self.root), meta.parent)


class HttpTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.labeler = Labeler(fixture(root), root / "labels")
        self.client = TestClient(TestServer(create_app(self.labeler, 8892)))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)
        self.headers = {"Host": "localhost:8892", "X-Labeler-Token": self.labeler.token}

    async def test_http_labels_guards_and_video_ranges(self):
        response = await self.client.get("/api/state", headers=self.headers)
        self.assertEqual((await response.json())["labeled"], 0)
        response = await self.client.post("/api/label", headers={"Host": "localhost:8892"},
                                          json={"episode_index": 0, "destination": "put-in-q1"})
        self.assertEqual(response.status, 403)
        response = await self.client.get("/api/state", headers={"Host": "evil.test:8892"})
        self.assertEqual(response.status, 403)
        response = await self.client.post("/api/build", headers=self.headers, json={})
        self.assertEqual(response.status, 400)
        response = await self.client.post("/api/label", headers=self.headers,
                                          json={"episode_index": 0, "destination": "put-in-q3"})
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["labeled"], 1)
        response = await self.client.get("/video/0/0", headers=self.headers | {"Range": "bytes=0-4"})
        self.assertEqual(response.status, 206)
        self.assertEqual(await response.read(), b"video")
        response = await self.client.get("/video/0/9", headers=self.headers)
        self.assertEqual(response.status, 400)

    async def test_export_errors_are_visible_and_release_busy_state(self):
        self.labeler.busy = True
        with patch.object(self.labeler, "build", side_effect=OSError("disk full")):
            await self.labeler.export()
        self.assertFalse(self.labeler.busy)
        self.assertEqual(self.labeler.error, "disk full")
        self.assertIsNone(self.labeler.output)

    async def test_lerobot_round_trip_reads_both_labels_as_task_conditioning(self):
        from recorder import Recorder
        from test_recorder import options, row
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        root = Path(self.temp.name)
        args = options()
        args.task = "pick-n-put-cheese"
        args.categories = ["pick-n-put-cheese", "pick-n-put-salami"]
        recorder = Recorder(root / "recordings", args)
        for task in args.categories:
            stamp = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
            recorder.ingest(stamp, row())
            recorder.start(2)
            identity = recorder.active["id"]
            for i in range(3):
                recorder.ingest(stamp + i * 100_000_000, row())
            recorder.stop()
            recorder.manage("save", identity, task)
        source = await recorder.build()
        labeler = Labeler(source, root / "real-labels")
        labeler.save(0, "put-in-q1")
        labeler.save(1, "put-in-q4")
        output = labeler.build()
        dataset = LeRobotDataset("local/destination-label-test", root=output, video_backend="pyav")
        labels = read_lines(output / "meta/episode_labels.jsonl")
        for i, label in enumerate(labels):
            frame = int(dataset.episode_data_index["from"][i])
            sample = dataset[frame]
            self.assertEqual(sample["task"], label["task"])
            self.assertEqual(sample["task"], f"{label['original_task']} {label['destination']}")
            self.assertEqual(tuple(sample[CAMERAS[0]].shape), (3, 224, 224))
            self.assertEqual(tuple(sample[CAMERAS[1]].shape), (3, 224, 224))
            self.assertEqual(tuple(sample["action"].shape), (8,))


if __name__ == "__main__":
    unittest.main()
