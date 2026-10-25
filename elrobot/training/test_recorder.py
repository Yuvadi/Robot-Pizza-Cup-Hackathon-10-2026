import asyncio
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyarrow.parquet as pq
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from recorder import Recorder, create_app, decode, timers


def options():
    return SimpleNamespace(task="place cheese on pizza", repo="yaditya/place-toppings-on-pizza",
                           public=True, port=8890, bus_serial="FOLLOWER")


def row():
    data = io.BytesIO()
    Image.new("RGB", (224, 224), (150, 100, 50)).save(data, format="JPEG")
    joint = dict(range_min=1000, range_max=3000, position=2000, position_norm=0.5,
                 goal=2100, goal_norm=0.55, current_ma=0, velocity=0)
    return dict(global_frame_id=b"\x01", joints=[joint.copy() for _ in range(8)],
                images=[{"jpeg": data.getvalue()} for _ in range(2)])


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.r = Recorder(Path(self.temp.name), options())
        self.stamp = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        self.r.ingest(self.stamp, row())

    def record(self, frames=3):
        self.r.start(2)
        identity = self.r.active["id"]
        for i in range(frames):
            self.r.ingest(self.stamp + i * 100_000_000, row())
        self.r.stop()
        return identity

    def test_save_restart_delete_and_frame_fidelity(self):
        identity = self.record()
        self.r.manage("save", identity)
        saved = self.r.records("saved")[0]
        data = pq.read_table(self.r.root / "saved" / identity / "frames.parquet").to_pylist()
        self.assertEqual(saved["frames"], 3)
        self.assertEqual([r["timestamp_ns_since_episode_start"] for r in data], [0, 100_000_000, 200_000_000])
        self.assertAlmostEqual(data[0]["joints"][0]["goal_norm"], 0.55)
        self.assertEqual(data[0]["images"], row()["images"])
        other = Recorder(self.r.root, options())
        self.assertEqual(len(other.records("saved")), 1)
        other.export = Path("previous-build")
        other.manage("delete", identity)
        self.assertEqual(other.records("saved"), [])
        self.assertTrue((other.root / "trash" / identity / "frames.parquet").exists())
        self.assertIsNone(other.export)

    def test_timer_and_freshness_validation(self):
        for invalid in [0, 301, 2.5, True]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.r.start(invalid)
        self.r.latest_stamp -= 2_000_000_000
        with self.assertRaisesRegex(ValueError, "Fresh"):
            self.r.start(2)

    def test_new_collection_exact_labels_and_persistent_settings(self):
        with tempfile.TemporaryDirectory() as root:
            args = options()
            args.task = "pick-n-put-cheese"
            args.categories = ["pick-n-put-cheese", "pick-n-put-salami"]
            args.camera_order = "1,0"
            args.fps = 15
            args.image_size = 320
            recorder = Recorder(Path(root), args)
            self.assertEqual(recorder.state()["categories"], args.categories)
            self.assertEqual(recorder.state()["camera_order"], [1, 0])
            Recorder(Path(root), args)
            args.camera_order = "0,1"
            with self.assertRaisesRegex(ValueError, "settings changed"):
                Recorder(Path(root), args)

    def test_capture_gate_counts_unique_camera_timestamps_not_just_stream(self):
        self.r.args.fps = 15
        for i in range(46):
            stamp = self.stamp + round(i * 1e9 / 15)
            self.r.ingest(stamp, row(), [stamp, self.stamp])
        self.assertFalse(self.r.capture_ready())
        with patch.object(self.r, "fresh", return_value=True), \
                self.assertRaisesRegex(ValueError, "Capture rate not ready"):
            self.r.start(2)
        self.r.samples.clear()
        for i in range(46):
            stamp = self.stamp + round(i * 1e9 / 15)
            self.r.ingest(stamp, row(), [stamp, stamp])
        self.assertTrue(self.r.capture_ready())
        self.assertEqual(self.r.capture_rates(), [15., 15., 15.])
        with patch.object(self.r, "fresh", return_value=True):
            self.r.start(2)
        self.r.stop()

    def test_decoder_preserves_higher_resolution_and_reorders_camera_bytes(self):
        image = io.BytesIO()
        Image.new("RGB", (426, 320), (0, 50, 100)).save(image, format="JPEG")
        original = row()
        joints = []
        for values in original["joints"]:
            joint = Mock()
            joint.get_monotonic_stamp_ns.return_value = self.stamp
            for key, value in values.items():
                getattr(joint, "get_" + key).return_value = value
            joints.append(joint)
        cameras = [Mock(), Mock()]
        blobs = [image.getvalue(), image.getvalue() + b"extra"]
        for camera, blob in zip(cameras, blobs):
            camera.get_jpeg.return_value = blob
            camera.get_monotonic_stamp_ns.return_value = self.stamp
        frame = Mock()
        frame.get_joints.return_value = joints
        frame.get_images.return_value = cameras
        frame.get_monotonic_stamp_ns.return_value = self.stamp
        frame.get_global_frame_id.return_value = b"frame"
        _, decoded = decode(frame, self.stamp, (1, 0), 320)
        self.assertEqual(decoded["images"], [{"jpeg": blobs[1]}, {"jpeg": blobs[0]}])
        with self.assertRaisesRegex(ValueError, "shortest side"):
            decode(frame, self.stamp, (1, 0), 224)

    def test_save_category_updates_all_frames_and_survives_restart(self):
        for task in ("pick_and_place_salami", "pick_and_place_cheese"):
            self.r.latest_stamp = self.stamp = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
            identity = self.record(frames=67)
            source = self.r.root / "drafts" / identity / "frames.parquet"
            before = pq.read_table(source).to_pylist()
            self.r.manage("save", identity, task)
            after = pq.read_table(self.r.root / "saved" / identity / "frames.parquet").to_pylist()
            self.assertEqual(after, [frame | {"task": task} for frame in before])
        other = Recorder(self.r.root, options())
        self.assertEqual({episode["task"] for episode in other.records("saved")},
                         {"pick_and_place_salami", "pick_and_place_cheese"})

    def test_unknown_category_keeps_draft_unchanged(self):
        identity = self.record()
        for task in ("", "unknown", "../salami", 5):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, "category"):
                self.r.manage("save", identity, task)
        self.assertEqual(self.r.records("saved"), [])
        self.assertEqual(self.r.records("drafts")[0]["task"], options().task)

    def test_failed_relabel_preserves_original_frames_and_draft(self):
        identity = self.record()
        source = self.r.root / "drafts" / identity
        before = (source / "frames.parquet").read_bytes()
        with patch("recorder.pq.ParquetWriter", side_effect=OSError("Disk write failed")), \
                self.assertRaisesRegex(OSError, "Disk write failed"):
            self.r.manage("save", identity, "pick_and_place_salami")
        self.assertEqual((source / "frames.parquet").read_bytes(), before)
        self.assertFalse((source / "frames.relabel.partial").exists())
        self.assertEqual(self.r.records("saved"), [])

    def test_failed_episode_cannot_be_saved(self):
        self.r.start(2)
        identity = self.r.active["id"]
        self.r.stop("feed lost")
        with self.assertRaisesRegex(ValueError, "valid"):
            self.r.manage("save", identity)
        self.r.manage("discard", identity)
        self.assertEqual(self.r.records("drafts"), [])

    def test_duplicate_gap_and_calibration_rejected(self):
        self.r.start(2)
        self.r.ingest(self.stamp, row())
        for stamp, data in [(self.stamp, row()), (self.stamp + 600_000_000, row())]:
            with self.assertRaises(ValueError):
                self.r.ingest(stamp, data)
        changed = row()
        changed["joints"][0]["range_max"] = 3100
        with self.assertRaisesRegex(ValueError, "Calibration changed"):
            self.r.ingest(self.stamp + 100_000_000, changed)
        self.r.stop("invalid test")

    def test_path_traversal_rejected(self):
        for identity in ["../saved", "", "/etc/passwd"]:
            with self.assertRaises(ValueError):
                self.r.manage("delete", identity)

    def test_new_episode_cannot_mix_calibration(self):
        identity = self.record()
        self.r.manage("save", identity)
        self.r.latest_stamp = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        self.r.latest["joints"][0]["range_max"] += 1
        with self.assertRaisesRegex(ValueError, "differ"):
            self.r.start(2)

    def test_decoder_checks_freshness_and_joint_goals(self):
        data = row()
        joints = []
        for value in data["joints"]:
            j = Mock()
            j.get_monotonic_stamp_ns.return_value = self.stamp
            for key, v in value.items():
                getattr(j, "get_" + key).return_value = v
            joints.append(j)
        image = Mock()
        image.get_monotonic_stamp_ns.return_value = self.stamp
        image.get_jpeg.return_value = data["images"][0]["jpeg"]
        frame = Mock()
        frame.get_joints.return_value = joints
        frame.get_images.return_value = [image, image]
        frame.get_monotonic_stamp_ns.return_value = self.stamp
        frame.get_global_frame_id.return_value = b"\x01"
        stamp, decoded = decode(frame, self.stamp)
        self.assertEqual(decoded, data)
        with self.assertRaises(ValueError):
            decode(frame, self.stamp + 2_000_000_000)
        joints[0].get_goal_norm.return_value = -0.1
        with self.assertRaisesRegex(ValueError, "goal"):
            decode(frame, self.stamp)

    def test_upload_replaces_only_dataset_paths(self):
        api = Mock()
        api.repo_info.return_value.private = False
        with patch("huggingface_hub.HfApi", return_value=api):
            self.r.upload(Path("export"))
        api.create_repo.assert_called_once_with(options().repo, repo_type="dataset", private=False, exist_ok=True)
        self.assertEqual(api.upload_folder.call_args.kwargs["delete_patterns"], ["data/**", "videos/**", "meta/**"])
        api.repo_info.return_value.private = True
        with patch("huggingface_hub.HfApi", return_value=api), self.assertRaises(ValueError):
            self.r.upload(Path("export"))


class AsyncTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.r = Recorder(Path(self.temp.name), options())
        self.stamp = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
        self.r.ingest(self.stamp, row())

    async def test_timer_auto_stops_for_review(self):
        self.r.start(2)
        self.r.ingest(self.stamp, row())
        self.r.ingest(self.stamp + 100_000_000, row())
        self.r.active["clock"] -= 2
        with patch.object(self.r, "fresh", return_value=True):
            task = asyncio.create_task(timers(self.r))
            await asyncio.sleep(.1)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self.assertIsNone(self.r.active)
        self.assertEqual(self.r.records("drafts")[0]["status"], "review")

    async def test_converter_round_trip(self):
        categories = ("pick_and_place_salami", "pick_and_place_cheese")
        for category in categories:
            self.stamp = time.clock_gettime_ns(time.CLOCK_BOOTTIME)
            self.r.latest_stamp = self.stamp
            self.r.start(2)
            identity = self.r.active["id"]
            for i in range(4):
                self.r.ingest(self.stamp + i * 100_000_000, row())
            self.r.stop()
            self.r.manage("save", identity, category)
        result = await self.r.build()
        info = json.loads((result / "meta/info.json").read_text())
        self.assertEqual(info["total_episodes"], 2)
        self.assertEqual(info["total_frames"], 8)
        self.assertEqual(info["fps"], 10)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset("local/recorder-test", root=result, video_backend="pyav")
        tasks = [json.loads(line)["task"] for line in (result / "meta/tasks.jsonl").read_text().splitlines()]
        self.assertEqual(set(tasks), set(categories))
        self.assertEqual(info["total_tasks"], 2)
        self.assertEqual({ds[i]["task"] for i in range(len(ds))}, set(categories))
        sample = ds[0]
        self.assertEqual(tuple(sample["observation.images.camera_0"].shape), (3, 224, 224))
        self.assertEqual(tuple(sample["action"].shape), (8,))
        self.assertIn(sample["task"], categories)

    async def test_high_resolution_15fps_round_trip_and_training_preflight(self):
        args = options()
        args.task = "pick-n-put-cheese"
        args.categories = ["pick-n-put-cheese", "pick-n-put-salami"]
        args.camera_order, args.fps, args.image_size = "1,0", 15, 320
        self.r = Recorder(self.r.root / "new", args)
        blob = io.BytesIO()
        Image.new("RGB", (426, 320), (50, 100, 150)).save(blob, format="JPEG")
        data = row()
        data["images"] = [{"jpeg": blob.getvalue()} for _ in range(2)]
        base = time.clock_gettime_ns(time.CLOCK_BOOTTIME) - 3_000_000_000
        for i in range(46):
            stamp = base + round(i * 1e9 / 15)
            self.r.ingest(stamp, data, [stamp, stamp])
        with patch.object(self.r, "fresh", return_value=True):
            self.r.start(2)
        identity = self.r.active["id"]
        for i in range(4):
            stamp = base + 3_000_000_000 + round(i * 1e9 / 15)
            self.r.ingest(stamp, data, [stamp, stamp])
        self.r.stop()
        self.r.manage("save", identity, "pick-n-put-salami")
        result = await self.r.build()
        info = json.loads((result / "meta/info.json").read_text())
        self.assertEqual(info["fps"], 15)
        self.assertEqual(info["total_frames"], 4)
        self.assertEqual(info["features"]["observation.images.camera_0"]["shape"], [320, 426, 3])
        profile = json.loads((result / "meta/collection.json").read_text())
        self.assertEqual(profile["camera_order"], [1, 0])
        from train import check_dataset
        check_dataset("local/new-toppings-test", result)

    async def test_http_requires_token_and_blocks_invalid_requests(self):
        app = create_app(self.r)
        app.cleanup_ctx.clear()
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        headers = {"Host": "localhost:8890"}
        response = await client.post("/api/start", json={"seconds": 2}, headers=headers)
        self.assertEqual(response.status, 403)
        headers["X-Recorder-Token"] = self.r.token
        response = await client.post("/api/start", json={"seconds": 0}, headers=headers)
        self.assertEqual(response.status, 400)
        response = await client.post("/api/upload", json={}, headers=headers)
        self.assertEqual(response.status, 400)
        response = await client.get("/api/state", headers={"Host": "evil.test:8890"})
        self.assertEqual(response.status, 403)
        response = await client.get("/camera/0", headers=headers)
        self.assertEqual(response.status, 200)
        response = await client.post("/api/start", json={"seconds": 2}, headers=headers)
        self.assertEqual(response.status, 200)
        response = await client.post("/api/start", json={"seconds": 2}, headers=headers)
        self.assertEqual(response.status, 400)
        response = await client.post("/api/stop", json={}, headers=headers)
        self.assertEqual(response.status, 200)

    async def test_http_save_requires_category_and_persists_selection(self):
        self.r.start(2)
        identity = self.r.active["id"]
        for i in range(3):
            self.r.ingest(self.stamp + i * 100_000_000, row())
        self.r.stop()
        app = create_app(self.r)
        app.cleanup_ctx.clear()
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        headers = {"Host": "localhost:8890", "X-Recorder-Token": self.r.token}
        state = await (await client.get("/api/state", headers=headers)).json()
        self.assertIn("pick_and_place_salami", state["categories"])
        self.assertIn("pick_and_place_cheese", state["categories"])
        for payload in ({"id": identity}, {"id": identity, "task": "unknown"},
                        {"id": identity, "task": None}):
            response = await client.post("/api/save", json=payload, headers=headers)
            self.assertEqual(response.status, 400)
        response = await client.post("/api/save", json={"id": identity, "task": "pick_and_place_salami"},
                                     headers=headers)
        self.assertEqual(response.status, 200)
        self.assertEqual((await response.json())["saved"][0]["task"], "pick_and_place_salami")


if __name__ == "__main__":
    unittest.main()
