"""Pizza kitchen: turn an order into a recipe of trained tasks and run each step to completion.

Every step runs through follow.rollout, so all freshness, calibration, range, and
max-delta checks still apply. A step ends when the arm has moved and then held still
(settle detection), or when a segment times out. The SmolVLM verifier then checks the
pizza quadrant. Safety stops are retried automatically until the order is done or
the operator presses Stop. Torque is never changed by this program.
"""

import argparse
import asyncio
import io
import json
import logging
import secrets
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from aiohttp import web
from PIL import Image

ELROBOT = Path(__file__).resolve().parents[1]
ROOT = ELROBOT.parent
TOPPINGS = ("cheese", "salami")
ORDER_ITEMS = ("cheese", "mystery", "salami")
MAX_PIECES = 20
QUADRANTS = (1, 2, 3, 4)
CORNERS = ("tl", "tr", "br", "bl")
DEFAULT_LAYOUT = {"camera": 0, "box": None, "quadrants": {"1": "tl", "2": "tr", "3": "br", "4": "bl"}}
LOG = logging.getLogger("kitchen")


@dataclass
class Step:
    topping: str
    quadrant: int
    task: str
    source: str = ""
    piece: int = 1
    status: str = "pending"
    attempts: int = 0
    recoveries: int = 0
    verification: dict | None = None


def recipe(order: dict, template: str, mystery_slot: str = "cheese", max_pieces: int = MAX_PIECES) -> list[Step]:
    """Order {"q1": {"cheese": 2, "salami": 1, "mystery": 0}, ...} -> one step per piece.

    Cheese goes first, then the mystery item, then salami on top. A quadrant may also be
    a list of item names (one piece each). The mystery item sits in the cheese or salami
    bin, so its steps reuse that bin's trained task.
    """
    if mystery_slot not in TOPPINGS:
        raise ValueError(f"mystery_slot must be one of {TOPPINGS}.")
    if not isinstance(order, dict) or set(order) - {f"q{q}" for q in QUADRANTS}:
        raise ValueError("Order keys must be q1..q4.")
    counts = {}
    for key, items in order.items():
        if isinstance(items, list):
            if set(items) - set(ORDER_ITEMS) or len(set(items)) != len(items):
                raise ValueError(f"{key}: items must be distinct names from {ORDER_ITEMS}.")
            items = {item: 1 for item in items}
        if not isinstance(items, dict) or set(items) - set(ORDER_ITEMS):
            raise ValueError(f"{key}: items must map {ORDER_ITEMS} to piece counts.")
        for item, count in items.items():
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= max_pieces:
                raise ValueError(f"{key} {item}: count must be an integer 0..{max_pieces}.")
        counts[int(key[1])] = items
    steps = []
    for item in ORDER_ITEMS:
        source = mystery_slot if item == "mystery" else item
        for q in QUADRANTS:
            for piece in range(1, counts.get(q, {}).get(item, 0) + 1):
                steps.append(Step(item, q, template.format(topping=source, quadrant=q), source, piece))
    if not steps:
        raise ValueError("The order is empty; choose at least one topping.")
    return steps


def offsets(joints) -> list[int]:
    return [(j.position - j.minimum) % 4096 for j in joints]


class SettleDetector:
    """Reports 'settled' once the arm moved away from its start pose and then held still."""

    def __init__(self, move_ticks=200, still_ticks=25, settle_seconds=2.0, clock=time.monotonic):
        self.move_ticks, self.still_ticks, self.settle_seconds = move_ticks, still_ticks, settle_seconds
        self.clock = clock
        self.start = None
        self.moved = False
        self.window = deque()

    def update(self, joints) -> bool:
        now, pose = self.clock(), offsets(joints)
        if self.start is None:
            self.start = pose
        if max(abs(a - b) for a, b in zip(pose, self.start)) > self.move_ticks:
            self.moved = True
        self.window.append((now, pose))
        while len(self.window) > 1 and now - self.window[1][0] >= self.settle_seconds:
            self.window.popleft()
        if not self.moved or now - self.window[0][0] < self.settle_seconds:
            return False
        poses = np.array([p for _, p in self.window])
        return int((poses.max(0) - poses.min(0)).max()) <= self.still_ticks

    async def __call__(self, joints, frame):
        return "settled" if self.update(joints) else None


def crop_quadrant(image: Image.Image, layout: dict, quadrant: int) -> Image.Image:
    box = layout.get("box")
    if not box:
        raise ValueError("Pizza area is not calibrated.")
    x0, y0, x1, y1 = box
    w, h = image.size
    left, top, right, bottom = x0 * w, y0 * h, x1 * w, y1 * h
    mx, my = (left + right) / 2, (top + bottom) / 2
    corner = layout["quadrants"][str(quadrant)]
    xs = (left, mx) if corner[1] == "l" else (mx, right)
    ys = (top, my) if corner[0] == "t" else (my, bottom)
    return image.crop((round(xs[0]), round(ys[0]), round(xs[1]), round(ys[1])))


def crop_box(image: Image.Image, box) -> Image.Image:
    w, h = image.size
    return image.crop((round(box[0] * w), round(box[1] * h), round(box[2] * w), round(box[3] * h)))


def valid_box(box) -> bool:
    return (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) for v in box)
            and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)


def change_score(before: Image.Image, after: Image.Image) -> float:
    size = (64, 64)
    a = np.asarray(before.convert("L").resize(size), dtype=np.float32) / 255
    b = np.asarray(after.convert("L").resize(size), dtype=np.float32) / 255
    return float(np.abs(a - b).mean())


class Verifier:
    """Before/after pixel change in the target quadrant, plus a SmolVLM yes/no gain.

    Change is the pass signal by default. SmolVLM2-500M is weak on 224px crops, so its
    gain (after minus before yes-probability) only gates when strict=True.
    """

    MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def __init__(self, min_gain=0.05, min_change=0.03, strict=False):
        self.min_gain, self.min_change, self.strict = min_gain, min_change, strict
        self.model = self.processor = None

    def load(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(self.MODEL)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()
            tok = self.processor.tokenizer
            self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in (" Yes", "Yes", " yes")]
            self.no = [tok.encode(w, add_special_tokens=False)[0] for w in (" No", "No", " no")]

    def yes_probability(self, image: Image.Image, question: str) -> float:
        import torch

        self.load()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image.convert("RGB")], return_tensors="pt")
        inputs = inputs.to("cuda", torch.bfloat16)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        yes, no = torch.logsumexp(logits[self.yes], 0), torch.logsumexp(logits[self.no], 0)
        return float(torch.sigmoid(yes - no))

    @staticmethod
    def question(topping):
        if topping == "mystery":
            return "Is there a topping on this part of the pizza? Answer yes or no."
        return f"Is there {topping} on this part of the pizza? Answer yes or no."

    def verify(self, topping, quadrant, before, after, layout) -> dict:
        crop_before, crop_after = crop_quadrant(before, layout, quadrant), crop_quadrant(after, layout, quadrant)
        change = change_score(crop_before, crop_after)
        question = self.question(topping)
        yes_before = self.yes_probability(crop_before, question)
        yes_after = self.yes_probability(crop_after, question)
        gain = yes_after - yes_before
        passed = change >= self.min_change and (not self.strict or gain >= self.min_gain)
        return {"passed": passed, "change": round(change, 4), "yes_before": round(yes_before, 3),
                "yes_after": round(yes_after, 3), "vlm_gain": round(gain, 3), "strict": self.strict,
                "min_change": self.min_change, "min_gain": self.min_gain}


def frame_images(frame, order) -> list[Image.Image]:
    images = frame.get_images() or []
    if len(images) != 2:
        raise ValueError("Station must provide exactly 2 cameras.")
    result = []
    for index in order:
        with Image.open(io.BytesIO(bytes(images[index].get_jpeg()))) as image:
            result.append(image.convert("RGB"))
    return result


def jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class Kitchen:
    def __init__(self, args, policy, verifier, open_station=None, sleep=asyncio.sleep):
        import follow

        self.follow = follow
        self.args, self.policy, self.verifier = args, policy, verifier
        self.open_station = open_station or follow.open_station
        self.sleep = sleep
        self.order_ids = [int(v) for v in args.camera_order.split(",")]
        self.ids = follow.motor_ids(args.motor_ids)
        self.token = secrets.token_urlsafe(24)
        self.layout_path = args.work / "layout.json"
        self.layout = (json.loads(self.layout_path.read_text()) if self.layout_path.exists()
                       else json.loads(json.dumps(DEFAULT_LAYOUT)))
        self.layout.setdefault("bins", {})
        self.steps: list[Step] = []
        self.status = "idle"
        self.message = ""
        self.log = deque(maxlen=300)
        self.task = None
        self.station = None
        self.latest = [None, None]
        self.io_lock = asyncio.Lock()

    def note(self, text):
        line = f"{time.strftime('%H:%M:%S')} {text}"
        self.log.append(line)
        print(line, flush=True)

    def state(self):
        return {"status": self.status, "message": self.message, "steps": [asdict(s) for s in self.steps],
                "log": list(self.log)[-80:], "layout": self.layout, "bus_serial": self.args.bus_serial,
                "verifier": self.verifier is not None, "task_template": self.args.task_template,
                "max_pieces": MAX_PIECES, "verify_pick": self.args.verify_pick}

    def save_layout(self, data):
        camera, box, quadrants = data.get("camera"), data.get("box"), data.get("quadrants")
        if camera not in (0, 1):
            raise ValueError("camera must be 0 or 1.")
        if not valid_box(box):
            raise ValueError("box must be [x0, y0, x1, y1] normalized with x0<x1, y0<y1.")
        if (not isinstance(quadrants, dict) or set(quadrants) != {"1", "2", "3", "4"}
                or sorted(quadrants.values()) != sorted(CORNERS)):
            raise ValueError("quadrants must map 1..4 to distinct tl/tr/br/bl corners.")
        self.layout.update({"camera": camera, "box": [float(v) for v in box], "quadrants": quadrants})
        self.write_layout()

    def write_layout(self):
        self.layout_path.parent.mkdir(parents=True, exist_ok=True)
        self.layout_path.write_text(json.dumps(self.layout, indent=2))

    def empty_path(self, slot):
        return self.args.work / f"empty-{slot}.png"

    def save_bin(self, data):
        """Ingredient bin region: {"slot", "camera", "box"}; box null removes it."""
        slot, camera, box = data.get("slot"), data.get("camera"), data.get("box")
        if slot not in TOPPINGS:
            raise ValueError(f"slot must be one of {TOPPINGS}.")
        self.empty_path(slot).unlink(missing_ok=True)
        if box is None:
            self.layout["bins"].pop(slot, None)
        else:
            if camera not in (0, 1) or not valid_box(box):
                raise ValueError("Bin needs camera 0/1 and a normalized [x0, y0, x1, y1] box.")
            self.layout["bins"][slot] = {"camera": camera, "box": [float(v) for v in box], "empty_ref": False}
        self.write_layout()

    async def capture_empty(self, slot):
        """Store the current look of an EMPTY bin; later steps wait while the bin matches it."""
        spec = self.layout["bins"].get(slot)
        if spec is None:
            raise ValueError(f"Draw and save the {slot} bin box first.")
        image = (await self.snapshot())[spec["camera"]]
        self.args.work.mkdir(parents=True, exist_ok=True)
        crop_box(image, spec["box"]).save(self.empty_path(slot))
        spec["empty_ref"] = True
        self.write_layout()

    def bin_crop(self, images, slot):
        spec = self.layout["bins"].get(slot)
        return None if spec is None else crop_box(images[spec["camera"]], spec["box"])

    def looks_empty(self, images, slot) -> float | None:
        """Change score vs. the empty-bin reference when it is below the threshold, else None."""
        spec = self.layout["bins"].get(slot)
        if not spec or not spec.get("empty_ref") or not self.empty_path(slot).exists():
            return None
        with Image.open(self.empty_path(slot)) as ref:
            score = change_score(ref.convert("RGB"), self.bin_crop(images, slot))
        return score if score < self.args.empty_match else None

    async def wait_for_ingredient(self, step):
        """Sends nothing while the source bin looks empty; resumes when it is refilled."""
        warned = False
        while True:
            spec = self.layout["bins"].get(step.source)
            if not spec or not spec.get("empty_ref"):
                return
            score = self.looks_empty(await self.snapshot(), step.source)
            if score is None:
                if warned:
                    self.note(f"{step.source} bin refilled; continuing.")
                return
            if not warned:
                step.status = "waiting"
                self.note(f"{step.source} bin looks EMPTY (match {score:.3f}); refill it. No commands are sent while waiting.")
                warned = True
            await self.sleep(2)

    def checks_enabled(self, step):
        return self.verifiable() or (self.args.verify_pick > 0 and step.source in self.layout["bins"])

    def check(self, step, before, after) -> dict:
        result = {"passed": True}
        if self.verifiable():
            cam = self.layout["camera"]
            result = self.verifier.verify(step.topping, step.quadrant, before[cam], after[cam], self.layout)
        if self.args.verify_pick > 0 and step.source in self.layout["bins"]:
            change = change_score(self.bin_crop(before, step.source), self.bin_crop(after, step.source))
            result["bin_change"] = round(change, 4)
            result["picked"] = change >= self.args.verify_pick
            result["passed"] = bool(result["passed"]) and result["picked"]
        return result

    async def connection(self):
        if self.station is None:
            self.station = await self.open_station(self.args, self.ids, 1)
        return self.station

    def disconnect(self):
        if self.station is not None:
            try:
                self.station[0].close_connection()
            except Exception:
                LOG.exception("Closing Station connection failed")
            self.station = None

    async def fetch(self):
        _, fetch, _ = await self.connection()
        frame = await fetch()
        try:
            self.latest = [jpeg(image) for image in frame_images(frame, self.order_ids)]
        except Exception:
            LOG.exception("Could not cache camera preview")
        return frame

    async def snapshot(self):
        """Fresh dataset-ordered camera images; raises if Station has no fresh frame."""
        async with self.io_lock:
            frame = await self.fetch()
        return frame_images(frame, self.order_ids)

    async def send(self, targets):
        _, _, send = await self.connection()
        await send(targets)

    def segment_args(self, task):
        a = self.args
        return SimpleNamespace(execute=True, rollout_seconds=a.segment_seconds, camera_order=a.camera_order,
                               motor_ids=a.motor_ids, task=task, max_delta_ticks=a.max_delta_ticks,
                               max_invalid_retries=self.follow.MAX_INVALID_RETRIES, control_fps=a.control_fps)

    async def warmup(self, task):
        frame = await self.fetch()
        batch, _ = self.follow.decode_frame(frame, self.order_ids, self.policy.config.image_features)
        batch["task"] = [task]
        await self.follow.warmup_policy(self.policy, batch)

    def verifiable(self):
        return self.verifier is not None and bool(self.layout.get("box"))

    async def run_step(self, step: Step):
        """Run one recipe step until verified. Retries forever; only cancellation (Stop) exits."""
        before = None
        failures = 0
        while True:
            try:
                await self.wait_for_ingredient(step)
                if before is None and self.checks_enabled(step):
                    before = await self.snapshot()
                step.attempts += 1
                step.status = "running"
                self.note(f"{step.task}: attempt {step.attempts} (warm-up, then rollout segment "
                          f"up to {self.args.segment_seconds:g}s)")
                async with self.io_lock:
                    await self.warmup(step.task)
                    detector = SettleDetector(self.args.move_ticks, self.args.still_ticks, self.args.settle_seconds)
                    reason = await self.follow.rollout(self.segment_args(step.task), self.policy,
                                                       self.fetch, self.send, monitor=detector)
                failures = 0
                self.note(f"{step.task}: segment ended ({reason or 'timeout'}); checking result")
                if before is None:
                    if reason == "settled":
                        step.verification = {"passed": True, "unverified": True}
                        return
                    continue
                step.status = "verifying"
                after = await self.snapshot()
                result = await asyncio.to_thread(self.check, step, before, after)
                step.verification = result
                self.note(f"{step.task}: verification {result}")
                if result["passed"]:
                    return
                self.note(f"{step.task}: not verified; continuing the same task")
            except asyncio.CancelledError:
                raise
            except Exception as error:
                failures += 1
                step.recoveries += 1
                step.status = "recovering"
                delay = min(2 * failures, 10)
                self.note(f"{step.task}: stopped by {type(error).__name__}: {error} "
                          f"-> recovery {step.recoveries}, retrying in {delay}s")
                if not isinstance(error, ValueError):
                    self.disconnect()
                self.policy.reset()
                await self.sleep(delay)

    async def run_order(self):
        try:
            self.status, self.message = "cooking", ""
            for step in self.steps:
                if step.status == "done":
                    continue
                await self.run_step(step)
                step.status = "done"
                self.note(f"{step.task}: done")
            self.status, self.message = "finished", "Order complete. Torque unchanged."
        except asyncio.CancelledError:
            for step in self.steps:
                if step.status != "done":
                    step.status = "stopped" if step.status != "pending" else "pending"
            self.status, self.message = "stopped", "Stopped by operator. No new commands; last goals may still be active."
            self.note(self.message)
            raise
        finally:
            self.policy.reset()

    def start(self, order, confirm, mystery_slot="cheese"):
        if self.task is not None and not self.task.done():
            raise ValueError("An order is already cooking; press Stop first.")
        phrase = f"COOK {self.args.bus_serial}"
        if confirm != phrase:
            raise ValueError(f"Type {phrase} to authorize robot motion.")
        self.steps = recipe(order, self.args.task_template, mystery_slot)
        self.log.clear()
        self.note("Order accepted: " + ", ".join(s.task for s in self.steps))
        if not self.verifiable():
            self.note("WARNING: verifier inactive (no pizza area calibrated); steps finish on settle only.")
        self.task = asyncio.create_task(self.run_order())
        self.task.add_done_callback(lambda t: t.cancelled() or t.exception())

    async def stop(self):
        if self.task is not None and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        self.policy.reset()

    async def inspect(self):
        if self.verifier is None or not self.layout.get("box"):
            raise ValueError("Calibrate the pizza area and enable the verifier first.")
        image = (await self.snapshot())[self.layout["camera"]]

        def run():
            return {f"q{q}": {t: round(self.verifier.yes_probability(
                crop_quadrant(image, self.layout, q), Verifier.question(t)), 3) for t in TOPPINGS}
                for q in QUADRANTS}
        return await asyncio.to_thread(run)


def create_app(kitchen: Kitchen, port: int):
    @web.middleware
    async def guard(request, handler):
        if request.host not in (f"localhost:{port}", f"127.0.0.1:{port}"):
            raise web.HTTPForbidden(text="Loopback host required.")
        if request.method != "GET" and request.headers.get("X-Kitchen-Token") != kitchen.token:
            raise web.HTTPForbidden(text="Missing local kitchen token.")
        try:
            return await handler(request)
        except (ValueError, KeyError, TypeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        except (TimeoutError, RuntimeError, OSError) as error:
            return web.json_response({"error": str(error)}, status=503)

    app = web.Application(middlewares=[guard], client_max_size=16384)
    no_store = {"Cache-Control": "no-store"}

    async def index(request):
        html = Path(__file__).with_suffix(".html").read_text().replace("__TOKEN__", kitchen.token)
        return web.Response(text=html, content_type="text/html", headers=no_store)

    async def state(request):
        return web.json_response(kitchen.state(), headers=no_store)

    async def camera(request):
        index = int(request.match_info["camera"])
        if index not in (0, 1):
            raise ValueError("camera must be 0 or 1.")
        if kitchen.task is None or kitchen.task.done():
            if not kitchen.io_lock.locked():
                try:
                    await kitchen.snapshot()
                except (TimeoutError, RuntimeError, ValueError, OSError):
                    if kitchen.latest[index] is None:
                        raise
        data = kitchen.latest[index]
        if data is None:
            raise RuntimeError("No camera frame yet. Station must be running with follower torque enabled.")
        return web.Response(body=data, content_type="image/jpeg", headers=no_store)

    async def order(request):
        data = await request.json()
        kitchen.start(data["order"], data.get("confirm", ""), data.get("mystery_slot", "cheese"))
        return web.json_response(kitchen.state())

    async def stop(request):
        await kitchen.stop()
        return web.json_response(kitchen.state())

    async def layout(request):
        kitchen.save_layout(await request.json())
        return web.json_response(kitchen.state())

    async def inspect(request):
        return web.json_response(await kitchen.inspect())

    async def bin_box(request):
        kitchen.save_bin(await request.json())
        return web.json_response(kitchen.state())

    async def bin_empty(request):
        await kitchen.capture_empty((await request.json()).get("slot"))
        return web.json_response(kitchen.state())

    async def cleanup(app):
        await kitchen.stop()
        kitchen.disconnect()

    app.on_cleanup.append(cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/api/state", state)
    app.router.add_get("/camera/{camera}", camera)
    app.router.add_post("/api/order", order)
    app.router.add_post("/api/stop", stop)
    app.router.add_post("/api/layout", layout)
    app.router.add_post("/api/inspect", inspect)
    app.router.add_post("/api/bin", bin_box)
    app.router.add_post("/api/bin/empty", bin_empty)
    return app


def main(kitchen_cls=None, description=__doc__, **defaults):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--task-template", default="place {topping} in quadrant {quadrant}")
    parser.add_argument("--bus-serial", required=True)
    parser.add_argument("--motor-ids", required=True)
    parser.add_argument("--camera-order", required=True, choices=["0,1", "1,0"])
    parser.add_argument("--max-delta-ticks", type=int, default=100)
    parser.add_argument("--control-fps", type=float, default=10)
    parser.add_argument("--segment-seconds", type=float, default=90,
                        help="Max rollout time before re-checking a step (default 90).")
    parser.add_argument("--move-ticks", type=int, default=200, help="Movement that counts as 'started'.")
    parser.add_argument("--still-ticks", type=int, default=25, help="Max joint wobble while 'settled'.")
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--no-verifier", action="store_true")
    parser.add_argument("--verify-change", type=float, default=0.03,
                        help="Min mean pixel change (0..1) in the target quadrant to count as placed.")
    parser.add_argument("--verify-strict", action="store_true",
                        help="Also require the SmolVLM yes-probability gain >= --verify-gain.")
    parser.add_argument("--verify-gain", type=float, default=0.05)
    parser.add_argument("--verify-pick", type=float, default=0.02,
                        help="Min pixel change in the source bin box to count as picked (0 disables).")
    parser.add_argument("--empty-match", type=float, default=0.03,
                        help="Bin counts as empty when it differs from the empty reference by less than this.")
    parser.add_argument("--work", type=Path, default=ROOT / "data/kitchen")
    parser.add_argument("--server", default="localhost")
    parser.add_argument("--queue", default="inference/normvla")
    parser.add_argument("--norma-core", type=Path, default=ELROBOT / "norma-core")
    parser.add_argument("--station-config", type=Path, default=ELROBOT / "norma-core/bin/station.yaml")
    parser.add_argument("--port", type=int, default=8893)
    parser.set_defaults(**defaults)
    args = parser.parse_args()

    import torch
    import yaml

    import follow

    if not 1 <= args.max_delta_ticks <= follow.MAX_DELTA_TICKS:
        parser.error(f"--max-delta-ticks must be in 1..{follow.MAX_DELTA_TICKS}.")
    if not 1 <= args.control_fps <= 30 or args.segment_seconds <= 0 or args.settle_seconds <= 0:
        parser.error("Invalid --control-fps, --segment-seconds, or --settle-seconds.")
    if args.server not in ("localhost", "127.0.0.1"):
        parser.error("Only same-host WSL Station is supported.")
    follow.motor_ids(args.motor_ids)
    follow.verify_station_config(yaml.safe_load(args.station_config.read_text()), args.bus_serial, args.queue)
    for t in TOPPINGS:
        args.task_template.format(topping=t, quadrant=1)
    if not (args.checkpoint / "config.json").is_file():
        parser.error("--checkpoint must contain config.json and model weights.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    logging.basicConfig(level=logging.WARNING)
    policy_type = json.loads((args.checkpoint / "config.json").read_text()).get("type")
    if policy_type == "smolvla":
        policy = follow.SmolVLAPolicy.from_pretrained(str(args.checkpoint)).to("cuda").eval()
    elif policy_type == "act":
        from act_checkpoint import load_act_checkpoint

        policy = load_act_checkpoint(args.checkpoint).to("cuda").eval()
    else:
        parser.error(f"Unsupported policy type {policy_type!r}.")
    follow.validate_policy(policy)
    verifier = None
    if not args.no_verifier:
        verifier = Verifier(args.verify_gain, args.verify_change, args.verify_strict)
        verifier.load()
    args.work.mkdir(parents=True, exist_ok=True)
    kitchen = (kitchen_cls or Kitchen)(args, policy, verifier)
    print(f"Kitchen ready on http://localhost:{args.port}/ (tasks like "
          f"{args.task_template.format(topping='cheese', quadrant=1)!r})", flush=True)
    web.run_app(create_app(kitchen, args.port), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
