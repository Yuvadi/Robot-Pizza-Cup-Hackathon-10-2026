# Hardware troubleshooting

## `There is no status packet` during teleop

The follower bus often drops after torque turns on. Common causes:

1. **Follower PSU** — USB does not power the servos. Unplug the follower barrel jack for 5 s, replug, confirm all six motor LEDs are steady red.
2. **Loose 3-pin cable** — especially at `wrist_roll` (id 5) or `wrist_flex` (id 4). Reseat the daisy chain gripper → base.
3. **Input voltage error** — large leader/follower pose mismatch makes the follower jerk and brown out. Align both arms to a similar pose before starting teleop.
4. **USB** — plug each Waveshare board into a **direct Mac port** (not a passive hub) if both arms enumerate but teleop still flakes.

Safer teleop flags (also the defaults in `scripts/teleoperate.sh`):

```bash
--robot.num_read_retries=5 \
--teleop.num_read_retries=5 \
--robot.max_relative_target=10
```

If it still fails, power-cycle the follower and retry with `--fps=15`.

## Feetech timeout / comms error

Check the **red motor LEDs** on the daisy chain before changing software.

| What you see | Likely cause | Fix |
| --- | --- | --- |
| All LEDs steady red, gripper → base | Wiring OK | Look at USB port / permissions |
| One or more motors dark, chain stops mid-way | Loose 3-pin cable or missing power | Reseat cables; confirm PSU is plugged in |
| LEDs blinking | Motor error: overload or wrong voltage | Unforce the joint; confirm 5 V / 7.4 V vs 12 V PSU matches the arm |

SO-101 ships in **5 V / 7.4 V** and **12 V** variants. They are not interchangeable.

## Port not found / `lerobot-find-port` says "No difference was found"

`lerobot-find-port` compares `/dev` before vs after you unplug. It fails if the `tty.usbmodem*` node does not disappear (common on macOS: unplugged PSU instead of USB, unplugged the robot cable, or pressed Enter before macOS dropped the device).

Skip it. List MotorBus ports directly:

```bash
bash scripts/list_ports.sh
```

You want `/dev/tty.usbmodem...` (not `/dev/tty`, Bluetooth, or `ttys000`). Plug **one arm at a time**, copy that path into `FOLLOWER_PORT` or `LEADER_PORT`.

If you still want the official finder:

1. Plug USB **into the Mac** (Waveshare USB, not only the 5 V/12 V barrel jack).
2. Confirm `ls /dev/tty.usbmodem*` shows a device **before** you start.
3. When asked, unplug that **USB cable from the Mac**, wait ~2 seconds, then press Enter.
4. Do not unplug the power supply instead.

On Linux: `sudo chmod 666 /dev/ttyACM0` (or add your user to `dialout`). Ports change when you replug USB — update `station.env`.

A USB hub / dongle is allowed, but it must still enumerate **two** `tty.usbmodem*` (or `ttyACM*`) devices — one Waveshare MotorBus per arm. You cannot put leader and follower on the same serial port: both buses use motor ids 1–6 and would collide. If `list_ports.sh` shows only one path with both arms cabled:

- each arm has its own controller board, USB cable, and barrel-jack PSU (USB does not power the servos)
- both boards have jumpers on channel **B** (USB)
- the second USB cable is seated in the hub, not only the 3-pin servo daisy chain
- try a powered hub, or plug the second board directly into the Mac to confirm it enumerates

You should see two different ids, e.g. `tty.usbmodem5B79...` and `tty.usbmodem575E...`.

## Follower does not track the leader

- Recalibrate both arms with the **same physical mid-range pose**.
- Confirm `FOLLOWER_ID` / `LEADER_ID` match the ids used during calibration.
- Confirm you did not swap `FOLLOWER_PORT` and `LEADER_PORT`.

## Camera black / wrong feed

- Run `uv run python scripts/list_cameras.py` and pick different `CAMERA_INDEX` / `CAMERA_INDEX_TOP`. Open `/tmp/lerobot-cameras/index-*.jpg` to tell them apart.
- Quit other apps that hold the webcam (Zoom, Photo Booth, Chrome).
- Confirm the USB camera is not a phone continuity camera on macOS.
- Two USB webcams: plug them into separate ports (not one unpowered hub). If one stream stalls, swap cables or drop to 640×480 @ 30 FPS.

## `OpenCV: out device of bound (0-1): 2`

AVFoundation only has cameras `0` and `1`. Index `2` is not a webcam (ffmpeg often lists it as **Capture screen 0**).

On this station that pair is usually **USB table cam + FaceTime**. Confirm with:

```bash
system_profiler SPCameraDataType
uv run python scripts/list_cameras.py
```

The overhead camera must appear as a third named device (not FaceTime) **before** you set `CAMERA_INDEX_TOP`. Unplug/replug it, use a different USB port, and skip FaceTime for `CAMERA_INDEX`.

## Second USB camera is plugged in but not listed

macOS never published it. This station’s bus currently has **one** `icspring camera` plus FaceTime. OpenCV cannot invent index `2`.

Isolation test:

1. Quit Zoom / Photo Booth / Chrome / teleop.
2. Unplug the **working** table USB camera.
3. Plug **only** the overhead camera into a **Mac USB-C port** (USB-C adapter is fine; skip the dock/hub that already holds the first cam).
4. Run `uv run python scripts/list_cameras.py` and open `/tmp/lerobot-cameras/index-0.jpg`.
5. If index 0 is the overhead view, that camera works. Plug the table cam into a **different host port**, then list again — you need two USB names, not FaceTime.
6. If index 0 is still missing / black / the old view, the overhead cable, adapter, or camera is not enumerating. Try another cable and a powered USB 2 hub.

Two identical icspring UVC cams on one hub/dock often collapse to a single device. Different ports (or one cam on a USB-C dongle, the other on the dock) is the usual fix.

## macOS OpenCV / ffmpeg class warnings

`lerobot-record` / `lerobot-train` may print `Class AVFFrameReceiver is implemented in both ...`. Harmless duplicate dylibs from OpenCV, PyAV, and Homebrew ffmpeg. Ignore unless the camera actually fails to open.

## Recording keys do nothing

Use an interactive terminal and keep it focused. Equivalents:

- `n` or `→` — save episode, go to next
- `r` or `←` — redo current episode
- `q` or `Esc` — finish session, encode, optionally upload

Press `n` after each demo you want to keep. Quitting with `q` mid-episode discards the in-progress episode (`No episodes saved` in the log).

## MP4 won’t play in QuickTime / Finder

LeRobot’s default codec is **AV1**, which QuickTime often cannot decode. `scripts/record.sh` uses **H.264** (`h264_videotoolbox` on macOS) so new recordings play in QuickTime.

For older AV1 datasets:

- Open with [VLC](https://www.videolan.org/vlc/) or [IINA](https://iina.io/)
- Or convert: `ffmpeg -i file-000.mp4 -c:v h264_videotoolbox -b:v 2M ~/Desktop/episode.mp4`

Prefer Rerun on the **saved dataset** (not the live record window):

```bash
uv run rerun "$HOME/.cache/huggingface/lerobot/${HF_USER}/${DATASET_NAME}_<timestamp>"
# or: bash scripts/viz_dataset.sh
```

Live Rerun during `record.sh` freezes the camera on purpose (`static=True`). See [data collection](data-collection.md#so-101--inspect-and-replay).
