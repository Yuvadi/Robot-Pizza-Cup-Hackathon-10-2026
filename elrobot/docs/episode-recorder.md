# Place Toppings on Pizza - browser episode recorder (WSL)

This is a read-only recording UI for NormaCore Station, not a robot controller.
It records actual synchronized camera/joint/action frames directly to Parquet
compatible with the repository's `transform_elrobot.py`. No History pointers
or manual export commands are required.

## Launch

Start Station and your usual supervised leader teleoperation separately. Stop
all policy processes. From `elrobot/training` in WSL:

```bash
.venv/bin/python recorder.py --root ../../data/pick-and-place-recordings --public
```

Open <http://localhost:8890> in your Windows browser.

The default recording task is **pick_and_place_cheese**. After each recording,
choose **pick_and_place_salami** or **pick_and_place_cheese** when saving.
An explicit `--task` is also offered as an additional category for compatibility
with existing recording workflows. The default upload repository remains
**yaditya/place-toppings-on-pizza**, follower **5B61032837**. The recorder displays
the dataset name **Place Toppings on Pizza**. `--public` selects
public uploads; omit it for private uploads. These choices are shown in the UI.
Use `--repo YOUR_ACCOUNT/YOUR_DATASET` to choose a different upload destination.
The example uses a new local root so existing cheese recordings are not mixed in.
The active Station configuration must pin `inference/normvla` to this follower.
Verify ascending physical motor IDs 1 through 8 and the camera order in Station.
The normvla frame format itself contains no physical bus or camera identity.
Do not switch cameras, calibrations, or followers within a dataset.

Dependencies come from the training environment. If setting up a new environment,
use its existing `uv sync` workflow; `aiohttp` is declared in `pyproject.toml`.

## Record, review, save

1. Verify the two live synchronized camera views and reset the scene.
2. Set an episode timer (2–300 seconds; default 45).
3. Click **Record new episode**, then demonstrate using the leader.
4. The timer stops capture automatically, or click **Stop & review** early.
   **Stopping capture never stops robot motion or teleoperation.**
5. Under **Review drafts**, choose the category, then click **Save episode** to
   keep the complete demonstration, or **Discard**. Save is disabled until you
   choose a category. Each draft has its own selection.
6. Reset the scene while not recording and repeat. Saved episodes can be deleted.

The saved table shows each episode's category. Saving writes that exact task
label into every raw Parquet frame and the episode metadata without changing
joint values, JPEGs, timestamps, or frame IDs. Both categories can coexist in one
dataset, but the follower and calibration must still match. Existing saved
episodes retain their original labels; they are not automatically reclassified.

Recordings use bounded 32-frame Parquet row groups, not an entire in-memory
episode. State, goal actions, source frame IDs, JPEGs, and source-relative
timestamps remain aligned. Raw camera dimensions must be 224x224, and normalized
states/actions must be in [0,1]. A gap over 0.5 seconds, stale sensors,
calibration change, or connection loss invalidates the active draft. Invalid or
interrupted drafts cannot be saved. Retrying the connection never resumes them.

Both previews use the *same synchronized normvla stream* as recording, not
independent camera streams. Station may stop producing that stream with all
follower torque disabled. A stale preview is cleared and Record is disabled.
Check Station and your existing supervised teleoperation setup; this UI never
enables torque to force the stream to resume.

## Local storage and recovery

Default location is `data/cheese-recordings/` under the repository. Override with
`--root /absolute/path`. Only one recorder may own a data directory at a time.

- `drafts/<id>/`: current/review/interrupted captures and episode metadata.
- `saved/<id>/`: kept raw Parquet and metadata; survives restarts.
- `trash/<id>/`: deleted/discarded episodes, excluded from conversion.
- `exports/<id>/`: immutable converted LeRobot snapshots.
- `conversion.log`: latest transformer output.

Delete/discard moves only the selected episode to local trash. It does not
delete Station history or remote data. Disk cleanup is manual. Restarted
recordings cannot resume; interrupted drafts are marked invalid.

**Build / save LeRobot dataset** converts all saved episodes into a fresh v2.1
snapshot at 10 FPS, with two H.264 videos per episode. It uses
`transform_elrobot.py`, preserving normalized source goals without action shifts.
The combined export includes both categories in `meta/tasks.jsonl`, with each
episode/frame assigned the corresponding task index; separate exports or folder
moves are not necessary.
Adding/deleting a saved episode invalidates the current build; upload rebuilds
it. Recording and episode edits are disabled during conversion/upload.

## Hugging Face upload

Log in **inside WSL, in this training environment**, once:

```bash
.venv/bin/hf auth login
```

Use a token with write access. Never paste tokens into the UI or source files.
**Build & upload** shows the exact destination, episode count, visibility, and a
confirmation before any upload. It uses the cached Hugging Face credentials.
The initial application launch does not upload anything.

An existing repository must match the requested public/private visibility;
the app does not change visibility automatically. Upload replaces the managed
`data/`, `videos/`, and `meta/` paths in one commit so deleted episodes do not
remain in the current dataset. It leaves unrelated repository files untouched.
Repository history and previously downloaded copies can still contain removed
episodes. Public camera images and telemetry are downloadable by anyone;
obtain consent from people visible in the images.

The HTTP server binds only to loopback, validates the Host header, and requires
a per-process token for writes. Do not expose it through a tunnel or bind it
to a public interface.

## Validation

```bash
.venv/bin/python -m unittest test_recorder test_follow test_safety
```

Tests cover timer, save/restart/delete, rejected recordings, source fidelity,
HTTP guards, upload file selection (mocked, no network upload), and conversion
back into a dataset loaded with LeRobot.
