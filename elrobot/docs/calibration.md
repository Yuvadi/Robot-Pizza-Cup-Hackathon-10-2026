# Guided manual calibration and saved profiles

Run Station in WSL, and use the same WSL instance for this tool. No third-party
Python dependencies are needed: the tool imports the installed Station client.
It reads `st3215/inference`, not the torque-gated training queue.

## Important distinctions

- Joint calibration establishes each servo's offset and normalized range.
- Camera/workplane registration maps pixels to physical coordinates. It is a
  separate calibration in [Multi-Modal-Approach](../../Multi-Modal-Approach/README.md).
- The old `calibrate.sh` invoked a six-motor SO-101 automatic routine. It now
  guides **manual calibration for both eight-motor arms** and saves profiles.
  There is no automatic motion to mechanical stops.

## Before changing calibration

Stop teleoperation, inference policies, and mirroring. Support both arms before
disabling torque; they may fall when torque is disabled. Use the correct supplies:
5 V leader, 12 V follower. Verify serials and IDs in Station, not from guesses.

**Opening Station's calibration page automatically resets that bus's calibration.**
Save existing frozen ranges first, before opening the page:

```bash
# From elrobot; both buses must be connected with torque OFF and frozen ranges.
python3 calibration.py \
  --leader-serial YOUR_LEADER_SERIAL \
  --follower-serial YOUR_FOLLOWER_SERIAL \
  --snapshot-only
```

This does not reset, freeze, enable torque, or send any hardware commands.
Unfrozen or invalid calibration, stale telemetry, wrong motor IDs, enabled torque,
or servo errors fail explicitly. If an arm is not calibrated yet, snapshot fails;
use the guided workflow to establish its ranges.

## Guided workflow

Set `LEADER_SERIAL` and `FOLLOWER_SERIAL` in the existing `elrobot.env`, then:

```bash
bash scripts/calibrate.sh
```

Or invoke the Python command above without `--snapshot-only`. For nonstandard
IDs, pass `--motor-ids` with exactly eight ascending comma-separated IDs.
The current wrapper expects both arms to share this ID ordering.

The wizard prompts for leader, then follower. All calibration actions happen in
Station's UI, not invisibly in the script:

1. Confirm the selected serial and eight motors; disable torque with support.
2. Open the calibration page for that bus. This resets its previous ranges.
3. Gently move each joint/gripper through the intended safe range. Do not force
   joints or use Auto Calibrate for this workflow.
4. Click **Save/Done** to freeze. This writes settings to servo EEPROM and saves
   ranges into Station metadata. Inspect the Station terminal for write errors.
5. Leave torque OFF, then type the exact `SAVE SERIAL` prompt in the wizard.

The tool checks fresh telemetry for all eight motors, frozen valid ranges,
zero torque, and no servo error status. It then writes:

```text
elrobot/calibrations/<UTC-session>/
  leader-<SERIAL>.json
  follower-<SERIAL>.json
```

Each file includes serial, role, motor order, frozen ranges (including wraparound
arcs), time, and observed register bytes. Earlier files are never overwritten.
If the second arm fails, the first profile remains saved and the command fails;
it does not claim both arms succeeded. Profiles stay ignored by Git.

If one arm fails because torque is still enabled, its calibration is not
necessarily lost. Support it, stop other controllers/mirroring, and turn torque
off in Station. Do not reopen the calibration page (that resets the ranges).
Retry just that arm's existing frozen profile:

```bash
bash scripts/calibrate.sh --role follower --snapshot-only
```

This creates a new timestamped session; the earlier leader profile stays intact.
`SAVE SERIAL` is input to the running wizard, not a PowerShell command.

## What is saved, and what can be restored

The JSON profiles are portable records for comparison and future adapters.
**They are not full EEPROM dumps or automatic restore commands.** The script
cannot independently prove every EEPROM write succeeded merely from frozen
range telemetry. Preserve both the motors' EEPROM and Station's `station_data`.

Do not replay a freeze command using saved centered ranges: freezing can write
a new offset and other motor settings. Do not write back the captured register
snapshot: it can contain volatile state and is not a restore image.

For a full Station-data backup, stop all control programs and Station first,
then copy its active `station.yaml` and entire `station_data` directory to a new
backup folder. This can be large because it includes camera/motor history.
Such a backup still does not replace a motor EEPROM dump.

After recalibration, older datasets/policies may no longer share the same
normalized joint mapping. Revalidate or recollect demonstrations before deployment.

Hardware-free tests, from `elrobot`:

```bash
python3 -m unittest -v test_calibration
```
