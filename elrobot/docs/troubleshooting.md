# ElRobot troubleshooting

Commands below run from `elrobot`. Start Station with `bash scripts/run.sh` and open http://localhost:8889. Camera framing: [ElRobot README — Cameras](../README.md#cameras).

## Bash scripts fail in WSL

Run scripts with Bash. From `elrobot/scripts`, use `bash install.sh`, not
`install.sh`: Bash does not search the current directory for commands by default.

An error such as `: invalid option nameset: pipefail` or `$'\r': command not found`
means the script has Windows CRLF line endings. Convert all ElRobot shell scripts,
including sourced helpers, to Unix LF endings. From `elrobot/scripts`:

```bash
sed -i 's/\r$//' ./*.sh
bash install.sh
```

The repository's `.gitattributes` keeps shell scripts LF on future Git checkouts.
When editing them on Windows, keep the editor's line-ending setting on **LF**.
WSL runs the Linux binary; USB hardware must also be made available to WSL.

## No camera stream in Station

Arms show up, but the camera icon next to **3D** stays grey (“No active cameras”) and the camera dropdown stays on **None**.

Station is not missing a config switch. `usb-video` is on by default. The camera driver asks for access once, when the backend starts, and then stays quiet if macOS says no.

### macOS camera permission is Terminal's

`run.sh` starts `norma-core/bin/station` as a child of the terminal you launched it from. macOS grants the camera to **that app** (Terminal, iTerm, …), not to a separate Station name.

1. **System Settings → Privacy & Security → Camera** — turn on the terminal app you used.
2. Restart Station:

   ```bash
   bash scripts/run.sh stop
   bash scripts/run.sh
   ```

### Which `station.yaml` the process reads

`run.sh` reads `norma-core/bin/station.yaml`, next to the binary. `elrobot/station.yaml` is a starter you copy there. This is enough:

```yaml
usb-video:
  enabled: true
  resize_target: 224
  formats: []      # empty = pick a format the camera offers
  frame-skip: 0
```

`formats: []` is automatic selection. A bad `formats` entry disables the driver. `lerobot-find-cameras` and SO-101 `station.env` indexes do not apply.

### The USB camera never finishes enumerating

Station ignores the built-in FaceTime camera. Wrist and side are two separate UVC devices.

In **System Information → Camera**, a name can appear while the device still has no video interfaces. `KD-USB Cameras` (Sonix) did this: it was listed as a USB device, but no UVC interface was attached, so Station had nothing to open. One half-enumerated dongle is not the wrist view plus the side view.

Plug the cameras into a **powered USB hub**, or straight into the Mac. Use a data cable. Unplug the camera, wait a few seconds, plug it back in, then restart Station. Quit Photo Booth, Zoom, or Chrome if they already hold the webcam. **System Information → Camera** should list two external cameras. The camera icon next to **3D** then becomes clickable.
