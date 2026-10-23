#!/usr/bin/env python3
"""Print OpenCV camera indexes that actually open, and save a preview frame."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2

PREVIEW_DIR = Path("/tmp/lerobot-cameras")
BACKEND = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY


def usb_camera_names() -> list[str]:
    if platform.system() != "Darwin":
        return []
    raw = subprocess.check_output(["ioreg", "-p", "IOUSB", "-w", "0"], text=True, errors="replace")
    names: list[str] = []
    for match in re.finditer(r"\+-o ([^@\n]+)", raw):
        name = match.group(1).strip()
        if re.search(r"camera|webcam|uvc", name, re.I):
            names.append(name)
    return names


def print_macos_devices() -> None:
    if platform.system() != "Darwin":
        return
    usb = usb_camera_names()
    print(f"USB cameras on the bus ({len(usb)}): {usb or 'none'}")
    raw = subprocess.check_output(["system_profiler", "SPCameraDataType"], text=True).rstrip()
    print("macOS camera devices:")
    print(raw)
    print()


def main() -> None:
    print_macos_devices()
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    found_indexes: list[int] = []
    consecutive_misses = 0
    for index in range(10):
        cap = cv2.VideoCapture(index, BACKEND)
        ok = cap.isOpened()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok else 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok else 0
        preview = ""
        if ok:
            grabbed, frame = cap.read()
            if grabbed and frame is not None:
                path = PREVIEW_DIR / f"index-{index}.jpg"
                cv2.imwrite(str(path), frame)
                preview = f"  preview={path}"
        cap.release()
        if not ok:
            consecutive_misses += 1
            if found_indexes and consecutive_misses >= 1:
                break
            continue
        consecutive_misses = 0
        found_indexes.append(index)
        print(f"index={index}  {width}x{height}{preview}")
    if not found_indexes:
        print("No cameras opened. Unplug/replug USB, quit Zoom/Chrome, retry.")
        raise SystemExit(1)
    print(f"Open {PREVIEW_DIR}/index-*.jpg to see which camera is which.")
    print(
        f"Valid OpenCV indexes: {found_indexes}. "
        "Do not use the next integer unless a new device appears in the list above."
    )
    print("Set CAMERA_INDEX (front) and CAMERA_INDEX_TOP (overhead) in station.env.")
    print("If a preview is your face, that index is FaceTime — do not use it for front or top.")
    if platform.system() == "Darwin" and len(usb_camera_names()) < 2:
        print()
        print("Second USB camera is not on the bus (not an OpenCV index problem).")
        print("1. Unplug the working table cam. Plug ONLY the overhead cam into a Mac USB-C port.")
        print("2. Re-run this script. Preview 0 should be the overhead view.")
        print("3. Then plug the table cam into a DIFFERENT Mac port (not the same hub/dock).")
        print("4. Re-run. You need three devices: FaceTime + two USB names.")


if __name__ == "__main__":
    main()
