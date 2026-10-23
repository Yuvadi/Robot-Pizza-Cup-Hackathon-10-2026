"""Verify a LeRobot checkpoint directory looks loadable."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console

console = Console()

REQUIRED = ("config.json", "model.safetensors")


def main(checkpoint_dir: str) -> None:
    path = Path(checkpoint_dir)
    if not path.exists():
        console.print(f"[red]Checkpoint not found:[/red] {path}")
        raise SystemExit(1)

    missing = [name for name in REQUIRED if not (path / name).exists()]
    if missing:
        console.print(f"[red]Missing files in {path}:[/red] {', '.join(missing)}")
        raise SystemExit(1)

    config = json.loads((path / "config.json").read_text())
    policy_type = config.get("type") or config.get("policy_type") or "unknown"
    console.print(f"Checkpoint OK: [bold]{policy_type}[/bold] @ {path}")
    extra = [p.name for p in path.iterdir() if p.is_file()]
    console.print(f"Files: {', '.join(sorted(extra))}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        console.print("Usage: python -m training.eval <checkpoint_dir>")
        raise SystemExit(2)
    main(sys.argv[1])
