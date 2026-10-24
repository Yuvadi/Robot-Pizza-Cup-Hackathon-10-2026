"""Validate a Station-converted LeRobot v2.1 dataset, then fine-tune SmolVLA."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def check_dataset(repo_id: str, root: Path | None = None) -> None:
    dataset = LeRobotDataset(repo_id, root=root, video_backend="pyav")
    info = dataset.meta.info
    if info["codebase_version"] != "v2.1":
        raise ValueError("This environment requires a LeRobot v2.1 dataset.")
    for key in ("observation.state", "action"):
        if tuple(info["features"][key]["shape"]) != (8,):
            raise ValueError(f"{key} must contain exactly 8 joints.")
        values = np.asarray(dataset.hf_dataset.with_format("numpy", columns=[key])[key])
        if values.shape != (len(dataset), 8) or not np.isfinite(values).all():
            raise ValueError(f"{key}: expected a finite (frames, 8) array.")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"{key}: expected Station-normalized values in [0, 1].")
    cameras = list(dataset.meta.video_keys)
    expected = ["observation.images.camera_0", "observation.images.camera_1"]
    if cameras != expected:
        raise ValueError(f"Expected cameras in order {expected}, got {cameras}.")
    if info["fps"] != 10:
        raise ValueError("Expected the Station dataset's 10 FPS timing.")
    sample = dataset[0]
    for key in expected:
        if tuple(sample[key].shape) != (3, 224, 224):
            raise ValueError(f"{key}: expected decoded RGB tensor (3, 224, 224).")
    print(
        f"Dataset: {repo_id}, {info['total_episodes']} episodes, "
        f"{info['total_frames']} frames, {info['fps']} FPS",
        flush=True,
    )
    print(f"Tasks: {dataset.meta.tasks}", flush=True)
    print(f"Decoded cameras: {cameras}; 8 state and action values.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="PeteVLA/cheese2")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/cheese-smolvla"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in WSL. Check nvidia-smi and the Windows NVIDIA driver.")
    device = torch.device("cuda")
    torch.ones((32, 32), device=device).matmul(torch.ones((32, 32), device=device))
    torch.cuda.synchronize()
    print(f"CUDA computation OK: {torch.cuda.get_device_name(0)}, torch {torch.__version__}", flush=True)
    check_dataset(args.dataset, args.dataset_root)
    if args.check_only:
        return
    if args.output.exists():
        raise FileExistsError(f"Choose a new --output directory: {args.output}")
    command = [
        sys.executable, "-m", "lerobot.scripts.train",
        f"--dataset.repo_id={args.dataset}",
        "--dataset.video_backend=pyav",
        "--policy.path=lerobot/smolvla_base",
        "--policy.device=cuda",
        "--policy.load_vlm_weights=false",
        "--policy.push_to_hub=false",
        f"--output_dir={args.output.resolve()}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch_size}",
        "--num_workers=0",
        "--save_freq=1000",
        "--log_freq=20",
        "--wandb.enable=false",
        "--eval_freq=0",
    ]
    if args.dataset_root is not None:
        command.append(f"--dataset.root={args.dataset_root.resolve()}")
    print("Starting:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
