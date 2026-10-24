"""Load an ACT checkpoint with explicit external mean/std processors."""

import dataclasses
import json
from pathlib import Path

import draccus
import torch
from lerobot.configs.types import NormalizationMode
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from safetensors.torch import load_file


def processor_stats(checkpoint: Path, name: str, expected_steps: list[str]) -> dict:
    processor = json.loads((checkpoint / name).read_text())
    steps = processor["steps"]
    if [step["registry_name"] for step in steps] != expected_steps:
        raise ValueError(f"{name}: unsupported processor sequence.")
    for step in steps:
        config = step["config"]
        if step["registry_name"] == "rename_observations_processor" and config["rename_map"]:
            raise ValueError("ACT observation renaming is not supported by this bridge.")
        if step["registry_name"] == "device_processor" and config.get("float_dtype") is not None:
            raise ValueError("ACT processor dtype conversion is not supported.")
    normalizer = next(step for step in steps if "normalizer_processor" in step["registry_name"])
    config = normalizer["config"]
    if config["eps"] != 1e-8 or any(value != "MEAN_STD" for value in config["norm_map"].values()):
        raise ValueError("ACT external processors must use MEAN_STD with eps=1e-8.")
    state_file = normalizer["state_file"]
    if Path(state_file).name != state_file:
        raise ValueError("ACT processor state must be a file inside the checkpoint.")
    tensors = load_file(str(checkpoint / state_file))
    stats = {}
    for feature, feature_config in config["features"].items():
        expected_shape = tuple(feature_config["shape"])
        if feature_config["type"] == "VISUAL":
            expected_shape = (3, 1, 1)
        values = {}
        for statistic in ("mean", "std"):
            value = tensors[f"{feature}.{statistic}"]
            if tuple(value.shape) != expected_shape or not torch.isfinite(value).all():
                raise ValueError(f"{feature}.{statistic}: invalid ACT normalization statistics.")
            if statistic == "std" and (value <= 0).any():
                raise ValueError(f"{feature}: ACT standard deviations must be positive.")
            values[statistic] = value
        stats[feature] = values
    return stats


def load_act_checkpoint(checkpoint: Path) -> ACTPolicy:
    raw = json.loads((checkpoint / "config.json").read_text())
    if raw.pop("type") != "act":
        raise ValueError("Expected an ACT checkpoint.")
    fields = {field.name for field in dataclasses.fields(ACTConfig)}
    compatible_metadata = {
        "dtype": None, "use_peft": False, "pretrained_path": None,
        "pretrained_revision": None, "optimizer_fused": None,
    }
    for key in set(raw) - fields:
        if key not in compatible_metadata or raw[key] != compatible_metadata[key]:
            raise ValueError(f"Unsupported ACT configuration field: {key}={raw[key]!r}")
        del raw[key]
    config = draccus.decode(ACTConfig, raw)
    if any(mode != NormalizationMode.MEAN_STD for mode in config.normalization_mapping.values()):
        raise ValueError("External ACT checkpoint requires MEAN_STD feature normalization.")
    stats = processor_stats(checkpoint, "policy_preprocessor.json", [
        "rename_observations_processor", "to_batch_processor", "device_processor", "normalizer_processor",
    ])
    output_stats = processor_stats(checkpoint, "policy_postprocessor.json", [
        "unnormalizer_processor", "device_processor",
    ])
    if set(stats) != set(config.input_features) | {"action"} or set(output_stats) != {"action"}:
        raise ValueError("ACT processor features do not match the policy.")
    for key in ("mean", "std"):
        if not torch.equal(stats["action"][key], output_stats["action"][key]):
            raise ValueError("ACT input and output action statistics disagree.")
    # All backbone weights are loaded strictly below; avoid an unnecessary download.
    backbone_weights = config.pretrained_backbone_weights
    config.pretrained_backbone_weights = None
    policy = ACTPolicy(config, dataset_stats=stats)
    config.pretrained_backbone_weights = backbone_weights
    weights = load_file(str(checkpoint / "model.safetensors"))
    if not weights or any(not key.startswith("model.") for key in weights):
        raise ValueError("Expected ACT model weights with external normalization processors.")
    policy.model.load_state_dict({key.removeprefix("model."): value for key, value in weights.items()},
                                 strict=True)
    print("Loaded ACT weights strictly with external camera/state/action normalization. "
          "ACT does not use the language task to select behavior.", flush=True)
    return policy
