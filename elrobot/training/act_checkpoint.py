"""Load ACT checkpoints: external mean/std processor checkpoints, or LeRobot 0.3.x native ones.

ACT has no language input. Checkpoints trained with a task one-hot in
`observation.environment_state` ship `task_vocab.json`; they are wrapped so the
`task` string in each batch selects the matching one-hot input.
"""

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


TASK_VOCAB = "task_vocab.json"
TASK_FEATURE = "observation.environment_state"


class TaskConditionedACT(torch.nn.Module):
    """Adds the task one-hot expected by a task-conditioned ACT policy."""

    def __init__(self, policy: ACTPolicy, tasks: list[str]):
        super().__init__()
        self.policy = policy
        self.tasks = list(tasks)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)

    def one_hot(self, tasks, device) -> torch.Tensor:
        rows = []
        for task in tasks:
            if task not in self.tasks:
                raise ValueError(f"Task {task!r} is not one this ACT checkpoint was trained on: {self.tasks}")
            row = torch.zeros(len(self.tasks), device=device)
            row[self.tasks.index(task)] = 1
            rows.append(row)
        return torch.stack(rows)

    def select_action(self, batch: dict) -> torch.Tensor:
        tasks = batch.get("task")
        if not tasks:
            raise ValueError("Task-conditioned ACT needs batch['task'].")
        state = batch["observation.state"]
        return self.policy.select_action({**batch, TASK_FEATURE: self.one_hot(tasks, state.device)})

    def reset(self):
        self.policy.reset()


def load_act(checkpoint: Path):
    """ACT policy from either checkpoint layout, task-conditioned when task_vocab.json is present."""
    checkpoint = Path(checkpoint)
    if (checkpoint / "policy_preprocessor.json").exists():
        policy = load_act_checkpoint(checkpoint)
    else:
        policy = ACTPolicy.from_pretrained(str(checkpoint))
    feature = policy.config.input_features.get(TASK_FEATURE)
    vocab_path = checkpoint / TASK_VOCAB
    if feature is None:
        if vocab_path.exists():
            raise ValueError(f"{TASK_VOCAB} present but the policy has no {TASK_FEATURE} input.")
        return policy
    if not vocab_path.exists():
        raise ValueError(f"Policy uses {TASK_FEATURE}; {TASK_VOCAB} is required to map tasks.")
    tasks = json.loads(vocab_path.read_text())["tasks"]
    if (not isinstance(tasks, list) or len(set(tasks)) != len(tasks)
            or tuple(feature.shape) != (len(tasks),) or not all(isinstance(t, str) and t for t in tasks)):
        raise ValueError(f"{TASK_VOCAB} must list {feature.shape[0]} distinct task strings.")
    print(f"Task-conditioned ACT: tasks {tasks}", flush=True)
    return TaskConditionedACT(policy, tasks)
