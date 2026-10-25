import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from act_checkpoint import processor_stats


class ACTProcessorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.step = {
            "registry_name": "normalizer_processor",
            "state_file": "stats.safetensors",
            "config": {
                "eps": 1e-8, "norm_map": {"ACTION": "MEAN_STD"},
                "features": {"action": {"type": "ACTION", "shape": [8]}},
            },
        }
        self.stats = {"action.mean": torch.zeros(8), "action.std": torch.ones(8)}

    def load(self):
        (self.root / "processor.json").write_text(json.dumps({"steps": [self.step]}))
        save_file(self.stats, str(self.root / "stats.safetensors"))
        return processor_stats(self.root, "processor.json", ["normalizer_processor"])

    def test_loads_exact_mean_and_std(self):
        result = self.load()
        self.assertTrue(torch.equal(result["action"]["mean"], self.stats["action.mean"]))
        self.assertTrue(torch.equal(result["action"]["std"], self.stats["action.std"]))

    def test_rejects_bad_statistics(self):
        for value in (torch.zeros(8), torch.full((8,), float("nan")), torch.ones(7)):
            self.stats["action.std"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load()

    def test_rejects_unknown_normalization(self):
        self.step["config"]["norm_map"]["ACTION"] = "MIN_MAX"
        with self.assertRaisesRegex(ValueError, "MEAN_STD"):
            self.load()

    def test_rejects_external_state_path(self):
        self.step["state_file"] = "../stats.safetensors"
        with self.assertRaisesRegex(ValueError, "inside"):
            self.load()

    def test_rejects_unknown_processor(self):
        self.step["registry_name"] = "unknown"
        with self.assertRaisesRegex(ValueError, "sequence"):
            self.load()


class TaskConditionedACTTests(unittest.TestCase):
    TASKS = ["pick-n-put-cheese put-in-q1", "pick-n-put-salami put-in-q2"]

    def save_native(self, root: Path, with_env=True, vocab=True):
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        inputs = {"observation.state": PolicyFeature(FeatureType.STATE, (8,))}
        if with_env:
            inputs["observation.environment_state"] = PolicyFeature(FeatureType.ENV, (2,))
        for cam in ("observation.images.camera_0", "observation.images.camera_1"):
            inputs[cam] = PolicyFeature(FeatureType.VISUAL, (3, 64, 64))
        config = ACTConfig(input_features=inputs, output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
                           pretrained_backbone_weights=None, dim_model=32, dim_feedforward=64, n_heads=2,
                           n_encoder_layers=1, n_decoder_layers=1, chunk_size=4, n_action_steps=4,
                           use_vae=False, device="cpu")
        stats = {k: {"mean": torch.zeros(s), "std": torch.ones(s)} for k, s in
                 (("observation.state", 8), ("action", 8))}
        for cam in ("observation.images.camera_0", "observation.images.camera_1"):
            stats[cam] = {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)}
        ACTPolicy(config, dataset_stats=stats).save_pretrained(root)
        if vocab:
            (root / "task_vocab.json").write_text(json.dumps(
                {"feature": "observation.environment_state", "tasks": self.TASKS}))

    def batch(self, task):
        b = {"observation.state": torch.zeros(1, 8), "task": [task]}
        for cam in ("observation.images.camera_0", "observation.images.camera_1"):
            b[cam] = torch.rand(1, 3, 64, 64)
        return b

    def test_native_checkpoint_gets_task_one_hot(self):
        from act_checkpoint import TaskConditionedACT, load_act

        with tempfile.TemporaryDirectory() as tmp:
            self.save_native(Path(tmp))
            policy = load_act(Path(tmp)).eval()
        self.assertIsInstance(policy, TaskConditionedACT)
        self.assertEqual(tuple(policy.config.output_features["action"].shape), (8,))
        seen = {}
        inner = policy.policy.select_action
        def spy(b):
            seen["env"] = b["observation.environment_state"]
            return inner(b)
        policy.policy.select_action = spy
        policy.reset()
        with torch.no_grad():
            policy.select_action(self.batch(self.TASKS[1]))
        self.assertTrue(torch.equal(seen["env"], torch.tensor([[0.0, 1.0]])))
        with self.assertRaisesRegex(ValueError, "not one this ACT checkpoint"):
            policy.select_action(self.batch("pick-n-put-cheese put-in-q4"))

    def test_task_feature_without_vocab_is_rejected(self):
        from act_checkpoint import load_act

        with tempfile.TemporaryDirectory() as tmp:
            self.save_native(Path(tmp), vocab=False)
            with self.assertRaisesRegex(ValueError, "task_vocab.json is required"):
                load_act(Path(tmp))

    def test_plain_native_checkpoint_loads_unwrapped(self):
        from act_checkpoint import TaskConditionedACT, load_act

        with tempfile.TemporaryDirectory() as tmp:
            self.save_native(Path(tmp), with_env=False, vocab=False)
            policy = load_act(Path(tmp))
        self.assertNotIsInstance(policy, TaskConditionedACT)
        with torch.no_grad():
            action = policy.eval().select_action(self.batch("anything"))
        self.assertEqual(tuple(action.shape), (1, 8))


if __name__ == "__main__":
    unittest.main()
