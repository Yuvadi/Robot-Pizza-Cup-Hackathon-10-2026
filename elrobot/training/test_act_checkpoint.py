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


if __name__ == "__main__":
    unittest.main()
