import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

import range_penalty
from range_penalty import corrected_forward


class TrainingLossTests(unittest.TestCase):
    def setUp(self):
        self.losses = torch.full((1, 3, 32), 100., requires_grad=True)
        with torch.no_grad():
            self.losses[:, :2, :8] = 2
        self.policy = SimpleNamespace(
            config=SimpleNamespace(
                adapt_to_pi_aloha=False, max_action_dim=32,
                action_feature=SimpleNamespace(shape=(8,)),
            ),
            normalize_inputs=lambda batch: batch,
            normalize_targets=lambda batch: batch,
            prepare_images=lambda batch: (None, None),
            prepare_state=lambda batch: None,
            prepare_language=lambda batch: (None, None),
            prepare_action=lambda batch: batch["action"],
            model=SimpleNamespace(forward=lambda *args: self.losses),
        )
        self.batch = {
            "action": torch.zeros(1, 3, 8),
            "action_is_pad": torch.tensor([[False, False, True]]),
        }

    def forward(self):
        return corrected_forward(self.policy, SmolVLAPolicy.forward, self.batch)

    def test_installed_loss_includes_padding_but_correction_excludes_it(self):
        old_loss, _ = SmolVLAPolicy.forward(self.policy, self.batch)
        self.assertGreater(old_loss.item(), 80)
        loss, metrics = self.forward()
        self.assertEqual(loss.item(), 2)
        self.assertEqual(metrics["loss"], 2)
        self.assertEqual(metrics["losses_after_rm_padding"].shape, (1, 3, 8))
        loss.backward()
        expected = torch.zeros_like(self.losses)
        expected[:, :2, :8] = 1 / 16
        torch.testing.assert_close(self.losses.grad, expected)

    def test_all_padded_batch_has_zero_loss_and_gradient(self):
        self.batch["action_is_pad"].fill_(True)
        loss, _ = self.forward()
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(self.losses.grad.abs().sum().item(), 0)

    def test_nonpadded_batch_uses_all_real_targets(self):
        self.batch["action_is_pad"].fill_(False)
        loss, _ = self.forward()
        self.assertAlmostEqual(loss.item(), (2 + 2 + 100) / 3, places=5)

    def test_invalid_or_missing_mask_fails(self):
        for mask in (None, torch.zeros(1, 3), torch.zeros(3, dtype=torch.bool)):
            with self.subTest(mask=mask):
                self.batch["action_is_pad"] = mask
                with self.assertRaisesRegex(ValueError, "boolean"):
                    self.forward()

    def test_unexpected_loss_shape_fails(self):
        self.losses = torch.ones(1, 3, 16)
        with self.assertRaisesRegex(ValueError, "loss shape"):
            self.forward()

    def test_nonfinite_valid_loss_fails(self):
        with torch.no_grad():
            self.losses[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.forward()

    def test_entrypoint_enables_correction_and_restores_installed_forward(self):
        original = SmolVLAPolicy.forward

        def train(*args, **kwargs):
            loss, _ = SmolVLAPolicy.forward(self.policy, self.batch)
            self.assertEqual(loss.item(), 2)

        with patch("sys.argv", ["range_penalty.py", "--steps=1"]):
            with patch.object(range_penalty.runpy, "run_module", side_effect=train) as run:
                range_penalty.main()
        run.assert_called_once_with("lerobot.scripts.train", run_name="__main__")
        self.assertIs(SmolVLAPolicy.forward, original)

    def test_penalty_entrypoint_receives_corrected_base_loss(self):
        original = SmolVLAPolicy.forward

        def penalize(policy, forward, batch, weight, noise=None, time=None):
            self.assertEqual(weight, 10)
            loss, metrics = forward(policy, batch, noise=noise, time=time)
            metrics["range_penalty"] = 0
            return loss, metrics

        def train(*args, **kwargs):
            loss, _ = SmolVLAPolicy.forward(self.policy, self.batch)
            self.assertEqual(loss.item(), 2)

        with patch("sys.argv", ["range_penalty.py", "--range-penalty-weight=10"]):
            with patch.object(range_penalty, "penalized_forward", side_effect=penalize):
                with patch.object(range_penalty.runpy, "run_module", side_effect=train):
                    range_penalty.main()
        self.assertIs(SmolVLAPolicy.forward, original)


if __name__ == "__main__":
    unittest.main()
