import unittest
from types import SimpleNamespace

import torch

from lerobot.configs.types import FeatureType, NormalizationMode
from range_penalty import action_range_penalty, penalized_forward


class RangePenaltyTests(unittest.TestCase):
    def test_valid_values_and_endpoints_have_zero_penalty(self):
        values = torch.tensor([[[0., .5, 1.]]], requires_grad=True)
        loss = action_range_penalty(values)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(values.grad, torch.zeros_like(values)))

    def test_penalty_gradient_moves_both_violations_inward(self):
        values = torch.tensor([[[-.2, 1.3]]], requires_grad=True)
        loss = action_range_penalty(values)
        self.assertAlmostEqual(loss.item(), (.04 + .09) / 2)
        loss.backward()
        self.assertLess(values.grad[0, 0, 0].item(), 0)
        self.assertGreater(values.grad[0, 0, 1].item(), 0)

    def test_padding_is_excluded_from_loss_and_gradient(self):
        values = torch.tensor([[[-.2], [100.]]], requires_grad=True)
        loss = action_range_penalty(values, torch.tensor([[False, True]]))
        self.assertAlmostEqual(loss.item(), .04)
        loss.backward()
        self.assertEqual(values.grad[0, 1, 0].item(), 0)
        self.assertEqual(action_range_penalty(values, torch.ones(1, 2, dtype=torch.bool)).item(), 0)

    def test_estimated_endpoint_inverse_normalization_and_gradients(self):
        projection = torch.nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.eye(4))
        policy = SimpleNamespace(
            config=SimpleNamespace(
                adapt_to_pi_aloha=False,
                action_feature=SimpleNamespace(type=FeatureType.ACTION, shape=(2,)),
                normalization_mapping={FeatureType.ACTION: NormalizationMode.MEAN_STD},
            ),
            normalize_targets=lambda batch: batch,
            prepare_action=lambda batch: batch["action"],
            model=SimpleNamespace(action_out_proj=projection),
            unnormalize_outputs=SimpleNamespace(
                buffer_action={"mean": torch.tensor([.5, .5]), "std": torch.tensor([2., 2.])},
            ),
        )
        batch = {"action": torch.zeros(1, 2, 4),
                 "action_is_pad": torch.tensor([[False, True]])}
        noise = torch.tensor([[[2., -2., 100., 100.], [100., 100., 100., 100.]]])
        time = torch.tensor([.5])

        def original(policy, batch, noise, time):
            policy.model.action_out_proj(noise * 0)
            return projection.weight.sum() * 0 + 2, {"loss": 2}

        loss, metrics = penalized_forward(policy, original, batch, 10, noise, time)
        self.assertAlmostEqual(metrics["range_penalty"], 2.25)
        self.assertAlmostEqual(loss.item(), 24.5)
        self.assertEqual(len(projection._forward_hooks), 0)

        def gradient_forward(policy, batch, noise, time):
            policy.model.action_out_proj(torch.ones_like(noise))
            return projection.weight.sum() * 0, {}

        loss, _ = penalized_forward(policy, gradient_forward, batch, 10, noise, time)
        loss.backward()
        self.assertGreater(projection.weight.grad[:2].abs().sum().item(), 0)
        self.assertEqual(projection.weight.grad[2:].abs().sum().item(), 0)

    def test_mask_shape_is_validated(self):
        with self.assertRaises(ValueError):
            action_range_penalty(torch.zeros(1, 2, 8), torch.zeros(1, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
