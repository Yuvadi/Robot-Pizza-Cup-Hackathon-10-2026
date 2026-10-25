"""Correct SmolVLA loss padding, with an optional action range penalty."""

import argparse
import math
import runpy
import sys

import torch
from lerobot.configs.types import NormalizationMode
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def corrected_forward(policy, original_forward, batch, noise=None, time=None):
    if policy.config.adapt_to_pi_aloha:
        raise ValueError("Corrected loss supports Station actions, not ALOHA conversion.")
    is_pad = batch.get("action_is_pad")
    actions = batch["action"]
    if (is_pad is None or is_pad.dtype != torch.bool
            or is_pad.shape != actions.shape[:2]):
        raise ValueError("action_is_pad must be a boolean (batch, chunk) mask.")
    dimensions = policy.config.action_feature.shape[0]
    if actions.ndim != 3 or actions.shape[-1] != dimensions:
        raise ValueError("Action targets must have shape (batch, chunk, real joints).")
    _, metrics = original_forward(policy, batch, noise=noise, time=time)
    # LeRobot 0.3.3 exposes the unreduced, differentiable flow-matching loss here.
    losses = metrics["losses_after_forward"]
    if (losses.ndim != 3 or losses.shape[:2] != is_pad.shape
            or losses.shape[-1] != policy.config.max_action_dim
            or dimensions > losses.shape[-1]):
        raise ValueError("Unexpected SmolVLA unreduced loss shape.")
    real_losses = losses[:, :, :dimensions]
    valid = (~is_pad).unsqueeze(-1).expand_as(real_losses)
    loss = real_losses.masked_select(valid).sum() / valid.sum().clamp_min(1)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite corrected imitation loss.")
    metrics.update(
        losses_after_in_ep_bound=losses.masked_fill(is_pad.unsqueeze(-1), 0),
        losses_after_rm_padding=real_losses.masked_fill(~valid, 0),
        imitation_loss=loss.detach().item(),
        loss=loss.detach().item(),
    )
    return loss, metrics


def action_range_penalty(actions: torch.Tensor, is_pad: torch.Tensor | None = None) -> torch.Tensor:
    violations = torch.relu(-actions).square() + torch.relu(actions - 1).square()
    if is_pad is None:
        return violations.mean()
    if is_pad.shape != actions.shape[:2] or is_pad.dtype != torch.bool:
        raise ValueError("action_is_pad must be a boolean (batch, chunk) mask.")
    valid = (~is_pad).unsqueeze(-1).expand_as(actions)
    return violations.masked_select(valid).sum() / valid.sum().clamp_min(1)


def penalized_forward(policy, original_forward, batch, weight, noise=None, time=None):
    if policy.config.adapt_to_pi_aloha:
        raise ValueError("Range penalty supports Station actions, not ALOHA conversion.")
    action_type = policy.config.action_feature.type
    if policy.config.normalization_mapping[action_type] != NormalizationMode.MEAN_STD:
        raise ValueError("Range penalty requires MEAN_STD action normalization.")
    actions = policy.prepare_action(policy.normalize_targets(batch))
    if noise is None:
        noise = policy.model.sample_noise(actions.shape, actions.device)
    if time is None:
        time = policy.model.sample_time(actions.shape[0], actions.device)
    velocities = []

    def capture(module, inputs, output):
        velocities.append(output)

    handle = policy.model.action_out_proj.register_forward_hook(capture)
    try:
        base_loss, metrics = original_forward(policy, batch, noise=noise, time=time)
    finally:
        handle.remove()
    if len(velocities) != 1:
        raise RuntimeError("Expected exactly one action velocity projection during training.")
    t = time[:, None, None]
    # Flow matching uses x_t = t * noise + (1-t) * action and v = noise-action.
    estimate = t * noise + (1 - t) * actions - t * velocities[0]
    estimate = estimate[:, :, :policy.config.action_feature.shape[0]]
    stats = policy.unnormalize_outputs.buffer_action
    # Unnormalize.forward is no_grad; use its formula here to preserve gradients.
    decoded = estimate * stats["std"] + stats["mean"]
    penalty = action_range_penalty(decoded, batch.get("action_is_pad"))
    loss = base_loss + weight * penalty
    if not torch.isfinite(loss):
        raise ValueError("Non-finite range-penalized training loss.")
    metrics.update(
        imitation_loss=base_loss.detach().item(),
        range_penalty=penalty.detach().item(),
        loss=loss.detach().item(),
    )
    return loss, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--range-penalty-weight", type=float, default=0)
    args, remaining = parser.parse_known_args()
    if not math.isfinite(args.range_penalty_weight) or args.range_penalty_weight < 0:
        parser.error("--range-penalty-weight must be nonnegative and finite.")
    original_forward = SmolVLAPolicy.forward
    updates = 0

    def forward(self, batch, noise=None, time=None):
        nonlocal updates
        def imitation(policy, batch, noise=None, time=None):
            return corrected_forward(policy, original_forward, batch, noise=noise, time=time)

        if args.range_penalty_weight:
            result = penalized_forward(
                self, imitation, batch, args.range_penalty_weight, noise=noise, time=time,
            )
        else:
            result = imitation(self, batch, noise=noise, time=time)
        updates += 1
        if args.range_penalty_weight and (updates == 1 or updates % 20 == 0):
            metrics = result[1]
            print(f"Range loss step={updates}: imitation={metrics['imitation_loss']:.6f} "
                  f"penalty={metrics['range_penalty']:.8f} total={metrics['loss']:.6f}",
                  flush=True)
        return result

    SmolVLAPolicy.forward = forward
    print("Corrected imitation loss enabled: action_is_pad; mean over valid steps "
          "and real joints only.", flush=True)
    if args.range_penalty_weight:
        print(f"Range penalty enabled: weight={args.range_penalty_weight}; "
              "estimated endpoint, dataset coordinates [0, 1], real joints only.", flush=True)
    sys.argv = [sys.argv[0], *remaining]
    try:
        runpy.run_module("lerobot.scripts.train", run_name="__main__")
    finally:
        SmolVLAPolicy.forward = original_forward


if __name__ == "__main__":
    main()
