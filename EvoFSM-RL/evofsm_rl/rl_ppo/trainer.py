"""PPO + GAE trainer for the EvoFSM-RL evolution loop.

Drop-in replacement for :func:`evofsm_rl.rl.grpo.grpo_step` (used inside
``run_l_c_evolution``). Same trajectory data shape (reuses
:class:`evofsm_rl.rl.grpo.TrajectoryData`), same replay-file format on
disk, same buffer-fills-then-fires cadence. The only differences:

  * Advantages come from GAE on per-step value-head estimates instead
    of within-group reward subtraction.
  * Loss is the PPO clipped surrogate (not the REINFORCE log-prob × adv
    GRPO uses).
  * A second loss term (MSE between V_θ(s_t) and GAE returns) trains
    the value head jointly with the LoRA policy.
  * The value head has its own optimizer (separate from the LoRA's).

The trainer does not touch ``grpo.py`` — it duplicates the replay-load
+ forward-with-grad helper into private functions here. This keeps the
GRPO module load-bearing for the existing B4 paths untouched while
letting us swap in PPO via a different trainer class.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import torch

from evofsm_rl.rl.grpo import TrajectoryData, cleanup_replay_data
from evofsm_rl.rl_ppo.gae import compute_gae
from evofsm_rl.rl_ppo.ppo_loss import ppo_clipped_loss, value_loss
from evofsm_rl.rl_ppo.value_head import LinearValueHead, load_value_head


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Replay helpers (private — mirrors grpo._load_replay_step but kept
# separate so changes to one don't bleed into the other).
# ─────────────────────────────────────────────────────────────────────


def _load_replay_step(path: str, device: str) -> dict[str, Any]:
    """Load one step's replay ``.pt`` to a target device."""
    raw = torch.load(path, map_location="cpu", weights_only=True)
    moved: dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device) if v.numel() > 0 else v
        else:
            moved[k] = v
    return moved


def _build_forward_kwargs(
    step_data: dict[str, Any], device: str, *, include_action: bool = True,
) -> tuple[dict[str, Any], int, int]:
    """Reconstruct forward kwargs for one replay step.

    Returns ``(fwd_kwargs, input_len, action_len)``. When
    ``include_action=False`` the input sequence is left at length
    ``input_len`` (no action tokens appended) — useful for the value-
    head forward, where we only need V(s_t) at the prompt-end position.
    Otherwise the action tokens are concatenated so the standard
    autoregressive log-prob slice is valid.
    """
    action_ids: torch.Tensor = step_data["action_token_ids"]
    input_len: int = int(step_data["input_len"])

    action_ids_1d = action_ids.flatten()
    action_len = int(action_ids_1d.shape[-1])

    input_ids: torch.Tensor = step_data["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    if not include_action:
        # Use just the prompt portion so the last position is the prompt
        # end (= state representation).
        full_ids = input_ids[:, :input_len]
        full_len = int(full_ids.shape[-1])
    else:
        if input_ids.shape[-1] >= input_len + action_len:
            full_ids = input_ids
        else:
            full_ids = torch.cat(
                [input_ids[:, :input_len], action_ids_1d.unsqueeze(0)], dim=-1,
            )
        full_len = int(full_ids.shape[-1])

    skip_keys = {"action_token_ids", "input_len", "step_log_prob", "input_ids"}
    fwd_kwargs: dict[str, Any] = {"input_ids": full_ids.to(device)}
    for key, val in step_data.items():
        if key in skip_keys:
            continue
        if not isinstance(val, torch.Tensor) or val.numel() == 0:
            continue
        # Per-token fields are padded out to full_len if needed; if we
        # are *truncating* (include_action=False) then slice instead.
        if val.dim() >= 2 and val.shape[-1] == input_len:
            if full_len > input_len:
                pad_val = 1 if key == "attention_mask" else 0
                pad_shape = list(val.shape)
                pad_shape[-1] = full_len - input_len
                pad = torch.full(
                    pad_shape, pad_val, dtype=val.dtype, device=val.device,
                )
                val = torch.cat([val, pad], dim=-1)
            elif full_len < input_len:
                val = val[..., :full_len]
        fwd_kwargs[key] = val.to(device)

    return fwd_kwargs, input_len, action_len


def _compute_step_log_prob_with_grad(
    model: Any, step_data: dict[str, Any], device: str,
) -> torch.Tensor:
    """Forward pass returning ``log P(action | state)`` with gradient.

    Identical math to :func:`evofsm_rl.rl.grpo._compute_step_log_prob_with_grad`
    but lives here so PPO can evolve independently of the GRPO module.
    """
    fwd_kwargs, input_len, action_len = _build_forward_kwargs(
        step_data, device, include_action=True,
    )

    outputs = model(**fwd_kwargs)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    action_ids_1d = step_data["action_token_ids"].flatten().to(device)
    action_logits = logits[0, input_len - 1 : input_len - 1 + action_len, :]
    log_probs = torch.log_softmax(action_logits, dim=-1)
    token_log_probs = log_probs.gather(
        1, action_ids_1d.unsqueeze(1),
    ).squeeze(1)
    return token_log_probs.sum()


def _compute_value_with_grad(
    model: Any,
    value_head: LinearValueHead,
    step_data: dict[str, Any],
    device: str,
    *,
    disable_adapter: bool = True,
) -> torch.Tensor:
    """Forward pass returning V_θ(s_t) under the value head.

    The forward runs on the **base model with the LoRA adapter
    disabled** (when ``disable_adapter=True``) so the hidden state the
    head sees is a function of fixed parameters only — the value head
    is the only differentiable thing between the hidden state and the
    loss. This keeps V(s) independent of the policy LoRA, per the
    "value head does NOT share the policy LoRA" design decision.

    Returns:
        Scalar tensor V_θ(s_t). Gradient flows back into the value
        head's parameters only.
    """
    fwd_kwargs, _input_len, _action_len = _build_forward_kwargs(
        step_data, device, include_action=False,
    )
    fwd_kwargs["output_hidden_states"] = True

    if disable_adapter and hasattr(model, "disable_adapter_layers"):
        model.disable_adapter_layers()
        try:
            # No grad through the base model — only through the head.
            with torch.no_grad():
                outputs = model(**fwd_kwargs)
        finally:
            model.enable_adapter_layers()
    else:
        with torch.no_grad():
            outputs = model(**fwd_kwargs)

    # ``hidden_states`` is a tuple of (n_layers + 1) tensors of shape
    # (batch, seq_len, hidden_size). We want the last layer.
    hidden = outputs.hidden_states[-1]
    # Detach the hidden state so the only grad path is through the head.
    hidden = hidden.detach()
    return value_head(hidden)  # shape (1,)


# ─────────────────────────────────────────────────────────────────────
# PPOTrainer
# ─────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class PPOTrainerConfig:
    """Hyperparameters for :class:`PPOTrainer`.

    Defaults follow standard PPO + the design decisions documented in
    :mod:`evofsm_rl.rl_ppo.__init__`.
    """

    clip_eps: float = 0.2
    value_loss_coef: float = 0.5
    gae_gamma: float = 0.99
    gae_lambda: float = 1.0
    gae_lambda_after_iter: int | None = None
    gae_lambda_after_value: float = 0.95
    max_grad_norm: float = 1.0
    min_n_active: int = 1
    value_head_disable_adapter: bool = True


class PPOTrainer:
    """PPO + GAE trainer with a frozen-base value head.

    Wraps the LoRA-attached policy ``model`` + a separate
    :class:`LinearValueHead`. The trainer holds two optimizers (one for
    the LoRA params, one for the value head) and a trajectory buffer.
    Call :meth:`add_trajectory` after each rollout and :meth:`step` once
    the buffer is full (typically every K iterations, matching the
    GRPO cadence).
    """

    def __init__(
        self,
        model: Any,
        value_head: LinearValueHead,
        *,
        lora_lr: float = 3e-4,
        value_lr: float = 1e-4,
        value_head_path: Path | str | None = None,
        clip_eps: float = 0.2,
        value_loss_coef: float = 0.5,
        gae_gamma: float = 0.99,
        gae_lambda: float = 1.0,
        gae_lambda_after_iter: int | None = None,
        gae_lambda_after_value: float = 0.95,
        max_grad_norm: float = 1.0,
        min_n_active: int = 1,
        device: str = "cuda",
        log_path: Path | str | None = None,
        value_head_disable_adapter: bool = True,
    ):
        self.model = model
        self.value_head = value_head
        self.device = str(device)
        self.config = PPOTrainerConfig(
            clip_eps=float(clip_eps),
            value_loss_coef=float(value_loss_coef),
            gae_gamma=float(gae_gamma),
            gae_lambda=float(gae_lambda),
            gae_lambda_after_iter=gae_lambda_after_iter,
            gae_lambda_after_value=float(gae_lambda_after_value),
            max_grad_norm=float(max_grad_norm),
            min_n_active=int(min_n_active),
            value_head_disable_adapter=bool(value_head_disable_adapter),
        )

        if value_head_path is not None:
            load_value_head(self.value_head, value_head_path)
            logger.info(
                "PPOTrainer: loaded pretrained value head from %s",
                value_head_path,
            )

        lora_params = [p for p in model.parameters() if p.requires_grad]
        if not lora_params:
            raise ValueError(
                "PPOTrainer: model has no trainable parameters. "
                "Call attach_lora(model) before constructing the trainer."
            )
        self.lora_optimizer = torch.optim.AdamW(lora_params, lr=float(lora_lr))
        self.value_optimizer = torch.optim.AdamW(
            self.value_head.parameters(), lr=float(value_lr),
        )

        self.trajectory_buffer: list[TrajectoryData] = []
        self.log_path: Path | None = Path(log_path) if log_path else None
        self._fire_counter: int = 0

    # ── Buffer ─────────────────────────────────────────────────────

    def add_trajectory(self, traj: TrajectoryData) -> None:
        """Append a rollout-produced trajectory to the buffer."""
        if traj is None:
            return
        self.trajectory_buffer.append(traj)

    def buffer_size(self) -> int:
        return len(self.trajectory_buffer)

    def clear_buffer(self, *, cleanup_replay: bool = True) -> int:
        """Empty the buffer (and optionally delete replay files).

        Returns the number of replay files removed.
        """
        n = 0
        if cleanup_replay and self.trajectory_buffer:
            n = cleanup_replay_data(self.trajectory_buffer)
        self.trajectory_buffer.clear()
        return n

    # ── Lambda schedule ───────────────────────────────────────────

    def _effective_lambda(self, iteration: int) -> float:
        cfg = self.config
        if cfg.gae_lambda_after_iter is None:
            return cfg.gae_lambda
        if iteration <= int(cfg.gae_lambda_after_iter):
            return cfg.gae_lambda
        return cfg.gae_lambda_after_value

    # ── Step ──────────────────────────────────────────────────────

    def step(self, *, iteration: int = 0) -> dict[str, float]:
        """One PPO update using all buffered trajectories.

        Pipeline:
          1. For each trajectory, compute per-step V(s_t) under the
             current value head (no grad through model).
          2. Run :func:`compute_gae` to get advantages + returns.
          3. For each step: forward through LoRA-attached base, get
             current log-prob; forward through value head, get
             prediction. Accumulate policy and value losses.
          4. Combined loss = ``policy_loss + value_loss_coef * value_loss``.
          5. One backward + step for the LoRA, one backward + step for
             the value head. (We share the autograd graph by calling
             ``.backward()`` once on the combined loss; the LoRA
             optimizer steps the policy params and the value optimizer
             steps the head params.)

        Returns a metrics dict for logging.
        """
        active: list[TrajectoryData] = [
            t for t in self.trajectory_buffer if t.replay_paths
        ]
        mean_reward = (
            sum(t.reward for t in self.trajectory_buffer)
            / len(self.trajectory_buffer)
            if self.trajectory_buffer else 0.0
        )

        if len(active) < max(1, self.config.min_n_active):
            logger.info(
                "PPOTrainer.step: only %d active trajectories (need >= %d); "
                "optimizer not stepped.",
                len(active), max(1, self.config.min_n_active),
            )
            metrics = {
                "iteration": int(iteration),
                "n_trajectories": len(self.trajectory_buffer),
                "n_active": len(active),
                "mean_reward": float(mean_reward),
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "loss": 0.0,
                "grad_norm": 0.0,
                "advantage_std": 0.0,
                "advantage_max": 0.0,
                "gae_lambda": float(self._effective_lambda(iteration)),
                "skipped_min_n_active": int(len(active) > 0),
            }
            self._write_log(metrics)
            return metrics

        lam = self._effective_lambda(iteration)
        gamma = self.config.gae_gamma

        # ── Pass 1: per-step V(s_t) under the *current* value head, no
        # grad. These values are snapshotted for GAE and for the
        # value-loss target (``returns = advantages + values``).
        was_training = self.model.training
        self.model.eval()
        self.value_head.eval()

        per_traj_values: list[list[float]] = []
        try:
            for traj in active:
                step_values: list[float] = []
                for replay_path in traj.replay_paths:
                    step_data = _load_replay_step(replay_path, self.device)
                    with torch.no_grad():
                        v = _compute_value_with_grad(
                            self.model, self.value_head, step_data,
                            self.device,
                            disable_adapter=self.config.value_head_disable_adapter,
                        )
                    step_values.append(float(v.item()))
                    del step_data, v
                    if torch.cuda.is_available() and self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                per_traj_values.append(step_values)
        finally:
            self.model.train(was_training)

        # ── Compute GAE per trajectory.
        # If trajectory has per_step_rewards (PRM-shaped), pass them so GAE
        # uses dense per-step rewards instead of the sparse terminal-only R.
        per_traj_adv: list[list[float]] = []
        per_traj_ret: list[list[float]] = []
        for traj, step_values in zip(active, per_traj_values):
            psr = getattr(traj, "per_step_rewards", None)
            # Validate length matches step count; if mismatch, fall back to None
            if psr is not None and len(psr) != len(step_values):
                psr = None
            adv, ret = compute_gae(
                step_values, float(traj.reward),
                per_step_rewards=psr,
                gamma=gamma, lam=lam,
            )
            per_traj_adv.append(adv)
            per_traj_ret.append(ret)

        # ── Normalize advantages across all steps in this fire.
        # Standard PPO practice: (A - mean) / (std + 1e-8). Stabilises
        # gradient magnitudes when per-step rewards have variable scale
        # (e.g. PRM scores summed over long trajectories can produce
        # advantage magnitudes ~10× larger than the sparse-R baseline).
        # When advantages are uniform (e.g. all rollouts succeed/fail
        # equally), std → 0 and we skip normalisation.
        _all_adv_flat = [a for adv in per_traj_adv for a in adv]
        if len(_all_adv_flat) >= 2:
            _adv_mean = sum(_all_adv_flat) / len(_all_adv_flat)
            _adv_var = sum((a - _adv_mean) ** 2 for a in _all_adv_flat) / max(len(_all_adv_flat) - 1, 1)
            _adv_std = max(_adv_var ** 0.5, 1e-8)
            if _adv_std > 1e-6:
                per_traj_adv = [
                    [(a - _adv_mean) / _adv_std for a in adv]
                    for adv in per_traj_adv
                ]

        # ── Pass 2: per-step backward — recompute log prob + V with
        # grad, accumulate PPO + value losses. Per-step backward keeps
        # peak activation memory ~ one step's worth (~5-8 GB on Qwen3-
        # VL-8B at 7k-token context) instead of holding the whole
        # buffer's autograd graph simultaneously.
        self.model.train()
        self.value_head.train()
        self.lora_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        n_steps_total = sum(len(t.replay_paths) for t in active)
        if n_steps_total == 0:
            # Defensive — active filter already required replay_paths non-empty.
            n_steps_total = 1

        scale = 1.0 / len(active)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        all_adv_values: list[float] = []

        try:
            for traj, advs, rets in zip(active, per_traj_adv, per_traj_ret):
                T = len(traj.replay_paths)
                if T == 0:
                    continue
                traj_scale = scale / T
                for step_idx, replay_path in enumerate(traj.replay_paths):
                    step_data = _load_replay_step(replay_path, self.device)

                    # Old log-prob: detached float saved at rollout time.
                    old_log_prob_val = float(step_data.get("step_log_prob", 0.0))
                    # Some replay payloads may have dropped step_log_prob —
                    # fall back to traj.step_log_probs if available.
                    if "step_log_prob" not in step_data:
                        if step_idx < len(traj.step_log_probs):
                            old_log_prob_val = float(
                                traj.step_log_probs[step_idx]
                            )

                    # Forward 1: current LoRA policy, with grad.
                    new_lp = _compute_step_log_prob_with_grad(
                        self.model, step_data, self.device,
                    )
                    new_lp_t = new_lp.unsqueeze(0)  # (1,)
                    old_lp_t = torch.tensor(
                        [old_log_prob_val],
                        device=new_lp.device,
                        dtype=new_lp.dtype,
                    )
                    adv_t = torch.tensor(
                        [float(advs[step_idx])],
                        device=new_lp.device,
                        dtype=new_lp.dtype,
                    )

                    # Forward 2: value head (LoRA disabled during the
                    # base forward inside _compute_value_with_grad).
                    v_pred = _compute_value_with_grad(
                        self.model, self.value_head, step_data, self.device,
                        disable_adapter=self.config.value_head_disable_adapter,
                    )
                    v_pred_t = v_pred  # already shape (1,)
                    ret_t = torch.tensor(
                        [float(rets[step_idx])],
                        device=v_pred.device,
                        dtype=v_pred.dtype,
                    )

                    pol_loss = ppo_clipped_loss(
                        new_lp_t, old_lp_t, adv_t,
                        clip_eps=self.config.clip_eps,
                    )
                    val_loss = value_loss(v_pred_t, ret_t)

                    # Per-step combined loss is scaled by ``traj_scale``
                    # (= 1 / (N_active * T_j)). Summing across all
                    # steps gives the average per-trajectory loss, and
                    # averaging across trajectories the final PPO loss.
                    step_loss = traj_scale * (
                        pol_loss + self.config.value_loss_coef * val_loss
                    )
                    step_loss.backward()

                    total_policy_loss += float(pol_loss.detach().item()) * traj_scale
                    total_value_loss += float(val_loss.detach().item()) * traj_scale
                    all_adv_values.append(float(advs[step_idx]))

                    del step_data, new_lp, new_lp_t, v_pred, v_pred_t
                    del pol_loss, val_loss, step_loss
                    if torch.cuda.is_available() and self.device.startswith("cuda"):
                        torch.cuda.empty_cache()

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            grad_norm_lora = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, self.config.max_grad_norm,
                ).item()
            )
            grad_norm_value = float(
                torch.nn.utils.clip_grad_norm_(
                    self.value_head.parameters(), self.config.max_grad_norm,
                ).item()
            )
            self.lora_optimizer.step()
            self.value_optimizer.step()
            self.lora_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
        finally:
            self.model.train(was_training)

        # Metrics
        if all_adv_values:
            mean_adv = sum(all_adv_values) / len(all_adv_values)
            adv_var = sum((a - mean_adv) ** 2 for a in all_adv_values) / len(all_adv_values)
            adv_std = adv_var ** 0.5
            adv_max = max(abs(a) for a in all_adv_values)
        else:
            mean_adv = 0.0
            adv_std = 0.0
            adv_max = 0.0

        self._fire_counter += 1
        metrics = {
            "iteration": int(iteration),
            "fire": int(self._fire_counter),
            "n_trajectories": len(self.trajectory_buffer),
            "n_active": len(active),
            "mean_reward": float(mean_reward),
            "policy_loss": float(total_policy_loss),
            "value_loss": float(total_value_loss),
            "loss": float(
                total_policy_loss + self.config.value_loss_coef * total_value_loss
            ),
            "grad_norm": float(grad_norm_lora),
            "grad_norm_value": float(grad_norm_value),
            "mean_advantage": float(mean_adv),
            "advantage_std": float(adv_std),
            "advantage_max": float(adv_max),
            "gae_lambda": float(lam),
            "gae_gamma": float(gamma),
            "skipped_min_n_active": 0,
        }
        self._write_log(metrics)
        return metrics

    # ── Logging ───────────────────────────────────────────────────

    def _write_log(self, metrics: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        import json
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(metrics) + "\n")


__all__ = [
    "PPOTrainer",
    "PPOTrainerConfig",
]
