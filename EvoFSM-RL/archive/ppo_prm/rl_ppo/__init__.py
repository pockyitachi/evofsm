"""PPO (Proximal Policy Optimization) + GAE trainer for LoRA fine-tuning.

Drop-in alternative to :mod:`evofsm_rl.rl.grpo`. Where GRPO uses
within-group reward baselines for advantage, PPO uses a learned value
function (a linear head on top of the frozen base model's last-token
hidden state) to compute GAE per-step advantages, then applies the
PPO-clipped surrogate loss to the LoRA parameters.

Key design choices (settled with user):
  * Value head: ``Linear(hidden_size → 1)`` on the last-token hidden
    state of the **frozen base** model (does NOT share the policy LoRA).
    Pretrained on existing trajectories before the PPO loop starts.
  * Per-step rewards are zero everywhere except the terminal step,
    which carries the trajectory-level binary reward R ∈ {0, 1}. PPO+GAE
    spreads the credit per-step automatically.
  * GAE λ default 1.0 (pure Monte Carlo). Optional adaptive schedule
    (λ=1.0 → λ=0.95 after iter N) for ablation.
  * Single epoch per rollout buffer (strictly on-policy — same as GRPO).
  * Standard knobs: clip ε=0.2, value loss coef=0.5.

The module is intentionally a sibling of ``rl/`` rather than a child;
``rl/grpo.py`` is the load-bearing GRPO path and is NOT touched.
"""

from evofsm_rl.rl_ppo.evolution_loop import (
    PPOEvolutionConfig,
    run_evolution_ppo,
    run_l_c_evolution_ppo,
)
from evofsm_rl.rl_ppo.gae import compute_gae
from evofsm_rl.rl_ppo.ppo_loss import ppo_clipped_loss, value_loss
from evofsm_rl.rl_ppo.trainer import PPOTrainer
from evofsm_rl.rl_ppo.value_head import (
    LinearValueHead,
    attach_value_head,
    load_value_head,
    save_value_head,
)

__all__ = [
    "LinearValueHead",
    "PPOEvolutionConfig",
    "PPOTrainer",
    "attach_value_head",
    "compute_gae",
    "load_value_head",
    "ppo_clipped_loss",
    "run_evolution_ppo",
    "run_l_c_evolution_ppo",
    "save_value_head",
    "value_loss",
]
