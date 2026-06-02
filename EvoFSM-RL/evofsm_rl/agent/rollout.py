"""Rollout loop — Story 1.5 (Qwen3-VL-M3A: M3A clone with Qwen3-VL-8B base).

This file rewrites baseline_v0's single-shot JSON-only agent into a
near-verbatim port of ``android_world.agents.m3a.M3A``. Same per-step
two-phase reasoning (action → summary), same Reason/Action format, same
NL summary history. The ONLY substitution is the LLM call: where M3A
talks to GPT-4o through ``infer.MultimodalLlmWrapper.predict_mm``, we
build a Qwen3-VL chat-template prompt and call ``model.generate``.

Per-step flow (see ``M3A.step`` for the canonical reference):

  1. Get state (raw screenshot + UI element list + screen geometry).
  2. Render M3A-style UI element description list.
  3. Draw SoM marks on a copy of the raw screenshot ("before" frame).
  4. Build action prompt and call the model with [raw, before_som].
  5. Parse "Reason: …\\nAction: {…}".
       - if ill-formed: append a summary noting it, return done=False.
       - if action.action_type == 'status': return done=True.
       - if index is out-of-range: append a summary noting it, return.
  6. Execute the action via env.execute_action (index passes through).
  7. Wait, grab the after-state, draw SoM on the after frame.
  8. Burn 'before' / 'after' text labels onto the two frames.
  9. Build summary prompt and call the model with [before, after].
  10. Append a step_data dict (with summary) to history; return done=False.

The per-step history rendered into future prompts is a list of strings
shaped like ``"Step N - <model summary>"`` — exactly M3A's format.

Why this is worth two LLM calls per step (vs. baseline_v0's one):
the summary phase forces the model to write down what just happened,
and that text is what the next-step action selector reads to decide
"the goal is done → emit status: complete". See the analysis report
in traces/baseline_50task_v02_a11y/ for the empirical motivation.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from typing import Any

import numpy as np
from PIL import Image

from android_world.agents import base_agent
from android_world.agents import m3a_utils
from android_world.env import interface, json_action

from evofsm_rl.agent import a11y
from evofsm_rl.agent.action import (
    ParseResult,
    parse_action,
)
from evofsm_rl.agent.prompts import (
    build_action_messages,
    build_action_prompt,
    build_summary_messages,
    build_summary_prompt,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Generation config
# ─────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class GenerationConfig:
    """Generation hyperparameters for the Qwen3-VL policy.

    Defaults mirror ``configs/model.yaml:generation`` (greedy, 512 tokens).
    Overridable at agent construction time so RL rollouts can enable
    sampling without touching the yaml (yaml represents the base policy
    in its canonical eval form — see model.yaml's rationale block).
    """

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float = 1.0

    @classmethod
    def from_yaml(cls, gen: dict[str, Any]) -> "GenerationConfig":
        """Build from ``model.yaml:generation`` dict."""
        return cls(
            max_new_tokens=int(gen.get("max_new_tokens", 512)),
            do_sample=bool(gen.get("do_sample", False)),
            temperature=gen.get("temperature"),
            top_p=gen.get("top_p"),
            repetition_penalty=float(gen.get("repetition_penalty", 1.0)),
        )

    def to_kwargs(self) -> dict[str, Any]:
        """Render as kwargs for ``model.generate(...)``."""
        kw: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.do_sample:
            if self.temperature is not None:
                kw["temperature"] = float(self.temperature)
            if self.top_p is not None:
                kw["top_p"] = float(self.top_p)
        return kw


# ─────────────────────────────────────────────────────────────────────────
# Generation helper
# ─────────────────────────────────────────────────────────────────────────


def generate_text(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    images: list[Image.Image],
    device: str,
    generation_config: GenerationConfig,
    *,
    collect_log_probs: bool = False,
) -> tuple[str, float, int] | tuple[str, float, int, dict[str, Any]]:
    """Run one VLM forward + greedy-generate pass with N images.

    M3A's ``llm.predict_mm(prompt, images)`` accepts an arbitrary-length
    image list; we mirror that — Qwen3-VL natively supports multi-image
    inputs by emitting one ``<|image_pad|>`` placeholder per image in the
    rendered chat template.

    When ``collect_log_probs=True`` (B4 / GRPO path), also runs a second
    forward pass on the full (prompt + generated) sequence with
    ``torch.no_grad`` to record ``log P(action | state)`` under the
    current policy, and returns the raw CPU tensors the caller needs to
    save for later gradient-bearing recomputation.

    Returns:
        * default (``collect_log_probs=False``): ``(decoded_text,
          wall_seconds, input_token_count)`` — byte-identical to the
          pre-Story-4.2 contract, so B3 callers are unaffected.
        * with ``collect_log_probs=True``: a 4-tuple with an extra
          ``extras`` dict. ``extras`` contains CPU copies of every
          tensor the processor returned (``input_ids``,
          ``attention_mask``, ``pixel_values``, ``image_grid_thw``,
          ``mm_token_type_ids``, and any other field current or future
          transformers versions add) plus ``step_log_prob`` (float),
          ``action_token_ids`` (CPU tensor), and ``input_len`` (int).
          Hard-coding a subset would silently break every time the
          upstream processor API grows; passing through verbatim keeps
          the replay file forward-compatible.
    """
    import torch

    t0 = time.monotonic()

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=images, return_tensors="pt", padding=True
    )
    inputs = {
        k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()
    }
    input_len = int(inputs["input_ids"].shape[-1])

    if not collect_log_probs:
        # B3 path: single generate(), unchanged.
        with torch.no_grad():
            generated = model.generate(
                **inputs, **generation_config.to_kwargs(),
            )
        new_tokens = generated[0, input_len:]
        tokenizer = getattr(processor, "tokenizer", processor)
        raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return raw_text, time.monotonic() - t0, input_len

    # ── GRPO path: single generate() with output_scores=True ──
    #
    # Earlier versions ran generate() then a second full forward pass on
    # the (prompt + action) sequence to collect log-probs. That doubled
    # peak GPU memory (two passes of ~7k-token activations on Qwen3-VL-8B
    # pushed the 80 GB A100 to OOM), and pulled in transformers M-RoPE
    # bookkeeping (``mm_token_type_ids``) whose upstream signature keeps
    # shifting. We now piggy-back on generate() itself: ``output_scores=
    # True`` returns one logits tensor per generated token, which is all
    # we need for log P(a_t | s_{<t}). Memory stays at B3 parity — no
    # second forward pass runs. We still save the full processor output
    # to the replay file so the later gradient-bearing forward in
    # grpo._compute_step_log_prob_with_grad can reconstruct the sequence
    # at update time (that path DOES need the full forward; it uses
    # current LoRA weights and must propagate through them).
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            **generation_config.to_kwargs(),
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated = gen_out.sequences
        scores = gen_out.scores  # tuple of (batch, vocab) per generated token

    new_tokens = generated[0, input_len:]
    action_len = int(new_tokens.shape[-1])
    tokenizer = getattr(processor, "tokenizer", processor)
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # Sum per-token log-probs. ``scores[t]`` has shape (batch, vocab) and
    # is the logits distribution over ``new_tokens[t]``. With greedy
    # decoding these are the raw logits; with sampling-based generation
    # transformers applies temperature/top-k before returning them, which
    # is still the distribution the action was sampled from — that's the
    # behaviour GRPO wants.
    step_log_prob_t = torch.zeros((), device=new_tokens.device)
    if action_len > 0 and len(scores) == action_len:
        for t in range(action_len):
            log_probs_t = torch.log_softmax(scores[t][0].float(), dim=-1)
            step_log_prob_t = step_log_prob_t + log_probs_t[new_tokens[t]]
    step_log_prob = float(step_log_prob_t.item())

    def _cpu(t: Any) -> Any:
        return t.detach().cpu() if isinstance(t, torch.Tensor) else t

    # Pack every tensor field from ``inputs`` so the replay save below
    # (and grpo.py's later gradient-bearing recomputation) sees the full
    # processor output, not a hardcoded subset. Forward compatible with
    # whatever future transformers versions add to the processor output.
    extras: dict[str, Any] = {}
    for key, val in inputs.items():
        extras[key] = _cpu(val)
    extras["step_log_prob"] = step_log_prob
    extras["action_token_ids"] = _cpu(new_tokens)
    extras["input_len"] = input_len
    return raw_text, time.monotonic() - t0, input_len, extras


# ─────────────────────────────────────────────────────────────────────────
# Agent class — mirrors M3A.M3A
# ─────────────────────────────────────────────────────────────────────────


class Qwen3VLAgent(base_agent.EnvironmentInteractingAgent):
    """M3A-shaped Android agent backed by Qwen3-VL-8B-Instruct.

    Class structure mirrors ``android_world.agents.m3a.M3A`` 1:1; the
    only methodological substitution is the LLM call (Qwen3-VL local
    generate vs. GPT-4o API).

    Public surface:
        Qwen3VLAgent(model, processor, env, device,
                     generation_config=..., name=...)
            .step(goal)               → AgentInteractionResult(done, step_data)
            .reset(go_home=False)     → clears history
            .set_task_guidelines(list[str])  → optional task-specific prompt extras
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        env: interface.AsyncEnv,
        *,
        device: str,
        generation_config: GenerationConfig | None = None,
        name: str = "Qwen3-VL-M3A",
        wait_after_action_seconds: float = 2.0,
        l_c_prompt_text: str | None = None,
        collect_log_probs: bool = False,
        replay_dir: Any = None,  # str | Path | None
    ):
        """
        Args:
            model: Loaded Qwen3-VL model (from ``load_base_model``).
            processor: Matching ``AutoProcessor`` instance.
            env: Connected AndroidWorld ``AsyncEnv``.
            device: "mps" / "cuda" / "cpu" — must match the model's device.
            generation_config: Sampling hyperparameters. Defaults to
                greedy + 512 max-new-tokens.
            name: Used by the episode runner's logs.
            wait_after_action_seconds: Seconds to wait after each action
                before grabbing the after-state. Mirrors M3A's default 2.0s.
            l_c_prompt_text: Optional pre-rendered LAYER-2 block to splice
                into every action-selection prompt as "Workflow knowledge
                (transferred from related apps)". Used by the B2 Static-FSM
                baseline for Tier-B apps. Default ``None`` leaves prompts
                byte-identical to B1. Can be reassigned at runtime via
                :meth:`set_l_c_prompt_text` when an eval driver iterates
                over multiple apps with different L_C rows.
            collect_log_probs: If True, record ``log P(action | state)``
                for every action-phase generation and save a small replay
                file per step under ``replay_dir``. Needed by B4 / GRPO
                (Story 4.3+). Off by default — B1/B2/B3 rollout paths are
                unaffected.
            replay_dir: Base directory for per-episode replay subdirs.
                Ignored when ``collect_log_probs=False``. Each call to
                :meth:`reset` allocates a fresh
                ``{replay_dir}/episode_{NNNN}/`` subdir; steps write
                ``step_{idx}.pt`` files into it. GRPO
                (``evofsm_rl.rl.grpo.grpo_step``) loads those files
                back at update time and
                :func:`evofsm_rl.rl.grpo.cleanup_replay_data` deletes
                them once consumed.
        """
        # We disable the base class's adaptive ``transition_pause`` so we
        # can do M3A's explicit fixed wait between action and the after-grab.
        super().__init__(env=env, name=name, transition_pause=None)
        self._model = model
        self._processor = processor
        self._device = device
        self._generation_config = generation_config or GenerationConfig()
        self.wait_after_action_seconds = wait_after_action_seconds

        # Per-episode state (cleared on reset).
        self.history: list[dict[str, Any]] = []
        self.additional_guidelines: list[str] | None = None

        # L_C (per-category LAYER-2 prompt text). Not reset with history —
        # it's keyed by app, not by episode, so the driver controls its
        # lifetime via set_l_c_prompt_text() when crossing app boundaries.
        self._l_c_prompt_text: str | None = l_c_prompt_text

        # GRPO / B4 log-prob collection. When disabled (default), every
        # branch below is skipped — the existing step() code path stays
        # byte-identical to the B3 version.
        self._collect_log_probs: bool = bool(collect_log_probs)
        from pathlib import Path as _Path  # local import to avoid top-level churn
        self._replay_dir_base: Any = (
            _Path(replay_dir) if replay_dir is not None else None
        )
        self._replay_dir: Any = None            # set per-episode in reset()
        self._episode_counter: int = 0
        self._step_log_probs: list[float] = []

        # Metrics — also reset per episode. Exposed for batch eval drivers
        # (baseline_10task.py reads agent._self_reported etc).
        # `_self_reported` counts agent-emitted `status` actions per
        # episode (max 1, since the episode terminates on the first one).
        # AndroidWorld doesn't track this — we track it ourselves so we
        # can distinguish "agent thinks it finished" from "AW eval says
        # the device state matches the goal" (`success`).
        self._parse_failures: int = 0
        self._alias_hits: int = 0
        self._clamp_hits: int = 0
        self._self_reported: int = 0
        self._step_idx: int = 0

        # Some callers forget to put model in eval mode; cheap to enforce.
        try:
            self._model.eval()
        except AttributeError:
            pass

    # ── M3A surface ─────────────────────────────────────────────────

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def set_l_c_prompt_text(self, l_c_prompt_text: str | None) -> None:
        """Swap the L_C prompt block (call when crossing app boundaries).

        Pass ``None`` to return to the B1 / zero-shot prompt shape for
        apps whose Play Store category has no L_C (Tier-C case).
        """
        self._l_c_prompt_text = l_c_prompt_text

    def get_trajectory_data(
        self,
        task_name: str,
        seed: int,
        fsm_variant_id: str,
        reward: float,
        *,
        success: float | None = None,
    ) -> Any:
        """Package the just-finished episode into a
        :class:`evofsm_rl.rl.grpo.TrajectoryData` for GRPO.

        Meaningful only when the agent was constructed with
        ``collect_log_probs=True``. Call after
        :meth:`harness.run_template` returns and *before* the next
        :meth:`reset` wipes ``history`` / the replay dir bookkeeping.

        Args:
            task_name: AndroidWorld template id.
            seed: Task seed (so the GRPO logger can trace back).
            fsm_variant_id: ID of the L_C variant that generated this
                rollout. Drives within-group advantage computation.
            reward: Scalar reward (typically
                :func:`evofsm_rl.rl.grpo.compute_reward` from
                ``(success, n_steps, step_budget)``).
            success: Final success (0.0 / 0.5 / 1.0). When omitted, a
                best-effort fallback inspects the last-step
                ``action.goal_status == "complete"`` flag; callers that
                know the real ``task.is_successful`` verdict should
                pass it explicitly.
        """
        # Late import so the agent module has no compile-time dependency
        # on the RL package.
        from evofsm_rl.rl.grpo import TrajectoryData

        if success is None:
            last_action: dict[str, Any] = {}
            if self.history:
                last = self.history[-1]
                if isinstance(last.get("action_output_json"), dict):
                    last_action = last["action_output_json"]
                elif isinstance(last.get("action"), dict):
                    last_action = last["action"]
            success = 1.0 if last_action.get("goal_status") == "complete" else 0.0

        replay_paths: list[str] = []
        if self._replay_dir is not None and self._replay_dir.exists():
            replay_paths = sorted(
                str(p) for p in self._replay_dir.glob("step_*.pt")
            )

        return TrajectoryData(
            task_name=task_name,
            seed=int(seed),
            fsm_variant_id=str(fsm_variant_id),
            success=float(success),
            n_steps=int(self._step_idx),
            step_log_probs=list(self._step_log_probs),
            replay_paths=replay_paths,
            reward=float(reward),
        )

    def reset(self, go_home: bool = False) -> None:
        """Reset env state + clear episode history."""
        super().reset(go_home=go_home)
        self.history = []
        self._step_idx = 0
        self._parse_failures = 0
        self._alias_hits = 0
        self._clamp_hits = 0
        self._self_reported = 0
        # GRPO-only state. No-ops when collect_log_probs=False.
        self._step_log_probs = []
        if self._collect_log_probs and self._replay_dir_base is not None:
            self._episode_counter += 1
            self._replay_dir = (
                self._replay_dir_base
                / f"episode_{self._episode_counter:04d}"
            )
            self._replay_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._replay_dir = None

    # ── Core step ───────────────────────────────────────────────────

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """One M3A-style two-phase turn.

        Returns done=True iff the model emits a ``status`` action.
        Framework terminates the episode then; ``is_successful`` judges
        true success via the task's rule-based checker.
        """
        self._step_idx += 1
        step_data: dict[str, Any] = {
            "step": self._step_idx,
            "timestamp": time.time(),     # absolute wall-clock at step start (Story 2.0)
            "goal": goal,
            "raw_screenshot": None,
            "before_screenshot_with_som": None,
            "before_ui_elements": [],
            "before_ui_elements_text": None,
            "after_screenshot_with_som": None,
            "after_ui_elements": [],
            "after_ui_elements_text": None,
            "action_prompt": None,
            "action_output": None,
            "action_output_json": None,
            "action_reason": None,
            "action_raw_response": None,
            "action_wall_s": None,
            "action_input_tokens": None,
            "summary_prompt": None,
            "summary": None,
            "summary_raw_response": None,
            "summary_wall_s": None,
            "summary_input_tokens": None,
            "exec_error": None,
        }
        logger.info("----------step %d----------", self._step_idx)

        # ── 1. Observe ────────────────────────────────────────────
        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary

        before_ui_elements = state.ui_elements
        step_data["before_ui_elements"] = before_ui_elements

        before_ui_elements_text = a11y.generate_ui_elements_description_list(
            before_ui_elements, logical_screen_size,
        )
        step_data["before_ui_elements_text"] = before_ui_elements_text

        raw_screenshot = state.pixels.copy()
        step_data["raw_screenshot"] = raw_screenshot.copy()

        before_screenshot_with_som = a11y.draw_set_of_marks(
            raw_screenshot,
            before_ui_elements,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
        step_data["before_screenshot_with_som"] = before_screenshot_with_som.copy()

        # ── 2. Action selection (LLM call #1) ─────────────────────
        history_lines = [
            f"Step {i + 1}- {step_info['summary']}"
            for i, step_info in enumerate(self.history)
        ]
        action_prompt_text = build_action_prompt(
            goal,
            history_lines,
            before_ui_elements_text,
            self.additional_guidelines,
            l_c_prompt_text=self._l_c_prompt_text,
        )
        step_data["action_prompt"] = action_prompt_text

        raw_pil = Image.fromarray(raw_screenshot)
        som_pil = Image.fromarray(before_screenshot_with_som)
        action_messages = build_action_messages(
            action_prompt_text, raw_pil, som_pil,
        )

        try:
            if self._collect_log_probs:
                # GRPO / B4 path: get raw tensors + log-prob back from
                # generate_text, save replay to disk, record per-step
                # log-prob for later trajectory packaging.
                action_output, t_action, action_in_tok, extras = generate_text(
                    self._model,
                    self._processor,
                    action_messages,
                    images=[raw_pil, som_pil],
                    device=self._device,
                    generation_config=self._generation_config,
                    collect_log_probs=True,
                )
                self._step_log_probs.append(float(extras["step_log_prob"]))
                step_data["step_log_prob"] = float(extras["step_log_prob"])
                if self._replay_dir is not None:
                    import torch as _torch
                    replay_path = self._replay_dir / f"step_{self._step_idx}.pt"
                    # Save everything ``generate_text`` packaged for us,
                    # minus the scalar float (recomputable from the
                    # tensors) — this keeps replay forward-compatible
                    # with any future processor fields. ``input_len``
                    # stays even though it's an int: grpo.py needs it
                    # to locate the action-token slice.
                    replay_data = {
                        k: v for k, v in extras.items()
                        if k != "step_log_prob"
                    }
                    _torch.save(replay_data, replay_path)
                    step_data["replay_path"] = str(replay_path)
            else:
                action_output, t_action, action_in_tok = generate_text(
                    self._model,
                    self._processor,
                    action_messages,
                    images=[raw_pil, som_pil],
                    device=self._device,
                    generation_config=self._generation_config,
                )
                step_data["step_log_prob"] = None
        except Exception as e:  # pragma: no cover — hardware-level failures
            logger.exception("Action-phase generation failed at step %d", self._step_idx)
            step_data["summary"] = (
                f"Generation error during action selection: {type(e).__name__}: {e}. "
                "No action performed."
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        step_data["action_output"] = action_output
        step_data["action_raw_response"] = action_output
        step_data["action_wall_s"] = t_action
        step_data["action_input_tokens"] = action_in_tok
        logger.info(
            "action output (%.1fs, %d tokens): %s",
            t_action, action_in_tok, action_output[:200],
        )

        # ── 3. Parse Reason/Action ────────────────────────────────
        reason, action_str = m3a_utils.parse_reason_action_output(action_output)
        if not reason or not action_str:
            logger.info("Action output not in 'Reason: …\\nAction: …' format.")
            self._parse_failures += 1
            step_data["summary"] = (
                "Output for action selection is not in the correct format, so no"
                " action is performed."
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        logger.info("Reason: %s", reason)
        logger.info("Action: %s", action_str)
        step_data["action_reason"] = reason

        # Run our robust JSON parser on the Action: payload (handles fences,
        # alias drift, clamping). Returns ParseResult with .action filled.
        parse_res: ParseResult = parse_action(
            action_str,
            screen_width=logical_screen_size[0],
            screen_height=logical_screen_size[1],
        )
        if parse_res.aliased_from is not None:
            self._alias_hits += 1
        if parse_res.clamped:
            self._clamp_hits += 1

        if not parse_res.ok:
            logger.info("Failed to parse action JSON: %s", parse_res.error)
            self._parse_failures += 1
            step_data["summary"] = (
                "Can not parse the output to a valid action. Please make sure to pick"
                " the action from the list with required parameters (if any) in the"
                " correct JSON format!"
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        converted_action = parse_res.action
        step_data["action_output_json"] = converted_action

        # ── 4. Pre-execute validation (mirror M3A's index-range check) ──
        if (
            converted_action.action_type
            in (json_action.CLICK, json_action.LONG_PRESS,
                json_action.INPUT_TEXT, json_action.SCROLL)
            and converted_action.index is not None
        ):
            n_ui = len(before_ui_elements)
            if converted_action.index >= n_ui:
                logger.info(
                    "Index out of range (idx=%d, n_ui=%d).",
                    converted_action.index, n_ui,
                )
                step_data["summary"] = (
                    "The parameter index is out of range. Remember the index must be in"
                    " the UI element list!"
                )
                self.history.append(step_data)
                return base_agent.AgentInteractionResult(False, step_data)

            # Add a mark to the raw (unmarked) screenshot so the summary
            # phase can see WHICH element we touched — same behavior as
            # M3A's m3a_utils.add_ui_element_mark on step_data['raw_screenshot'].
            target = before_ui_elements[converted_action.index]
            if a11y.validate_ui_element(target, logical_screen_size):
                a11y.add_ui_element_mark(
                    step_data["raw_screenshot"],
                    target,
                    converted_action.index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )

        # ── 5. status / answer short-circuit ──────────────────────
        if converted_action.action_type == json_action.STATUS:
            self._self_reported += 1
            if converted_action.goal_status == "infeasible":
                logger.info("Agent stopped — mission deemed infeasible.")
            step_data["summary"] = "Agent thinks the request has been completed."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(True, step_data)

        if converted_action.action_type == json_action.ANSWER:
            logger.info("Agent answered: %s", converted_action.text)

        # ── 6. Execute on device ──────────────────────────────────
        try:
            self.env.execute_action(converted_action)
        except Exception as e:
            logger.exception("execute_action failed")
            step_data["exec_error"] = f"{type(e).__name__}: {e}"
            step_data["summary"] = (
                "Can not execute the action, make sure to select the action with"
                " the required parameters (if any) in the correct JSON format!"
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        # ── 7. Wait for the screen to stabilize (M3A's fixed sleep) ─
        time.sleep(self.wait_after_action_seconds)

        # ── 8. Grab after-state ───────────────────────────────────
        after_state = self.env.get_state(wait_to_stabilize=False)
        # Re-read geometry — orientation can change between turns (e.g. video apps).
        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary

        after_ui_elements = after_state.ui_elements
        step_data["after_ui_elements"] = after_ui_elements
        after_ui_elements_text = a11y.generate_ui_elements_description_list(
            after_ui_elements, logical_screen_size,
        )
        step_data["after_ui_elements_text"] = after_ui_elements_text

        after_screenshot_with_som = a11y.draw_set_of_marks(
            after_state.pixels.copy(),
            after_ui_elements,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )

        # Burn 'before' / 'after' text labels onto the two SoM frames so
        # the summary-phase model can tell them apart.
        a11y.add_screenshot_label(step_data["before_screenshot_with_som"], "before")
        a11y.add_screenshot_label(after_screenshot_with_som, "after")
        step_data["after_screenshot_with_som"] = after_screenshot_with_som.copy()

        # ── 9. Summary (LLM call #2) ──────────────────────────────
        summary_prompt_text = build_summary_prompt(
            action_str,
            reason,
            goal,
            before_ui_elements_text,
            after_ui_elements_text,
        )
        step_data["summary_prompt"] = summary_prompt_text

        before_pil = Image.fromarray(step_data["before_screenshot_with_som"])
        after_pil = Image.fromarray(after_screenshot_with_som)
        summary_messages = build_summary_messages(
            summary_prompt_text, before_pil, after_pil,
        )

        try:
            summary_text, t_summary, summary_in_tok = generate_text(
                self._model,
                self._processor,
                summary_messages,
                images=[before_pil, after_pil],
                device=self._device,
                generation_config=self._generation_config,
            )
        except Exception as e:  # pragma: no cover
            logger.exception("Summary-phase generation failed at step %d", self._step_idx)
            summary_text = (
                f"[summary generation crashed: {type(e).__name__}: {e}]"
            )
            t_summary = 0.0
            summary_in_tok = 0

        step_data["summary_raw_response"] = summary_text
        step_data["summary_wall_s"] = t_summary
        step_data["summary_input_tokens"] = summary_in_tok
        # Mirror M3A: the saved summary is "Action selected: <action>. <model summary>".
        step_data["summary"] = f"Action selected: {action_str}. {summary_text.strip()}"
        logger.info(
            "summary (%.1fs, %d tokens): %s",
            t_summary, summary_in_tok, summary_text.strip()[:200],
        )

        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)

    # ── Trajectory persistence (Story 2.0 — Epic 2 prep) ─────────────

    def save_episode(
        self,
        output_dir: Any,                  # str | pathlib.Path
        *,
        success: float,
        template: str,
        seed: int,
        app: str,
        tier: str,
    ) -> Any:                              # pathlib.Path of the per-episode dir
        """Persist the just-completed episode's trajectory to disk.

        Layout (matches Story 2.0 spec):
            {output_dir}/{template}_seed{seed}/
                meta.json          — episode-level metadata
                episode.jsonl      — one JSON line per step
                step_{i}_before.png   — SoM-marked screenshot at step start
                step_{i}_after.png    — SoM-marked screenshot after action
                                         (omitted for early-exit steps:
                                          status emit / parse fail / exec fail)

        The per-step `reward` is sparse-terminal: 0.0 for all but the last
        step, which gets the final ``success`` (so 0.0 / 0.5 / 1.0 from
        AndroidWorld's rule-based ``task.is_successful(env)``).

        Must be called AFTER ``harness.run_template`` returns (so ``success``
        is known) and BEFORE the next episode's ``agent.reset()`` (which
        would wipe ``self.history``). The sweep script wires this in.

        Returns the per-episode directory path (or None if history is empty).
        """
        import pathlib

        if not self.history:
            logger.warning(
                "save_episode called but agent.history is empty for %s; skipping.",
                template,
            )
            return None

        ep_dir = pathlib.Path(output_dir) / f"{template}_seed{seed}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        n_steps = len(self.history)
        meta = {
            "schema_version": 1,           # bump if JSONL field set changes
            "template": template,
            "seed": seed,
            "app": app,
            "tier": tier,
            "success": float(success),
            "n_steps": n_steps,
            "self_reported": int(self._self_reported),
            "parse_failures": int(self._parse_failures),
            "alias_hits": int(self._alias_hits),
            "clamp_hits": int(self._clamp_hits),
            "agent_name": self.name,
            "wall_s_total": sum(
                (s.get("action_wall_s") or 0.0) + (s.get("summary_wall_s") or 0.0)
                for s in self.history
            ),
        }
        with (ep_dir / "meta.json").open("w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)

        with (ep_dir / "episode.jsonl").open("w") as fh:
            for i, sd in enumerate(self.history):
                is_last = (i == n_steps - 1)
                step_idx = sd.get("step", i + 1)

                # Screenshots — write PNG side-files; missing 'after' for
                # early-exit steps is normal (status / parse-fail / exec-fail).
                before_path = None
                after_path = None
                before_arr = sd.get("before_screenshot_with_som")
                if before_arr is not None:
                    fname = f"step_{step_idx}_before.png"
                    Image.fromarray(before_arr).save(ep_dir / fname)
                    before_path = fname
                after_arr = sd.get("after_screenshot_with_som")
                if after_arr is not None:
                    fname = f"step_{step_idx}_after.png"
                    Image.fromarray(after_arr).save(ep_dir / fname)
                    after_path = fname

                # Action — JSONAction.as_dict if parsed, else None.
                action_obj = sd.get("action_output_json")
                action_dict = (
                    action_obj.as_dict(skip_none=True)
                    if action_obj is not None and hasattr(action_obj, "as_dict")
                    else None
                )

                # Parse error inference: model output exists but JSON
                # didn't parse to a JSONAction. The synthetic 'summary'
                # written at parse-fail time captures the error message.
                parse_error = None
                if (
                    action_dict is None
                    and sd.get("action_output") is not None
                    and sd.get("exec_error") is None
                ):
                    parse_error = sd.get("summary")

                row = {
                    "step": step_idx,
                    "timestamp": sd.get("timestamp"),
                    "goal": sd.get("goal"),
                    "before_ui_elements_text": sd.get("before_ui_elements_text"),
                    "after_ui_elements_text": sd.get("after_ui_elements_text"),
                    "before_screenshot_path": before_path,
                    "after_screenshot_path": after_path,
                    "action": action_dict,
                    "action_reason": sd.get("action_reason"),
                    "summary": sd.get("summary"),
                    "reward": float(success) if is_last else 0.0,
                    "action_wall_s": sd.get("action_wall_s"),
                    "summary_wall_s": sd.get("summary_wall_s"),
                    "action_input_tokens": sd.get("action_input_tokens"),
                    "parse_error": parse_error,
                    "exec_error": sd.get("exec_error"),
                    "action_raw_response": sd.get("action_output"),
                    "summary_raw_response": sd.get("summary_raw_response"),
                }
                fh.write(json.dumps(row, ensure_ascii=False, default=str))
                fh.write("\n")

        logger.info(
            "save_episode: wrote %d steps to %s (success=%.2f)",
            n_steps, ep_dir, success,
        )
        return ep_dir


__all__ = [
    "GenerationConfig",
    "Qwen3VLAgent",
    "generate_text",
]
