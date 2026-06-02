"""Prompts for EvoFSM-RL Android agent — Story 1.5 (Qwen3-VL-M3A baseline).

This module is intentionally a near-verbatim port of
``android_world.agents.m3a``'s text prompts. The only deltas vs. upstream
M3A are:
  * String constants are duplicated here (rather than re-exported) so we
    can swap them out for ablations / fine-tuning without touching the
    AndroidWorld submodule.
  * ``build_action_messages`` / ``build_summary_messages`` package the
    rendered prompt + screenshots into Qwen3-VL's multimodal chat-template
    format. M3A speaks the OpenAI Vision schema; Qwen3-VL speaks
    ``[{"type": "image", "image": ...}, {"type": "text", "text": ...}]``.

Two-phase per-step inference (matches M3A 1:1):

  Phase A — action selection
      input:  PROMPT_PREFIX + goal + history + ui-elements + GUIDANCE +
              "Reason: ... Action: {...}" instruction
              + [raw_screenshot, before_screenshot_with_som]
      output: free-form "Reason: <why>\\nAction: <json>"

  Phase B — summarization
      input:  PROMPT_PREFIX + goal + before/after ui-elements +
              chosen action + the model's own reason
              + [before_screenshot_with_som, after_screenshot_with_som]
      output: 1-line natural-language summary of what happened.
              That summary becomes the next step's history entry.

Why two phases (and why we abandoned baseline_v0's single-phase JSON-only
prompt): see analysis in ``traces/baseline_50task_v02_a11y/`` — without a
post-action reflection step, the 8B VLM never decides "the goal is done"
because its history contains only past JSON dicts, not outcomes. M3A's
summary loop is the load-bearing piece for ``status: complete`` emission.
"""

from __future__ import annotations

from typing import Any


# ─────────────────────────────────────────────────────────────────────────
# M3A prompt strings (verbatim from android_world.agents.m3a)
# ─────────────────────────────────────────────────────────────────────────
# We duplicate rather than re-export so:
#   1. Future ablations (FSM-conditioned guidance, RL-curriculum prompts)
#      can swap these strings without touching the AndroidWorld submodule.
#   2. Fine-tuning on M3A-style targets reads the canonical text from one
#      place — this file.
# Doubled braces ({{...}}) are mandatory: these strings get .format()-ed
# downstream to inject {goal}, {history}, {ui_elements}, etc.
# ─────────────────────────────────────────────────────────────────────────


PROMPT_PREFIX = (
    'You are an agent who can operate an Android phone on behalf of a user.'
    " Based on user's goal/request, you may\n"
    '- Answer back if the request/goal is a question (or a chat message),'
    ' like user asks "What is my schedule for today?".\n'
    '- Complete some tasks described in the requests/goals by'
    ' performing actions (step by step) on the phone.\n\n'
    'When given a user request, you will try to complete it step by step.'
    ' At each step, you will be given the current screenshot (including the'
    ' original screenshot and the same screenshot with bounding'
    ' boxes and numeric indexes added to some UI elements) and a history of'
    ' what you have done (in text). Based on these pieces of information and'
    ' the goal, you must choose to perform one of the'
    ' action in the following list (action description followed by the JSON'
    ' format) by outputing the action in the correct JSON format.\n'
    '- If you think the task has been completed, finish the task by using the'
    ' status action with complete as goal_status:'
    ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
    "- If you think the task is not feasible (including cases like you don't"
    ' have enough information or can not perform some necessary actions),'
    ' finish by using the `status` action with infeasible as goal_status:'
    ' `{{"action_type": "status", "goal_status": "infeasible"}}`\n'
    "- Answer user's question:"
    ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
    '- Click/tap on an element on the screen. We have added marks (bounding'
    ' boxes with numeric indexes on their TOP LEFT corner) to most of the UI'
    ' elements in the screenshot, use the numeric index to indicate which'
    ' element you want to click:'
    ' `{{"action_type": "click", "index": <target_index>}}`.\n'
    '- Long press on an element on the screen, similar with the click action'
    ' above, use the numeric label on the bounding box to indicate which'
    ' element you want to long press:'
    ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
    '- Type text into a text field (this action contains clicking the text'
    ' field, typing in the text and pressing the enter, so no need to click on'
    ' the target field to start), use the numeric label'
    ' on the bounding box to indicate the target text field:'
    ' `{{"action_type": "input_text", "text": <text_input>,'
    ' "index": <target_index>}}`\n'
    '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
    '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
    '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
    '- Scroll the screen or a scrollable UI element in one of the four'
    ' directions, use the same numeric index as above if you want to scroll a'
    ' specific UI element, leave it empty when scroll the whole screen:'
    ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
    ' "index": <optional_target_index>}}`\n'
    '- Open an app (nothing will happen if the app is not'
    ' installed): `{{"action_type": "open_app", "app_name": <name>}}`\n'
    '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
)


GUIDANCE = (
    'Here are some useful guidelines you need to follow:\n'
    'General:\n'
    '- Usually there will be multiple ways to complete a task, pick the'
    ' easiest one. Also when something does not work as expected (due'
    ' to various reasons), sometimes a simple retry can solve the problem,'
    " but if it doesn't (you can see that from the history),"
    ' SWITCH to other solutions.\n'
    '- Sometimes you may need to navigate the phone to gather information'
    ' needed to complete the task, for example if user asks'
    ' "what is my schedule tomorrow", then you may want to open the calendar'
    ' app (using the `open_app` action), look up information there, answer'
    " user's question (using the `answer` action) and finish (using"
    ' the `status` action with complete as goal_status).\n'
    '- For requests that are questions (or chat messages), remember to use'
    ' the `answer` action to reply to user explicitly before finish!'
    ' Merely displaying the answer on the screen is NOT sufficient (unless'
    ' the goal is something like "show me ...").\n'
    '- If the desired state is already achieved (e.g., enabling Wi-Fi when'
    " it's already on), you can just complete the task.\n"
    'Action Related:\n'
    '- Use the `open_app` action whenever you want to open an app'
    ' (nothing will happen if the app is not installed), do not use the'
    ' app drawer to open an app unless all other ways have failed.\n'
    '- Use the `input_text` action whenever you want to type'
    ' something (including password) instead of clicking characters on the'
    ' keyboard one by one. Sometimes there is some default text in the text'
    ' field you want to type in, remember to delete them before typing.\n'
    '- For `click`, `long_press` and `input_text`, the index parameter you'
    ' pick must be VISIBLE in the screenshot and also in the UI element'
    ' list given to you (some elements in the list may NOT be visible on'
    ' the screen so you can not interact with them).\n'
    '- Consider exploring the screen by using the `scroll`'
    ' action with different directions to reveal additional content.\n'
    '- The direction parameter for the `scroll` action can be confusing'
    " sometimes as it's opposite to swipe, for example, to view content at the"
    ' bottom, the `scroll` direction should be set to "down". It has been'
    ' observed that you have difficulties in choosing the correct direction, so'
    ' if one does not work, try the opposite as well.\n'
    'Text Related Operations:\n'
    '- Normally to select certain text on the screen: <i> Enter text selection'
    ' mode by long pressing the area where the text is, then some of the words'
    ' near the long press point will be selected (highlighted with two pointers'
    ' indicating the range) and usually a text selection bar will also appear'
    ' with options like `copy`, `paste`, `select all`, etc.'
    ' <ii> Select the exact text you need. Usually the text selected from the'
    ' previous step is NOT the one you want, you need to adjust the'
    ' range by dragging the two pointers. If you want to select all text in'
    ' the text field, simply click the `select all` button in the bar.\n'
    "- At this point, you don't have the ability to drag something around the"
    ' screen, so in general you can not select arbitrary text.\n'
    '- To delete some text: the most traditional way is to place the cursor'
    ' at the right place and use the backspace button in the keyboard to'
    ' delete the characters one by one (can long press the backspace to'
    ' accelerate if there are many to delete). Another approach is to first'
    ' select the text you want to delete, then click the backspace button'
    ' in the keyboard.\n'
    '- To copy some text: first select the exact text you want to copy, which'
    ' usually also brings up the text selection bar, then click the `copy`'
    ' button in bar.\n'
    '- To paste text into a text box, first long press the'
    ' text box, then usually the text selection bar will appear with a'
    ' `paste` button in it.\n'
    '- When typing into a text field, sometimes an auto-complete dropdown'
    ' list will appear. This usually indicating this is a enum field and you'
    ' should try to select the best match by clicking the corresponding one'
    ' in the list.\n'
)


ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '{l_c_section}'  # empty by default; L_C-injection fills this slot. See build_action_prompt.
    + '\nThe current user goal/request is: {goal}\n\n'
    'Here is a history of what you have done so far:\n{history}\n\n'
    'The current screenshot and the same screenshot with bounding boxes'
    ' and labels added are also given to you.\n'
    'Here is a list of detailed'
    ' information for some of the UI elements (notice that some elements in'
    ' this list may not be visible in the current screen and so you can not'
    ' interact with it, can try to scroll the screen to reveal it first),'
    ' the numeric indexes are'
    ' consistent with the ones in the labeled screenshot:\n{ui_elements}\n'
    + GUIDANCE
    + '{additional_guidelines}'
    + '\nNow output an action from the above list in the correct JSON format,'
    ' following the reason why you do that. Your answer should look like:\n'
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    'Your Answer:\n'
)


SUMMARY_PROMPT_TEMPLATE = (
    PROMPT_PREFIX
    + '\nThe (overall) user goal/request is: {goal}\n'
    'Now I want you to summerize the latest step.\n'
    'You will be given the screenshot before you performed the action (which'
    ' has a text label "before" on the bottom right), the action you chose'
    ' (together with the reason) and the screenshot after the action was'
    ' performed (which has a text label "after" on the bottom right).\n'
    'Also here is the list of detailed information for some UI elements'
    ' in the before screenshot:\n{before_elements}\n'
    'Here is the list for the after screenshot:\n{after_elements}\n'
    'This is the action you picked: {action}\n'
    'Based on the reason: {reason}\n\n'
    'By comparing the two screenshots (plus the UI element lists) and the'
    ' action performed, give a brief summary of this step. This summary'
    ' will be added to action history and used in future action selection,'
    ' so try to include essential information you think that will be most'
    ' useful for future action selections like what you'
    ' intended to do, why, if it worked as expected, if not'
    ' what might be the reason (be critical, the action/reason might be'
    ' wrong), what should/should not be done next and so on. Some more'
    ' rules/tips you should follow:\n'
    '- Keep it short (better less than 50 words) and in a single line\n'
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    ' you can just assume they work as expected.\n'
    '- Given this summary will be added into action history, it can be used as'
    ' memory to include information that needs to be remembered, or shared'
    ' between different apps.\n\n'
    'Summary of this step: '
)


# ─────────────────────────────────────────────────────────────────────────
# Prompt-string builders (mirror m3a._action_selection_prompt /
# m3a._summarize_prompt)
# ─────────────────────────────────────────────────────────────────────────


L_C_SECTION_HEADER = "# Workflow knowledge (transferred from related apps)"
L_C_SECTION_INTRO = (
    "The following abstract workflow patterns were learned from similar apps in the\n"
    "same category. They describe general strategies, common failure modes, and\n"
    "verification signals. They do NOT describe this specific app's UI — you must\n"
    "still rely on the screenshot to identify actual elements."
)


def _render_l_c_section(l_c_prompt_text: str | None) -> str:
    """Wrap a rendered L_C Layer-2 block with the section header + intro.

    Returns the empty string when ``l_c_prompt_text`` is ``None`` (B1
    path — the empty slot is byte-equivalent to the pre-injection
    template output). Returns a block starting with two newlines so
    that the section visually separates from the preceding
    ``PROMPT_PREFIX``.
    """
    if l_c_prompt_text is None:
        return ""
    return (
        "\n\n"
        + L_C_SECTION_HEADER
        + "\n\n"
        + L_C_SECTION_INTRO
        + "\n\n"
        + l_c_prompt_text
        + "\n"
    )


def build_action_prompt(
    goal: str,
    history: list[str],
    ui_elements: str,
    additional_guidelines: list[str] | None = None,
    *,
    l_c_prompt_text: str | None = None,
) -> str:
    """Render the text portion of the per-step action-selection prompt.

    Args:
        goal: The user's task goal (from ``task.goal``).
        history: Per-step natural-language summaries from prior turns.
            Pass ``[]`` for turn 1.
        ui_elements: Pre-rendered UI element description list (one
            element per line, in the verbose JSON-per-element format
            from ``a11y.generate_ui_elements_description_list``).
        additional_guidelines: Optional task-specific guideline lines.
        l_c_prompt_text: Optional pre-rendered LAYER-2 block (from
            ``Layer2.to_prompt_text``) describing abstract workflow
            knowledge transferred from same-category source-pool apps.
            When ``None`` (default), the prompt is byte-identical to
            the B1 zero-shot baseline. When provided, a
            ``# Workflow knowledge`` section is spliced in right after
            ``PROMPT_PREFIX`` so the injected content sits in the
            stable prefix (KV-cache friendly across steps of one
            episode) and ahead of the episode-specific goal/history/UI.
    """
    if history:
        history_str = '\n'.join(history)
    else:
        history_str = 'You just started, no action has been performed yet.'

    extra_guidelines = ''
    if additional_guidelines:
        extra_guidelines = 'For The Current Task:\n'
        for guideline in additional_guidelines:
            extra_guidelines += f'- {guideline}\n'

    return ACTION_SELECTION_PROMPT_TEMPLATE.format(
        l_c_section=_render_l_c_section(l_c_prompt_text),
        goal=goal,
        history=history_str,
        ui_elements=ui_elements if ui_elements else 'Not available',
        additional_guidelines=extra_guidelines,
    )


def build_summary_prompt(
    action: str,
    reason: str,
    goal: str,
    before_elements: str,
    after_elements: str,
) -> str:
    """Render the text portion of the per-step summarization prompt."""
    return SUMMARY_PROMPT_TEMPLATE.format(
        goal=goal,
        before_elements=before_elements,
        after_elements=after_elements,
        action=action,
        reason=reason,
    )


# ─────────────────────────────────────────────────────────────────────────
# Qwen3-VL chat-template message packers
# ─────────────────────────────────────────────────────────────────────────
# Qwen3-VL's processor.apply_chat_template expects a list-of-dicts where
# user content is a list of typed parts. M3A passes images directly to the
# OpenAI vision endpoint; we package them into Qwen's conventional format.
# ─────────────────────────────────────────────────────────────────────────


def build_action_messages(
    prompt_text: str,
    raw_screenshot,           # PIL.Image (no SoM marks)
    som_screenshot,           # PIL.Image (same screenshot + green boxes/indices)
) -> list[dict[str, Any]]:
    """Pack the action-selection prompt + 2 screenshots for Qwen3-VL.

    Mirrors M3A's ``llm.predict_mm(action_prompt, [raw, som])`` —
    the two images are sent in the same order so the model sees the
    raw screen first, then the labelled overlay.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": raw_screenshot},
                {"type": "image", "image": som_screenshot},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def build_summary_messages(
    prompt_text: str,
    before_screenshot,        # PIL.Image (with SoM + 'before' label)
    after_screenshot,         # PIL.Image (with SoM + 'after' label)
) -> list[dict[str, Any]]:
    """Pack the summarization prompt + before/after screenshots for Qwen3-VL.

    Mirrors M3A's ``llm.predict_mm(summary_prompt, [before, after])``.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": before_screenshot},
                {"type": "image", "image": after_screenshot},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


__all__ = [
    "PROMPT_PREFIX",
    "GUIDANCE",
    "ACTION_SELECTION_PROMPT_TEMPLATE",
    "SUMMARY_PROMPT_TEMPLATE",
    "L_C_SECTION_HEADER",
    "L_C_SECTION_INTRO",
    "build_action_prompt",
    "build_summary_prompt",
    "build_action_messages",
    "build_summary_messages",
]
