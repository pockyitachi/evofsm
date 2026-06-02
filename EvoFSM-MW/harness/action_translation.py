"""Translate Qwen3-VL `mobile_use` actions <-> AndroidWorld `JSONAction`.

SCAFFOLD — not implemented yet.

The env (`androidworld:evofsm-tasks193`) executes coordinate actions natively
(verified: `actuation.py` click/long_press/double_tap via x,y; swipe via
`direction=[x,y,x2,y2]`). So this layer is a thin field-name mapper, NOT env work.

mobile_use action            -> JSONAction
  click {coordinate:[x,y]}     -> {action_type:"click", x, y}
  long_press {coordinate}      -> {action_type:"long_press", x, y, (duration)}
  swipe {coordinate,coordinate2} -> {action_type:"swipe", direction:[x,y,x2,y2]}
  type {text}                  -> {action_type:"input_text", text}
  answer {text}                -> {action_type:"answer", text}
  system_button Back/Home/Menu/Enter -> navigate_back / navigate_home / keycode
  wait                         -> {action_type:"wait"}
  terminate {status}           -> {action_type:"status", goal_status}

Note: mobile_use coords are in a 999x999 space; scale to the device screen
(AVD is 2400x1080) before emitting x,y — mirror MobileWorld's qwen3vl agent
rel->abs conversion.
"""

raise NotImplementedError("EvoFSM-MW harness.action_translation — scaffold only")
