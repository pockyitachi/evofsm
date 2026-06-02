# Patch layer on top of androidworld:evofsm — adds android_world_plus
# extension app tasks (6 apps / 77 tasks) without touching the 2026+swipe base.
# JSONL task_id 0-115 preserved; new tasks occupy 116-192 (android family → 193).
FROM androidworld:evofsm

# 6 new app task modules + dense-reward / pysqlite3 / activity-mapping merges.
# COPY merges into the existing android_world package (overwrites matching
# files, adds new ones); base files not listed here stay untouched.
COPY patch_aw/ /usr/local/lib/python3.11/site-packages/android_world/

# registry_ext.py: imports the 6 modules + appends 77 tasks after index 115.
COPY patch_skyrl/registry_ext.py /data/skyrl_server/registry_ext.py
