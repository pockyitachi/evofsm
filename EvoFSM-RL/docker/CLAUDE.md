# CLAUDE.md — docker/ working context

Container build for the verl/SkyRL emulator spike. See `../CLAUDE.md` for the
project-wide picture; this file is only what you need when working in `docker/`.

## File map
- `evofsm/Dockerfile` — the only file here. Layers our AVD onto a pre-built
  AndroidWorld base. `FROM androidworld:2026plusswipe` → `COPY AWAvd2.{ini,avd}`
  → `sed` the ini's `path=` to `/root/.android/avd/AWAvd2.avd` (with a `grep`
  assert that the rewrite landed). Tagged `androidworld:evofsm`.

## Gotchas
- **Build context is `android-sdk/avd`, not the repo root.** The `COPY` paths
  (`AWAvd2.ini`, `AWAvd2.avd`) are relative to that dir; build from elsewhere
  and the copy fails. Invoke with `-f EvoFSM-RL/docker/evofsm/Dockerfile ...
  android-sdk/avd` (see README).
- **Base image is external** (ziqiang's `androidworld:2026plusswipe`: full SDK +
  emulator 35.3.11.0 + `android_world` + `skyrl_server`). It must already be in
  the local Docker registry — this Dockerfile does not build it.
- **No code is patched.** Runtime selection (which AVD, which snapshot, ADB
  port) is entirely via env vars at `docker run` — `AVD_NAME`, `ENV_SNAPSHOT`,
  `ADB_SERVER_PORT`. Don't add `RUN` steps to hard-code these.
- The in-container AVD lives at `/root/.android/avd/AWAvd2.avd`; the `sed`
  rewrite of `path=` is load-bearing because the host ini points at the
  on-disk linqiang path, which doesn't exist inside the container.
- The baked snapshot is `apps_ready_dec2025` (the canonical AVD with all apps);
  see `../CLAUDE.md` "Canonical AVD" for what's inside it.
