# docker/ — containerized AndroidWorld emulator for the verl/SkyRL spike

A thin Docker layer that bakes **our** canonical AVD (`AWAvd2`) into a
pre-built AndroidWorld emulator base image, so containerized rollouts run the
same environment that produced the B1–B4 numbers.

## Layout

| Path | Role |
|---|---|
| `evofsm/Dockerfile` | Single-stage image. `FROM androidworld:2026plusswipe` (ziqiang's base: full SDK + emulator 35.3.11.0 + `android_world` + `skyrl_server`); copies in `AWAvd2.avd` + `AWAvd2.ini` and rewrites the ini's `path=` to the in-container location. No code patched — runtime selection is all via env vars. |

## Usage

Build (context = `android-sdk/avd`, which holds `AWAvd2.avd` + `AWAvd2.ini`):

```bash
docker build -f EvoFSM-RL/docker/evofsm/Dockerfile -t androidworld:evofsm android-sdk/avd
```

Run — the base image picks the AVD/snapshot from env vars (no code changes):

```bash
AVD_NAME=AWAvd2  ENV_SNAPSHOT=apps_ready_dec2025  ADB_SERVER_PORT=<private>  ...
```

The image only swaps in our AVD; everything else (SDK, emulator binary,
`android_world`, `skyrl_server`) comes from the `androidworld:2026plusswipe`
base. See `CLAUDE.md` in this directory for working context, and `../CLAUDE.md`
for the project-wide picture.
