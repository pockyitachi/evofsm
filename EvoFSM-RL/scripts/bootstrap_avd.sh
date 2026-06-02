#!/usr/bin/env bash
# bootstrap_avd.sh — Create a Pixel_6_API_33 AVD for EvoFSM-RL.
#
# Usage:
#   ./scripts/bootstrap_avd.sh          # auto-detect arch
#   ./scripts/bootstrap_avd.sh arm64    # force arm64 (Apple Silicon Mac)
#   ./scripts/bootstrap_avd.sh x86_64   # force x86_64 (Linux / cloud GPU box)
#
# Prerequisites:
#   - ANDROID_HOME set (e.g. ~/Library/Android/sdk)
#   - sdkmanager, avdmanager, emulator on PATH
#
# Per ADR-001: native AVD on both hosts. No Docker.

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────
AVD_NAME="Pixel_6_API_33"
API_LEVEL="33"
DEVICE="pixel_6"          # avdmanager device id
RAM_MB="4096"
HEAP_MB="512"

# ── Detect or accept arch ────────────────────────────────────────────
detect_arch() {
    local machine
    machine="$(uname -m)"
    case "$machine" in
        arm64|aarch64) echo "arm64-v8a" ;;
        x86_64|amd64)  echo "x86_64" ;;
        *)
            echo "ERROR: unsupported architecture '$machine'" >&2
            exit 1
            ;;
    esac
}

if [[ "${1:-}" == "arm64" ]]; then
    ABI="arm64-v8a"
elif [[ "${1:-}" == "x86_64" ]]; then
    ABI="x86_64"
elif [[ -z "${1:-}" ]]; then
    ABI="$(detect_arch)"
else
    echo "Usage: $0 [arm64|x86_64]" >&2
    exit 1
fi

SYSTEM_IMAGE="system-images;android-${API_LEVEL};google_apis;${ABI}"

echo "=== EvoFSM-RL AVD Bootstrap ==="
echo "  AVD name:     $AVD_NAME"
echo "  ABI:          $ABI"
echo "  System image: $SYSTEM_IMAGE"
echo "  RAM:          ${RAM_MB} MB"
echo "  Heap:         ${HEAP_MB} MB"
echo ""

# ── Preflight checks ────────────────────────────────────────────────
check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' not found on PATH." >&2
        echo "  Ensure ANDROID_HOME is set and tools are on PATH." >&2
        echo "  See EvoFSM-RL/docs/adr/001-emulator-path.md" >&2
        exit 1
    fi
}

check_tool sdkmanager
check_tool avdmanager
check_tool emulator

if [[ -z "${ANDROID_HOME:-}" ]]; then
    echo "WARNING: ANDROID_HOME not set. Trying to infer from sdkmanager path."
    ANDROID_HOME="$(dirname "$(dirname "$(command -v sdkmanager)")")"
    export ANDROID_HOME
    echo "  Inferred ANDROID_HOME=$ANDROID_HOME"
fi

# ── Accept licenses (non-interactive) ────────────────────────────────
echo "--- Accepting SDK licenses ---"
yes 2>/dev/null | sdkmanager --licenses >/dev/null 2>&1 || true

# ── Install system image + platform ──────────────────────────────────
echo "--- Installing system image (may take a few minutes on first run) ---"
sdkmanager --install "$SYSTEM_IMAGE" "platforms;android-${API_LEVEL}" "platform-tools" "emulator"

# ── Create AVD (idempotent — delete if exists) ───────────────────────
if avdmanager list avd 2>/dev/null | grep -q "Name: ${AVD_NAME}"; then
    echo "--- AVD '$AVD_NAME' already exists. Deleting and re-creating. ---"
    avdmanager delete avd --name "$AVD_NAME"
fi

echo "--- Creating AVD '$AVD_NAME' ---"
echo "no" | avdmanager create avd \
    --name "$AVD_NAME" \
    --package "$SYSTEM_IMAGE" \
    --device "$DEVICE" \
    --force

# ── Patch config.ini for RAM / heap ─────────────────────────────────
AVD_DIR="${HOME}/.android/avd/${AVD_NAME}.avd"
CONFIG_INI="${AVD_DIR}/config.ini"

if [[ -f "$CONFIG_INI" ]]; then
    echo "--- Patching config.ini (RAM=${RAM_MB}, heap=${HEAP_MB}) ---"
    # Remove existing lines if present, then append
    sed -i.bak '/^hw\.ramSize/d; /^vm\.heapSize/d; /^hw\.gpu\.enabled/d; /^hw\.gpu\.mode/d' "$CONFIG_INI"
    cat >> "$CONFIG_INI" <<EOF
hw.ramSize=${RAM_MB}
vm.heapSize=${HEAP_MB}
hw.gpu.enabled=yes
hw.gpu.mode=auto
EOF
    rm -f "${CONFIG_INI}.bak"
fi

# ── Verify ───────────────────────────────────────────────────────────
echo ""
echo "=== Done! ==="
echo ""
echo "AVD '$AVD_NAME' is ready. To start it:"
echo ""
echo "  emulator -avd $AVD_NAME -no-snapshot-load -gpu auto"
echo ""
echo "To verify from Python:"
echo ""
echo "  from android_world.env import env_launcher"
echo "  env = env_launcher.load_and_setup_env(console_port=5554, grpc_port=8554)"
echo ""
avdmanager list avd 2>/dev/null | grep -A2 "Name: ${AVD_NAME}" || true
