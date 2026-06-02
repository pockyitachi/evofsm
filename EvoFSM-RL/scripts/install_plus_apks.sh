#!/usr/bin/env bash
# install_plus_apks.sh — Install the 6 Plus-repo apps on a running AVD.
#
# Why a separate script: vanilla AndroidWorld's setup.py only knows about
# its own 17 apps (see `_APPS` in android_world/env/setup_device/setup.py).
# The 6 Plus-repo apps (bluecoins, calculator, maps_me, pi_music, snapseed,
# wikipedia) come from BMOCA / AndroidLab and must be sideloaded manually.
# The original repo author stored them at a hardcoded Linux path in
# bmoca_apps.py; we replace that with a portable, check-in-able folder.
#
# Supports BOTH formats from APKMirror:
#   .apk   — single-file installable APK  (preferred)
#   .apkm  — APKMirror's zipped Android App Bundle (split APKs)
#            → we unzip + adb install-multiple automatically
#
# Usage:
#   ./scripts/install_plus_apks.sh               # install from EvoFSM-RL/apks/
#   ./scripts/install_plus_apks.sh /some/folder  # install from custom folder
#
# Prerequisites:
#   - A running emulator (emulator -avd Pixel_6_API_33 -grpc 8554)
#   - adb on PATH
#   - unzip on PATH (macOS and Linux have it by default)
#   - All 6 APKs/APKMs in the apks/ folder (see apks/README.md)

set -euo pipefail

# ── Required packages (authoritative list — DO NOT change without ──
# ── updating android_world/task_evals/single/*.py expectations) ──
REQUIRED_PACKAGES=(
    "com.google.android.calculator"        # Google Calculator
    "org.wikipedia"                        # Wikipedia
    "com.rammigsoftware.bluecoins"         # Bluecoins
    "com.mapswithme.maps.pro"              # MAPS.ME
    "com.Project100Pi.themusicplayer"      # Pi Music Player
    "com.niksoftware.snapseed"             # Snapseed
)

# ── Locate APK folder ─────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APK_DIR="${1:-${SCRIPT_DIR}/../apks}"

if [[ ! -d "$APK_DIR" ]]; then
    echo "ERROR: APK folder not found: $APK_DIR" >&2
    echo "  Create it with APKs listed in EvoFSM-RL/apks/README.md" >&2
    exit 1
fi

# ── Preflight: adb + unzip ────────────────────────────────────────
if ! command -v adb &>/dev/null; then
    echo "ERROR: adb not found on PATH." >&2
    exit 1
fi
if ! command -v unzip &>/dev/null; then
    echo "ERROR: unzip not found on PATH (needed for .apkm)." >&2
    exit 1
fi

if ! adb get-state 2>/dev/null | grep -q device; then
    echo "ERROR: No device connected. Start the emulator first:" >&2
    echo "  emulator -avd Pixel_6_API_33 -grpc 8554" >&2
    exit 1
fi

# ── Scratch dir for .apkm extraction ──────────────────────────────
TMP_DIR="$(mktemp -d -t evofsm_apkm.XXXXXX)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

# ── Helper: install a single .apk via `adb install` ──────────────
install_apk() {
    local apk="$1"
    echo "--- Installing $(basename "$apk") ---"
    # -r: replace existing  -g: grant all runtime permissions  -t: allow test
    if adb install -r -g -t "$apk"; then
        echo "  ✅ OK"
        return 0
    else
        echo "  ❌ FAILED"
        return 1
    fi
}

# ── Helper: extract .apkm and install via `adb install-multiple` ──
# APKMirror .apkm files are zip archives containing:
#   base.apk, split_config.<arch>.apk, split_config.<dpi>.apk, split_<lang>.apk
# `adb install-multiple` needs all splits passed together.
install_apkm() {
    local apkm="$1"
    local name="$(basename "$apkm" .apkm)"
    local extract_dir="$TMP_DIR/$name"
    echo "--- Installing $(basename "$apkm") (via install-multiple) ---"

    mkdir -p "$extract_dir"
    if ! unzip -qo "$apkm" -d "$extract_dir"; then
        echo "  ❌ FAILED: could not unzip .apkm"
        return 1
    fi

    # Gather all .apk files inside the extracted archive
    shopt -s nullglob
    local splits=("$extract_dir"/*.apk)
    shopt -u nullglob

    if [[ ${#splits[@]} -eq 0 ]]; then
        echo "  ❌ FAILED: no .apk splits inside $name.apkm"
        return 1
    fi

    echo "  Splits: ${#splits[@]}"
    for s in "${splits[@]}"; do echo "    - $(basename "$s")"; done

    if adb install-multiple -r -g -t "${splits[@]}"; then
        echo "  ✅ OK"
        return 0
    else
        echo "  ❌ FAILED"
        return 1
    fi
}

# ── Enumerate inputs (both .apk and .apkm) ────────────────────────
echo "=== Installing Plus-repo APKs from: $APK_DIR ==="
shopt -s nullglob
inputs=("$APK_DIR"/*.apk "$APK_DIR"/*.apkm)
shopt -u nullglob

if [[ ${#inputs[@]} -eq 0 ]]; then
    echo "ERROR: No .apk or .apkm files in $APK_DIR" >&2
    echo "  See apks/README.md for the download list." >&2
    exit 1
fi

echo "Found ${#inputs[@]} file(s):"
for f in "${inputs[@]}"; do echo "  - $(basename "$f")"; done
echo ""

# ── Install each one ──────────────────────────────────────────────
installed_count=0
failed_count=0
for f in "${inputs[@]}"; do
    case "$f" in
        *.apk)
            if install_apk "$f"; then ((installed_count++)); else ((failed_count++)); fi
            ;;
        *.apkm)
            if install_apkm "$f"; then ((installed_count++)); else ((failed_count++)); fi
            ;;
        *)
            echo "--- Skipping $(basename "$f") (unknown extension) ---"
            ;;
    esac
    echo ""
done

# ── Verify required packages are now present ──────────────────────
echo "=== Verifying required packages ==="
all_good=true
for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if adb shell pm list packages 2>/dev/null | grep -q "package:${pkg}$"; then
        echo "  ✅ $pkg"
    else
        echo "  ❌ $pkg  (missing — download APK and rerun)"
        all_good=false
    fi
done

echo ""
echo "=== Summary ==="
echo "  Installed: $installed_count"
echo "  Failed:    $failed_count"
echo ""

if $all_good; then
    echo "✅ All 6 Plus-repo packages present. Ready to run full smoke test."
    exit 0
else
    echo "⚠  Some required packages are still missing."
    echo "   Check apks/README.md for the download list."
    exit 1
fi
