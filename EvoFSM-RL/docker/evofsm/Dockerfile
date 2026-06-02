# C2 image — our own AndroidWorld emulator image for the verl/SkyRL spike.
# FROM ziqiang's already-built base (full SDK + emulator 35.3.11.0 + android_world
# + skyrl_server). We only swap in OUR AVD so the env matches our B1-B4 numbers.
#
# Build (context = android-sdk/avd, which holds AWAvd2.avd + AWAvd2.ini):
#   docker build -f EvoFSM-RL/docker/evofsm/Dockerfile -t androidworld:evofsm android-sdk/avd
#
# Run (all selection is via env vars — no code patched):
#   AVD_NAME=AWAvd2  ENV_SNAPSHOT=apps_ready_dec2025  ADB_SERVER_PORT=<private>
FROM androidworld:2026plusswipe

# Our AVD + its registration ini (path rewritten to the in-container location).
COPY AWAvd2.ini /root/.android/avd/AWAvd2.ini
COPY AWAvd2.avd /root/.android/avd/AWAvd2.avd
RUN sed -i 's|^path=.*|path=/root/.android/avd/AWAvd2.avd|' /root/.android/avd/AWAvd2.ini && \
    grep -q '^path=/root/.android/avd/AWAvd2.avd$' /root/.android/avd/AWAvd2.ini
