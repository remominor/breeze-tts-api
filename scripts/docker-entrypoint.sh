#!/usr/bin/env bash
set -euo pipefail

# Runtime paths are intentionally configurable so Unraid can map host shares
# without rebuilding the image.  BREEZE_WEIGHTS_PATH takes precedence over the
# filename form for users who keep weights outside BREEZE_MODEL_ROOT.
model_root="${BREEZE_MODEL_ROOT:-/models}"
model_dir="${BREEZE_MODEL_DIR:-${model_root}/Breeze-TTS-2}"
weights_path="${BREEZE_WEIGHTS_PATH:-${model_root}/${BREEZE_WEIGHTS_FILE:-Breeze-TTS-2-int8-hybrid.safetensors}}"
voice_dir="${BREEZE_PROFILE_DIR:-${BREEZE_VOICE_DIR:-/data/profiles}}"

exec python -m breeze_infer.api "${model_dir}" \
  --weights "${weights_path}" \
  --voice-dir "${voice_dir}" \
  --host "${BREEZE_HOST:-0.0.0.0}" \
  --port "${BREEZE_PORT:-7860}" \
  --fast-backbone-decode \
  --fast-depth-decoder \
  --fast-codec \
  "$@"
