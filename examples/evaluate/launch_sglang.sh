#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-"qwen2_5_vl_3b"}"
MODEL_PATH="${MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}"
PORT="${PORT:-30000}"
DP_SIZE="${DP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${MODEL_NAME}" \
  --dp-size "${DP_SIZE}" \
  --tp "${TP_SIZE}" \
  --trust-remote-code \
  --mem-fraction-static "${MEM_FRACTION}" \
  --log-level info
