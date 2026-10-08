#!/bin/bash
# Start the PrimitiveSkill environment server in a tmux session.

DEVICES="${DEVICES:-[0,1]}"
MAX_ENVS_PER_GPU="${MAX_ENVS_PER_GPU:-128}"
PORT="${PORT:-8000}"

tmux new-session -s prim_env_server "export VK_ICD_FILENAMES=\$HOME/.local/share/vulkan/icd.d/nvidia_egl_icd.json && \
    python -m vagen.envs.primitive_skill.serve \
    --devices='${DEVICES}' \
    --max_envs_per_gpu=${MAX_ENVS_PER_GPU} \
    --port=${PORT}"
