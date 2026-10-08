#!/usr/bin/env bash
set -euo pipefail

BASEDIR=$(cd "$(dirname "$0")" && pwd)

cd "${BASEDIR}/verl"
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e .

cd "${BASEDIR}"
pip install -e .
pip install "trl==0.26.2"
