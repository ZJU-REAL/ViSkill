SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASEDIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

python -m vagen.evaluate.run_eval \
  --config "${SCRIPT_DIR}/config.yaml" \
  fileroot="${BASEDIR}" \
  run.backend=sglang \
  backends.sglang.model=qwen2_5_vl_3b
