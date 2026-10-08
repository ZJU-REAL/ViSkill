#!/usr/bin/env bash
set -euo pipefail

pip install fire httpx fastapi torch numba

# Dependencies for the original VAGEN Navigation environment.
pip install ai2thor
python -m vagen.envs.navigation.pre_download_scenes

# Dependencies for PrimitiveSkill.
pip install "mani_skill<=3.0.0b22"
python -m mani_skill.utils.download_asset partnet_mobility_cabinet
python -m mani_skill.utils.download_asset ycb
