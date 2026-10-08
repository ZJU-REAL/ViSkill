#!/bin/bash
set -euo pipefail

BASEDIR=$(pwd)
SCRIPTDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

python3 -m vagen.skills.cold_start \
    --config-path=${SCRIPTDIR} \
    --config-name='cold_start_frozenlake' \
    output=${BASEDIR}/skill_library_cold_start/frozenlake \
    seed=[0,10000,1] \
    size=[20,40,80,120,160,200] \
    render.trajectory.thumbnail_size=120