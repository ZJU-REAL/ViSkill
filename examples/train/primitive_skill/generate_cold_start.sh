#!/bin/bash
set -euo pipefail

BASEDIR=$(pwd)
SCRIPTDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

python3 -m vagen.skills.cold_start \
    --config-path=${SCRIPTDIR} \
    --config-name='cold_start_maniskill' \
    output=${BASEDIR}/skill_library_cold_start/maniskill \
    seed=[0,5000,1] \
    size=[50,100,200,300,400,500] \
    render.trajectory.thumbnail_size=180
