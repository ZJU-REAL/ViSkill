#!/bin/bash
set -euo pipefail

BASEDIR=$(pwd)
SCRIPTDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

python3 -m vagen.skills.cold_start \
    --config-path=${SCRIPTDIR} \
    --config-name='cold_start_sokoban' \
    output=${BASEDIR}/skill_library_cold_start/sokoban \
    seed=[0,10000,1] \
    size=[5,10,20,30,40,50] \
    render.trajectory.thumbnail_size=null \
    

