#!/bin/bash

OUTDIR=~/NSERC/conda_envs
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

mapfile -t ENVS < <(
    conda env list | awk 'NR>2 {print $1}' | sed 's/^\*//' | grep -vE '^(#|base$)'
)

for ENV in "${ENVS[@]}"; do
    [[ -z "$ENV" ]] && continue
    echo "Exporting env: $ENV"

    conda env export -n "$ENV" | sed '/^prefix:/d' > "${OUTDIR}/${ENV}.yml"
done 

echo "Done exporting."
