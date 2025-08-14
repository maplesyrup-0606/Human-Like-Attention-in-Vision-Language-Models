#!/bin/bash

# ----------------------------------------------------------------------
# Script for the LLaVA CLI tool (robust path handling)
# ----------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(realpath "$SCRIPT_DIR/../..")"
fi

# Change to the LLaVA directory
cd "$REPO_ROOT/LLaVA"

# Run LLaVA CLI with robust, repo-rooted paths
python -m llava.serve.cli \
  --model-path liuhaotian/llava-v1.5-7b \
  --load-4bit \
  --image-file "$REPO_ROOT/data/images/MSCOCO_images/000000002149.jpg" \
  --scanpath "$REPO_ROOT/data/scanpaths/coco_scanpaths/000000002149_scanpath.npy"