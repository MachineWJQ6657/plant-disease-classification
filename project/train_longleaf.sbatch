#!/bin/bash
#SBATCH --job-name=tomato-fresh-3seed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=tomato-%j.log
# Activate a requirements-train.txt environment before submitting from project/.
# Supply the partition/account settings for your own allocation.
set -euo pipefail
: "${DATA_DIR:?Set DATA_DIR to the portable pinned raw-data directory}"
OUT_DIR="${OUT_DIR:-rerun/fresh_views}"
CACHE_DIR="${CACHE_DIR:-.build/fresh-cache}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python experiment_fresh.py --data "$DATA_DIR" --manifest results/manifest.csv --out "$OUT_DIR" --cache "$CACHE_DIR" --device cuda --seeds 664 665 666 --workers 4 --threads 4
python summarize_fresh.py --out "$OUT_DIR"
