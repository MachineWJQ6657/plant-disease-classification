#!/bin/bash
#SBATCH --job-name=tomato-3seed
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=tomato-%j.log
# Submit from project/ inside a requirements-train.txt environment; add your partition/account.
set -euo pipefail
: "${DATA_DIR:?Set DATA_DIR to the directory filled by: python data.py --data DATA_DIR}"
python train.py --data "$DATA_DIR" --out rerun --device cuda
python analysis.py --results rerun
