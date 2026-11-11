#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for seed in 664 665 666; do
    python verify.py --out "results/fresh_views/seed_$seed"
done
python demo.py examples/field.jpg
python demo.py examples/field_second.jpg
