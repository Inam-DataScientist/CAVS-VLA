#!/usr/bin/env bash
# Reproduce every number and figure in the paper from scratch (CPU only, ~15 min).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest
python -m safeland shield                                   # certificate + reserve map
python -m safeland run --set experiment.name=main            # 3 methods x 10 seeds x 100k episodes
# robustness sweep over gust probability
for p in 0.1 0.2 0.3 0.4; do
  python -m safeland run --set world.wind_prob=$p --set experiment.name=wind_$p
done
echo "results in results/"
