#!/usr/bin/env bash
# Full experimental pipeline (steps 2-6 of the README). Usage:
#   bash scripts/run_pipeline.sh configs/full_nuplan_womd.yaml main 8
set -euo pipefail
CFG=${1:-configs/default.yaml}
RUN=${2:-main}
NGPU=${3:-1}
CKPT_DIR=$(python - "$CFG" <<'PY'
import sys
from cavs_vla.config import load_config
print(load_config(sys.argv[1]).train.ckpt_dir)
PY
)
CKPT="$CKPT_DIR/$RUN/best.pt"

python -m cavs_vla build-data --config "$CFG"
python -m cavs_vla kinematic-check --config "$CFG" | tee results_kinematic.json
python -m cavs_vla estimate-disturbance --config "$CFG"
python -m cavs_vla build-hj --config "$CFG"
if [ "$NGPU" -gt 1 ]; then
  torchrun --nproc_per_node="$NGPU" -m cavs_vla train --config "$CFG" --run "$RUN"
else
  python -m cavs_vla train --config "$CFG" --run "$RUN"
fi
python -m cavs_vla calibrate   --config "$CFG" --ckpt "$CKPT"
python -m cavs_vla verify      --config "$CFG" --ckpt "$CKPT"
python -m cavs_vla eval-open   --config "$CFG" --ckpt "$CKPT"
python -m cavs_vla eval-closed --config "$CFG" --ckpt "$CKPT"
python -m cavs_vla eval-closed --config "$CFG" --ckpt "$CKPT" --methods B1,B7 --tag mismatch --set sim.accel_noise=1.0 sim.tau_a=0.3
python -m cavs_vla eval-closed --config "$CFG" --ckpt "$CKPT" --methods B1,B7 --tag latency  --set sim.replan_interval=5 sim.latency_steps=2
python -m cavs_vla eval-closed --config "$CFG" --ckpt "$CKPT" --methods B1,B6,B7 --tag cav4 --set sim.num_controlled=4
python -m cavs_vla mine-failures --config "$CFG" --ckpt "$CKPT" --method B1 --out results/failures_train.json
python -m cavs_vla ceg --config "$CFG" --ckpt "$CKPT" --failures results/failures_train.json --run "${RUN}_ceg"
python -m cavs_vla eval-closed --config "$CFG" --ckpt "$CKPT_DIR/${RUN}_ceg/best.pt" --tag ceg
