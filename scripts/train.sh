#!/bin/bash
#
# Copyright © 2026 Adobe Inc. and its licensors. All rights reserved.
#
# This file constitutes Licensed Materials under the Adobe Research License.
# Use is limited to noncommercial research purposes.
# See the LICENSE file at the project root for the complete license terms and disclaimer.
#
# Train one LDR model from scratch across your nodes (8 GPUs each) using torchrun.
# Training steps: 10000 for uniform / parabola / looming, 20000 for collision / bouncing / joint_5task.
set -eu
res=$1; data=$2; steps=$3; out=logs/${data}_${res}
mkdir -p "$out/ckpts"
if [ -z "${LDR_RANK:-}" ]; then
  # launcher: one worker per node
  read -ra nodes <<< "$NODES"
  nn=${#nodes[@]}
  for r in "${!nodes[@]}"; do
    ssh "${nodes[$r]}" "cd $PWD && LDR_RANK=$r NN=$nn MASTER=$MASTER PORT=${PORT} bash scripts/train.sh $res $data $steps" </dev/null >> "$out/node$r.log" 2>&1 &
  done; wait; echo "done -> $out (per-node logs: $out/node*.log)"
else
  # worker: 8 GPUs on this node
  source /opt/conda/etc/profile.d/conda.sh
  conda activate ldr
  # global batch 256
  bs=$([ "$res" = 256 ] && echo 2 || echo 4)
  acc=$(( 256 / (NN*8*bs) ))
  resume=$(ls -1v "$out"/ckpts/ckpt_0*.pt 2>/dev/null | tail -1)
  torchrun --nnodes="$NN" --node_rank="$LDR_RANK" --nproc_per_node=8 --master_addr="$MASTER" --master_port="${PORT}" train.py --img_size "$res" --total_steps "$steps" --task "$data" --batch_size "$bs" --grad_accum "$acc" --data "data/train/$data.hdf5" --eval_data "data/eval/$data.hdf5" --split "data/splits/$data.json" --output_dir "$out/ckpts" ${resume:+--resume "$resume"}
fi
