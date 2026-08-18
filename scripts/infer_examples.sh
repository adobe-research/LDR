#!/bin/bash
#
# Copyright © 2026 Adobe Inc. and its licensors. All rights reserved.
#
# This file constitutes Licensed Materials under the Adobe Research License.
# Use is limited to noncommercial research purposes.
# See the LICENSE file at the project root for the complete license terms and disclaimer.
#
# Roll out every folder in examples/ to logs/examples/<case>.mp4 (run from the repo root).
# A "_" in the folder name marks an appearance-shift stress test case.
set -eu
mkdir -p logs/examples
for d in examples/*/; do
  case=$(basename "$d")
  if [[ "$case" == *_* ]]; then
    # appearance-shift stress case
    ckpt=checkpoints/128x128/single_task/uniform.pt
    img=128
    extra="--out_size 256"
  else
    # single-task case
    ckpt=checkpoints/256x256/single_task/$case.pt
    img=256
    extra=""
  fi
  echo "[$case] ckpt=$ckpt img=$img"
  python infer.py --ckpt "$ckpt" --frames_dir "$d" --img_size "$img" $extra --out "logs/examples/$case.mp4"
done
echo "done -> logs/examples/"
