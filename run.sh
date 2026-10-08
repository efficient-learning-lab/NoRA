#!/usr/bin/env bash
set -euo pipefail

data="breastw"
model_type="NoRA"

echo "Running $model_type on $data."
python main.py \
    --dataname "$data" \
    --model_type $model_type \