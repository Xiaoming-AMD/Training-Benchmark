#!/bin/bash

set -x


# NNODES, MBS, GBS, TRAIN_ITERS
CONFIGS=(
    "1 2 16   50"
    "1 8 256  10"
    "1 8 1024 10"
    "2 1 16   50"
    "2 8 256  10"
    "2 8 1024 10"
)

cd ../../third_party/Primus
for CONFIG in "${CONFIGS[@]}"; do
    read node_num mbs gbs iters <<< "$CONFIG"
    export NNODES=${node_num}
    export MBS=${mbs}
    export GBS=${gbs}
    export TRAIN_ITERS=${iters}
    bash examples/customer_package/run_qwen3_8b_pretrain_mi355x.sh
done
