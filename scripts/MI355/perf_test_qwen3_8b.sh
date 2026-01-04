#!/bin/bash

set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_BENCHMARK_DIR="$( dirname "$( dirname "$SCRIPT_DIR" )" )"
PRIMUS_PATH="$TRAINING_BENCHMARK_DIR/third_party/Primus"

export MODEL_NAME="qwen3_8B"

CONFIGS=(
# NNODES  MBS   GBS    TRAIN_ITERS
    "1      2     16     50"
    "1      8    256     10"
    "1      8   1024     10"
    "2      1     16     50"
    "2      8    256     10"
    "2      8   1024     10"
)

cd "$PRIMUS_PATH" || exit
for CONFIG in "${CONFIGS[@]}"; do
    read -r node_num mbs gbs iters <<< "$CONFIG"
    export NNODES=${node_num}         # Number of nodes
    export MBS=${mbs}                 # Micro batch size
    export GBS=${gbs}                 # Global batch size
    export TRAIN_ITERS=${iters}       # Training iterations
    bash "${TRAINING_BENCHMARK_DIR}"/scripts/MI355/run_pretrain_mi355x.sh
done
