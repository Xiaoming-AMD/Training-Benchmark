#!/bin/bash

set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_BENCHMARK_DIR="$( dirname "$( dirname "$SCRIPT_DIR" )" )"
PRIMUS_PATH="$TRAINING_BENCHMARK_DIR/third_party/Primus"

export MODEL_NAME="llama3.1_70B"

# NNODES, MBS, GBS, RECOMPUTE_LAYERS, TRAIN_ITERS
CONFIGS=(
    "4 2 256  0 10"
    "4 2 1024 0 10"
    "8 4 256  30 10"
    "8 4 1024 30 10"
)

cd "$PRIMUS_PATH" || exit
for CONFIG in "${CONFIGS[@]}"; do
    read -r node_num mbs gbs recompute_layers iters <<< "$CONFIG"
    export NNODES=${node_num}
    export MBS=${mbs}
    export GBS=${gbs}
    export RECOMPUTE_LAYERS=${recompute_layers}
    export TRAIN_ITERS=${iters}
    bash "${TRAINING_BENCHMARK_DIR}"/scripts/MI355/run_pretrain_mi355x.sh
done

export MODEL_NAME="llama3.1_70B-Zero1"

# NNODES, MBS, GBS, TP, PP, VPP, RECOMPUTE_LAYERS, TRAIN_ITERS
CONFIGS=(
    "4 1 16 4 8 5 0 10"
    "8 1 16 8 8 5 0 10"
)

cd "$PRIMUS_PATH" || exit
for CONFIG in "${CONFIGS[@]}"; do
    read -r node_num mbs gbs tp pp vpp recompute_layers iters <<< "$CONFIG"
    export NNODES=${node_num}
    export MBS=${mbs}
    export GBS=${gbs}
    export TP=${tp}
    export PP=${pp}
    export VPP=${vpp}
    export RECOMPUTE_LAYERS=${recompute_layers}
    export TRAIN_ITERS=${iters}
    bash "${TRAINING_BENCHMARK_DIR}"/scripts/MI355/run_pretrain_mi355x.sh
done