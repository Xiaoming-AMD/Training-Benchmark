#!/bin/bash

set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_BENCHMARK_DIR="$( dirname "$( dirname "$SCRIPT_DIR" )" )"
PRIMUS_PATH="$TRAINING_BENCHMARK_DIR/third_party/Primus"

export MODEL_NAME="qwen3_30B_A3B"
export MANUAL_GC=True
export NUMA_BINDING=True
export ENABLE_SYNC_FREE_MOE=True
export ENABLE_TURBO_DEEPEP=True

# NNODES, MBS, GBS, TP, PP, EP, RECOMPUTE_LAYERS, TRAIN_ITERS
CONFIGS=(
    "4 1 16 1 2 8 0 10"
    "4 8 256 1 1 8 4 10"
    "4 8 1024 1 1 8 4 10"
    "8 1 16 4 1 8 0 10"
    "8 4 256 1 1 8 0 10"
    "8 8 1024 1 1 8 2 10"
)

cd "$PRIMUS_PATH" || exit
for CONFIG in "${CONFIGS[@]}"; do
    read -r node_num mbs gbs tp pp ep recompute_layers train_iters <<< "$CONFIG"
    export NNODES=${node_num}
    export MBS=${mbs}
    export GBS=${gbs}
    export TP=${tp}
    export PP=${pp}
    export EP=${ep}
    export RECOMPUTE_LAYERS=${recompute_layers}
    export TRAIN_ITERS=${train_iters}
    bash "${TRAINING_BENCHMARK_DIR}"/scripts/MI355/run_pretrain_mi355x.sh
done
