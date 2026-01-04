#!/bin/bash

set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_BENCHMARK_DIR="$( dirname "$( dirname "$SCRIPT_DIR" )" )"
PRIMUS_PATH="$TRAINING_BENCHMARK_DIR/third_party/Primus"

CONFIGS=(
#    MODEL_NAME         NNODES MBS GBS TP PP VPP RECOMPUTE_LAYERS TRAIN_ITERS
    "llama3.1_70B        4     2   256 1  1  1   0                10"
    "llama3.1_70B        4     2  1024 1  1  1   0                10"
    "llama3.1_70B        8     4   256 1  1  1  30                10"
    "llama3.1_70B        8     4  1024 1  1  1  30                10"
    "llama3.1_70B-Zero1  4     1    16 4  8  5   0                10"
    "llama3.1_70B-Zero1  8     1    16 8  8  5   0                10"
)

cd "$PRIMUS_PATH" || exit
for CONFIG in "${CONFIGS[@]}"; do
    read -r model_name node_num mbs gbs tp pp vpp recompute_layers iters <<< "$CONFIG"
    export MODEL_NAME="${model_name}"           # Set model name
    export NNODES=${node_num}                   # Number of nodes
    export MBS=${mbs}                           # Micro batch size
    export GBS=${gbs}                           # Global batch size
    export TP=${tp:-1}                          # Tensor parallelism
    export PP=${pp:-1}                          # Pipeline parallelism
    export VPP=${vpp:-1}                        # Virtual pipeline parallelism
    export RECOMPUTE_LAYERS=${recompute_layers} # Recomputation layers
    export TRAIN_ITERS=${iters}                 # Training iterations
    bash "${TRAINING_BENCHMARK_DIR}"/scripts/MI355/run_pretrain_mi355x.sh
done
