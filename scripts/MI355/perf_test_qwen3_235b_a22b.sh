#!/bin/bash

set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_BENCHMARK_DIR="$( dirname "$( dirname "$SCRIPT_DIR" )" )"
PRIMUS_PATH="$TRAINING_BENCHMARK_DIR/third_party/Primus"

export MODEL_NAME="qwen3_235B_A22B"
export MANUAL_GC=True
export NUMA_BINDING=True
export ENABLE_SYNC_FREE_MOE=True
export ENABLE_TURBO_DEEPEP=True

CONFIGS=(
#   NNODES  MBS   GBS    PP  EP  PIPELINE_LAYOUT             TRAIN_ITERS
    "4      1     16     4   8  'Et*23|t*24|t*24|t*23,L'     10"
    "4      1    256     4   8  'Et*7|(t*8|)*10,t*7,L'       10"
    "4      1   1024     4   8  'Et*7|(t*8|)*10,t*7,L'       10"
    "8      1     16     4   8  'Et*23|t*24|t*24|t*23,L'     10"
    "8      1    256     4   8  'Et*7|(t*8|)*10,t*7,L'       10"
    "8      1   1024     4   8  'Et*7|(t*8|)*10,t*7,L'       10"
)

cd "$PRIMUS_PATH" || exit
for CONFIG in "${CONFIGS[@]}"; do
    read -r node_num mbs gbs pp ep pipeline_layout train_iters <<< "$CONFIG"
    export NNODES=${node_num}                   # Number of nodes
    export MBS=${mbs}                           # Micro batch size
    export GBS=${gbs}                           # Global batch size
    export PP=${pp}                             # Pipeline parallel
    export EP=${ep}                             # Expert parallel
    export PIPELINE_LAYOUT=${pipeline_layout}   # Pipeline layout string
    export TRAIN_ITERS=${train_iters}           # Training iterations
    bash "${TRAINING_BENCHMARK_DIR}"/scripts/MI355/run_pretrain_mi355x.sh
done
