#!/bin/bash

set -x

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TRAINING_BENCHMARK_DIR="$( dirname "$( dirname "$SCRIPT_DIR" )" )"
PRIMUS_PATH="$TRAINING_BENCHMARK_DIR/third_party/Primus"

export MODEL_NAME="qwen2.5_32B"
export MANUAL_GC=False
export NUMA_BINDING=False

CONFIGS=(
#   NNODES  GPUS_PER_NODE  MBS   GBS    TP  PP  RECOMPUTE_LAYERS  TRAIN_ITERS  LOG_AVG_SKIP_ITERATIONS
    "1       1              2     32    1   1   0                 5           2"
    "1       8              8    256    1   1   0                 5           2"
)

cd "$PRIMUS_PATH" || exit
for CONFIG in "${CONFIGS[@]}"; do
    read -r node_num gpus_per_node mbs gbs tp pp recompute_layers train_iters log_avg_skip_iterations <<< "$CONFIG"
    export NNODES=${node_num}                   # Number of nodes
    export GPUS_PER_NODE=${gpus_per_node}       # GPUs per node
    export MBS=${mbs}                           # Micro batch size
    export GBS=${gbs}                           # Global batch size
    export TP=${tp}                             # Tensor parallel
    export PP=${pp}                             # Pipeline parallel
    export RECOMPUTE_LAYERS=${recompute_layers} # Recomputation layers
    export TRAIN_ITERS=${train_iters}           # Training iterations
    export LOG_AVG_SKIP_ITERATIONS=${log_avg_skip_iterations} # Skip the first several iterations when calculating throughput average
    bash "${TRAINING_BENCHMARK_DIR}"/scripts/MI355/run_pretrain_mi355x.sh
done
