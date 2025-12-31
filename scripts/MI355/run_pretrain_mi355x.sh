#!/bin/bash
# shellcheck disable=SC2155
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

###################### Training Docker and Variables ##########################
export DOCKER_IMAGE="docker.io/tasimage/primus:pr-464-v25.09-ainic"
export HF_TOKEN=${HF_TOKEN:-"your_hf_token"}
export CLEAN_DOCKER_CONTAINER=1
export SKIP_TRAIN=0

######################## Vultr Cluster Settings ###############################
export NNODES=${NNODES:-1}
export USING_AINIC=${USING_AINIC:-1}
export NCCL_SOCKET_IFNAME=enp193s0f1np1
export GLOO_SOCKET_IFNAME=enp193s0f1np1

########################### Training Config ###################################
export MODEL_NAME=${MODEL_NAME:-llama3.1_8B}
export MBS=${MBS:-8}
export GBS=${GBS:-256}
export TP=${TP:-1}
export PP=${PP:-1}
export EP=${EP:-1}
export SEQ_LENGTH=${SEQ_LENGTH:-4096}
export RECOMPUTE_LAYERS=${RECOMPUTE_LAYERS:-0}
export LEGACY_GG=${LEGACY_GG:-False}
export TRAIN_ITERS=${TRAIN_ITERS:-10}
export MANUAL_GC=${MANUAL_GC:-False}
export ENABLE_SYNC_FREE_MOE=${ENABLE_SYNC_FREE_MOE:-False}
export ENABLE_TURBO_DEEPEP=${ENABLE_TURBO_DEEPEP:-False}
export VPP=${VPP:-1}

# Optional pipeline layout: if PIPELINE_LAYOUT is set externally, pass it through;
# otherwise do not configure pipeline_model_parallel_layout at all.
PIPELINE_LAYOUT=${PIPELINE_LAYOUT:-""}

FEATURE_ARGS=()

PRIMUS_TURBO_ENABLED="False"
ensure_primus_turbo() {
    if [ "$PRIMUS_TURBO_ENABLED" = "False" ]; then
        FEATURE_ARGS+=("--enable_primus_turbo" "True")
        PRIMUS_TURBO_ENABLED="True"
    fi
}

if [ "$MANUAL_GC" = "True" ]; then
    FEATURE_ARGS+=("--manual_gc" "True")
    FEATURE_ARGS+=("--manual_gc_interval" "1")
fi

if [ "$ENABLE_SYNC_FREE_MOE" = "True" ]; then
    ensure_primus_turbo
    FEATURE_ARGS+=("--turbo_sync_free_moe_stage" "1")
fi

if [ "$ENABLE_TURBO_DEEPEP" = "True" ]; then
    ensure_primus_turbo
    FEATURE_ARGS+=("--use_turbo_deepep" "True")
    FEATURE_ARGS+=("--turbo_deepep_num_cu" "32")
    FEATURE_ARGS+=("--turbo_deepep_use_comm_stream" "False")
    FEATURE_ARGS+=("--moe_shared_expert_overlap" "False")
    FEATURE_ARGS+=("--moe_router_dtype" "fp32")
fi

if [ -n "$PIPELINE_LAYOUT" ]; then
    FEATURE_ARGS+=("--pipeline_model_parallel_layout" "$PIPELINE_LAYOUT")
elif [ "$VPP" -gt 1 ]; then
    FEATURE_ARGS+=("--num_virtual_stages_per_pipeline_rank" "$VPP")
fi

###################### Training Launch Config #################################
export HSA_NO_SCRATCH_RECLAIM=1
export NVTE_CK_USES_BWD_V3=1

export NUMA_BINDING=${NUMA_BINDING:-False}
if [ "$NUMA_BINDING" = "True" ]; then
    export ENABLE_NUMA_BINDING=1
    export HSA_KERNARG_POOL_SIZE=12582912
fi

####################### Training Experiments ##################################
export PRIMUS_TEAM="date-$(date +%Y%m%d)"
export PRIMUS_USER=user-tas
export PRIMUS_EXP_NAME="${MODEL_NAME}_MI355X_NNODES${NNODES}_MBS${MBS}_GBS${GBS}"

LOG_DIR=./output/$PRIMUS_TEAM/$PRIMUS_USER/$PRIMUS_EXP_NAME
export LOG_FILE=$LOG_DIR/training.log
export EXPORT_CONFIG=$LOG_DIR/config.yaml
mkdir -p "$LOG_DIR"

########################## Training Job #######################################
export EXP="examples/megatron/configs/MI355X/${MODEL_NAME}-BF16-pretrain.yaml"

bash ./examples/run_slurm_pretrain.sh \
    --micro_batch_size "$MBS" \
    --global_batch_size "$GBS" \
    --seq_length "$SEQ_LENGTH" \
    --tensor_model_parallel_size "$TP" \
    --pipeline_model_parallel_size "$PP" \
    --expert_model_parallel_size "$EP" \
    --moe_use_legacy_grouped_gemm "$LEGACY_GG" \
    --recompute_granularity "full" \
    --recompute_method "block" \
    --recompute_num_layers "${RECOMPUTE_LAYERS}" \
    --cross_entropy_fusion_impl "te" \
    --cross_entropy_loss_fusion "True" \
    "${FEATURE_ARGS[@]}" \
    --train_iters "$TRAIN_ITERS" 2>&1 | tee "$LOG_FILE"
