#!/bin/bash
# ============================================
# IGPO Evaluation Script
# 
# This script runs evaluation only, without training.
# It uses a dedicated evaluation entry point with minimal configuration.
# ============================================

set -e

# ============================================
# Environment Variables
# ============================================
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export RAY_memory_monitor_refresh_ms=0

# ============================================
# User Configuration (modify these)
# ============================================
MODEL_PATH=${MODEL_PATH:-"/root/Qwen2.5-7B-Instruct"}
VAL_FILES=${VAL_FILES:-"./data/test.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"./eval_results"}
MAX_TURNS=${MAX_TURNS:-10}
SEARCH_ENGINE=${SEARCH_ENGINE:-"rag"}  # "rag" or "online_search"
N_GPUS=${N_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-32}

# ============================================
# Create output directory
# ============================================
mkdir -p ${OUTPUT_DIR}

echo "============================================"
echo "IGPO Evaluation"
echo "============================================"
echo "Model: ${MODEL_PATH}"
echo "Validation data: ${VAL_FILES}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Max turns: ${MAX_TURNS}"
echo "Search engine: ${SEARCH_ENGINE}"
echo "GPUs: ${N_GPUS}"
echo "============================================"

# ============================================
# Run Evaluation
# ============================================
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_eval \
    model.path=${MODEL_PATH} \
    data.val_files=${VAL_FILES} \
    data.batch_size=${BATCH_SIZE} \
    eval.output_dir=${OUTPUT_DIR} \
    eval.max_turns=${MAX_TURNS} \
    eval.search_engine=${SEARCH_ENGINE} \
    trainer.n_gpus_per_node=${N_GPUS} \
    2>&1 | tee ${OUTPUT_DIR}/eval.log

echo "============================================"
echo "Evaluation completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================"
