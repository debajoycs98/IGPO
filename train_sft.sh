#!/usr/bin/env bash
set -euo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.fsdp_sft_trainer"}

NUM_GPUS=${NUM_GPUS:-8}

MODEL_ID=${MODEL_ID:-Qwen/Qwen3-8B}
MODEL_PATH=${MODEL_PATH:-/root/Qwen3-8B}
# huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"

TRAIN_FILES=${TRAIN_FILES:-/ossfs/workspace/data_sft_message.parquet}
VAL_FILES=${VAL_FILES:-/ossfs/workspace/data_sft_message.parquet}

SP_SIZE=${SP_SIZE:-1}
LIGER=${LIGER:-False}
MULTITURN=${MULTITURN:-True}
LORA_RANK=${LORA_RANK:-0}
RM_PAD=${RM_PAD:-True}

micro_bsz=1
NUM_GPUS=8

project_name="verl-test"
exp_name="$(basename "${MODEL_ID,,}")-sft-minimal"
ckpts_home=${ckpts_home:-/root/${project_name}/${exp_name}}
LOG_FILE="${ckpts_home}/verl-sft-train-$(date +'%Y%m%d-%H%M%S').log"

mkdir -p "${ckpts_home}"

exec > >(tee "$LOG_FILE") 2>&1
set -x

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=64 \
    data.max_length=30720 \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    data.response_dict_keys=['answer'] \
    data.multiturn.enable="${MULTITURN}" \
    data.multiturn.messages_key=messages \
    data.truncation=right \
    optim.lr=7e-6 \
    data.micro_batch_size_per_gpu=${micro_bsz} \
    model.partial_pretrain="${MODEL_PATH}" \
    model.lora_rank="${LORA_RANK}" \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.use_liger="${LIGER}" \
    model.enable_gradient_checkpointing=true \
    ulysses_sequence_parallel_size="${SP_SIZE}" \
    use_remove_padding="${RM_PAD}" \
    trainer.default_local_dir="${ckpts_home}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=10 \
    trainer.total_training_steps=200 \
    trainer.logger=['console','tensorboard'] \
    +trainer.save_freq=50 \
    +trainer.test_freq=50 \
    trainer.default_hdfs_dir=null $@

# rm -rf "${ckpts_home:?}/*"