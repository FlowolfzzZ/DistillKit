#!/bin/bash
eval "$(/data/home/wtma/anaconda3/bin/conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=6,7
export WANDB_DIR=/ssd4/hyzhang/Kit/wandb_logs
conda activate mergekit_py312
HF_DATASETS_DISABLE_CACHE=1 distillkit examples/online_distillation/Qwen3_32B_only_ass.yml -v
