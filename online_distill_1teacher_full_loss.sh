#!/bin/bash
eval "$(/data/home/wtma/anaconda3/bin/conda shell.bash hook)"
export CUDA_VISIBLE_DEVICES=4,5
export WANDB_DIR=/ssd4/hyzhang/Kit/wandb_logs
conda activate mergekit_py312
distillkit examples/online_distillation/Qwen3_32B_full_loss.yml -v
