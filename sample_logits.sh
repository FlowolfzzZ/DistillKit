#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate distillkit

export HF_ENDPOINT="https://hf-mirror.com"

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m distillkit.sample_logits_vllm \
  --model /ssd4/lizijian/Models/Qwen3-8B \
  --dataset /ssd4/lizijian/Datasets/allenai/tulu-3-sft-mixture \
  --output ./output/Qwen3-8B_tulu_logits/ \
  --compression-config examples/logit_compression/legacy_logit_compression_config.yml \
  --apply-chat-template \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --samples 1000 \
  --max-workers 1
