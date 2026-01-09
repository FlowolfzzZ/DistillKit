#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate distillkit

export HF_ENDPOINT="https://hf-mirror.com"

export CUDA_VISIBLE_DEVICES=0,1,2,3

teacher_models=("Qwen3-8B" "DeepSeek-R1-Distill-Qwen-32B")

for teacher_model in ${teacher_models[@]}
do
  echo "Sampling logits for model: ${teacher_model}"
  python -m distillkit.sample_logits_vllm \
    --model /ssd4/lizijian/Models/${teacher_model} \
    --dataset /ssd4/lizijian/Datasets/allenai/tulu-3-sft-mixture \
    --output ./output/${teacher_model}_tulu_logits/ \
    --compression-config examples/logit_compression/legacy_logit_compression_config.yml \
    --apply-chat-template \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --samples 10 \
    --max-workers 1
done

teacher_order=$(printf "'%s'," "${teacher_models[@]}")
teacher_order="${teacher_order%,}"

output_file=$(printf "%s+" "${teacher_models[@]}")
output_file="${output_file%+}/data_0.parquet"

python merge_multi_teacher_logits.py --teacher_order "[$teacher_order]" --output_file "$output_file"