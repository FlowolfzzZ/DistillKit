#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda activate distillkit

export HF_ENDPOINT="https://hf-mirror.com"

export CUDA_VISIBLE_DEVICES=0,1,2,3

python distillkit/main.py examples/online_distillation/DeepSeek-R1-Distill-Qwen-32B.yml
