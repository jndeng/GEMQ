#!/bin/bash
set -euo pipefail

model_name="mistralai/Mixtral-8x7B-v0.1"
model_path="results/real_quant_models/${model_name}/GEMQ/C4-Seed0-WT2_A4-G16-D4-E2.0"

prompt="Although the experiment failed repeatedly, the researchers eventually"
max_new_tokens=200

CUDA_VISIBLE_DEVICES=0 TORCH_LOGS="graph_breaks,recompiles" python -m gemq.benchmark_generate \
    --model_path $model_path \
    --model_name $model_name \
    --attn_impl eager \
    --prompt "$prompt" \
    --num_samples 10 \
    --max_new_tokens $max_new_tokens \
    --top_k 200 \
    --compile