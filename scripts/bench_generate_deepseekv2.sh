#!/bin/bash
set -euo pipefail

model_name="deepseek-ai/DeepSeek-V2-Lite"
mode="QT"  # FP | QT

prompt="Although the experiment failed repeatedly, the researchers eventually"
max_new_tokens=200

if [ "$mode" == "FP" ]; then
    model_path=$model_name
    CUDA_VISIBLE_DEVICES=0 TORCH_LOGS="graph_breaks,recompiles" python -m gemq.benchmark_generate \
        --model_path $model_path \
        --model_name $model_name \
        --attn_impl eager \
        --prompt "$prompt" \
        --num_samples 10 \
        --max_new_tokens $max_new_tokens \
        --top_k 200 \
        --temperature 0.8 \
        --compile \
        --is_fp
else
    model_path="results/real_quant_models/${model_name}/GEMQ/C4-Seed0-WT2_A4-G16-D4-E2.0"

    # NOTE: if `trust_remote_code` is used, make sure `modeling_deepseek.py` is present in the local model folder.
    hf download deepseek-ai/DeepSeek-V2-Lite modeling_deepseek.py --local-dir $model_path

    CUDA_VISIBLE_DEVICES=0 TORCH_LOGS="graph_breaks,recompiles" python -m gemq.benchmark_generate \
        --model_path $model_path \
        --model_name $model_name \
        --trust_remote_code \
        --attn_impl eager \
        --prompt "$prompt" \
        --num_samples 10 \
        --max_new_tokens $max_new_tokens \
        --top_k 200 \
        --temperature 0.8 \
        --compile
fi