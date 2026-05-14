#!/bin/bash
set -euo pipefail

# Model settings
model_name="mistralai/Mixtral-8x7B-v0.1"
model="mistralai/Mixtral-8x7B-v0.1"
model_str=""  # used to specify which model is used for stats computation;  empty string for using fp model

# Dataset settings
dataset="c4"
nsamples=128
seqlen=2048
seed=0


# =============================================================================
#  Step1: Compute statistics - Layer output gradients
# =============================================================================
# NOTE: For Mixtral-8x7B, 3 GPUs (each with 80GB memory) are needed to compute model gradients
layer_grads_path="cache/${model_name}/LayerGrads_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}${model_str}.pt"
CUDA_VISIBLE_DEVICES=0,1,2 python -m gemq.compute_model_stats \
    --mode "layer_grads" \
    --model ${model} \
    --model_name ${model_name} \
    --calib_dataset ${dataset} \
    --seed ${seed} \
    --nsamples ${nsamples} \
    --seqlen ${seqlen} \
    --layer_grads_path ${layer_grads_path}


# =============================================================================
#  Step2: Compute statistics - Weighted layer reconstruction errors
# =============================================================================
wbits="1,2,3"
layer_re_path="cache/${model_name}/LayerRE_${dataset}-N${nsamples}-L${seqlen}-Seed${seed}_B${wbits}${model_str}_faster.pkl"
python -m gemq.compute_model_stats \
    --mode "layer_re" \
    --model ${model} \
    --model_name ${model_name} \
    --calib_dataset ${dataset} \
    --seed ${seed} \
    --nsamples ${nsamples} \
    --seqlen ${seqlen} \
    --wbits ${wbits} \
    --layer_grads_path ${layer_grads_path} \
    --layer_re_path ${layer_re_path} \
    --forward_batch_size 1
