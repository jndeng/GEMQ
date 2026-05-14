#!/bin/bash
set -euo pipefail

# NOTE: There are two implementations of DeepSeek-V2-Lite: the one in HuggingFace Transformers and the one in the official DeepSeek repo.
# The former yields significantly higher eval perplexity, while the latter might not be compatible with latest Transformers versions.
# We can use the `use_official_impl` flag to switch between the two implementations.

# ===============================
#  Model settings
# ===============================
model_name="deepseek-ai/DeepSeek-V2-Lite"
model="deepseek-ai/DeepSeek-V2-Lite"
use_official_impl=true     # use official modeling code (set false to use HF Transformers implementation)

# ===============================
#  Dataset settings
# ===============================
calib_dataset="wikitext2"
nsamples=128
seqlen=2048

# ===============================
#  Quantization settings
# ===============================
quantizer="gptq"
bpe=2.0                    # bits per expert
mixed_prec=true            # enable expert-level mixed-precision quantization (set false for uniform quantization)
bit_cfg="configs/${model_name}/GEMQ/C4-Seed0_E${bpe}_B1,2,3_c2c3.pkl"

# ===============================
#  Router fine-tuning
# ===============================
# NOTE: 2x 80G GPUs are required for DeepSeek-V2-Lite router fine-tuning
finetune_routers=true      # whether to finetune the routers after quantization
rft_epochs=1
rft_lr=1e-4

# ===============================
#  Evaluation settings
# ===============================
eval_downstream=false      # whether to run downstream eval after quantization
downstream_tasks="piqa,arc_easy,arc_challenge,hellaswag,winogrande,mathqa,mmlu"

# ===============================
#  I/O settings
# ===============================
real_quant=true            # whether to pack + save INT weights (set false for pseudo quantization)
save_model=true            # whether to save the quantized model



# ===============================
#  AUTO argument construction
# ===============================
model_args=(--model "$model" --model_name "$model_name")
if [[ "${use_official_impl}" == "true" ]]; then
    model_args+=(--trust_remote_code)
fi

data_args=(--calib_dataset "$calib_dataset" --nsamples "$nsamples" --seqlen "$seqlen")

bpe_int=$(printf "%.0f" "$bpe")
quant_args=(--quantizer "$quantizer" --expert_wbits "$bpe_int" --groupsize 128 --mse --reproduce_mcmoe)
if [[ "${mixed_prec}" == "true" ]]; then
    qtype="$(basename "$(dirname "$bit_cfg")")"
    quant_args+=(--mixed --bit_cfg "$bit_cfg")
else
    qtype="Uniform"
fi

rft_tag=""
if [[ "${finetune_routers}" == "true" ]]; then
    rft_tag="_RFT"
    quant_args+=(--finetune_routers --rft_epochs "$rft_epochs" --rft_lr "$rft_lr")
fi

eval_args=()
if [[ "${eval_downstream}" == "true" ]]; then
    eval_args=(--eval_downstream --downstream_tasks "$downstream_tasks")
    if [[ "${use_official_impl}" == "true" ]]; then
        eval_args+=(--disable_cache)
    fi
fi

fname="${bit_cfg##*/}"
alloc_prefix="${fname%%_*}"
prefix="${alloc_prefix}-WT2"
if [[ "${save_model}" == "true" ]]; then
    if [[ "${real_quant}" == "true" ]]; then
        save_path="results/real_quant_models/${model_name}/${qtype}/${prefix}_A4-G16-D4-E${bpe}${rft_tag}"
        io_args=(--real_quant --save_path "$save_path")
    else
        save_path="results/fake_quant_models/${model_name}/${qtype}/${prefix}_A4-G16-D4-E${bpe}${rft_tag}"
        io_args=(--save_path "$save_path")
    fi
else
    save_path="None"
    io_args=()
fi


# ===============================
#  Run
# ===============================
echo "=============================================="
echo ">>> Quantization Job Summary"
echo "----------------------------------------------"
echo " Model:            ${model_name}"
echo " Implementation:   $( [[ "${use_official_impl}" == "true" ]] && echo 'Official (trust_remote_code)' || echo 'Transformers')"
echo " Dataset:          ${calib_dataset} (nsamples=${nsamples}, seqlen=${seqlen})"
echo "----------------------------------------------"
echo " Quantizer:        ${quantizer}"
echo " Expert bits:      ${bpe} (mixed: ${mixed_prec})"
echo " Bit config:       ${bit_cfg}"
echo " Finetune routers: ${finetune_routers} (epochs=${rft_epochs}, lr=${rft_lr})"
echo " Save path:        ${save_path}"
echo "----------------------------------------------"
echo ">>> Running quantization ..."
echo "=============================================="

python -m gemq.quantize \
    "${model_args[@]}" \
    "${data_args[@]}" \
    "${quant_args[@]}" \
    "${eval_args[@]}" \
    "${io_args[@]}"
