# GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs

[![arXiv](https://img.shields.io/badge/arXiv-2605.23078-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2605.23078)&nbsp;

GEMQ is a post-training quantization framework for Mixture-of-Experts (MoE) LLMs that enables extreme low-bit quantization (down to 1.5 bits per expert) with minimal accuracy degradation. It works by:
1. automatically assigning different bit-widths to experts based on their importance;
2. fine-tuning the routers so they can better work with quantized experts;
3. optionally using progressive quantization to refine the bit allocation.


### What's in this repo
* An ILP solver for global expert-level bit allocation
* GPTQ-based quantization and router fine-tuning pipelines
* Efficient low-bit MoE triton kernels for **real** quantized inference


## Installation

```bash
conda create -n gemq python=3.10 -y
conda activate gemq
git clone https://github.com/jndeng/GEMQ
cd GEMQ
pip install -e .
```

> [!NOTE]
>
> This project currently uses **gurobipy** as the integer linear programming (ILP) solver for bit allocation. A Gurobi license may be required for certain MoE models with a large number of experts, such as the DeepSeek and Qwen series.



## Usage

> All scripts for Mixtral-8×7B and DeepSeek-V2-Lite are provided in `scripts`.

### 1. Bit Allocation

> [!NOTE]
>
> We provide pre-generated bit allocation configs under `configs`, which can be used directly for quantization. You may skip this section if you do not want to regenerate them.

To generate the configs from scratch, follow the steps below.


1. Download the first shard of the C4 training dataset (c4-train.00000-of-01024.json) from [allenai/c4](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz) and save it under `./data`.

2. Run `scripts/compute_stats_<model>.sh` to compute model statistics on the calibration dataset. The resulting statistics (gradients and perturbation errors) will be saved under `cache`.


3. Run `scripts/allocate_<model>.sh` to solve the ILP for bit allocation using the generated model statistics. The allocation results (bit configs) will be saved under `configs`. 


### 2. Mixed-Precision Quantization

Simply run `scripts/quantize_<model>.sh` for model quantization. Please refer to the script for the detailed available options.

The evaluation code runs automatically after quantization. If you want to evaluate the model on downstream tasks, please ensure that [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is installed.

Quantized models will be saved under `results`.


### 3. Inference

Use `scripts/bench_generate_<model>.sh` to run inference demos and benchmark the real quantized models.


## Acknowledgements
This repository builds upon several excellent open-source projects, including [MC-MoE](https://github.com/Aaronhuang-778/Mixture-Compressor-MoE), [GPTQ](https://github.com/IST-DASLab/gptq), [HQQ](https://github.com/dropbox/hqq), [GemLite](https://github.com/dropbox/gemlite), and [gpt-fast](https://github.com/meta-pytorch/gpt-fast). We sincerely thank the authors and contributors for making their code publicly available.

## Citation
If you find GEMQ useful for your research or project, please consider citing our paper.
