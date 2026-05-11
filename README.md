# GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2505.05799-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2505.05799)&nbsp; -->

GEMQ is a mixed-precision quantization framework for Mixture-of-Experts (MoE) LLMs that enables extreme low-bit quantization (down to 1.5 bits per expert) with minimal accuracy degradation. GEMQ achieves this through:
* a global linear-programming formulation for expert-wise mixed-precision bit allocation based on quantization error analysis;
* efficient router fine-tuning to adapt routing policies to quantized experts;
* optional progressive quantization that iteratively refines expert importance estimation.


#### What's in this repo
* An ILP solver for expert-level bit allocation
* GPTQ-based quantization and router fine-tuning pipelines
* Efficient low-bit MoE Triton kernels for **real** quantized inference


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
> This project uses **gurobipy** as the integer linear programming (ILP) solver for bit allocation. A Gurobi license may be required for certain MoE models, especially those with a large number of experts.



## Usage

Demo scripts for Mixtral-8×7B and DeepSeek-V2-Lite are provided in `scripts`.

### Bit Allocation

We provide pre-generated bit allocation configs under `configs`, which can be used directly for quantization. You may skip this section if you do not want to regenerate them. To generate the configs from scratch, follow the steps below.


1. Download the first shard of the C4 training dataset (c4-train.00000-of-01024.json) from [allenai/c4](https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz) and save it under `./data`.

2. Run `scripts/compute_stats_<model>.sh` to compute model statistics on the calibration dataset. The resulting statistics (gradients and perturbation errors) will be saved under `cache`.


3. Run `scripts/allocate_<model>.sh` to solve the ILP for bit allocation using the generated model statistics. The allocation results (bit configs) will be saved under `configs`. 


### Mixed-Precision Quantization

Simply run `scripts/quantize_<model>.sh` for model quantization. Please refer to the scripts for detailed usage instructions.

The evaluation code will run automatically after quantization.

Quantized models will be saved under `results`.


### Inference

Use `scripts/bench_generate_<model>.sh` to run and benchmark the real quantized models.


## Acknowledgements
This repository builds upon several excellent open-source projects, including [MC-MoE](https://github.com/Aaronhuang-778/Mixture-Compressor-MoE), [GPTQ](https://github.com/IST-DASLab/gptq), and [HQQ](https://github.com/dropbox/hqq) / [GemLite](https://github.com/dropbox/gemlite). We sincerely thank the authors and contributors for making their code publicly available.

## Citation
If you find GEMQ useful for your research or project, please consider citing our paper.
