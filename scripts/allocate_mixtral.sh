#!/bin/bash
set -euo pipefail

# Settings
model_name="mistralai/Mixtral-8x7B-v0.1"
bits_per_expert=2.0  # target average bits-per-expert
wbits="1,2,3"        # candidate bit-widths
ilp_solver="gemq"    # ILP solver
extra_constr="c2c3"  # extra constraints for bit allocation
# path to the weighted layer reconstruction errors (i.e., ILP coefficients)
layer_re_path="cache/${model_name}/LayerRE_c4-N128-L2048-Seed0_B1,2,3_faster.pkl"

python -m gemq.allocate_bits \
    --model_name ${model_name} \
    --layer_re_path ${layer_re_path} \
    --bit_budget ${bits_per_expert} \
    --bit_candidates ${wbits} \
    --ilp_solver ${ilp_solver} \
    --extra_constr ${extra_constr}
