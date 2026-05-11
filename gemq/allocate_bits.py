import os
import os.path as osp
import argparse
import json
import pickle
import re

from gemq.utils.model_utils import get_model_info
from gemq.allocation.ilp_solvers import GEMQSolver


def auto_parse_filename(layer_re_path):
    calib_str = ""
    if "math+c4" in layer_re_path:
        calib_str = "MATH+C4"
    elif "c4" in layer_re_path:
        calib_str = "C4"
    elif "math" in layer_re_path:
        calib_str = "MATH"
    else:
        raise ValueError(f"Cannot parse calibration dataset from layer_re_path: {layer_re_path}")

    # extract seed number from layer_re_path
    match = re.search(r'Seed(\d+)', layer_re_path)
    seed_num = match.group(1) if match else "00"
    calib_str += f"-Seed{seed_num}"

    model_str = ""
    if "Uni" in layer_re_path:
        model_str = "_QT"
    elif "QTFT" in layer_re_path:
        model_str = "_QTFT"
    
    return calib_str, model_str


def run_gemq_solver(args):
    # parse info
    m = get_model_info(args.model_name)
    bpe = args.bit_budget
    bit_cands = list(map(int, args.bit_candidates.split(",")))

    # auto compute the total bit budget for global ilp
    # we assume the shared experts will get the highest bit;
    # all shared experts are merged to one single FFN, we need to consider this in bpl calculation;
    # NOTE: this is not the actual bpl because the shared experts are merged to one single expert
    bpl = (
        bpe * (m.num_routed_experts_per_layer + m.num_shared_experts_per_layer) -
        (max(0, m.num_shared_experts_per_layer - 1)) * max(bit_cands)
    )
    total_bits = bpl * (m.num_layers - m.first_k_dense_layers)

    # build a solver and solve
    global_solver = GEMQSolver(
        layer_re_path=args.layer_re_path,
        x_space=bit_cands,
        extra_constr=args.extra_constr, # NOTE: this args is valid only when using x_space=(1,2,3)
        start_layer_idx=m.first_k_dense_layers,
    )
    opt_set = global_solver.solve_all(total_bits=total_bits)

    # auto generate the save path if not specified
    save_path = args.save_path
    if not save_path:
        bc_str = ",".join(map(str, bit_cands))
        calib_str, model_str = auto_parse_filename(args.layer_re_path)
        const_str = "" if args.extra_constr == "none" else f"_{args.extra_constr}"
        save_path = f"configs/{args.model_name}/GEMQ/{calib_str}_E{bpe:.1f}_B{bc_str}{const_str}{model_str}.pkl"
    
    # save results
    os.makedirs(osp.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(opt_set, f)
    print("Bit config file saved to:", save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Bit allocation for MoE models.")
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Which model to perform bit allocation on",
    )
    parser.add_argument(
        "--layer_re_path", type=str, default="",
        help="Path to the pre-computed weighted layer reconstruction errors",
    )
    parser.add_argument(
        "--bit_budget", type=float, required=True,
        help="Average bits per expert (bpe) budget",
    )
    parser.add_argument(
        "--bit_candidates", type=str, default="1,2,3",
        help="Available bit candidates for allocation, comma-separated string",
    )
    parser.add_argument(
        "--ilp_solver", type=str, required=True, choices=["gemq"],
        help="Solver to use for bit allocation",
    )
    parser.add_argument(
        "--extra_constr", type=str, default="none",
        help="Whether to enable extra constraints for GEMQ solver",
    )
    parser.add_argument(
        "--save_path", type=str, default="",
        help="Path to save the bit allocation results (leave empty to auto-generate)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # parse args
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    if args.ilp_solver == "gemq":
        run_gemq_solver(args)
    else:
        raise ValueError(f"Unknown solver: {args.ilp_solver}")
