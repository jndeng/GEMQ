import os
import argparse
import time
import math
import gc
import json
from functools import partial
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from hqq.models.hf.base import AutoHQQHFModel

from gemq.quantizers.gptq import MCMoeGPTQWeightQuantizer, GPTQWeightQuantizer
from gemq.utils.data_utils import get_calib_loader
from gemq.utils.model_utils import *
from gemq.utils.quant_utils import *
from gemq.utils.eval_utils import evaluate_perplexity, run_lm_eval

logging.set_verbosity_error()


def save_quantized_model(model, tokenizer, save_path, save_dtype, real_quant):
    """
    Save the real/pseudo quantized model.
    """
    if real_quant:
        tokenizer.save_pretrained(save_path)
        AutoHQQHFModel.save_quantized(model, save_path)
    else:
        dtype = torch.float16 if save_dtype == "float16" else torch.bfloat16
        model = model.to(dtype)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)


def finetune_routers(model, dataloader, args):
    """
    Fine-tune all router modules in the MoE model.
    """
    # disable kv cahce
    use_cache = model.config.use_cache
    model.config.use_cache = False
    org_dtype = next(model.parameters()).dtype

    model.train()
    model = model.to(torch.bfloat16)

    # prepare dataset
    input_ids = []
    for data in dataloader:
        input_ids.append(data[0])  # (1, seqlen)
    input_ids = torch.cat(input_ids, dim=0)  # (nsamples, seqlen)

    # enable gradients for all routers
    router_params = get_router_params(model, args.model_name)  # NOTE: return a list of parameters
    for p in model.parameters():
        p.requires_grad = False
    for p in router_params:
        p.requires_grad = True

    # sanity check
    org_pmean, org_gmean = 0.0, 0.0
    for name, param in model.named_parameters():
        if get_module_type(name, args.model_name) == LinearModuleType.GATE:
            org_gmean += param.mean().item()
        else:
            org_pmean += param.mean().item()
    if args.verbose:
        print("Router stats before fine-tuning:", org_gmean)

    # start fine-tuning
    optimizer = torch.optim.AdamW(router_params, lr=args.rft_lr, weight_decay=args.rft_wd)
    for epoch in range(args.rft_epochs):
        loss_sum = 0.
        start = time.time()
        for i in range(args.nsamples // args.rft_batch_size):
            idx = i * args.rft_batch_size
            data = input_ids[idx: idx + args.rft_batch_size].to("cuda")  # (bsz, seqlen)
            outputs = model(input_ids=data, labels=data)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            if i % 32 == 0:
                print(f"[epoch {epoch} | iter {i:>3d}] loss: {loss_sum / (i+1):.6f}")
        elapse = time.time() - start
        print(f"epoch {epoch:>2} loss: {loss_sum / len(dataloader):.6f}, elapse: {elapse:.2f} seconds")

        # sanity check
        if epoch == 0:
            pmean, gmean = 0., 0.
            for name, param in model.named_parameters():
                if get_module_type(name, args.model_name) == LinearModuleType.GATE:
                    gmean += param.mean().item()
                else:
                    pmean += param.mean().item()

            assert math.fabs(org_pmean - pmean) < 1e-8, "Other parameters are changing during router fine-tuning!"
            assert math.fabs(org_gmean - gmean) > 1e-10, "Routers are not changing during fine-tuning!"
            print("Sanity check passed!")
            if args.verbose:
                print("Sum of routers params after finetuning:", gmean)

    # restore
    model = model.to(org_dtype)
    model.config.use_cache = use_cache


@torch.no_grad()
def quantize_weights_gptq(model, dataloader, args):
    """
    Perform mixed-precision weight-only quantization with GPTQ quantizer.
    At the end of this function, the model weights are replaced with dequantized weights in fp16.

    NOTE: only supports single GPU quantization.
    """
    # disable KV cache for quantization
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # build a bit allocation config for each Linear module
    bit_cfg = build_alloc_cfg(model, args)

    # prepare decoder inputs and kwargs for model forward
    inps, layer_kwargs = compute_decoder_inputs(model, dataloader, args.model_name, "cuda")

    # retrieve decoder blocks
    layers = get_blocks(model, args.model_name)

    # perform quantization for each block
    quant_modules = {}
    outs = torch.zeros_like(inps)
    for i in tqdm(range(len(layers)), desc="GPTQ Quantizing"):
        if args.verbose:
            print("+" + "="*57 + "+")
            print(f"| block {i:<24} | {'bit':<3} |  gs  | {'time (s)':>9} |")
            print("+" + "-"*57 + "+")
        start = time.time()

        # retrieve linear modules in the current block
        layer = layers[i].to("cuda")
        named_linears = get_named_linears(layer)

        # create a quantizer for each linear module that requires quantization
        quantizers = {}
        for name, m in named_linears.items():
            wbits = bit_cfg[i][name]

            # skip
            if wbits >= 16:
                continue

            # NOTE: adjust groupsize to fit the hidden size
            hidden_size = m.weight.shape[1]
            if hidden_size % args.groupsize == 0:
                groupsize = args.groupsize
            else:
                assert hidden_size % 64 == 0, "Currently only supports groupsize=64 as fallback."
                groupsize = 64
                if args.verbose:
                    print(f"Forcing groupsize from {args.groupsize} to 64 for module: {name}")

            if args.reproduce_mcmoe:
                quantizers[name] = MCMoeGPTQWeightQuantizer(
                    m.weight.data, name, wbits, args.blocksize, args.percdamp,
                    groupsize, args.actorder, args.static_groups, args.mse
                )
            else:
                quantizers[name] = GPTQWeightQuantizer(
                    m.weight.data, name, wbits, args.blocksize, args.percdamp,
                    groupsize, args.actorder, args.static_groups, args.mse
                )
            
            # collect quantized modules for real quantization saving
            quant_modules[f"{i}.{name}"] = m

        # update Hessian using a batch of input data for each linear module
        def update_hessian_hook(m, x, y, quantizer):
            x = x[0].detach()
            quantizer.add_batch(x)

        handles = []
        for name in named_linears:
            if name not in quantizers:
                continue
            handles.append(
                named_linears[name].register_forward_hook(
                    partial(update_hessian_hook, quantizer=quantizers[name])
                )
            )
        for j in range(args.nsamples):
            outs[j] = layer(inps[j: j+1], **layer_kwargs)[0]
        for h in handles:
            h.remove()

        elapse = time.time() - start
        if args.verbose:
            print(f"| {'pre-quantization':<30}   {' ':<3} | {' ':>4} | {elapse:>9.2f} |")
            print("+" + "-"*57 + "+")

        # quantize each linear module
        for name, m in named_linears.items():
            if name not in quantizers:
                continue

            # quantize
            start = time.time()
            Q, scales, zeros = quantizers[name].quantize()

            # dequantize
            W = quantizers[name].dequantize(Q, scales, zeros)

            # replace weights and register quantization params
            m.weight.data = W.reshape_as(m.weight.data)
            m.register_buffer("quant_scales", scales)
            m.register_buffer("quant_zeros", zeros)
            m.register_buffer("quant_nbits", torch.tensor(quantizers[name].nbits))
            m.register_buffer("quant_groupsize", torch.tensor(quantizers[name].groupsize))

            elapse = time.time() - start
            if args.verbose:
                print(f"| {name:<30} | {quantizers[name].nbits:<3} | {quantizers[name].groupsize:>4} | {elapse:>9.2f} |")

        # compute layer outputs using quantized weights
        start = time.time()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j: j+1], **layer_kwargs)[0]

        elapse = time.time() - start
        if args.verbose:
            print("+" + "-"*57 + "+")
            print(f"| {'post-quantization':<30}   {' ':<3} | {' ':>4} | {elapse:>9.2f} |")
            print("+" + "="*57 + "+")

        # update inputs for the next layer
        inps, outs = outs, inps

        # save memory
        layers[i] = layer.to("cpu")
        del quantizers
        gc.collect()
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache  # restore

    return quant_modules


def parse_args():
    parser = argparse.ArgumentParser(description="GEMQ for MoE-LLMs Quantization.")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Whether to enable verbose logging"
    )
    
    # model args
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Name of the model; used to load model-specific modules",
    )
    parser.add_argument(
        "--model_dtype", type=str, default="float16", choices=["auto", "float16", "bfloat16"],
        help="Data type of the model weights; use `auto` to load the model in the default dtype",
    )
    parser.add_argument(
        "--use_fast", action="store_true",
        help="Whether to use the fast tokenizer implementation",
    )
    parser.add_argument(
        "--attn_impl", type=str, default="eager", choices=["eager", "sdpa"],
        help="Implementation of attention to use",
    )
    parser.add_argument(
        "--disable_cache", action="store_true",
        help="Disable KV cache",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Enable `trust_remote_code` when loading the model from HuggingFace Hub",
    )

    # dataset args
    parser.add_argument(
        "--calib_dataset", type=str, default="wikitext2",
        help="Which calibration dataset to use",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128,
        help="Number of calibration sequences"
    )
    parser.add_argument(
        "--seqlen", type=int, default=2048,
        help="Length of each sequence",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size of the data loader"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for sampling the calibration data"
    )

    # mixed-precision bit allocation
    parser.add_argument(
        "--mixed", action="store_true",
        help="Whether to mixed-precision quantization"
    )
    parser.add_argument(
        "--bit_cfg", type=str, default="",
        help="Path to the bit allocation config file; leave blank to use uniform allocation"
    )

    # quantization args
    parser.add_argument(
        "--eval_fp", action="store_true",
        help="Whether to skip quantization and evaluate the full-precision model"
    )
    parser.add_argument(
        "--quantizer", type=str, default="gptq",
        help="Which quantizer to use"
    )
    parser.add_argument(
        "--attn_wbits", type=int, default=4,
        help="#bits for quantization of attention modules"
    )
    parser.add_argument(
        "--gate_wbits", type=int, default=16,
        help="#bits for quantization of gate (router) modules"
    )
    parser.add_argument(
        "--dense_wbits", type=int, default=4,
        help="#bits for quantization of dense modules"
    )
    parser.add_argument(
        "--expert_wbits", type=int, default=4,
        help="#bits for quantization of expert modules"
    )
    parser.add_argument(
        "--groupsize", type=int, default=128,
        help="Groupsize to use for quantization"
    )
    
    # quantizer-specific args
    parser.add_argument(
        "--blocksize", type=int, default=128,
        help="Blocksize to use for quantization"
    )
    parser.add_argument(
        "--percdamp", type=float, default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening"
    )
    parser.add_argument(
        "--mse", action="store_true",
        help="Whether to seach for quantization parameters (range)"
    )
    parser.add_argument(
        "--actorder", action="store_true",
        help="Whether to apply the activation order GPTQ heuristic (never used)"
    )
    parser.add_argument(
        "--static_groups", action="store_true",
        help="Whether to use static groups; recommended when using `--actorder` for more efficient inference. (never used)"
    )
    parser.add_argument(
        "--reproduce_mcmoe", action="store_true",
        help="Whether to use the GPTQ implementation from MC-MoE"
    )

    # router fine-tuning args
    parser.add_argument(
        "--finetune_routers", action="store_true",
        help="Whether to finetune the router modules after quantization"
    )
    parser.add_argument(
        "--rft_epochs", type=int, default=1,
        help="Number of epochs for the router fine-tuning"
    )
    parser.add_argument(
        "--rft_batch_size", type=int, default=1,
        help="Batch size for the router fine-tuning"
    )
    parser.add_argument(
        "--rft_lr", type=float, default=0.0001,
        help="Learning rate for the router fine-tuning"
    )
    parser.add_argument(
        "--rft_wd", type=float, default=0.0001,
        help="Weight decay for the router fine-tuning"
    )

    # evaluation args
    parser.add_argument(
        "--eval_downstream", action="store_true",
        help="Whether to run evaluation on downstream tasks"
    )
    parser.add_argument(
        "--downstream_tasks", type=str, default="piqa,arc_easy,arc_challenge,hellaswag,winogrande,mathqa,mmlu",
        help="Tasks to evaluate on; ignored if `--eval_downstream` is False"
    )
    parser.add_argument(
        "--lm_eval_batchsize", type=int, default=32,
        help="Batch size for lm_eval downstream evaluation"
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=0,
        help="Few-shot examples to use for downstream evaluation"
    )

    # i/o args
    parser.add_argument(
        "--real_quant", action="store_true",
        help="Whether to conduct real quantization and save the int weights (using HQQ)"
    )
    parser.add_argument(
        "--save_path", type=str, default="",
        help="Save quantized checkpoint under this path"
    )
    parser.add_argument(
        "--save_dtype", type=str, default="float16", choices=["float16", "bfloat16"],
        help="Data type to save the quantized model"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    # parse args
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    # load pre-trained model
    print("Loading model ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=args.use_fast, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cpu", torch_dtype=args.model_dtype,
        attn_implementation=args.attn_impl, trust_remote_code=args.trust_remote_code,
    )
    model.seqlen = 2048
    model.eval()

    # load calibration dataset
    print("Loading calibration data ...")
    dataloader = get_calib_loader(tokenizer, args)

    # quantize model weights
    if not args.eval_fp:
        # quantize and get a name-module mapping of quantized modules
        print(f"Start quantizing model weights ...")
        quantizer = args.quantizer.lower().split("-")[0]
        if quantizer == "gptq":
            quant_modules = quantize_weights_gptq(model, dataloader, args)
        else:
            raise ValueError(f"Unsupported weight quantizer: {args.quantizer}")
    
    # finetune routers
    if args.finetune_routers:
        model = dispatch_model_to_all_devices(model)
        
        print("Evaluating quantized model before fine-tuning ...")
        evaluate_perplexity(model, tokenizer, ["wikitext2", "c4"], args.model_name, offload=False)

        print("Fine-tuning routers ...")
        finetune_routers(model, dataloader, args)

    # evaluate model
    print("Evaluating model ...")
    model.eval()
    if args.eval_downstream or args.finetune_routers:
        # move all model weights onto gpus and use model() for forwarding
        if not args.finetune_routers:
            model = dispatch_model_to_all_devices(model)

        evaluate_perplexity(model, tokenizer, ["wikitext2", "c4"], args.model_name, offload=False)
        if args.eval_downstream:
            if args.disable_cache:
                model.config.use_cache = False
            try:
                run_lm_eval(
                    model, tokenizer, tasks=args.downstream_tasks.split(","),
                    batch_size=args.lm_eval_batchsize, num_fewshot=args.num_fewshot
                )
            except:
                print("Downstream evaluation failed. Skipping ...")
    else:
        # memory-efficient evaluation with layer offloading
        evaluate_perplexity(model, tokenizer, ["wikitext2", "c4"], args.model_name, offload=True)

    # save model
    if args.save_path:
        print("Saving model ...")
        os.makedirs(args.save_path, exist_ok=True)

        if args.real_quant:
            # for real quant, replace nn.Linear to HQQLinear for weight packing and saving
            replace_linears(model, args.model_name, quant_modules, quant_weight=True)
            check_packing(model, quant_modules, args)
        else:
            # for fake quant, remove the extra quantization parameters for saving
            for name, m in quant_modules.items():
                m.quant_scales = None
                m.quant_zeros = None
                m.quant_nbits = None
                m.quant_groupsize = None

        # save the quantized model
        save_quantized_model(model, tokenizer, args.save_path, args.save_dtype, args.real_quant)
        print(f"Quantized model saved to:", args.save_path)
