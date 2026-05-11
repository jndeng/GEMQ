import argparse
import time
import json
import contextlib
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from hqq.models.hf.base import AutoHQQHFModel

from gemq.inference.kv_cache import StaticCache
from gemq.inference.patch import prepare_for_inference


def device_sync(device="cuda"):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    else:
        print(f"device={device} is not yet suppported")


def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = F.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def prefill(model, x, kv_cache, input_pos, **sampling_kwargs):
    outputs = model(x, past_key_values=kv_cache, cache_position=input_pos)
    return sample(outputs.logits, **sampling_kwargs)[0]


def decode_one_token(model, x, kv_cache, input_pos, **sampling_kwargs):
    outputs = model(x, past_key_values=kv_cache, cache_position=input_pos)
    return sample(outputs.logits, **sampling_kwargs)[0]


def decode_n_tokens(model, cur_token, kv_cache, input_pos, num_new_tokens, **sampling_kwargs):
    new_tokens = []
    for i in range(num_new_tokens):
        next_token = decode_one_token(model, cur_token, kv_cache, input_pos, **sampling_kwargs)
        input_pos += 1
        new_tokens.append(next_token.clone())
        cur_token = next_token.clone().view(1, -1)
    return new_tokens


def load_model(args, compute_dtype=torch.float16, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=args.trust_remote_code
    )

    if args.is_fp:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, dtype=compute_dtype, device_map=device,
            trust_remote_code=args.trust_remote_code
        )
    else:
        model = AutoHQQHFModel.from_quantized(
            args.model_path, compute_dtype=compute_dtype, device=device,
            trust_remote_code=args.trust_remote_code
        )

    # patch model for inference
    if args.compile:
        print("Patching model for inference ...")
        prepare_for_inference(model, args.model_name, is_fp=args.is_fp)

    model = model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(
    model, prompt: torch.Tensor, max_new_tokens: int, kv_cache: StaticCache,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """
    device, dtype = prompt.device, prompt.dtype
    stats = {}

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(0)
    empty = torch.empty(T + max_new_tokens, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    t0 = time.perf_counter()
    device_sync()
    next_token = prefill(
        model, prompt.view(1, -1), kv_cache, input_pos, **sampling_kwargs
    )
    device_sync()
    elapsed_time = time.perf_counter() - t0
    stats["prefill_latency"] = elapsed_time # in seconds
    stats["prefill_throughput"] = T / stats["prefill_latency"] # tokens per second

    seq[T] = next_token
    input_pos = torch.tensor([T], device=device, dtype=torch.long)

    t0 = time.perf_counter()
    device_sync()
    generated_tokens = decode_n_tokens(
        model, next_token.view(1, -1), kv_cache, input_pos, max_new_tokens - 1, **sampling_kwargs
    )
    device_sync()
    elapsed_time = time.perf_counter() - t0
    stats["decode_latency"] = elapsed_time # in seconds
    stats["decode_throughput"] = (max_new_tokens - 1) / stats["decode_latency"] # tokens per second

    seq[T + 1:] = torch.cat(generated_tokens)

    return seq, stats


def main(args):
    """
    Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # NOTE: only supports single GPU inference for now
    device = "cuda"
    compute_dtype = torch.float16
    print(f"Using device={device}, dtype={compute_dtype}")
    
    # load and patch model
    print("Loading model ...")
    t0 = time.time()
    model, tokenizer = load_model(args, compute_dtype, device)
    device_sync()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # compile model
    if args.compile:
        global decode_one_token
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

    # encode prompt
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    prompt_length = inputs.input_ids.size(1) # inputs.input_ids [1, S]

    # setup kv cache; static cache is used for compatibility with torch.compile
    max_seq_length = min(prompt_length + args.max_new_tokens, model.config.max_position_embeddings)
    kv_cache = StaticCache(model.config, max_cache_len=max_seq_length)

    # run for `num_samples` rounds
    start = -1 if args.compile else 0
    for i in range(start, args.num_samples):
        device_sync()
        torch.cuda.reset_peak_memory_stats()
        kv_cache.reset()

        # setup profiler
        if i != args.num_samples - 1 or not args.profile:
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()

        # run generation
        t0 = time.perf_counter()
        with prof:
            outputs, stats = generate(
                model,
                inputs.input_ids[0], # [T,]
                args.max_new_tokens,
                kv_cache,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        device_sync()
        t = time.perf_counter() - t0

        # print generated text
        if i == -1:
            print(f"Compilation time: {t:.2f} seconds")
            continue
        print("\n\n" + "="*40 + f"\n Round {i}\n" + "="*40)
        print(tokenizer.decode(outputs.tolist(), skip_special_tokens=True))

        # print stats
        tokens_generated = outputs.size(0) - prompt_length
        print("-"*40 + "\n" + "-"*40)
        print(f"Context length:   {prompt_length}")
        print(f"Generated length: {tokens_generated}")
        print(f"Memory used:      {torch.cuda.max_memory_reserved() / 1024**3:.02f} GB")
        print("-"*53)
        print(f"| {'Stage':<10} | {'Latency (sec)':>13} | {'Throughput (tok/sec)':>20} |")
        print(f"| {'-'*10} | {'-'*13} | {'-'*20} |")
        print(f"| {'Prefill':<10} | {stats['prefill_latency']:>13.2f} | {stats['prefill_throughput']:>20.2f} |")
        print(f"| {'Decode':<10} | {stats['decode_latency']:>13.2f} | {stats['decode_throughput']:>20.2f} |")
        print(f"| {'-'*10} | {'-'*13} | {'-'*20} |")
        print(f"| {'Overall':<10} | {t:>13.2f} | {tokens_generated / t:>20.2f} |")
        print("-"*53)

    # save profiling trace
    if hasattr(prof, "export_chrome_trace"):
        prof.export_chrome_trace(f"{args.profile}.json")


def parse_args():
    parser = argparse.ArgumentParser(description="GEMQ Inference")

    # model args
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Name of the model; used to load model-specific modules",
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--trust_remote_code", action="store_true",
        help="Enable `trust_remote_code` when loading the model from HuggingFace Hub",
    )
    parser.add_argument(
        "--is_fp", action="store_true",
        help="Whether the current model is in full-precision"
    )
    parser.add_argument(
        "--attn_impl", type=str, default="eager", choices=["eager", "sdpa"],
        help="Implementation of attention to use",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Whether to compile the model with torch.compile",
    )

    # inference args
    parser.add_argument(
        "--prompt", type=str, default="Hello, my name is",
        help="Input prompt."
    )
    parser.add_argument(
        "--num_samples", type=int, default=5,
        help="Number of samples (rounds)."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=200,
        help="Maximum number of new tokens."
    )
    parser.add_argument(
        "--top_k", type=int, default=200,
        help="Top-k for sampling."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Temperature for sampling."
    )
    parser.add_argument(
        "--profile", type=str, default="",
        help="Profile path."
    )

    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=4))

    main(args)
