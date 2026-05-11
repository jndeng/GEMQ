import gc
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from gemq.utils.model_utils import get_blocks, move_embed, move_head


def get_testenc(tokenizer, dataset, seqlen):
    if dataset == "wikitext2":
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    elif dataset == "c4":
        testdata = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")
        testenc = tokenizer(" ".join(testdata[:1100]["text"]), return_tensors="pt")
        testenc = testenc.input_ids[:, :(256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        testenc = TokenizerWrapper(testenc)

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")

    return testenc


def compute_perplexity(model, input_ids, dataset_name) -> float:
    """
    Compute the perplexity of the model on the given dataset.
    """
    nlls = []
    nsamples = input_ids.numel() // model.seqlen
    for i in tqdm(range(nsamples), desc=f"Evaluating [{dataset_name}]"):
        batch = input_ids[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(model.device)
        lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:].to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen)).item()
    return ppl


def compute_perplexity_offload(model, model_name, input_ids, dataset_name):
    """
    Compute the perplexity of the model on the given dataset.
    This function uses dynamic weights offloading for memory-efficient evaluation.
    """
    # disable kv cache since we are running batch generation for evaluation
    use_cache = model.config.use_cache
    model.config.use_cache = False

    nsamples = input_ids.numel() // model.seqlen

    # retrieve blocks that require quantization
    layers = get_blocks(model, model_name)

    # get input and kwargs to the first layer decoding layer
    inps = []
    layer_kwargs = {}

    move_embed(model, model_name, "cuda")
    layers[0] = layers[0].to("cuda")
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

            if hasattr(self.module, "attention_type"):
                self.attention_type = self.module.attention_type

        def forward(self, inp, **kwargs):
            inps.append(inp)  # NOTE: inp is (bsz, seqlen, hidden_size)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = input_ids[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to("cuda")
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module  # restore
    inps = torch.cat(inps, dim=0)  # (nsamples, seqlen, hidden_size)
    
    # for memory savings
    move_embed(model, model_name, "cpu")
    layers[0] = layers[0].cpu()
    gc.collect()
    torch.cuda.empty_cache()


    # forward pass with dynamic offloading
    outs = torch.zeros_like(inps)
    for i in tqdm(range(len(layers)), desc=f"Evaluating [{dataset_name}]"):
        layer = layers[i].to("cuda")
        for j in range(nsamples):
            outs[j] = layer(inps[j: j+1], **layer_kwargs)[0]
        layers[i] = layer.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        inps, outs = outs, inps
    move_head(model, model_name, "cuda")

    # compute perplexity
    nlls = []
    nsamples = input_ids.numel() // model.seqlen
    for i in range(nsamples):
        hidden_states = model.model.norm(inps[i:i + 1])
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = input_ids[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:].to("cuda")
        # loss_fct = nn.CrossEntropyLoss()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen)).item()

    model.config.use_cache = use_cache  # restore

    return ppl


@torch.inference_mode()
def evaluate_perplexity(model, tokenizer, datasets, model_name, offload=True):
    """
    Evaluate the model on a given dataset.
    """
    # NOTE: disable kv cache since we are running batch generation for evaluation
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # for each dataset
    for dataset in datasets:
        testenc = get_testenc(tokenizer, dataset, model.seqlen)
        if offload:
            ppl = compute_perplexity_offload(model, model_name, testenc.input_ids, dataset)
        else:
            ppl = compute_perplexity(model, testenc.input_ids, dataset)
        print(f"[{dataset}] ppl: {ppl:.4f}")

    # restore
    model.config.use_cache = use_cache



def run_lm_eval(model, tokenizer, tasks=["mmlu"], batch_size=32, num_fewshot=0):
    try:
        from lm_eval import evaluator, utils
        from lm_eval.models.huggingface import HFLM
    except:
        print("lm_eval package not found. Skipping downstream evaluation ...")
        return


    # wrap the model with lm_eval's HFLM
    lm_eval_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size="auto",
    )

    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        log_samples=False
    )

    outputs = []
    acc_sum = 0.0
    for task in tasks:
        if "acc_norm,none" in results["results"][task]:
            acc = results["results"][task]["acc_norm,none"]
            acc_sum += acc
            header = f"{task} (acc_norm)"

        elif "acc,none" in results["results"][task]:
            acc = results["results"][task]["acc,none"]
            acc_sum += acc
            header = f"{task} (acc)"

        else:
            raise ValueError(f"Unknown metric for task {task}")
        
        output_str = f"{header:<25}: {acc*100:.2f} (%)"
        outputs.append(output_str)
    print("\n".join(outputs))
    print(f"Avg: {acc_sum/len(tasks)*100:.2f} (%)")
