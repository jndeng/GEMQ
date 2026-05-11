import gc
import time
from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MoE
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils.modeling import get_balanced_memory


class ModelType(Enum):
    # Dense models
    LLAMA2 = auto()
    QWEN3 = auto()

    # MoE models
    MIXTRAL = auto()
    DEEPSEEKV2 = auto()
    OLMOE = auto()
    

class LinearModuleType(Enum):
    ATTN = auto()
    GATE = auto()
    EXPERT = auto()
    DENSE = auto()
    OTHERS = auto()


NAME_TO_MODEL = {
    "meta-llama/Llama-2-7b-hf": ModelType.LLAMA2,
    "Qwen/Qwen3-8B": ModelType.QWEN3,

    "mistralai/Mixtral-8x7B-v0.1": ModelType.MIXTRAL,
    "deepseek-ai/DeepSeek-V2-Lite": ModelType.DEEPSEEKV2,
    "allenai/OLMoE-1B-7B-0924": ModelType.OLMOE,
}


@dataclass
class ModelInfo:
    num_layers: int
    first_k_dense_layers: int
    num_routed_experts_per_layer: int
    num_shared_experts_per_layer: int
    num_experts_per_token: int


def dispatch_model_to_all_devices(model):
    """
    Dispatch model to all available devices.
    """
    print("Dispatching model weights to all devices ... ", end="")
    t0 = time.time()
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[
            "LlamaDecoderLayer",
            "Qwen3DecoderLayer",
            "MixtralDecoderLayer",
            "DeepseekV2DecoderLayer",

        ],
        max_memory=get_balanced_memory(model),
    )
    model = dispatch_model(model, device_map=device_map)
    torch.cuda.synchronize()
    print(f"Done in {(time.time() - t0)/60:.2f} minutes")
    return model


def get_model_info(model_name):
    """
    Get basic model info (#layers, #experts, etc.).
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type == ModelType.MIXTRAL:
        model_info = ModelInfo(
            num_layers=32,
            first_k_dense_layers=0,
            num_routed_experts_per_layer=8,
            num_shared_experts_per_layer=0,
            num_experts_per_token=2,
        )
    elif model_type == ModelType.DEEPSEEKV2:
        model_info = ModelInfo(
            num_layers=27,
            first_k_dense_layers=1,
            num_routed_experts_per_layer=64,
            num_shared_experts_per_layer=2,
            num_experts_per_token=6,
        )
    elif model_type == ModelType.OLMOE:
        model_info = ModelInfo(
            num_layers=16,
            first_k_dense_layers=0,
            num_routed_experts_per_layer=64,
            num_shared_experts_per_layer=0,
            num_experts_per_token=8,
        )
    else:
        raise NotImplementedError(f"Model type {model_type} not supported for getting model info.")

    return model_info


def get_blocks(model, model_name):
    """
    Retrieve a list of decoder blocks (layers).
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type in (
        ModelType.LLAMA2, ModelType.QWEN3,
        ModelType.MIXTRAL, ModelType.DEEPSEEKV2, ModelType.OLMOE,
    ):
        blocks = model.model.layers
    else:
        raise NotImplementedError(f"Model type {model_type} not supported for getting blocks.")
    return blocks


def move_embed(model, model_name, device):
    """
    Move the embedding layer to the specified device.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type in (
        ModelType.LLAMA2, ModelType.QWEN3,
        ModelType.MIXTRAL, ModelType.DEEPSEEKV2, ModelType.OLMOE,
    ):
        model.model.embed_tokens = model.model.embed_tokens.to(device)


def move_head(model, model_name, device):
    """
    Move the LM head to the specified device.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type in (
        ModelType.LLAMA2, ModelType.QWEN3,
        ModelType.MIXTRAL, ModelType.DEEPSEEKV2, ModelType.OLMOE,
    ):
        model.model.norm = model.model.norm.to(device)
        model.lm_head = model.lm_head.to(device)


def get_named_linears(module):
    """
    Return name-module pairs for linear sub-modules.
    module: a decoder layer
    """
    is_gate = lambda name: name.endswith("gate")
    return {
        name: m for name, m in module.named_modules()
        if (isinstance(m, nn.Linear) or is_gate(name))
    }


def get_moe_block(layer, model_name):
    """
    Get the MoE block from a decoder layer.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type == ModelType.MIXTRAL:
        moe_block = layer.block_sparse_moe
    elif model_type == ModelType.DEEPSEEKV2:
        moe_block = layer.mlp
    elif model_type == ModelType.OLMOE:
        moe_block = layer.mlp
    
    return moe_block


def get_shared_expert_block(moe_block, model_name):
    """
    Get the shared expert FFN from a moe block.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type == ModelType.DEEPSEEKV2:
        shared_expert = moe_block.shared_experts
    else:
        raise NotImplementedError(f"Model type {model_type} does not have shared experts.")
    return shared_expert


def get_sublinear_names(model_name):
    """
    Get names of sub-linear modules in a FFN.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type == ModelType.MIXTRAL:
        sublinear_names = ["w1", "w2", "w3"]
    elif model_type in (ModelType.DEEPSEEKV2, ModelType.OLMOE):
        sublinear_names = ["gate_proj", "up_proj", "down_proj"]
    
    return sublinear_names


def get_module_type(module_name, model_name):
    """
    Parse the type of **linear module** based on its name.
    """
    model_type = NAME_TO_MODEL[model_name]

    if model_type == ModelType.LLAMA2:
        if "attn" in module_name:
            mtype = LinearModuleType.ATTN
        else:
            mtype = LinearModuleType.DENSE
    
    elif model_type == ModelType.QWEN3:
        if "attn" in module_name:
            mtype = LinearModuleType.ATTN
        else:
            mtype = LinearModuleType.DENSE

    elif model_type == ModelType.MIXTRAL:
        if "attn" in module_name:
            mtype = LinearModuleType.ATTN
        elif "gate" in module_name:
            mtype = LinearModuleType.GATE
        elif "experts" in module_name:
            mtype = LinearModuleType.EXPERT
        else:
            mtype = LinearModuleType.OTHERS

    elif model_type == ModelType.DEEPSEEKV2:
        if "attn" in module_name:
            mtype = LinearModuleType.ATTN
        elif ("gate" in module_name) and ("proj" not in module_name):
            mtype = LinearModuleType.GATE
        elif "mlp" in module_name and ("experts" not in module_name):
            mtype = LinearModuleType.DENSE
        elif ("experts" in module_name) or ("shared_experts" in module_name):
            mtype = LinearModuleType.EXPERT
        else:
            mtype = LinearModuleType.OTHERS
    
    elif model_type == ModelType.OLMOE:
        if "attn" in module_name:
            mtype = LinearModuleType.ATTN
        elif ("gate" in module_name) and ("proj" not in module_name):
            mtype = LinearModuleType.GATE
        elif ("experts" in module_name):
            mtype = LinearModuleType.EXPERT
        else:
            mtype = LinearModuleType.OTHERS


    return mtype


def get_expert_id(name, model_name):
    """
    Get the expert id from the name of the Linear module.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type in (ModelType.MIXTRAL, ModelType.OLMOE):
        exp_id = int(name.split(".")[-2])
    elif model_type == ModelType.DEEPSEEKV2:
        exp_id = 64 if "shared_experts" in name else int(name.split(".")[-2])

    return exp_id


def get_all_expert_names(model_name):
    """
    Get all expert linear module names (including both routed and shared) in the model.
    """
    model_type = NAME_TO_MODEL[model_name]
    model_info = get_model_info(model_name)
    if model_type == ModelType.MIXTRAL:
        all_expert_names = [f"block_sparse_moe.experts.{i}" for i in range(model_info.num_routed_experts_per_layer)]
    elif model_type == ModelType.DEEPSEEKV2:
        all_expert_names = [f"mlp.experts.{i}" for i in range(model_info.num_routed_experts_per_layer)] + ["mlp.shared_experts"]
    elif model_type == ModelType.OLMOE:
        all_expert_names = [f"mlp.experts.{i}" for i in range(model_info.num_routed_experts_per_layer)]

    return all_expert_names


def get_router_params(model, model_name):
    """
    Get all router parameters in the model.
    """
    layers = get_blocks(model, model_name)
    router_params = []
    for layer in layers:
        linears = get_named_linears(layer)
        for name, m in linears.items():
            mtype = get_module_type(name, model_name)
            if mtype == LinearModuleType.GATE:
                for param in m.parameters(): # in case bias exists
                    router_params.append(param)

    return router_params


def compute_decoder_inputs(model, dataloader, model_name, device="cuda"):
    """
    Prepare input data for the first decoder block, and shared kwargs for all blocks.
    """
    layers = get_blocks(model, model_name)

    # get input and kwargs to the first layer decoding layer
    # NOTE: kwargs are shared across all layers
    inps = []
    layer_kwargs = {}

    move_embed(model, model_name, device)
    layers[0] = layers[0].to(device)
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

            # NOTE: ad-hoc for Qwen3
            if hasattr(self.module, "attention_type"):
                self.attention_type = self.module.attention_type

        def forward(self, inp, **kwargs):
            inps.append(inp)  # NOTE: inp is (bsz, seqlen, hidden_size)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module  # restore
    inps = torch.cat(inps, dim=0)  # (nsamples, seqlen, hidden_size)
    
    # for memory savings
    move_embed(model, model_name, "cpu")
    layers[0] = layers[0].cpu()

    gc.collect()
    torch.cuda.empty_cache()

    return inps, layer_kwargs


def compute_gate_stats_hook_mixtral(m, x, y, inps, outs, weights, counts):
    """
    Hook function to compute gate statistics (expert frequency and weights) for MixtralSparseMoeBlock block.
    """
    assert isinstance(m, MixtralSparseMoeBlock)

    hidden_states = x[0]  # (bsz, seqlen, hidden_size)
    final_hidden_states = y[0]  # (bsz, seqlen, hidden_size)
    router_logits = y[1]  # (bsz * sequence_length, n_experts)
    
    # compute gate outputs
    # routing_weights:  (batch * sequence_length, topk)
    # selected_experts: (batch * sequence_length, topk)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, m.top_k, dim=-1)

    # compute weights
    actw = torch.zeros(m.num_experts, device=router_logits.device)
    actw.scatter_add_(0, selected_experts.view(-1), routing_weights.view(-1))
    weights.append(actw.to("cpu"))

    # compute counts
    actc = torch.zeros(m.num_experts, dtype=torch.long, device=router_logits.device)
    ones = torch.ones_like(selected_experts.view(-1), device=router_logits.device)
    actc.scatter_add_(0, selected_experts.view(-1), ones)
    counts.append(actc.to("cpu"))

    # save inputs and outputs
    inps.append(x[0])  # (bsz, seqlen, hidden_size)
    outs.append(y[0])  # (bsz, seqlen, hidden_size)


def compute_gate_stats_hook_deepseekmoe(m, x, y, inps, outs, weights, counts):
    """
    Hook function to compute gate statistics (expert frequency and weights) for DeepseekV2MoE block.
    """
    # NOTE: this function should be compatible with deepseekmoe, but only tested on DeepseekV2MoE.
    assert isinstance(m, DeepseekV2MoE)

    hidden_states = x[0]  # (bsz, seqlen, hidden_size)
    device = hidden_states.device

    # NOTE: get weights before renormalization
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    # topk_idx:    (bsz*seqlen, topk)
    # topk_weight: (bsz*seqlen, topk)
    hidden_states = hidden_states.view(-1, hidden_dim)     # (bsz*seqlen, hidden_size)
    logits = F.linear(hidden_states, m.gate.weight, None)  # (bsz*seqlen, 64)
    scores = logits.softmax(dim=-1, dtype=torch.float)     # (bsz*seqlen, 64)
    topk_weight, topk_idx = torch.topk(scores, k=m.num_experts_per_tok, dim=-1, sorted=False)

    # =================================
    # for routed experts
    # =================================
    # compute weights
    actw = torch.zeros(m.gate.n_routed_experts, device=device)
    actw.scatter_add_(0, topk_idx.view(-1), topk_weight.view(-1))

    # compute counts
    actc = torch.zeros(m.gate.n_routed_experts, dtype=torch.long, device=device)
    ones = torch.ones_like(topk_idx.view(-1), device=device)
    actc.scatter_add_(0, topk_idx.view(-1), ones)

    # =================================
    # for shared expert
    # =================================
    shared_actw = scores.shape[0]
    shared_actc = scores.shape[0]

    # =================================
    # combine shared and routed experts
    # =================================
    # NOTE: the shared expert is always put at the end
    shared_actw = actw.new_ones(1) * shared_actw
    actw = torch.cat([actw, shared_actw], dim=0)  # (num_experts + 1,)
    shared_actc = actc.new_ones(1) * shared_actc
    actc = torch.cat([actc, shared_actc], dim=0)  # (num_experts + 1,)
    weights.append(actw.to("cpu"))
    counts.append(actc.to("cpu"))

    # save inputs and outputs
    inps.append(x[0])  # (bsz, seqlen, hidden_size)
    outs.append(y[0])  # (bsz, seqlen, hidden_size)


def compute_gate_stats_hook_olmoe(m, x, y, inps, outs, weights, counts):
    """
    Hook function to compute gate statistics (expert frequency and weights) for OlmoeSparseMoeBlock block.
    """
    assert isinstance(m, OlmoeSparseMoeBlock)

    hidden_states = x[0]  # (bsz, seqlen, hidden_size)
    final_hidden_states = y[0]  # (bsz, seqlen, hidden_size)
    router_logits = y[1]  # (bsz * sequence_length, n_experts)
    
    # compute gate outputs
    # routing_weights:  (batch * sequence_length, topk)
    # selected_experts: (batch * sequence_length, topk)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, m.top_k, dim=-1)

    # compute weights
    actw = torch.zeros(m.num_experts, device=router_logits.device)
    actw.scatter_add_(0, selected_experts.view(-1), routing_weights.view(-1))
    weights.append(actw.to("cpu"))

    # compute counts
    actc = torch.zeros(m.num_experts, dtype=torch.long, device=router_logits.device)
    ones = torch.ones_like(selected_experts.view(-1), device=router_logits.device)
    actc.scatter_add_(0, selected_experts.view(-1), ones)
    counts.append(actc.to("cpu"))

    # save inputs and outputs
    inps.append(x[0])  # (bsz, seqlen, hidden_size)
    outs.append(y[0])  # (bsz, seqlen, hidden_size)


def get_gate_stats_hook_fn(model_name):
    """
    Get the appropriate hook function for computing router statistics based on model type.
    """
    model_type = NAME_TO_MODEL[model_name]
    if model_type == ModelType.MIXTRAL:
        hook_fn = compute_gate_stats_hook_mixtral
    elif model_type == ModelType.DEEPSEEKV2:
        hook_fn = compute_gate_stats_hook_deepseekmoe
    elif model_type == ModelType.OLMOE:
        hook_fn = compute_gate_stats_hook_olmoe
    else:
        raise NotImplementedError(f"Model type {model_type} not supported for gate stats computation.")

    return hook_fn
