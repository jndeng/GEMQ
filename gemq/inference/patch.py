import importlib
import gc

import torch

from hqq.core.quantize import HQQLinear
import gemlite
from gemlite.core import GemLiteLinearTriton, TORCH_TO_DTYPE

from gemq.utils.model_utils import NAME_TO_MODEL, ModelType, get_blocks
from gemq.inference.moe_block import(
    FusedMixtralMoEBlock, QuantFusedMixtralMoEBlock,
    FusedDeepseekV2MoEBlock, QuantFusedDeepseekV2MoEBlock
)

# monkey-patch to support 3-bit quantization
from gemq.triton_kernels.utils import pack_weights_over_cols_triton
gemlite.core.pack_weights_over_cols_triton = pack_weights_over_cols_triton
importlib.reload(gemlite)
GemLiteLinearTriton.SUPPORTED_BITS_TRITON.append(3)


def create_gemlite_from_hqq(hqq_layer, **kwargs):
    # get meta info
    device = kwargs.get("device", hqq_layer.W_q.device)
    out_features, in_features = hqq_layer.meta["shape"]
    nbits = hqq_layer.meta["nbits"]
    group_size = hqq_layer.meta["group_size"]
    assert hqq_layer.meta["axis"] == 1, "Only axis==1 is supported."

    # get quantized weights and quantization params
    W_q = hqq_layer.unpack(reshape=False, dtype=torch.uint8)
    if nbits == 3:
        W_q = W_q[:out_features * in_features // group_size]
    W_q = W_q.view(out_features, in_features)
    scales = hqq_layer.meta["scale"].clone()
    zeros  = hqq_layer.meta["zero"].clone()
    bias   = hqq_layer.bias.clone() if (hqq_layer.bias is not None) else None
    

    # set to proper device and dtype
    dtype = kwargs.get("dtype", scales.dtype)
    assert scales.dtype in (torch.float16, torch.bfloat16, torch.float32), "Invalid scales.dtype, should floating point."
    
    W_q = W_q.to(device)
    scales = scales.to(dtype=dtype, device=device)
    zeros = zeros.to(dtype=dtype, device=device)
    if bias is not None:
        bias = bias.to(device=device, dtype=dtype)


    # create a new GemLite linear layer
    gemlite_linear = GemLiteLinearTriton(
        nbits,
        group_size,
        in_features,
        out_features,
        input_dtype=TORCH_TO_DTYPE[dtype],
        output_dtype=TORCH_TO_DTYPE[dtype],
        scaled_activations=False,
    )

    # re-pack weights for inference on triton kernels
    gemlite_linear.pack(
        W_q, scales, zeros, bias=bias, packing_bitwidth=32
    )

    return gemlite_linear


def replace_linear_recursive(module, **kwargs):
    """
    Replace all HQQ linears in the model with GemLite linears recursively.
    """
    for name, child in module.named_children():
        if isinstance(child, HQQLinear):
            setattr(module, name, create_gemlite_from_hqq(child, **kwargs))
        else:
            replace_linear_recursive(child, **kwargs)


def replace_moe_blocks(model, model_name, config, is_fp=False):
    """
    Replace MoE blocks in HF models for efficient inference.
    """

    def _replace_mixtral(layer, del_orig=True):
        org_block = layer.block_sparse_moe

        if is_fp:
            layer.block_sparse_moe = FusedMixtralMoEBlock.from_hf(config, org_block)
        else:
            layer.block_sparse_moe = QuantFusedMixtralMoEBlock.from_hf(config, org_block)
        
        if del_orig:
            del org_block
            gc.collect()
            torch.cuda.empty_cache()

    def _replace_deepseekv2(layer, del_orig=True):
        org_block = layer.mlp

        if is_fp:
            layer.mlp = FusedDeepseekV2MoEBlock.from_hf(config, org_block)
        else:
            layer.mlp = QuantFusedDeepseekV2MoEBlock.from_hf(config, org_block)
        
        if del_orig:
            del org_block
            gc.collect()
            torch.cuda.empty_cache()


    model_type = NAME_TO_MODEL[model_name]
    layers = get_blocks(model, model_name)
    for i in range(len(layers)):
        layer = layers[i]

        if model_type == ModelType.MIXTRAL:
            _replace_mixtral(layer, del_orig=True)
        elif model_type == ModelType.DEEPSEEKV2:
            if i > 0: # NOTE: the first layer of this model is a dense layer instead of MoE
                _replace_deepseekv2(layer, del_orig=True)
        else:
            raise NotImplementedError


def prepare_for_inference(model, model_name, is_fp=False):
    """
    Prepare the model for accelerated inference by replacing blocks in the model.
    """

    if is_fp:
        # replace MoE blocks to fused ones
        replace_moe_blocks(model, model_name, model.config, is_fp=True)
    else:
        # NOTE: for quantized models, we need to first replace HQQ linears with GemLite linears
        replace_linear_recursive(model)
        replace_moe_blocks(model, model_name, model.config, is_fp=False)
    
    gc.collect()
    torch.cuda.empty_cache()

    return model
