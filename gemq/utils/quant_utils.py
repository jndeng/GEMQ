import pickle

from hqq.core.quantize import HQQLinear, BaseQuantizeConfig
from hqq.core.quantize import Quantizer

from gemq.utils.model_utils import *


def build_alloc_cfg(model, args):
    """
    Build a bit allocation config for the model. The config includes bitwidth for
    each Linear modules required quantization.

    Returns:
        bit_cfg: a list containing alloction cfg for Linear modules in each block/layer
        e.g.,
        [
            {"module1": 1, "module2": 3, "module3": 2},  # layer1 cfg
            ...  
        ]
    """
    # load bit allocation config
    # NOTE: current cfg only supports mixed-precision for expert-ffn
    if args.mixed:
        with open(args.bit_cfg, "rb") as f:
            expert_bit_cfg = pickle.load(f)

    layers = get_blocks(model, args.model_name)

    bit_cfg = []
    for i in range(len(layers)):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        layer_bit_cfg = {}
        for name, m in named_linears.items():
            mtype = get_module_type(name, args.model_name)
            if mtype == LinearModuleType.ATTN:
                layer_bit_cfg[name] = args.attn_wbits
            elif mtype == LinearModuleType.GATE:
                layer_bit_cfg[name] = args.gate_wbits
            elif mtype == LinearModuleType.DENSE:
                layer_bit_cfg[name] = args.dense_wbits
            elif mtype == LinearModuleType.EXPERT:
                if args.mixed:
                    expert_id = get_expert_id(name, args.model_name)
                    layer_bit_cfg[name] = expert_bit_cfg[i][expert_id]
                else:
                    layer_bit_cfg[name] = args.expert_wbits
        bit_cfg.append(layer_bit_cfg)

    return bit_cfg


def create_hqq_linear_from_quantized_weights(
    W_q, scale, zero, org_shape, nbits, group_size, bias=None, device="cuda"
):
    """
    Create a HQQLinear layer from quantized weights without running on-the-fly HQQ quantization.

    NOTE: We assume the grouped quantization is performed at hidden_dim (i.e., axis=1).
    """
    # create a dummy HQQLinear layer
    # skip initialization (HQQ quantization) by passing a None linear_layer
    quant_config = BaseQuantizeConfig(nbits, group_size)
    quant_layer = HQQLinear(None, quant_config, compute_dtype=W_q.dtype, device=device)
    
    # set W_q and meta
    W_q = W_q.float()

    # store meta-data
    meta = {
        "nbits": nbits,
        "group_size": group_size,
        "shape": org_shape,
        "scale": scale,
        "zero": zero,
        "axis": 1,
        "packing": Quantizer.bit_to_packing[nbits],
        "quant_scale": False,
        "quant_zero": False,
    }
    meta["unpack_view_dtype"] = Quantizer.unpack_view_dtype[meta["packing"]]

    # pack bits
    meta["view_as_float"] = False
    W_q = Quantizer.pack[meta["packing"]](W_q)

    # set attributes
    quant_layer.W_q = W_q
    quant_layer.meta = meta
    if bias is not None:
        quant_layer.bias = bias.data.clone().to(device=device, dtype=quant_layer.compute_dtype)

    quant_layer.cuda(device)
    quant_layer.ready = True  # to enable dequantization

    return quant_layer


def getattr_nested(obj, attr_path):
    """
    Get a nested attribute of an object given a dot-separated attribute path.
    """
    parts = attr_path.split(".")
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]  # for ModuleList
        else:
            obj = getattr(obj, p)
    return obj


def setattr_nested(obj, attr_path, value):
    """
    Set a nested attribute of an object given a dot-separated attribute path.
    """
    parts = attr_path.split(".")
    for p in parts[:-1]:
        if p.isdigit():
            obj = obj[int(p)]  # for ModuleList
        else:
            obj = getattr(obj, p)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def replace_linears(model, model_name, quant_modules, quant_weight=True):
    """
    Replace quantized nn.Linear in model with HQQLinear modules.
    """
    layers = get_blocks(model, model_name)
    for name, m in quant_modules.items():
        # retrieve quantization params
        W = m.weight.data
        scales = m.quant_scales
        zeros = m.quant_zeros
        nbits = m.quant_nbits.item()
        group_size = m.quant_groupsize.item()

        if quant_weight:
            num_groups = scales.shape[0]
            Q = torch.round(
                W.view(num_groups, -1) / scales + zeros
            ).clamp(0, 2**nbits - 1)
        else:
            Q = W

        # Replace with HQQLinear
        hqq_linear = create_hqq_linear_from_quantized_weights(
            Q, scales, zeros, m.weight.shape, nbits, group_size, bias=m.bias, device="cuda"
        )
        setattr_nested(layers, name, hqq_linear)


def check_packing(model, quant_modules, args):
    """
    Sanity check for weight packing and unpacking.
    """
    layers = get_blocks(model, args.model_name)

    max_rec_err = 0.0
    for name, m in quant_modules.items():
        qlinear = getattr_nested(layers, name)
        W_r = qlinear.dequantize()

        avg_err = (m.weight.data.cuda() - W_r.cuda()).abs().mean().item()
        max_rec_err = max(max_rec_err, avg_err)

    print(f"Max reconstruction error among all quantized linear layers: {max_rec_err}")
