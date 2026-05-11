import torch
from torch import Tensor

import triton
import triton.language as tl

from gemq.triton_kernels.utils import dequantize


def get_autotune_config(pre_hook=None):
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE_N':8,  'BLOCK_SIZE_K':128}, num_warps=16, num_stages=1, pre_hook=pre_hook))
    configs.append(triton.Config({'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':256}, num_warps=16, num_stages=2, pre_hook=pre_hook))

    configs.append(triton.Config({'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':16}, num_warps=1, num_stages=1, pre_hook=pre_hook))
    configs.append(triton.Config({'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32}, num_warps=1, num_stages=1, pre_hook=pre_hook))
    configs.append(triton.Config({'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64}, num_warps=1, num_stages=1, pre_hook=pre_hook))
    
    configs.append(triton.Config({'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':16}, num_warps=1, num_stages=2, pre_hook=pre_hook))
    configs.append(triton.Config({'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32}, num_warps=1, num_stages=1, pre_hook=pre_hook))
    configs.append(triton.Config({'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32}, num_warps=2, num_stages=2, pre_hook=pre_hook))
    configs.append(triton.Config({'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64}, num_warps=2, num_stages=1, pre_hook=pre_hook))
    configs.append(triton.Config({'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128}, num_warps=2, num_stages=2, pre_hook=pre_hook))

    configs.append(triton.Config({'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':16}, num_warps=2, num_stages=1, pre_hook=pre_hook))
    configs.append(triton.Config({'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':32}, num_warps=4, num_stages=2, pre_hook=pre_hook))
    configs.append(triton.Config({'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':64}, num_warps=4, num_stages=2, pre_hook=pre_hook))

    configs.append(triton.Config({'BLOCK_SIZE_N':512, 'BLOCK_SIZE_K':64}, num_warps=2, num_stages=1, pre_hook=pre_hook))
    return configs


def get_max_autotune_config(pre_hook=None):
    configs = []
    for N in [8, 32, 128, 512]:
        for K in [16, 64, 128, 256]:
            for num_warps in [2, 4, 16]:
                for num_stages in [1, 2]:
                    configs.append(triton.Config({"BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K}, num_warps=num_warps, num_stages=num_stages, pre_hook=pre_hook))
    return configs


@triton.autotune(
    configs=get_autotune_config(),
    key=["N", "K"],
    reset_to_zero=["c_ptr"] # NOTE: reset output for atomic adds
)
@triton.jit
def dequant_splitk_gemv_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    # Matrix dimensions
    N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_qk, stride_qn,
    # Quantization parameters
    nbits: tl.constexpr,
    group_size: tl.constexpr,
    # Meta parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    GEMV for C = matmul(A, dequantize(B, scales, zeros))
    assert M == 1

    A is of shape [1, K]: float16
    B is of shape [K//elements_per_sample, N]: int32 as a packed matrix
    C is of shape [1, N]: float16

    scales and zeros is of shape [num_groups, N]: float16
    NOTE: dequant is computed as: B * scales + zeros

    NOTE: fix compute dtype to float16 for simplicity
    """
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    
    # create offsets
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # load A
    a_ptrs = a_ptr + offs_k * stride_ak # NOTE: 1D
    a = tl.load(a_ptrs, mask=offs_k < K, other=0.0)

    # load B, scales and zeros
    elements_per_sample = 32 // nbits # NOTE: B is assumed to be packed in int32
    b_ptrs = b_ptr + (offs_k[:, None] // elements_per_sample) * stride_bk + offs_n[None, :] * stride_bn

    s_ptrs = scales_ptr + (offs_k[:, None] // group_size) * stride_qk + offs_n[None, :] * stride_qn
    z_ptrs = zeros_ptr  + (offs_k[:, None] // group_size) * stride_qk + offs_n[None, :] * stride_qn
    
    mask_2d = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    b = tl.load(b_ptrs, mask=mask_2d, other=0.0)
    s = tl.load(s_ptrs, mask=mask_2d, other=0.0)
    z = tl.load(z_ptrs, mask=mask_2d, other=0.0)

    # unpack and dequantize
    q_shift = (offs_k % elements_per_sample * nbits).to(tl.int32)[:, None] 
    unpack_mask = (1 << nbits) - 1
    b = dequantize(b, s, z, q_shift, unpack_mask)

    # dot product
    acc = tl.sum(a[:, None].to(tl.float32) * b.to(tl.float32), axis=0)

    # accumulate
    c_ptrs = c_ptr + offs_n * stride_cn
    tl.atomic_add(c_ptrs, acc, mask=offs_n < N, sem="relaxed")


def dequant_splitk_gemv_triton(
    x: Tensor, w_q: Tensor, scales: Tensor, zeros: Tensor, nbits: int, group_size: int
) -> Tensor:
    """
    Matmul with quantized weights: output = matmul(x, dequantize(w_q, scales, zeros))
    NOTE: assert M == 1

    Args:
        x:      [1, K]
        w_q:    [K // elements_per_sample, N]
        scales: [num_groups, N]
        zeros:  [num_groups, N]
    Returns:
        output: [1, N]
    """
    M, K = x.shape
    _, N = w_q.shape

    # allocates output
    output = torch.zeros((M, N), device=x.device, dtype=torch.float16)

    # 2D grid
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]), triton.cdiv(K, META["BLOCK_SIZE_K"]))

    dequant_splitk_gemv_kernel[grid](
        x, w_q, output,
        scales, zeros,
        N, K,
        x.stride(0), x.stride(1),
        w_q.stride(0), w_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        nbits, group_size,
        # NOTE: comment out for autotune
        # BLOCK_SIZE_N=64,
        # BLOCK_SIZE_K=64,
    )

    return output


@triton.autotune(
    configs=get_autotune_config(),
    key=["N", "K"],
    reset_to_zero=["c_ptr"] # NOTE: reset output for atomic adds
)
@triton.jit
def dequant_sel_splitk_gemv_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, idx_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    nbits_ptr, b_stride_ptr,
    # Matrix dimensions
    N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_qk, stride_qn,
    # Quantization parameters
    group_size: tl.constexpr,
    # Meta parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)
    
    # create offsets
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # load idx and expert-specific params
    idx = tl.load(idx_ptr)
    nbit = tl.load(nbits_ptr + idx)
    stride_b = tl.load(b_stride_ptr + idx)

    # load A
    a_ptrs = a_ptr + offs_k * stride_ak # NOTE: 1D
    a = tl.load(a_ptrs, mask=offs_k < K, other=0.0)

    # load B, scales and zeros
    elements_per_sample = 32 // nbit # NOTE: B is assumed to be packed in int32
    b_ptrs = b_ptr + stride_b + (offs_k[:, None] // elements_per_sample) * stride_bk + offs_n[None, :] * stride_bn

    stride_zs = idx * K // group_size * stride_qk
    s_ptrs = scales_ptr + stride_zs + (offs_k[:, None] // group_size) * stride_qk + offs_n[None, :] * stride_qn
    z_ptrs = zeros_ptr + stride_zs + (offs_k[:, None] // group_size) * stride_qk + offs_n[None, :] * stride_qn
    
    mask_2d = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    b = tl.load(b_ptrs, mask=mask_2d, other=0.0)
    s = tl.load(s_ptrs, mask=mask_2d, other=0.0)
    z = tl.load(z_ptrs, mask=mask_2d, other=0.0)

    # unpack and dequantize
    q_shift = (offs_k % elements_per_sample * nbit).to(tl.int32)[:, None] 
    unpack_mask = (1 << nbit) - 1
    b = dequantize(b, s, z, q_shift, unpack_mask)

    # dot product
    acc = tl.sum(a[:, None].to(tl.float32) * b.to(tl.float32), axis=0)

    # accumulate
    c_ptrs = c_ptr + offs_n * stride_cn
    tl.atomic_add(c_ptrs, acc, mask=offs_n < N, sem="relaxed")


def dequant_sel_splitk_gemv_triton(
    x: Tensor, idx: Tensor, w_q: Tensor, scales: Tensor, zeros: Tensor,
    nbits: Tensor, wq_strides: Tensor,
    group_size: int, # NOTE: we assume same group_size for all experts
) -> Tensor:
    """
    Matmul with quantized weights: output = matmul(x, dequantize(w_q[idx], scales[idx], zeros[idx], nbits[idx]))
    Designed for mixed-prec quantized and stacked w_q (e.g., mixed-prec quantized MoE)

    Args:
        x:          [1, K]
        idx:        [1,] index of experts to be multiplied x with (\in [0, E-1])
        wq:         [sum(K // elements_per_sample[e]), N] stacked quantized weights of all experts
        scales:     [sum(K // group_size[e]), N] stacked scales of all experts
        zeros:      [sum(K // group_size[e]), N] stacked zeros of all experts
        nbits:      [E,] number of bits for each expert
        wq_strides: [E,] stride of quantized weights for each expert
        group_size: int, assumed same for all experts
    Returns:
        output:     [1, N]
    """
    M, K = x.shape
    _, N = w_q.shape

    # allocates output
    output = torch.zeros((M, N), device=x.device, dtype=torch.float16)

    # 2D grid
    grid = lambda META: (triton.cdiv(N, META["BLOCK_SIZE_N"]), triton.cdiv(K, META["BLOCK_SIZE_K"]))

    dequant_sel_splitk_gemv_kernel[grid](
        x, w_q, idx, output,
        scales, zeros,
        nbits, wq_strides,
        N, K,
        x.stride(0), x.stride(1),
        w_q.stride(0), w_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        group_size,
        # NOTE: comment out for autotune
        # BLOCK_SIZE_N=64,
        # BLOCK_SIZE_K=64,
    )

    return output
