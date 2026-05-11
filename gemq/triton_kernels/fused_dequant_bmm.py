import torch
from torch import Tensor

import triton
import triton.language as tl

from gemq.triton_kernels.utils import dequantize


def get_autotune_config(pre_hook=None):
    configs = []
    for N in [4, 8, 32, 64, 128]:
        for K in [128, 256, 512]:
            for num_warps in [8, 16]:
                for num_stages in [1, 2]:
                    configs.append(triton.Config({"BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K}, num_warps=num_warps, num_stages=num_stages, pre_hook=pre_hook))
    return configs


@triton.autotune(
    configs=get_autotune_config(),
    key=["N", "K", "A"],
)
@triton.jit
def fused_dequant_up_proj_kernel(
    # Pointers to matrices
    x_ptr, idx_ptr, w1_ptr, w3_ptr, x1_ptr, x3_ptr,
    w1_scales_ptr, w1_zeros_ptr, w3_scales_ptr, w3_zeros_ptr,
    nbits_ptr, stride_in_ptr,
    # Matrix dimensions
    N, K, A,
    # Strides
    stride_xm, stride_xk,
    stride_ik, stride_in,
    stride_ok, stride_on,
    stride_qk, stride_qn,
    # Quantization parameters
    group_size: tl.constexpr,
    # Meta parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)

    # create offsets
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < A * N
    offs_k_base = tl.arange(0, BLOCK_SIZE_K)

    # NOTE: assumes BLOCK_SIZE_N divides N, i.e., no cross-expert within a block
    tl.device_assert((N % BLOCK_SIZE_N) == 0, "N must be multiple of BLOCK_SIZE_N")
    block_idx = pid_n * BLOCK_SIZE_N // N  # get a scalar block_idx
    
    # load expert params as scalars
    idx = tl.load(idx_ptr + block_idx)
    nbit = tl.load(nbits_ptr + idx)
    stride_w = tl.load(stride_in_ptr + idx)

    # pre-calculate scalar constants outside the loop
    elements_per_sample = 32 // nbit
    unpack_mask = (1 << nbit) - 1
    stride_zs = idx * (K // group_size) * stride_qk

    # split K dimension
    x1_acc = tl.full([BLOCK_SIZE_K, BLOCK_SIZE_N], 0, tl.float32)
    x3_acc = tl.full([BLOCK_SIZE_K, BLOCK_SIZE_N], 0, tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # update k offsets
        offs_k = offs_k_base + k
        mask_k = offs_k < K
        mask_2d = mask_k[:, None] & mask_n[None, :]
        q_shift = (offs_k[:, None] % elements_per_sample * nbit).to(tl.int32)
        
        # load x
        x_ptrs = x_ptr + offs_k * stride_xk
        x = tl.load(x_ptrs, mask_k, other=0.0)

        # ========== x @ w1 ==========
        # load w1
        w1_ptrs = w1_ptr + stride_w + (offs_k[:, None] // elements_per_sample) * stride_ik + (offs_n[None, :] % N) * stride_in
        w1 = tl.load(w1_ptrs, mask=mask_2d, other=0.0)

        # load scales and zeros for w1
        s1_ptrs = w1_scales_ptr + stride_zs + (offs_k[:, None] // group_size) * stride_qk + (offs_n[None, :] % N) * stride_qn
        z1_ptrs = w1_zeros_ptr + stride_zs + (offs_k[:, None] // group_size) * stride_qk + (offs_n[None, :] % N) * stride_qn
        s1 = tl.load(s1_ptrs, mask=mask_2d, other=0.0)
        z1 = tl.load(z1_ptrs, mask=mask_2d, other=0.0)

        # dequantize w1
        w1 = dequantize(w1, s1, z1, q_shift, unpack_mask)

        # dot product
        x1_acc += x[:, None].to(tl.float32) * w1.to(tl.float32) # NOTE: float32 for accumulation


        # ========== x @ w3 ==========
        # load w3
        w3_ptrs = w3_ptr + stride_w + (offs_k[:, None] // elements_per_sample) * stride_ik + (offs_n[None, :] % N) * stride_in
        w3 = tl.load(w3_ptrs, mask=mask_2d, other=0.0)

        # load scales and zeros for w3
        s3_ptrs = w3_scales_ptr + stride_zs + (offs_k[:, None] // group_size) * stride_qk + (offs_n[None, :] % N) * stride_qn
        z3_ptrs = w3_zeros_ptr + stride_zs + (offs_k[:, None] // group_size) * stride_qk + (offs_n[None, :] % N) * stride_qn
        s3 = tl.load(s3_ptrs, mask=mask_2d, other=0.0)
        z3 = tl.load(z3_ptrs, mask=mask_2d, other=0.0)

        # dequantize w3
        w3 = dequantize(w3, s3, z3, q_shift, unpack_mask)

        # dot product
        x3_acc += x[:, None].to(tl.float32) * w3.to(tl.float32) # NOTE: float32 for accumulation

    # accumulate over K dimension
    x1_acc = tl.sum(x1_acc, axis=0)
    x3_acc = tl.sum(x3_acc, axis=0)

    # store results
    tl.store(x1_ptr + offs_n * stride_on, x1_acc, mask=mask_n)
    tl.store(x3_ptr + offs_n * stride_on, x3_acc, mask=mask_n)


def fused_dequant_up_proj_triton(
    x: Tensor, idx: Tensor, w1: Tensor, w3: Tensor,
    w1_scales: Tensor, w1_zeros: Tensor, w3_scales: Tensor, w3_zeros: Tensor,
    nbits: Tensor, w_strides: Tensor,
    group_size: int, # NOTE: we assume same group_size for all experts
):
    """
    Compute matmul(x, dequant(w1)) and matmul(x, dequant(w3)) for a group of experts selected by idx. 

    Args:
        x:          [1, K]
        idx:        [A,] index of experts to be multiplied x with (\in [0, E-1])
        w1:         [sum(cdiv(K, elements_per_sample[e])), N] quantized, packed and stacked weights of all w1 experts
        w3:         [sum(cdiv(K, elements_per_sample[e])), N] quantized, packed and stacked weights of all w3 experts
        w1_scales:  [sum(cdiv(K, group_size[e])), N] stacked scales of all w1 experts
        w1_zeros:   [sum(cdiv(K, group_size[e])), N] stacked zeros of all w1 experts
        w3_scales:  [sum(cdiv(K, group_size[e])), N] stacked scales of all w3 experts
        w3_zeros:   [sum(cdiv(K, group_size[e])), N] stacked zeros of all w3 experts
        nbits:      [E,] number of bits for each expert
        wq_strides: [E,] stride of quantized weights for each expert
        group_size: int, assumed same for all experts
    Returns:
        x1:         [1, A*N]
        x3:         [1, A*N]
    """
    _, K = x.shape
    _, N = w1.shape
    A, = idx.shape

    # allocates output
    x1 = torch.empty((1, A * N), device=x.device, dtype=torch.float16)
    x3 = torch.empty((1, A * N), device=x.device, dtype=torch.float16)

    # 1D grid
    grid = lambda META: (triton.cdiv(A * N, META["BLOCK_SIZE_N"]),)

    fused_dequant_up_proj_kernel[grid](
        x, idx, w1, w3, x1, x3,
        w1_scales, w1_zeros, w3_scales, w3_zeros,
        nbits, w_strides,
        N, K, A,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        x1.stride(0), x1.stride(1),
        w1_scales.stride(0), w1_scales.stride(1),
        group_size,
        # NOTE: comment out for autotune
        # BLOCK_SIZE_N=64,
        # BLOCK_SIZE_K=64,
    )

    return x1, x3


@triton.autotune(
    configs=get_autotune_config(),
    key=["N", "K", "A"],
)
@triton.jit
def fused_dequant_down_proj_kernel(
    # Pointers to matrices
    x1_ptr, x3_ptr, idx_ptr, w2_ptr, x2_ptr,
    w2_scales_ptr, w2_zeros_ptr,
    nbits_ptr, stride_in_ptr,
    # Matrix dimensions
    N, K, A,
    # Strides
    stride_xm, stride_xk,  # Strides for x1 and x3 (assumed same)
    stride_ik, stride_in,
    stride_ok, stride_on,
    stride_qk, stride_qn,
    # Quantization parameters
    group_size: tl.constexpr,
    # Meta parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_n = tl.program_id(axis=0)

    # create offsets
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < A * N
    offs_k_base = tl.arange(0, BLOCK_SIZE_K)

    # NOTE: assumes BLOCK_SIZE_N divides N, i.e., no cross-expert within a block
    tl.device_assert((N % BLOCK_SIZE_N) == 0, "N must be multiple of BLOCK_SIZE_N")
    block_idx = pid_n * BLOCK_SIZE_N // N  # get a scalar block_idx
    
    # load expert params as scalars
    idx = tl.load(idx_ptr + block_idx)
    nbit = tl.load(nbits_ptr + idx)
    stride_w = tl.load(stride_in_ptr + idx)

    # pre-calculate scalar constants outside the loop
    elements_per_sample = 32 // nbit
    unpack_mask = (1 << nbit) - 1
    stride_zs = idx * (K // group_size) * stride_qk

    # split K dimension
    x2_acc = tl.full([BLOCK_SIZE_K, BLOCK_SIZE_N], 0, tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        # update k offsets
        offs_k = offs_k_base + k
        mask_k = offs_k < K
        mask_2d = mask_k[:, None] & mask_n[None, :]
        q_shift = (offs_k[:, None] % elements_per_sample * nbit).to(tl.int32)

        # load 1D x1 and x3
        x1_ptrs = x1_ptr + (block_idx * K + offs_k) * stride_xk
        x1 = tl.load(x1_ptrs, mask=mask_k, other=0.0)

        x3_ptrs = x3_ptr + (block_idx * K + offs_k) * stride_xk
        x3 = tl.load(x3_ptrs, mask=mask_k, other=0.0)

        # compute x = silu(x1) * x3
        x1 = x1.to(tl.float32)
        x1 = x1 * tl.sigmoid(x1)
        x = x1 * x3.to(tl.float32)  # x is [BLOCK_SIZE_K]

        # load w2
        w2_ptrs = w2_ptr + stride_w + (offs_k[:, None] // elements_per_sample) * stride_ik + (offs_n[None, :] % N) * stride_in
        w2 = tl.load(w2_ptrs, mask=mask_2d, other=0.0)

        # load scales and zeros for w2 
        s2_ptrs = w2_scales_ptr + stride_zs + (offs_k[:, None] // group_size) * stride_qk + (offs_n[None, :] % N) * stride_qn
        z2_ptrs = w2_zeros_ptr + stride_zs + (offs_k[:, None] // group_size) * stride_qk + (offs_n[None, :] % N) * stride_qn
        s2 = tl.load(s2_ptrs, mask=mask_2d, other=0.0)
        z2 = tl.load(z2_ptrs, mask=mask_2d, other=0.0)

        # dequantize w2
        w2 = dequantize(w2, s2, z2, q_shift, unpack_mask)

        # dot product
        x2_acc += x[:, None].to(tl.float32) * w2.to(tl.float32) # NOTE: float32 for accumulation

    # accumulate over K dimension
    x2_acc = tl.sum(x2_acc, axis=0)

    # store results
    tl.store(x2_ptr + offs_n * stride_on, x2_acc, mask=mask_n)


def fused_dequant_down_proj_triton(
    x1: Tensor, x3: Tensor, idx: Tensor, w2: Tensor,
    w2_scales: Tensor, w2_zeros: Tensor,
    nbits: Tensor, w_strides: Tensor,
    group_size: int, # NOTE: we assume same group_size for all experts
):
    """
    Compute matmul(sili(x1) * x3, dequant(w2)) for a group of experts selected by idx.

    Args:
        x1:         [1, A*K]
        x3:         [1, A*K]
        idx:        [A,] index of experts to be multiplied x with (\in [0, E-1])
        w2:         [sum(cdiv(K, elements_per_sample[e])), N] quantized, packed and stacked weights of all w2 experts
        w2_scales:  [sum(cdiv(K, group_size[e])), N] stacked scales of all w2 experts
        w2_zeros:   [sum(cdiv(K, group_size[e])), N] stacked zeros of all w2 experts
        nbits:      [E,] number of bits for each expert
        wq_strides: [E,] stride of quantized weights for each expert
        group_size: int, assumed same for all experts
    Returns:
        x2:         [1, A*N]
    """
    _, AK = x1.shape
    _, N = w2.shape
    A, = idx.shape
    K = AK // A

    # allocates output
    x2 = torch.empty((1, A * N), device=x1.device, dtype=torch.float16)

    # 1D grid
    grid = lambda META: (triton.cdiv(A * N, META["BLOCK_SIZE_N"]),)

    fused_dequant_down_proj_kernel[grid](
        x1, x3, idx, w2, x2,
        w2_scales, w2_zeros,
        nbits, w_strides,
        N, K, A,
        x1.stride(0), x1.stride(1),
        w2.stride(0), w2.stride(1),
        x2.stride(0), x2.stride(1),
        w2_scales.stride(0), w2_scales.stride(1),
        group_size,
        # NOTE: comment out for autotune
        # BLOCK_SIZE_N=64,
        # BLOCK_SIZE_K=64,
    )

    return x2
