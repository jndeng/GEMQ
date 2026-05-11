import torch
from torch import Tensor

import triton
import triton.language as tl

from gemq.triton_kernels.utils import dequantize



def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
    ]

def get_autotune_config():
    return get_cuda_autotune_config()


def get_fast_autotune_config_nvidia():
    configs = []
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':32,}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':64,}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':128,}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':32,  'BLOCK_SIZE_K':256,}, num_warps=4, num_stages=5))

    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':32,}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':64,}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':128,}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':64,  'BLOCK_SIZE_K':256,}, num_warps=8, num_stages=4))

    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':32,}, num_warps=8, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':64,}, num_warps=4, num_stages=5))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':128,}, num_warps=4, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':128, 'BLOCK_SIZE_K':256,}, num_warps=4, num_stages=4))
    
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':64,}, num_warps=8, num_stages=4))
    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':256, 'BLOCK_SIZE_K':128,}, num_warps=8, num_stages=4))

    configs.append(triton.Config({'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':512, 'BLOCK_SIZE_K':128,}, num_warps=8, num_stages=3))
    return configs


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)
@triton.jit
def dequant_gemm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    scales_ptr, zeros_ptr,
    # Matrix dimensions
    M, N, K,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    GEMM for C = matmul(A, dequantize(B, scales, zeros))

    A is of shape (M, K): float16
    B is of shape (K//elements_per_sample, N): int32 as a packed matrix
    C is of shape (M, N): float16

    scales and zeros is of shape (num_groups, N): float16
    NOTE: dequant is computed as: B * scales + zeros

    NOTE: fix compute dtype to float16 for simplicity
    """
    pid = tl.program_id(axis=0)
    
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M) # number of program ids along the M axis
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N) # number of program ids along the N axis
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K) # number of program ids along the K axis

    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # create offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k_init = tl.arange(0, BLOCK_SIZE_K)

    # create pointers for the first blocks of A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k_init[None, :] * stride_ak)

    elements_per_sample = 32 // nbits

    # accumulate over K dimension
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        k_start = k * BLOCK_SIZE_K
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < K
        
        # load the current block of A and B, generate a mask by checking the K dimension
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)

        b_ptrs = b_ptr + (offs_k[:, None] // elements_per_sample) * stride_bk + offs_bn[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

        # load the current block of scales and zeros
        s_ptrs = scales_ptr + (offs_k[:, None] // group_size * stride_qk + offs_bn[None, :] * stride_qn)
        z_ptrs = zeros_ptr  + (offs_k[:, None] // group_size * stride_qk + offs_bn[None, :] * stride_qn)
        s = tl.load(s_ptrs, mask=offs_k[:, None] < K, other=0.0)
        z = tl.load(z_ptrs, mask=offs_k[:, None] < K, other=0.0)

        # unpack and dequantize
        q_shift = (offs_k % elements_per_sample * nbits).to(tl.int32)[:, None] 
        unpack_mask = (1 << nbits) - 1
        b = dequantize(b, s, z, q_shift, unpack_mask)

        # matmul
        accumulator = tl.dot(a, b, acc=accumulator)

        # advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak

    
    c = accumulator.to(tl.float16)
    
    # write back the block of the output matrix C with masks
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def dequant_gemm_triton(
    x: Tensor, w_q: Tensor, scales: Tensor, zeros: Tensor,
    nbits: int, group_size: int
) -> Tensor:
    """
    Matmul with quantized weights: output = matmul(x, dequantize(w_q, scales, zeros))

    Args:
        x:      [M, K]
        w_q:    [K // elements_per_sample, N]
        scales: [num_groups, N]
        zeros:  [num_groups, N]
    Returns:
        output: [M, N]

    NOTE: this function can handle BLOCK_SIZE_K > group_size
    """
    compute_dtype = torch.float16

    M, K = x.shape
    _, N = w_q.shape

    # allocates output
    output = torch.empty((M, N), device=x.device, dtype=compute_dtype)

    # 1D grid
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    dequant_gemm_kernel[grid](
        x, w_q, output,
        scales, zeros,
        M, N, K,
        x.stride(0), x.stride(1),
        w_q.stride(0), w_q.stride(1),
        output.stride(0), output.stride(1),
        scales.stride(0), scales.stride(1),
        nbits, group_size,
        # NOTE: comment out for autotune
        # BLOCK_SIZE_M=64,
        # BLOCK_SIZE_N=64,
        # BLOCK_SIZE_K=32,
    )

    return output
