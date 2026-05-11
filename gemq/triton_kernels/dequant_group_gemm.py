import torch
from torch import Tensor

import triton
import triton.language as tl

from gemq.triton_kernels.utils import dequantize


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, "NUM_SM": 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, "NUM_SM": 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, "NUM_SM": 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, "NUM_SM": 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, "NUM_SM": 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, "NUM_SM": 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, "NUM_SM": 128}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, "NUM_SM": 128}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 16,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, "NUM_SM": 128}, num_stages=2, num_warps=4),
    ]


def get_cuda_autotune_config_1():
    return [
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "NUM_SM": 84}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "NUM_SM": 128}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32, "NUM_SM": 84}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 32, "NUM_SM": 128}),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "NUM_SM": torch.cuda.get_device_properties("cuda").multi_processor_count}),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "NUM_SM": torch.cuda.get_device_properties("cuda").multi_processor_count}),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K", "A", "E"],
)
@triton.jit
def dequant_group_gemm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, indices_ptr,
    scales_ptr, zeros_ptr,
    nbits_ptr, group_sizes_ptr,
    b_stride_ptr, zs_stride_ptr,
    # Matrix dimensions
    M, N, K, A, E,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_c, stride_cm, stride_cn,
    stride_qk, stride_qn,
    # Meta parameters
    NUM_SM: tl.constexpr, # number of virtual SM
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    tile_idx = tl.program_id(0) # tile index
    num_m_tiles = tl.cdiv(M, BLOCK_SIZE_M) # number of tiles (programs) along the M axis
    num_n_tiles = tl.cdiv(N, BLOCK_SIZE_N) # number of tiles (programs) along the N axis
    num_k_tiles = tl.cdiv(K, BLOCK_SIZE_K) # number of tiles (programs) along the K axis
    num_tiles = num_m_tiles * num_n_tiles

    # traverse each group (expert)
    last_problem_end = 0
    for g in range(A):
        # retrieve group (expert) specific parameters
        idx = tl.load(indices_ptr + g)
        nbit = tl.load(nbits_ptr + idx)
        group_size = tl.load(group_sizes_ptr + idx)
        stride_bb = tl.load(b_stride_ptr + idx)
        stride_zs = tl.load(zs_stride_ptr + idx)

        aa_ptr = a_ptr
        bb_ptr = b_ptr + stride_bb
        cc_ptr = c_ptr + g * stride_c
        ss_ptr = scales_ptr + stride_zs
        zz_ptr = zeros_ptr + stride_zs
        
        # iterate through the tiles (in the output matrix) in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem

            # figure out tile coordinates (tile index in the current gemm problem)
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular quant gemm here
            offs_am = (tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
            offs_bn = (tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
            offs_k = tl.arange(0, BLOCK_SIZE_K)

            offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
            offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            

            # create pointers for the first blocks of A and B
            # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
            offs_k_init = tl.arange(0, BLOCK_SIZE_K)
            aa_ptrs = aa_ptr + (offs_am[:, None] * stride_am + offs_k_init[None, :] * stride_ak)

            elements_per_sample = 32 // nbit

            # accumulate over K dimension
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for k in range(0, num_k_tiles):
                k_start = k * BLOCK_SIZE_K
                offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
                mask_k = offs_k < K

                # load the current block of A and B, generate a mask by checking the K dimension
                a = tl.load(aa_ptrs, mask=mask_k[None, :], other=0.0)

                bb_ptrs = bb_ptr + (offs_k[:, None] // elements_per_sample * stride_bk + offs_bn[None, :] * stride_bn)
                b = tl.load(bb_ptrs, mask=mask_k[:, None], other=0.0)

                # load the current block of scales and zeros
                ss_ptrs = ss_ptr + (offs_k[:, None] // group_size * stride_qk + offs_bn[None, :] * stride_qn)
                zz_ptrs = zz_ptr  + (offs_k[:, None] // group_size * stride_qk + offs_bn[None, :] * stride_qn)
                s = tl.load(ss_ptrs, mask=mask_k[:, None], other=0.0)
                z = tl.load(zz_ptrs, mask=mask_k[:, None], other=0.0)

                # unpack and dequantize
                q_shift = (offs_k % elements_per_sample * nbit).to(tl.int32)[:, None] 
                unpack_mask = (1 << nbit) - 1
                b = dequantize(b, s, z, q_shift, unpack_mask)

                # matmul
                accumulator = tl.dot(a, b, acc=accumulator)

                # advance the ptrs to the next K block
                aa_ptrs += BLOCK_SIZE_K * stride_ak

            c = accumulator.to(tl.float16)

            # write back the block of the output matrix C with masks
            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_cn = tl.max_contiguous(tl.multiple_of(offs_cn, BLOCK_SIZE_N), BLOCK_SIZE_N)
            cc_ptrs = cc_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
            cc_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
            tl.store(cc_ptrs, c, mask=cc_mask)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM
        
        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def dequant_group_gemm_triton(
    x: Tensor, indices: Tensor, wq: Tensor, scales: Tensor, zeros: Tensor,
    nbits: Tensor, group_sizes: Tensor, wq_strides: Tensor, zs_strides: Tensor,
    compute_dtype=torch.float16
) -> Tensor:
    """
    Matmul with quantized weights: output[e] = matmul(x, dequantize(w_q[e], scales[e], zeros[e]))

    Args:
        x:          [M, K]
        indices:    [A,] indices of experts to be multiplied x with (\in [0, E-1])
        wq:         [sum(K // elements_per_sample[e]), N] stacked quantized weights of all experts
        scales:     [sum(K // group_size[e]), N] stacked scales of all experts
        zeros:      [sum(K // group_size[e]), N] stacked zeros of all experts
        nbits:      [E,] number of bits for each expert
        group_size: [E,] group sizes for each expert
        w_q_sizes:  [E,] size of quantized weights for each expert
    Returns:
        output:     [A, M, N]
    """
    M, K = x.shape
    _, N = wq.shape
    A, = indices.shape
    E, = nbits.shape

    # allocates output
    output = torch.empty((A, M, N), device=x.device, dtype=compute_dtype)

    # 1D grid
    grid = lambda META: (META["NUM_SM"], )

    dequant_group_gemm_kernel[grid](
        x, wq, output, indices,
        scales, zeros,
        nbits, group_sizes, wq_strides, zs_strides,
        M, N, K, A, E,
        x.stride(0), x.stride(1),
        wq.stride(0), wq.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        scales.stride(0), scales.stride(1),
        # NOTE: comment out for autotune
        # NUM_SM=torch.cuda.get_device_properties("cuda").multi_processor_count,
        # BLOCK_SIZE_M=64,
        # BLOCK_SIZE_N=64,
        # BLOCK_SIZE_K=32,
    )

    return output
