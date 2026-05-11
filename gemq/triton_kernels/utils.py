import torch
import triton
import triton.language as tl


_powers_of_2 = [2**n for n in range(10)][::-1]
def highest_divisor(n: int, max_val: int) -> int:
    if(max_val == 1): 
        return 1
    
    for d in _powers_of_2:
        if n % d == 0 and d <= max_val:
            return d


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def supports_ws():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


@triton.jit
def or_fn(a, b):
    return a | b


@triton.jit
def pack_weights_over_cols_kernel(
    W_q_ptr,
    W_q_out_ptr,
    num_input_cols,
    num_cols,
    unroll: tl.constexpr, # BLOCK_SIZE
    elements_per_sample: tl.constexpr,
    W_nbits: tl.constexpr,
    out_dtype: tl.constexpr,
):
    """
    support non-divisible case (e.g., 3-bit)
    """
    pid     = tl.program_id(0)
    pid_row = (pid // num_cols) * unroll
    pid_col = (pid % num_cols)

    for r in range(unroll):
        start_col = pid_col * elements_per_sample
        # NOTE: arange's range must be a power of 2; we will mask out the extra elements later
        cols = tl.arange(0, triton.next_power_of_2(elements_per_sample))
        mask = (start_col + cols < num_input_cols) & (cols < elements_per_sample)

        base_shifts = (cols * W_nbits).to(out_dtype)
        shifts = tl.where(mask, base_shifts, 0)

        # load  
        offset = pid_row * num_input_cols + start_col + cols
        # offset = tl.max_contiguous(tl.multiple_of(offset, elements_per_sample), elements_per_sample)
        values = tl.load(W_q_ptr + offset, mask=mask, other=0).to(out_dtype) # pad zeros

        # pack
        result = tl.reduce(values << shifts, axis=0, combine_fn=or_fn)

        # store
        output_offset = pid_row * num_cols + pid_col
        tl.store(W_q_out_ptr + output_offset, result)
        pid_row += 1


def pack_weights_over_cols_triton(W_q, W_nbits, packing_bitwidth=32, transpose=True) -> tuple[torch.Tensor, int]:
    """
    Pack quantized weights along columns (dim=1).
    
    support non-divisible case (e.g., 3-bit)

    W_q: Quantized weight tensor in uint8 with shape [out_features, in_features]
    """
    assert packing_bitwidth == 32, "Unsuported bitpacking width"
    assert W_nbits in [4, 3, 2, 1], f"Untested quantization bitwidth: {W_nbits}"

    elements_per_sample = packing_bitwidth // W_nbits
    num_rows, num_input_cols = W_q.shape
    # num_cols = num_input_cols // elements_per_sample
    num_cols = triton.cdiv(num_input_cols, elements_per_sample) # for non-divisible case (e.g., 3-bit)

    dtype = torch.int32
    out_dtype = tl.int32
    W_q_out = torch.empty((num_rows, num_cols), dtype=dtype, device=W_q.device)

    unroll  = highest_divisor(num_rows, max_val=64) 
    grid = (triton.cdiv(num_rows * num_cols, unroll), )

    pack_weights_over_cols_kernel[grid](
        W_q.contiguous(),
        W_q_out,
        num_input_cols,
        num_cols,
        unroll,
        elements_per_sample,
        W_nbits,
        out_dtype,
        num_stages=2,
        num_warps=1,
    )
    
    if transpose:
        W_q_out = W_q_out.t()

    return W_q_out, elements_per_sample


@triton.jit
def unpack_over_cols_kernel(
    W_q_packed_ptr,
    W_q_unpacked_ptr,
    num_rows,
    num_cols,
    num_output_cols,
    elements_per_sample: tl.constexpr,
    W_nbits: tl.constexpr,
    unroll: tl.constexpr,
    output_dtype: tl.constexpr,
):
    pid           = tl.program_id(0)
    num_blocks    = tl.cdiv(num_output_cols, unroll)
    pid_row       = pid // num_blocks
    pid_col_block = pid % num_blocks
    
    # load
    cols          = pid_col_block * unroll + tl.arange(0, unroll)
    packed_cols   = cols // elements_per_sample
    offset        = pid_row * num_cols + packed_cols
    packed_values = tl.load(W_q_packed_ptr + offset)

    # unpack
    shifts   = (cols % elements_per_sample) * W_nbits
    mask_val = (1 << W_nbits) - 1
    unpacked_values = ((packed_values >> shifts) & mask_val).to(output_dtype)

    # store the unpacked values
    unpacked_offsets = pid_row * num_output_cols + cols
    tl.store(W_q_unpacked_ptr + unpacked_offsets, unpacked_values)


def unpack_over_cols_triton(
    W_q_packed: torch.Tensor,
    W_nbits: int,
    packing_bitwidth: int,
    num_output_cols: int,
    dtype: torch.dtype = torch.uint8,
) -> torch.Tensor:
    """
    Unpack quantized weights along columns (dim=1).

    support non-divisible case (e.g., 3-bit)
    """

    # get input dimensions
    num_rows, num_cols  = W_q_packed.shape
    # elements_per_sample = num_output_cols // num_cols
    elements_per_sample = packing_bitwidth // W_nbits  # for non-divisible case (e.g., 3-bit)

    # allocate output tensor
    W_q_unpacked = torch.empty((num_rows, num_output_cols), dtype=dtype, device=W_q_packed.device)
    output_dtype = tl.int32

    unroll = highest_divisor(num_cols, max_val=256) 
    grid = (num_rows * triton.cdiv(num_output_cols, unroll),)

    # launch the kernel
    unpack_over_cols_kernel[grid](
        W_q_packed.contiguous(),
        W_q_unpacked,
        num_rows,
        num_cols,
        num_output_cols,
        elements_per_sample,
        W_nbits,
        unroll,
        output_dtype,
        num_stages=2,
        num_warps=1
    )

    return W_q_unpacked


@triton.jit
def dequantize(b, scales, zeros, q_shift, unpack_mask: tl.constexpr):
    """
    Dequantize packed quantized matrix B.
    NOTE: dequant is computed as: B * scales + zeros

    Args:
        b:       [BLOCK_SIZE_K, BLOCK_SIZE_N] int32 packed quantized matrix
        scales:  [BLOCK_SIZE_K, BLOCK_SIZE_N] float16 scales
        zeros:   [BLOCK_SIZE_K, BLOCK_SIZE_N] float16 zeros
        q_shift: [BLOCK_SIZE_K, 1] int32 shift amount for unpacking
    Returns:
        dequantized B: [BLOCK_SIZE_K, BLOCK_SIZE_N] float16
    """
    b = (b >> q_shift) & unpack_mask # int32 -> int32

    # dequantize
    # NOTE: fixed to float16
    b = tl.fma(b.to(tl.float16), scales, zeros) # b*scales + zeros

    return b
