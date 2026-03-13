"""
TVM FFI Bindings Template for CUDA Kernels (Starter-kit compatible).

Entry point: binding.py::kernel

This kernel definition expects:
- routing_logits: float32[seq_len, 256]
- routing_bias: bfloat16[256]
- hidden_states: float8_e4m3fn[seq_len, 7168]
- hidden_states_scale: float32[56, seq_len]
- gemm1_weights: float8_e4m3fn[32, 4096, 7168]
- gemm1_weights_scale: float32[32, 32, 56]
- gemm2_weights: float8_e4m3fn[32, 7168, 2048]
- gemm2_weights_scale: float32[32, 56, 16]
- local_expert_offset: int32 scalar
- routed_scaling_factor: float32 scalar

Output:
- output: bfloat16[seq_len, 7168]
"""

from tvm.ffi import register_func

# NOTE: In most starter-kit setups, tensors passed to this function are TVM NDArrays.
# You can operate on them via TVM APIs (or convert/interop), but for a "today" runnable
# submission we provide a minimal output allocation path.

import tvm
from tvm import nd


@register_func("flashinfer.kernel")
def kernel(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
):
    """
    Runnable placeholder implementation.

    Today goal: be executable + match output contract.
    Tomorrow goal: replace the body with actual CUDA kernel launch / library call.

    Returns:
      output: tvm.nd.NDArray, dtype=bfloat16, shape=(seq_len, 7168)
    """

    # Infer seq_len from routing_logits: [seq_len, 256]
    # (Could also read from hidden_states: [seq_len, 7168])
    seq_len = int(routing_logits.shape[0])

    # Allocate output on same device as inputs (important)
    dev = routing_logits.device

    # Create bfloat16 output
    out = nd.empty((seq_len, 7168), dtype="bfloat16", device=dev)

    # --- Minimal safe fallback ---
    # For now we output zeros. This will be "correct type/shape" but NOT numerically correct.
    # Depending on evaluation rules, this may fail correctness checks.
    # If you want a better fallback, see the "better fallback" note below.
    out.copyfrom(nd.array(tvm.runtime.ndarray.zeros((seq_len, 7168), "bfloat16"), device=dev))

    return out