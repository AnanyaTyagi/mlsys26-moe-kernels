"""
TVM FFI binding for FlashInfer-Bench (CUDA solution).

Strategy (baseline correctness):
- Convert inputs to torch.Tensor via DLPack
- Call flashinfer.fused_moe.trtllm_fp8_block_scale_moe (baseline op)
- Return bfloat16 output, converting back to TVM NDArray if needed
"""

from __future__ import annotations

from tvm.ffi import register_func

import torch
from torch.utils.dlpack import from_dlpack as torch_from_dlpack
from torch.utils.dlpack import to_dlpack as torch_to_dlpack

import flashinfer
import flashinfer.fused_moe  # ensure submodule is loaded


def _is_torch(x) -> bool:
    return isinstance(x, torch.Tensor)


def _to_torch(x) -> torch.Tensor:
    """Convert TVM NDArray / torch.Tensor -> torch.Tensor (zero-copy when possible)."""
    if _is_torch(x):
        return x
    if hasattr(x, "to_dlpack"):
        return torch_from_dlpack(x.to_dlpack())
    if hasattr(x, "__dlpack__"):
        return torch_from_dlpack(x.__dlpack__())
    raise TypeError(f"Unsupported input type for DLPack conversion: {type(x)}")


def _to_original_container(y_torch: torch.Tensor, like):
    """Return torch tensor or TVM NDArray depending on what `like` is."""
    if _is_torch(like):
        return y_torch
    import tvm
    return tvm.nd.from_dlpack(torch_to_dlpack(y_torch))


def _scalar_to_py(v, pytype):
    if isinstance(v, (int, float)):
        return pytype(v)
    if _is_torch(v):
        return pytype(v.item())
    if hasattr(v, "numpy"):
        return pytype(v.numpy().item())
    raise TypeError(f"Unsupported scalar type: {type(v)}")


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
    # Convert to torch
    routing_logits_t = _to_torch(routing_logits)
    routing_bias_t = _to_torch(routing_bias) if routing_bias is not None else None
    hidden_states_t = _to_torch(hidden_states)
    hidden_states_scale_t = _to_torch(hidden_states_scale)
    gemm1_weights_t = _to_torch(gemm1_weights)
    gemm1_weights_scale_t = _to_torch(gemm1_weights_scale)
    gemm2_weights_t = _to_torch(gemm2_weights)
    gemm2_weights_scale_t = _to_torch(gemm2_weights_scale)

    local_expert_offset_py = _scalar_to_py(local_expert_offset, int)
    routed_scaling_factor_py = _scalar_to_py(routed_scaling_factor, float)

    out_t = flashinfer.fused_moe.trtllm_fp8_block_scale_moe(
        routing_logits=routing_logits_t,
        routing_bias=routing_bias_t,
        hidden_states=hidden_states_t,
        hidden_states_scale=hidden_states_scale_t,
        gemm1_weights=gemm1_weights_t,
        gemm1_weights_scale=gemm1_weights_scale_t,
        gemm2_weights=gemm2_weights_t,
        gemm2_weights_scale=gemm2_weights_scale_t,
        num_experts=256,
        top_k=8,
        n_group=8,
        topk_group=4,
        intermediate_size=2048,
        local_expert_offset=local_expert_offset_py,
        local_num_experts=32,
        routed_scaling_factor=routed_scaling_factor_py,
        routing_method_type=0,
        use_shuffled_weight=False,
        weight_layout=0,
        do_finalize=True,
    )

    if out_t.dtype != torch.bfloat16:
        out_t = out_t.to(torch.bfloat16)

    return _to_original_container(out_t, routing_logits)