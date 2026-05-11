import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2MoE
from gemlite.core import GemLiteLinearTriton

from gemq.triton_kernels.dequant_group_gemm import dequant_group_gemm_triton
from gemq.triton_kernels.dequant_gemm import dequant_gemm_triton
from gemq.triton_kernels.dequant_gemv import dequant_splitk_gemv_triton
from gemq.triton_kernels.fused_dequant_bmm import fused_dequant_up_proj_triton, fused_dequant_down_proj_triton


class FusedMixtralMoEBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.w1 = nn.Parameter(torch.empty(self.num_experts, self.ffn_dim, self.hidden_dim))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, self.ffn_dim, self.hidden_dim))

    @classmethod
    def from_hf(cls, config, hf_block: MixtralSparseMoeBlock):
        device = next(hf_block.parameters()).device
        dtype = next(hf_block.parameters()).dtype

        # create fused block
        fused_block = cls(config)

        # copy gate weights
        fused_block.gate.load_state_dict(hf_block.gate.state_dict())

        # copy and stack experts weights
        fused_block.w1.data = torch.stack([expert.w1.weight.data for expert in hf_block.experts])
        fused_block.w2.data = torch.stack([expert.w2.weight.data for expert in hf_block.experts])
        fused_block.w3.data = torch.stack([expert.w3.weight.data for expert in hf_block.experts])
        
        fused_block = fused_block.to(dtype).to(device)

        return fused_block

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        x = hidden_states.view(-1, hidden_dim)
        
        scores = self.gate(x) # [T, E]
        expert_weights = F.softmax(scores, dim=-1)
        expert_weights, expert_indices = torch.topk(expert_weights, self.top_k, dim=-1) # [T, A], [T, A]
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True) # [T, A]
        
        w1_weights = self.w1[expert_indices] # [T, A, D, D]
        w3_weights = self.w3[expert_indices] # [T, A, D, D]
        w2_weights = self.w2[expert_indices] # [T, A, D, D]

        x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights))
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
        expert_outs =  torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)

        outs = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)

        final_hidden_states = outs.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, None


class QuantFusedMixtralMoEBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # quantized weights and meta args
        self.w1_wq, self.w1_scales, self.w1_zeros = None, None, None
        self.w2_wq, self.w2_scales, self.w2_zeros = None, None, None
        self.w3_wq, self.w3_scales, self.w3_zeros = None, None, None

        # meta tensors
        self.w1_nbits, self.w1_group_sizes = None, None
        self.w2_nbits, self.w2_group_sizes = None, None
        self.w3_nbits, self.w3_group_sizes = None, None
        self.block_group_size = None

        self.w1_wq_strides, self.w1_zs_strides = None, None
        self.w2_wq_strides, self.w2_zs_strides = None, None
        self.w3_wq_strides, self.w3_zs_strides = None, None

    @classmethod
    def from_hf(cls, config, hf_block: MixtralSparseMoeBlock):
        # create fused block
        fused_block = cls(config)

        # copy gate weights
        fused_block.gate.weight.data = hf_block.gate.weight.data.clone()

        # copy and stack quantized experts weights
        if not isinstance(hf_block.experts[0].w1, GemLiteLinearTriton):
            raise ValueError("Expected linears in MixtralSparseMoeBlock. Please patch and replace them first.")

        fused_block.w1_wq     = torch.cat([expert.w1.W_q.data for expert in hf_block.experts])
        fused_block.w1_scales = torch.cat([expert.w1.scales.data for expert in hf_block.experts])
        fused_block.w1_zeros  = torch.cat([expert.w1.zeros.data for expert in hf_block.experts])
        fused_block.w2_wq     = torch.cat([expert.w2.W_q.data for expert in hf_block.experts])
        fused_block.w2_scales = torch.cat([expert.w2.scales.data for expert in hf_block.experts])
        fused_block.w2_zeros  = torch.cat([expert.w2.zeros.data for expert in hf_block.experts])
        fused_block.w3_wq     = torch.cat([expert.w3.W_q.data for expert in hf_block.experts])
        fused_block.w3_scales = torch.cat([expert.w3.scales.data for expert in hf_block.experts])
        fused_block.w3_zeros  = torch.cat([expert.w3.zeros.data for expert in hf_block.experts])

        # create meta tensors
        fused_block.w1_nbits = torch.tensor([expert.w1.W_nbits for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w1_wq.device)
        fused_block.w1_group_sizes = torch.tensor([expert.w1.group_size for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w1_wq.device)
        fused_block.w2_nbits = torch.tensor([expert.w2.W_nbits for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w2_wq.device)
        fused_block.w2_group_sizes = torch.tensor([expert.w2.group_size for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w2_wq.device)
        fused_block.w3_nbits = torch.tensor([expert.w3.W_nbits for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w3_wq.device)
        fused_block.w3_group_sizes = torch.tensor([expert.w3.group_size for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w3_wq.device)

        # make sure group sizes are the same across all experts
        block_group_size = torch.unique(
            torch.cat([fused_block.w1_group_sizes, fused_block.w2_group_sizes, fused_block.w3_group_sizes])
        )
        assert len(block_group_size) == 1, "Group sizes must be the same across all experts."
        fused_block.block_group_size = block_group_size.item()

        cumsum = lambda x: x.cumsum(dim=0)[:-1] # exclusive cumsum
        fused_block.w1_wq_strides = cumsum(torch.tensor([0] + [expert.w1.W_q.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w1_wq.device))
        fused_block.w1_zs_strides = cumsum(torch.tensor([0] + [expert.w1.scales.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w1_wq.device))
        fused_block.w2_wq_strides = cumsum(torch.tensor([0] + [expert.w2.W_q.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w2_wq.device))
        fused_block.w2_zs_strides = cumsum(torch.tensor([0] + [expert.w2.scales.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w2_wq.device))
        fused_block.w3_wq_strides = cumsum(torch.tensor([0] + [expert.w3.W_q.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w3_wq.device))
        fused_block.w3_zs_strides = cumsum(torch.tensor([0] + [expert.w3.scales.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w3_wq.device))

        return fused_block

    def forward_single_expert(self, expert_idx: Tensor, hidden_states) -> Tensor:
        x  = hidden_states # [T, D]

        # compute topk expert outputs
        x1 = dequant_group_gemm_triton(
            x, expert_idx, self.w1_wq, self.w1_scales, self.w1_zeros,
            self.w1_nbits, self.w1_group_sizes, self.w1_wq_strides, self.w1_zs_strides
        )[0] # [T, D]
        x3 = dequant_group_gemm_triton(
            x, expert_idx, self.w3_wq, self.w3_scales, self.w3_zeros,
            self.w3_nbits, self.w3_group_sizes, self.w3_wq_strides, self.w3_zs_strides
        )[0] # [T, D]
        x2 = dequant_group_gemm_triton(
            F.silu(x1) * x3, expert_idx, self.w2_wq, self.w2_scales, self.w2_zeros,
            self.w2_nbits, self.w2_group_sizes, self.w2_wq_strides, self.w2_zs_strides
        )[0] # [T, D]

        return x2 # [T, D]

    def forward_n_tokens(self, hidden_states: Tensor) -> Tensor:
        """
        Fallback to the original implementation for multiple tokens input.
        This function is NOT compatible with torch.compile due to the dynamic inputs in the loop.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
    
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # one hot encode the selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # [E, A, T]

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = self.forward_single_expert(expert_idx, current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits 

    def forward_one_token(self, hidden_states: Tensor) -> Tensor:
        """
        Forward a single token input through the MoE block.
        This function is compatible with torch.compile for accelerated inference.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        x = hidden_states.view(-1, self.hidden_dim)
        assert x.shape[0] == 1, f"This function only supports single token forward. But got {x.shape[0]} tokens."

        # route
        scores = self.gate(x) # [1, E]
        expert_weights = F.softmax(scores, dim=-1)
        expert_weights, expert_indices = torch.topk(expert_weights, self.top_k, dim=-1) # [1, A], [1, A]
        expert_weights /= expert_weights.sum(dim=-1, keepdim=True) # [1, A]

        # gate and up project
        x1, x3 = fused_dequant_up_proj_triton(
            x, expert_indices[0], self.w1_wq, self.w3_wq,
            self.w1_scales, self.w1_zeros, self.w3_scales, self.w3_zeros,
            self.w1_nbits, self.w1_wq_strides, group_size=self.block_group_size
        ) # [1, A*D]

        # down project
        x2 = fused_dequant_down_proj_triton(
            x1, x3, expert_indices[0], self.w2_wq,
            self.w2_scales, self.w2_zeros, self.w2_nbits, self.w2_wq_strides, group_size=self.block_group_size
        ) # [1, A*D]

        # combine topk expert outputs
        expert_outs = x2.view(1, self.top_k, hidden_dim) # [1, A, D]
        outs = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)

        # the original implementation also returns router logits, but we don't need them for inference
        final_hidden_states = outs.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, None

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.shape[0] == hidden_states.shape[1] == 1:
            return self.forward_one_token(hidden_states)
        return self.forward_n_tokens(hidden_states)


class FusedDeepseekV2MoEBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # grouped routing experts
        self.w1 = nn.Parameter(torch.empty(self.num_experts, self.ffn_dim, self.hidden_dim))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.ffn_dim))
        self.w3 = nn.Parameter(torch.empty(self.num_experts, self.ffn_dim, self.hidden_dim))

        # shared experts
        self.shared_w1 = nn.Parameter(torch.empty(self.num_shared_experts * self.ffn_dim, self.hidden_dim))
        self.shared_w2 = nn.Parameter(torch.empty(self.hidden_dim, self.num_shared_experts * self.ffn_dim))
        self.shared_w3 = nn.Parameter(torch.empty(self.num_shared_experts * self.ffn_dim, self.hidden_dim))

    @classmethod
    def from_hf(cls, config, hf_block: DeepseekV2MoE):
        device = next(hf_block.parameters()).device
        dtype = next(hf_block.parameters()).dtype

        # create fused block
        fused_block = cls(config)

        # copy gate weights
        fused_block.gate.load_state_dict(hf_block.gate.state_dict())

        # copy and stack experts weights
        fused_block.w1.data = torch.stack([expert.gate_proj.weight.data for expert in hf_block.experts])
        fused_block.w2.data = torch.stack([expert.down_proj.weight.data for expert in hf_block.experts])
        fused_block.w3.data = torch.stack([expert.up_proj.weight.data for expert in hf_block.experts])
        fused_block.shared_w1.data = hf_block.shared_experts.gate_proj.weight.data.clone()
        fused_block.shared_w2.data = hf_block.shared_experts.down_proj.weight.data.clone()
        fused_block.shared_w3.data = hf_block.shared_experts.up_proj.weight.data.clone()

        fused_block = fused_block.to(dtype).to(device)

        return fused_block

    def forward(self, hidden_states: Tensor) -> Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        x = hidden_states.view(-1, hidden_dim)
        
        scores = self.gate(x) # [T, E]
        expert_weights = F.softmax(scores, dim=-1)
        expert_weights, expert_indices = torch.topk(expert_weights, self.top_k, dim=-1) # [T, A], [T, A]
        if self.norm_topk_prob:
            expert_weights /= expert_weights.sum(dim=-1, keepdim=True) # [T, A]
        
        # compute expert outputs
        w1_weights = self.w1[expert_indices] # [T, A, D, D]
        w3_weights = self.w3[expert_indices] # [T, A, D, D]
        w2_weights = self.w2[expert_indices] # [T, A, D, D]
        x1 = F.silu(torch.einsum("ti,taoi -> tao", x, w1_weights))
        x3 = torch.einsum("ti, taoi -> tao", x, w3_weights)
        expert_outs = torch.einsum("tao, taio -> tai", (x1 * x3), w2_weights)

        # weight expert outputs
        outs = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)

        # add shared experts
        shared_x1 = F.silu(F.linear(x, self.shared_w1))
        shared_x3 = F.linear(x, self.shared_w3)
        shared_expert_outs = F.linear(shared_x1 * shared_x3, self.shared_w2)
        outs += shared_expert_outs

        final_hidden_states = outs.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class QuantFusedDeepseekV2MoEBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        # quantized weights and meta args
        self.w1_wq, self.w1_scales, self.w1_zeros = None, None, None
        self.w2_wq, self.w2_scales, self.w2_zeros = None, None, None
        self.w3_wq, self.w3_scales, self.w3_zeros = None, None, None
        self.shared_w1_wq, self.shared_w1_scales, self.shared_w1_zeros = None, None, None
        self.shared_w2_wq, self.shared_w2_scales, self.shared_w2_zeros = None, None, None
        self.shared_w3_wq, self.shared_w3_scales, self.shared_w3_zeros = None, None, None

        # meta tensors
        self.w1_nbits, self.w1_group_sizes = None, None
        self.w2_nbits, self.w2_group_sizes = None, None
        self.w3_nbits, self.w3_group_sizes = None, None
        self.block_group_size = None

        self.w1_wq_strides, self.w1_zs_strides = None, None
        self.w2_wq_strides, self.w2_zs_strides = None, None
        self.w3_wq_strides, self.w3_zs_strides = None, None

        self.shared_nbits, self.shared_group_size = None, None

    @classmethod
    def from_hf(cls, config, hf_block: DeepseekV2MoE):
        # create fused block
        fused_block = cls(config)

        # copy gate weights
        fused_block.gate.weight.data = hf_block.gate.weight.data.clone()

        # copy and stack quantized experts weights
        # NOTE: a simple check here to ensure all nn.linear modules are already replaced
        if not isinstance(hf_block.experts[0].gate_proj, GemLiteLinearTriton):
            raise ValueError("Expected linears in DeepseekV2MoE. Please patch and replace them first.")

        # routed experts
        fused_block.w1_wq     = torch.cat([expert.gate_proj.W_q.data for expert in hf_block.experts])
        fused_block.w1_scales = torch.cat([expert.gate_proj.scales.data for expert in hf_block.experts])
        fused_block.w1_zeros  = torch.cat([expert.gate_proj.zeros.data for expert in hf_block.experts])
        fused_block.w2_wq     = torch.cat([expert.down_proj.W_q.data for expert in hf_block.experts])
        fused_block.w2_scales = torch.cat([expert.down_proj.scales.data for expert in hf_block.experts])
        fused_block.w2_zeros  = torch.cat([expert.down_proj.zeros.data for expert in hf_block.experts])
        fused_block.w3_wq     = torch.cat([expert.up_proj.W_q.data for expert in hf_block.experts])
        fused_block.w3_scales = torch.cat([expert.up_proj.scales.data for expert in hf_block.experts])
        fused_block.w3_zeros  = torch.cat([expert.up_proj.zeros.data for expert in hf_block.experts])
        
        # create meta tensors
        fused_block.w1_nbits = torch.tensor([expert.gate_proj.W_nbits for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w1_wq.device)
        fused_block.w1_group_sizes = torch.tensor([expert.gate_proj.group_size for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w1_wq.device)
        fused_block.w2_nbits = torch.tensor([expert.down_proj.W_nbits for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w2_wq.device)
        fused_block.w2_group_sizes = torch.tensor([expert.down_proj.group_size for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w2_wq.device)
        fused_block.w3_nbits = torch.tensor([expert.up_proj.W_nbits for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w3_wq.device)
        fused_block.w3_group_sizes = torch.tensor([expert.up_proj.group_size for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w3_wq.device)

        # make sure group sizes are the same across all experts
        block_group_size = torch.unique(torch.cat([fused_block.w1_group_sizes, fused_block.w2_group_sizes, fused_block.w3_group_sizes]))
        assert len(block_group_size) == 1, "Group sizes must be the same across all experts."
        fused_block.block_group_size = block_group_size.item()

        cumsum = lambda x: x.cumsum(dim=0)[:-1] # exclusive cumsum
        fused_block.w1_wq_strides = cumsum(torch.tensor([0] + [expert.gate_proj.W_q.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w1_wq.device))
        fused_block.w1_zs_strides = cumsum(torch.tensor([0] + [expert.gate_proj.scales.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w1_wq.device))
        fused_block.w2_wq_strides = cumsum(torch.tensor([0] + [expert.down_proj.W_q.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w2_wq.device))
        fused_block.w2_zs_strides = cumsum(torch.tensor([0] + [expert.down_proj.scales.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w2_wq.device))
        fused_block.w3_wq_strides = cumsum(torch.tensor([0] + [expert.up_proj.W_q.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w3_wq.device))
        fused_block.w3_zs_strides = cumsum(torch.tensor([0] + [expert.up_proj.scales.numel() for expert in hf_block.experts], dtype=torch.int32, device=fused_block.w3_wq.device))

        # copy shared experts
        fused_block.shared_w1_wq     = hf_block.shared_experts.gate_proj.W_q.data.clone()
        fused_block.shared_w1_scales = hf_block.shared_experts.gate_proj.scales.data.clone()
        fused_block.shared_w1_zeros  = hf_block.shared_experts.gate_proj.zeros.data.clone()
        fused_block.shared_w2_wq     = hf_block.shared_experts.down_proj.W_q.data.clone()
        fused_block.shared_w2_scales = hf_block.shared_experts.down_proj.scales.data.clone()
        fused_block.shared_w2_zeros  = hf_block.shared_experts.down_proj.zeros.data.clone()
        fused_block.shared_w3_wq     = hf_block.shared_experts.up_proj.W_q.data.clone()
        fused_block.shared_w3_scales = hf_block.shared_experts.up_proj.scales.data.clone()
        fused_block.shared_w3_zeros  = hf_block.shared_experts.up_proj.zeros.data.clone()

        # NOTE: assume w1, w2, w3 have the same nbits and group_size for shared experts
        fused_block.shared_nbits = hf_block.shared_experts.gate_proj.W_nbits
        fused_block.shared_group_size = hf_block.shared_experts.gate_proj.group_size

        return fused_block

    def forward_single_expert(self, expert_idx: Tensor, hidden_states, is_shared=False) -> Tensor:
        x  = hidden_states # [T, D]

        if is_shared:
            x1 = dequant_gemm_triton(
                x, self.shared_w1_wq, self.shared_w1_scales, self.shared_w1_zeros, self.shared_nbits, self.shared_group_size
            ) # [T, D]
            x3 = dequant_gemm_triton(
                x, self.shared_w3_wq, self.shared_w3_scales, self.shared_w3_zeros, self.shared_nbits, self.shared_group_size
            ) # [T, D]
            x2 = dequant_gemm_triton(
                F.silu(x1) * x3, self.shared_w2_wq, self.shared_w2_scales, self.shared_w2_zeros, self.shared_nbits, self.shared_group_size
            ) # [T, D]

            return x2

        # compute topk expert outputs
        x1 = dequant_group_gemm_triton(
            x, expert_idx, self.w1_wq, self.w1_scales, self.w1_zeros,
            self.w1_nbits, self.w1_group_sizes, self.w1_wq_strides, self.w1_zs_strides
        )[0] # [T, D]
        x3 = dequant_group_gemm_triton(
            x, expert_idx, self.w3_wq, self.w3_scales, self.w3_zeros,
            self.w3_nbits, self.w3_group_sizes, self.w3_wq_strides, self.w3_zs_strides
        )[0] # [T, D]
        x2 = dequant_group_gemm_triton(
            F.silu(x1) * x3, expert_idx, self.w2_wq, self.w2_scales, self.w2_zeros,
            self.w2_nbits, self.w2_group_sizes, self.w2_wq_strides, self.w2_zs_strides
        )[0] # [T, D]

        return x2 # [T, D]

    def forward_n_tokens(self, hidden_states: Tensor) -> Tensor:
        """
        Forward several tokens simultaneously through the MoE block.
        Fallback to the original implementation for multiple tokens input.
        This function is NOT compatible with torch.compile due to the dynamic inputs in the loop.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
    
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = F.linear(hidden_states.float(), self.gate.weight.float(), bias=None)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # one hot encode the selected experts to create an expert mask
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        # [E, A, T]

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            # expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = self.forward_single_expert(expert_idx, current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        # shared experts
        final_hidden_states += self.forward_single_expert(None, hidden_states, is_shared=True)
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states

    def forward_one_token(self, hidden_states: Tensor) -> Tensor:
        """
        Forward a single token input through the MoE block.
        This function is compatible with torch.compile for accelerated inference.
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        x = hidden_states.view(-1, self.hidden_dim) # [1, D]
        assert x.shape[0] == 1, f"Only supports single token forward. Got {x.shape[0]} tokens."

        # route
        router_logits = F.linear(x.float(), self.gate.weight.float(), bias=None) # [1, E]
        expert_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        expert_weights, expert_indices = torch.topk(expert_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
        # cast back to the input dtype
        expert_weights = expert_weights.to(hidden_states.dtype)
        
        # gate and up project
        x1, x3 = fused_dequant_up_proj_triton(
            x, expert_indices[0], self.w1_wq, self.w3_wq,
            self.w1_scales, self.w1_zeros, self.w3_scales, self.w3_zeros,
            self.w1_nbits, self.w1_wq_strides, group_size=self.block_group_size
        ) # [1, A*D]
        
        # down project
        x2 = fused_dequant_down_proj_triton(
            x1, x3, expert_indices[0], self.w2_wq,
            self.w2_scales, self.w2_zeros, self.w2_nbits, self.w2_wq_strides, group_size=self.block_group_size
        ) # [1, A*D]

        # combine topk expert outputs
        expert_outs = x2.view(1, self.top_k, hidden_dim) # [1, A, D]
        outs = torch.einsum("tai,ta -> ti", expert_outs, expert_weights)

        # add shared experts
        x1 = dequant_splitk_gemv_triton(x, self.shared_w1_wq, self.shared_w1_scales, self.shared_w1_zeros, self.shared_nbits, self.shared_group_size) # [1, D]
        x3 = dequant_splitk_gemv_triton(x, self.shared_w3_wq, self.shared_w3_scales, self.shared_w3_zeros, self.shared_nbits, self.shared_group_size) # [1, D]
        x2 = dequant_splitk_gemv_triton(F.silu(x1) * x3, self.shared_w2_wq, self.shared_w2_scales, self.shared_w2_zeros, self.shared_nbits, self.shared_group_size) # [1, D]
        outs += x2 # [1, D]
    
        final_hidden_states = outs.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.shape[0] == hidden_states.shape[1] == 1:
            return self.forward_one_token(hidden_states)
        return self.forward_n_tokens(hidden_states)
