""" This module contains the implementation of the Mixture of Experts (MoE) layer. """

import torch
import torch.nn as nn
from transformers import BertConfig, ViTConfig

from d2dmoe.common import ACTIVATION_NAME_MAP
from d2dmoe.utils import inverse_argsort


class Router(nn.Sequential):
    """Base class for the router network that takes the hidden states of the model and outputs the routing scores for the experts."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        width: int,
        activation: str,
        depth: int = 1,
        bias: bool = False,
        output_activation: str = "identity",
    ):
        """
        Args:
            hidden_size: The size of the hidden states of the model.
            num_experts: The number of experts.
            width: The width of the router network.
            activation: The activation function to use in the router network.
            depth: The depth of the router network.
            bias: Whether to include bias in the linear layers of the router network.
            output_activation: The activation function to use in the output layer of the router network.
        """
        layers = []
        if depth == 1:
            layers.append(nn.Linear(hidden_size, num_experts, bias=bias))
        else:
            layers.append(nn.Linear(hidden_size, width, bias=bias))
            layers.append(ACTIVATION_NAME_MAP[activation]())
            for i in range(depth - 2):
                layers.append(nn.Linear(width, width, bias=bias))
                layers.append(ACTIVATION_NAME_MAP[activation]())
            layers.append(nn.Linear(width, num_experts, bias=bias))
        layers.append(ACTIVATION_NAME_MAP[output_activation]())
        super().__init__(*layers)


class TopKGating(nn.Module):
    """This module implements the Top-K gating mechanism for the Mixture of Experts layer."""
    def __init__(self, num_experts: int, k: int = None):
        super().__init__()
        self.num_experts = num_experts
        if k is None:
            k = num_experts
        self.set_k(k)

    def set_k(self, k: int):
        assert k <= self.num_experts, f"{k=} must be less than or equal to {self.num_experts=}"
        self.k = k

    @staticmethod
    def topk_mask(scores: torch.Tensor, k: int):
        top_scores, top_indices = scores.topk(k, dim=-1, sorted=False)
        mask = torch.zeros_like(scores).scatter(-1, top_indices, 1)
        return mask

    def forward(self, routing_tensor: torch.Tensor):
        expert_mask = self.topk_mask(routing_tensor, self.k)
        # set all non-topk expert scores to 0
        routing_tensor = routing_tensor * expert_mask

        return expert_mask, routing_tensor


class DynamicKGating(nn.Module):
    def __init__(self, num_experts: int, tau: float = None):
        super().__init__()
        self.num_experts = num_experts
        if tau is None:
            tau = 1 / num_experts
        self.set_tau(tau)

    def set_tau(self, tau: float):
        assert tau > 0 and tau <= 1, f"{tau=} must be between 0 and 1"
        self.tau = tau

    @staticmethod
    def dynk_mask(expert_shares: torch.Tensor, threshold: float):
        # we are interested in cumulative contribution of "selected" experts
        # staring with those with the highest contribution
        sorted_expert_shares, expert_share_indices = torch.sort(expert_shares, dim=-1, descending=True)
        cumulative_expert_shares = torch.cumsum(sorted_expert_shares, dim=-1)
        cumulative_mask = cumulative_expert_shares < threshold
        # one more expert is needed to actually cross the threshold
        # and at least one expert must be executed
        cumulative_mask[..., 1:] = cumulative_mask[..., :-1].clone()
        # cumulative_mask = cumulative_mask.roll(shifts=1, dims=-1) # not currently implemented for the MPS device.
        cumulative_mask[..., 0] = True
        # map the selected sorted experts to routing tensor
        mask = cumulative_mask.gather(dim=-1, index=inverse_argsort(expert_share_indices)).int()
        return mask

    def forward(self, routing_tensor: torch.Tensor):
        expert_mask = self.dynk_mask(routing_tensor, self.tau)
        # set all above threshold expert scores to 0
        routing_tensor = routing_tensor * expert_mask

        return expert_mask, routing_tensor


class Experts(nn.Module):
    def __init__(
        self, hidden_size: int, num_experts: int, expert_size: int, activation: nn.Module, combine_method: str
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.expert_size = expert_size
        self.w1 = nn.Parameter(torch.zeros((self.num_experts, self.hidden_size, self.expert_size)))
        self.b1 = nn.Parameter(torch.zeros((self.num_experts, self.expert_size)))
        self.w2 = nn.Parameter(torch.zeros((self.num_experts, self.expert_size, self.hidden_size)))
        self.b2 = nn.Parameter(torch.zeros(self.hidden_size))
        self.intermediate_act = activation()
        self.combine_method = combine_method

    def forward(self, x: torch.Tensor, dispatch_tensor: torch.Tensor, combine_tensor: torch.Tensor):
        # x is of size (batch_size, sequence_length, hidden_size)
        # dispatch_tensor is of size (batch_size, sequence_length, num_experts)
        # combine_tensor is of size (batch_size, sequence_length, num_experts)
        expert_inputs = torch.einsum("bnd,bne->bned", x, dispatch_tensor)
        orig_shape = expert_inputs.shape
        expert_inputs = expert_inputs.reshape(-1, self.num_experts, self.hidden_size).permute(1, 0, 2)
        hidden_states = torch.einsum("...nh,...hd->...nd", expert_inputs, self.w1)
        bias1 = torch.einsum(
            "...ni,...id->...nd",
            expert_inputs.sum(dim=-1).unsqueeze(-1).bool().int(),
            self.b1.unsqueeze(1),
        )
        hidden_states = hidden_states + bias1
        hidden_states = self.intermediate_act(hidden_states)
        hidden_states = torch.einsum("...nd,...dh->...nh", hidden_states, self.w2)
        hidden_states = hidden_states.permute(1, 0, 2).reshape(*orig_shape)

        if self.combine_method == "weighted_sum":
            hidden_states = (combine_tensor.unsqueeze(-1) * hidden_states).sum(dim=-2)
        elif self.combine_method == "sum":
            hidden_states = hidden_states.sum(dim=-2)
        else:
            raise ValueError(f"combine_method={self.combine_method} is not supported")

        hidden_states = hidden_states + self.b2
        return hidden_states

    def extra_repr(self):
        return f"in_features={self.hidden_size}, num_experts={self.num_experts}, expert_size={self.expert_size}, intermediate_act={str(self.intermediate_act)}, out_features={self.hidden_size}"


class MoELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        expert_size: int,
        activation: nn.Module,
        router_output_normalization: str = None,
        experts_combine_method: str = "sum",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_size = expert_size
        self.num_experts = num_experts
        self.experts = Experts(hidden_size, num_experts, expert_size, activation, experts_combine_method)
        self.router = None
        self.router_output_normalization = router_output_normalization
        self.gate = None
        self._gating_enabled = False

    def forward(self, x: torch.Tensor):
        # x is of size (batch_size, sequence_length, dim)
        if self.gating_enabled():
            routing_logits = self.router(x)
            if self.router_output_normalization == "softmax":
                routing_scores = torch.softmax(routing_logits, dim=-1)
            elif self.router_output_normalization == "sum":
                routing_scores = routing_logits / routing_logits.sum(dim=-1, keepdim=True)
            else:
                routing_scores = routing_logits
            expert_mask, routing_tensor = self.gate(routing_scores)
        else:
            expert_mask = torch.ones(x.size(0), x.size(1), self.num_experts, dtype=x.dtype, device=x.device)
            routing_tensor = torch.ones(x.size(0), x.size(1), self.num_experts, dtype=x.dtype, device=x.device)
        output = self.experts(x, expert_mask, routing_tensor)
        return output

    def gating_enabled(self):
        return self._gating_enabled

    def enable_gating(self):
        self._gating_enabled = True

    def disable_gating(self):
        self._gating_enabled = False


class MoEBertConfig(BertConfig):
    def __init__(self, num_experts=None, expert_size=None, exper_split=False, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_size = expert_size
        self.expert_split = exper_split


class MoEViTConfig(ViTConfig):
    def __init__(self, num_experts=None, expert_size=None, expert_split=False, **kwargs):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_size = expert_size
        self.expert_split = expert_split
