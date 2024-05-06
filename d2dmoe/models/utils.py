import logging
from typing import Dict, Union

import numpy as np
import torch
from k_means_constrained import KMeansConstrained
from torch import nn
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.vit.modeling_vit import ViTLayer

from d2dmoe.models.moe import MoELayer
from d2dmoe.utils import get_module_by_name, set_module_by_name


def parameter_clustering_split(
    w1: nn.Linear, w2: nn.Linear, num_experts: int, method: str = "balanced_k_means"
) -> Dict[str, torch.Tensor]:
    """See section `3.2 Expert Construction` in https://arxiv.org/pdf/2110.01786.pdf"""
    intermediate_size, hidden_size = w1.weight.shape

    expert_size = int(intermediate_size / num_experts)
    assert expert_size * num_experts == intermediate_size, f"Experts split is uneven with num_experts: {num_experts}."
    logging.info(f"Splitting {intermediate_size} neurons into {num_experts} experts of size {expert_size}")
    w1_matrix_norm = torch.nn.functional.normalize(w1.weight.detach().cpu(), p=2.0, dim=1)
    if method == "balanced_k_means":
        clf = KMeansConstrained(n_clusters=num_experts, size_min=expert_size, size_max=expert_size)
        labels = clf.fit_predict(w1_matrix_norm.numpy())

    elif method == "random":
        labels = np.repeat(np.arange(num_experts), expert_size)[np.random.permutation(intermediate_size)]

    else:
        raise NotImplementedError(f"Unknown clustering method: {method}")
    
    b1_moe = torch.zeros((num_experts, expert_size))
    w1_moe = torch.zeros((num_experts, hidden_size, expert_size))
    b2_moe = w2.bias
    w2_moe = torch.zeros((num_experts, expert_size, hidden_size))
    
    filled_neuron_counts = [0 for _ in range(num_experts)]
    for neuron_index, expert_index in enumerate(labels):
        exp_ix = filled_neuron_counts[expert_index]
        w1_moe[expert_index, :, exp_ix] = w1.weight[neuron_index, :]
        b1_moe[expert_index, exp_ix] = w1.bias[neuron_index]
        w2_moe[expert_index, exp_ix, :] = w2.weight[:, neuron_index]
        filled_neuron_counts[expert_index] += 1

    return {"w1": w1_moe, "b1": b1_moe, "w2": w2_moe, "b2": b2_moe}


def load_moe_weights(layer: MoELayer, moe_weights: Dict[str, Dict[str, torch.Tensor]]):
    """Load clustering weights into an Experts layer."""
    layer.experts.w1.data = moe_weights["w1"]
    layer.experts.b1.data = moe_weights["b1"]
    layer.experts.w2.data = moe_weights["w2"]
    layer.experts.b2.data = moe_weights["b2"]


def replace_layer_with_moe(original_layer: Union[BertLayer, ViTLayer], moe_class: MoELayer, num_experts=None, expert_size=None):
    w1 = original_layer.intermediate.dense
    activation = type(original_layer.intermediate.intermediate_act_fn)
    hidden_size = w1.in_features
    intermediate_size = w1.out_features
    if num_experts is not None:
        assert intermediate_size % num_experts == 0, "intermediate_size has to be divisible by the num_experts"
        expert_size = intermediate_size // num_experts
    elif expert_size is not None:
        assert intermediate_size % expert_size == 0, "intermediate_size has to be divisible by the expert_size"
        num_experts = intermediate_size // expert_size
    moe_layer = moe_class(hidden_size, num_experts, expert_size, activation)
    return moe_layer
