import logging
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertLayer
from transformers.models.vit.modeling_vit import ViTForImageClassification, ViTLayer

from d2dmoe.models.moe import DynamicKGating, MoEBertConfig, MoELayer, MoEViTConfig, Router
from d2dmoe.models.utils import load_moe_weights, parameter_clustering_split, replace_layer_with_moe
from d2dmoe.utils import find_module_names, get_module_by_name, set_module_by_name


class D2DMoERouter(Router):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__(
            hidden_size,
            num_experts,
            width=num_experts,
            depth=2,
            bias=True,
            activation="relu",
            output_activation="abs",
        )


class D2DMoELayer(MoELayer):
    def __init__(self, hidden_size: int, num_experts: int, expert_size: int, activation: nn.Module):
        super().__init__(hidden_size, num_experts, expert_size, activation)
        self.router = D2DMoERouter(hidden_size, num_experts)
        self.gate = DynamicKGating(num_experts)

    def set_for_eval(self, tau: float):
        self.enable_gating()
        self.gate.set_tau(tau)


class D2DMoEBertForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config: MoEBertConfig):
        self.num_experts = config.num_experts
        self.expert_size = config.expert_size
        assert (
            self.num_experts is not None or self.expert_size is not None
        ), "either num_experts or expert_size must be passed."
        assert not (
            self.num_experts is not None and self.expert_size is not None
        ), "num_experts and expert_size cannot be both passed."
        self.moe_module_names = []
        super().__init__(config)
        if config.expert_split:
            self.replace_ffns_with_moes()
            self.config.expert_split = False


    @staticmethod
    def ffn_filter_condition(m: nn.Module):
        if isinstance(m, BertLayer):
            return True

    def replace_ffns_with_moes(self):
        modules_to_moefy = find_module_names(self, self.ffn_filter_condition)
        for i, layer_name in enumerate(modules_to_moefy):
            logging.info(f"Replacing {layer_name} with MoE layer with {self.num_experts} experts")
            original_layer = get_module_by_name(self, layer_name)
            moefied_layer = replace_layer_with_moe(original_layer, D2DMoELayer, self.num_experts, self.expert_size)
            moe_weights = parameter_clustering_split(
                get_module_by_name(self, layer_name + ".intermediate.dense"),
                get_module_by_name(self, layer_name + ".output.dense"),
                self.num_experts,
            )
            load_moe_weights(moefied_layer, moe_weights)
            moe_layer_name = layer_name + ".moe"
            set_module_by_name(self, moe_layer_name, moefied_layer)
            self.moe_module_names.append(moe_layer_name)

            # move droupout and layernorm to the end of the layer
            set_module_by_name(self, layer_name + ".dropout", original_layer.output.dropout)
            set_module_by_name(self, layer_name + ".LayerNorm", original_layer.output.LayerNorm)

            # delete intermediate and output layers
            del original_layer.intermediate
            del original_layer.output

            # adjust feed_forward_chunk to use MoE
            self.bert.encoder.layer[i].feed_forward_chunk = partial(
                self._feed_forward_chunk, self.bert.encoder.layer[i]
            )

    def _feed_forward_chunk(self, layer, attention_output):
        if hasattr(self, "moe"):
            moe_output = layer.moe(attention_output)
            moe_output = layer.dropout(moe_output)
            layer_output = layer.LayerNorm(moe_output + attention_output)
        else:
            intermediate_output = layer.intermediate(attention_output)
            layer_output = layer.output(intermediate_output, attention_output)
        return layer_output


class D2DMoEViTForImageClassification(ViTForImageClassification):
    def __init__(self, config: MoEViTConfig):
        self.num_experts = config.num_experts
        self.expert_size = config.expert_size
        assert (
            self.num_experts is not None or self.expert_size is not None
        ), "either num_experts or expert_size must be passed."
        assert not (
            self.num_experts is not None and self.expert_size is not None
        ), "num_experts and expert_size cannot be both passed."
        self.moe_module_names = []
        super().__init__()
        if config.expert_split:
            self.replace_ffns_with_moes()
            self.config.expert_split = False

            
    @staticmethod
    def ffn_filter_condition(m: nn.Module):
        if isinstance(m, ViTLayer):
            return True

    def replace_ffns_with_moes(self):
        modules_to_moefy = find_module_names(self, self.ffn_filter_condition)
        for i, layer_name in enumerate(modules_to_moefy):
            logging.info(f"Replacing {layer_name} with MoE layer with {self.num_experts} experts")
            original_layer = get_module_by_name(self, layer_name)
            moefied_layer = replace_layer_with_moe(original_layer, D2DMoELayer, self.num_experts, self.expert_size)
            moe_weights = parameter_clustering_split(
                get_module_by_name(self, layer_name + ".intermediate.dense"),
                get_module_by_name(self, layer_name + ".output.dense"),
                self.num_experts,
            )
            load_moe_weights(moefied_layer, moe_weights)
            moe_layer_name = layer_name + ".moe"
            set_module_by_name(self, moe_layer_name, moefied_layer)
            self.moe_module_names.append(moe_layer_name)

            # move droupout and layernorm to the end of the layer
            set_module_by_name(self, layer_name + ".dropout", original_layer.output.dropout)

            # delete intermediate and output layers
            del original_layer.intermediate
            del original_layer.output

            # adjust feed_forward_chunk to use MoE
            self.vit.encoder.layer[i] = partial(self._vit_layer_forward, self.vit.encoder.layer[i])

    def _vit_layer_forward(
        self,
        layer,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = layer.attention(
            layer.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = layer.layernorm_after(hidden_states)
        if hasattr(layer, "moe"):
            layer_output = layer.moe(layer_output)
            layer_output = layer.dropout(layer_output)
            layer_output = layer_output + hidden_states

        else:
            layer_output = layer.intermediate(layer_output)
            # second residual connection is done here
            layer_output = layer.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs
