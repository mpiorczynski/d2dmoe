import logging
import numbers
from collections import Counter
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from accelerate import Accelerator
from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
from fvcore.nn.jit_handles import elementwise_flop_counter, get_shape

from d2dmoe.models.moe import DynamicKGating, Experts, MoELayer, Router, TopKGating
from d2dmoe.utils import (
    add_save_activations_hook,
    add_save_outputs_hook,
    find_module_names,
    get_module_by_name,
    get_single_sample,
    human_readable,
    remove_hooks,
    to_model_device,
)


def benchmark(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader) -> Tuple[FlopCountAnalysis, Dict]:
    """Benchmark the model and return the cost of the model and the number of parameters."""
    model.eval()
    inputs = next(iter(data_loader))
    sample = get_single_sample(inputs)
    sample = to_model_device(sample, model)
    with torch.inference_mode():
        model_costs = flop_count(
            model,
            (sample["input_ids"], sample["token_type_ids"], sample["attention_mask"]),
        )
        param_count = parameter_count(model)
    logging.info(f"Ops by operator:\n{model_costs.by_operator()}")
    logging.info(f"Ops by module:\n{flop_count_table(model_costs, max_depth=7)}")
    logging.info(f"Total ops: {model_costs.total()} ({human_readable(model_costs.total())})")
    unsupported = model_costs.unsupported_ops()
    if len(unsupported) > 0:
        for k, v in unsupported.items():
            logging.warning(f"Unsupported op: {k} (occurrences: {v})")
    uncalled = model_costs.uncalled_modules()
    if len(uncalled) > 0:
        for m in uncalled:
            logging.warning(f"Uncalled module: {m}")
    return model_costs, param_count


def benchmark_moe(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader):
    """Benchmark the model and return the cost of the model without experts and the cost of a single expert for a single token."""
    model.eval()
    model_costs, model_params = benchmark(model, data_loader)
    # find MoE modules and order them (already done in the model)
    moe_module_names = find_module_names(model, lambda m: isinstance(m, MoELayer))
    # add hooks on gating networks and expert modules
    experts_modules_names = {}
    for moe_module_name in moe_module_names:
        moe_module = get_module_by_name(model, moe_module_name)
        # find the experts module
        experts_names = find_module_names(moe_module, lambda m: isinstance(m, Experts))
        assert len(experts_names) == 1, f"{len(experts_names)=}"
        experts_modules_names[moe_module_name] = f"{moe_module_name}.{experts_names[0]}"
    # add hooks
    experts_modules_names_list = list(experts_modules_names.values())
    experts_inputs, _, experts_handles = add_save_activations_hook(model, experts_modules_names_list)
    # push an example though forward
    batch = next(iter(data_loader))
    sample = get_single_sample(batch)
    sample = to_model_device(sample, model)
    with torch.no_grad():
        _ = model(**sample)
    # push a single sample though each of the modules and calculate its costs
    cost_without_experts = model_costs.total()
    token_expert_costs = {}
    for moe_name in moe_module_names:
        experts_name = experts_modules_names[moe_name]
        experts_module = get_module_by_name(model, experts_name)
        # calculate cost of the gating network
        gating_cost = model_costs.by_module()[moe_name] - model_costs.by_module()[experts_name]
        # calculate the cost of a single expert for a single sample
        experts_input = experts_inputs[experts_name]
        assert experts_input[0].dim() == 3
        assert experts_input[1].dim() == 3
        assert experts_input[0].size(0) == 1, "Expected a single sample"
        token_expert_mask = torch.zeros_like(experts_input[1])
        token_expert_mask[0, 0, 0] = 1.0
        token_routing_tensor = torch.zeros_like(experts_input[1])
        token_routing_tensor[0, 0, 0] = 1.0
        experts_input = (experts_input[0], token_expert_mask, token_routing_tensor)
        with torch.no_grad():
            token_expert_cost = flop_count(experts_module, experts_input).total()
        seq_length = experts_input[0].size(1)
        # all experts for each token in the sequence is executed
        # so we divide by these two values
        token_expert_cost /= experts_module.num_experts * seq_length
        logging.info(
            f"MoE {moe_name} single token's expert cost: {token_expert_cost} for sequence length {seq_length}"
        )
        cost_without_experts -= model_costs.by_module()[moe_name] - gating_cost
        token_expert_costs[moe_name] = token_expert_cost
    remove_hooks(experts_handles)
    logging.info(f"Model cost without experts: {cost_without_experts}")
    # find router modules and their costs
    router_module_names = find_module_names(model, lambda m: isinstance(m, Router))
    router_costs = [model_costs.by_module()[router_name] for router_name in router_module_names]
    logging.info(f"Router costs: {router_costs}")
    return cost_without_experts, token_expert_costs, router_costs, model_params


def online_evaluate_moe(
    accelerator: Accelerator,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    cost_without_experts: float,
    token_expert_costs: Dict[str, float],
    criterion_class: torch.nn.Module = nn.CrossEntropyLoss,
):
    """Evaluate the model and return the loss, accuracy, and the average FLOPs per token."""
    criterion = criterion_class(reduction="sum")
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    total_average_flops = cost_without_experts
    moe_executed_tokens = {name: 0 for name in token_expert_costs.keys()}
    expert_average_costs = {}
    gating_modules_names = find_module_names(model, lambda m: isinstance(m, (TopKGating, DynamicKGating)))
    gating_data, gating_handles = add_save_outputs_hook(model, gating_modules_names)

    with torch.inference_mode():
        for batch in data_loader:
            y = batch.pop("labels")
            y_pred = model(**batch).logits
            # each element of gating_data_list is a tuple (expert_mask, routing_tensor), so we select only the final routing decisions
            gating_data = {k: v[0] for k, v in gating_data.items()}
            # gating data should be a dict with tensor values of size (batch_size, sequence_length, num_experts) now
            y_pred, y, gating_data = accelerator.gather_for_metrics((y_pred, y, gating_data))
            y_pred_max = y_pred.argmax(dim=-1)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            correct += (y_pred_max == y).sum().item()
            total += y.numel()
            for moe_name in token_expert_costs.keys():
                moe_executed_tokens[moe_name] += (gating_data[moe_name] > 0.0).long().sum().item()
    for moe_name, token_expert_cost in token_expert_costs.items():
        expert_average_cost = moe_executed_tokens[moe_name] * token_expert_cost / total
        logging.info(f"Averaged FLOPs for MoE {moe_name}: {expert_average_cost}")
        expert_average_costs[moe_name] = expert_average_cost
        total_average_flops += expert_average_cost
    remove_hooks(gating_handles)
    return running_loss / total, correct / total, total_average_flops, expert_average_costs


def count_scaled_dot_product_attention_ops(inputs: List[Any], _outputs: List[Any]) -> Counter[str]:
    q_samples, q_heads, q_tokens, q_dim = get_shape(inputs[0])
    k_samples, k_heads, k_tokens, k_dim = get_shape(inputs[1])
    v_samples, v_heads, v_tokens, v_dim = get_shape(inputs[2])
    assert q_dim == k_dim
    assert q_tokens == k_tokens == v_tokens
    assert q_heads == k_heads == v_heads
    assert q_samples == k_samples == v_samples
    # query @ key.transpose(-2, -1)
    q_k_matmul_ops = q_samples * q_heads * q_tokens * q_tokens * q_dim
    # / math.sqrt(query.size(-1))
    div_ops = q_samples * q_heads * q_tokens * q_tokens
    # + attn_mask
    if get_shape(inputs[3]) is not None or isinstance(inputs[3], numbers.Number):
        mask_add_ops = q_samples * q_heads * q_tokens * q_tokens
    else:
        mask_add_ops = 0
    # torch.softmax(..., dim=-1)
    softmax_ops = 2 * q_samples * q_heads * q_tokens * q_tokens
    # attn_weight @ value
    v_matmul_ops = q_samples * q_heads * q_tokens * q_tokens * v_dim
    return Counter(
        {
            "matmul": q_k_matmul_ops + v_matmul_ops,
            "div": div_ops,
            "add": mask_add_ops,
            "softmax": softmax_ops,
        }
    )


OP_HANDLERS = {
    "aten::add": elementwise_flop_counter(0, 1),
    "aten::add_": elementwise_flop_counter(0, 1),
    "aten::radd": elementwise_flop_counter(0, 1),
    "aten::sub": elementwise_flop_counter(0, 1),
    "aten::sub_": elementwise_flop_counter(0, 1),
    "aten::rsub": elementwise_flop_counter(0, 1),
    "aten::mul": elementwise_flop_counter(0, 1),
    "aten::mul_": elementwise_flop_counter(0, 1),
    "aten::rmul": elementwise_flop_counter(0, 1),
    "aten::div": elementwise_flop_counter(0, 1),
    "aten::div_": elementwise_flop_counter(0, 1),
    "aten::rdiv": elementwise_flop_counter(0, 1),
    "aten::exp": elementwise_flop_counter(0, 1),
    "aten::cumsum": elementwise_flop_counter(0, 1),
    "aten::ne": elementwise_flop_counter(0, 1),
    "aten::gelu": elementwise_flop_counter(0, 1),
    "aten::silu_": elementwise_flop_counter(0, 1),
    "aten::dropout_": elementwise_flop_counter(0, 1),
    "aten::sigmoid": elementwise_flop_counter(0, 1),
    "aten::softmax": elementwise_flop_counter(0, 2),
    "aten::log_softmax": elementwise_flop_counter(0, 2),
    "aten::argmax": elementwise_flop_counter(0, 1),
    "aten::one_hot": elementwise_flop_counter(0, 1),
    "aten::flatten": elementwise_flop_counter(0, 0),
    "aten::unflatten": elementwise_flop_counter(0, 0),
    "aten::mean": elementwise_flop_counter(1, 0),
    "aten::sum": elementwise_flop_counter(1, 0),
    "aten::abs": elementwise_flop_counter(0, 1),
    "aten::tanh": elementwise_flop_counter(0, 1),
    "aten::relu": elementwise_flop_counter(0, 1),
    "aten::where": elementwise_flop_counter(0, 1),
    "aten::le": elementwise_flop_counter(0, 1),
    "aten::topk": elementwise_flop_counter(1, 1),
    "aten::sort": elementwise_flop_counter(1, 1),
    "aten::argsort": elementwise_flop_counter(1, 1),
    "aten::scatter": elementwise_flop_counter(1, 1),
    "aten::gather": elementwise_flop_counter(1, 1),
    "aten::adaptive_max_pool2d": elementwise_flop_counter(1, 0),
    "aten::scaled_dot_product_attention": count_scaled_dot_product_attention_ops,
}


def flop_count(model: torch.nn.Module, input) -> FlopCountAnalysis:
    return FlopCountAnalysis(model, input).set_op_handle(**OP_HANDLERS)
