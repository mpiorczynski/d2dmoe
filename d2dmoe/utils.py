import logging
import random
from functools import reduce
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def human_readable(number: int):
    """Convert a number to a human readable format."""
    units = ["", "K", "M", "B", "T", "P", "E", "Z", "Y"]
    unit_index = 0

    while abs(number) >= 1000 and unit_index < len(units) - 1:
        number /= 1000.0
        unit_index += 1

    return "{:.1f}{}".format(number, units[unit_index])


def get_single_sample(inputs: Dict[str, torch.Tensor], index: int = 0):
    """Get a single sample from a batch of inputs."""
    return {k: v[index : (index + 1)].squeeze().unsqueeze(0) for k, v in inputs.items()}


def to_model_device(inputs: Dict, model: nn.Module):
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    return inputs


def setup_logging():
    logging.basicConfig(
        format=("[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"),
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
        force=True,
    )
    logging.info("Configured logging.")


def inverse_argsort(sorted_indices: torch.Tensor):
    """https://stackoverflow.com/questions/52127723/pytorch-better-way-to-get-back-original-tensor-order-after-torch-sort"""
    return sorted_indices.argsort(dim=-1)


def find_module_names(model: nn.Module, filter: Callable[[nn.Module], bool]):
    """Find all module names in a model that satisfy the filter."""
    found_names = []
    for name, module in model.named_modules():
        if filter(module):
            found_names.append(name)
    return found_names


def get_module_by_name(module: nn.Module, name: str):
    names = name.split(sep=".")
    return reduce(getattr, names, module)


def get_parent_module_name(name: str):
    names = name.split(sep=".")
    return ".".join(names[:-1])


def set_module_by_name(module: nn.Module, name: str, replacement: nn.Module):
    names = name.split(sep=".")
    parent = reduce(getattr, names[:-1], module)
    setattr(parent, names[-1], replacement)


def get_module_name(module: nn.Module, submodule: nn.Module):
    for name, m in module.named_modules():
        if m is submodule:
            return name


def add_save_activations_hook(model: nn.Module, module_names: List[str]):
    module_inputs = {}
    module_outputs = {}
    module_id_to_name = {}
    hook_handles = []
    for name in module_names:
        module = get_module_by_name(model, name)
        module_id_to_name[id(module)] = name

        def save_activations_hook(m: nn.Module, input: torch.Tensor, output: torch.Tensor):
            module_name = module_id_to_name[id(m)]
            module_inputs[module_name] = input
            module_outputs[module_name] = output

        handle = module.register_forward_hook(save_activations_hook)
        hook_handles.append(handle)
    return module_inputs, module_outputs, hook_handles


def add_save_inputs_hook(model: nn.Module, module_names: List[str]):
    module_inputs = {}
    module_id_to_name = {}
    hook_handles = []
    for name in module_names:
        module = get_module_by_name(model, name)
        module_id_to_name[id(module)] = name

        def save_activations_hook(m, input, _output):
            module_name = module_id_to_name[id(m)]
            module_inputs[module_name] = input

        handle = module.register_forward_hook(save_activations_hook)
        hook_handles.append(handle)
    return module_inputs, hook_handles


def add_save_outputs_hook(model: nn.Module, module_names: List[str]):
    gating_outputs = {}
    module_id_to_name = {}
    hook_handles = []

    for name in module_names:
        module = get_module_by_name(model, name)
        module_id_to_name[id(module)] = name

        def save_gating_hook(m: nn.Module, _input: torch.Tensor, output: torch.Tensor):
            module_name = module_id_to_name[id(m)]
            gating_outputs[module_name] = output

        handle = module.register_forward_hook(save_gating_hook)
        hook_handles.append(handle)
    return gating_outputs, hook_handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()

def inverse_dict(d):
    return {v: k for k, v in d.items()}