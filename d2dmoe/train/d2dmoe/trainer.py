import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    DataCollator,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertIntermediate
from transformers.models.vit.modeling_vit import ViTForImageClassification, ViTIntermediate
from transformers.trainer_utils import speed_metrics

from d2dmoe.arguments import TrainingArguments
from d2dmoe.common import LOSS_NAME_MAP
from d2dmoe.flops import benchmark_moe, online_evaluate_moe
from d2dmoe.models.d2dmoe import D2DMoELayer
from d2dmoe.utils import (
    add_save_inputs_hook,
    add_save_outputs_hook,
    find_module_names,
    get_module_by_name,
    get_module_name,
    remove_hooks,
)


class SparsityEnforecementTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        assert isinstance(
            self.model, (BertForSequenceClassification, ViTForImageClassification)
        ), f"SparsityEnforecementTrainer only supports BertForSequenceClassification and ViTForImageClassification, but got {self.model.__class__.__name__}"
        # add hooks to save intermediate activations
        self.intermediate_module_names = find_module_names(
            self.model, lambda m: isinstance(m, (BertIntermediate, ViTIntermediate))
        )
        self.intermediate_activations, self.intermediate_activations_handles = add_save_outputs_hook(
            self.model, self.intermediate_module_names
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        sparsity_loss = 0.0
        sparse_activations_sum = 0.0
        total_activations_sum = 0
        for module_name, acts in self.intermediate_activations.items():
            assert isinstance(acts, torch.Tensor), "Activations must be a tensor"
            sparsity_loss += acts.sum(dim=-1).mean()
            sparse_activations_sum += (acts <= 0).sum().item()
            total_activations_sum += acts.numel()
        sparsity_loss /= len(self.intermediate_module_names)
        sparsity = sparse_activations_sum / total_activations_sum
        loss += self.args.sparsity_enforcement_weight * sparsity_loss
        self.log({"Train/Sparsity": sparsity, "Train/Sparsity loss": sparsity_loss.item()})
        return (loss, outputs) if return_outputs else loss


class D2DMoERoutersTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.moe_modules_dict = {
            moe_layer_name: get_module_by_name(moe_layer_name) for moe_layer_name in self.model.moe_module_names
        }
        self.setup_for_training(self.model)
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # get_captured_layer_name_map returns dict of {moe_layer_name: moe_module}
        def get_captured_layer_name_map(model, moe_modules):
            captured_layer_name_map = {}
            for moe_name, moe_module in moe_modules.items():
                captured_layer_name_map[moe_name] = get_module_name(model, moe_module.experts.intermediate_act)
            return captured_layer_name_map

        self.captured_layer_name_map = get_captured_layer_name_map(unwrapped_model, self.moe_modules_dict)
        self.saved_inputs, input_handles = add_save_inputs_hook(unwrapped_model, self.moe_modules_dict.keys())
        self.saved_outputs, output_handles = add_save_outputs_hook(
            unwrapped_model, self.captured_layer_name_map.values()
        )
        self.hook_handles = input_handles + output_handles
        self.router_criterion_type = LOSS_NAME_MAP["mse"]
        criterion_args = {}
        self.router_criterion = self.router_criterion_type(reduction="mean", **criterion_args)

    def setup_for_training(self, model):
        model.eval()
        model.requires_grad_(False)
        for moe_name, moe_module in self.moe_modules_dict.items():
            assert moe_module.router is not None
            moe_module.router.train()
            moe_module.router.requires_grad_(True)
            moe_module.disable_gating()

    def set_for_eval_with_gating(self, tau):
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, D2DMoELayer):
                m.set_for_eval(tau)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """Perform a training step on a batch of inputs."""
        self.setup_for_training(model)
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            _ = model(**inputs)
        metrics = {}
        router_losses = []
        for moe_name, moe in self.moe_modules_dict.items():
            router = moe.router
            input = self.saved_inputs[moe_name][0]
            intermediate_output = self.saved_outputs[self.captured_layer_name_map[moe_name]]
            with torch.no_grad():
                # assumes ReLU activation
                # intermediate_output size is (num_experts, batch_size * seq_len, expert_dim)
                assert torch.all(intermediate_output >= 0.0), f"{intermediate_output=}"
                router_label = intermediate_output.sum(dim=-1)
                router_label = router_label / router_label.max()
                router_label = router_label.view(router_label.size(0), input.size(0), input.size(1))
                router_label = router_label.permute(1, 2, 0).detach()
                assert torch.all((router_label >= 0.0) & (router_label <= 1.0)), f"{router_label}"
            router_output = router(input)
            router_loss = self.router_criterion(router_output, router_label)
            router_losses.append(router_loss)
            metrics[f"Train/Router {moe_name} loss"] = router_loss.item()

        loss = torch.stack(router_losses).mean()
        metrics["Train/Average loss"] = loss.item()
        self.log(metrics)

        self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        _ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """Run evaluation and returns metrics."""
        # handle multipe eval datasets
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        num_samples = len(eval_dataloader)
        start_time = time.time()

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics = self.evaluation_loop(eval_dataloader)
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size),
            )
        )

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def evaluation_loop(self, dataloader):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        self.set_for_eval_with_gating(1.0)
        metrics = {}
        cost_without_experts, token_expert_costs, model_params = benchmark_moe(unwrapped_model, dataloader)
        for tau_to_use in self.args.tau_to_eval:
            if self.accelerator.is_main_process:
                logging.info(f"Testing on testset for tau={tau_to_use}.")
            self.set_for_eval_with_gating(self, tau_to_use)
            test_loss, test_acc, _, _ = online_evaluate_moe(
                self.accelerator,
                self.model,
                self.test_loader,
                cost_without_experts,
                token_expert_costs,
            )
            if self.accelerator.is_main_process:
                logging.info(f"Testing on trainset for tau={tau_to_use}.")
            train_loss, train_acc, total_average_flops, _ = online_evaluate_moe(
                self.accelerator,
                self.model,
                self.train_loader,
                cost_without_experts,
                token_expert_costs,
            )
            metrics[f"Eval with tau={tau_to_use}/Test loss"] = test_loss
            metrics[f"Eval with tau={tau_to_use}/Test accuracy"] = test_acc
            metrics[f"Eval with tau={tau_to_use}/Train loss"] = train_loss
            metrics[f"Eval with tau={tau_to_use}/Train accuracy"] = train_acc
            metrics[f"Eval with tau={tau_to_use}/Model FLOPs"] = total_average_flops

        metrics["Eval/Model params"] = model_params[""]

        return metrics
