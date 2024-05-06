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
from transformers.trainer_utils import speed_metrics

from d2dmoe.arguments import TrainingArguments
from d2dmoe.common import LOSS_NAME_MAP
from d2dmoe.flops import benchmark_moe, online_evaluate_moe
from d2dmoe.models.moefication import MoEficationLayer
from d2dmoe.utils import add_save_inputs_hook, add_save_outputs_hook, get_module_by_name, get_module_name, remove_hooks


class MoEficationRoutersTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None, # type: ignore
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
            moe_layer_name: get_module_by_name(model, moe_layer_name) for moe_layer_name in self.model.moe_module_names
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
        self.router_criterion_type = LOSS_NAME_MAP[args.router_loss_type]
        criterion_args = args.router_loss_args if args.router_loss_args is not None else {}
        self.router_criterion = self.router_criterion_type(reduction="mean", **criterion_args)

    def setup_for_training(self, model):
        model.eval()
        model.requires_grad_(False)
        for moe_name, moe_module in self.moe_modules_dict.items():
            assert moe_module.router is not None
            moe_module.router.train()
            moe_module.router.requires_grad_(True)
            moe_module.disable_gating()

    def set_for_eval_with_gating(self, k):
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, MoEficationLayer):
                m.set_for_eval(k)

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
        self.set_for_eval_with_gating(1)
        metrics = {}
        cost_without_experts, token_expert_costs, model_params = benchmark_moe(unwrapped_model, dataloader)
        for k_to_use in self.args.k_to_eval:
            if self.accelerator.is_main_process:
                logging.info(f"Testing on testset for k={k_to_use}.")
            self.set_for_eval_with_gating(self, k_to_use)
            test_loss, test_acc, _, _ = online_evaluate_moe(
                self.accelerator,
                self.model,
                self.test_loader,
                cost_without_experts,
                token_expert_costs,
            )
            if self.accelerator.is_main_process:
                logging.info(f"Testing on trainset for k={k_to_use}.")
            train_loss, train_acc, total_average_flops, _ = online_evaluate_moe(
                self.accelerator,
                self.model,
                self.train_loader,
                cost_without_experts,
                token_expert_costs,
            )
        metrics[f"Eval with k={k_to_use}/Test loss"] = test_loss
        metrics[f"Eval with k={k_to_use}/Test accuracy"] = test_acc
        metrics[f"Eval with k={k_to_use}/Train loss"] = train_loss
        metrics[f"Eval with k={k_to_use}/Train accuracy"] = train_acc
        metrics[f"Eval with k={k_to_use}/Model FLOPs"] = total_average_flops
        metrics["Eval/Model params"] = model_params[""]

        return metrics
