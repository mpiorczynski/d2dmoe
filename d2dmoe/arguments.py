from dataclasses import dataclass, field
from typing import List, Optional, Union

from transformers import TrainingArguments as HfTrainingArguments


@dataclass
class TrainingArguments(HfTrainingArguments):
    sparsity_enforcement_weight: Optional[float] = field(
        default=None, metadata={"help": "Regularization weight for sparsity inducing regularization."}
    )
    k_to_eval: Optional[List[int]] = field(
        default=None, metadata={"help": "Number of experts to select for evaluation."}
    )
    tau_to_eval: Optional[List[float]] = field(
        default=None, metadata={"help": "Threshold to select the number of experts for evaluation."}
    )
    def __post_init__(self):
        super().__post_init__()
        if self.k_to_eval is not None and not isinstance(self.k_to_eval, list):
            self.k_to_eval = [self.k_to_eval]
        if self.tau_to_eval is not None and not isinstance(self.tau_to_eval, list):
            self.tau_to_eval = [self.tau_to_eval]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to train on."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    padding: str = field(
        default="max_length",
        metadata={"help": "Padding strategy to use. Can be 'max_length' or 'longest' or 'do_not_pad'."},
    )
    truncation: bool = field(
        default=True,
        metadata={"help": "Truncate the samples to max_seq_length or not."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The data directory to load the dataset from."}
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    num_experts: int = field(default=None, metadata={"help": "Number of experts in MoE layers."})
    expert_size: int = field(default=None, metadata={"help": "Size of each expert in MoE layers."})
    expert_split: Optional[bool] = field(
        default=True, metadata={"help": "Whether to perform the experts split step."}
    )

    def __post_init__(self):
        self.tokenizer_name = self.tokenizer_name if self.tokenizer_name else self.model_name_or_path
