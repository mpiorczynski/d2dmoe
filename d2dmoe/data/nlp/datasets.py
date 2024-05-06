import logging
import os
import random

import datasets
import transformers
from datasets import load_dataset

from d2dmoe.data.utils import clip_num_samples


def log_dataset_info(dataset: datasets.Dataset, tokenizer: transformers.PreTrainedTokenizerBase):
    """Log some information about a dataset."""
    logging.info(f"Dataset columns: {dataset.column_names}")
    logging.info(f"Dataset length: {len(dataset)}")
    for index in random.sample(range(len(dataset)), 3):
        logging.info(f"Sample {index} of the dataset: {tokenizer.decode(dataset[index]['input_ids'])}.")


def get_glue_dataset(
    task_name,
    tokenizer,
    padding="max_length",
    max_seq_length=128,
    truncation=True,
    max_train_samples=None,
    max_eval_samples=None,
):
    train_dataset = load_dataset("glue", task_name, split="train")
    validation_dataset = load_dataset("glue", task_name, split="validation")

    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }
    if task_name not in task_to_keys.keys():
        raise NotImplementedError()

    sentence1_key, sentence2_key = task_to_keys[task_name]

    if max_seq_length > tokenizer.model_max_length:
        logging.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=truncation)
        return result

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Running tokenizer on {task_name}:train",
    )
    validation_dataset = validation_dataset.map(
        preprocess_function,
        batched=True,
        desc=f"Running tokenizer on {task_name}:validation",
    )

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
    )
    validation_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
    )

    train_dataset = clip_num_samples(train_dataset, max_train_samples)
    validation_dataset = clip_num_samples(validation_dataset, max_eval_samples)

    log_dataset_info(train_dataset, tokenizer)

    return train_dataset, validation_dataset


def get_rte(
    tokenizer, padding="max_length", max_seq_length=128, truncation=True, max_train_samples=None, max_eval_samples=None
):
    return get_glue_dataset(
        "rte",
        tokenizer=tokenizer,
        padding=padding,
        max_seq_length=max_seq_length,
        truncation=truncation,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def get_qqp(
    tokenizer, padding="max_length", max_seq_length=128, truncation=True, max_train_samples=None, max_eval_samples=None
):
    return get_glue_dataset(
        "qqp",
        tokenizer=tokenizer,
        padding=padding,
        max_seq_length=max_seq_length,
        truncation=truncation,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def get_qnli(
    tokenizer, padding="max_length", max_seq_length=128, truncation=True, max_train_samples=None, max_eval_samples=None
):
    return get_glue_dataset(
        "qnli",
        tokenizer=tokenizer,
        padding=padding,
        max_seq_length=max_seq_length,
        truncation=truncation,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def get_mrpc(
    tokenizer, padding="max_length", max_seq_length=128, truncation=True, max_train_samples=None, max_eval_samples=None
):
    return get_glue_dataset(
        "mrpc",
        tokenizer=tokenizer,
        padding=padding,
        max_seq_length=max_seq_length,
        truncation=truncation,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def get_sst2(
    tokenizer, padding="max_length", max_seq_length=128, truncation=True, max_train_samples=None, max_eval_samples=None
):
    return get_glue_dataset(
        "sst2",
        tokenizer=tokenizer,
        padding=padding,
        max_seq_length=max_seq_length,
        truncation=truncation,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
    )


def get_ag_news(
    tokenizer, padding="max_length", max_seq_length=128, truncation=True, max_train_samples=None, max_eval_samples=None
):
    train_dataset = load_dataset("ag_news", split="train")
    test_dataset = load_dataset("ag_news", split="test")

    if max_seq_length > tokenizer.model_max_length:
        logging.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        result = tokenizer(examples["text"], padding=padding, max_length=max_seq_length, truncation=truncation)
        return result

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on ag_news:train",
    )
    test_dataset = test_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on ag_news:test",
    )

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
    )

    train_dataset = clip_num_samples(train_dataset, max_train_samples)
    test_dataset = clip_num_samples(test_dataset, max_eval_samples)

    log_dataset_info(train_dataset, tokenizer)

    return train_dataset, test_dataset


def get_emotion(
    tokenizer,
    padding="max_length",
    max_seq_length=128,
    truncation=True,
    max_train_samples=None,
    max_eval_samples=None,
):
    train_dataset = load_dataset("dair-ai/emotion", split="train")
    validation_dataset = load_dataset("dair-ai/emotion", split="validation")

    if max_seq_length > tokenizer.model_max_length:
        logging.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        result = tokenizer(examples["text"], padding=padding, max_length=max_seq_length, truncation=truncation)
        return result

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on emotion:train",
    )
    validation_dataset = validation_dataset.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on emotion:validation",
    )

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
    )
    validation_dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "label"],
    )

    train_dataset = clip_num_samples(train_dataset, max_train_samples)
    validation_dataset = clip_num_samples(validation_dataset, max_eval_samples)

    log_dataset_info(train_dataset, tokenizer)

    return train_dataset, validation_dataset


DATASET_TO_DATALOADERS = {
    "rte": get_rte,
    "qqp": get_qqp,
    "qnli": get_qnli,
    "mrpc": get_mrpc,
    "sst2": get_sst2,
    "ag_news": get_ag_news,
    "emotion": get_emotion,
}

DATASET_TO_SEQUENCE_LENGTH = {
    "rte": 128,
    "qqp": 128,
    "qnli": 128,
    "mrpc": 128,
    "sst2": 128,
    "ag_news": 128,
    "emotion": 128,
}

DATASET_TO_NUM_CLASSES = {
    "rte": 2,
    "qqp": 2,
    "qnli": 2,
    "mrpc": 2,
    "sst2": 2,
    "ag_news": 4,
    "emotion": 6,
}

DATASET_TO_METRIC_NAME = {
    "rte": "accuracy",
    "qqp": "accuracy",
    "qnli": "accuracy",
    "mrpc": "accuracy",
    "sst2": "accuracy",
    "ag_news": "accuracy",
    "emotion": "accuracy",
}
