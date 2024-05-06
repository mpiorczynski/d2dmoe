import logging
from typing import Callable

import datasets
import numpy as np
from transformers import AutoTokenizer, EvalPrediction

from d2dmoe.common import METRIC_TO_FUNCTION


def clip_num_samples(dataset: datasets.Dataset, max_samples: int = None):
    """Clip the number of samples in a dataset."""
    if max_samples is not None:
        max_samples = min(len(dataset), max_samples)
        logging.info(f"Clipping dataset to {max_samples} samples.")
        dataset = dataset.select(range(max_samples))
    else:
        logging.info("No dataset clipping.")
    return dataset


def get_compute_metrics(metric_name) -> Callable:
    metric = METRIC_TO_FUNCTION[metric_name]

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = {metric_name: metric(y_pred=preds, y_true=p.label_ids)}
        return result

    return compute_metrics


def get_tokenizer(tokenizer_name: str):
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def get_label_names(dataset: datasets.Dataset):
    if "label" in dataset.features:
        return dataset.features["label"].names
    elif "labels" in dataset.features:
        return dataset.features["labels"].names
    else:
        raise ValueError("No label names found in the dataset.")
