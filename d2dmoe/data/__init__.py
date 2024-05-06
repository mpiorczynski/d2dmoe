from d2dmoe.data.cv.datasets import DATASET_TO_DATALOADERS as DATASET_TO_DATALOADERS_CV
from d2dmoe.data.cv.datasets import DATASET_TO_METRIC_NAME as DATASET_TO_METRIC_NAME_CV
from d2dmoe.data.cv.datasets import DATASET_TO_NUM_CLASSES as DATASET_TO_NUM_CLASSES_CV
from d2dmoe.data.nlp.datasets import DATASET_TO_DATALOADERS as DATASET_TO_DATALOADERS_NLP
from d2dmoe.data.nlp.datasets import DATASET_TO_METRIC_NAME as DATASET_TO_METRIC_NAME_NLP
from d2dmoe.data.nlp.datasets import DATASET_TO_NUM_CLASSES as DATASET_TO_NUM_CLASSES_NLP
from d2dmoe.data.utils import get_compute_metrics, get_label_names, get_tokenizer

DATASET_TO_DATALOADERS = {**DATASET_TO_DATALOADERS_CV, **DATASET_TO_DATALOADERS_NLP}
DATASET_TO_METRIC_NAME = {**DATASET_TO_METRIC_NAME_CV, **DATASET_TO_METRIC_NAME_NLP}
DATASET_TO_NUM_CLASSES = {**DATASET_TO_NUM_CLASSES_CV, **DATASET_TO_NUM_CLASSES_NLP}

__all__ = [
    "DATASET_TO_DATALOADERS",
    "DATASET_TO_METRIC_NAME",
    "DATASET_TO_NUM_CLASSES",
    "get_compute_metrics",
    "get_tokenizer",
    "get_label_names",
]