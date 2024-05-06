import pytest
from transformers import AutoConfig, BertForSequenceClassification

from d2dmoe.data import DATASET_TO_DATALOADERS
from d2dmoe.data.utils import get_tokenizer
from d2dmoe.models.moe import MoEBertConfig
from d2dmoe.models.moefication import MoEficationBertForSequenceClassification


@pytest.fixture
def bert(model_name_or_path, num_labels):
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    return BertForSequenceClassification.from_pretrained(model_name_or_path, config=config)


@pytest.fixture
def moefication(model_name_or_path, num_experts):
    config = MoEBertConfig.from_pretrained(model_name_or_path, num_experts=num_experts)
    return MoEficationBertForSequenceClassification.from_pretrained(model_name_or_path, config=config)


@pytest.fixture
def dataset(model_name_or_path):
    tokenizer = get_tokenizer(model_name_or_path)

    train_dataset, _ = DATASET_TO_DATALOADERS["rte"](
        tokenizer=tokenizer,
        max_train_samples=8,
        max_eval_samples=8,
    )
    return train_dataset
