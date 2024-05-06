import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, default_data_collator

from d2dmoe.flops import benchmark


def sample_to_dataloader(sample: dict) -> DataLoader:
    return DataLoader(
        [sample], batch_size=1, collate_fn=default_data_collator
    )

def test_benchmark(model_name_or_path, text):
    # given
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    model = BertModel.from_pretrained(model_name_or_path)
    encoded_input = tokenizer(text, return_tensors='pt')
    dataloader = sample_to_dataloader(encoded_input)
    
    # when
    model_costs, param_count = benchmark(model, dataloader)
    
    # then
    assert param_count[''] == sum(p.numel() for p in model.parameters())