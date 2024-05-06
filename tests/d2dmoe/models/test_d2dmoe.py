from d2dmoe.models.d2dmoe import D2DMoEBertForSequenceClassification
from d2dmoe.models.moe import MoEBertConfig


def test_d2dmoe_bert_for_sequence_classification(model_name_or_path, num_experts):
    config = MoEBertConfig.from_pretrained(model_name_or_path, num_experts=num_experts)
    model = D2DMoEBertForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    assert model.num_experts == num_experts
