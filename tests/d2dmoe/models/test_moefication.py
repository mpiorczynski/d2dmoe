from d2dmoe.models.moe import MoEBertConfig
from d2dmoe.models.moefication import MoEficationBertForSequenceClassification


def test_moefication_bert_for_sequence_classification(model_name_or_path, num_experts):
    config = MoEBertConfig.from_pretrained(model_name_or_path, num_experts=num_experts)
    model = MoEficationBertForSequenceClassification.from_pretrained(model_name_or_path, config=config)

    assert model.num_experts == num_experts
