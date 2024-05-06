from transformers import default_data_collator

from d2dmoe.arguments import TrainingArguments
from d2dmoe.train.d2dmoe.trainer import SparsityEnforecementTrainer


def test_sparsity_enforcement(bert, dataset, tmpdir):
    # given
    training_args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=1,
        logging_strategy='no',
        per_device_train_batch_size=4,
        sparsity_enforcement_weight=1e-3, 
    )

    trainer = SparsityEnforecementTrainer(
        model=bert,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=dataset
    )

    # when
    trainer.train()

    # then
    assert trainer.state.global_step == 1
    

