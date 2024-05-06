from transformers import default_data_collator

from d2dmoe.arguments import TrainingArguments
from d2dmoe.data import DATASET_TO_DATALOADERS
from d2dmoe.train.moefication.trainer import MoEficationRoutersTrainer


def test_router_training(moefication, dataset, tmpdir):
    # given
    training_args = TrainingArguments(
        output_dir=tmpdir,
        max_steps=1,
        logging_strategy='no',
        per_device_train_batch_size=4,
        router_loss_type='bcewl',
        k_to_eval=1,
    )

    trainer = MoEficationRoutersTrainer(
        model=moefication,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=dataset
    )

    # when
    trainer.train()

    # then
    assert trainer.state.global_step == 1