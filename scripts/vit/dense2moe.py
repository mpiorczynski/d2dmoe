import logging

from transformers import HfArgumentParser, default_data_collator, set_seed

from d2dmoe.arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from d2dmoe.data import DATASET_TO_DATALOADERS, DATASET_TO_METRIC_NAME
from d2dmoe.data.utils import get_compute_metrics, get_label_names, get_tokenizer
from d2dmoe.models.d2dmoe import D2DMoEViTForImageClassification
from d2dmoe.models.moe import MoEViTConfig
from d2dmoe.models.moefication import MoEficationViTForImageClassification
from d2dmoe.train.d2dmoe.trainer import D2DMoERoutersTrainer
from d2dmoe.train.moefication.trainer import MoEficationRoutersTrainer
from d2dmoe.train.utils import get_last_checkpoint
from d2dmoe.utils import setup_logging


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    setup_logging()
    set_seed(training_args.seed)

    train_dataset, _, eval_dataset = DATASET_TO_DATALOADERS[data_args.dataset_name](data_args.data_dir)
    labels = get_label_names(train_dataset)
    num_labels = len(labels)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    compute_metrics = get_compute_metrics[DATASET_TO_METRIC_NAME[data_args.dataset_name]]
    config = MoEViTConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.dataset_name,
        cache_dir=model_args.cache_dir,
        label2id=label2id,
        id2label=id2label,
        num_experts=model_args.num_experts,
        expert_size=model_args.expert_size,
        expert_split=model_args.expert_split,
    )

    if training_args.method == "moefication":
        model_class = MoEficationViTForImageClassification
        trainer_class = MoEficationRoutersTrainer
    elif training_args.method == "d2dmoe":
        model_class = D2DMoEViTForImageClassification
        trainer_class = D2DMoERoutersTrainer
    else:
        raise ValueError(f"Unknown training method: {training_args.method}")
   
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )
   
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        logging.info("*** Train ***")
        checkpoint = get_last_checkpoint(training_args)
        # train
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        # save model
        trainer.save_model()
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logging.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset)
        # save and log metrics
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()