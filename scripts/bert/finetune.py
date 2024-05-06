import logging

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    HfArgumentParser,
    Trainer,
    default_data_collator,
    set_seed,
)

from d2dmoe.arguments import DataTrainingArguments, ModelArguments, TrainingArguments
from d2dmoe.data import DATASET_TO_DATALOADERS, DATASET_TO_METRIC_NAME
from d2dmoe.data.utils import get_compute_metrics, get_label_names, get_tokenizer
from d2dmoe.train import SparsityEnforecementTrainer
from d2dmoe.train.utils import get_last_checkpoint
from d2dmoe.utils import setup_logging


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    setup_logging()
    set_seed(training_args.seed)

    tokenizer = get_tokenizer(model_args.tokenizer_name)
    train_dataset, eval_dataset = DATASET_TO_DATALOADERS[data_args.dataset_name](
        tokenizer=tokenizer,
        padding=data_args.padding,
        max_seq_length=data_args.max_seq_length,
        truncation=data_args.truncation,
        max_train_samples=data_args.max_train_samples,
        max_eval_samples=data_args.max_eval_samples,
    )
    labels = get_label_names(train_dataset)
    num_labels = len(labels)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    compute_metrics = get_compute_metrics[DATASET_TO_METRIC_NAME[data_args.dataset_name]]
    config = BertConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.dataset_name,
        cache_dir=model_args.cache_dir,
        label2id=label2id,
        id2label=id2label,
    )

    model = BertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    if training_args.method == "moefication":
        trainer_class = Trainer
    elif training_args.method == "d2dmoe":
        trainer_class = SparsityEnforecementTrainer
    else:
        raise ValueError(f"Unknown training method: {training_args.method}")
    
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
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