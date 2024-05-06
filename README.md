# d2dmoe
`d2dmoe` is a Python package implementing Dense To Dynamic-k Mixture-of-Experts (D2DMoE), a simple and practical method that reduces the computational costs of Transformer models during the inference by converting standard pretrained model checkpoint into its MoE version.
D2DMoE is based on the MoEfication[1], also implemented in `d2dmoe`, but brings several improvements to this method.
`d2dmoe` is built on top of PyTorch and benefits from the user-friendly Hugging Face Transformers API.

## Installation
`d2dmoe` can be installed via PyPI by simply executing the following statement:
```
pip install d2dmoe
```

If you'd like to use provided training and inference scripts or play with the examples, you must install the library from source. Clone this repository and install `d2dmoe` with the following commands:
```
git clone https://github.com/mpiorczynski/d2dmoe.git
cd d2dmoe
pip install -e .
```

## Usage
To perform sparsity enforcement you need to execute:
```
# create D2DMoE for the task of sequence classification from the BERT-base checkpoint
from transformers import BertForSequenceClassification
from d2dmoe.train import SparsityEnforecementTrainer

bert = BertForSequenceClassification.from_pretrained(
    model_name_or_path="mpiorczynski/relu-bert-base-uncased",
  	...
)

sparsity_trainer = SparsityEnforecementTrainer(
    model=bert,
    args=training_args,
    output_dir='checkpoints/sparsified_bert_base',
    sparsity_enforcement_weight=1e-3,
    ...
)
sparsity_trainer.train()
sparsity_trainer.save_model()
```
To convert sparsified model into its MoE version you need to execute:
```
from d2dmoe.models import MoEBertConfig, D2DMoEBertForSequenceClassification
from d2dmoe.train import D2DMoERoutersTrainer

config = MoEBertConfig.from_pretrained(
    'checkpoints/sparsified_bert_base',
    num_experts=128,
    expert_split=True,
    ...
)
d2dmoe = D2DMoEBertForSequenceClassification.from_pretrained(
    model_name_or_path='checkpoints/sparsified_bert_base',
    config=config,
    ...
)

routers_trainer = D2DMoERoutersTrainer(
    model=d2dmoe,
    output_dir='checkpoints/d2dmoe_bert_base',
    num_train_epochs=10,
    learning_rate=0.001,
    per_device_train_batch_size=64,
    tau_to_eval=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
  	...
)
routers_trainer.train()
routers_trainer.save_model()
```

For complete examples of usage see `examples` directory.


## Supported models

As D2DMoE relies on the activation sparsity phenomenon exhibited in ReLU-based Transformer models, we have provided pretrained models with the ReLU activation function.
We derived checkpoints listed below by replacing the GELU with ReLU activation function and continuing pretraining through several iterations to adapt them to the change of the activation function.

| Model     | URL                                                           |
|-----------|---------------------------------------------------------------|
| ViT-B     | [mpiorczynski/relu-vit-base-patch16-224](https://huggingface.co/mpiorczynski/relu-vit-base-patch16-224) |
| BERT-base | [mpiorczynski/relu-bert-base-uncased](https://huggingface.co/mpiorczynski/relu-bert-base-uncased)    |

## Dev
You need to have Python 3.10 and pip.
To install dependencies you need to execute:
```
make install-dependencies
```
To format and check code with linter run:
```
make prepare-code
```
Run tests using following:
```
make test
```
To build the package and upload it to the Python Package Index (PyPI) run:
```
make build-release:
```

## References
[1] Zhang, Zhengyan, et al. "Moefication: Transformer feed-forward layers are mixtures of experts." arXiv preprint arXiv:2110.01786 (2021). \
[2] Piórczyński, Mikołaj, et al. "Exploiting Transformer Activation Sparsity with Dynamic Inference." arXiv preprint arXiv:2310.04361 (2023).