import pytest
from train.conftest import *


@pytest.fixture
def model_name_or_path():
    return 'mpiorczynski/relu-bert-base-uncased'

@pytest.fixture
def num_experts():
    return 2

@pytest.fixture
def num_labels():
    return 2

@pytest.fixture
def text():
    return "Replace me by any text you'd like."
