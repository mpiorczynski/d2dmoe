import pytest


@pytest.mark.parametrize(
    "dataset_name, expected_num_classed",
    [
        ("oxford_iiit_pet", 37),
        ("food101", 101),
        ("rte", 2),
        ("qqp", 2),
        ("qnli", 2),
    ],
)
def test_get_num_classes(dataset_name, expected_num_classed):
    from d2dmoe.data import DATASET_TO_NUM_CLASSES

    assert DATASET_TO_NUM_CLASSES[dataset_name] == expected_num_classed
