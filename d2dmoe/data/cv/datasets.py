import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from d2dmoe.data.cv.utils import GaussianBlur, GrayScale, Solarization, torchvision_to_hf_dataset


def get_mnist(dataset_path):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    train_data = torchvision.datasets.MNIST(dataset_path, train=True, download=True, transform=transform)
    train_eval_data = train_data
    test_data = torchvision.datasets.MNIST(dataset_path, train=False, download=True, transform=transform)
    return (
        torchvision_to_hf_dataset(train_data),
        torchvision_to_hf_dataset(train_eval_data),
        torchvision_to_hf_dataset(test_data),
    )


def get_cifar10(dataset_path, normalization=None):
    if normalization == "0.5":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean, std)
    elif normalization == "skip":
        normalize = transforms.Compose([])
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
        normalize = transforms.Normalize(mean, std)
    transform_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_train = transforms.Compose(
        [
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.1),
        ]
    )
    train_data = torchvision.datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform_train)
    train_eval_data = torchvision.datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform_eval)
    test_data = torchvision.datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform_eval)
    return (
        torchvision_to_hf_dataset(train_data),
        torchvision_to_hf_dataset(train_eval_data),
        torchvision_to_hf_dataset(test_data),
    )

def get_oxford_pets(dataset_path, normalization=None, image_size=None):
    if normalization == "0.5":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean, std)
    elif normalization == "skip":
        normalize = transforms.Compose([])
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean, std)
    img_size = 224 if image_size is None else image_size

    # based on https://arxiv.org/pdf/2204.07118.pdf
    # https://github.com/facebookresearch/deit/blob/main/augment.py
    transform_train = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([GrayScale(p=0.9), Solarization(p=0.9), GaussianBlur(p=0.9)]),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_data = torchvision.datasets.OxfordIIITPet(
        dataset_path, split="trainval", transform=transform_train, download=True
    )
    train_eval_data = torchvision.datasets.OxfordIIITPet(
        dataset_path, split="trainval", transform=transform_eval, download=True
    )
    test_data = torchvision.datasets.OxfordIIITPet(dataset_path, split="test", transform=transform_eval, download=True)
    return (
        torchvision_to_hf_dataset(train_data),
        torchvision_to_hf_dataset(train_eval_data),
        torchvision_to_hf_dataset(test_data),
    )

def get_food101(dataset_path, normalization=None, image_size=None):
    if normalization == "0.5":
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        normalize = transforms.Normalize(mean, std)
    elif normalization == "skip":
        normalize = transforms.Compose([])
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        normalize = transforms.Normalize(mean, std)
    
    img_size = 224 if image_size is None else image_size

    # based on https://arxiv.org/pdf/2204.07118.pdf
    # https://github.com/facebookresearch/deit/blob/main/augment.py
    transform_train = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(img_size, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomChoice([GrayScale(p=0.9), Solarization(p=0.9), GaussianBlur(p=0.9)]),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_eval = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_data = torchvision.datasets.Food101(dataset_path, split="train", transform=transform_train, download=True)
    train_eval_data = torchvision.datasets.Food101(
        dataset_path, split="train", transform=transform_eval, download=True
    )
    test_data = torchvision.datasets.Food101(dataset_path, split="test", transform=transform_eval, download=True)
    return (
        torchvision_to_hf_dataset(train_data),
        torchvision_to_hf_dataset(train_eval_data),
        torchvision_to_hf_dataset(test_data),
    )

DATASET_TO_DATALOADERS = {
    "food101": get_food101,
    "oxford_iiit_pet": get_oxford_pets,
}

DATASET_TO_METRIC_NAME = {
    "food101": "accuracy",
    "oxford_iiit_pet": "accuracy",
}

DATASET_TO_NUM_CLASSES = {
    "food101": 101,
    "oxford_iiit_pet": 37,
}
