import random

from datasets import Dataset
from PIL import ImageFilter, ImageOps
from torchvision import transforms

from d2dmoe.utils import inverse_dict


class GaussianBlur:
    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(self.radius_min, self.radius_max)))
        return img


class Solarization:
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GrayScale:
    def __init__(self, p=0.2):
        self.p = p
        self.transform = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transform(img)
        else:
            return img


def torchvision_to_hf_dataset(torchvision_dataset):
    idx_to_class = inverse_dict(torchvision_dataset.class_to_idx)

    def data_generator():
        for idx in range(len(torchvision_dataset)):
            yield {"pixel_values": torchvision_dataset[idx][0], "labels": idx_to_class[torchvision_dataset[idx][1]]}

    hf_dataset = Dataset.from_generator(data_generator)
    hf_dataset.set_format(type="torch", columns=["pixel_values", "labels"])
    hf_dataset = hf_dataset.class_encode_column("labels")
    return hf_dataset
