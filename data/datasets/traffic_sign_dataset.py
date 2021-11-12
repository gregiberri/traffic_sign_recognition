import csv
import os
from warnings import warn
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from data.utils.split_train_val_test import split_train_val
from utils.device import DEVICE


def make_transform_composition(transformation_list, split):
    """
    Just casually iterating through the transformations from the transform list, load them and make the composition.
    :param transformation_list: list containing lists of the function names and parameters: [['function1', [function1_params], ['function2', [function2_params]. ...]
    :param split:
    :return: transform
    """
    compose_list = []
    if split == 'val':
        transformation_list = transformation_list[-3:]

    for item in transformation_list:
        if hasattr(transforms, item[0]):
            function = getattr(transforms, item[0])
            compose_list.append(function(*item[1]))
        else:
            warn(f'Skipping unknown transform: {item[0]}.')
    return transforms.Compose(compose_list)


class TrafficSignDataloader(data.Dataset):
    """
    Dataloader to load the traffic signs.
    """

    def __init__(self, config, split):
        self.config = config
        self.split = split

        self.paths, self.labels = self.load_labels_and_paths()
        self.transforms = make_transform_composition(self.config.transforms, self.split)

    def __getitem__(self, item):
        path, label = self.paths[item], self.labels[item]

        # make the input tensors
        with Image.open(path) as im:
            torch_input_image = self.transforms(im)
        torch_label = torch.as_tensor(int(label), dtype=torch.long)

        return {'paths': path, 'input_images': torch_input_image, 'labels': torch_label}

    def __len__(self):
        return len(self.labels)

    def load_labels_and_paths(self):
        """
        Load data paths and labels according to the current split (train/val/test)
        from the corresponding csv file in the dataset path.
        During training if the csv file does not exist make one for the 3 splits.
        During eval if the csv file does not exist use the whole dataset for evaluation.

        :return: [paths, labels]: list of the paths and list of the corresponding labels
        """
        split_file_path = os.path.join(self.config.dataset_path, self.split + '.csv')
        if not os.path.exists(split_file_path):
            if self.split == 'train':
                split_train_val(self.config.train_val_split, self.config.dataset_path)
            elif self.split == 'val':
                ...
            else:
                raise ValueError(f'Wrong split: {self.split}')

        with open(split_file_path, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

        return list(zip(*data))
