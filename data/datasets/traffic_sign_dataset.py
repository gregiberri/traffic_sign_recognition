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


def make_transform_composition(transformation_list):
    """
    Just casually iterating through the transformations from the transform list, load them and make the composition.
    :param transformation_list: list containing lists of the function names and parameters: [['function1', [function1_params], ['function2', [function2_params]. ...]
    :return: transform
    """
    compose_list = []
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
        self.transforms = make_transform_composition(self.config.transforms)

    def __getitem__(self, item):
        path, label = self.paths[item], self.labels[item]
        with Image.open(path) as im:
            input_image = np.array(im, dtype=np.float32) / 255

        # make the input tensors
        input_image = np.transpose(input_image, [2, 0, 1])
        torch_input_image = torch.as_tensor(input_image, dtype=torch.float32, device=DEVICE)
        torch_label = torch.as_tensor(int(label), dtype=torch.long, device=DEVICE)
        # torch_one_hot_labels = one_hot(torch_label, num_classes=self.config.num_classes)

        # run the transform on the image
        torch_input_image = self.transforms(torch_input_image)

        return {'paths': path, 'input_images': torch_input_image, 'label_numbers': torch_label}

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
