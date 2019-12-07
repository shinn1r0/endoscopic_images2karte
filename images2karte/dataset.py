from random import sample

import accimage
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset


class Images2KarteTrainDataset(Dataset):
    """
    Images to Karte Dataset
    """
    def __init__(self, data_dict, transform=None, limit=None):
        """
        Args:
            dataset_file (str):
            transform (callable): torchvision transform
            limit (int):
        """
        self.image_paths = data_dict["image_paths"]
        self.labels_list = data_dict["label"]
        self.image_path = list()
        self.new_labels_list = list()
        for each_image_paths, each_labels in zip(self.image_paths, self.labels_list):
            for each_image_path in each_image_paths:
                self.image_path.append(each_image_path)
                self.new_labels_list.append(each_labels)

        self.transform = transform

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        image = accimage.Image(self.image_path[idx])
        labels = self.new_labels_list[idx]

        if self.transform:
            image = self.transform(image)

        return image, labels


class Images2KarteTestDataset(Dataset):
    """
    Images to Karte Dataset
    """
    def __init__(self, data_dict, transform=None, limit=None):
        """
        Args:
            dataset_file (str):
            transform (callable): torchvision transform
            limit (int):
        """
        self.image_paths = data_dict["image_paths"]
        self.labels_list = data_dict["label"]

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path_list = self.image_paths[idx]
        image_list = [accimage.Image(x) for x in image_path_list]
        if self.transform:
            image_list = [self.transform(img) for img in image_list]
        labels = self.labels_list[idx]

        return image_list, labels


class ImagesSeq2KarteDataset(Dataset):
    """
    Images to Karte Dataset
    """
    def __init__(self, dataset_file, transform=None, transform_3d=None,
                 image_num_limit=None, limit=None, channel_first=False):
        """
        Args:
            dataset_file (str):
            transform (callable): torchvision transform
            transform_3d (callable): torchvision transform
            image_num_limit (int):
            limit (int):
            channel_first (bool):
        """
        with open(dataset_file, mode="rb") as f:
            data_dict = joblib.load(f)
        self.image_paths = data_dict["image_paths"]
        self.labels_list = data_dict["label"]

        self.transform = transform
        self.transform_3d = transform_3d

        self.max_image_num = self.get_max_image_num()
        self.image_num_limit = image_num_limit
        self.channel_first = channel_first

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.image_num_limit is not None and len(self.image_paths[idx]) > self.image_num_limit:
            image_path_list = sorted(sample(self.image_paths[idx], self.image_num_limit))
        else:
            image_path_list = self.image_paths[idx]
        image_list = [accimage.Image(x) for x in image_path_list]
        if self.transform:
            image_list = [self.transform(img) for img in image_list]
        if self.channel_first:
            image_tensor = torch.stack(image_list, dim=1)
        else:
            image_tensor = torch.stack(image_list, dim=0)
        labels = self.labels_list[idx]

        if self.transform_3d:
            image_tensor = self.transform_3d(image_tensor)

        if self.channel_first:
            channel_num, image_num, image_height, image_width = image_tensor.size()
            if self.image_num_limit is not None:
                input_tensor = torch.zeros(channel_num, self.image_num_limit, image_height, image_width)
            else:
                input_tensor = torch.zeros(channel_num, self.max_image_num, image_height, image_width)
            input_tensor[:, :image_num, :, :] = image_tensor
        else:
            image_num, channel_num, image_height, image_width = image_tensor.size()
            if self.image_num_limit is not None:
                input_tensor = torch.zeros(self.image_num_limit, channel_num, image_height, image_width)
            else:
                input_tensor = torch.zeros(self.max_image_num, channel_num, image_height, image_width)
            input_tensor[:image_num, :, :, :] = image_tensor

        return input_tensor, labels

    def get_max_image_num(self):
        max_image_num = max([len(x) for x in self.image_paths])
        return max_image_num
