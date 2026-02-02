# -*- coding: utf-8 -*-
"""
功能: 提供点云数据集与 DataLoader 组装。
输入: 点云张量列表、标签列表、批大小与是否打乱。
输出: PyTorch DataLoader。
运行: 在训练/评估脚本中调用 get_data_loader。
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class PointCloudDataset(Dataset):
    def __init__(self, data, labels):
        """
        data: list of tensors, each tensor shape (num_points, 4) for (x, y, z, t)
        labels: list of integers, class labels
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    """
    Custom collate function to pad sequences and create masks.
    batch: list of tuples (data, label)

    Returns:
        padded_data: tensor of shape (batch_size, max_seq_len, 4)
        labels: tensor of shape (batch_size)
        mask: tensor of shape (batch_size, max_seq_len)
    """
    data, labels = zip(*batch)

    padded_data = pad_sequence(data, batch_first=True, padding_value=0)
    mask = (padded_data.sum(dim=2) != 0).float()
    labels = torch.tensor(labels)

    return padded_data, labels, mask


def get_data_loader(data, labels, batch_size, shuffle):
    """
    Returns a DataLoader for the PointCloudDataset with the custom collate function.

    data: list of tensors, each tensor shape (num_points, 4) for (x, y, z, t)
    labels: list of integers, class labels
    batch_size: size of each batch
    shuffle: whether to shuffle the data

    Returns:
        DataLoader object
    """
    dataset = PointCloudDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    return data_loader

