import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class PointCloudDataset(Dataset):
    def __init__(self, data, labels):
        """
        data: list of tensors, each tensor shape (num_points, 4) for (x, y, z, t)
        labels: list of integers, class labels
        """
        self.data = data  # 每个样本为形状 (num_points, 4) 的点云张量
        self.labels = labels  # 标签，整数

    def __len__(self):
        return len(self.data)  # 返回数据集的大小

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # 返回样本和对应的标签


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

    # Pad sequences to the same length
    padded_data = pad_sequence(data, batch_first=True, padding_value=0)  # (batch_size, max_seq_len, 4)

    # Create mask: 1 for valid data, 0 for padded data
    mask = (padded_data.sum(dim=2) != 0).float()  # (batch_size, max_seq_len)

    # Convert labels to tensor
    labels = torch.tensor(labels)  # (batch_size)

    return padded_data, labels, mask


def get_data_loader(data, labels, batch_size=32, shuffle=True):
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

