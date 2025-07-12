import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import gc


# 1. 数据处理模块
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
    # Pad sequences
    padded_data = pad_sequence(data, batch_first=True, padding_value=0)
    # Create mask
    mask = (padded_data.sum(dim=2) != 0).float()
    labels = torch.tensor(labels)
    return padded_data, labels, mask


# 2. 内存管理模块
def manage_memory():
    """
    Manually manage memory by deleting unnecessary variables and collecting garbage.
    """
    gc.collect()


def manage_gpu_memory():
    """
    Clear GPU cache to free up memory.
    """
    torch.cuda.empty_cache()


# 3. 教师数学模型模块
class TeacherMathModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # 简单的聚类中心（示例）
        self.cluster_centers = torch.randn(num_classes, 4).cuda()  # 假设有 num_classes 个聚类中心

    def compute_soft_labels(self, data):
        """
        根据数学模型选择最接近的聚类并生成软标签（置信度）。
        data: tensor of shape (batch_size, seq_len, 4)
        Returns:
            soft_labels: tensor of shape (batch_size, num_classes)
        """
        # 计算每个点到每个聚类中心的距离
        # data: (batch_size, seq_len, 4)
        # cluster_centers: (num_classes, 4)
        batch_size, seq_len, _ = data.size()
        # 扩展维度以计算距离
        data_exp = data.unsqueeze(2)  # (batch_size, seq_len, 1, 4)
        centers_exp = self.cluster_centers.unsqueeze(0).unsqueeze(0)  # (1, 1, num_classes, 4)
        distances = torch.norm(data_exp - centers_exp, dim=3)  # (batch_size, seq_len, num_classes)

        # 选择最近的聚类中心
        min_distances, _ = torch.min(distances, dim=2)  # (batch_size, seq_len)

        # 生成每个类的置信度（示例：根据距离的反比）
        confidences = 1 / (distances + 1e-5)  # 避免除零
        soft_labels = confidences.sum(dim=1)  # (batch_size, num_classes)
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)  # 归一化为概率分布
        return soft_labels


# 4. 学生网络模块
class StudentLSTMModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=2, num_classes=10):
        super(StudentLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask):
        """
        x: (batch_size, seq_len, input_dim)
        mask: (batch_size, seq_len)
        """
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim)
        # Apply mask
        lstm_out = lstm_out * mask.unsqueeze(-1)  # (batch_size, seq_len, hidden_dim)
        # Aggregate: average over valid time steps
        out = lstm_out.sum(dim=1) / mask.sum(dim=1, keepdim=True)  # (batch_size, hidden_dim)
        out = self.fc(out)  # (batch_size, num_classes)
        return out


class StudentTransformerModel(nn.Module):
    def __init__(self, input_dim=4, num_heads=4, num_layers=2, hidden_dim=128, num_classes=10):
        super(StudentTransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, mask):
        """
        x: (batch_size, seq_len, input_dim)
        mask: (batch_size, seq_len)
        """
        # Prepare for transformer: (seq_len, batch_size, hidden_dim)
        x = self.input_linear(x)  # (batch_size, seq_len, hidden_dim)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)

        # Create key_padding_mask: (batch_size, seq_len), True for padding
        key_padding_mask = (mask == 0)  # (batch_size, seq_len)

        transformer_out = self.transformer(x,
                                           src_key_padding_mask=key_padding_mask)  # (seq_len, batch_size, hidden_dim)

        # Aggregate: average over valid time steps
        transformer_out = transformer_out.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        transformer_out = transformer_out * mask.unsqueeze(-1)  # (batch_size, seq_len, hidden_dim)
        out = transformer_out.sum(dim=1) / mask.sum(dim=1, keepdim=True)  # (batch_size, hidden_dim)

        out = self.fc(out)  # (batch_size, num_classes)
        return out


# 5. 训练模块
def train_student(teacher_model, student_model, data_loader, optimizer, loss_fn, device, num_epochs=10):
    student_model.train()
    teacher_model.cuda().eval()  # Ensure teacher is on GPU and in eval mode

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_data, labels, mask in data_loader:
            # Move data to GPU
            batch_data = batch_data.to(device)  # (batch_size, seq_len, 4)
            labels = labels.to(device)  # (batch_size)
            mask = mask.to(device)  # (batch_size, seq_len)

            # Generate soft labels from teacher
            with torch.no_grad():
                soft_labels = teacher_model.compute_soft_labels(batch_data)  # (batch_size, num_classes)

            # Forward pass through student model
            outputs = student_model(batch_data, mask)  # (batch_size, num_classes)

            # Compute loss
            loss = loss_fn(outputs, soft_labels)
            total_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Memory management
            del batch_data, labels, mask, soft_labels, outputs, loss
            manage_memory()
            manage_gpu_memory()

        avg_loss = total_loss / len(data_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


# 6. 评估模块
def evaluate_student(teacher_model, student_model, data_loader, loss_fn, device):
    student_model.eval()
    teacher_model.cuda().eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, labels, mask in data_loader:
            # Move data to GPU
            batch_data = batch_data.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            # Generate soft labels from teacher
            soft_labels = teacher_model.compute_soft_labels(batch_data)

            # Forward pass through student model
            outputs = student_model(batch_data, mask)

            # Compute loss
            loss = loss_fn(outputs, soft_labels)
            total_loss += loss.item()

            # For accuracy, use hard labels
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Memory management
            del batch_data, labels, mask, soft_labels, outputs, loss
            manage_memory()
            manage_gpu_memory()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    print(f'Evaluation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


# 主函数
def main():
    # 配置参数
    num_classes = 10
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 示例数据生成（实际使用时替换为真实数据加载）
    # 假设有 1000 个样本，每个样本是一个不等长的点云，包含 (x, y, z, t)
    np.random.seed(0)
    torch.manual_seed(0)
    data = []
    labels = []
    for i in range(1000):
        num_points = np.random.randint(50, 150)  # 每个样本有 50 到 150 个点
        point_cloud = torch.randn(num_points, 4)  # 随机生成点云数据
        data.append(point_cloud)
        labels.append(np.random.randint(0, num_classes))  # 随机生成标签

    # 创建数据集和数据加载器
    dataset = PointCloudDataset(data, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 实例化教师数学模型
    teacher_model = TeacherMathModel(num_classes=num_classes)

    # 实例化学生网络（选择 LSTM 或 Transformer）
    # student_model = StudentLSTMModel(input_dim=4, hidden_dim=128, num_layers=2, num_classes=num_classes).to(device)
    student_model = StudentTransformerModel(input_dim=4, num_heads=4, num_layers=2, hidden_dim=128,
                                            num_classes=num_classes).to(device)

    # 定义优化器和损失函数
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    loss_fn = nn.KLDivLoss(reduction='batchmean')  # 适用于软标签

    # 训练学生网络
    print("开始训练...")
    train_student(teacher_model, student_model, data_loader, optimizer, loss_fn, device, num_epochs=num_epochs)

    # 评估学生网络
    print("开始评估...")
    evaluate_student(teacher_model, student_model, data_loader, loss_fn, device)

    # 保存学生模型
    torch.save(student_model.state_dict(), 'student_model.pth')
    print("学生模型已保存。")


if __name__ == "__main__":
    main()
