import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timesfm
from sklearn.model_selection import train_test_split

# 设置计算精度
torch.set_float32_matmul_precision("high")
patch_len = 32

# 计算设备
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    device_count = torch.cuda.device_count()
else:
    device = torch.device("cpu")
    device_count = 1

# --------------------------
# 1. 定义带分类头的TimesFM模型
# --------------------------
class TimesFMAnomalyClassifier(nn.Module):
    def __init__(self, base_model): # base_model选用timeFM模型
        super().__init__()
        # 保留TimesFM的特征提取部分（预训练模型）
        self.base_model = base_model.model
        # 冻结预训练权重（可选，根据数据量决定是否微调）
        for param in self.base_model.parameters():
            param.requires_grad = False  # 若数据量大，可设为True进行微调
        
        # 分需根据base_model的输出特征维度调整（这里假设特征维度为256，需根据实际情况修改）类头：输入为TimesFM的特征维度，输出二分类概率（异常/正常）
        # 
        self.classifier_head = nn.Sequential(
            nn.Linear(in_features=1280*4, out_features=128),  # 中间层
            nn.ReLU(),
            nn.Dropout(0.3),  # 防止过拟合
            nn.Linear(in_features=128, out_features=1)  # 输出层（sigmoid前的logits）
        ).to(device)

    def forward(self, inputs, masks):
        # 1. 用TimesFM提取特征（需根据原模型输出格式调整）
        # 假设base_model的forward返回序列的最后一个时间步的特征（需结合原模型结构确认）
        # 若原模型输出为预测序列，可改为提取中间隐藏层特征（需查看TimesFM源码）
        with torch.no_grad():  # 若冻结权重，可不计算梯度
            # 这里需要根据TimesFM的实际输出格式调整特征提取方式
            # 示例：假设输入x是(batch_size, seq_len)，base_model返回特征向量
            input_embeddings, output_embeddings = self.base_model.extract_features(inputs, masks)  # 假设存在提取特征的方法
        
        # print(output_embeddings.shape)
        output_embeddings = torch.reshape(output_embeddings, (output_embeddings.shape[0], -1))
        # print(output_embeddings.shape)
        # 2. 分类头预测
        logits = self.classifier_head(output_embeddings)  # (batch_size, 1)
        # print(logits.shape)
        # exit()
        return logits  # 后续用sigmoid转为概率


# --------------------------
# 2. 加载预训练模型并初始化分类器
# --------------------------
# 加载TimesFM预训练模型（使用原预测配置，仅用于特征提取）
base_model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("/home/mist/my_projs/timesfm/pretrain_model/timesfm-2.5-200m-pytorch", torch_compile=True)
base_model.compile(
    timesfm.ForecastConfig(
        max_context=1024,
        max_horizon=256,
        normalize_inputs=True,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
)

# 初始化分类模型
model = TimesFMAnomalyClassifier(base_model).to(device)


# --------------------------
# 3. 构建带标签的数据集（示例）
# --------------------------
class AnomalyDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        sequences: 时间序列列表，每个元素是np.array(seq_len,)
        labels: 标签列表，每个元素是0（正常）或1（异常）
        """
        self.sequences = []
        self.masks = []
        for seq in sequences:
            seq_len = len(seq)
            pad_length = (patch_len - (seq_len%patch_len)) % patch_len
            sequence = np.concatenate([np.zeros(pad_length), seq])
            mask = np.concatenate([np.ones(pad_length, dtype=bool), np.zeros(seq_len, dtype=bool)])
            
            self.sequences.append(torch.tensor(sequence, dtype=torch.float32))
            self.masks.append(torch.tensor(mask, dtype=torch.bool))
        self.labels = [torch.tensor(label, dtype=torch.float32) for label in labels]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.masks[idx], self.labels[idx]


# 生成示例数据（含异常标签）
def generate_sample(seq_len, is_anomaly=False):
    """生成时间序列，最后一个点是否异常"""
    # 正常序列：正弦曲线+噪声
    x = np.linspace(0, 10, seq_len)
    seq = np.sin(x) + np.random.normal(0, 0.1, seq_len)
    if is_anomaly:
        # 异常点：最后一个点偏离正常范围
        seq[-1] += np.random.uniform(3, 5)  # 大幅偏离
    return seq

# 生成1000个样本（800正常，200异常），每个时间序列有100个点
np.random.seed(42)
sequences = []
labels = []
for _ in range(800):
    sequences.append(generate_sample(seq_len=100, is_anomaly=False))
    labels.append(0)
for _ in range(200):
    sequences.append(generate_sample(seq_len=100, is_anomaly=True))
    labels.append(1)

# 划分训练集和测试集
train_seqs, test_seqs, train_labels, test_labels = train_test_split(
    sequences, labels, test_size=0.2, random_state=42
)

# 构建数据加载器
train_dataset = AnomalyDataset(train_seqs, train_labels)
test_dataset = AnomalyDataset(test_seqs, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --------------------------
# 4. 训练模型
# --------------------------
# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵（输入logits）
optimizer = optim.Adam(model.classifier_head.parameters(), lr=1e-3)  # 仅优化分类头（若冻结base_model）

# 训练循环
model.train()
for epoch in range(10):  # 训练10轮
    total_loss = 0.0
    for seqs, masks, labels in train_loader: # 按batch取数
        seqs = seqs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        # 前向传播
        logits = model(seqs, masks).squeeze(1)  # (batch_size,)

        loss = criterion(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")


# --------------------------
# 5. 评估模型
# --------------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for seqs, masks, labels in test_loader:
        seqs = seqs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        logits = model(seqs, masks).squeeze(1)
        preds = (torch.sigmoid(logits) > 0.5).float()  # 概率>0.5视为异常
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")