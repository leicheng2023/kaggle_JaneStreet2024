import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
import torch  # 导入 PyTorch 库


class GRUModel(nn.Module):
    """
    定义一个基于 GRU（门控循环单元）的神经网络模型。
    
    参数:
    - input_num: 输入特征的数量，默认为 30。
    - hidden_num: GRU 隐藏层单元的数量，默认为 60。
    - head_type: 头部类型，可以是 'pretrain' 或 'prediction'，决定输出层的类型。
    - head_dropout: 在输出层中使用的 dropout 概率，默认为 0.0。
    """
    def __init__(
            self,
            input_num: int = 30,
            hidden_num: int = 60,
            head_type: str = 'prediction',
            head_dropout: int = 0.0):
        super().__init__()

        # 确保 head_type 参数值只能是 'pretrain' 或 'prediction'
        assert head_type in ['pretrain', 'prediction'], 'head type should be either pretrain or prediction'

        # 定义一个 GRU 层，输入维度是 input_num，隐藏层维度是 hidden_num，序列的第一个维度是 batch
        self.gru = nn.GRU(input_num, hidden_num, batch_first=True, num_layers=1)
        
        # 根据 head_type 参数，选择使用预训练头部（PretrainHead）还是预测头部（PredictionHead）
        if head_type == "pretrain":
            self.head = PretrainHead(hidden_num, input_num, head_dropout)  # 预训练头部
        elif head_type == "prediction":
            self.head = PredictionHead(hidden_num, head_dropout)  # 预测头部

    def forward(self, x1, mask_point=None):  # x1 是输入的张量，mask_point 是一个可选参数
        """
        前向传播函数。
        
        参数:
        - x1: 输入张量，形状为 (batch_size, sequence_length, features)。
        - mask_point: 可选参数，用于指定序列中的哪个时间点进行预测。
        
        返回:
        - output: 模型的输出。
        - _: GRU 隐藏层的输出（可忽略）。
        """
        # 将输入 x1 通过 GRU 层进行处理，返回隐藏层输出 l_x1
        l_x1, _ = self.gru(x1)  # 例如输入形状为 (5000, 40, 30)，输出形状与输入相同
        
        # 将 GRU 的输出传入头部，得到最终输出
        output = self.head(l_x1, mask_point)
        
        return output, _  # 返回输出和 GRU 的隐藏状态


class PretrainHead(nn.Module):
    """
    预训练头部，用于重构任务。
    
    参数:
    - hidden_num: 隐藏层单元的数量。
    - feature_num: 输入特征的数量（即模型输出的目标特征数）。
    - dropout: Dropout 层的丢弃概率。
    """
    def __init__(self, hidden_num, feature_num, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # 定义 Dropout 层
        self.linear = nn.Linear(hidden_num, feature_num)  # 定义全连接层，将 hidden_num 映射到 feature_num

    def forward(self, x):
        """
        前向传播函数。
        
        参数:
        - x: 输入张量，形状为 (batch_size, sequence_length, hidden_num)。
        
        返回:
        - 输出张量，形状为 (batch_size, sequence_length, feature_num)。
        """
        x = self.linear(self.dropout(x))  # 先通过 dropout 再通过线性层
        return x


class PredictionHead(nn.Module):
    """
    预测头部，用于最终的分类或回归任务。
    
    参数:
    - hidden_num: GRU 隐藏层单元的数量。
    - dropout: Dropout 层的丢弃概率。
    """
    def __init__(self, hidden_num, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)  # 定义 Dropout 层
        self.hidden = nn.BatchNorm1d(hidden_num)  # 使用 Batch Normalization 进行归一化
        self.linear = nn.Linear(hidden_num, 1)  # 定义全连接层，将 hidden_num 映射到 1

    def forward(self, x, mask_point=None):
        """
        前向传播函数。
        
        参数:
        - x: 输入张量，形状为 (batch_size, sequence_length, hidden_num)。
        - mask_point: 可选参数，指定预测位置，形状为 (batch_size)。
        
        返回:
        - 输出张量，形状为 (batch_size)，即预测结果。
        """
        x = self.dropout(x)  # 先通过 Dropout 层
        
        if mask_point is None:
            # 如果没有提供 mask_point，则使用序列的最后一个时间步进行预测
            output = self.linear(self.hidden(x[:, -1, :]))
        else:
            # 如果提供了 mask_point，按照指定位置进行预测
            bs = x.shape[0]  # 获取 batch size
            output = self.linear(self.hidden(x[torch.arange(0, bs).long(), mask_point.long(), :]))
        
        return output
