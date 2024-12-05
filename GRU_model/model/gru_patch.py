import torch.nn as nn
import torch
import math

# 定义GRU Patch模型
class GRUPatchModel(nn.Module):
    def __init__(
            self,
            input_dim=6,  # 输入特征维度
            hidden_dim=30,  # GRU隐藏层维度
            N_patch=5,  # 每个patch的大小
            step=1,  # 采样步长
            seq_len=40,  # 输入序列的长度
            patch_axis=0):  # patch的轴
        super().__init__()
        assert patch_axis in [0, 1], 'patch should be either 0 or 1'  # 确保patch轴是0或1
        self.N_patch = N_patch  # 保存patch大小
        self.step = step  # 保存步长
        self.patch_num = math.floor((seq_len - N_patch) / step) + 1  # 计算patch的数量
        self.input_dim = input_dim  # 保存输入维度
        self.patch_axis = patch_axis  # 保存patch轴
        self.gru = nn.GRU(N_patch, hidden_dim, batch_first=True, num_layers=1)  # 初始化GRU层
        self.flatten = Flatten_Head(num_feature=input_dim, nf=hidden_dim)  # 初始化Flatten头

    def forward(self, x1):  # 前向传播
        # x1的形状为 (5000, 40, 6)，需要转置
        x1 = x1.permute(0, 2, 1).contiguous()  # 变换为 (5000, 6, 40)

        # 创建patch
        x1 = x1.unfold(dimension=-1, size=self.N_patch, step=self.step)  # 变换为 (5000, 6, 8, 5)
        if self.patch_axis == 1:
            x1 = x1.permute(0, 1, 3, 2).contiguous()  # 如果patch_axis是1，调整维度顺序

        # 调整形状以适配GRU输入
        x1 = torch.reshape(x1, (x1.shape[0] * x1.shape[1], x1.shape[2], x1.shape[3]))  # 变换为 (5000*6, 8, 5)

        # 通过GRU层
        x1, _ = self.gru(x1)  # 输出形状为 (5000*6, 8, 30)

        # 重新调整形状
        x1 = torch.reshape(x1, (-1, self.input_dim, x1.shape[1], x1.shape[2]))  # 变换为 (5000, 6, 8, 30)

        # 扁平化输出
        return self.flatten(x1), 0  # 返回扁平化后的结果和一个0（通常用于损失计算）

# 定义扁平化头
class Flatten_Head(nn.Module):
    def __init__(self, num_feature: int, nf: int, head_dropout: int = 0.1, last_day: bool = True):
        super().__init__()
        self.num_feature = num_feature  # 特征数量
        self.last_day = last_day  # 是否使用最后一天的输出

        self.linears = nn.ModuleList()  # 线性层列表
        self.dropouts = nn.ModuleList()  # Dropout层列表
        self.flattens = nn.ModuleList()  # 扁平化层列表

        # 初始化每个特征的线性层和Dropout层
        for i in range(self.num_feature):
            self.flattens.append(nn.Flatten(start_dim=-2))  # 扁平化层
            self.linears.append(nn.Linear(nf, 1))  # 线性层，输出1维
            self.dropouts.append(nn.Dropout(head_dropout))  # Dropout层

        # 批归一化层
        self.batch0 = nn.BatchNorm1d(nf)  # 针对特征维度的批归一化
        self.batch1 = nn.BatchNorm1d(num_feature)  # 针对特征数量的批归一化

        # 最终输出的线性层
        self.outlinear = nn.Sequential(
            nn.ReLU(),  # ReLU激活函数
            nn.Linear(num_feature, 1))  # 最终线性层

    def forward(self, x):
        x_out = []  # 用于存储每个特征的输出
        for i in range(self.num_feature):
            if self.last_day:
                z = self.linears[i](self.dropouts[i](x[:, i, -1, :]))  # 处理最后一天的输出
            else:
                z = self.flattens[i](x[:, i, :, :])  # 扁平化
                z = self.linears[i](self.dropouts[i](z))  # 线性变换

            x_out.append(z)  # 添加到输出列表

        x = torch.stack(x_out, dim=1)  # 堆叠特征输出，形状为 (5000, 6, 1)
        x = torch.squeeze(x, -1)  # 压缩维度，变为 (5000, 6)
        return self.outlinear(self.batch1(x))  # 通过批归一化和线性层，最终输出形状为 (5000, 1)
