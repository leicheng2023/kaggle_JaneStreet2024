import os
import random
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader



def set_seed(seed: int = 0):
    random.seed(seed)  # 设置 Python 内置 random 模块的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置 Python 的哈希随机种子（确保在不同进程间的随机一致性）
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的 CPU 随机种子
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)  # 如果 MPS（苹果芯片的 Metal API 加速）可用，设置其随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 如果 GPU 可用，设置 CUDA 的随机种子
        torch.cuda.manual_seed_all(seed)  # 设置所有可用 GPU 的 CUDA 随机种子
    torch.backends.cudnn.benchmark = False  # 禁用 CUDNN 的自动优化，确保结果可复现
    torch.backends.cudnn.deterministic = True  # 使 CUDNN 使用确定性算法，确保结果的可复现性
    print('Set random seed as {} for pytorch'.format(seed))  # 打印设置的随机种子值

def standardize_label(factor_input):
    factor_output = (factor_input - np.nanmean(factor_input)) / np.nanstd(factor_input) # Z-score标准化
    return factor_output

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x, dim=0)  # 计算真实值的均值
    my = torch.mean(y, dim=0)  # 计算预测值的均值
    xm, ym = x - mx, y - my  # 减去均值，得到去均值后的变量
    r_num = torch.sum(xm * ym)  # 计算协方差的分子部分
    x_square_sum = torch.sum(xm * xm)  # 计算 x 的平方和
    y_square_sum = torch.sum(ym * ym)  # 计算 y 的平方和
    r_den = torch.sqrt(x_square_sum * y_square_sum)  # 计算协方差的分母部分（两个标准差的乘积）
    r = r_num / r_den  # 皮尔逊相关系数
    return torch.mean(r)  # 返回皮尔逊相关系数的均值

def pearson_r_loss(y_true, y_pred):
    return -pearson_r(y_true, y_pred)  # 返回皮尔逊相关系数的负值作为损失函数

def mse_loss(y_true, y_pred):
    """
    计算均方误差损失（MSE）。

    参数:
    - y_true: 真实标签 (Tensor)
    - y_pred: 预测值 (Tensor)

    返回:
    - MSE 损失 (Tensor)
    """
    error = y_pred - y_true     # 计算预测值与真实值的误差
    squared_error = error ** 2     # 计算误差的平方
    mse = torch.mean(squared_error)     # 计算均方误差

    return mse


def load_train_data(x1, y, seq_len=40, step=5):
    """
    加载并处理训练数据，将输入数据分割成合适的样本格式并标准化。

    参数:
    - x1: 输入特征数据，形状为 (样本数, 时间步数, 特征数)。
    - y: 标签数据，形状为 (样本数, 时间步数)。
    - seq_len: 序列长度，用于分割时间步的长度，默认为 40。
    - step: 每次移动的时间步，默认为 5。

    返回:
    - x1_train: 处理后的训练输入数据，形状为 (训练样本数, seq_len, 特征数)。
    - x1_val: 处理后的验证输入数据，形状为 (验证样本数, seq_len, 特征数)。
    - y_train: 处理后的训练标签数据，形状为 (训练样本数, 1)。
    - y_val: 处理后的验证标签数据，形状为 (验证样本数, 1)。
    """
    
    # 初始化 x1_in_sample 和 y_in_sample 用于存储处理后的输入和标签数据
    x1_in_sample = np.zeros(
        (int(x1.shape[0] * x1.shape[1] / step), seq_len, x1.shape[2]))  # 形状为 (总样本数, 序列长度, 特征数)
    y_in_sample = np.zeros((int(y.shape[0] * y.shape[1] / step), 1))  # 标签数据形状为 (总样本数, 1)

    n_sample = 0  # 用于统计有效样本数
    
    # 在时间轴上采样数据
    for j in range(0, y.shape[1] - seq_len + 1, step):
        s_index = n_sample  # 当前时间段开始的样本索引
        
        # 遍历所有股票样本
        for i in range(y.shape[0]):
            # 从第 i 个样本的时间步 j 开始，获取一个长度为 seq_len 的序列
            x1_one = x1[i, j:j + seq_len]
            # 获取该时间段的最后一个时间步的标签
            y_one = y[i, j + seq_len - 1]
            

            # 检查特征是否存在缺失值，或者特征是否全为 0，或者标签缺失，跳过无效样本
            # if (np.isnan(x1_one).any() or (x1_one[-1, :] == 0).any() or np.isnan(y_one).any()):
            if (np.isnan(x1_one).any() or (x1_one[-1, :] == 0).all() or np.isnan(y_one).any()):
                continue
            
            # # 获取最后一个时间步的特征值并广播，用于归一化处理
            # x1_one_last = np.tile(x1_one[-1, :], (x1_one.shape[0], 1))  # 复制最后一个时间步的特征值
            # x1_one = x1_one / x1_one_last  # 将序列特征进行归一化处理（除以最后一个时间步的值）

            # 将归一化后的输入特征和对应的标签保存到样本集中
            x1_in_sample[n_sample, :, :] = x1_one
            y_in_sample[n_sample, :] = y_one
            n_sample += 1  # 增加有效样本数
        
        e_index = n_sample  # 当前时间段结束的样本索引
        
        # 如果没有有效样本，跳过标准化
        if e_index == s_index:
            continue
        
        # 对该时间段的标签数据进行标准化处理
        y_in_sample[s_index:e_index, 0] = standardize_label(y_in_sample[s_index:e_index, 0])

    # 将无效部分去掉，只保留有效样本
    x1_in_sample = x1_in_sample[:n_sample, :]  # 保留前 n_sample 个有效输入样本
    y_in_sample = y_in_sample[:n_sample, :]  # 保留前 n_sample 个有效标签样本

    # 按 90% 的比例划分训练集和验证集
    split = int(y_in_sample.shape[0] * 0.9)  # 划分点
    x1_train = x1_in_sample[:split, :, :]  # 训练集输入数据
    x1_val = x1_in_sample[split:, :, :]  # 验证集输入数据
    y_train = y_in_sample[:split, :]  # 训练集标签数据
    y_val = y_in_sample[split:, :]  # 验证集标签数据

    return x1_train, x1_val, y_train, y_val  # 返回训练集和验证集


def gru_dataloader(df,
                   train_start_date: str,
                   train_end_date: str,
                   test_start_date: str,
                   test_end_date: str,
                   seq_len: int,
                   step: int):
    """
    GRU 模型的数据加载器，处理给定 DataFrame 中的特征和标签并生成训练和测试集。

    参数：
    df : pd.DataFrame
        包含股票特征和标签的数据框，至少包括 'datetime' 和 'stock_code' 列。
    train_start_date : str
        训练数据的开始日期，格式如 'YYYY-MM-DD'。
    train_end_date : str
        训练数据的结束日期，格式如 'YYYY-MM-DD'。
    test_start_date : str
        测试数据的开始日期，格式如 'YYYY-MM-DD'。
    test_end_date : str
        测试数据的结束日期，格式如 'YYYY-MM-DD'。
    seq_len : int
        序列长度，用于构造训练和测试数据的时间步长。
    step : int
        步长，用于生成样本序列时的步长。

    返回：
    x1_train : np.ndarray
        训练数据集的输入特征，经过时间序列处理。
    x1_valid : np.ndarray
        验证数据集的输入特征。
    X_test : np.ndarray
        测试数据集的输入特征。
    y_train : np.ndarray
        训练数据集的标签。
    y_valid : np.ndarray
        验证数据集的标签。
    y_test : np.ndarray
        测试数据集的标签。
    train_date : np.ndarray
        训练集日期列表。
    test_date : np.ndarray
        测试集日期列表。
    sample_stock : np.ndarray
        股票代码列表。

    说明：
    该函数从输入的 DataFrame 中提取训练和测试数据，处理为适合 GRU 模型的格式。
    """

    # 筛选时间范围内的数据
    df = df[(df.datetime >= train_start_date) & (df.datetime <= test_end_date)]
    
    # 获取所有的唯一时间戳和股票代码
    sample_datetime = np.sort(df.datetime.unique())
    sample_stock = np.sort(df.stock_code.unique())
    
    # 获取特征列（从第 3 列到倒数第 2 列）
    features = df.columns[2:-1]
    
    # 获取标签列（最后一列）
    labels = df.columns[[-1]]
    
    # 初始化 X 矩阵，形状为 (股票数量, 日期数量, 特征数量)
    X = np.zeros((len(sample_stock), len(sample_datetime), len(features)))

    # 循环遍历每个特征，并按股票和时间对数据进行透视
    for i, f in enumerate(tqdm(features,colour = 'green')):
        featurei = df.pivot(index='stock_code', columns='datetime', values=f)
        featurei = featurei.sort_index(axis=0).sort_index(axis=1)
        X[:, :, i] = featurei.values  # 将透视结果填充到 X 矩阵中
    
    # 提取标签数据，并按股票和时间进行透视
    label = df.pivot(index='stock_code', columns='datetime', values=labels[-1])
    label = label.sort_index(axis=0).sort_index(axis=1)
    label = label.values  # 将标签转为 numpy 数组

    # 使用 PostNTradingDate 函数计算训练集的结束时间的下一个交易日
    # from higgsboom.FuncUtils.DateTime import PostNTradingDate
    # train_end = np.where(sample_datetime == np.datetime64(PostNTradingDate(train_end_date, 1)))[0][0]
    train_end = np.where(sample_datetime == train_end_date)[0][0]

    
    # 测试集的结束位置
    test_end = int(len(sample_datetime))
    
    # 划分训练集和测试集数据
    X_train, X_test = X[:, :train_end], X[:, train_end - seq_len + 1:test_end]
    y_train, y_test = label[:, :train_end], label[:, train_end - seq_len + 1:test_end]
    
    # 获取训练和测试集日期
    train_date, test_date = sample_datetime[:train_end], sample_datetime[train_end:test_end]

    # 使用 load_train_data 函数构造训练和验证数据集
    x1_train, x1_valid, y_train, y_valid = load_train_data(
        X_train,
        y_train,
        seq_len=seq_len,
        step=step
    )

    # 返回构造好的训练和测试集以及相关的日期和股票代码
    return x1_train, x1_valid, X_test, y_train, y_valid, y_test, train_date, test_date, sample_stock


class Newdataset(Dataset):
    def __init__(self, data1, label) -> None:
        super().__init__()
        self.data1 = data1.astype(np.float32)
        self.label = label.astype(np.float32)

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        return self.data1[index], self.label[index]

def load_test_data(x1, y, seq_len=40):
    """
    该函数用于加载测试数据，并对其进行预处理，返回经过标准化处理的特征和标签。

    参数:
    - x1: np.array, 输入的特征矩阵，维度为 (num_samples, time_steps, num_features)。
    - y: np.array, 标签矩阵，维度为 (num_samples, time_steps)。
    - seq_len: int, 序列长度，默认值为 40，表示取多少个时间步的数据作为输入。

    返回:
    - x1_test: np.array, 标准化后的测试特征矩阵，维度为 (n_sample, seq_len, num_features)。
    - y_test: np.array, 标准化后的测试标签，维度为 (n_sample, 1)。
    - nonan_index: list, 存储非空样本的索引列表。
    """

    # 初始化存储特征和标签的数组
    x1_in_sample = np.zeros(
        (int(x1.shape[0]), seq_len, x1.shape[2]))  # 特征样本，形状为 (样本数, 序列长度, 特征数)
    y_in_sample = np.zeros((int(y.shape[0]), 1))  # 标签样本，形状为 (样本数, 1)
    n_sample = 0  # 记录有效样本数量
    s_index = n_sample  # 开始索引
    nonan_index = []  # 存储没有空值的样本索引

    # 遍历每个样本
    for i in range(y.shape[0]):
        x1_one = x1[i, :]  # 取出第 i 个样本的特征
        y_one = y[i, -1]  # 取出第 i 个样本的最后一个标签

        # 如果当前样本包含 NaN 或特征的最后一个时间步全部为 0 值，跳过该样本
        if (np.isnan(x1_one).any() or (x1_one[-1, :] == 0).all()):
            continue

        nonan_index.append(i)  # 记录没有空值的样本索引

        # # 复制最后一个时间步的特征，用于归一化操作
        # x1_one_last = np.tile(x1_one[-1, :], (x1_one.shape[0], 1))
        # x1_one = x1_one / x1_one_last  # 将每个时间步的特征值归一化为相对于最后一个时间步的比例

        # 将处理后的样本存入对应位置
        x1_in_sample[n_sample, :, :] = x1_one
        y_in_sample[n_sample, :] = y_one
        n_sample += 1

    # 取出当前有效样本的结束索引
    e_index = n_sample

    # 对标签进行标准化处理，将其转换为标准正态分布
    y_in_sample[s_index:e_index, 0] = standardize_label(y_in_sample[s_index:e_index, 0])

    # 去除多余的无效样本，保留有效的样本数据
    x1_in_sample = x1_in_sample[:n_sample, :]
    y_in_sample = y_in_sample[:n_sample, :]

    # 返回预处理后的特征、标签和没有空值的样本索引
    x1_test = x1_in_sample
    y_test = y_in_sample

    return x1_test, y_test, nonan_index


def predict_test_set(X_test, y_test, modeltrainer, test_date, sample_stock,seq_len):
    """
    该函数用于对测试集进行预测，并返回每个股票在每个测试日期的预测值。

    参数:
    - X_test: np.array, 测试集特征矩阵，维度为 (num_samples, time_steps, num_features)。
    - y_test: np.array, 测试集标签矩阵，维度为 (num_samples, time_steps)。
    - modeltrainer: ModelTrainer 对象，训练好的模型，用于预测。
    - test_date: list, 测试日期列表。
    - sample_stock: list, 样本股票代码。
    - seq_len: int,时间步数

    返回:
    - fac_1: pd.DataFrame, 每个股票在每个日期的预测值，行是股票，列是日期。
    """

    # 初始化一个 DataFrame，用于存储预测的因子数据（股票的预测值）
    # 大小为 (测试样本数量, 测试日期长度 - 10)
    fac_1 = pd.DataFrame(np.nan * np.zeros((X_test.shape[0], len(test_date))))

    i_panel = 0  # 用于跟踪列的索引（即时间步的索引）
    
    # 遍历测试日期，使用滚动窗口的方式进行预测
    for i in tqdm(range(len(test_date)),colour='blue'):
        # 使用 i 到 i+seq_len 作为时间序列窗口提取特征和标签数据
        x1_test, y1_test, nonan_index = load_test_data(X_test[:, i:i+seq_len, :], y_test[:, i:i+seq_len],seq_len=seq_len)

        # 创建测试数据集并加载数据
        test_ds = Newdataset(x1_test, y1_test)  # 创建自定义的数据集
        test_dl = DataLoader(test_ds, batch_size=len(x1_test))  # 创建数据加载器

        # 使用模型进行预测
        y_pred = modeltrainer.predict(test_dl)
        
        # 将预测结果存入 fac_1 中，nonan_index 表示有效样本的索引
        fac_1.iloc[nonan_index, i_panel] = y_pred[:, -1]  # 将预测值的最后一个时间步的结果填入

        i_panel += 1  # 更新列的索引

    # 为 fac_1 的列设置日期标签
    fac_1.columns = test_date[:i_panel]
    
    # 为 fac_1 的行设置股票索引
    fac_1.index = sample_stock

    fac_1.rename_axis(index = 'stock_code',columns = 'datetime',inplace = True)

    fac_1 = fac_1.T

    return fac_1
