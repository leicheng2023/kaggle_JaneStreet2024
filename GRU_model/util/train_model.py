import time
from util.function import *  # 导入自定义的函数库
from tqdm import tqdm  # 导入进度条工具
import matplotlib.pyplot as plt  # 导入绘图工具


class ModelTrainer:
    """
    ModelTrainer类用于训练、验证和测试模型，支持早停法，并绘制训练和验证过程中IC的变化曲线。
    """
    def __init__(self,
                 model,  # 传入的神经网络模型
                 optimizer,  # 优化器
                 device,  # 训练使用的设备（如CPU或GPU）
                 name,  # 模型名称，用于保存模型文件时命名
                 early_stop=5,  # 早停参数，当验证集IC多次没有提升时提前终止训练
                 n_epochs=100,  # 最大训练轮次
                 seed=0):  # 随机种子，保证实验结果的可复现性
        self.model = model  # 保存传入的模型
        self.optimizer = optimizer  # 保存传入的优化器
        self.device = device  # 保存训练设备（如'cuda'或'cpu'）
        self.early_stop = early_stop  # 保存早停次数
        self.n_epochs = n_epochs  # 保存最大训练轮次
        self.seed = seed  # 保存随机种子
        self.name = name  # 保存模型名称

        self.model.to(self.device)  # 将模型移动到指定的设备上（如GPU）

    # def _train_epoch(self, train_dl, epoch):
    #     """
    #     执行一个训练轮次。
    #     :param train_dl: 训练数据加载器
    #     :param epoch: 当前训练轮次的编号
    #     :return: 该轮训练的平均IC值
    #     """
    #     ic = 0  # 初始化累计IC（信息系数）值
    #     self.model.train()  # 将模型设置为训练模式
    #     tqdm_ = tqdm(iterable=train_dl)  # 使用tqdm包装训练数据加载器，显示进度条
    #     i = 0
    #     for i, batch in enumerate(tqdm_):  # 遍历每个批次
    #         x1, labels = batch  # 从批次中获取输入数据和标签
    #         x1 = x1.to(self.device)  # 将输入数据移动到指定设备
    #         labels = labels.to(self.device)  # 将标签移动到指定设备
    #         out, _ = self.model(x1)  # 模型前向传播，获取预测结果

    #         loss = pearson_r_loss(labels, out)  # 计算Pearson相关性损失

    #         self.optimizer.zero_grad()  # 清除之前的梯度
    #         loss.backward()  # 反向传播计算梯度
    #         self.optimizer.step()  # 更新模型参数

    #         ic_i = pearson_r(labels, out).item()  # 计算当前批次的IC值
    #         ic += ic_i  # 累计IC值
    #         tqdm_.set_description("epoch:{:d} train IC:{:.4f}".format(epoch, ic / (i + 1)))  # 更新进度条显示的信息
    #     return ic / (i + 1)  # 返回平均IC值

    # def _eval_epoch(self, val_dl, epoch):
    #     """
    #     执行一个验证轮次。
    #     :param val_dl: 验证数据加载器
    #     :param epoch: 当前验证轮次的编号
    #     :return: 该轮验证的平均IC值
    #     """
    #     self.model.eval()  # 将模型设置为评估模式
    #     ic = 0  # 初始化累计IC值
    #     tqdm_ = tqdm(iterable=val_dl)  # 使用tqdm包装验证数据加载器，显示进度条
    #     i = 0
    #     for i, batch in enumerate(tqdm_):  # 遍历每个批次
    #         x1, labels = batch  # 从批次中获取输入数据和标签
    #         x1 = x1.to(self.device)  # 将输入数据移动到指定设备
    #         labels = labels.to(self.device)  # 将标签移动到指定设备
    #         out, _ = self.model(x1)  # 模型前向传播，获取预测结果

    #         ic += pearson_r(labels, out).item()  # 累计每个批次的IC值

    #         tqdm_.set_description(
    #             "epoch:{:d} test IC:{:.4f} ".format(epoch, ic / (i + 1)))  # 更新进度条显示的信息
    #     return ic / (i + 1)  # 返回平均IC值

    # def fit(self, train_dl, val_dl, model_path):
    #     """
    #     训练和验证模型。
    #     :param train_dl: 训练数据加载器
    #     :param val_dl: 验证数据加载器
    #     :param model_path: 模型保存路径
    #     :return: 训练集和验证集的IC值列表
    #     """
    #     print(f'current device: {self.device}')  # 输出当前使用的设备
    #     print(f'begin time: {time.ctime()}')  # 输出当前时间
    #     print(self.model)  # 输出模型结构
    #     set_seed(self.seed)  # 设置随机种子，保证结果可复现

    #     max_ic = -10000  # 初始化最大IC值
    #     max_epoch = 0  # 初始化最优轮次

    #     train_list = []  # 保存每轮训练的IC值
    #     val_list = []  # 保存每轮验证的IC值

    #     epoch = 0
    #     for epoch in range(self.n_epochs):  # 逐轮训练
    #         train_ic = self._train_epoch(train_dl, epoch)  # 训练一轮
    #         ic = self._eval_epoch(val_dl, epoch)  # 验证一轮

    #         train_list.append(train_ic)  # 记录训练IC值
    #         val_list.append(ic)  # 记录验证IC值

    #         if ic > max_ic:  # 如果当前验证IC值优于历史最优值
    #             max_ic = ic  # 更新最大IC值
    #             max_epoch = epoch  # 更新最优轮次
    #             torch.save(self.model, f'{model_path}/saved_model/{self.name}.pt')  # 保存模型
    #         else:
    #             if epoch - max_epoch >= self.early_stop:  # 如果IC没有提升，且超过早停轮数
    #                 break  # 结束训练

    #     # 绘制训练和验证IC变化曲线
    #     fig = plt.figure(figsize=[8, 6])
    #     plt.plot(
    #         np.arange(epoch + 1),
    #         train_list,
    #         label='train_scores'
    #     )
    #     plt.plot(
    #         np.arange(epoch + 1),
    #         val_list,
    #         label='valid_scores'
    #     )
    #     plt.legend()
    #     plt.title(f"scores for {self.name} best IC = {max_ic}")
    #     fig.savefig(f'{model_path}/saved_loss/{self.name}-loss.png')  # 保存IC变化图

    #     return train_list, val_list  # 返回训练集和验证集的IC值列表


    # def fit_rolling(self, train_dl, val_dl, model_path):
    #     """
    #     训练和验证模型。
    #     :param train_dl: 训练数据加载器
    #     :param val_dl: 验证数据加载器
    #     :param model_path: 模型保存路径
    #     :return: 训练集和验证集的IC值列表
    #     """
    #     print(self.name)
    #     set_seed(self.seed)  # 设置随机种子，保证结果可复现

    #     max_ic = -10000  # 初始化最大IC值
    #     max_epoch = 0  # 初始化最优轮次

    #     train_list = []  # 保存每轮训练的IC值
    #     val_list = []  # 保存每轮验证的IC值

    #     epoch = 0
    #     for epoch in range(self.n_epochs):  # 逐轮训练
    #         train_ic = self._train_epoch(train_dl, epoch)  # 训练一轮
    #         ic = self._eval_epoch(val_dl, epoch)  # 验证一轮

    #         train_list.append(train_ic)  # 记录训练IC值
    #         val_list.append(ic)  # 记录验证IC值

    #         if ic > max_ic:  # 如果当前验证IC值优于历史最优值
    #             max_ic = ic  # 更新最大IC值
    #             max_epoch = epoch  # 更新最优轮次
    #             torch.save(self.model, f'{model_path}/saved_model/{self.name}.pt')  # 保存模型
    #         else:
    #             if epoch - max_epoch >= self.early_stop:  # 如果IC没有提升，且超过早停轮数
    #                 break  # 结束训练

    #     # 绘制训练和验证IC变化曲线
    #     fig = plt.figure(figsize=[8, 6])
    #     plt.plot(
    #         np.arange(epoch + 1),
    #         train_list,
    #         label='train_scores'
    #     )
    #     plt.plot(
    #         np.arange(epoch + 1),
    #         val_list,
    #         label='valid_scores'
    #     )
    #     plt.legend()
    #     plt.title(f"scores for {self.name} best IC = {max_ic}")
    #     fig.savefig(f'{model_path}/saved_loss/{self.name}-loss.png')  # 保存IC变化图

    #     return train_list, val_list  # 返回训练集和验证集的IC值列表


    def _train_epoch(self, train_dl, epoch,loss_fun: str = 'ic'):
        """
        执行一个训练轮次。
        :param train_dl: 训练数据加载器
        :param epoch: 当前训练轮次的编号
        :param loss_fun: 损失函数ic或者mse
        :return: 该轮训练的平均loss值
        """
        loss_accum = 0
        self.model.train()  # 将模型设置为训练模式
        tqdm_ = tqdm(iterable=train_dl)  # 使用tqdm包装训练数据加载器，显示进度条
        i = 0
        for i, batch in enumerate(tqdm_):  # 遍历每个批次
            x1, labels = batch  # 从批次中获取输入数据和标签
            x1 = x1.to(self.device)  # 将输入数据移动到指定设备
            labels = labels.to(self.device)  # 将标签移动到指定设备
            result = self.model(x1) # 模型前向传播，获取预测结果
            if isinstance(result,tuple):
                out, _ = result  
            else:
                out = result
            if loss_fun == 'ic':
                loss = pearson_r_loss(labels, out)  # 计算Pearson相关性损失,要保证两者的维度完全一致

                self.optimizer.zero_grad()  # 清除之前的梯度
                loss.backward()  # 反向传播计算梯度
                self.optimizer.step()  # 更新模型参数

                ic_i = pearson_r(labels, out).item()  # 计算当前批次的IC值
                loss_accum += ic_i  # 累计IC值
                tqdm_.set_description("epoch:{:d} train IC:{:.4f}".format(epoch, loss_accum / (i + 1)))  # 更新进度条显示的信息
            elif loss_fun == 'mse':
                loss = mse_loss(labels, out)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mse_i = mse_loss(labels, out).item()
                loss_accum += mse_i
                tqdm_.set_description("epoch:{:d} train mse:{:.4f}".format(epoch, loss_accum / (i + 1)))  # 更新进度条显示的信息
        return loss_accum / (i + 1)  # 返回平均loss值

    def _eval_epoch(self, val_dl, epoch,loss_fun:str='ic'):
        """
        执行一个验证轮次。
        :param val_dl: 验证数据加载器
        :param epoch: 当前验证轮次的编号
        :param loss_fun: 损失函数ic或者mse
        :return: 该轮训练的平均loss值
        """
        self.model.eval()  # 将模型设置为评估模式
        loss_accum = 0  # 初始化累计IC值
        tqdm_ = tqdm(iterable=val_dl)  # 使用tqdm包装验证数据加载器，显示进度条
        i = 0
        for i, batch in enumerate(tqdm_):  # 遍历每个批次
            x1, labels = batch  # 从批次中获取输入数据和标签
            x1 = x1.to(self.device)  # 将输入数据移动到指定设备
            labels = labels.to(self.device)  # 将标签移动到指定设备
            result = self.model(x1) # 模型前向传播，获取预测结果
            if isinstance(result,tuple):
                out, _ = result  
            else:
                out = result
            if loss_fun == 'ic':
                loss_accum += pearson_r(labels, out).item()  # 累计每个批次的IC值

                tqdm_.set_description(
                    "epoch:{:d} test IC:{:.4f} ".format(epoch, loss_accum / (i + 1)))  # 更新进度条显示的信息
            elif loss_fun == 'mse':
                loss_accum += mse_loss(labels, out).item()
                tqdm_.set_description(
                    "epoch:{:d} test mse:{:.4f} ".format(epoch, loss_accum / (i + 1))
                )
        return loss_accum / (i + 1)  # 返回平均IC值

    def fit(self, train_dl, val_dl, model_path,loss_fun:str='ic'):
        """
        训练和验证模型。

        :param train_dl: 训练数据加载器
        :param val_dl: 验证数据加载器
        :param model_path: 模型保存路径
        :param loss_fun: 损失函数ic或者mse
        :return: 训练集和验证集的loss值列表
        """
        print(f'current device: {self.device}')  # 输出当前使用的设备
        print(f'begin time: {time.ctime()}')  # 输出当前时间
        print(self.model)  # 输出模型结构
        set_seed(self.seed)  # 设置随机种子，保证结果可复现

        max_ic = -10000  # 初始化最大IC值
        min_mse = 10000 # 初始化最小mse
        max_epoch = 0  # 初始化最优轮次

        train_list = []  # 保存每轮训练的loss值
        val_list = []  # 保存每轮验证的loss值

        epoch = 0
        for epoch in range(self.n_epochs):  # 逐轮训练
            train_loss = self._train_epoch(train_dl, epoch,loss_fun)  # 训练一轮
            loss = self._eval_epoch(val_dl, epoch,loss_fun)  # 验证一轮

            train_list.append(train_loss)  # 记录训练IC值
            val_list.append(loss)  # 记录验证IC值
            if loss_fun == 'ic':
                if loss > max_ic:  # 如果当前验证IC值优于历史最优值
                    max_ic = loss  # 更新最大IC值
                    max_epoch = epoch  # 更新最优轮次
                    torch.save(self.model, f'{model_path}/saved_model/{self.name}.pt')  # 保存模型
                else:
                    if epoch - max_epoch >= self.early_stop:  # 如果IC没有提升，且超过早停轮数
                        break  # 结束训练
            elif loss_fun == 'mse':
                if loss < min_mse:
                    min_mse = loss
                    max_epoch = epoch
                    torch.save(self.model,f'{model_path}/saved_model/{self.name}.pt')
                else:
                    if epoch - max_epoch >= self.early_stop:
                        break

        # 绘制训练和验证loss变化曲线
        fig = plt.figure(figsize=[8, 6])
        plt.plot(
            np.arange(epoch + 1),
            train_list,
            label='train_scores'
        )
        plt.plot(
            np.arange(epoch + 1),
            val_list,
            label='valid_scores'
        )
        plt.legend()

        if loss_fun == 'ic':
            plt.title(f"scores for {self.name} best {loss_fun} = {max_ic}")
        elif loss_fun == 'mse':
            plt.title(f"scores for {self.name} best {loss_fun} = {min_mse}")

        fig.savefig(f'{model_path}/saved_loss/{self.name}-loss.png')  # 保存IC变化图

        return train_list, val_list  # 返回训练集和验证集的loss值列表


    def fit_rolling(self, train_dl, val_dl, model_path,loss_fun:str='ic'):
        """
        训练和验证模型。
        :param train_dl: 训练数据加载器
        :param val_dl: 验证数据加载器
        :param model_path: 模型保存路径
        :param loss_fun: 损失函数ic或者mse
        :return: 训练集和验证集的loss值列表
        """
        print(self.name)
        set_seed(self.seed)  # 设置随机种子，保证结果可复现

        max_ic = -10000  # 初始化最大IC值
        min_mse = 10000 # 初始化最小mse
        max_epoch = 0  # 初始化最优轮次

        train_list = []  # 保存每轮训练的loss值
        val_list = []  # 保存每轮验证的loss值

        epoch = 0
        for epoch in range(self.n_epochs):  # 逐轮训练
            train_loss = self._train_epoch(train_dl, epoch,loss_fun)  # 训练一轮
            loss = self._eval_epoch(val_dl, epoch,loss_fun)  # 验证一轮

            train_list.append(train_loss)  # 记录训练IC值
            val_list.append(loss)  # 记录验证IC值
            if loss_fun == 'ic':
                if loss > max_ic:  # 如果当前验证IC值优于历史最优值
                    max_ic = loss  # 更新最大IC值
                    max_epoch = epoch  # 更新最优轮次
                    torch.save(self.model, f'{model_path}/saved_model/{self.name}.pt')  # 保存模型
                else:
                    if epoch - max_epoch >= self.early_stop:  # 如果IC没有提升，且超过早停轮数
                        break  # 结束训练
            elif loss_fun == 'mse':
                if loss < min_mse:
                    min_mse = loss
                    max_epoch = epoch
                    torch.save(self.model,f'{model_path}/saved_model/{self.name}.pt')
                else:
                    if epoch - max_epoch >= self.early_stop:
                        break

        # 绘制训练和验证loss变化曲线
        fig = plt.figure(figsize=[8, 6])
        plt.plot(
            np.arange(epoch + 1),
            train_list,
            label='train_scores'
        )
        plt.plot(
            np.arange(epoch + 1),
            val_list,
            label='valid_scores'
        )
        plt.legend()

        if loss_fun == 'ic':
            plt.title(f"scores for {self.name} best {loss_fun} = {max_ic}")
        elif loss_fun == 'mse':
            plt.title(f"scores for {self.name} best {loss_fun} = {min_mse}")

        fig.savefig(f'{model_path}/saved_loss/{self.name}-loss.png')  # 保存IC变化图

        return train_list, val_list  # 返回训练集和验证集的loss值列表


    def predict(self, test_dl):
        """
        使用训练好的模型进行预测。
        :param test_dl: 测试数据加载器
        :return: 预测结果
        """
        x1, labels = next(iter(test_dl))  # 获取一个批次的输入数据和标签
        x1 = x1.to(self.device)  # 将输入数据移动到指定设备
        self.model.eval()  # 将模型设置为评估模式
        y_pred, _ = self.model(x1)  # 模型前向传播，获取预测结果
        y_pred = y_pred.cpu().detach().numpy()  # 将预测结果从设备移到CPU，并转换为numpy格式

        return y_pred  # 返回预测结果