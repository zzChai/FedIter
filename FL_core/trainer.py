import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
import time

'''
模型训练器,包括对模型进行训练和测试。
模型训练使用交叉熵损失函数，优化器可以是 SGD 或 Adam,本实验用SGD
'''
class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.device = args.device
        self.client_optimizer = args.client_optimizer  #默认是sgd
        self.lr = args.lr_local  #客户端学习率，默认0.1
        self.wdecay = args.wdecay  #权重衰减
        self.momentum = args.momentum
        self.num_epoch = args.num_epoch   # epoch 数
        self.batch_size = args.batch_size # local batch size B


    def get_model(self):
        return self.model

    def set_model(self, model):
        #模型参数的拷贝方式，将一个模型的参数值复制到另一个模型中。
        self.model.load_state_dict(model.cpu().state_dict())

    #用于参与方本地训练
    def train(self, data, tracking=True):
        model = self.model
        #将model移动到设备self.device上
        model = model.to(self.device)
        #设置model为训练模式
        model.train()

        #将交叉熵损失函数实例化并将其移动到指定的设备
        #其常用于多分类问题中，用于衡量模型输出的概率分布与真实标签的差距。
        criterion = nn.CrossEntropyLoss().to(self.device)

        if self.client_optimizer == 'sgd':
            '''
            使用随机梯度下降优化器
            optim.SGD() 函数会接收模型的参数、学习率（lr）、动量（momentum）和权重衰减（weight_decay）等参数。
            其中，学习率用于控制每次参数更新的步长，动量用于控制参数更新的方向，权重衰减用于防止过拟合，
            通过惩罚大的权重来降低模型的复杂度。
            '''
            optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.wdecay)
        else:
            #Adam 优化器，动态调整每个参数的学习率，从而加快收敛速度,但本实验不用
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        #r_train_loss, r_train_acc = 0., 0
        batch_time, batch_count, batch_size, total_time = 0.0, 0, 0, time.time()
        #epoch训练
        #这里的epoch数是自适应调整的
        #遍历从 0 到 self.num_epoch - 1 的整数，并把每个整数赋值给变量 epoch
        for epoch in range(self.num_epoch):
            train_loss, correct, total = 0., 0, 0  #.是表示浮点数的小数点

            #遍历训练数据中的每个 batch
            #input 是一个输入张量，labels 是对应的标签张量，
            for input, labels in data:
                batch_count+=1  #tao_i
                '''
                if(batch_count>=传来的迭代次数)
                    break
                '''
                #将输入和标签移到指定的设备（CPU 或 GPU）上进行计算。
                input, labels = input.to(self.device), labels.to(self.device)
                #每次更新前清空梯度，否则会累加到下一次更新中，导致模型参数不正确
                optimizer.zero_grad()
                #将输入的数据input通过神经网络模型model进行前向传播，得到输出结果output
                output = model(input)
                '''
                计算损失值。通过计算损失值来评估模型在给定数据上的预测能力。
                criterion 是损失函数，而 output 是模型在输入数据上的预测结果，labels 则是与 output 对应的真实标签。
                labels.long() 将标签从 float 类型转换为 long 类型。
                该语句的结果是一个标量（单个值），表示该批数据的平均损失。
                '''
                loss = criterion(output, labels.long())
                #torch.max()函数在output的第一维度上取得最大值和最大值对应的索引，其中_是占位符，表示并不需要最大值本身，
                # 只需要最大值对应的索引。preds是模型预测的类别标签，表示在输入数据上模型预测为哪个类别。
                _, preds = torch.max(output.data, 1)

                #计算每个 epoch 的训练损失
                #loss.detach().cpu().item() 得到当前 batch 的训练损失，
                # 乘以 input.size(0) 得到当前 batch 中的样本数量，这样就得到了当前 batch 的总损失。
                train_loss += loss.detach().cpu().item() * input.size(0)
                '''
                计算模型在当前 batch 上的预测正确数
                preds == labels.data 会返回一个布尔类型的张量，表示每个样本的预测结果是否与真实标签相同，
                然后 torch.sum() 对这个张量进行求和操作，得到当前 batch 中预测正确的样本数。
                最后使用.detach().cpu().data.numpy() 将结果从 tensor 转化为 numpy 数组，
                并累加到 correct 变量中，用于计算整个 epoch 中预测正确的样本数。
                '''
                correct += torch.sum(preds == labels.data).detach().cpu().data.numpy()
                #input.size(0)返回的是当前 mini-batch 的大小，而total是用来记录当前客户端已经处理过的样本总数
                #用于计算模型的准确率
                total += input.size(0)
                batch_size=input.size(0)

                #backward()用于计算梯度， backward()方法之前，需要将之前的梯度清零
                loss.backward()
                #更新模型参数。
                # 在反向传播计算得到梯度之后，调用该方法可以将梯度应用于模型参数。
                # 该方法会根据在构造优化器时传入的参数（如学习率、动量等）自动更新模型参数，
                # 从而使模型的损失函数的值更接近最优值
                optimizer.step()

                #清空cuda内存缓存的函数
                torch.cuda.empty_cache()

            #训练准确率，即预测正确的样本数占总样本数的比例。
            train_acc = correct / total
            #平均损失函数值。
            train_loss = train_loss / total
            #在控制台中输出当前训练的状态信息
            sys.stdout.write('\rEpoch {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(epoch+1,
                                                                                     self.num_epoch,
                                                                                     train_loss, train_acc))

            #r_train_loss=train_loss
            #r_train_acc=train_acc
        batch_time = (time.time()-total_time) / batch_count
        if tracking:
            #在每个 epoch 训练完成后，如果tracking参数为 True，则在控制台打印一个换行符，以便下一行输出
            print()
        #将Trainer类中的self.model属性更新为当前的模型
        self.model = model
        batch_loss=train_loss*batch_size
        #返回训练集的准确率 train_acc 和损失 train_loss
        return train_acc , train_loss, batch_time, batch_loss

    def train_E0(self, data, tracking=True):
        model = self.model
        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss().to(self.device)

        if self.client_optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wdecay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wdecay)

        correct, total = 0, 0
        batch_loss = []
        for input, labels in data:
            input, labels = input.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, labels.long())
            _, preds = torch.max(output.data, 1)

            batch_loss.append(loss * input.size(0)) ##### loss sum
            total += input.size(0).detach().cpu().data.numpy()
            correct += torch.sum(preds == labels.data).detach().cpu().data.numpy()

            torch.cuda.empty_cache()

        train_acc = correct / total
        avg_loss = sum(batch_loss) / total

        avg_loss.backward()
        optimizer.step()

        sys.stdout.write('\rTrainLoss {:.6f} TrainAcc {:.4f}'.format(avg_loss, train_acc))

        if tracking:
            print()
        self.model = model

        return train_acc, avg_loss.detach().cpu()

    def test(self, model, data):
        model = model.to(self.device)
        model.eval()

        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            test_loss, correct, total = 0., 0, 0
            y_true, y_score = np.empty((0)), np.empty((0))
            for input, labels in data:
                input, labels = input.to(self.device), labels.to(self.device)
                output = model(input)
                loss = criterion(output, labels.long())
                _, preds = torch.max(output.data, 1)

                test_loss += loss.detach().cpu().item() * input.size(0)
                correct += torch.sum(preds == labels.data).detach().cpu().data.numpy() #preds.eq(labels).sum()
                total += input.size(0)

                y_true = np.append(y_true, labels.detach().cpu().numpy(), axis=0)
                y_score = np.append(y_score, preds.detach().cpu().numpy(), axis=0)

                torch.cuda.empty_cache()

        try:
            auc = roc_auc_score(y_true, y_score)
        except:
            auc = 0

        if total > 0:
            test_loss /= total
            test_acc = correct / total
        else:
            test_acc = correct
        return test_acc, test_loss, auc
