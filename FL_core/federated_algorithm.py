from copy import deepcopy
from collections import OrderedDict
import torch
import numpy as np


class FederatedAlgorithm:
    def __init__(self, train_sizes, init_model):
        #这个属性是一个整数列表，表示每个客户端的训练集大小
        self.train_sizes = train_sizes
        '''
        init_model是初始模型，其类型可能是OrderedDict或nn.Module。
        如果init_model是OrderedDict类型，则param_keys等于模型的键。
        如果它是nn.Module类型，则使用cpu()方法将模型放置在CPU上，并获取模型的状态字典的键。
        '''
        if type(init_model) == OrderedDict:
            self.param_keys = init_model.keys()
        else:
            self.param_keys = init_model.cpu().state_dict().keys()

    def update(self, local_models, client_indices, global_model=None):
        pass



class FedAvg(FederatedAlgorithm):
    def __init__(self, train_sizes, init_model):
        super().__init__(train_sizes, init_model)

    '''
    根据客户端本地模型的权重计算全局模型的更新值
    local_models: 客户端本地模型列表，其中每个元素是一个PyTorch模型
    client_indices: 客户端编号列表，其中每个元素是一个整数，表示该客户端的编号；
    global_model: 全局模型
    '''
    def update(self, local_models, client_indices, global_model=None):
        #计算被选中的客户端的训练数据总量
        num_training_data = sum([self.train_sizes[index] for index in client_indices])
        '''
        创建了一个有序字典对象 update_model，它是一个空的字典，用于存储模型参数的更新结果。
        在此之后，update_model 可以通过 .update() 方法
        或索引操作（update_model[key] = value）向其中添加键值对，以便存储模型参数的更新结果。
        '''
        update_model = OrderedDict()
        #遍历所有被选中的客户端
        for index in range(len(client_indices)):
            '''
            第index个客户端的本地模型。cpu()方法将模型的所有参数移动到CPU上，以便进行下一步的计算和处理。
            state_dict()方法返回模型的参数字典，它将模型中的每个参数映射到其相应的权重。
            因此，local_model是一个包含客户端本地模型权重的字典。
            '''
            local_model = local_models[index].cpu().state_dict()
            #选择的第index个客户端的训练集大小
            num_local_data = self.train_sizes[client_indices[index]]
            #计算加权权重
            weight = num_local_data / num_training_data
            #根据本地模型的权重对全局模型进行更新
            '''
            local_model 是从一个客户端收到的本地模型的权重，它是一个由 state_dict 组成的字典。
            param_keys 是由全局模型的 state_dict 中的所有键组成的列表。
            update_model 是一个 OrderedDict 对象，存储了要更新的全局模型的权重。
            '''
            for k in self.param_keys:
                if index == 0:
                    update_model[k] = weight * local_model[k]
                else:
                    update_model[k] += weight * local_model[k]
        return update_model


class FedAdam(FederatedAlgorithm):
    def __init__(self, train_sizes, init_model, args):
        super().__init__(train_sizes, init_model)
        self.beta_1 = args.beta_1  # 0.9
        self.beta_2 = args.beta_2  # 0.999
        self.epsilon = args.epsilon  # 1e-8
        self.lr_global = args.lr_global
        self.m, self.v = OrderedDict(), OrderedDict()
        for k in self.param_keys:
            self.m[k], self.v[k] = 0., 0.

    def update(self, local_models, client_indices, global_model):
        num_training_data = sum([self.train_sizes[index] for index in client_indices])
        gradient_update = OrderedDict()
        for index in range(len(local_models)):
            local_model = local_models[index].cpu().state_dict()
            num_local_data = self.train_sizes[client_indices[index]]
            weight = num_local_data / num_training_data
            for k in self.param_keys:
                if index == 0:
                    gradient_update[k] = weight * local_model[k]
                else:
                    gradient_update[k] += weight * local_model[k]
                torch.cuda.empty_cache()

        global_model = global_model.cpu().state_dict()
        update_model = OrderedDict()
        for k in self.param_keys:
            g = gradient_update[k]
            self.m[k] = self.beta_1 * self.m[k] + (1 - self.beta_1) * g
            self.v[k] = self.beta_2 * self.v[k] + (1 - self.beta_2) * torch.mul(g, g)
            m_hat = self.m[k] / (1 - self.beta_1)
            v_hat = self.v[k] / (1 - self.beta_2)
            update_model[k] = global_model[k] - self.lr_global * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return update_model