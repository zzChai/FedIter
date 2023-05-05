import torch
import wandb
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import sys
import multiprocessing as mp

from FL_core.client import Client
from FL_core.trainer import Trainer


#联邦学习服务器
class Server(object):
    def __init__(self, data, init_model, args, selection, fed_algo):
        self.train_data = data['train']['data']   #获取训练数据
        self.train_sizes = data['train']['data_sizes']  #训练数据的大小
        self.test_data = data['test']['data']   #获取测试数据
        self.test_sizes = data['test']['data_sizes']   #测试数据的大小
        '''
        data['test']['data_sizes'] 是一个字典，其键是客户端索引，对应的值是该客户端的测试数据大小。
        因此，data['test']['data_sizes'].keys() 返回了测试集中的所有客户端索引，
        并将这些索引存储在 self.test_clients 列表中。
        '''
        self.test_clients = data['test']['data_sizes'].keys()   #获取测试集中所有客户端的索引
        self.device = args.device #从传入的参数args中获取设备信息
        self.args = args
        self.global_model = init_model  #传入的初始模型赋值给全局模型
        self.selection_method = selection  #保存传入的客户端选择方法，这里是random
        #联邦聚合算法，这里是加权平均
        self.federated_method = fed_algo

        #获取CPU核心数量
        self.nCPU = mp.cpu_count() // 2 if args.nCPU is None else args.nCPU

        self.total_num_client = args.total_num_client  #从参数args中获取客户端总数
        self.num_clients_per_round = args.num_clients_per_round  #从参数args中获取每轮选取的客户端数量，默认10
        self.total_round = args.num_round #从参数args中获取总共的轮数，默认200轮

        self.trainer = Trainer(init_model, args) #创建Trainer实例
        self.test_on_training_data = False #是否在训练集上测试的标志位

        self._init_clients(init_model) #初始化客户端列表


    def _init_clients(self, init_model):
        self.client_list = []
        #遍历所有的客户端,client_index从0到total_num_client-1依次取值
        for client_index in range(self.total_num_client):
            '''
            这行代码是为了初始化客户端对象时，给定该客户端是否包含测试数据。
            它的作用是检查该客户端索引是否在测试集客户端索引列表（self.test_clients）中，
            如果在，则将其对应的测试数据赋值给local_test_data，否则将其设为一个空数组。
            所有客户端都有训练集，但并不是所有客户端都有测试集
            这里的目的是确保测试数据仅在测试集客户端上使用，而不会被训练集客户端使用，
            因为在联邦学习中，测试数据应该是不可用的，只能在模型最终训练完成后用于评估模型的性能。
            '''
            local_test_data = np.array([]) if client_index not in self.test_clients else self.test_data[client_index]
            #deepcopy 用于在不改变原始模型的情况下创建一份深拷贝。
            c = Client(client_index, self.train_data[client_index], local_test_data, deepcopy(init_model), self.args)
            self.client_list.append(c)

    def train(self):
        #训练total_round轮
        for round_index in range(self.total_round):
            print(f'\n>> round {round_index}')
            # get global model
            self.global_model = self.trainer.get_model()
            # 所有客户端的索引
            client_indices = [*range(self.total_num_client)]

            # pre-client-selection，默认随机选取客户端
            if self.args.method == 'Random':
                #随机选到的客户端下标
                client_indices = self.selection_method.select(self.num_clients_per_round)
                print(f'Selected clients: {sorted(client_indices)[:10]}')

            # local training
            local_models, local_losses, accuracy = [], [], []
            #使用多处理，默认false
            if self.args.use_mp:
                iter = 0
                with mp.pool.ThreadPool(processes=self.nCPU) as pool:
                    iter += 1
                    result = list(tqdm(pool.imap(self.local_training, client_indices), desc='>> Local training'))
                    result = np.array(result)
                    local_model, local_loss, local_acc = result[:,0], result[:,1], result[:,2]
                    local_models.extend(local_model.tolist())
                    local_losses.extend(local_loss.tolist())
                    accuracy.extend(local_acc.tolist())
                    sys.stdout.write(
                        '\rClient {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(len(local_losses), len(client_indices),
                                                                                 local_loss.mean().item(),
                                                                                 local_acc.mean().item()))
                print()
            else:
                #tqdm用于在命令行中实现进度条显示，desc参数用于指定进度条前缀，leave参数用于指定是否保留进度条
                #每个参与方进行本地训练
                for client_index in tqdm(client_indices, desc='>> Local training', leave=True):
                    #获得当前client训练后的本地模型、损失值、准确率
                    local_model, local_loss, local_acc = self.local_training(client_index)
                    local_models.append(local_model)
                    local_losses.append(local_loss)
                    accuracy.append(local_acc)
                    sys.stdout.write(
                        '\rClient {}/{} TrainLoss {:.6f} TrainAcc {:.4f}'.format(len(local_losses),
                                                                                 len(client_indices),
                                                                                 local_loss, local_acc))

            # client selection
            '''
                if self.args.method == 'myFL':
                    selected_client_indices = self.selection_method.select(self.num_clients_per_round,
                                                                           local_losses, round_index)
                    local_models = np.take(local_models, selected_client_indices).tolist()
                    client_indices = np.take(client_indices, selected_client_indices).tolist()
                    local_losses = np.take(local_losses, selected_client_indices)
                    accuracy = np.take(accuracy, selected_client_indices)
                    torch.cuda.empty_cache()
                    print(len(selected_client_indices), len(client_indices))
            '''
            #self.weight_variance(local_models)
            '''
            #将每个客户端在本地训练得到的损失和准确率的平均值记录到W&B（Weights & Biases）实验记录平台上
            wandb.log({
                'Train/Loss': sum(local_losses) / len(client_indices),
                'Train/Acc': sum(accuracy) / len(client_indices)
            })
            '''

            print('{} Clients TrainLoss {:.6f} TrainAcc {:.4f}'.format(len(client_indices),
                                                                       sum(local_losses) / len(client_indices),
                                                                       sum(accuracy) / len(client_indices)))

            # 聚合局部模型并更新全局模型
            #简单加权聚合
            global_model_params = self.federated_method.update(local_models, client_indices, self.global_model)
            #模型加载函数。其中self.global_model是要加载模型的实例，global_model_params是已经训练好的模型参数字典，
            #load_state_dict()函数会将已经训练好的参数加载到当前模型的实例中，从而用于后续的推理或者微调训练。
            self.global_model.load_state_dict(global_model_params)
            #将当前模型的参数加载到 Trainer 实例中的模型中
            self.trainer.set_model(self.global_model)

            # test
            if self.test_on_training_data:
                self.test(self.total_num_client, phase='Train')

            self.test_on_training_data = False
            self.test(len(self.test_clients), phase='Test')

            torch.cuda.empty_cache()

    #在每个客户端上进行本地训练的函数
    def local_training(self, client_index):
        #从Client对象数组中拿到当前Client
        client = self.client_list[client_index]
        #传入的全局模型是深度复制的，以防止原模型受到影响
        #返回本地模型、本地损失和本地准确率
        local_model, local_acc, local_loss = client.train(deepcopy(self.global_model), tracking=False)
        #清除 GPU 缓存的函数
        torch.cuda.empty_cache()

        #返回本地训练后的模型深度拷贝、损失、和准确率
        return deepcopy(local_model.cpu()), local_loss, local_acc # / self.train_sizes[client_index]

    def local_testing(self, client_index):
        client = self.client_list[client_index]
        phase = 'train' if self.test_on_training_data else 'test'
        result = client.test(self.global_model, phase)
        torch.cuda.empty_cache()
        return result

    def test(self, num_clients_for_test, phase='Test'):
        metrics = {'loss': [], 'acc': []}
        if self.args.use_mp:
            iter = 0
            with mp.pool.ThreadPool(processes=self.nCPU) as pool:
                iter += 1
                result = pool.map(self.local_testing, [*range(num_clients_for_test)])
                losses, accs = [x['loss'] for x in result], [x['acc'] for x in result]
                metrics['loss'].extend(losses)
                metrics['acc'].extend(accs)
                sys.stdout.write(
                    '\rClient {}/{} {}Loss {:.6f} {}Acc {:.4f}'.format(len(result) * iter, num_clients_for_test,
                                                                       phase, sum(losses) / len(result),
                                                                       phase, sum(accs) / len(result)))
            print()
        else:
            for client_index in tqdm(range(num_clients_for_test), desc=f'>> Local test on {phase} set', leave=True):
                result = self.local_testing(client_index)
                metrics['loss'].append(result['loss'])
                metrics['acc'].append(result['acc'])
                sys.stdout.write(
                    '\rClient {}/{} {}Loss {:.6f} {}Acc {:.4f}'.format(client_index, num_clients_for_test,
                                                                       phase, result['loss'], phase, result['acc']))
        wandb.log({
            f'{phase}/Loss': sum(metrics['loss']) / num_clients_for_test,
            f'{phase}/Acc': sum(metrics['acc']) / num_clients_for_test
        })
        print('ALL Clients {}Loss {:.6f} {}Acc {:.4f}'.format(phase, sum(metrics['loss']) / num_clients_for_test,
                                                              phase, sum(metrics['acc']) / num_clients_for_test))

    def weight_variance(self, local_models):
        variance = 0
        for k in tqdm(local_models[0].state_dict().keys(), desc='>> compute weight variance'):
            tmp = []
            for local_model_param in local_models:
                tmp.extend(torch.flatten(local_model_param.cpu().state_dict()[k]).tolist())
            variance += torch.var(torch.tensor(tmp), dim=0)
        variance /= len(local_models)
        print('variance of model weights {:.8f}'.format(variance))