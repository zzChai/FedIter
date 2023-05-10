
import os
import wandb
import argparse
import torch

from data import reddit, federated_emnist
from model import BLSTM, CNN
from FL_core.server import *
from FL_core.client_selection import *
from FL_core.federated_algorithm import *



def get_args():
    #argparse.ArgumentParser()是argparse模块中的一个类，用于创建一个解析器对象，
    # 可以通过该对象来定义程序需要接受的命令行参数、选项和参数类型等信息。
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu cuda index')
    parser.add_argument('--dataset', type=str, default='FederatedEMNIST', help='dataset', choices=['Reddit','FederatedEMNIST'])
    parser.add_argument('--data_dir', type=str, default='E:/code/my-Federated-Learning/dataset/FederatedEMNIST/', help='dataset directory')
    parser.add_argument('--model', type=str, default='CNN', help='model', choices=['BLSTM','CNN'])
    parser.add_argument('--method', type=str, default='Random', choices=['Random', 'myFL'],
                        help='client selection')
    #用于聚合的联邦算法
    parser.add_argument('--fed_algo', type=str, default='FedAvg', choices=['FedAvg', 'FedAdam'],
                        help='Federated algorithm for aggregation')

    parser.add_argument('--client_optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='client optim')
    parser.add_argument('--lr_local', type=float, default=0.1, help='learning rate for client optim')
    parser.add_argument('--lr_global', type=float, default=0.001, help='learning rate for server optim')
    parser.add_argument('--wdecay', type=float, default=0, help='weight decay for optim')

    parser.add_argument('--momentum', type=float, default=0, help='momentum for SGD')

    parser.add_argument('--beta_1', type=float, default=0.9, help='beta_1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.999, help='beta_2 for Adam')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for Adam')

    parser.add_argument('--alpha_1', type=float, default=0.75, help='alpha_1 for myFL')
    parser.add_argument('--alpha_2', type=float, default=0.01, help='alpha_2 for myFL')
    parser.add_argument('--alpha_3', type=float, default=0.6, help='alpha_3 for myFL') # 重要超参，使用loss来选择客户端的比例

    parser.add_argument('-E', '--num_epoch', type=int, default=1, help='number of epochs')
    parser.add_argument('-B', '--batch_size', type=int, default=64, help='batch size of each client data')
    parser.add_argument('-R', '--num_round', type=int, default=200, help='total number of rounds')
    parser.add_argument('-A', '--num_clients_per_round', type=int, default=10, help='number of participated clients')
    parser.add_argument('--total_num_clients', type=int, default=None, help='total number of clients')

    parser.add_argument('--maxlen', type=int, default=400, help='maxlen for NLP dataset')

    parser.add_argument('--comment', type=str, default='', help='comment')

    parser.add_argument('--fix_seed', action='store_true', default=False, help='fix random seed')
    parser.add_argument('--parallel', action='store_true', default=False, help='use multi GPU')
    parser.add_argument('--use_mp', action='store_true', default=False, help='use multiprocessing')
    parser.add_argument('--nCPU', type=int, default=None, help='number of CPU cores for multiprocessing')

    args = parser.parse_args()
    return args


def load_data(args):
    if args.dataset == 'Reddit':
        #从本地文件系统中加载Reddit数据集，并将其转化为PyTorch的Dataset对象
        #args.data_dir是数据集存放的本地目录路径，args是一个命令行参数对象，用于指定数据集的相关参数
        return reddit.RedditDataset(args.data_dir, args)
    elif args.dataset == 'FederatedEMNIST':
        '''
        从本地文件系统中加载Federated EMNIST数据集，并将其转化为PyTorch的Dataset对象
         FederatedEMNISTDataset，它是一个联邦学习数据集。
         这种数据集由多个客户端数据集组成，每个客户端的数据集可以是来自不同用户的本地数据集。
         因此，该数据集需要保存有关联邦学习所需的客户端数量、客户端数据大小等元数据。
        '''
        return federated_emnist.FederatedEMNISTDataset(args.data_dir, args)


def create_model(args):
    if args.model == 'BLSTM':
        model = BLSTM.BLSTM(vocab_size=args.maxlen, num_classes=args.num_classes)
    elif args.model == 'CNN':
        #创建一个CNN模型，具体的实现在model/CNN.py文件中的CNN_DropOut类中
        #False参数表示这个模型中不使用dropout，即不随机舍弃一部分神经元，从而减少模型的过拟合。
        model = CNN.CNN_DropOut(False)
    if args.parallel:
        #使用多GPU训练，那么将模型放到DataParallel中
        '''
        torch.nn.DataParallel将模型在多个GPU上进行复制，每个GPU都处理模型的一部分输入数据，
        然后将各自的结果汇总，从而得到模型的最终输出。
        args.parallel为True，即开启了多GPU并行训练，
        就会将模型通过torch.nn.DataParallel转换为支持多GPU并行训练的模型，并将模型输出设备设置为第一个GPU。
        如果args.parallel为False，就不进行转换，直接返回原始的模型。
        '''
        model = torch.nn.DataParallel(model, output_device=0)
    return model


def client_selection_method(args, dataset):
    '''
        if args.method == 'myFL':
        return myFederatedLearning(args.total_num_client, args.device, args)
    else:
        return RandomSelection(args.total_num_client, args.device)
    '''
    return RandomSelection(args.total_num_client, args.device)


def federated_algorithm(dataset, model, args):
    '''
    提取了训练集中每个客户端的数据集大小
    dataset是一个字典，其中包含训练集和测试集，
    因此dataset['test']表示测试集，dataset['test']['data_sizes']表示测试集中每个客户端数据的大小。
    '''
    train_sizes = dataset['train']['data_sizes']
    if args.fed_algo == 'FedAdam':
        return FedAdam(train_sizes, model, args=args)
    else:
        # FedAvg,默认用FedAvg聚合
        return FedAvg(train_sizes, model)



if __name__ == '__main__':
    # set up
    args = get_args()
    '''
    #实验跟踪的一种方法
    wandb.init(
        project=f'myFL-{args.dataset}',
        name=f"{args.method}-{args.fed_algo}-{args.num_clients_per_round}{args.comment}",
        config=args,
        dir='.',
        save_code=True
    )
    '''

    #args.gpu_id指定的GPU可用，则使用指定的GPU作为运行设备；否则使用CPU作为运行设备
    #torch.device()函数来创建设备对象，并将其存储在args.device变量中
    args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    #将当前的GPU设备设置为args.device所指定的GPU设备，使得后续的运算都在该设备上进行。
    torch.cuda.set_device(args.device)
    print('Current cuda device: {}'.format(torch.cuda.current_device()))

    # set data
    #获取Dataset对象
    data = load_data(args)
    #将data数据集中的类别数量赋值给命令行参数args.num_classes。
    #数据集中的标签数目，也就是需要进行分类的类别数目
    args.num_classes = data.num_classes
    #训练集的客户端数量和测试集的客户端数量分别赋值
    args.total_num_client, args.test_num_clients = data.train_num_clients, data.test_num_clients
    #包含训练集和测试集
    dataset = data.dataset

    # set model
    #这里是CNN模型
    model = create_model(args)
    #选中的客户端的下标数组,是随机选择下一轮客户端下标
    client_selection = client_selection_method(args, dataset)
    #这里选用的是FedAvg聚合算法
    fed_algo = federated_algorithm(dataset, model, args)

    # 设置联邦优化算法
    #创建服务器对象
    ServerExecute = Server(dataset, model, args, client_selection, fed_algo)

    #数据预处理，所有客户端，都执行一个epoch,求得每个客户端的batch_time,存client
    #所有客户端迭代时间、收敛贡献、数据价值，可以有个初始值


    # 训练n轮
    ServerExecute.train()


    # save code
    '''
    from glob import glob
    code = wandb.Artifact(f'myFL-{args.dataset}', type='code')
    for path in glob('**/*.py', recursive=True):
        code.add_file(path)
    wandb.run.use_artifact(code)
    '''
