import numpy as np


class ClientSelection:
    def __init__(self, total, device):
        #初始化了客户端总数total和设备信息device
        self.total = total
        self.device = device

    def select(self, n, metric):
        pass


'''Random Selection'''
#确定在每轮训练中参与的客户端。
class RandomSelection(ClientSelection):
    def __init__(self, total, device):
        super().__init__(total, device)

    #n表示每轮选择的客户端数目，metric为度量指标，本实现中没有用到
    def select(self, n, metric=None):
        #选中客户端的下标数组
        selected_client_indexs = np.random.choice(self.total, size=n, replace=False)
        return selected_client_indexs



'''my Federated Learning'''
class myFederatedLearning(ClientSelection):
    def __init__(self, total, device, args):
        super().__init__(total, device)
        self.alpha_1 = args.alpha_1 # 加权概率相关
        self.alpha_2 = args.alpha_2 # 加权概率相关
        self.alpha_3 = args.alpha_3 # loss控制选择的客户端比例

    def select(self, n, metric, seed=0):
        # set sampling distribution
        probs = np.exp(np.array(metric) * self.alpha_2)

        num_select = int(self.alpha_1 * self.total)
        argsorted_value_list = np.argsort(metric)
        drop_client_indexs = argsorted_value_list[:self.total - num_select]
        probs[drop_client_indexs] = 0
        probs /= sum(probs)


        num_select = int((1 - self.alpha_3) * n)
        np.random.seed(seed)
        selected = np.random.choice(self.total, num_select, p=probs, replace=False)

        not_selected = np.array(list(set(np.arange(self.total)) - set(selected)))
        selected2 = np.random.choice(not_selected, n - num_select, replace=False)
        selected_client_indexs = np.append(selected, selected2, axis=0)
        print(f'{len(selected_client_indexs)} selected users: {selected_client_indexs}')
        return selected_client_indexs.astype(int)


def modified_exp(x, SAFETY=2.0):
    mrn = np.finfo(x.dtype).max
    threshold = np.log(mrn / x.size) - SAFETY
    xmax = x.max()
    if xmax > threshold:
        return np.exp(x - (xmax - threshold))
    else:
        return np.exp(x)