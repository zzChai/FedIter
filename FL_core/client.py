from FL_core.trainer import Trainer


class Client(object):
    def __init__(self, client_index, local_train_data, local_test_data, model, args):
        self.client_index = client_index
        self.local_train_data = local_train_data
        self.local_test_data = local_test_data
        self.device = args.device
        self.trainer = Trainer(model, args) #每个Client对象都有个对应的Trainer对象
        self.num_epoch = args.num_epoch # local epochs E

    def train(self, global_model, tracking=True):
        self.trainer.set_model(global_model)
        if self.num_epoch == 0:
            acc, loss = self.trainer.train_E0(self.local_train_data, tracking)
        else:
            #走这条
            acc, loss = self.trainer.train(self.local_train_data, tracking)
        #拿到这个client更新后的模型
        model = self.trainer.get_model()
        return model, acc, loss

    def test(self, model, mode='test'):
        if mode == 'train':
            acc, loss, auc = self.trainer.test(model, self.local_train_data)
        else:
            acc, loss, auc = self.trainer.test(model, self.local_test_data)
        return {'loss': loss, 'acc': acc, 'auc': auc}

    def get_client_index(self):
        return self.client_index
