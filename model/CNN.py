
import torch
import torch.nn as nn

'''
CNN_DropOut类继承自torch.nn.Module类，表示它是一个可训练的PyTorch模型。
在这个类的初始化函数中，会定义一些CNN模型的层，包括卷积层、池化层、全连接层等，
同时还会定义一些参数，如卷积核大小、卷积核数量、池化层大小等。
'''
class CNN_DropOut(torch.nn.Module):
    def __init__(self, only_digits=False, num_channel=1):
        super(CNN_DropOut, self).__init__()
        self.conv2d_1 = nn.Conv2d(num_channel, 32, kernel_size=3)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout_1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(9216, 128)
        self.dropout_2 = nn.Dropout(0.5)
        self.linear_2 = nn.Linear(128, 10 if only_digits else 62)
        self.relu = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.relu(x)
        x = self.conv2d_2(x)
        x = self.relu(x)
        x = self.max_pooling(x)
        x = self.dropout_1(x)
        x = self.flatten(x)
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.linear_2(x)
        #x = self.softmax(self.linear_2(x))
        return x
