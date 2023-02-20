# coding=utf-8
"""
* project : free_code
* author : hewei
* file : vgg_net.py
* corp : NSFOCUS Corporation
* time : 2023-02-19
"""

import torch
from torch import nn
import torchvision

epoch = 40
batch_size = 128
lr = 0.1


# VGG 16
class VGGNet(nn.Module):

    def __init__(self, layer_info, num_class=1000):
        super(VGGNet, self).__init__()
        self.in_channel = 3
        # 第一层 3*224*224 -> 64*112*112
        self.layer1 = self.__build_feature_layer(64, layer_info[0])

        # 第二层 64*112*112 -> 128*56*56
        self.layer2 = self.__build_feature_layer(128, layer_info[1])

        # 第三层 128*56*56 -> 256*28*28
        self.layer3 = self.__build_feature_layer(256, layer_info[2])

        # 第四层 256*28*28 -> 512*14*14
        self.layer4 = self.__build_feature_layer(512, layer_info[3])
        #
        # 第五层 512*14*14 -> 512*7*7
        self.layer5 = self.__build_feature_layer(512, layer_info[4])

        self.out = nn.Sequential(nn.Flatten(),
                                 nn.Linear(512*7*7, 512), nn.ReLU(), nn.Dropout(p=0.5),
                                 # nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.5),
                                 nn.Linear(512, num_class)
                                 )

    def __build_feature_layer(self, out_channel, layer_num):
        layer = []
        for _ in range(layer_num):
            layer.append(
                nn.Conv2d(in_channels=self.in_channel, out_channels=out_channel, kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 1)))
            layer.append(nn.ReLU())
            self.in_channel = out_channel
        layer.append(nn.MaxPool2d(kernel_size=(2, 2), stride=2))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        out_put = self.out(x)
        return out_put


def vgg_11():
    model = VGGNet([1, 1, 1, 2, 2], 10)
    return model


def vgg_13():
    model = VGGNet([2, 2, 2, 2, 2], 10)
    return model


def vgg_16():
    model = VGGNet([2, 2, 3, 3, 3], 10)
    return model


def vgg_19():
    model = VGGNet([2, 2, 4, 4, 4], 10)
    return model


def train_model(net, train_data, optim, loss_func):
    for i in range(epoch):
        print('--- Epoch %d ---' % (i + 1))
        steps = 0
        total_loss = 0
        total_num = 0
        acc_num = 0
        net.train()
        for x, label in train_data:
            y = net(x)
            loss = loss_func(y, label)
            optim.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
            optim.step()

            total_loss += loss.data
            pred = torch.argmax(nn.Softmax(dim=1)(y), dim=1)
            acc_num += (pred == label).sum()
            total_num += y.shape[0]
            steps += 1
            if steps % 10 == 0:
                print('epoch: {}, loss: {}, acc: {}'.format(i + 1, total_loss / steps, acc_num / total_num))
        print('epoch: {}, loss: {}, acc: {}'.format(i + 1, total_loss / steps, acc_num / total_num))


def predict(net, test_data):
    net.eval()
    acc_num = 0
    total_num = 0
    net.eval()
    for x, label in test_data:
        y = net(x)
        pred = torch.argmax(nn.Softmax(dim=1)(y), dim=1)
        acc_num += (pred == label).sum()
        total_num += y.shape[0]

    print('train data acc: {}'.format(acc_num / total_num))


def get_dataset():
    tran = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((224, 224))])

    CIFAR_train = torchvision.datasets.CIFAR10(root='./Dataset', train=True, download=True,
                                               transform=tran)

    CIFAR_test = torchvision.datasets.CIFAR10(root='./Dataset', train=False, download=True,
                                              transform=tran)

    CIFAR_train_data = torch.utils.data.DataLoader(CIFAR_train, batch_size=batch_size, shuffle=True)
    CIFAR_test_data = torch.utils.data.DataLoader(CIFAR_test, batch_size=batch_size, shuffle=False)
    return CIFAR_train_data, CIFAR_test_data


if __name__ == '__main__':
    my_model = vgg_11()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    train_data, test_data = get_dataset()

    train_model(my_model, train_data, optimizer, loss_func)
    predict(my_model, test_data)
    torch.save(my_model.state_dict(), './my_vgg_net_11_40.pkl')
    # print(len(MNIST_train))
    # print(len(MNIST_test))
