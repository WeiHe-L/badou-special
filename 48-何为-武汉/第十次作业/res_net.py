import torch
from torch import nn
import torchvision

epoch = 40
batch_size = 64
lr = 0.01


# BasicBlock 两个 卷积   有两种模式  identity_block 和 conv-block(下采样)，
class BasicBlock(nn.Module):
    expansion = 1  # 输出通道数的倍乘

    def __init__(self, in_channel, out_channel, stride=(1, 1), down_sample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.down_sample = down_sample
        self.sample_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=(1, 1), stride=stride),
            nn.BatchNorm2d(out_channel * self.expansion))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample:
            residual = self.sample_conv(x)

        out += residual
        out = self.relu(out)
        return out


# Bottleneck 三个 卷积   有两种模式  identity_block 和 conv-block(下采样)，
class Bottleneck(nn.Module):
    expansion = 4  # 输出通道数的倍乘

    def __init__(self, in_channel, out_channel, stride=(1, 1), down_sample=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=(1, 1), stride=(1, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.strde = stride
        self.down_sample = down_sample
        self.sample_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=(1, 1), stride=stride),
            nn.BatchNorm2d(out_channel * self.expansion))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            residual = self.sample_conv(x)

        out += residual
        out = self.relu(out)
        return out


# resnet 分为5大层
class ResNet(nn.Module):

    def __init__(self, block, layer_info, num_class=1000):
        super(ResNet, self).__init__()
        self.in_channel = 64
        # 第一层
        self.layer1 = nn.Sequential(  # 3*32*32
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 第二层
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer2 = self.__build_layer(block, 64, layer_info[0])

        # 第三层
        self.layer3 = self.__build_layer(block, 128, layer_info[1], stride=(2, 2))

        # 第四层
        self.layer4 = self.__build_layer(block, 256, layer_info[2], stride=(2, 2))

        # 第五层
        self.layer5 = self.__build_layer(block, 512, layer_info[3], stride=(2, 2))

        self.out = nn.Sequential(nn.AvgPool2d(7),
                                 nn.Flatten(),
                                 nn.Linear(512 * block.expansion, num_class))

    def __build_layer(self, block, out_channel, num, stride=(1, 1)):
        if stride != (1, 1) or self.in_channel != out_channel * block.expansion:
            down_sample = True
        else:
            down_sample = False

        layer = [block(self.in_channel, out_channel, stride=stride, down_sample=down_sample)]
        self.in_channel = out_channel * block.expansion
        for _ in range(num - 1):
            layer.append(block(self.in_channel, out_channel))
        return nn.Sequential(*layer)

    def forward(self, x):

        x = self.layer1(x)

        x = self.max_pool(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        out_put = self.out(x)
        return out_put


def res_net_18():
    model = ResNet(BasicBlock, [2, 2, 2, 2], 10)
    return model


def res_net_34():
    model = ResNet(BasicBlock, [3, 4, 6, 3], 10)
    return model


def res_net_50():
    model = ResNet(BasicBlock, [3, 4, 6, 3], 10)
    return model


def res_net_101():
    model = ResNet(BasicBlock, [3, 4, 23, 3], 10)
    return model


def res_net_152():
    model = ResNet(BasicBlock, [3, 8, 36, 3], 10)
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

    MNIST_train_data = torch.utils.data.DataLoader(CIFAR_train, batch_size=batch_size, shuffle=True)
    MNIST_test_data = torch.utils.data.DataLoader(CIFAR_test, batch_size=batch_size, shuffle=False)
    return MNIST_train_data, MNIST_test_data


if __name__ == '__main__':
    my_model = res_net_50()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    train_data, test_data = get_dataset()

    train_model(my_model, train_data, optimizer, loss_func)
    predict(my_model, test_data)
    torch.save(my_model.state_dict(), './my_res_net_40.pkl')
    # print(len(MNIST_train))
    # print(len(MNIST_test))
