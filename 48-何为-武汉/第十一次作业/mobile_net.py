import torch
from torch import nn
import torchvision

epoch = 10
batch_size = 64
lr = 0.01


class ConBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)):
        super(ConBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=kernel_size,
                      stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super(MobileBlock, self).__init__()
        # depth wise
        self.depth_wise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, bias=False, kernel_size=(3, 3),
                      stride=stride, padding=(1, 1), groups=in_channels),
            nn.BatchNorm2d(in_channels, eps=0.001),
            nn.ReLU(inplace=True)
        )
        # point wise
        self.point_wise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, kernel_size=(1, 1),
                      stride=(1, 1)),
            nn.BatchNorm2d(out_channels, eps=0.001),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depth_wise(x)
        out = self.point_wise(x)
        return out


class MobileNet(nn.Module):
    # inception V1

    def __init__(self, num_class=1000):
        super(MobileNet, self).__init__()

        # # 3*224*224 -> 32*112*112
        self.conv1 = ConBlock(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        self.mobile_block1 = nn.Sequential(
            # 32*112*112 -> 64*112*112
            MobileBlock(32, 64, stride=(1, 1)),

            # 64*112*112 -> 128*56*56
            MobileBlock(64, 128, stride=(2, 2)),

            # 128*56*56 -> 128*56*56
            MobileBlock(128, 128, stride=(1, 1)),

            # 128*56*56 -> 256*28*288
            MobileBlock(128, 256, stride=(2, 2)),

            # 256*28*288 -> 256*28*288
            MobileBlock(256, 256, stride=(1, 1)),

            # 256*28*288 -> 512*14*14
            MobileBlock(256, 512, stride=(2, 2))
        )

        self.mobile_block2 = nn.Sequential(
            # 512*14*14 -> 512*14*14
            MobileBlock(512, 512, stride=(1, 1)),

            # 512*14*14 -> 512*14*14
            MobileBlock(512, 512, stride=(1, 1)),

            # 512*14*14 -> 512*14*14
            MobileBlock(512, 512, stride=(1, 1)),

            # 512*14*14 -> 512*14*14
            MobileBlock(512, 512, stride=(1, 1)),

            # 512*14*14 -> 512*14*14
            MobileBlock(512, 512, stride=(1, 1)),
        )

        self.mobile_block3 = nn.Sequential(
            # 512*14*14 -> 1024*7*7
            MobileBlock(512, 1024, stride=(2, 2)),

            # 1024*7*7 -> 1024*7*7
            MobileBlock(1024, 1024, stride=(1, 1)),
        )

        self.out = nn.Sequential(
            nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        # 3*224*224 -> 32 * 112 * 112
        x = self.conv1(x)

        # 32*112*112 -> 512*14*14
        x = self.mobile_block1(x)

        # 512*14*14 -> 512*14*14
        x = self.mobile_block2(x)

        # 512*14*14 -> 1024*7*7
        x = self.mobile_block3(x)

        out = self.out(x)

        return out


def mobile_net_v1():
    model = MobileNet(10)
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

    CIFAR_train_data = torch.utils.data.DataLoader(CIFAR_train, batch_size=batch_size, shuffle=True)
    CIFAR_test_data = torch.utils.data.DataLoader(CIFAR_test, batch_size=batch_size, shuffle=False)
    return CIFAR_train_data, CIFAR_test_data


if __name__ == '__main__':
    my_model = mobile_net_v1()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    train_data, test_data = get_dataset()

    train_model(my_model, train_data, optimizer, loss_func)
    predict(my_model, test_data)
    torch.save(my_model.state_dict(), './my_google_net_v1_10.pkl')
    # print(len(MNIST_train))
    # print(len(MNIST_test))
