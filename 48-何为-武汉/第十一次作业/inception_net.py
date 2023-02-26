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


class AuxClass(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxClass, self).__init__()
        # 768*17*17 -> 768*5*5
        self.pool = nn.AvgPool2d(kernel_size=(5, 5), stride=(3, 3))

        self.conv = nn.Sequential(
            # 768*5*5 -> 128*5*5
            ConBlock(in_channels=in_channels, out_channels=128, kernel_size=(1, 1), stride=(1, 1)),
            # 128*5*5 -> 768*1*1
            ConBlock(in_channels=128, out_channels=768, kernel_size=(5, 5), stride=(1, 1))
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(1 * 1 * 768, num_classes),
        )

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        out = self.out(x)
        return out


class InceptionA(nn.Module):
    def __init__(self, in_channel, out_branch_1, in_branch_2, out_branch_2, in_branch_3, out_branch_3, out_branch_4):
        super(InceptionA, self).__init__()
        self.branch_1 = ConBlock(in_channels=in_channel, out_channels=out_branch_1, kernel_size=(1, 1), stride=(1, 1))

        self.branch_2 = nn.Sequential(
            ConBlock(in_channels=in_channel, out_channels=in_branch_2, kernel_size=(1, 1), stride=(1, 1)),
            ConBlock(in_channels=in_branch_2, out_channels=out_branch_2, kernel_size=(3, 3), stride=(1, 1),
                     padding=(1, 1))
        )

        self.branch_3 = nn.Sequential(
            ConBlock(in_channels=in_channel, out_channels=in_branch_3, kernel_size=(1, 1), stride=(1, 1)),
            ConBlock(in_channels=in_branch_3, out_channels=out_branch_3, kernel_size=(3, 3), stride=(1, 1),
                     padding=(1, 1)),
            ConBlock(in_channels=out_branch_3, out_channels=out_branch_3, kernel_size=(3, 3), stride=(1, 1),
                     padding=(1, 1))
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConBlock(in_channels=in_channel, out_channels=out_branch_4, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        branch1 = self.branch_1(x)
        branch2 = self.branch_2(x)
        branch3 = self.branch_3(x)
        branch4 = self.branch_4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return out


class InceptionB(nn.Module):
    def __init__(self, in_channel, out_branch_1, in_branch_2, out_branch_2, in_branch_3, out_branch_3, out_branch_4):
        super(InceptionB, self).__init__()
        self.branch_1 = ConBlock(in_channels=in_channel, out_channels=out_branch_1, kernel_size=(1, 1), stride=(1, 1))

        self.branch_2 = nn.Sequential(
            ConBlock(in_channels=in_channel, out_channels=in_branch_2, kernel_size=(1, 1), stride=(1, 1)),

            ConBlock(in_channels=in_branch_2, out_channels=in_branch_2, kernel_size=(3, 1), stride=(1, 1),
                     padding=(1, 0)),

            ConBlock(in_channels=in_branch_2, out_channels=out_branch_2, kernel_size=(1, 3), stride=(1, 1),
                     padding=(0, 1))
        )

        self.branch_3 = nn.Sequential(
            ConBlock(in_channels=in_channel, out_channels=in_branch_3, kernel_size=(1, 1), stride=(1, 1)),

            ConBlock(in_channels=in_branch_3, out_channels=in_branch_3, kernel_size=(3, 1), stride=(1, 1),
                     padding=(1, 0)),
            ConBlock(in_channels=in_branch_3, out_channels=in_branch_3, kernel_size=(1, 3), stride=(1, 1),
                     padding=(0, 1)),

            ConBlock(in_channels=in_branch_3, out_channels=in_branch_3, kernel_size=(3, 1), stride=(1, 1),
                     padding=(1, 0)),
            ConBlock(in_channels=in_branch_3, out_channels=out_branch_3, kernel_size=(1, 3), stride=(1, 1),
                     padding=(0, 1))
        )

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConBlock(in_channels=in_channel, out_channels=out_branch_4, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        branch1 = self.branch_1(x)
        branch2 = self.branch_2(x)
        branch3 = self.branch_3(x)
        branch4 = self.branch_4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return out


class InceptionC(nn.Module):
    def __init__(self, in_channel, out_branch_1, in_branch_2, out_branch_2, in_branch_3, out_branch_3, out_branch_4):
        super(InceptionC, self).__init__()
        self.branch_1 = ConBlock(in_channels=in_channel, out_channels=out_branch_1, kernel_size=(1, 1), stride=(1, 1))

        self.branch_2_a = ConBlock(in_channels=in_channel, out_channels=in_branch_2, kernel_size=(1, 1), stride=(1, 1))

        self.branch_2_b_1 = ConBlock(in_channels=in_branch_2, out_channels=out_branch_2, kernel_size=(3, 1),
                                     stride=(1, 1), padding=(1, 0))

        self.branch_2_b_2 = ConBlock(in_channels=in_branch_2, out_channels=out_branch_2, kernel_size=(1, 3),
                                     stride=(1, 1), padding=(0, 1))

        self.branch_3_a = nn.Sequential(
            ConBlock(in_channels=in_channel, out_channels=in_branch_3, kernel_size=(1, 1), stride=(1, 1)),

            ConBlock(in_channels=in_branch_3, out_channels=out_branch_3, kernel_size=(3, 3), stride=(1, 1),
                     padding=(1, 1)))
        self.branch_3_b_1 = ConBlock(in_channels=out_branch_3, out_channels=out_branch_3, kernel_size=(3, 1),
                                     stride=(1, 1), padding=(1, 0))
        self.branch_3_b_2 = ConBlock(in_channels=out_branch_3, out_channels=out_branch_3, kernel_size=(1, 3),
                                     stride=(1, 1), padding=(0, 1))

        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConBlock(in_channels=in_channel, out_channels=out_branch_4, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        branch1 = self.branch_1(x)

        branch_2_a = self.branch_2_a(x)

        branch_2_b_1 = self.branch_2_b_1(branch_2_a)
        branch_2_b_2 = self.branch_2_b_2(branch_2_a)
        branch2 = torch.cat([branch_2_b_1, branch_2_b_2], dim=1)

        branch_3_a = self.branch_3_a(x)

        branch_3_b_1 = self.branch_3_b_1(branch_3_a)
        branch_3_b_2 = self.branch_3_b_2(branch_3_a)
        branch3 = torch.cat([branch_3_b_1, branch_3_b_2], dim=1)

        branch4 = self.branch_4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return out


class InceptionD(nn.Module):
    def __init__(self, in_channel, in_branch_1, out_branch_1, in_branch_2, out_branch_2):
        super(InceptionD, self).__init__()
        self.branch_1 = nn.Sequential(
            ConBlock(in_channels=in_channel, out_channels=in_branch_1, kernel_size=(1, 1), stride=(1, 1)),
            ConBlock(in_channels=in_branch_1, out_channels=out_branch_1, kernel_size=(3, 3), stride=(2, 2))
        )
        self.branch_2 = nn.Sequential(
            ConBlock(in_channels=in_channel, out_channels=in_branch_2, kernel_size=(1, 1), stride=(1, 1)),
            ConBlock(in_channels=in_branch_2, out_channels=out_branch_2, kernel_size=(3, 3), stride=(1, 1),
                     padding=(1, 1)),
            ConBlock(in_channels=out_branch_2, out_channels=out_branch_2, kernel_size=(3, 3), stride=(2, 2))
        )
        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch1 = self.branch_1(x)
        branch2 = self.branch_2(x)
        branch3 = self.branch_3(x)

        out = torch.cat([branch1, branch2, branch3], dim=1)
        return out


class InceptionE(nn.Module):
    def __init__(self, in_channel, in_branch_1, out_branch_1, in_branch_2, out_branch_2):
        super(InceptionE, self).__init__()
        self.branch_1 = nn.Sequential(
            ConBlock(in_channels=in_channel, out_channels=in_branch_1, kernel_size=(1, 1), stride=(1, 1)),
            ConBlock(in_channels=in_branch_1, out_channels=out_branch_1, kernel_size=(3, 3), stride=(2, 2))
        )
        self.branch_2 = nn.Sequential(
            ConBlock(in_channels=in_channel, out_channels=in_branch_2, kernel_size=(1, 1), stride=(1, 1)),

            ConBlock(in_channels=in_branch_2, out_channels=in_branch_2, kernel_size=(1, 7), stride=(1, 1),
                     padding=(0, 3)),
            ConBlock(in_channels=in_branch_2, out_channels=in_branch_2, kernel_size=(7, 1), stride=(1, 1),
                     padding=(3, 0)),
            ConBlock(in_channels=in_branch_2, out_channels=out_branch_2, kernel_size=(3, 3), stride=(2, 2))
        )
        self.branch_3 = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        branch1 = self.branch_1(x)
        branch2 = self.branch_2(x)
        branch3 = self.branch_3(x)

        out = torch.cat([branch1, branch2, branch3], dim=1)
        return out


class Inception(nn.Module):

    def __init__(self, in_channel, out_1x1, in_3x3, out_3x3, in_5x5, out_5x5, pooling):
        super(Inception, self).__init__()

        # 四种类型的卷积  1x1 1x1+3x3 1x1+5x5 pool+1x1
        # 1x1 卷积
        self.branch_1 = ConBlock(in_channels=in_channel, out_channels=out_1x1, kernel_size=(1, 1), stride=(1, 1))

        # 1x1 + 3x3 卷积
        self.branch_2 = nn.Sequential(
            #
            ConBlock(in_channels=in_channel, out_channels=in_3x3, kernel_size=(1, 1), stride=(1, 1)),

            ConBlock(in_channels=in_3x3, out_channels=out_3x3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )

        # 1x1 + 5x5 卷积
        self.branch_3 = nn.Sequential(
            ConBlock(in_channels=in_channel, out_channels=in_5x5, kernel_size=(1, 1), stride=(1, 1)),

            ConBlock(in_channels=in_5x5, out_channels=out_5x5, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        )

        # max_pool + 1x1 卷积
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            ConBlock(in_channels=in_channel, out_channels=pooling, kernel_size=(1, 1), stride=(1, 1)),
        )

    def forward(self, x):
        branch1 = self.branch_1(x)
        branch2 = self.branch_2(x)
        branch3 = self.branch_3(x)
        branch4 = self.branch_4(x)

        out = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return out


class GoogleNetInceptionV3(nn.Module):
    # inception V1

    def __init__(self, num_class=1000):
        super(GoogleNetInceptionV3, self).__init__()

        self.block1 = nn.Sequential(
            # 3*299*299 -> 32*149*149
            ConBlock(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2)),
            # 32*149*149 -> 32*147*147
            ConBlock(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
            # 32*147*147 -> 64*147*147
            ConBlock(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )

        # 64*147*147 -> 64*73*73
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.block2 = nn.Sequential(
            # 64*73*73 -> 80*71*71
            ConBlock(in_channels=64, out_channels=80, kernel_size=(3, 3), stride=(1, 1)),
            # 80*71*71 -> 192*35*35
            ConBlock(in_channels=80, out_channels=192, kernel_size=(3, 3), stride=(2, 2)),
            # # 192*35*35 -> 228*35*35
            # ConBlock(in_channels=192, out_channels=228, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        # 192*35*35 -> 228*35*35
        self.block3 = nn.Sequential(
            InceptionA(in_channel=192, out_branch_1=64, in_branch_2=48, out_branch_2=64, in_branch_3=64,
                       out_branch_3=96, out_branch_4=32),
            InceptionA(in_channel=256, out_branch_1=64, in_branch_2=48, out_branch_2=64, in_branch_3=64,
                       out_branch_3=96, out_branch_4=64),
            InceptionA(in_channel=288, out_branch_1=64, in_branch_2=48, out_branch_2=64, in_branch_3=64,
                       out_branch_3=96, out_branch_4=64)
        )

        # 228*35*35 -> 768*17*17
        self.block4 = InceptionD(in_channel=288, in_branch_1=384, out_branch_1=384, in_branch_2=64, out_branch_2=96)

        # 768*17*17 -> 768*17*17
        self.block5 = nn.Sequential(
            InceptionB(in_channel=768, out_branch_1=192, in_branch_2=128, out_branch_2=192, in_branch_3=128,
                       out_branch_3=192, out_branch_4=192),
            # InceptionB(in_channel=768, out_branch_1=192, in_branch_2=128, out_branch_2=192, in_branch_3=128,
            #            out_branch_3=192, out_branch_4=192),
            InceptionB(in_channel=768, out_branch_1=192, in_branch_2=160, out_branch_2=192, in_branch_3=160,
                       out_branch_3=192, out_branch_4=192),
            InceptionB(in_channel=768, out_branch_1=192, in_branch_2=160, out_branch_2=192, in_branch_3=160,
                       out_branch_3=192, out_branch_4=192),
            InceptionB(in_channel=768, out_branch_1=192, in_branch_2=192, out_branch_2=192, in_branch_3=192,
                       out_branch_3=192, out_branch_4=192)
        )
        # 768*17*17
        self.aux = AuxClass(in_channels=768, num_classes=num_class)

        # 768*17*17 -> 1280*8*8
        self.block6 = InceptionE(in_channel=768, in_branch_1=192, out_branch_1=320, in_branch_2=192, out_branch_2=192)

        # 1280*8*8 -> 2048 * 8 * 8
        self.block7 = nn.Sequential(
            InceptionC(in_channel=1280, out_branch_1=320, in_branch_2=384, out_branch_2=384, in_branch_3=448,
                       out_branch_3=384, out_branch_4=192),
            InceptionC(in_channel=2048, out_branch_1=320, in_branch_2=384, out_branch_2=384, in_branch_3=448,
                       out_branch_3=384, out_branch_4=192)
        )

        # 2048*8*8 -> 2048*1*1
        self.pool2 = nn.MaxPool2d(kernel_size=8, stride=1, ceil_mode=True)

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, num_class)
        )

    def forward(self, x):
        # 3*299*299 -> 64 * 147 * 147
        x = self.block1(x)

        # 64*147*147 -> 64*73*73
        x = self.pool1(x)

        # 64*73*73 -> 192*35*35
        x = self.block2(x)

        # 192*35*35 -> 228*35*35
        x = self.block3(x)

        # 228*35*35 -> 768*17*17
        x = self.block4(x)

        # 768*17*17 -> 768*17*17
        x = self.block5(x)

        aux = self.aux(x)

        # 768*17*17 -> 1280*8*8
        x = self.block6(x)

        # 1280*8*8 -> 2048 * 8 * 8
        x = self.block7(x)

        # 2048*8*8 -> 2048*1*1
        x = self.pool2(x)

        out = self.out(x)

        return out, aux


def google_net_v3():
    model = GoogleNetInceptionV3(10)
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
            y, y1 = net(x)
            loss = loss_func(y, label)
            loss1 = loss_func(y1, label)
            loss = loss + loss1 * 0.3

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
        y, _ = net(x)
        pred = torch.argmax(nn.Softmax(dim=1)(y), dim=1)
        acc_num += (pred == label).sum()
        total_num += y.shape[0]

    print('train data acc: {}'.format(acc_num / total_num))


def get_dataset():
    tran = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Resize((299, 299))])

    CIFAR_train = torchvision.datasets.CIFAR10(root='./Dataset', train=True, download=True,
                                               transform=tran)

    CIFAR_test = torchvision.datasets.CIFAR10(root='./Dataset', train=False, download=True,
                                              transform=tran)

    CIFAR_train_data = torch.utils.data.DataLoader(CIFAR_train, batch_size=batch_size, shuffle=True)
    CIFAR_test_data = torch.utils.data.DataLoader(CIFAR_test, batch_size=batch_size, shuffle=False)
    return CIFAR_train_data, CIFAR_test_data


if __name__ == '__main__':
    my_model = google_net_v3()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    train_data, test_data = get_dataset()

    train_model(my_model, train_data, optimizer, loss_func)
    predict(my_model, test_data)
    torch.save(my_model.state_dict(), './my_google_net_v3_10.pkl')
    # print(len(MNIST_train))
    # print(len(MNIST_test))
