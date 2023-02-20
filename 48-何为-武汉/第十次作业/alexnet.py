import torch
from torch import nn
import torchvision

epoch = 40
batch_size = 64
lr = 0.01


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Sequential(  # 3*224*224
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv2 = nn.Sequential(  # 96*27*27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )
        self.conv3 = nn.Sequential(  # 256*13*13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(  # 384*13*13
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(  # 384*13*13
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2)
        )

        self.out = nn.Sequential(nn.Flatten(),
                                 nn.Linear(256 * 6 * 6, 4096), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                                 nn.Linear(4096, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out_put = self.out(x)
        return out_put


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
    my_model = MyNet()
    optimizer = torch.optim.SGD(my_model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    train_data, test_data = get_dataset()

    train_model(my_model, train_data, optimizer, loss_func)
    predict(my_model, test_data)
    torch.save(my_model.state_dict(), './my_alex_net_40.pkl')
    # print(len(MNIST_train))
    # print(len(MNIST_test))
