import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=stride,
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1),
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2),
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # (in, out, 分组卷积的步数)
        self.layers = nn.Sequential(
            Block(32, 64, 1),
            Block(64, 128, 2),
            Block(128, 128, 1),
            Block(128, 256, 2),
            Block(256, 256, 1),
            Block(256, 512, 2),
            Block(512, 512, 1),
            Block(512, 512, 1),
            Block(512, 512, 1),
            Block(512, 512, 1),
            Block(512, 512, 1),
            Block(512, 1024, 2),
            Block(1024, 1000, 1))

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(1000, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.layers(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        y_pred = self.softmax(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

if __name__ == '__main__':
    model = MobileNet()
    print(model)

    input = torch.randn(2, 3, 224, 224)
    target = torch.randint(0, 10, (2, 1))
    print(target)
    target = nn.functional.one_hot(target, num_classes=10).float().view(-1, 10)

    print(target)
    print(model(input, target))


