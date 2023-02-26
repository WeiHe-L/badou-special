import torch
import torch.nn as nn

#卷积+标准化+relu三合一模块，     b, out_channnels, ...
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

# b, 224+pool_features, size不变
class InceptionModuleA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionModuleA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
             BasicConv2d(in_channels, 48, kernel_size=1),
             BasicConv2d(48, 64, kernel_size=5, padding=2))

        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1))

        self.branch_pool_1x1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_features, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1x1(x)
        out2 = self.branch5x5(x)
        out3 = self.branch3x3(x)
        out4 = self.branch_pool_1x1(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

#b,384+96+288=768, size/2
class InceptionModuleB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModuleB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=2))

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch3x3(x)
        out2 = self.branch3x3dbl(x)
        out3 = self.branch_pool(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

#b, 192+192+192+192=768, size不变
class InceptionModuleC(nn.Module):
    def __init__(self, in_channels, channels):
        super(InceptionModuleC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        self.branch7x7 = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channels, 192, kernel_size=(7, 1), padding=(3, 0)))

        self.branch7x7dbl = nn.Sequential(
            BasicConv2d(in_channels, channels, kernel_size=1),
            BasicConv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(channels, 192, kernel_size=(1, 7), padding=(0, 3)))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1x1(x)
        out2 = self.branch7x7(x)
        out3 = self.branch7x7dbl(x)
        out4 = self.branch_pool(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out

#b, 320+192+768=1280, size/2
class InceptionModuleD(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModuleD, self).__init__()
        self.branch3x3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, stride=2))

        self.branch7x7x3 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(192, 192, kernel_size=3, stride=2))

        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        out1 = self.branch3x3(x)
        out2 = self.branch7x7x3(x)
        out3 = self.branch_pool(x)
        out = torch.cat([out1, out2, out3], dim=1)
        return out

#b, 320+768+768+192=2048, size
class InceptionModuleE(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModuleE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1x1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_1x3 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_3x1 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1x1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_3x3 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_1x3 = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3x1 = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, 192, kernel_size=1))

    def forward(self, x):
        out1 = self.branch1x1(x)

        out2_1 = self.branch3x3_1x1(x)
        out2 = torch.cat([self.branch3x3_1x3(out2_1), self.branch3x3_3x1(out2_1)], dim=1)

        out3_1 = self.branch3x3dbl_1x1(x)
        out3_2 = self.branch3x3dbl_3x3(out3_1)
        out3 = torch.cat([self.branch3x3dbl_1x3(out3_2), self.branch3x3dbl_3x1(out3_2)], dim=1)

        out4 = self.branch_pool(x)
        out = torch.cat([out1, out2, out3, out4], dim=1)
        return out


class Inception_V3(nn.Module):
    def __init__(self, num_classes=10): #b,3,299,299
        super(Inception_V3, self).__init__()
        # b,32,149,149 > b,32,147,147 > b,64,147,147 > b,64,73,73
        self.block1 = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2))

        # b, 80, 71, 71 > b, 192, 71, 71 > b, 192, 35, 35
        self.block2 = nn.Sequential(
            BasicConv2d(64, 80, kernel_size=3),
            BasicConv2d(80, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2))

        # b, 256, 35, 35 > b, 288, 35, 35 > b, 288, 35, 35
        self.block3 = nn.Sequential(
            InceptionModuleA(192, 32),
            InceptionModuleA(256, 64),
            InceptionModuleA(288, 64))

        # b, 768, 17, 17 > b, 768, 17, 17 > b, 768, 17, 17 > b, 768, 17, 17 > b, 768, 17, 17
        self.block4 = nn.Sequential(
            InceptionModuleB(288),
            InceptionModuleC(768, 128),
            InceptionModuleC(768, 160),
            InceptionModuleC(768, 160),
            InceptionModuleC(768, 192))

        # b, 1280, 8, 8 > b, 2048, 8, 8 > b, 2048, 8, 8
        self.block5 = nn.Sequential(
            InceptionModuleD(768),
            InceptionModuleE(1280),
            InceptionModuleE(2048))

        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(2048, num_classes)

        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        y_pred = self.softmax(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred

if __name__ == '__main__':
    model = Inception_V3()
    print(model)

    input = torch.randn(2, 3, 299, 299)
    target = torch.randint(0, 10, (2, 1))
    target = nn.functional.one_hot(target, num_classes=10).float().view(-1, 10)

    print(target)
    print(model(input, target))





