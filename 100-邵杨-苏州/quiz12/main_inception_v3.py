import time
import numpy as np
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from inception_v3 import Inception_V3


transform1 = transforms.Compose([transforms.Resize((299, 299)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform2 = transforms.Compose([transforms.Resize((299, 299)),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
data_train = datasets.CIFAR10('./c10data', train=True, transform=transform1,
                         download=True)
data_test = datasets.CIFAR10('./c10data', train=False, transform=transform2,
                         download=False)
train_loader = DataLoader(data_train, batch_size=50, shuffle=True)
train_test = DataLoader(data_test, batch_size=50, shuffle=False)


def evaluate(model, tests, labels):
    with torch.no_grad():  # no grad when test and predict
        outputs = model(tests)
        predicted = torch.argmax(outputs, 1)
        total = labels.shape[0]
        correct = torch.sum(predicted == labels)
    return correct / total

def main():
    epochs = 10              #训练轮数
    learning_rate = 0.001     # 学习率
    model = Inception_V3()     # 建立模型
    print(model)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 选择优化器
    # 训练过程
    print('Start Training...')
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        watch_loss = []
        test_acc = []
        for inputs, targets in train_loader:
            targets = nn.functional.one_hot(targets, num_classes=10).float().view(-1, 10)
            optim.zero_grad()  # 梯度归零
            loss = model(inputs, targets)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        print('Evaluating ...')
        model.eval()
        for tests, targets in train_test:
            acc = evaluate(model, tests, targets)
            test_acc.append(acc)
        print('Accuracy of the network on the test images: {:.1f}%'.format(np.mean(test_acc)*100))

        stop_time = time.time()
        print('time is:{:.4f}s'.format(stop_time-start_time))
    torch.save(model.state_dict(), "Inception_V3_model.pth")

if __name__ == '__main__':
    main()
    print('===Finish Training===')