import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils import data
from process import train_val, test, tb_writer
from criteria import accuracy, metrics
from model import get_model
from dataloader import dataloader
import time


def train(model, device, dataloaders, criterion, optimizer, epochs, writer):
    # 训练开始
    print("{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}".format('Epoch', 'Train Loss', 'val_loss', 'val_acc', 'Test Loss', 'Test_acc'))
    # 初始最小的损失
    best_loss = np.inf
    # 开始训练、测试
    for epoch in range(epochs):
        # 训练，return: loss
        train_loss, val_loss, val_acc = train_val(model, device, dataloaders['train'], dataloaders['val'], optimizer, criterion, epoch, writer)
        # 测试，return: loss + accuracy
        test_loss, test_acc = test(model, device, dataloaders['test'], criterion, epoch, writer)
        # 判断损失是否最小
        if test_loss < best_loss:
            best_loss = test_loss # 保存最小损失
            # 保存模型
            timestr = time.strftime("%Y%m%d_%H%M%S")
            torch.save(model.state_dict(), 'model.pth')
        # 输出结果
        print("{0:>15} | {1:>15} | {2:>15} | {3:>15} | {4:>15} | {5:>15}".format(epoch, train_loss, val_loss, val_acc, test_loss, test_acc))
        writer.flush()
    writer.close()

if __name__ == '__main__': 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('using {} device'.format(device.type))

    model = get_model().to(device)

    criterion = nn.NLLLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001)

    writer = tb_writer()
    dataloaders = dataloader()
    epochs=100
    train(model, device, dataloaders, criterion, optimizer, epochs, writer)
 