import time
import torch
from torch.utils.tensorboard import SummaryWriter # SummaryWriter() 向事件文件写入事件和概要
from criteria import metrics




log_path = 'logdir/'   # 定义日志路径

# 定义函数：获取tensorboard writer
def tb_writer():
    timestr = time.strftime("%Y%m%d_%H%M%S") # 时间格式
    writer = SummaryWriter(log_path+timestr) # 写入日志
    return writer


# # 记录错误分类的图片
# def misclassified_images(pred, writer, target, images, output, epoch, count=10):
#     misclassified = (pred != target.data) # 判断是否一致
#     for index, image_tensor in enumerate(images[misclassified][:count]):
#         img_name = 'Epoch:{}-->Predict:{}-->Actual:{}'.format(epoch, LABEL[pred[misclassified].tolist()[index]],
#                                                               LABEL[target.data[misclassified].tolist()[index]])
#         writer.add_image(img_name, image_tensor, epoch)


# 定义训练函数
def train_val(model, device, train_loader, val_loader, optimizer, criterion, epoch, writer):
    model.train()
    total_loss = 0.0
    val_loss = 0.0
    val_acc = 0
    for batch_id, (images, labels) in enumerate(train_loader):
        # 部署到device上
        images, labels = images.to(device), labels.to(device)
        # 梯度置0
        optimizer.zero_grad()
        # 模型输出
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累计损失
        total_loss += loss.item() * images.size(0)
    # 平均训练损失
    train_loss = total_loss / len(train_loader.dataset)
    #写入到writer中
    writer.add_scalar('Training Loss', train_loss, epoch)
    # 写入到磁盘
    writer.flush()
    
    model.eval() 
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) # 前向传播输出
            loss = criterion(outputs, labels) # 损失
            val_loss += loss.item() * images.size(0) # 累计损失
            _, pred = torch.max(outputs, dim=1) # 获取最大概率的索引
            correct = pred.eq(labels.view_as(pred)) # 返回：tensor([ True,False,True,...,False])
            accuracy = torch.mean(correct.type(torch.FloatTensor)) # 准确率
            val_acc += accuracy.item() * images.size(0) # 累计准确率
        # 平均验证损失
        val_loss = val_loss / len(val_loader.dataset)
        # 平均准确率
        val_acc = val_acc / len(val_loader.dataset)
        
    return train_loss, val_loss, val_acc 

# 定义测试函数
def test(model, device, test_loader, criterion, epoch, writer):
    model.eval()
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    correct = 0.0 # 正确数
    with torch.no_grad():
        for batch_id, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            # 输出
            outputs = model(images)
            # 损失
            loss = criterion(outputs, labels)
            # 累计损失
            total_loss += loss.item()
            # 获取预测概率最大值的索引
            _, predicted = torch.max(outputs, dim=1)
            # 累计正确预测的数
            correct += predicted.eq(labels.view_as(predicted)).sum().item()
            # 错误分类的图片
            # misclassified_images(predicted, writer, labels, images, outputs, epoch)
            # precision, recall, f1 = metrics(outputs, labels)
            # total_precision += precision
            # total_recall += recall
            # total_f1 += f1
        # 平均损失
        test_loss = total_loss / len(test_loader.dataset)
        # avg_percision = total_precision / len(test_loader.dataset)
        # avg_recall = total_recall / len(test_loader.dataset)
        # avg_f1 = total_f1 / len(test_loader.dataset)
        # 计算正确率
        accuracy = 100 * correct / len(test_loader.dataset)
        # 将test的结果写入write
        writer.add_scalar("Test Loss", test_loss, epoch)
        writer.add_scalar("accuracy", accuracy, epoch)
        # writer.add_scalar("percision", avg_percision, epoch)
        # writer.add_scalar("recall", avg_recall, epoch)
        # writer.add_scalar("f1", avg_f1, epoch)
        writer.flush()
        return test_loss, accuracy
