import torch
from torch import nn
from torch.nn.modules import conv
from resnetnoca import resnet50
from torchsummary import summary



# 自定义池化层
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size=None):
        super(AdaptiveConcatPool2d,self).__init__()
        size = size or (1, 1) # kernel大小
        # 自适应算法能够自动帮助我们计算核的大小和每次移动的步长。
        self.avgPooling = nn.AdaptiveAvgPool2d(size) # 自适应平均池化
        self.maxPooling = nn.AdaptiveMaxPool2d(size) # 最大池化
    def forward(self, x):
        # 拼接avg和max
        return torch.cat([self.maxPooling(x), self.avgPooling(x)], dim=1)

# 迁移学习：获取预训练模型，并替换池化层和全连接层
def get_model():
    # 获取欲训练模型 restnet50
    model = resnet50(pretrained=False)
    # # 冻结模型参数
    # for param in model.parameters():
    #     param.requires_grad = False 
    # 替换最后2层：池化层和全连接层
    # 池化层
    model.avgpool = AdaptiveConcatPool2d()
    # 全连接层
    model.fc = nn.Sequential(
        nn.Flatten(), # 拉平
        nn.BatchNorm1d(4096), # 加速神经网络的收敛过程，提高训练过程中的稳定性
        nn.Dropout(0.5), # 丢掉部分神经元
        nn.Linear(4096, 512), # 全连接层
        nn.ReLU(), # 激活函数
        nn.BatchNorm1d(512), 
        nn.Dropout(0.5),
        nn.Linear(512, 2), # 2个输出
        nn.LogSoftmax(dim=1) # 损失函数：将input转换成概率分布的形式，输出2个概率
    )
    return model

if __name__ == '__main__': 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = get_model().to(device)
    model.eval()
    summary(model, (3, 28, 28), 1, 'cpu')

    # input a tensor to get the predict result
    input = torch.rand([1, 3, 28, 28])
    if device != 'cpu':
        input.cuda()
    predict_res = model(input).argmax(1).item()
    print(f"Predicted class: {predict_res}")

    # check the parameters of the defined model using named_parameters() or parameters()
    # 这里包括在模型中定义的所有字段
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         # print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    #         print(f"Layer: {name} | Size: {param.size()} \n")
