import torch
from torch import nn
from torch.nn.modules import conv
from resnet import resnet50
from torchsummary import summary




# 迁移学习：获取预训练模型，并替换池化层和全连接层
def get_model():
    # 获取欲训练模型 restnet50
    model = resnet50(pretrained=True) 
    # # 冻结模型参数
    # for param in model.parameters():
    #     param.requires_grad = False 
    # 替换最后2层：池化层和全连接层
    # 池化层
    # 全连接层
    model.fc = nn.Linear(2048, 2)# 2个输出
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
