from torchvision import datasets, transforms
from torch.utils.data import DataLoader, dataset

class Createdataloaders:
    def __init__(self):
        # 数据集所在目录路径
        data_dir = '/home/ccut/文档/data/data'
        # train路径
        train_dir = data_dir + '/train'
        # val路径
        val_dir = data_dir + '/val'
        # test路径
        test_dir = data_dir + '/test'

        # 分为为train, val, test定义transform
        image_transforms = {
            'train' : transforms.Compose([  #将图片的所有操作全部放在一起，就像一个管道一样
                transforms.RandomResizedCrop(size=300, scale=(0.8, 1.1)), #功能：随机长宽比裁剪原始图片, 表示随机crop出来的图片会在的0.08倍至1.1倍之间
                transforms.RandomRotation(degrees=10), #功能：根据degrees随机旋转一定角度, 则表示在（-10，+10）度之间随机旋转
                transforms.ColorJitter(0.4, 0.4, 0.4), #功能：修改亮度、对比度和饱和度
                transforms.RandomHorizontalFlip(), #功能：水平翻转
                transforms.CenterCrop(size=256), #功能：根据给定的size从中心裁剪，size - 若为sequence,则为(h,w)，若为int，则(size,size)
                transforms.ToTensor(), #numpy --> tensor
                # 功能：对数据按通道进行标准化（RGB），即先减均值，再除以标准差
                transforms.Normalize([0.485, 0.456, 0.406],# mean
                                    [0.229, 0.224, 0.225])# std 
            ]),
            
            'val' : transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],# mean
                                    [0.229, 0.224, 0.225])# std 
            ]),
            
            'test' : transforms.Compose([
                transforms.Resize(300),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],# mean
                                    [0.229, 0.224, 0.225])# std 
            ])
        }

        # 从文件中读取数据
        self.datasets = {
            'train' : datasets.ImageFolder(train_dir, transform=image_transforms['train']), # 读取train中的数据集，并transform，将图片做变换
            'val' : datasets.ImageFolder(val_dir, transform=image_transforms['val']),  # 读取val中的数据集，并transform
            'test' : datasets.ImageFolder(test_dir, transform=image_transforms['test']) #  读取test中的数据集，并transform
        }

        # 定义BATCH_SIZE
        BATCH_SIZE = 16 # 每批读取128张图片
        # DataLoader : 创建iterator, 按批遍历读取数据
        self.dataloaders = {
            'train' : DataLoader(self.datasets['train'], batch_size=BATCH_SIZE, shuffle=True), # 训练集  shuffle：将图片进行打乱
            'val' : DataLoader(self.datasets['val'], batch_size=BATCH_SIZE, shuffle=True), # 验证集
            'test' : DataLoader(self.datasets['test'], batch_size=BATCH_SIZE, shuffle=True) # 测试集
        }

def dataloader():
    data = Createdataloaders()
    dataloaders = data.dataloaders
    # datasets = data.datasets
    return dataloaders

if __name__ == '__main__':
    dataloaders = dataloader()
    train_features, train_labels = next(iter(dataloaders['train']))
    print(train_labels)