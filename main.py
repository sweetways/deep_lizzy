import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import train
import config
import configparser
from models import faster_rcnn

# 导入你的数据集、模型和训练/测试函数
# from your_dataset import YourDataset
# from your_models import YourModel1, YourModel2
# from your_train_test_functions import train, test


def read_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    # 读取默认的超参数
    batch_size = config.getint('DEFAULT', 'batch_size')
    learning_rate = config.getfloat('DEFAULT', 'learning_rate')
    num_epochs = config.getint('DEFAULT', 'num_epochs')
    
    # 读取模型特定的超参数
    model1_hidden_size = config.getint('MODEL1', 'hidden_size')
    model1_num_layers = config.getint('MODEL1', 'num_layers')
    model2_hidden_size = config.getint('MODEL2', 'hidden_size')
    model2_num_layers = config.getint('MODEL2', 'num_layers')
    
    # 将超参数保存在一个字典中并返回
    return {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'model1': {
            'hidden_size': model1_hidden_size,
            'num_layers': model1_num_layers,
        },
        'model2': {
            'hidden_size': model2_hidden_size,
            'num_layers': model2_num_layers,
        },
    }

def main():
    #read_config('config.ini')
    model = faster_rcnn.get_model(21)
    train.train(model)


if __name__ == "__main__":
    main()