# train.py
import os
import torch
import logging
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch import optim, nn


from utils.logger import Logger

def train(model, num_epochs=100, batch_size=64, learning_rate=0.001):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建 Logger 对象
    logger = Logger(f'{checkpoint_dir}/train.log', logging.INFO, 'train').get_log()

    # 数据加载和预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder('path_to_your_train_dataset', transform=transform)
    val_dataset = ImageFolder('path_to_your_val_dataset', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 将模型移动到指定设备
    model = model.to(device)

    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # 获取模型的类名并创建一个目录来保存训练断点
    model_name = model.__class__.__name__
    checkpoint_dir = f'logs/{model_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证循环
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        train_acc = correct / total
        logger.info(f'Epoch: {epoch}, Train Loss: {loss.item()}, Train Accuracy: {train_acc}')

        # 保存训练断点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'{checkpoint_dir}/checkpoint_{epoch}.pth')
