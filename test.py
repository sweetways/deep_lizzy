import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

# 加载模型
model = torch.load('model.pth')
model.eval()

# 定义转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据
dataset = ImageFolder('path_to_your_dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 运行模型并保存结果
results = []
for inputs, labels in dataloader:
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    results.extend(preds.numpy())

# 可视化结果
plt.hist(results, bins=np.arange(min(results), max(results) + 2) - 0.5, rwidth=0.8)
plt.xticks(np.arange(min(results), max(results) + 1))
plt.title('Model predictions')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()

# 保存结果到文件
np.savetxt('results.txt', results, fmt='%d')