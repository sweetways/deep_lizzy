# 导入必要的库
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载COCO数据集
coco_train_dataset = CocoDetection(root='path/to/coco/train', annFile='path/to/coco/annotations/instances_train.json', transform=ToTensor())
coco_test_dataset = CocoDetection(root='path/to/coco/val', annFile='path/to/coco/annotations/instances_val.json', transform=ToTensor())

# 创建数据加载器
train_dataloader = DataLoader(coco_train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
test_dataloader = DataLoader(coco_test_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# 定义 Faster R-CNN 模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 91  # COCO dataset has 80 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# 将模型移动到设备
model.to(device)

# 定义优化器和学习率调度器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0005)

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_dataloader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # 更新学习率
    lr_scheduler.step()

# 测试模型
model.eval()
results = []
for images, targets in test_dataloader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    with torch.no_grad():
        outputs = model(images)

    results.append(outputs)