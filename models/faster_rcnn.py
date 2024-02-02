import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def get_model(num_classes):
    # 加载预训练的模型进行分类预测
    backbone = torchvision.models.resnet50(pretrained=True)

# Remove the last layer (the classifier)
    modules = list(backbone.children())[:-1]
    backbone = torch.nn.Sequential(*modules)
        
    # FasterRCNN需要知道骨干网络的输出通道数量。对于mobilenet_v2，它是1280，所以我们需要在这里添加它
    backbone.out_channels = 1280
    
    # 我们让RPN在每个空间位置生成5 x 3个锚点
    # 具有5种不同的大小和3种不同的宽高比。 
    # 我们有一个元组[元组[int]]
    # 因为每个特征映射可能具有不同的大小和宽高比
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # 定义一下我们将用于执行感兴趣区域裁剪的特征映射，以及重新缩放后裁剪的大小。 
    # 如果您的骨干返回Tensor，则featmap_names应为[0]。 
    # 更一般地，骨干应该返回一个OrderedDict[Tensor]
    # 并且在featmap_names中，你可以选择使用哪个特征映射。
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # 将这些pieces放在FasterRCNN模型中
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model