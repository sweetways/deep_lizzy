# data_loader.py
import os
import yaml
from torchvision.datasets import ImageFolder

def load_datasets(config_path, transform):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get absolute path from relative path
    base_dir = os.path.dirname(os.path.abspath(config_path))
    train_path = os.path.join(base_dir, config['train_path'])

    train_dataset = ImageFolder(train_path, transform=transform)
    return train_dataset