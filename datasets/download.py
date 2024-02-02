import os
import sys
import tarfile
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

DATASET_YEAR_DICT = {
    "VOC":{
        "2012": {
            "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
            "filename": "VOCtrainval_11-May-2012.tar",
            "md5": "6cd6e144f989b92b3379bac3b3de84fd",
            "base_dir": os.path.join("VOCdevkit", "VOC2012"),
        },
        "2007": {
            "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "filename": "VOCtrainval_06-Nov-2007.tar",
            "md5": "c52e279531787c972589f7e41ab4ae64",
            "base_dir": os.path.join("VOCdevkit", "VOC2007"),
        },
        "2007-test": {
            "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
            "filename": "VOCtest_06-Nov-2007.tar",
            "md5": "b6e924de25625d8de591ea690078ad9f",
            "base_dir": os.path.join("VOCdevkit", "VOC2007"),
        },
    },
    "COCO":{
        
    },
}

def download_dataset(dataset_name, year):
    # 从字典中获取数据集信息
    dataset_info = DATASET_YEAR_DICT[dataset_name][year]
    root = './datasets/VOC'
    # 获取下载 URL 和文件名
    url = dataset_info['url']
    filename = dataset_info['filename']
    md5 = dataset_info["md5"]

    # 检查文件是否已经存在
    if not os.path.exists(os.path.join(root, filename)):
        # 下载数据集
        print(f"Downloading {url}...")
        if dataset_name == "VOC":
            download_voc(url, root, filename, md5)
        elif dataset_name == "COCO":
            download_coco(url, root, filename, md5)
        print("Download complete.")
    else:
        print("文件已存在！")

def download_coco(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), 'r') as tar:
        tar.extractall(path = root)


def download_voc(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)



def main():
    dataset_name = "VOC"
    year = "2007"
    download_dataset(dataset_name, year)


if __name__ == '__main__':
    main()