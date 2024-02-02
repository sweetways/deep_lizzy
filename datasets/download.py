import os
import sys
import tarfile
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity



def download_voc12(url, root, filename, md5):
    download_url(url, root, filename, md5)
    with tarfile.open(os.path.join(root, filename), "r") as tar:
        tar.extractall(path=root)

def main():
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    filename = 'VOCtrainval_11-May-2012.tar'
    md5 = '6cd6e144f989b92b3379bac3b3de84fd'
    download_voc12(url, './VOC', filename, md5)
    
if __name__ == '__main__':
    main()