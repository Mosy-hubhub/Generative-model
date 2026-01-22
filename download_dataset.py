from torchvision.datasets import CIFAR10
import os

# 只下载，不创建Dataset
data_path = os.path.join('run', 'datasets', 'cifar10')
CIFAR10(data_path, train=True, download=True)
CIFAR10(data_path, train=False, download=True)  # 如果需要测试集