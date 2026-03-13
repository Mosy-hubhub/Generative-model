import os
import random
from torchvision.datasets import CIFAR10
from tqdm import tqdm

def export_cifar10_images(base_dir='cifar10_real_images', num_images=10000):
    # 1. 动态拼接带有数量后缀的文件夹名 (例如: cifar10_real_images_10000)
    output_dir = f"{base_dir}_{num_images}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading CIFAR-10 dataset...")
    # 拿到纯净的 PIL Image
    dataset = CIFAR10(root='./datasets', train=True, download=True)
    total_data = len(dataset)
    
    # 防呆设计：如果你要求的数量比数据集还大，就强制改成数据集最大容量
    num_images = min(num_images, total_data)
    
    print(f"Randomly selecting {num_images} out of {total_data} images...")
    
    # 2. 核心魔法：无放回地随机抽取指定数量的索引
    # 这样保证抽出来的图片绝对不会重复，且类别分布均匀
    random_indices = random.sample(range(total_data), num_images)
    
    print(f"Exporting {num_images} images to '{output_dir}'...")
    
    # 3. 遍历这些随机抽中的索引
    for i in tqdm(random_indices, desc=f"Saving {num_images} Real Images"):
        img, label = dataset[i]
        
        # 命名依然保留它原本在数据集里的绝对索引，方便溯源
        filename = f"{i:05d}_class_{label}.png"
        filepath = os.path.join(output_dir, filename)
        
        # 存为 PNG
        img.save(filepath, format='PNG')
        
    print(f"\n[Success] {num_images} real images safely exported to ./{output_dir}/")

if __name__ == "__main__":
    # 你可以在这里随便改数量！
    # 比如先搞个 10000 张用来做日常开发期的快速 FID 评测
    export_cifar10_images(num_images=5000)
    
    # 等你要终极打榜的时候，再跑一次：
    # export_cifar10_images(num_images=50000)