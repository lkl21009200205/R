import os
import json
import glob
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms

class FEMNISTMultiJsonDataset(Dataset):
    """加载并合并多个 FEMNIST JSON 文件"""

    def __init__(self, data_dir, transform=None):
        """
        data_dir: train/ 或 test/ 文件夹路径
        """
        self.transform = transform
        self.all_x = []
        self.all_y = []
        self.user_ids = []

        # 获取所有 json 文件
        json_pattern = os.path.join(data_dir, "*.json")
        json_files = sorted(glob.glob(json_pattern))

        if not json_files:
            raise FileNotFoundError(f"在 {data_dir} 中未找到任何 .json 文件")

        print(f"找到 {len(json_files)} 个 JSON 文件: {[os.path.basename(f) for f in json_files]}")

        # 遍历并合并所有文件
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            users = data.get('users', [])
            user_data = data.get('user_data', {})

            for user_id in users:
                x = user_data[user_id]['x']
                y = user_data[user_id]['y']

                self.all_x.extend(x)
                self.all_y.extend(y)
                self.user_ids.extend([user_id] * len(y))

        print(f"总共加载 {len(self.all_y)} 条样本")

    def __len__(self):
        return len(self.all_y)

    def __getitem__(self, idx):
        import numpy as np
        import torch

        img = np.array(self.all_x[idx], dtype=np.float32).reshape(28, 28) / 255.0
        label = self.all_y[idx]

        img = torch.from_numpy(img).unsqueeze(0)  # (1, 28, 28)

        if self.transform:
            img = self.transform(img)

        return img, label


# ==================== 联邦学习版本：保持用户分离 ====================

class FEMNISTFederatedDataset:
    """联邦学习场景：返回按用户划分的多个 Dataset"""

    def __init__(self, data_dir, transform=None):
        self.user_datasets = {}
        self.user_ids = []

        json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)

            for user_id in data.get('users', []):
                user_data = data['user_data'][user_id]
                self.user_datasets[user_id] = FEMNISTUserDataset(user_data, transform)
                self.user_ids.append(user_id)

    def get_user_dataset(self, user_id):
        return self.user_datasets[user_id]

    def get_all_datasets(self):
        """返回所有用户的 Dataset 列表"""
        return [self.user_datasets[uid] for uid in self.user_ids]

    def __len__(self):
        return len(self.user_ids)


class FEMNISTUserDataset(Dataset):
    """单个用户的本地数据"""

    def __init__(self, user_data, transform=None):
        self.x = user_data['x']
        self.y = user_data['y']
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        import numpy as np
        import torch

        img = np.array(self.x[idx], dtype=np.float32).reshape(28, 28) / 255.0
        label = self.y[idx]

        img = torch.from_numpy(img).unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        return img, label


# ==================== 修改后的 get_femnist 函数 ====================

def get_femnist(data_dir):
    """
    支持多个 JSON 文件的 FEMNIST 数据加载
    """
    mean = (0.1307,)
    std = (0.3081,)

    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Normalize(mean, std),
    ])

    # 自动检测：单个文件还是多个文件
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # 检查 train 目录下是单个文件还是多个文件
    train_files = glob.glob(os.path.join(train_dir, "*.json"))

    if len(train_files) == 1 and os.path.basename(train_files[0]) == 'train.json':
        # 单个文件模式（原始代码）
        train_dataset = FEMNISTDataset(train_files[0], transform_train)
        eval_dataset = FEMNISTDataset(os.path.join(test_dir, 'test.json'), transform_test)
    else:
        # 多个文件模式
        train_dataset = FEMNISTMultiJsonDataset(train_dir, transform_train)
        eval_dataset = FEMNISTMultiJsonDataset(test_dir, transform_test)

    return train_dataset, eval_dataset


# ==================== 联邦学习版本 ====================
