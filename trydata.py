import os
import json
import glob
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms

class FEMNISTMultiJsonDataset(Dataset):
    """Load and merge multiple FEMNIST JSON files."""

    def __init__(self, data_dir, transform=None):
        """
        data_dir: path to the train/ or test/ directory.
        """
        self.transform = transform
        self.all_x = []
        self.all_y = []
        self.user_ids = []

        # Get all JSON files.
        json_pattern = os.path.join(data_dir, "*.json")
        json_files = sorted(glob.glob(json_pattern))

        if not json_files:
            raise FileNotFoundError(f"No .json files found in {data_dir}")

        print(f"Found {len(json_files)} JSON files: {[os.path.basename(f) for f in json_files]}")

        # Iterate over and merge all files.
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

        print(f"Loaded {len(self.all_y)} samples in total")

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


# ==================== Federated learning version: keep users separate ====================

class FEMNISTFederatedDataset:
    """Federated learning scenario: return multiple datasets split by user."""

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
        """Return the list of datasets for all users."""
        return [self.user_datasets[uid] for uid in self.user_ids]

    def __len__(self):
        return len(self.user_ids)


class FEMNISTUserDataset(Dataset):
    """Local data for a single user."""

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


# ==================== Modified get_femnist function ====================

def get_femnist(data_dir):
    """
    Load FEMNIST data from multiple JSON files.
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

    # Automatically detect whether the dataset uses a single file or multiple files.
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # Check whether the train directory contains a single file or multiple files.
    train_files = glob.glob(os.path.join(train_dir, "*.json"))

    if len(train_files) == 1 and os.path.basename(train_files[0]) == 'train.json':
        # Single-file mode from the original code.
        train_dataset = FEMNISTDataset(train_files[0], transform_train)
        eval_dataset = FEMNISTDataset(os.path.join(test_dir, 'test.json'), transform_test)
    else:
        # Multi-file mode.
        train_dataset = FEMNISTMultiJsonDataset(train_dir, transform_train)
        eval_dataset = FEMNISTMultiJsonDataset(test_dir, transform_test)

    return train_dataset, eval_dataset


# ==================== Federated learning version ====================
