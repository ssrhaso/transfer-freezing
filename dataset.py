import os
import torch
import glob
from torch.utils.data import Dataset
from torchvision import transforms

class GTZANDataset(Dataset):
    def __init__(self, root_dir, split='train', augment=False):
        self.split_dir = os.path.join(root_dir, split)
        self.files = glob.glob(os.path.join(self.split_dir, "*.pt"))
        self.augment = augment
        
        # ImageNet Normalization
        self.norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        item = torch.load(path)
        
        spec = item['data']  # Already normalized to [0, 1]
        label = item['label']
        
        # Add channel dimension: (1, 224, 224)
        spec = spec.unsqueeze(0)
        
        # Convert to RGB (3 channels)
        spec = spec.repeat(3, 1, 1)
        
        # Apply ImageNet normalization
        spec = self.norm(spec)
        
        return spec, label
