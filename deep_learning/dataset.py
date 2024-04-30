from typing import List, Dict

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class FacesDataset(Dataset):
    """Create a PyTorch Dataset with custom data."""
    def __init__(self, image_path: List[str], label_hash: Dict[str, str]):
        self.image_paths = image_path
        self.label_hash = label_hash

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        label = img_path.split('_')[1].split('.')[0]
        image = Image.open(img_path).convert('RGB')

        # Convert into a three dimensional tensor
        image = torch.from_numpy(np.array(image)).float()
        image = image.permute(2, 0, 1)

        # Store pixel data and labels
        sample = {'image': image, 'label': self.label_hash[label]}
        return sample
