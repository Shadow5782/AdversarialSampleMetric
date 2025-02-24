# Description: Custom dataset for loading CIFAR10 images as png from a folder with a custom annotation
# Author: Johannes Geier
# Date: 19.12.2024

from torch.utils.data import Dataset
import torch
import torchvision
import os

class CIFARDataset(Dataset):
    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        self.image_paths = os.listdir(self.image_dir)
        # Check if the folder contains any non-image files
        for image_path in self.image_paths:
            if image_path[-4:] != ".png":
                raise Exception(f"Expected PNG Files in {self.image_dir}")
        self.amount_datapoints = len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Read image
        image = torchvision.io.decode_image(self.image_dir + "/" + self.image_paths[idx])
        # Normalize to [0, 1]
        image = image/255
        # Read true target and desired target
        target_string = self.image_paths[idx].split("_")
        true_target = torch.tensor(int(target_string[3]))
        adv_target = torch.tensor(int(target_string[5][:-4]))
        return image, true_target, adv_target
    
def collate_function(data: list[list[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Build Batch out of input list
    images = torch.stack([data[image][0] for image in range(0, len(data))])
    true_targets = torch.stack([data[target][1] for target in range(0, len(data))])
    adv_targets = torch.stack([data[target][2] for target in range(0, len(data))])

    return images, true_targets, adv_targets
