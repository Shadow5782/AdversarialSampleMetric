# Description: Small script to load original CIFAR10 dataset and store each image as png
# Author: Johannes Geier
# Date: 18.12.2024

import torchvision
import torch
import os

# Path to folder which will hold the original CIFAR10 files
data_path = "./2_Code/data/white_box_data"

# Path to folder where the train and test pngs will be stored
output_path = "./2_Code/data"

# Load original data
train_data = torchvision.datasets.CIFAR10(data_path,train=True,download=True,transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(data_path,train=False,download=True,transform=torchvision.transforms.ToTensor())

# Write train images
if not os.path.isdir(output_path + "/train/"):
    os.mkdir(output_path + "/train/")
for i in range(0,train_data.data.shape[0]):
    torchvision.io.write_png(torch.tensor(train_data.data[i].transpose(2,0,1)),f"{output_path}/train/index_{i}_true_{train_data.targets[i]}_target_{train_data.targets[i]}.png")

# Write test images
if not os.path.isdir(output_path + "/test/"):
    os.mkdir(output_path + "/test/")
for i in range(0,test_data.data.shape[0]):
    torchvision.io.write_png(torch.tensor(test_data.data[i].transpose(2,0,1)),f"{output_path}/test/index_{i}_true_{test_data.targets[i]}_target_{test_data.targets[i]}.png")