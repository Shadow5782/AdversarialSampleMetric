# Description: Functions which perform the PGD Untargeted L2 Attack
# Author: Johannes Geier
# Date: 19.12.2024

#################################################################################################
# Package used for attack: Torchattacks
# Kim Hoki. 2021. 
# "Torchattacks: A PyTorch Repository for Adversarial Attacks". 
# ArXiv-Preprint, 6 Pages. 
# https://doi.org/10.48550/arXiv.2010.01950
# https://github.com/Harry24k/adversarial-attacks-pytorch?tab=readme-ov-file

# PGD Attack:
# Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. 2018.
# "Towards Deep Learning Models Resistant to Adversarial Attacks". 
# In 6th International Conference on Learning Representations. 
# Vancouver, Canada, April 30 - May 3 2018. OpenReview, 23 Pages. 
# https://openreview.net/pdf?id=rJzIBfZAb.
#################################################################################################

import torch
from torch.utils.data import DataLoader
import torchvision
import torchattacks
import dataset
import os

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Function managing the attack on the model
def run_pgd_attack(model: torch.nn.Module, sample_path: str, storage_path: str, eps: float, alpha: float, iterations: int) -> tuple[int, int, int]:
    # Init PGD attack from torchattacks
    pgd_attack = torchattacks.PGDL2(model,eps=eps,alpha=alpha,steps=iterations)
    # Load images for attack
    data = dataset.CIFARDataset(sample_path)
    attack_dataloader = DataLoader(data,1024,False,collate_fn=dataset.collate_function)

    # Set model to eval
    model.eval()

    # Set counters for later metric calculation
    count_total = 0
    count_cor = 0
    count_adv = 0
    
    # Create folder for outputs
    if not os.path.isdir(storage_path):
        os.mkdir(storage_path)

    index = 0
    for images, true_labels, _ in attack_dataloader:
        # Move data to device
        images = images.to(device)
        true_labels = true_labels.to(device)

        # Perform attack
        adv_images = pgd_attack(images, true_labels)

    #     # Evaluate attack
    #     outputs = model(adv_images)
    #     _, predictions = torch.max(outputs, 1)
    #     count_total += adv_images.shape[0]
    #     count_cor += (predictions == true_labels).sum().item()

        # Store perturbated images
        for i in range(0,adv_images.shape[0]):
            torchvision.io.write_png((adv_images[i] * 255).to("cpu").type(torch.uint8),f"{storage_path}/index_{index}_true_{true_labels[i].item()}_target_{-1}.png")
            index += 1

    pgd_attack = 0

    # count_adv = count_total - count_cor

    return count_total, count_cor, count_adv

if __name__ == '__main__':
    from network_training import Net

    # Load net for testing
    network = Net()
    network.load_state_dict(torch.load("./2_Code/models/white_box_model/white_box_model.pth",weights_only=True))
    network = network.to(device)

    count_total, count_cor, count_adv = run_pgd_attack(network,sample_path="./2_Code/data/untargeted_data",storage_path="./2_Code/data/adv_samps",eps=0.1,alpha=2/255,iterations=40)

    print(count_total, count_cor, count_adv)