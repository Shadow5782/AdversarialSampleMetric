# Description: Functions which perform the Carlini & Wagner Targeted L2 Attack
# Author: Johannes Geier
# Date: 19.12.2024

########################################################################################
# Package used for attack: Torchattacks
# Kim Hoki. 2021. 
# "Torchattacks: A PyTorch Repository for Adversarial Attacks". 
# ArXiv-Preprint, 6 Pages. 
# https://doi.org/10.48550/arXiv.2010.01950
# https://github.com/Harry24k/adversarial-attacks-pytorch?tab=readme-ov-file

# Carlini & Wagner Attack:
# Nicholas Carlini and David Wagner. 2017. 
# "Towards Evaluating the Robustness of Neural Networks". 
# In 2017 IEEE Symposium on Security and Privacy (SP). 
# San Jose, CA, USA, May 22 - May 25 2017. IEEE, Piscataway Township, NJ, USA, 19 Pages.
# https://doi.org/10.1109/SP.2017.49.
########################################################################################

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
def run_cw_attack(model: torch.nn.Module, sample_path: str, storage_path: str, c: float, kappa: float, iterations: int, alpha: float, eps: float) -> tuple[int, int, int]:
    # Init CW attack from torchattacks
    cw_attack = torchattacks.CW(model,c=c,kappa=kappa,steps=iterations,lr=alpha)
    # cw_attack.set_mode_targeted_random(quiet=True)
    cw_attack.set_mode_targeted_by_label(quiet=True)

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
    for images, true_labels, target_labels in attack_dataloader:
        # Move data to device
        images = images.to(device)
        target_labels = target_labels.to(device)

        # Perform attack
        adv_images = cw_attack(images, target_labels)
        
        # Calculate perturbation deltas for every image
        delta = adv_images - images
        delta_norms = torch.norm(delta.view(images.shape[0], -1), p=2, dim=1)
        
        # Sort out perturbations greater than the allowed eps
        sorted_adv_images = []
        for i in range(0, images.shape[0]):
            if delta_norms[i].item() <= eps:
                sorted_adv_images.append(adv_images[i].clone())
            else:
                sorted_adv_images.append(images[i].clone())
        adv_images = torch.stack(sorted_adv_images)

    #     # Evaluate attack
    #     outputs = model(adv_images)
    #     _, predictions = torch.max(outputs, 1)
    #     count_total += adv_images.shape[0]
    #     count_adv += (predictions == target_labels).sum().item()

        # Store perturbated images
        for i in range(0,adv_images.shape[0]):
            torchvision.io.write_png((adv_images[i]*255).to("cpu").type(torch.uint8),f"{storage_path}/index_{index}_true_{true_labels[i].item()}_target_{target_labels[i].item()}.png")
            index += 1

    # count_cor = count_total - count_adv

    return count_total, count_cor, count_adv

if __name__ == '__main__':
    from network_training import Net

    # Load net for testing
    network = Net()
    network.load_state_dict(torch.load("./2_Code/models/white_box_model/white_box_model.pth",weights_only=True))
    network = network.to(device)

    count_total, count_cor, count_adv = run_cw_attack(network,sample_path="./2_Code/data/targeted_data",storage_path="./2_Code/data/adv_samps",c=1,kappa=20,iterations=1000,alpha=0.1,eps=0.5)

    print(count_total, count_cor, count_adv)