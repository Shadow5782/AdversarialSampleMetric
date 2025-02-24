# Description: Script 1/3 to create a substitute dataset for practical black-/grey-box attacks: Oracle labeling
# Author: Johannes Geier
# Date: 20.12.2024

############################################################################################################################
# Procedure used to create substitute data:
# Nicolas Papernot, Patrick McDaniel, Ian Goodfellow, Somesh Jha, Z. Berkay Celik, and Ananthram Swami. 2017. 
# "Practical Black-Box Attacks against Machine Learning."
# In Proceedings of the 2017 ACM on Asia Conference on Computer and Communications Security (ASIA CCS ’17). 
# Abu Dhabi, United Arab Emirates, April 2 - April 6 2017. Association for Computing Machinery, New York, NY, USA, 14 Pages.
# https://doi.org/10.1145/3052973.3053009.
############################################################################################################################

import torch
import dataset
import os

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    # Path to targeted model
    target_network_path = "evaluation_models/advTrainDF/DFWhiteBox.pth"
    # Path to substitude training data
    substitude_data_path = "data/dfAdvTrain_subst"

    # Max size for one batch during labeling
    max_batch_size = 1000

    # Load targeted net for labeling
    from evaluation_models.WhiteBox_Model import Net
    network = Net()
    network.load_state_dict(torch.load(target_network_path,weights_only=True))
    network = network.to(device)

    # Create substitude dataset and dataloader
    substitude_data = dataset.CIFARDataset(substitude_data_path)
    substitude_dataloader = torch.utils.data.DataLoader(
        substitude_data,
        batch_size=substitude_data.amount_datapoints if substitude_data.amount_datapoints <= max_batch_size else max_batch_size, 
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_function
    )

    # List all names of substitute data files
    substitude_data_pathlist = os.listdir(substitude_data_path)

    # Start labeling
    batch_index = 0
    for images, _, _ in substitude_dataloader:
        # Move images to desired device
        images: torch.Tensor = images.to(device)
        # Use predicted label (oracle)
        outputs: torch.Tensor = network(images)
        _, outputs = torch.max(outputs, 1)

        # Store new labels
        for i in range(0,outputs.shape[0]):
            new_name = substitude_data_pathlist[batch_index+i].split("_")
            os.rename(f"{substitude_data_path}/{substitude_data_pathlist[batch_index+i]}",f"{substitude_data_path}/index_{new_name[1].removesuffix('.png')}_true_{outputs[i].item()}_target_{outputs[i].item()}.png")

        batch_index += max_batch_size