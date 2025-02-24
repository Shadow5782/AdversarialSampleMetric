# Description: Script 3/3 to create a substitute dataset for practical black-/grey-box attacks: Jacobian data augmentation
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
import torchvision
import dataset

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    # Path to substitute data
    substitude_data_path = "data/dfAdvTrain_subst"
    # Path to targeted network
    target_network_path = "evaluation_models/advTrainDF/DFWhiteBox.pth"

    # Step size hyperparameter for jacobi augmentation (Same as in cited paper)
    step_size = 0.1

    # Create substitude dataset and dataloader
    substitude_data = dataset.CIFARDataset(substitude_data_path)
    substitude_data_loader = torch.utils.data.DataLoader(
        substitude_data,
        batch_size=1, 
        shuffle=True,
        num_workers=1,
        collate_fn=dataset.collate_function
    )

    # Load net for testing
    from evaluation_models.WhiteBox_Model import Net
    network = Net()
    network.load_state_dict(torch.load(target_network_path,weights_only=True))
    network = network.to(device)

    network.eval()

    # Start augmentation loop
    storage_index = substitude_data.amount_datapoints
    for images, _, _ in substitude_data_loader:
        # Move images to desired device
        images: torch.Tensor = images.to(device)

        # Calculate jacobi matrix
        jacobian: torch.Tensor = torch.autograd.functional.jacobian(network, images)
        jacobian = jacobian.reshape(jacobian.shape[1],jacobian.shape[3],jacobian.shape[4],jacobian.shape[5])
        
        # Calculate augmented images 
        images = images.repeat(jacobian.shape[0],1,1,1)
        augmentend_images = images + step_size * jacobian.sign()

        # Store augmented images
        for i in range(augmentend_images.shape[0]):
            torchvision.io.write_png((augmentend_images[i]*255).type(torch.uint8).to("cpu"),f"{substitude_data_path}/index_{storage_index}_true_0_target_0.png")
            storage_index += 1
    
