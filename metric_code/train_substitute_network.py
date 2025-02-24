# Description: Script 2/3 to create a substitute dataset for practical black-/grey-box attacks: Substitude model training
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
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import dataset
import os
from network_training import run_nn

random_seed = 321
torch.manual_seed(random_seed)

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
if __name__ == "__main__":
    # Path to substitute data
    substitude_data_path = "data/dfAdvTrain_subst"
    # Path to where model checkpoints and evaluation will be stored
    working_dir = "evaluation_models/advTrainDF"
    # Name of the model
    model_name = "DFBlackBox"

    # Create working dir folder
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    # Create dataset
    train_data = dataset.CIFARDataset(substitude_data_path)

    # Setup hyperparameters
    batch_size = 256
    learning_rate = 0.001
    log_interval = 1
    num_epochs = 20

    # Create train data dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=5,
        collate_fn=dataset.collate_function
    )

    # Setup network and optimizer
    from evaluation_models.BlackBox_Model import SubstituteNet # Black Box Model
    # from evaluation_models.GreyBox_Model import SubstituteNet # Grey Box Model
    network = SubstituteNet()
    network = network.to(device)
    optimizer = optim.Adam(params=network.parameters(),lr=learning_rate)

    # Train network with optimizer
    train_acc,test_acc,test_loss = run_nn(network,optimizer,train_loader,train_loader,num_epochs,log_interval,"Train")

    # Save model checkpoint
    torch.save(network.state_dict(), working_dir + "/" + model_name + ".pth")

    # Create evaluation graphs
    x_axis = []
    for i in range(0,num_epochs):
        for j in range(0,len(train_loader),log_interval):
            x_axis.append(i+(j*(1/len(train_loader))))

    fig, ax = plt.subplots(1,3,figsize=(19,5))

    # Graph for train ACC per batch
    ax[0].plot(x_axis,train_acc,label="Subst Adam")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("ACC")
    ax[0].legend()
    ax[0].set_title("Train ACC")

    x_axis_epochs = np.arange(1,num_epochs+1)

    # Graph train ACC per epoch
    ax[1].plot(x_axis_epochs,test_acc,label="Subst Adam")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("ACC")
    ax[1].set_xlim(1,num_epochs)
    ax[1].set_title("Full Train ACC")
    ax[1].legend()

    # Graph train loss per epoch
    ax[2].plot(x_axis_epochs,test_loss, label="Subst Adam")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Loss")
    ax[2].set_xlim(1,num_epochs)
    ax[2].set_title("Full Train Cross Entropy Loss")
    ax[2].legend()

    # Save plot
    fig.suptitle(f"Train ACC: {test_acc[-1]:.4}")
    plt.savefig(working_dir + "/" + model_name + ".png")