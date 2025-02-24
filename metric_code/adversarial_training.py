# Description: Functions for training a neural net with adversarial training
# Author: Johannes Geier
# Date: 16.02.2025

#################################################################################################
# Package used for PGD and CW attack: Torchattacks
# Kim Hoki. 2021. 
# "Torchattacks: A PyTorch Repository for Adversarial Attacks". 
# ArXiv-Preprint, 6 Pages. 
# https://doi.org/10.48550/arXiv.2010.01950
# https://github.com/Harry24k/adversarial-attacks-pytorch?tab=readme-ov-file

# PGD Attack:
# Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras and Adrian Vladu. 2018.
# "Towards Deep Learning Models Resistant to Adversarial Attacks". 
# In 6th International Conference on Learning Representations. 
# Vancouver, Canada, April 30 - May 3 2018. OpenReview, 23 Pages. 
# https://openreview.net/pdf?id=rJzIBfZAb.

# Carlini & Wagner Attack:
# Nicholas Carlini and David Wagner. 2017. 
# "Towards Evaluating the Robustness of Neural Networks". 
# In 2017 IEEE Symposium on Security and Privacy (SP). 
# San Jose, CA, USA, May 22 - May 25 2017. IEEE, Piscataway Township, NJ, USA, 19 Pages.
# https://doi.org/10.1109/SP.2017.49.

# DeepFool Attack:
# Seyed-Mohsen Moosavi-Dezfooli, Alhussein Fawzi and Pascal Frossard. 2016.
# "DeepFool: a simple and accurate method to fool deep neural networks"
# In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)
# Las Vegas, NV, USA, June 26 - July 1 2017. IEEE, Piscataway Township, NJ, USA, 19 Pages.
# https://openaccess.thecvf.com/content_cvpr_2016/html/Moosavi-Dezfooli_DeepFool_A_Simple_CVPR_2016_paper.html

# Concept of adversarial training:
# Alexey Kurakim, Ian J. Goodfellow and Samy Bengio. 2017.
# "Adversarial Machine Learning at Scale".
# In 5th International Conference on Learning Representations.
# Toulon, France, April 24 - April 26 2017. OpenReview, 17 Pages.
# https://openreview.net/pdf?id=BJm4T4Kgx.
#################################################################################################

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import dataset
import os
import torchattacks
from scipy.stats import truncnorm
from network_training import test_net

# Set random seeds for reproducability
random_seed = 321
torch.manual_seed(random_seed)

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# One epoch train run
def train_adv_net(epoch: int, network: nn.Module, optimizer: optim.Adam, train_loader: torch.utils.data.DataLoader, log_interval: int, generation_algo_mode: int, loss_weight: float) -> list[float]:
    # Setup network and evaluation storage
    network.train()
    train_accs = []
    print(f"Started Epoch {epoch}")
    # Counter for batches
    cnt = 0
    for data, target, _ in train_loader:
        # Move data to device
        data = data.to(device)
        target = target.to(device)

        # Create adversarial samples
        # Mode 0: PGD-Attack
        if generation_algo_mode == 0:
            # Split data 50/50
            data_pgd = data[:data.shape[0]//2]
            data = data[data.shape[0]//2:]

            # Split target data 50/50
            target_pgd = target[:data_pgd.shape[0]]
            target = target[data_pgd.shape[0]:]

            # Prepare attack
            eps = truncnorm.rvs(0, 2, loc=0, scale=8) # Max distance is randomly selected from a truncated normal distribution (Same as in cited paper)
            pgd_attack = torchattacks.PGDL2(network,eps=eps,alpha=2/255,steps=50)

            # Run attack
            pgd_output = pgd_attack(data_pgd,target_pgd)

            # Read size for loss calculation and set output
            size_data = data.shape[0]
            size_adv = data_pgd.shape[0]
            attack_output = pgd_output
            attack_target = target_pgd

        # Mode 1: CW-Attack
        elif generation_algo_mode == 1:
            data_cw = data[:data.shape[0]//2]
            data = data[data.shape[0]//2:]
            
            target_cw = target[:data_cw.shape[0]]
            target = target[data_cw.shape[0]:]

            cw_attack = torchattacks.CW(network,c=1,kappa=0,steps=50,lr=0.1)
            cw_attack.set_mode_targeted_by_label(quiet=True)

            # Randomize target labels for cw attack by shifting every target by a random number (ensures true != target)
            cw_attack_labels = target_cw + torch.tensor(np.random.randint(1,9,size=target_cw.shape[0]), device=device)
            cw_attack_labels = torch.where(cw_attack_labels > 9, cw_attack_labels - 10, cw_attack_labels)

            cw_output = cw_attack(data_cw,cw_attack_labels)

            size_data = data.shape[0]
            size_adv = data_cw.shape[0]
            attack_output = cw_output
            attack_target = target_cw

        # Mode 2: PGD-Attack and CW-Attack
        elif generation_algo_mode == 2:
            data_pgd = data[:data.shape[0]//4]
            data_cw = data[data.shape[0]//4:data.shape[0]//2]
            data = data[data.shape[0]//2:]

            target_pgd = target[:data_pgd.shape[0]]
            target_cw = target[data_pgd.shape[0]:data_pgd.shape[0]+data_cw.shape[0]]
            target = target[data_pgd.shape[0]+data_cw.shape[0]:]

            eps = truncnorm.rvs(0, 2, loc=0, scale=8)
            pgd_attack = torchattacks.PGDL2(network,eps=eps,alpha=2/255,steps=50)
            
            cw_attack = torchattacks.CW(network,c=1,kappa=0,steps=50,lr=0.1)
            cw_attack.set_mode_targeted_by_label(quiet=True)

            cw_attack_labels = target_cw + torch.tensor(np.random.randint(1,9,size=target_cw.shape[0]), device=device)
            cw_attack_labels = torch.where(cw_attack_labels > 9, cw_attack_labels - 10, cw_attack_labels)

            pgd_output = pgd_attack(data_pgd,target_pgd)
            cw_output = cw_attack(data_cw,cw_attack_labels)

            size_data = data.shape[0]
            size_adv = data_pgd.shape[0] + data_cw.shape[0]
            attack_output = torch.concat((pgd_output,cw_output),dim=0)
            attack_target = torch.concat((target_pgd,target_cw),dim=0)

        # Mode 3: DeepFool-Attack
        elif generation_algo_mode == 3:
            data_df = data[:data.shape[0]//2]
            data = data[data.shape[0]//2:]
            
            target_df = target[:data_df.shape[0]]
            target = target[data_df.shape[0]:]

            df_attack = torchattacks.DeepFool(network,steps=50,overshoot=0.02)

            df_output = df_attack(data_df,target_df)

            size_data = data.shape[0]
            size_adv = data_df.shape[0]
            attack_output = df_output
            attack_target = target_df

        else:
            raise Exception("generation_algo_mode not recognized")

        # Train network
        optimizer.zero_grad()
        y_pred = network(data)
        y_pred_adv = network(attack_output)
        train_loss:torch.Tensor = 1/(size_data + loss_weight * size_adv) * (F.cross_entropy(y_pred,target,reduction="sum") + loss_weight * F.cross_entropy(y_pred_adv,attack_target,reduction="sum"))
        train_loss.backward()
        optimizer.step()

        # Calc batch accuracy
        if cnt % log_interval == 0:
            acc = (torch.sum(torch.argmax(y_pred,dim=1) == target).item() + torch.sum(torch.argmax(y_pred_adv,dim=1) == attack_target).item())/(size_data + size_adv)
            print(f"Batch {cnt} \tTrain ACC: {acc} ")
            train_accs.append(acc)
        
        # Update batch counter
        cnt += 1
    
    return train_accs

# Main function managing epochs and tests
def run_advatrain_nn(network: nn.Module, optimizer: optim.Adam, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, n_epochs: int, log_interval: int, test_datasource:str, generation_algo_mode: int, loss_weight: float) -> tuple[list[float], list[float], list[float]]:
    # Prepare lists for loss and accuracy storage
    train_accs = []
    test_losses = []
    test_accs = []

    # Start training loop
    print("Started Training")
    for epoch in range(1,n_epochs+1):
        # Train NN
        train_acc = train_adv_net(epoch,network,optimizer,train_loader,log_interval,generation_algo_mode,loss_weight)
        # Store train ACCs
        train_accs.extend(train_acc)
        # Test NN
        test_loss, test_acc = test_net(epoch,network,test_loader,test_datasource)
        # Store test ACC and loss
        test_losses.append(test_loss)
        test_accs.append(test_acc)
    print("Finished Training")
    
    return train_accs,test_accs,test_losses

if __name__ == '__main__':
    # Path to the folder containing train and test data
    data_path = "data/"
    # Path to store the model and graphs to
    working_dir = "evaluation_models/advTrainDF"
    # Name of the model
    model_name = "DFWhiteBox"

    # Create working dir
    if not os.path.isdir(working_dir):
        os.mkdir(working_dir)

    # Create datasets
    train_data = dataset.CIFARDataset(data_path + "train")
    test_data = dataset.CIFARDataset(data_path + "test")

    # Setup hyperparameters
    batch_size = 512
    learning_rate = 0.001
    log_interval = 5
    num_epochs = 15

    # Create train data dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size, 
        shuffle=True,
        num_workers=5,
        collate_fn=dataset.collate_function
    )

    # Create test data dataloader
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1000, 
        shuffle=False,
        num_workers=2,
        collate_fn=dataset.collate_function
    )

    # Setup network and optimizer
    from evaluation_models.WhiteBox_Model import Net
    network = Net()
    network = network.to(device)
    optimizer = optim.Adam(params=network.parameters(),lr=learning_rate)

    # Train network with optimizer
    # generation_algo_mode: 0 = PGD Adversarial Training, 1 = CW Adversarial Training, 2 = PGD and CW Adversarial Training, 3 = DeepFool Adversarial Training
    train_acc,test_acc,test_loss = run_advatrain_nn(network,optimizer,train_loader,test_loader,num_epochs,log_interval,"Test",generation_algo_mode=3,loss_weight=0.3)

    # Evaluate the ACC on train data
    _, train_final_acc = test_net(0,network,train_loader,"Train")

    # Save model checkpoint
    torch.save(network.state_dict(), working_dir + "/" + model_name + ".pth")

    # Create evaluation graphs
    x_axis = []
    for i in range(0,num_epochs):
        for j in range(0,len(train_loader),log_interval):
            x_axis.append(i+(j*(1/len(train_loader))))

    fig, ax = plt.subplots(1,3,figsize=(19,5))

    # Graph for train ACC
    ax[0].plot(x_axis,train_acc,label="White Box Adam")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("ACC")
    ax[0].legend()
    ax[0].set_title("Train ACC")

    x_axis_epochs = np.arange(1,num_epochs+1)

    # Graph for test ACC
    ax[1].plot(x_axis_epochs,test_acc,label="White Box Adam")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("ACC")
    ax[1].set_xlim(1,num_epochs)
    ax[1].set_title("Test ACC")
    ax[1].legend()

    # Graph for test loss
    ax[2].plot(x_axis_epochs,test_loss, label="White Box Adam")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Loss")
    ax[2].set_xlim(1,num_epochs)
    ax[2].set_title("Test Cross Entropy Loss")
    ax[2].legend()

    # Save plot
    fig.suptitle(f"Train ACC: {train_final_acc:.4} / Test ACC: {test_acc[-1]:.4}")
    plt.savefig(working_dir + "/" + model_name + ".png")