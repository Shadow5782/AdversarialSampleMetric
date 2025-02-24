# Description: Functions for training a neural net
# Author: Johannes Geier
# Date: 19.12.2024

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import dataset
import os

# Set random seeds for reproducability
random_seed = 321
torch.manual_seed(random_seed)

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# One epoch train run
def train_net(epoch: int, network: nn.Module, optimizer: optim.Adam, train_loader: torch.utils.data.DataLoader, log_interval: int) -> list[float]:
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

        # Train network
        optimizer.zero_grad()
        y_pred = network(data)
        train_loss = F.cross_entropy(y_pred,target)
        train_loss.backward()
        optimizer.step()

        # Calc batch accuracy
        if cnt % log_interval == 0:
            acc = torch.sum(torch.argmax(y_pred,dim=1) == target).item()/len(data)
            print(f"Batch {cnt} \tTrain ACC: {acc} ")
            train_accs.append(acc)
        
        # Update batch counter
        cnt += 1
    
    return train_accs

# Test function that applies the test set to the trained net
def test_net(epoch: int, network: nn.Module, test_loader: torch.utils.data.DataLoader, datasource: str) -> tuple[float, float]:
    # Setup network and evaluation storage
    network.eval()
    loss = 0
    acc = 0
    # Iterate through test data
    for data, target, _ in test_loader:
        data = data.to(device)
        target = target.to(device)
        # Calc loss and accuracy for current batch
        y_pred = network(data)
        loss += F.cross_entropy(y_pred,target,reduction="sum").item()
        acc += torch.sum(torch.argmax(y_pred,dim=1) == target).item()
    # Calc loss and accuracy for all natches
    loss /= len(test_loader.dataset)
    acc /= len(test_loader.dataset)
    print(f"Epoch {epoch} \t{datasource} ACC: {acc} {datasource} Loss: {loss}")
    return loss, acc

# Main function managing epochs and tests
def run_nn(network: nn.Module, optimizer: optim.Adam, train_loader: torch.utils.data.DataLoader, test_loader: torch.utils.data.DataLoader, n_epochs: int, log_interval: int, test_datasource:str) -> tuple[list[float], list[float], list[float]]:
    # Prepare lists for loss and accuracy storage
    train_accs = []
    test_losses = []
    test_accs = []

    # Start training loop
    print("Started Training")
    for epoch in range(1,n_epochs+1):
        # Train NN
        train_acc = train_net(epoch,network,optimizer,train_loader,log_interval)
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
    data_path = "2_Code/data/"
    # Path to store the model and graphs to
    working_dir = "2_Code/evaluation_models/standardNet"
    # Name of the model
    model_name = "StandardWhiteBox"

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
    train_acc,test_acc,test_loss = run_nn(network,optimizer,train_loader,test_loader,num_epochs,log_interval,"Test")

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