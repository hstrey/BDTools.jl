#!/usr/bin/env python3
"""
BD Clean - PyTorch model for cleaning BD data
Usage: python bd_clean.py -i gt_clean.npz -o bd_clean.pt -n 2
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
import copy
import random
import sys


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 18, 9, padding="same")
        self.conv2 = nn.Conv1d(18, 18, 9, padding="same")
        self.conv3 = nn.Conv1d(18, 1, 1, padding="same")
        self.bn = nn.BatchNorm1d(18)
        self.dropout1 = nn.Dropout(0.10)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.sigmoid(x)
        
        x = self.conv2(x)
        x = self.bn(x)
        x = F.sigmoid(x)

        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = F.sigmoid(x)

        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = F.sigmoid(x)
        
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = F.sigmoid(x)

        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = F.sigmoid(x)
        
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = F.sigmoid(x)

        x = self.dropout1(x)
        
        output = self.conv3(x)
        
        return output


def custom_loss(y_true, y_pred):
    SS_res = torch.sum(torch.square(y_true - y_pred)) 
    SS_tot = torch.sum(torch.square(y_true - torch.mean(y_true))) 
    loss2 = (1.0 - SS_res/(SS_tot + torch.finfo(torch.float32).eps))
    return -loss2


def corr_loss(y_true, y_pred):
    c = torch.corrcoef(torch.stack((torch.flatten(y_true), torch.flatten(y_pred)), dim=0))[1,0]
    return -c/(1-c)


def load_data(filename):
    """Load and preprocess data from npz file"""
    data_dict = np.load(filename)
    
    ori32 = torch.tensor(data_dict["ori64"].transpose((2,1,0))[:,:,:].astype(np.float32))
    sim32 = torch.tensor(data_dict["sim64"].transpose((2,1,0))[:,:,:].astype(np.float32))
    ori32means = torch.tensor(data_dict["ori64means"].transpose((2,1,0))[:,:,:].astype(np.float32))
    sim32means = torch.tensor(data_dict["sim64means"].transpose((2,1,0))[:,:,:].astype(np.float32))
    ori32sigmas = torch.tensor(data_dict["ori64sigmas"].transpose((2,1,0))[:,:,:].astype(np.float32))
    sim32sigmas = torch.tensor(data_dict["sim64sigmas"].transpose((2,1,0))[:,:,:].astype(np.float32))
    
    dataset = TensorDataset(ori32, sim32, ori32means, sim32means, ori32sigmas, sim32sigmas)
    
    return dataset


def train(model, device, train_loader, optimizer, epoch, train_size, batch_size):
    model.train()
    for batch_idx, (data, target, _, _, _, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = corr_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx == int(train_size/batch_size):
            print('Train Epoch: {} \t\tCorr: {:.6f}'.format(
                epoch, loss.item()/(loss.item()-1)))


def test(model, device, test_loader):
    model.eval()
    corr = 0
    with torch.no_grad():
        for data, target, _, _, _, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            corr = corr_loss(output, target)
    c = corr.item()/(corr.item()-1)
    return c


def evaluate_model(model, dataset, test_dataloader):
    """Evaluate the model and print correlation metrics"""
    model.eval()
    
    # Get test data
    test_ori, test_sim, test_ori_mean, test_sim_mean, test_ori_sigmas, test_sim_sigmas = next(iter(test_dataloader))
    
    # Make predictions
    test_cleaned = model(test_ori)
    test_cleaned_denorm = test_cleaned * test_ori_sigmas + test_ori_mean
    test_ori_denorm = test_ori * test_ori_sigmas + test_ori_mean
    test_sim_denorm = test_sim * test_sim_sigmas + test_sim_mean
    
    # Print correlations
    print("\nModel Evaluation:")
    print("Corr test: ", torch.corrcoef(torch.stack((torch.flatten(test_ori), torch.flatten(test_sim)), dim=0))[1,0].item())
    print("Corr test denormed: ", torch.corrcoef(torch.stack((torch.flatten(test_ori_denorm), torch.flatten(test_sim_denorm)), dim=0))[1,0].item())
    print("Corr test cleaned: ", torch.corrcoef(torch.stack((torch.flatten(test_cleaned), torch.flatten(test_sim)), dim=0))[1,0].item())
    print("Corr test cleaned denormed: ", torch.corrcoef(torch.stack((torch.flatten(test_cleaned_denorm), torch.flatten(test_sim_denorm)), dim=0))[1,0].item())
    
    # Full dataset correlation
    print("Corr dataset: ", torch.corrcoef(torch.stack((torch.flatten(dataset[:][0]), torch.flatten(dataset[:][1])), dim=0))[1,0].item())
    with torch.no_grad():
        dataset_cleaned = model(dataset[:][0])
        print("Corr dataset cleaned: ", torch.corrcoef(torch.stack((torch.flatten(dataset_cleaned), torch.flatten(dataset[:][1])), dim=0))[1,0].item())


def main():
    parser = argparse.ArgumentParser(description='BD Clean - PyTorch model for cleaning BD data')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input npz file containing the data')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output pt file to save the trained model')
    parser.add_argument('-n', '--rounds', type=int, required=True,
                        help='Number of training rounds')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs per training run (default: 50)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='Train/test split ratio (default: 0.8)')
    parser.add_argument('--lr-values', type=str, default='0.04,0.05,0.06',
                        help='Learning rates to try (comma-separated, default: 0.04,0.05,0.06)')
    parser.add_argument('--eps-values', type=str, default='0.0,1e-8,1e-7',
                        help='Epsilon values to try (comma-separated, default: 0.0,1e-8,1e-7)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    # Parse lr and eps values
    lr_values = [float(x) for x in args.lr_values.split(',')]
    eps_values = [float(x) for x in args.eps_values.split(',')]
    
    # Check device
    if torch.backends.mps.is_available():
        print("Using MPS device")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA device")
        device = torch.device("cuda")
    else:
        print("Using CPU device")
        device = torch.device("cpu")
    
    print(f"PyTorch version: {torch.__version__}")
    
    # Load data
    print(f"\nLoading data from {args.input}...")
    dataset = load_data(args.input)
    
    # Split dataset
    train_size = int(args.train_split * len(dataset))
    test_size = len(dataset) - train_size
    print(f"Dataset size: {len(dataset)}")
    print(f"Train size: {train_size}, Test size: {test_size}")
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_size)
    
    # Print initial correlations
    print(f"Train set correlation: {torch.corrcoef(torch.stack((torch.flatten(train_dataset[:][0]), torch.flatten(train_dataset[:][1])), dim=0))[1,0].item():.4f}")
    print(f"Test set correlation: {torch.corrcoef(torch.stack((torch.flatten(test_dataset[:][0]), torch.flatten(test_dataset[:][1])), dim=0))[1,0].item():.4f}")
    
    # Initialize best model tracking
    best_c = [0.0]
    best_lr = []
    best_eps = []
    best_model = None
    
    # Training loop
    print(f"\nStarting training for {args.rounds} rounds...")
    for round_idx in range(args.rounds):
        print(f"\n=== Round {round_idx + 1}/{args.rounds} ===")
        
        for lr in lr_values:
            for eps in eps_values:
                print(f"\nTraining with lr={lr:.6f} and eps={eps:.1e}")
                
                # Initialize model
                model = Net().to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
                scheduler = StepLR(optimizer, step_size=20, gamma=0.9)
                
                # Reset dataloader for each run
                train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                
                # Train
                for epoch in range(1, args.epochs + 1):
                    train(model, device, train_dataloader, optimizer, epoch, train_size, args.batch_size)
                    c = test(model, device, test_dataloader)
                    
                    if c > best_c[-1]:
                        print(f"***************** Found better test model: {c:.6f} *****************")
                        best_c.append(c)
                        best_lr.append(lr)
                        best_eps.append(eps)
                        best_model = copy.deepcopy(model.state_dict())
                    
                    scheduler.step()
    
    # Save best model
    if best_model is not None:
        print(f"\nSaving best model to {args.output}")
        torch.save(best_model, args.output)
        
        print(f"\nBest model achieved correlation: {best_c[-1]:.6f}")
        print(f"Best learning rates used: {best_lr}")
        print(f"Best epsilon values used: {best_eps}")
        
        # Load and evaluate the best model
        model_saved = Net()
        model_saved.load_state_dict(best_model)
        model_saved.eval()
        
        evaluate_model(model_saved, dataset, test_dataloader)
    else:
        print("No model improvement found during training.")
        sys.exit(1)


if __name__ == "__main__":
    main()