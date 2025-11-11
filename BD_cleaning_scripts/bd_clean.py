#!/usr/bin/env python3
"""
BD Clean Enhanced - PyTorch neural network model for cleaning BD data
Usage: python bd_clean_enhanced.py -i input.npz -o model.pt -n 2
"""

import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import numpy as np
import copy
import random


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
    """Custom R-squared based loss function."""
    SS_res = torch.sum(torch.square(y_true - y_pred)) 
    SS_tot = torch.sum(torch.square(y_true - torch.mean(y_true))) 
    loss2 = (1.0 - SS_res/(SS_tot + torch.finfo(torch.float32).eps))
    return -loss2


def corr_loss(y_true, y_pred):
    """Correlation-based loss function."""
    c = torch.corrcoef(torch.stack((torch.flatten(y_true), torch.flatten(y_pred)), dim=0))[1,0]
    return -c/(1-c)


def load_data(filename, train_length=None, verbose=False):
    """
    Load and preprocess data from npz file.

    Args:
        filename: Path to the npz file containing the data
        train_length: Optional length to truncate the third dimension (uses later part)
        verbose: Enable verbose output

    Returns:
        TensorDataset containing the preprocessed data
    """
    if verbose:
        print(f"Loading data from: {filename}")

    data_dict = np.load(filename)

    # Check available keys
    if verbose:
        print(f"Available data keys: {list(data_dict.keys())}")

    # Load and transpose data
    ori32 = torch.tensor(data_dict["ori64"].transpose((2,1,0))[:,:,:].astype(np.float32))
    sim32 = torch.tensor(data_dict["sim64"].transpose((2,1,0))[:,:,:].astype(np.float32))
    ori32means = torch.tensor(data_dict["ori64means"].transpose((2,1,0))[:,:,:].astype(np.float32))
    sim32means = torch.tensor(data_dict["sim64means"].transpose((2,1,0))[:,:,:].astype(np.float32))
    ori32sigmas = torch.tensor(data_dict["ori64sigmas"].transpose((2,1,0))[:,:,:].astype(np.float32))
    sim32sigmas = torch.tensor(data_dict["sim64sigmas"].transpose((2,1,0))[:,:,:].astype(np.float32))

    # If train_length is specified, truncate arrays to use later part of third dimension
    if train_length is not None:
        current_length = ori32.shape[2]
        if train_length < current_length:
            start_idx = current_length - train_length
            ori32 = ori32[:, :, start_idx:]
            sim32 = sim32[:, :, start_idx:]
            ori32means = ori32means[:, :, start_idx:]
            sim32means = sim32means[:, :, start_idx:]
            ori32sigmas = ori32sigmas[:, :, start_idx:]
            sim32sigmas = sim32sigmas[:, :, start_idx:]

            if verbose:
                print(f"Truncated third dimension from {current_length} to {train_length} (using later part)")

    if verbose:
        print(f"Data shape: {ori32.shape}")
        print(f"Data type: {ori32.dtype}")

    dataset = TensorDataset(ori32, sim32, ori32means, sim32means, ori32sigmas, sim32sigmas)

    return dataset


def train(model, device, train_loader, optimizer, epoch, train_size, batch_size, verbose=False):
    """
    Train the model for one epoch.
    
    Args:
        model: Neural network model
        device: Torch device (cpu/cuda/mps)
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        epoch: Current epoch number
        train_size: Size of training dataset
        batch_size: Batch size for training
        verbose: Enable verbose output
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (data, target, _, _, _, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = corr_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Print progress
        if verbose and batch_idx % max(1, int(train_size/batch_size/10)) == 0:
            progress = 100. * batch_idx * batch_size / train_size
            print(f'  Batch {batch_idx:3d}/{int(train_size/batch_size):3d} [{progress:5.1f}%]\tLoss: {loss.item()/(loss.item()-1):.6f}', end='\r')
        
        if batch_idx == int(train_size/batch_size):
            avg_corr = (total_loss/num_batches) / ((total_loss/num_batches) - 1)
            if verbose:
                print(f'  Epoch {epoch:3d} completed - Avg Correlation: {avg_corr:.6f}' + ' ' * 20)
            else:
                print(f'Train Epoch: {epoch} \t\tCorr: {loss.item()/(loss.item()-1):.6f}')
            break


def test(model, device, test_loader):
    """
    Test the model on test data.
    
    Args:
        model: Neural network model
        device: Torch device (cpu/cuda/mps)
        test_loader: DataLoader for test data
    
    Returns:
        Correlation value
    """
    model.eval()
    corr = 0
    with torch.no_grad():
        for data, target, _, _, _, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            corr = corr_loss(output, target)
    c = corr.item()/(corr.item()-1)
    return c


def evaluate_model(model, dataset, test_dataloader, verbose=False):
    """
    Evaluate the model and print correlation metrics.
    
    Args:
        model: Trained neural network model
        dataset: Full dataset
        test_dataloader: DataLoader for test data
        verbose: Enable verbose output
    """
    model.eval()
    
    if verbose:
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
    else:
        print("\nModel Evaluation:")
    
    # Get test data
    test_ori, test_sim, test_ori_mean, test_sim_mean, test_ori_sigmas, test_sim_sigmas = next(iter(test_dataloader))
    
    # Make predictions
    test_cleaned = model(test_ori)
    test_cleaned_denorm = test_cleaned * test_ori_sigmas + test_ori_mean
    test_ori_denorm = test_ori * test_ori_sigmas + test_ori_mean
    test_sim_denorm = test_sim * test_sim_sigmas + test_sim_mean
    
    # Calculate correlations
    corr_test = torch.corrcoef(torch.stack((torch.flatten(test_ori), torch.flatten(test_sim)), dim=0))[1,0].item()
    corr_test_denorm = torch.corrcoef(torch.stack((torch.flatten(test_ori_denorm), torch.flatten(test_sim_denorm)), dim=0))[1,0].item()
    corr_test_cleaned = torch.corrcoef(torch.stack((torch.flatten(test_cleaned), torch.flatten(test_sim)), dim=0))[1,0].item()
    corr_test_cleaned_denorm = torch.corrcoef(torch.stack((torch.flatten(test_cleaned_denorm), torch.flatten(test_sim_denorm)), dim=0))[1,0].item()
    
    # Print results
    if verbose:
        print("\nTest Set Correlations:")
        print(f"  Original (normalized):   {corr_test:.6f}")
        print(f"  Original (denormalized): {corr_test_denorm:.6f}")
        print(f"  Cleaned (normalized):    {corr_test_cleaned:.6f}")
        print(f"  Cleaned (denormalized):  {corr_test_cleaned_denorm:.6f}")
        
        improvement = (corr_test_cleaned - corr_test) / corr_test * 100
        print(f"\n  Improvement: {improvement:+.2f}%")
    else:
        print(f"Corr test: {corr_test:.6f}")
        print(f"Corr test denormed: {corr_test_denorm:.6f}")
        print(f"Corr test cleaned: {corr_test_cleaned:.6f}")
        print(f"Corr test cleaned denormed: {corr_test_cleaned_denorm:.6f}")
    
    # Full dataset correlation
    corr_dataset = torch.corrcoef(torch.stack((torch.flatten(dataset[:][0]), torch.flatten(dataset[:][1])), dim=0))[1,0].item()
    
    with torch.no_grad():
        dataset_cleaned = model(dataset[:][0])
        corr_dataset_cleaned = torch.corrcoef(torch.stack((torch.flatten(dataset_cleaned), torch.flatten(dataset[:][1])), dim=0))[1,0].item()
    
    if verbose:
        print("\nFull Dataset Correlations:")
        print(f"  Original: {corr_dataset:.6f}")
        print(f"  Cleaned:  {corr_dataset_cleaned:.6f}")
        print("="*60)
    else:
        print(f"Corr dataset: {corr_dataset:.6f}")
        print(f"Corr dataset cleaned: {corr_dataset_cleaned:.6f}")


def main():
    """Main function to handle command line arguments and run training."""
    
    # Create argument parser with detailed help
    parser = argparse.ArgumentParser(
        description='BD Clean Enhanced - PyTorch neural network model for cleaning BD data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic training:
    %(prog)s -i data.npz -o model.pt -n 2

  With custom hyperparameters:
    %(prog)s -i data.npz -o model.pt -n 3 --epochs 100 --batch-size 16

  With specific learning rates:
    %(prog)s -i data.npz -o model.pt -n 2 --lr-values 0.01,0.02,0.03

  With truncated training length:
    %(prog)s -i data.npz -o model.pt -n 2 --train-length 370

  Verbose mode for detailed output:
    %(prog)s -i data.npz -o model.pt -n 2 --verbose

  With reproducible results:
    %(prog)s -i data.npz -o model.pt -n 2 --seed 42

Notes:
  - The input file should be an npz file with specific data keys
  - The model will be saved as a PyTorch state dict
  - Multiple learning rates and epsilon values are tested
  - Best model is selected based on test correlation
  - Use verbose mode to see detailed training progress
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        dest='input_file',
        help='Input npz file containing the training data'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        dest='output_file',
        help='Output pt file to save the trained model'
    )
    
    parser.add_argument(
        '-n', '--rounds',
        type=int,
        required=True,
        help='Number of training rounds to perform'
    )
    
    # Training parameters
    training_group = parser.add_argument_group('training parameters')
    
    training_group.add_argument(
        '-b', '--batch-size',
        type=int,
        default=8,
        dest='batch_size',
        help='Batch size for training (default: 8)'
    )
    
    training_group.add_argument(
        '-e', '--epochs',
        type=int,
        default=50,
        help='Number of epochs per training run (default: 50)'
    )
    
    training_group.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        dest='train_split',
        help='Train/test split ratio (default: 0.8)'
    )

    training_group.add_argument(
        '--train-length',
        type=int,
        default=None,
        dest='train_length',
        help='Optional: truncate arrays to this length in third dimension (uses later part)'
    )

    # Optimizer parameters
    optimizer_group = parser.add_argument_group('optimizer parameters')
    
    optimizer_group.add_argument(
        '--lr-values',
        type=str,
        default='0.04,0.05,0.06',
        dest='lr_values',
        help='Comma-separated learning rates to try (default: 0.04,0.05,0.06)'
    )
    
    optimizer_group.add_argument(
        '--eps-values',
        type=str,
        default='0.0,1e-8,1e-7',
        dest='eps_values',
        help='Comma-separated epsilon values to try (default: 0.0,1e-8,1e-7)'
    )
    
    optimizer_group.add_argument(
        '--scheduler-step',
        type=int,
        default=20,
        dest='scheduler_step',
        help='Step size for learning rate scheduler (default: 20)'
    )
    
    optimizer_group.add_argument(
        '--scheduler-gamma',
        type=float,
        default=0.9,
        dest='scheduler_gamma',
        help='Gamma for learning rate scheduler (default: 0.9)'
    )
    
    # Other options
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output with detailed progress'
    )
    
    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit as e:
        if e.code == 2:  # argparse error code for missing arguments
            print("\nFor more information and examples, run: python bd_clean_enhanced.py --help", file=sys.stderr)
        raise
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Check output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        if args.verbose:
            print(f"Random seed set to: {args.seed}")
    
    # Parse lr and eps values
    try:
        lr_values = [float(x.strip()) for x in args.lr_values.split(',')]
        eps_values = [float(x.strip()) for x in args.eps_values.split(',')]
    except ValueError:
        print("Error: Invalid learning rate or epsilon values", file=sys.stderr)
        sys.exit(1)
    
    # Detect and set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_name = "MPS (Apple Silicon)"
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = "CUDA (GPU)"
    else:
        device = torch.device("cpu")
        device_name = "CPU"
    
    if args.verbose:
        print("="*60)
        print("BD CLEAN ENHANCED - TRAINING SESSION")
        print("="*60)
        print(f"\nSystem Information:")
        print(f"  Device: {device_name}")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"\nTraining Configuration:")
        print(f"  Rounds: {args.rounds}")
        print(f"  Epochs per round: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rates: {lr_values}")
        print(f"  Epsilon values: {eps_values}")
        print(f"  Train/test split: {args.train_split:.1%}/{(1-args.train_split):.1%}")
        print("="*60)
    else:
        print(f"Using {device_name}")
        print(f"PyTorch version: {torch.__version__}")
    
    try:
        # Load data
        if args.verbose:
            print("\n" + "="*60)
            print("DATA LOADING")
            print("="*60)
        else:
            print(f"\nLoading data from {args.input_file}...")
        
        dataset = load_data(args.input_file, train_length=args.train_length, verbose=args.verbose)
        
        # Split dataset
        train_size = int(args.train_split * len(dataset))
        test_size = len(dataset) - train_size
        
        if args.verbose:
            print(f"\nDataset Statistics:")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Training samples: {train_size}")
            print(f"  Test samples: {test_size}")
            print("="*60)
        else:
            print(f"Dataset size: {len(dataset)}")
            print(f"Train size: {train_size}, Test size: {test_size}")
        
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=test_size)
        
        # Print initial correlations
        train_corr = torch.corrcoef(torch.stack((torch.flatten(train_dataset[:][0]), torch.flatten(train_dataset[:][1])), dim=0))[1,0].item()
        test_corr = torch.corrcoef(torch.stack((torch.flatten(test_dataset[:][0]), torch.flatten(test_dataset[:][1])), dim=0))[1,0].item()
        
        if args.verbose:
            print("\nInitial Correlations:")
            print(f"  Training set: {train_corr:.6f}")
            print(f"  Test set: {test_corr:.6f}")
        else:
            print(f"Train set correlation: {train_corr:.4f}")
            print(f"Test set correlation: {test_corr:.4f}")
        
        # Initialize best model tracking
        best_c = [0.0]
        best_lr = []
        best_eps = []
        best_model = None
        total_configs = len(lr_values) * len(eps_values) * args.rounds
        config_count = 0
        
        # Training loop
        if args.verbose:
            print("\n" + "="*60)
            print("TRAINING PHASE")
            print("="*60)
            print(f"Testing {len(lr_values) * len(eps_values)} hyperparameter combinations over {args.rounds} rounds")
            print(f"Total training runs: {total_configs}")
        else:
            print(f"\nStarting training for {args.rounds} rounds...")
        
        for round_idx in range(args.rounds):
            if args.verbose:
                print(f"\n{'='*60}")
                print(f"ROUND {round_idx + 1}/{args.rounds}")
                print(f"{'='*60}")
            else:
                print(f"\n=== Round {round_idx + 1}/{args.rounds} ===")
            
            for lr in lr_values:
                for eps in eps_values:
                    config_count += 1
                    
                    if args.verbose:
                        print(f"\n[Config {config_count}/{total_configs}] Learning rate: {lr:.6f}, Epsilon: {eps:.1e}")
                        print("-" * 40)
                    else:
                        print(f"\nTraining with lr={lr:.6f} and eps={eps:.1e}")
                    
                    # Initialize model
                    model = Net().to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr, eps=eps)
                    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
                    
                    # Reset dataloader for each run
                    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                    
                    # Train
                    best_epoch_corr = 0
                    for epoch in range(1, args.epochs + 1):
                        train(model, device, train_dataloader, optimizer, epoch, train_size, args.batch_size, verbose=args.verbose)
                        c = test(model, device, test_dataloader)
                        
                        if c > best_epoch_corr:
                            best_epoch_corr = c
                        
                        if c > best_c[-1]:
                            if args.verbose:
                                print(f"\n{'*'*60}")
                                print(f"NEW BEST MODEL FOUND!")
                                print(f"  Correlation: {c:.6f}")
                                print(f"  Previous best: {best_c[-1]:.6f}")
                                print(f"  Improvement: {(c - best_c[-1]):.6f}")
                                print(f"  Configuration: lr={lr:.6f}, eps={eps:.1e}, epoch={epoch}")
                                print(f"{'*'*60}")
                            else:
                                print(f"***************** Found better test model: {c:.6f} *****************")
                            
                            best_c.append(c)
                            best_lr.append(lr)
                            best_eps.append(eps)
                            best_model = copy.deepcopy(model.state_dict())
                        
                        scheduler.step()
                    
                    if args.verbose:
                        print(f"Best correlation for this config: {best_epoch_corr:.6f}")
        
        # Save best model
        if best_model is not None:
            if args.verbose:
                print("\n" + "="*60)
                print("SAVING BEST MODEL")
                print("="*60)
            
            torch.save(best_model, args.output_file)
            
            if args.verbose:
                print(f"Model saved to: {args.output_file}")
                print(f"\nBest Model Statistics:")
                print(f"  Final correlation: {best_c[-1]:.6f}")
                print(f"  Learning rates history: {best_lr}")
                print(f"  Epsilon values history: {best_eps}")
                print(f"  Total improvements: {len(best_c) - 1}")
            else:
                print(f"\nSaving best model to {args.output_file}")
                print(f"\nBest model achieved correlation: {best_c[-1]:.6f}")
                print(f"Best learning rates used: {best_lr}")
                print(f"Best epsilon values used: {best_eps}")
            
            # Load and evaluate the best model
            model_saved = Net()
            model_saved.load_state_dict(best_model)
            model_saved.eval()
            
            evaluate_model(model_saved, dataset, test_dataloader, verbose=args.verbose)
            
            if args.verbose:
                print("\n" + "="*60)
                print("TRAINING COMPLETED SUCCESSFULLY")
                print("="*60)
        else:
            print("\nError: No model improvement found during training.", file=sys.stderr)
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\nError during training: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()