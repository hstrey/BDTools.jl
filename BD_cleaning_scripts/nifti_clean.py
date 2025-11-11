#!/usr/bin/env python3
"""
NIFTI Clean Enhanced - Neural network model for cleaning NIFTI imaging data
Usage: python nifti_clean_enhanced.py -m model.pt -i input.nii -o output.nii
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from nilearn import image
import nibabel as nib
import numpy as np


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


def apply_brain_mask(img, mask_path='/shared/home/zeming/utils/MNI152_T1_2mm_brain_mask.nii.gz', threshold=0):
    """
    Apply brain mask to the input image.
    
    Args:
        img: Input NIfTI image
        mask_path: Path to brain mask file
        threshold: Threshold value for mask binarization
    
    Returns:
        Masked NIfTI image
    """
    mask_raw = nib.load(mask_path)
    
    # Resample mask to match input image and binarize
    mask = image.resample_to_img(mask_raw, img, interpolation='nearest').get_fdata()
    mask = mask > threshold
    
    # Apply mask
    affine = img.affine
    header = img.header
    data = img.get_fdata()
    masked_data = data[mask, :]
    
    # Reconstruct masked image
    result = np.zeros_like(data)
    result[mask, :] = masked_data
    
    return nib.Nifti1Image(result, affine=affine, header=header)


def process_nifti(model_path, input_path, output_path, batch_size=12, mask_path=None, verbose=False):
    """
    Process a NIFTI file using the trained neural network model.
    
    Args:
        model_path: Path to the trained model file (.pt)
        input_path: Path to the input NIFTI file
        output_path: Path to save the cleaned NIFTI file
        batch_size: Batch size for processing
        mask_path: Optional path to brain mask file
        verbose: Enable verbose output
    
    Returns:
        None
    """
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
    
    if verbose:
        print(f"Device: {device_name}")
        print(f"PyTorch version: {torch.__version__}")
    
    # Load model
    if verbose:
        print(f"\nLoading model from: {model_path}")
    
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    # Load NIFTI data
    if verbose:
        print(f"Loading NIFTI file: {input_path}")
    
    img = image.load_img(input_path)
    
    # Apply brain mask if specified
    if mask_path:
        if verbose:
            print(f"Applying brain mask: {mask_path}")
        img = apply_brain_mask(img, mask_path)
    elif mask_path is None:
        # Use default mask path if it exists
        default_mask = '/shared/home/zeming/utils/MNI152_T1_2mm_brain_mask.nii.gz'
        try:
            if verbose:
                print(f"Applying default brain mask: {default_mask}")
            img = apply_brain_mask(img, default_mask)
        except FileNotFoundError:
            if verbose:
                print("No brain mask applied (default mask not found)")
    
    # Extract and prepare data
    data = img.get_fdata()
    X, Y, Z, T = data.shape
    
    if verbose:
        print(f"\nData dimensions: {X} x {Y} x {Z} x {T}")
        print(f"Total voxels: {X * Y * Z}")
        print(f"Time points: {T}")
    
    # Calculate statistics for normalization
    voxel_means = np.mean(data, axis=3, keepdims=True)
    voxel_stds = np.std(data, axis=3, keepdims=True)
    
    # Normalize data
    data_normalized = (data - voxel_means) / (voxel_stds + 1e-8)
    data_demeaned = data - voxel_means
    
    # Reshape for processing
    data_2d = data_normalized.reshape(-1, T)
    data_dm_2d = data_demeaned.reshape(-1, T)
    means_2d = voxel_means.reshape(-1, 1)
    stds_2d = voxel_stds.reshape(-1, 1)
    
    # Create mask for non-zero voxels
    mask = ~np.all(data_2d == 0, axis=1)
    valid_voxels = np.sum(mask)
    
    if verbose:
        print(f"Valid voxels: {valid_voxels} ({100 * valid_voxels / len(mask):.1f}%)")
    
    # Extract valid data
    valid_data = data_2d[mask, :]
    valid_data_dm = data_dm_2d[mask, :]
    valid_means = means_2d[mask]
    valid_stds = stds_2d[mask]
    
    # Prepare input tensor
    data_tensor = torch.from_numpy(valid_data).float().to(device)
    data_tensor = data_tensor.unsqueeze(1)  # Add channel dimension
    
    # Process in batches
    if verbose:
        print(f"\nProcessing {valid_voxels} voxels in batches of {batch_size}...")
    
    outputs = []
    num_batches = (data_tensor.size(0) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(0, data_tensor.size(0), batch_size):
            if verbose and i % (batch_size * 100) == 0:
                progress = min(100, 100 * i // data_tensor.size(0))
                print(f"Progress: {progress}%", end='\r')
            
            batch = data_tensor[i:i + batch_size]
            output = model(batch)
            outputs.append(output.cpu())
    
    if verbose:
        print("Progress: 100%")
    
    # Concatenate outputs
    outputs = torch.cat(outputs, dim=0).squeeze(1).numpy()
    
    # Post-process outputs
    outputs_centered = outputs - np.mean(outputs, axis=1, keepdims=True)
    
    # Scale outputs to match original data
    scaling_factors = np.sum(valid_data_dm * outputs_centered, axis=1) / np.sum(outputs_centered ** 2, axis=1)
    outputs_scaled = outputs_centered * scaling_factors[:, np.newaxis] + valid_means
    
    # Reconstruct full data
    result = np.zeros_like(data_2d)
    result[mask] = outputs_scaled
    result_4d = result.reshape(X, Y, Z, T)
    
    # Save cleaned NIFTI
    cleaned_img = nib.Nifti1Image(result_4d, affine=img.affine, header=img.header)
    
    if verbose:
        print(f"\nSaving cleaned NIFTI to: {output_path}")
    
    cleaned_img.to_filename(output_path)
    
    if verbose:
        print("Processing complete!")


def main():
    """Main function to handle command line arguments and run processing."""
    
    # Create argument parser with detailed help
    parser = argparse.ArgumentParser(
        description='NIFTI Clean Enhanced - Neural network model for cleaning NIFTI imaging data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage:
    %(prog)s -m model.pt -i input.nii -o output.nii
  
  With custom brain mask:
    %(prog)s -m model.pt -i input.nii -o output.nii --mask brain_mask.nii
  
  With larger batch size for faster processing:
    %(prog)s -m model.pt -i input.nii -o output.nii --batch-size 32
  
  Verbose mode for detailed output:
    %(prog)s -m model.pt -i input.nii -o output.nii --verbose

Notes:
  - The model file should be a PyTorch state dict saved with torch.save()
  - Input NIFTI files should be 4D (x, y, z, time)
  - Default brain mask is applied if available, use --no-mask to skip
  - Batch size affects memory usage and processing speed
        """
    )
    
    # Required arguments
    parser.add_argument(
        '-m', '--model',
        type=str,
        required=True,
        dest='model_path',
        help='Path to the trained PyTorch model file (.pt)'
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        dest='input_path',
        help='Path to the input NIFTI file to be cleaned'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        dest='output_path',
        help='Path where the cleaned NIFTI file will be saved'
    )
    
    # Optional arguments
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=12,
        dest='batch_size',
        help='Batch size for processing (default: 12)'
    )
    
    parser.add_argument(
        '--mask',
        type=str,
        default=None,
        dest='mask_path',
        help='Path to custom brain mask file (optional)'
    )
    
    parser.add_argument(
        '--no-mask',
        action='store_const',
        const=False,
        dest='mask_path',
        help='Skip brain mask application entirely'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output with processing details'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input file exists
    import os
    if not os.path.exists(args.input_path):
        print(f"Error: Input file not found: {args.input_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check output directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Run processing
        process_nifti(
            model_path=args.model_path,
            input_path=args.input_path,
            output_path=args.output_path,
            batch_size=args.batch_size,
            mask_path=args.mask_path,
            verbose=args.verbose
        )
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error during processing: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()