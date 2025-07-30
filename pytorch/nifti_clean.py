#!/usr/bin/env python3
"""
Command-line tool for cleaning NIFTI files using a trained neural network model.
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


def apply_bm_ts(img, bm_path='/shared/home/zeming/utils/MNI152_T1_2mm_brain_mask.nii.gz'):
    bm_mask_raw = nib.load(bm_path)

    BM_THR = 0

    # Binarize
    bm_mask = image.resample_to_img(bm_mask_raw, img, interpolation='nearest').get_fdata()
    bm_mask = bm_mask > BM_THR
    affine = img.affine
    header = img.header
    # apply the gm mask to img
    img = img.get_fdata()
    img_masked = img[bm_mask, :]
    img = np.zeros_like(img)  # Same shape as the original 4D image
    # Place the masked data back into the 4D array at the corresponding voxel locations
    img[bm_mask, :] = img_masked

    img = nib.Nifti1Image(img, affine=affine, header=header)
    return img


def process_nifti(model_file, nifti_path, cleaned_nifti_path, batch_size=12, brain_mask_path=None):
    """
    Process a NIFTI file using the trained model.
    
    Args:
        model_file: Path to the trained model file
        nifti_path: Path to the input NIFTI file
        cleaned_nifti_path: Path to save the cleaned NIFTI file
        batch_size: Batch size for processing
        brain_mask_path: Optional path to brain mask file
    """
    # Find correct device
    if torch.backends.mps.is_available():
        print("Using MPS device")
        device = torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA device")
        device = torch.device("cuda")
    else:
        print("Using CPU device")
        device = torch.device("cpu")
    
    # Load model
    print(f"Loading model from {model_file}")
    model_saved = Net()
    model_saved.load_state_dict(torch.load(model_file, map_location=device))
    model_saved.eval()
    model_saved.to(device)
    
    # Load data
    print(f"Loading NIFTI file from {nifti_path}")
    img = image.load_img(nifti_path)
    
    # Apply brain mask
    if brain_mask_path:
        print(f"Applying brain mask from {brain_mask_path}")
        img = apply_bm_ts(img, brain_mask_path)
    else:
        print("Applying default brain mask")
        img = apply_bm_ts(img)
    
    # Process data
    print("Processing data...")
    data = img.get_fdata()
    X, Y, Z, T = data.shape
    
    # Calculate mean and standard deviation for each voxel across the time dimension
    voxel_means = np.mean(data, axis=3, keepdims=True)
    voxel_stds = np.std(data, axis=3, keepdims=True)
    
    data_dm = data - voxel_means  # De-mean the data
    data_dm = data_dm.reshape(-1, T)  # Reshape to 2D for processing
    
    # Normalize the data for each voxel
    data = (data - voxel_means) / (voxel_stds + 1e-8)  # Add a small value to avoid division by zero
    
    # Reshape the data
    data_2d = data.reshape(-1, T)
    means_2d = voxel_means.reshape(-1, 1)
    stds_2d = voxel_stds.reshape(-1, 1)
    
    # Create mask for valid data
    mask = ~np.all(data_2d == 0, axis=1)
    
    valid_data = data_2d[mask, :]
    # compute the mean and std of the valid data
    valid_data_demean = data_dm[mask, :]
    valid_means = means_2d[mask]
    valid_stds = stds_2d[mask]
    
    data_input = torch.from_numpy(valid_data).float().to(device)
    data_input = data_input.unsqueeze(1)  # Add channel dimension
    
    # Process in batches
    print(f"Processing {data_input.size(0)} voxels in batches of {batch_size}")
    outputs = []
    
    with torch.no_grad():
        for i in range(0, data_input.size(0), batch_size):
            batch = data_input[i:i+batch_size]
            out = model_saved(batch)
            outputs.append(out.cpu())
    
    outputs = torch.cat(outputs, dim=0).squeeze(1).numpy()
    
    # Demean the output along the 2 axis
    outputs_centered = outputs - np.mean(outputs, axis=1, keepdims=True)
    
    # Scale outputs
    bs = np.sum(valid_data_demean * outputs_centered, axis=1) / np.sum(outputs_centered ** 2, axis=1)
    
    # Reshape `bs` to match the dimensions of `outputs` for broadcasting
    outputs_scaled = outputs_centered * bs[:, np.newaxis] + valid_means
    
    # Reconstruct the full data
    result_denorm = np.zeros_like(data_2d)
    result_denorm[mask] = outputs_scaled
    
    result_denorm_4d = result_denorm.reshape(X, Y, Z, T)
    
    # Save the result
    new_img_denorm = nib.Nifti1Image(result_denorm_4d, affine=img.affine, header=img.header)
    print(f"Saving cleaned NIFTI file to {cleaned_nifti_path}")
    new_img_denorm.to_filename(cleaned_nifti_path)
    print("Processing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Clean NIFTI files using a trained neural network model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-m", "--model-file",
        type=str,
        required=True,
        help="Path to the trained model file (.pt)"
    )
    
    parser.add_argument(
        "-i", "--nifti-path",
        type=str,
        required=True,
        help="Path to the input NIFTI file"
    )
    
    parser.add_argument(
        "-o", "--cleaned-nifti-path",
        type=str,
        required=True,
        help="Path to save the cleaned NIFTI file"
    )
    
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=12,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "-bm", "--brain-mask",
        type=str,
        default=None,
        help="Path to brain mask file (optional, uses default if not provided)"
    )
    
    args = parser.parse_args()
    
    try:
        process_nifti(
            args.model_file,
            args.nifti_path,
            args.cleaned_nifti_path,
            args.batch_size,
            args.brain_mask
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()