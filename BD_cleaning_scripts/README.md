# BD Cleaning Scripts

This directory contains Python scripts for training and applying neural network models to clean BD (Bloch-Siegert) imaging data and NIFTI files.

## Contents

- `bd_clean.py` - Train neural network models on BD data
- `nifti_clean.py` - Apply trained models to clean NIFTI imaging files
- `environment.yml` - Conda environment specification

## Setup

### Creating the Conda Environment

This project requires Python 3.12 and various scientific computing libraries. The easiest way to set up the environment is using conda with the provided `environment.yml` file.

#### Step 1: Install Conda

If you don't have conda installed, download and install [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

#### Step 2: Create the Environment

Navigate to this directory and create the environment from the `environment.yml` file:

```bash
cd BD_cleaning_scripts
conda env create -f environment.yml
```

This will create a new conda environment named `pytorch` with Python 3.12 and all required dependencies.

#### Step 3: Activate the Environment

```bash
conda activate pytorch
```

You should now see `(pytorch)` in your command prompt, indicating the environment is active.

#### Step 4: Verify Installation

```bash
python --version  # Should show Python 3.12.x
python -c "import torch; print(torch.__version__)"  # Should show PyTorch version
```

## Script Usage

### bd_clean.py - Training Models

This script trains neural network models on BD data stored in NPZ format.

#### Basic Usage

```bash
python bd_clean.py -i data.npz -o model.pt -n 2
```

#### Required Arguments

- `-i, --input` - Input npz file containing the training data
- `-o, --output` - Output pt file to save the trained model
- `-n, --rounds` - Number of training rounds to perform

#### Optional Arguments

**Training Parameters:**
- `-b, --batch-size` - Batch size for training (default: 8)
- `-e, --epochs` - Number of epochs per training run (default: 50)
- `--train-split` - Train/test split ratio (default: 0.8)
- `--train-length` - Truncate time series to this length in TRs

**Optimizer Parameters:**
- `--lr-values` - Comma-separated learning rates to try (default: 0.04,0.05,0.06)
- `--eps-values` - Comma-separated epsilon values to try (default: 0.0,1e-8,1e-7)
- `--scheduler-step` - Step size for learning rate scheduler (default: 20)
- `--scheduler-gamma` - Gamma for learning rate scheduler (default: 0.9)

**Other Options:**
- `--seed` - Random seed for reproducibility
- `-v, --verbose` - Enable verbose output with detailed progress

#### Examples

**Basic training:**
```bash
python bd_clean.py -i data.npz -o model.pt -n 2
```

**With custom hyperparameters:**
```bash
python bd_clean.py -i data.npz -o model.pt -n 3 --epochs 100 --batch-size 16
```

**With specific learning rates:**
```bash
python bd_clean.py -i data.npz -o model.pt -n 2 --lr-values 0.01,0.02,0.03
```

**With truncated training length:**
```bash
python bd_clean.py -i data.npz -o model.pt -n 2 --train-length 370
```

**Verbose mode for detailed output:**
```bash
python bd_clean.py -i data.npz -o model.pt -n 2 --verbose
```

**With reproducible results:**
```bash
python bd_clean.py -i data.npz -o model.pt -n 2 --seed 42
```

#### Notes

- The input file should be an npz file with specific data keys (ori64, sim64, ori64means, sim64means, ori64sigmas, sim64sigmas)
- The model will be saved as a PyTorch state dict
- Multiple learning rates and epsilon values are tested during training
- The best model is selected based on test correlation
- Use verbose mode to see detailed training progress

### nifti_clean.py - Applying Models

This script applies a trained neural network model to clean NIFTI imaging files.

#### Basic Usage

```bash
python nifti_clean.py -m model.pt -i input.nii -o output.nii
```

#### Required Arguments

- `-m, --model` - Path to the trained PyTorch model file (.pt)
- `-i, --input` - Path to the input NIFTI file to be cleaned
- `-o, --output` - Path where the cleaned NIFTI file will be saved

#### Optional Arguments

- `-b, --batch-size` - Batch size for processing (default: 12)
- `--mask` - Path to custom brain mask file (optional)
- `--no-mask` - Skip brain mask application entirely
- `-v, --verbose` - Enable verbose output with processing details
- `--version` - Show version information

#### Examples

**Basic usage:**
```bash
python nifti_clean.py -m model.pt -i input.nii -o output.nii
```

**With custom brain mask:**
```bash
python nifti_clean.py -m model.pt -i input.nii -o output.nii --mask brain_mask.nii
```

**With larger batch size for faster processing:**
```bash
python nifti_clean.py -m model.pt -i input.nii -o output.nii --batch-size 32
```

**Verbose mode for detailed output:**
```bash
python nifti_clean.py -m model.pt -i input.nii -o output.nii --verbose
```

#### Notes

- The model file should be a PyTorch state dict saved with `torch.save()`
- Input NIFTI files should be 4D (x, y, z, time)
- A default brain mask is applied if available at `/shared/home/zeming/utils/MNI152_T1_2mm_brain_mask.nii.gz`
- Use `--no-mask` to skip brain mask application entirely
- Batch size affects memory usage and processing speed

## Getting Help

For detailed help on any script, use the `--help` flag:

```bash
python bd_clean.py --help
python nifti_clean.py --help
```

## Hardware Acceleration

Both scripts automatically detect and use available hardware acceleration:
- **Apple Silicon (M1/M2/M3)**: Uses MPS (Metal Performance Shaders)
- **NVIDIA GPU**: Uses CUDA
- **CPU**: Falls back to CPU processing

## Troubleshooting

### Environment Issues

If you encounter issues with the environment:

```bash
# Remove the existing environment
conda env remove -n pytorch

# Recreate it
conda env create -f environment.yml
```

### Memory Issues

If you run out of memory during training or processing:
- Reduce batch size with `-b` or `--batch-size`
- For training: Use `--train-length` to truncate the dataset
- Close other applications to free up memory

### Missing Dependencies

If you get import errors, ensure the environment is activated:

```bash
conda activate pytorch
```

And verify all packages are installed:

```bash
conda list
```

## License

See the main repository for license information.
