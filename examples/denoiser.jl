# install `Flux`, `CUDA` & `cuDNN` packages
using Revise
using CUDA
using Flux
using BDTools
using Plots

# ## Loading data
#
# First, we load a generated "ground truth" phantom,
# and the original phantom data from a NIfTI formatted file.
#
const DATA_DIR = "../PhantomData"
# phantomfile = joinpath(DATA_DIR, "BFC_time_series.nii")
phantomfile = joinpath(DATA_DIR, "104", "104.nii")
groundtruthfile = "examples/gt_raw.h5"
# groundtruthfile = "examples/gt_clean.h5"

# following function loads "ground truth" (GT) phantom data and performs its unit normalization
# `sim` variable contains GT time series as a 3D tensor with a first dimension as a time series length,
#  a second dimension is 1, and a third dimension is number of voxels from a masked phantom data
gt, removeidx, sim, _ = BDTools.load_ground_truth(groundtruthfile)

# following function loads original data, uses mask from the GT phantom to filter
# unused voxels, and performs unit normalization of the data (minimum & maximum are also returned)
# `remove` parameter is used to explicitly remove voxels by index
ori, _ = BDTools.load_phantom(gt, phantomfile, valid_slices=true, remove=removeidx)

# construct a denoiser model
model = DenoiseNet(BDTools.TrainParameters(; epochs=200); dev=Flux.gpu)
# and train it on a normalized data
losses = BDTools.train!(model, sim, ori)
# serialized the model to a file `epi.bson`
BDTools.Denoiser.save("examples/104", model)

# we can plot training losses
plot(losses)

# we can load model from a file
model = BDTools.Denoiser.load("examples/104.bson")

# and perform denoising of the original data
let i = 450
    original = ori[:,:,i:i]
    simulated = sim[:,:,i:i]
    denoised = BDTools.denoise(model, original)

    plot(simulated |> vec, label="prediction")
    plot!(original |> vec, label="original")
    plot!(denoised |> vec, label="denoised", c=:black)
end
