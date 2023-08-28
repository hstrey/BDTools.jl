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
phantomfile = joinpath(DATA_DIR, "BFC_time_series.nii")
groundtruthfile = "examples/gt.h5"

# following function loads "ground truth" (GT) phantom data and performs its unit normalization
gt, sim, _ = BDTools.load_ground_truth(groundtruthfile)
# following function loads original data, uses mask from the GT phantom to filter
# unused voxels, and performes unit normalization of the data (minimum & maximum are also returned)
ori, orimin, orimax = BDTools.load_phantom(gt, phantomfile, valid_slices=true)

# construct a denoiser model
model = DenoiseNet(BDTools.TrainParameters(; epochs=20); dev=Flux.gpu)
# and train it on a normalized data
losses = BDTools.train!(model, sim, ori)
# serialized the model to a file `epi.bson`
BDTools.Denoiser.save("examples/epi", model)

# we can plot training losses
plot(losses)

# we can load model from a file
model2 = BDTools.Denoiser.load("examples/epi.bson")

# and perform denoising of the original data
let i = 600
    original = ori[:,:,i:i]
    simulated = sim[:,:,i:i]
    denoised = BDTools.denoise(model2, original)

    plot(simulated |> vec, label="prediction")
    plot!(original |> vec, label="original")
    plot!(denoised |> vec, label="denoised", c=:black)
end

let i = 600
    original = ori[:,:,i:i]
    denoised = BDTools.denoise(model2, original)
    min = orimin[:,:,i:i]
    max = orimax[:,:,i:i]

    full_ori = BDTools.Denoiser.denormalize(original, min, max) |> vec
    full_dnzd = BDTools.Denoiser.destandardize(denoised, min, max) |> vec

    plot(full_ori, label="original")
    plot!(full_dnzd, label="denoised", c=:black)
end
