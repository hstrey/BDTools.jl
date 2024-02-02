module Denoiser

using Printf
using Flux
using BSON
using Statistics
using Random, StableRNGs
using NIfTI
using Statistics
using Polynomials
using ..BDTools: BDTools

export DenoiseNet, TrainParameters, train!, load_ground_truth, load_phantom, denoise, detrend

"""
    standardize(data::AbstractArray; dims=1)

Return standardized `data` over dimension `dims`.
"""
function standardize(data::AbstractArray; dims=1)
    μ = mean(data, dims=dims)
    σ = std(data, dims=dims, mean=μ)
    (data.-μ) ./ σ, μ, σ
end

"""
    detrend(data::AbstractArray)

Return detrended `data`.
"""
function detrend(data::AbstractArray; dims=1)
    l = length(data)
    data .- Polynomials.fit(1:l,data,1).(1:l)
end

"""
    destandardize(data, μ, σ)

Return destandardized data.
"""
destandardize(data, μ, σ) = data.*σ .+ μ

"""
    normalize(data::AbstractArray; dims=1)

Return 0-1 normalized `data` over dimension `dims`.
"""
function normalize(data::AbstractArray; dims=1)
    min = minimum(data, dims=dims)
    max = maximum(data, dims=dims)
    (data.-min) ./ (max.-min), min, max
end

"""
    denormalize(data, min, max)

Return denormalized data.
"""
denormalize(data, min, max) = data.*(max.-min).+min

"""
    training_set(data::AbstractArray; dims=3)

Form a training set from a standardized data by combining original dataset
with it's reverse over dimension `dims`.
"""
function training_set(data::AbstractArray; dims=3)
    rev = reverse(data, dims=1) # reverse time
    cat(data, rev, dims=dims)
end

"""
    load_ground_truth(fname::String)

Returns a `GroundTruth` object, a standardized ground truth data, and standardization parameters, i.e. μ and σ.
"""
function load_ground_truth(fname::String; datatype=Float32, normalization=standardize)
    # load ground truth data
    gt = BDTools.deserialize(fname, BDTools.GroundTruth)

    # simulated ground truth
    simulated = datatype.(permutedims(
        reshape(
            (@view gt.data[:,:,:,1]),
            (size(gt.data,1), :, 1))
        , [1,3,2]))

    # remove any time series that contain NaNs
    clean_idxs = [!any(isnan.(s)) for s in eachslice(simulated, dims=3)]
    cleaned_simulated = @view simulated[:,:,clean_idxs]

    return gt, findall(iszero, clean_idxs), normalization(cleaned_simulated)...
end

function mask_time_series(gt::BDTools.GroundTruth, data::AbstractArray)
    mask = map(idxs -> CartesianIndex(idxs...), eachcol(gt.maskindex))
    masked = mapslices(x->x[mask], data, dims=(1,2))
    dropdims(masked, dims=2)
end

"""
    load_phantom(gt::GroundTruth, phantomfile::String)

Return standardized phantom time series data, and standardization parameters, i.e. μ and σ.
"""
function load_phantom(gt::BDTools.GroundTruth, phantomfile::String;
                      datatype=Float32, valid_slices=false,
                      remove=Int[], normalization=standardize)
    phantom = niread(phantomfile);
    gtsize = size(gt.data)
    phsize = size(phantom)

    # use only phantom series corresponding to the ground truth dimensions
    tstart = phsize[4] - gtsize[1] + 1
    valid_data = convert(Array{datatype}, @view phantom[:,:,:,tstart:end])

    # mask data
    masked_data = mask_time_series(gt, valid_data)
    slice_index = valid_slices ? gt.sliceindex .- gt.sliceindex[1] .+ 1 :  1:size(valid_data,3)

    # reshape and standardize data
    masked_valid_data = @view masked_data[:, slice_index, :]
    ts_data = permutedims(reshape(masked_valid_data, (:, 1, size(masked_valid_data, 3))), [3,2,1])

    # mask remove
    remove_indices = [i ∉ remove for i in 1:size(ts_data,3)]
    clean_ts_data = @view ts_data[:, :, remove_indices]

    return normalization(clean_ts_data)
end

anynan(x) = any(y -> any(isnan, y), x)

"""
Training parameters for a denoising network.

- `adj_rate`: a learning rate adjustment factor (default value is `0.33`)
- `batch_size`: a training data batch size (default value is `8`)
- `epochs`: a number of training epochs (default value is `250`)
- `eta`: a learning rate (default value is `3e-4`)
- `kernel_size`: a convolutional layer kernel size (default value is `9`)
- `layers`: a number of denoiser network convolutional layers (default value is `6`)
- `save_path`: a path to a location where a NN model is saved
- `save_best`: if `true` then the denoiser model is saved on each successful loss improvement
- `seed`: a PRNG seed (default value is `180181`)
- `test_split`: a fraction of a phantom data dedicated to a testing set (default value is `0.33`)
"""
Base.@kwdef mutable struct TrainParameters
    adj_rate::Float64 = 1/5
    batch_size::Int = 8
    epochs::Int = 250
    eta::Float64 = 1e-5
    improve_steps::Int = 50
    kernel_size::Int = 9
    layers::Int = 6
    save_path::String = "./"
    save_best::Bool = false
    seed::Int = 180181
    test_split = 1/5
end

"""
    Denoiser Model

A wrapper type for denoiser model. It contains deep neural network parameters,
a loss function, and training parameters.
"""
mutable struct DenoiseNet{LF}
    model::Chain
    loss::LF
    device::Function
    parameters::TrainParameters
end

"""
    DenoiseNet(params = TrainParameters(); chain=denoiser, loss=negr2, dev=Flux.cpu)

Construct a denoiser CNN model with specified training parameters, [`params`](@ref TrainParameters).
Using keyword arguments following parameters of the model can be modified:
- `chain`: a CNN structure represented as `Flux.Chain`
- `loss`: a loss function used for training the model
- `dev`: a location of the network model (`Flux.cpu` or `Flux.gpu`)
"""
function DenoiseNet(params = TrainParameters(); chain=denoiser, loss=negr2, dev=Flux.cpu, kwargs...)
    net = chain(;layers = params.layers, kersize=params.kernel_size, kwargs...)
    DenoiseNet(net |> dev, loss, dev, params)
end

"""
    denoiser(;convdim=18, kersize=9, layers=6)

Construct a denoiser deep CNN network with a convolution dimension `convdim`,
a kernel size `kersize`, and a number of `layers`.
"""
function denoiser(;convdim=18, kersize=9, layers=6, kwargs...)
    Chain(
        # 1st layer, Conv+sigmoid
        Conv((kersize,), 1=>convdim, sigmoid, pad=SamePad()),
        # 6 layers, Conv+BN+sigmoid
        (Chain(
            Conv((kersize,), convdim=>convdim, pad=SamePad()),
            BatchNorm(convdim, sigmoid)
        ) for i in 1:layers)...,
        # last layer, Conv
        Dropout(0.2),
        Conv((1,), convdim=>1, pad=SamePad())
    )
end

function denoiser2(;convdim=18, kersize=9, layers=6, maxpool=30, input=600, kwargs...)
    h = layers >> 1
    hidden = round(Int, input/maxpool)
    Chain(
        # 1st layer, Conv+sigmoid
        Conv((kersize,), 1=>convdim, tanh, pad=SamePad()),
        # 3 layers, Conv+BN+sigmoid
        (Chain(
            Conv((kersize,), convdim=>convdim, pad=SamePad()),
            BatchNorm(convdim, tanh)
        ) for i in 1:h)...,
        # dense layer in the middle
        Flux.MaxPool((maxpool,), pad=SamePad()),
        Flux.flatten,
        Dense(hidden*convdim=>hidden*convdim),
        X -> reshape(X, hidden, convdim, :),
        Upsample(scale=(maxpool,1)),
        # 3 layers, Conv+BN+sigmoid
        (Chain(
            Conv((kersize,), convdim=>convdim, pad=SamePad()),
            BatchNorm(convdim, tanh)
        ) for i in 1:(layers-h))...,
        # last layer, Conv
        # Dropout(0.2),
        Conv((1,), convdim=>1, pad=SamePad())
    )
end

"""
    negr2(y_pred, y_true)

Returns a negative R^2 value for predicted, `y_pred`, and "true", `y_true`, values.
"""
function negr2(y_pred, y_true)
    SS_tot = sum(abs2, y_true .- mean(y_true))
    return negr2(y_pred, y_true, SS_tot)
end

function negr2(y_pred, y_true, SS_tot)
    SS_res =  sum(abs2, y_true .- y_pred)
    r2 = 1 - SS_res/(SS_tot + eps())
    return -r2
end

"""
    dataloader(sim, ori; dev=gpu)

Construct a `Flux.DataLoader` from simulated and original data sets.
The training dataset is created by combining original and reversed time series.

"""
function dataloader(sim, ori; dev=gpu, test_split=0.2, seed=180181, batch_size=8)
    # form a training sets
    simtrain = training_set(sim);
    oritrain = training_set(ori);

    # create a randomized collection of indices
    rng = StableRNG(seed)
    n = size(simtrain, 3)
    idxs = collect(1:n)
    shuffle!(rng, idxs)

    # split data
    m = round(Int, n*test_split)
    trainidxs = @view idxs[1:(n-m)]
    testidxs = @view idxs[(n-m)+1:end]
    train_original = dev(oritrain[:,:,trainidxs])
    train_simulated = dev(simtrain[:,:,trainidxs])

    # form a dataloader
    train_data = Flux.DataLoader((train_original, train_simulated); batchsize=batch_size, shuffle=true, rng);
    test_original = dev(oritrain[:, :, testidxs])
    test_simulated = dev(simtrain[:, :, testidxs])
    return (train_data, test_original, test_simulated)
end

"""
    train!(denoiser::Denoiser, sim::AbstractArray{T,3}, ori::AbstractArray{T,3})

Train a denoiser network given a ground truth data, `sim`, and an original phantom data, `ori`.
The data has to be standardized.
"""
function train!(denoiser::DenoiseNet, sim::AbstractArray{T,3}, ori::AbstractArray{T,3}) where {T <: AbstractFloat}
    args = denoiser.parameters
    dev = denoiser.device

    # form a dataloader
    train_data, test_simulated, test_original = dataloader(sim, ori; dev,
        batch_size=args.batch_size, test_split=args.test_split, seed=args.seed);

    # build model
    model = denoiser.model
    lossfn = denoiser.loss

    opt = Flux.setup(Adam(args.eta), model)

    @info("Beginning training loop...")
    best_loss = Inf
    last_improvement = 0
    losses = T[]
    for epoch_idx in 1:args.epochs

        # Train for a single epoch
        elapsed = @elapsed Flux.train!(model, train_data, opt) do m, x, y
            lossfn(m(x), y)
        end

        # Terminate on NaN
        if anynan(Flux.params(model))
            @error "NaN params"
            break
        end

        # Calculate loss
        loss = lossfn(model(test_original), test_simulated)
        push!(losses, loss)
        @info(@sprintf("[%d]: Test loss: %.7f. Elapsed: %.3fs", epoch_idx, loss, elapsed))

        #TODO: find why loss negative, meanwhile skip negative loss
        loss < 0 && continue

        # If this is the best loss we've seen so far, save the model out
        if loss <= best_loss
            @info(" -> New best loss: $(loss)!")
            if !isinf(best_loss) && args.save_best
                save("ph_tmp", denoiser, epoch = epoch_idx, loss = loss)
            end
            # skip first epoch
            if epoch_idx > 1
                best_loss = loss
                last_improvement = epoch_idx
            end
        end

        # If we haven't seen improvement in some epochs, drop our learning rate:
        if last_improvement > 0 && epoch_idx - last_improvement >= args.improve_steps && args.eta > 1e-6
            args.eta *= args.adj_rate
            Flux.adjust!(opt, args.eta)
            @warn(" -> Haven't improved in a while, dropping learning rate to $(args.eta)!")

            # After dropping learning rate, give it a few epochs to improve
            last_improvement = epoch_idx
        end
    end

    # return losses
    return losses, test_original, test_simulated
end

"""
    save(fname::String, model::DenoiseNet; kwargs...)

Saves a denoising CNN `model` to a file, `fname`.
"""
function save(fname::String, model::DenoiseNet; kwargs...)
    args = model.parameters
    BSON.bson(joinpath(args.save_path, "$fname.bson");
              model=Flux.state(model.model |> cpu),
              params = args,
              kwargs...)
end

"""
    load(fname::String; dev=Flux.cpu) -> DenoiseNet

Loads a denoising CNN model from a file, `fname`.
"""
function load(fname::String; dev=Flux.cpu)
    data = BSON.load(fname)
    params = data[:params]
    state = data[:model]
    net = denoiser(;layers = params.layers, kersize=params.kernel_size)
    Flux.loadmodel!(net, state)
    DenoiseNet(net |> dev, negr2, dev, params)
end

"""
    denoise(model::DenoiseNet, data)

Denoise `data` using a denoseing CNN `model`. The data input should be standardized.
"""
function denoise(model::DenoiseNet, data)
    denoised = model.model(data |> model.device) |> Flux.cpu
    return denoised
end

snr(measured, denoised) = sum(abs2, denoised) / sum(abs2, measured-denoised)

end # module Denoiser
