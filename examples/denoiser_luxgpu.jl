using HDF5: HDF5
using Random, Lux, ADTypes, Zygote, Printf, MLUtils, Statistics
using Optimization, OptimizationOptimisers, SciMLSensitivity, ComponentArrays
using OptimizationOptimJL
using LuxCUDA, CUDA
using Plots

CUDA.allowscalar(false)

const gdev = gpu_device()
const cdev = cpu_device()

struct GroundTruthCleaned
    data::Array{Float64,3}
end

function deserialize(filepath::String, ::Type{GroundTruthCleaned})
    HDF5.h5open(filepath, "r") do io
        GroundTruthCleaned(
            io["GroundTruthCleaned/data"] |> read
        )
    end
end

"""
    standardize(data::AbstractArray; dims=1)

Return standardized `data` over dimension `dims`.
"""
function standardize(data::AbstractArray; dims=1)
    μ = mean(data, dims=dims)
    σ = std(data, dims=dims, mean=μ)
    (data.-μ) ./ σ, μ, σ
end

gt = deserialize("gt_cleanTE3new370.h5", GroundTruthCleaned)
ori64, ori_mean, ori_std = standardize(reshape(gt.data[:,:,1],
    (size(gt.data[:,:,1])[1],1,size(gt.data[:,:,1])[2])))
sim64, sim_mean, sim_std = standardize(reshape(gt.data[:,:,2],
    (size(gt.data[:,:,2])[1],1,size(gt.data[:,:,2])[2])))
ori = Float32.(ori64)
sim = Float32.(sim64)

gt2 = deserialize("gt_cleanTE3new2.h5", GroundTruthCleaned)
ori64, ori_mean, ori_std = standardize(reshape(gt.data[:,:,1],
    (size(gt.data[:,:,1])[1],1,size(gt.data[:,:,1])[2])))
sim64, sim_mean, sim_std = standardize(reshape(gt.data[:,:,2],
    (size(gt.data[:,:,2])[1],1,size(gt.data[:,:,2])[2])))
ori = Float32.(ori64)
sim = Float32.(sim64)

(i_train, o_train), (i_test, o_test) = splitobs((ori, sim); at=0.8)
train_dataloader = DataLoader(collect.((i_train, o_train)); batchsize=8, shuffle=true) |> gdev
test_dataloader = DataLoader(collect.((i_test, o_test)); batchsize=8, shuffle=true)

i_test = copy(i_test) |> gdev
o_test = copy(o_test) |> gdev
i_train = copy(i_train) |> gdev
o_train = copy(o_train) |> gdev


rng = MersenneTwister()
Random.seed!(rng, 12345)

function negr2(y_pred, y_true)
    SS_tot = sum(abs2, vec(y_true) .- mean(vec(y_true)))
    return negr2(y_pred, y_true, SS_tot)
end

function negr2(y_pred, y_true, SS_tot)
    SS_res =  sum(abs2, vec(y_true) .- vec(y_pred))
    r2 = 1 - SS_res/(SS_tot + eps())
    return -r2
end

model = Chain(
        Conv((9,), 1=>18, sigmoid, pad=SamePad()),
        [Chain(Conv((9,), 18=>18, pad=SamePad()), BatchNorm(18, sigmoid)) for _ in 1:6]...,
        Dropout(0.2),
        Conv((1,), 18=>1, pad=SamePad())
)

# Flux implementation of alternate NN
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

ps, st = Lux.setup(rng, model)
ps_ca = ComponentArray(ps) |> gdev
stgpu = st |> gdev

function pcor(x,y)
    x_mean = mean(x)
    y_mean = mean(y)
    x_norm = x .- x_mean
    y_norm = y .- y_mean
    cv = mean(x_norm .* y_norm)
    return cv/std(x)/std(y)
end

function loss(p, (i_data,o_data))
    model_d = Lux.apply(model,i_data,p, stgpu)[1]
    #print(size(i_data))
    return negr2(vec(model_d),vec(o_data))
    #c = pcor(vec(model_d),vec(o_data))
    #return -c/(1-c) # stretch the range to -infty to +infty
end

losses = []
corrs = []
bestpara = nothing
bestc = 0.0

function callback(state, l)
    model_d = Lux.apply(model, i_test, state.u, stgpu)[1]
    c = pcor(vec(model_d),vec(o_test))
    if c > bestc
        global bestc = c
        global bestpara = state.u
    end
    push!(losses,l)
    push!(corrs,c)
    state.iter % 50 == 1 && @printf "Iteration: %5d, Loss: %.5g, Fidelity: %.5g\n" state.iter l c
    return false
end

function callback2(state, l)
    #model_d = Lux.apply(model,i_test,state.u, st)[1]
    #c = cor(vec(model_d),vec(o_test))
    #if c > bestc
    #    global bestc = c
    #    global bestpara = state.u
    #end
    push!(losses,l)
    #push!(corrs,c)
    @printf "Iteration: %5d, Loss: %.5g, Fidelity: %.5g\n" state.iter l
    return false
end

opt_func = OptimizationFunction(loss, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps_ca, train_dataloader)
opt = Optimisers.Adam(1e-4)

res_adam = solve(opt_prob, opt; callback, epochs = 250)

pcor(vec(i_test),vec(o_test)), pcor(vec(i_train),vec(o_train))

pcor(vec(Lux.apply(model,i_test,bestpara, stgpu)[1]),vec(o_test)), pcor(vec(Lux.apply(model,i_train,bestpara, stgpu)[1]),vec(o_train))

pcor(vec(Lux.apply(model,i_test,res_adam.u, stgpu)[1]),vec(o_test)), pcor(vec(Lux.apply(model,i_train,res_adam.u, stgpu)[1]),vec(o_train))

opt_prob_new = OptimizationProblem(opt_func, bestpara, (i_train,o_train))
res_bfgs = solve(opt_prob_new, Optimisers.Adam(0.001); callback, epochs=10000)

pcor(vec(Lux.apply(model,i_test,res_bfgs.u, stgpu)[1]),vec(o_test)), pcor(vec(Lux.apply(model,i_train,res_bfgs.u, stgpu)[1]),vec(o_train))

plot(losses)
plot(corrs)