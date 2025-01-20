using HDF5: HDF5
using Random, Lux, ADTypes, Zygote, Printf, MLUtils, Statistics
using Optimization, OptimizationOptimisers, SciMLSensitivity, ComponentArrays
using OptimizationOptimJL
using BSON
using Plots

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

function serialize(filepath::String, gt::GroundTruthCleaned)
    HDF5.h5open(filepath, "w") do io
        g = HDF5.create_group(io, "GroundTruthCleaned")
        dset = HDF5.create_dataset(g, "data", eltype(gt.data), size(gt.data))
        HDF5.write(dset, gt.data)
    end
end

"""
    standardize(data::AbstractArray; dims=1)

Return standardized `data` over dimension `dims`.
"""
function standardize(data::AbstractArray; dims=1)
    μ = mean(data, dims=dims)
    σ = std(data, dims=dims, mean=μ)
    (data.-μ) ./ σ
end

"""
    standardize(data::AbstractArray; dims=1)

Return standardized `data` over dimension `dims`.
"""
function normalize(data::AbstractArray; dims=1)
    μ = mean(data, dims=dims)
    σ = std(data, dims=dims, mean=μ)
    (data.-μ) ./ mean(σ)
end

function mean_corr(data1, data2)
    batch_size = size(data1)[3]
    mean([cor(data1[:,1,i],data2[:,1,i]) for i in 1:batch_size])
end

gt = deserialize("gt_cleanTE3_new.h5", GroundTruthCleaned)
ori64 =normalize(reshape(gt.data[:,:,1], (size(gt.data[:,:,1])[1],1,size(gt.data[:,:,1])[2])))
sim64 = normalize(reshape(gt.data[:,:,2], (size(gt.data[:,:,2])[1],1,size(gt.data[:,:,2])[2])))
ori = normalize(Float32.(ori64)[1:370,:,:])
sim = normalize(Float32.(sim64)[1:370,:,:])

(i_train, o_train), (i_test, o_test) = splitobs((ori, sim); at=0.8)
stand_i_train = standardize(i_train)
stand_o_train = standardize(o_train)
stand_i_test = standardize(i_test)
stand_o_test = standardize(o_test)
train_dataloader = DataLoader(collect.((i_train, o_train)); batchsize=8, shuffle=true)
test_dataloader = DataLoader(collect.((i_test, o_test)); batchsize=8, shuffle=true)

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

model = Chain(Conv((9,), 1=>8, sigmoid, pad=SamePad()),
                MaxPool((2,)),
                Conv((9,),8=>16, sigmoid, pad=SamePad()),
                MaxPool((2,)),
                Conv((9,),16=>32, sigmoid, pad=SamePad()),
                MaxPool((2,0)),
                ConvTranspose((9,), 32=>16, sigmoid, stride=2, pad=pad=(4, 4))
)

model = Chain(Conv((9,), 1=>8, sigmoid, pad=SamePad()),BatchNorm(8, sigmoid),
                MaxPool((2,)),
                Conv((9,),8=>16, sigmoid, pad=SamePad()),BatchNorm(16, sigmoid),
                MaxPool((2,)),
                Conv((9,),16=>32, sigmoid, pad=SamePad()),BatchNorm(32, sigmoid),
                MaxPool((2,)),
                ConvTranspose((9,), 32=>16, sigmoid, stride=2, pad=SamePad()),BatchNorm(16, sigmoid),
                ConvTranspose((9,), 16=>8, sigmoid, stride=2, pad=3),BatchNorm(8, sigmoid), Dropout(0.1),
                ConvTranspose((9,), 8=>1, sigmoid, stride=2, pad=SamePad())
                )

# variational autoencoder 

model = Chain(Conv((5,), 1=>8, sigmoid, pad=SamePad()),
                MaxPool((2,)),BatchNorm(8, sigmoid),
                Conv((5,),8=>16, sigmoid, pad=SamePad()),
                MaxPool((2,)),BatchNorm(16, sigmoid),
                Conv((5,),16=>32, sigmoid, pad=SamePad()),
                MaxPool((2,)),BatchNorm(32, sigmoid),
                ConvTranspose((5,), 32=>16, sigmoid, stride=2, pad=SamePad()),BatchNorm(16, sigmoid),
                ConvTranspose((5,), 16=>8, sigmoid, stride=2, pad=1), BatchNorm(8, sigmoid),Dropout(0.2),
                ConvTranspose((5,), 8=>1, sigmoid, stride=2, pad=SamePad())
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

function loss(p, (i_data,o_data))
    model_d = smodel(i_data,p)
    #print(size(i_data))
    #return negr2(vec(model_d),vec(o_data))
    c = cor(vec(model_d),vec(o_data))
    return -c/(1-c) # stretch the range to -infty to +infty
    #return MSELoss()(model_d, i_data)
end

function loss_s(p, (i_data,o_data))
    model_d = standardize(smodel(i_data,p))
    #print(size(i_data))
    #return negr2(vec(model_d),vec(standardize(o_data)))
    c = cor(vec(model_d),vec(standardize(o_data)))
    return -c/(1-c) # stretch the range to -infty to +infty
    #return MSELoss()(model_d, standardize(i_data))
end

function loss_c(p, (i_data,o_data))
    model_d = smodel(i_data,p)
    batch_size = size(model_d)[3]
    c = mean([cor(model_d[:,1,i],o_data[:,1,i]) for i in 1:batch_size])
    return -c/(1-c) # stretch the range to -infty to +infty
end

function callback(state, l)
    model_d = smodel(i_test,state.u)
    c = cor(vec(model_d),vec(o_test))
    if c > bestc
        global bestc = c
        global bestpara = state.u
    end
    push!(losses,l)
    push!(corrs,c)
    state.iter % 50 == 1 && @printf "Iteration: %5d, Training: %.5g, Fidelity: %.5g, Best: %.5g\n" state.iter l/(l-1) c bestc
    return false
end

function callback_c(state, l)
    model_d = smodel(i_test,state.u)
    batch_size = size(model_d)[3]
    c = mean([cor(model_d[:,1,i],o_test[:,1,i]) for i in 1:batch_size])
    if c > bestc
        global bestc = c
        global bestpara = state.u
    end
    push!(losses,l)
    push!(corrs,c)
    state.iter % 50 == 1 && @printf "Iteration: %5d, Training: %.5g, Fidelity: %.5g, Best: %.5g\n" state.iter l/(l-1) c bestc
    return false
end

function callback_s(state, l)
    model_d = standardize(smodel(i_test,state.u))
    c = cor(vec(model_d),vec(stand_o_test))
    if c > bestc
        global bestc = c
        global bestpara = state.u
    end
    push!(losses,l)
    push!(corrs,c)
    state.iter % 50 == 1 && @printf "Iteration: %5d, Training: %.5g, Fidelity: %.5g, Best: %.5g\n" state.iter l/(l-1) c bestc
    return false
end

function callback2(state, l)
    model_d = Lux.apply(model3,i_test,state.u, st)[1]
    c = cor(vec(model_d),vec(o_test))
    if c > bestc
        global bestc = c
        global bestpara = state.u
    end
    push!(losses,l)
    push!(corrs,c)
    @printf "Iteration: %5d, Loss: %.5g, Fidelity: %.5g\n" state.iter l/(l-1) c
    return false
end

ps, st = Lux.setup(rng, model)
ps_ca = ComponentArray(ps)
smodel = StatefulLuxLayer{true}(model, nothing, st)

losses = []
corrs = []
bestpara = nothing
bestc = 0.0

opt_func = OptimizationFunction(loss_s, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps_ca, train_dataloader)
opt = Optimisers.AdamW(;eta=5e-3,lambda=1e-5)

res_adam = solve(opt_prob, opt; callback=callback_s, epochs = 150)

"Mean Corr signal", mean_corr(i_test,o_test), mean_corr(i_train,o_train)

"Corr signal", cor(vec(i_test),vec(o_test)), cor(vec(i_train),vec(o_train))

"Corr stand signal", cor(vec(stand_i_test),vec(stand_o_test)), cor(vec(stand_i_train),vec(stand_o_train))

"Corr best model", cor(vec(smodel(i_test,bestpara)),vec(o_test)), cor(vec(smodel(i_train,bestpara)),vec(o_train))

"Mean Corr model", mean_corr(smodel(i_test,bestpara),o_test), mean_corr(smodel(i_train, bestpara),o_train)

"Corr stand model", cor(vec(standardize(smodel(i_test,bestpara))),vec(stand_o_test)), cor(vec(standardize(smodel(i_train,bestpara))),vec(stand_o_train))

"Corr opt model", cor(vec(smodel(i_test,res_adam.u)),vec(o_test)), cor(vec(smodel(i_train,res_adam.u)),vec(o_train))

# save weights
BSON.@save "gt_cleanTE3_370r.bson" model_p = bestpara

opt_prob_new = OptimizationProblem(opt_func, bestpara, (i_train,o_train))
res_bfgs = solve(opt_prob_new, BFGS(); callback=callback2, maxiters = 1000)
bestpara