using HDF5: HDF5
using Random, Lux, Zygote, Printf, MLUtils, Statistics, Optimisers
using BSON

#const xdev = reactant_device(; force=true)
#const cdev = cpu_device()

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

gt2 = deserialize("gt_cleanTE3_new2.h5", GroundTruthCleaned)
ori64_2 =normalize(reshape(gt2.data[:,:,1], (size(gt2.data[:,:,1])[1],1,size(gt2.data[:,:,1])[2])))
sim64_2 = normalize(reshape(gt2.data[:,:,2], (size(gt2.data[:,:,2])[1],1,size(gt2.data[:,:,2])[2])))

ori64t = cat(ori64,ori64_2,dims=3)
sim64t = cat(sim64,sim64_2,dims=3)

ori = normalize(Float32.(ori64t)[1:370,:,:])
sim = normalize(Float32.(sim64t)[1:370,:,:])

(i_train, o_train), (i_test, o_test) = splitobs((ori, sim); at=0.8)
stand_i_train = standardize(i_train)
stand_o_train = standardize(o_train)
stand_i_test = standardize(i_test)
stand_o_test = standardize(o_test)
train_dataloader = DataLoader(collect.((i_train, o_train)); batchsize=8, shuffle=true)
test_dataloader = DataLoader(collect.((i_test, o_test)); batchsize=8, shuffle=true)

model = Chain(
        Conv((9,), 1=>18, sigmoid, pad=SamePad()),
        [Chain(Conv((9,), 18=>18, pad=SamePad()), BatchNorm(18, sigmoid)) for _ in 1:6]...,
        Dropout(0.2),
        Conv((1,), 18=>1, pad=SamePad())
)

model = Chain(Conv((9,), 1=>10, sigmoid, pad=SamePad()),BatchNorm(10, sigmoid),
                MaxPool((2,)),
                Conv((9,),10=>20, sigmoid, pad=SamePad()),BatchNorm(20, sigmoid),
                MaxPool((2,)),
                Conv((9,),20=>40, sigmoid, pad=SamePad()),BatchNorm(40, sigmoid),
                MaxPool((2,)),
                ConvTranspose((9,), 40=>20, sigmoid, stride=2, pad=SamePad()),BatchNorm(20, sigmoid),
                ConvTranspose((9,), 20=>10, sigmoid, stride=2, pad=3),BatchNorm(10, sigmoid), Dropout(0.2),
                ConvTranspose((9,), 10=>1, sigmoid, stride=2, pad=SamePad())
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

function negr2(y_pred, y_true)
    SS_tot = sum(abs2, vec(y_true) .- mean(vec(y_true)))
    SS_res =  sum(abs2, vec(y_true) .- vec(y_pred))
    r2 = 1 - SS_res/(SS_tot + eps())
    return -r2
end

function mean_corr(data1, data2)
    batch_size = size(data1)[3]
    mean([cor(data1[:,1,i],data2[:,1,i]) for i in 1:batch_size])
end

rng = Xoshiro()
Random.seed!(rng, 0)


function pcor(x,y)
    x_mean = mean(x)
    y_mean = mean(y)
    x_norm = x .- x_mean
    y_norm = y .- y_mean
    cv = mean(x_norm .* y_norm)
    return cv/std(x)/std(y)
end

function loss_function(model, ps, st,(i_data,o_data))
    model_d = standardize(model(i_data,ps,st)[1])
    #print(size(i_data))
    #return negr2(vec(model_d),vec(standardize(o_data)))
    c = cor(vec(model_d),vec(standardize(o_data)))
    return -c/(1-c), st, (;) # stretch the range to -infty to +infty
    #return MSELoss()(model_d, standardize(i_data))
end

ps, st = Lux.setup(rng, model)

opt = AdamW(; eta=1e-3, lambda=1e-5)

train_state = Training.TrainState(model, ps, st, opt)

@printf "Total Trainable Parameters: %0.4f M\n" (Lux.parameterlength(ps)/1e6)
epochs = 150
best_ps = nothing
best_st = nothing
bestc = -1

for epoch in 1:epochs
    loss_total = 0.0f0

    start_time = time()
    for (i, X) in enumerate(train_dataloader)
        (_, loss, _, train_state) = Training.single_train_step(
            AutoZygote(), loss_function, X, train_state
        )

        model_d = standardize(model(i_test, train_state.parameters, Lux.testmode(train_state.states))[1])
        c = cor(vec(model_d),vec(stand_o_test))
        if c > bestc
            best_ps = train_state.parameters
            best_st = Lux.testmode(train_state.states)
            bestc = c
            @printf "New best correlation %.7f\n" c
        end

        if i % 50 == 0 || i == length(train_dataloader)
            @printf "Epoch %d, Iter %d, Loss: %.7f Best corr: %.7f\n" epoch i loss bestc
        end
    end
    total_time = time() - start_time

    train_loss = loss_total / length(train_dataloader)
    @printf "Epoch %d, Train Loss: %.7f, Time: %.4fs\n" epoch train_loss total_time

end

"Mean Corr signal", mean_corr(i_test,o_test), mean_corr(i_train,o_train)

"Corr signal", cor(vec(i_test),vec(o_test)), cor(vec(i_train),vec(o_train))

"Corr stand signal", cor(vec(stand_i_test),vec(stand_o_test)), cor(vec(stand_i_train),vec(stand_o_train))

"Corr best model", cor(vec(model(i_test,best_ps,best_st)[1]),vec(o_test)), cor(vec(model(i_train,best_ps,best_st)[1]),vec(o_train))

"Mean Corr model", mean_corr(model(i_test,best_ps,best_st)[1],o_test), mean_corr(model(i_train,best_ps,best_st)[1],o_train)

"Corr stand model", cor(vec(standardize(model(i_test,best_ps,best_st)[1])),vec(stand_o_test)), cor(vec(standardize(model(i_train,best_ps,best_st)[1])),vec(stand_o_train))

"Corr opt model", cor(vec(model(i_test,train_state.parameters,Lux.testmode(train_state.states))[1]),vec(o_test)), cor(vec(model(i_train,train_state.parameters,Lux.testmode(train_state.states))[1]),vec(o_train))