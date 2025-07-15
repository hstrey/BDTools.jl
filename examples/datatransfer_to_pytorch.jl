using HDF5: HDF5
using Random, Statistics, NPZ

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
    μ, σ, (data.-μ) ./ mean(σ)
end

function mean_corr(data1, data2)
    batch_size = size(data1)[3]
    mean([cor(data1[:,1,i],data2[:,1,i]) for i in 1:batch_size])
end

gt = deserialize("gt_clean_Bay2_041624.h5", GroundTruthCleaned)
ori64means, ori64sigmas, ori64 = normalize(reshape(gt.data[:,:,1], (size(gt.data[:,:,1])[1],1,size(gt.data[:,:,1])[2])))
sim64means, sim64sigmas, sim64 = normalize(reshape(gt.data[:,:,2], (size(gt.data[:,:,2])[1],1,size(gt.data[:,:,2])[2])))

data_dict = Dict("ori64means" => ori64means,
             "ori64sigmas" => ori64sigmas,
             "ori64" => ori64,
             "sim64means" => sim64means,
             "sim64sigmas" => sim64sigmas,
             "sim64" => sim64
             )

npzwrite("gt_clean_Bay2_041624.npz", data_dict)

ori64