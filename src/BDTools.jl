module BDTools

using PythonCall

const sitk = PythonCall.pynew()
function __init__()
    PythonCall.pycopy!(sitk, pyimport("SimpleITK"))
end
export sitk

using Optim: Optim
using Interpolations
using Interpolations: scale
using HDF5: HDF5

export StaticPhantom, staticphantom, GroundTruth, groundtruth, DenoiseNet

include("bfield.jl")
include("segmentation.jl")
include("ellipse.jl")
include("utils.jl")
include("acqtimes.jl")
include("qualitymeasures.jl")

"""
A static phantom model
"""
struct StaticPhantom
    data::Array{Float64,3}
    ellipses::Matrix{Float64}
    sliceinfo::Matrix{Float64}
    center::Vector{Float64}
    centersdir::Vector{Float64}
    interpolation::Interpolations.Extrapolation
    segments::Vector{SegmentedImage}
end

Base.show(io::IO, ph::StaticPhantom) = print(io, "Phantom: $(size(ph))")
Base.size(ph::StaticPhantom) = size(ph.data)

"""
    centers(ph::StaticPhantom)

Return centers of ellipses fitted to each slice of a static phantom `ph`.
"""
centers(ph::StaticPhantom) = ph.ellipses[:, 1:2]

"""
    predictcenter(ph::StaticPhantom, t::Real)

Return a center coordinate of a phantom slice given a parametric value `t`.
"""
predictcenter(ph::StaticPhantom, t::Real) = ph.center + ph.centersdir * t

"""
    fittedcenters(ph::StaticPhantom)

Return a projection of fitted ellipses' center to a center axis.
"""
fittedcenters(ph::StaticPhantom) = mapslices(c -> fittedcenter(ph, vec(c)), centers(ph), dims=2)

"""
    fittedcenter(ph::StaticPhantom)

Return a projection of fitted ellipses' center to a center axis.
"""
function fittedcenter(ph::StaticPhantom, p::Vector)
    p0 = ph.center
    dir = ph.centersdir
    p0 + (p - p0)' * dir * dir
end

"""
    mask(ph::StaticPhantom, z::Int; threshold=1.0)

Get coordinates of an internal segment of the phantom's `ph` slice `z`.
The `threshold` parameter allow to specify a percentage of masked coordinates.
"""
function mask(ph::StaticPhantom, z::Int; threshold=Inf)
    @assert 1 <= z <= length(ph.segments) "Incorrect slice index"
    maskmtx = ph.segments[z].image_indexmap .== 3
    # Mask voxels over threshold
    if !isinf(threshold)
        o, a, b = getellipse(ph, z)
        threshfn = ci -> (ci.I[1] - o[1])^2 / a^2 + (ci.I[2] - o[2])^2 / b^2 > threshold
        maskidxs = findall(maskmtx)
        threshed = threshfn.(maskidxs)
        maskmtx[maskidxs[threshed]] .= 0
    end
    maskmtx
end

"""
    getellipse(ph::StaticPhantom, z::Int)

Return ellipse properties for a slice `z` of a static phantom `ph`.
"""
function getellipse(ph::StaticPhantom, z::Int)
    @assert 1 <= z <= size(ph)[3] "Invalid slice index"
    # get center of a phantom
    cs = centers(ph)
    start = fittedcenter(ph, cs[z, :])
    # 3D origin
    origin = [start..., z]
    # ellipse axes
    a, b = ph.ellipses[z, 3:4]
    return (origin, a, b)
end


"""
    findinitialrotation(sph::StaticPhantom, z::Real; intensity=0.01)

Find an ellipse initial rotation angle that minimizes distortions from
elliptical rotation. An `intensity` parameter specifies an intensity percentage
for selection of points as parameters for optimization task that determines
phantom initial rotation.
"""
function findinitialrotation(ph::StaticPhantom, z::Int; intensity=0.01)
    # get a phantom slice
    @assert 1 <= z <= size(ph)[3] "Invalid slice index"
    img = @view ph.data[:, :, z]

    # mask all points outside the internal region of a phantom
    maskidx = mask(ph, z)
    masked = @view img[maskidx]

    # threshold intensity to get a collection of darkest points
    mini, maxi = extrema(masked)
    irng = maxi - mini
    mininidx = findall(masked .< mini + irng * intensity)

    # get points' coordinates
    pts = findall(maskidx)[mininidx]
    findinitialrotation(ph, z, pts)
end

function findinitialrotation(ph::StaticPhantom, z::Int, pts::Vector{CartesianIndex{2}})
    # get ellipse params
    ori, a, b = getellipse(ph, z)

    # Minimize sum all intensity differences for selected points `pts`
    f(γ) = begin
        res = 0
        for ci in pts
            p = [ci.I..., z]
            coords = map(α -> ellipserot(α, γ, a, b) * (p .- ori) .+ ori, 0.0:0.1:π)
            sim = map(c -> ph.interpolation(c...), coords)
            res += abs(-(extrema(sim)...))
        end
        res
    end

    res = Optim.optimize(f, 0.0, float(π))
    Optim.minimizer(res)
end

"""
    staticphantom(ph::Array{Float64, 4}, sliceinfo::Matrix{Int}; staticslices=1:200)

Construct a static phantom object from phantom time series `ph` and
slice motion data `sliceinfo`. The parameter `staticslices` allows to select
a number slices are used for construction of a model.
"""
function staticphantom(ph::Array{Float64,4}, sliceinfo::Matrix{Int};
    staticslices=1:200, interpolationtype=BSpline(Quadratic()))
    sno = size(sliceinfo, 1)
    @assert size(ph, 3) == sno "Incompatible data dimensions"

    # volume averaged over static slices
    staticph = dropdims(mean(view(ph, :, :, :, staticslices), dims=4), dims=4)

    # skip slices
    useslices = findall(.!iszero, sliceinfo[:, 1])

    # keep slices segmentation
    segments = Array{SegmentedImage}(undef, sno)

    # get ellipse parameters over averaged static volume (z-slices)
    ellipses = zeros(sno, 9)
    for i in 1:sno
        img = view(staticph, :, :, i)
        segments[i] = seg = segment3(img)
        ellipses[i, :] = fitellipse(img, seg, edge(seg); verbose=false)
    end

    # construct phantom center axis
    centers = fitline(ellipses[useslices, 1:2]')

    # construct interpolation object
    phantom_itp = phantominterp(staticph; interpolationtype)

    # Form phantom object
    StaticPhantom(staticph, ellipses, sliceinfo, vec(centers.μ), centers.v, phantom_itp, segments)
end

"""
Phantom Simulated Ground Truth

A ground truth representation of a phantom motion data given a rotation information.
It contains a tensor of original and predicted values for the masked volume of
an original phantom data, a slice index map, and a mask index map for translation to
the original phantom coordinate space.
"""
struct GroundTruth
    data::Array{Float64,4}
    sliceindex::Vector{Int}
    maskindex::Matrix{Int}
end

struct GroundTruthCleaned
    data::Array{Float64,3}
end

Base.show(io::IO, gt::GroundTruth) = print(io, "GroundTruth: $(size(gt.data))")

"""
    getindex(gt::GroundTruth, x::Int, y::Int)

Return an index of a masked phantom volume.
"""
Base.getindex(gt::GroundTruth, x::Int, y::Int) =
    findfirst(c -> c[1] == x && c[2] == y, eachcol(gt.maskindex))

"""
    getindex(gt::GroundTruth, x::Int, y::Int, z::Int, original::Bool=false)

Return a phantom predicted ground truth data in the coordinate `(x,y,z)`.
If `original` is set to `true`, original phantom data is returned.
"""
function Base.getindex(gt::GroundTruth, x::Int, y::Int, z::Int, original::Bool=false)
    midx = gt[x, y]
    isnothing(midx) && return
    view(gt.data, :, midx, z, (original + 1))
end

"""
    maskindex(mask::BitMatrix)

Construct a coordinate index for phantom slice mask.
"""
function maskindex(mask::BitMatrix)
    n = sum(mask)
    res = zeros(Int, 2, n)
    ci = 1
    for (c, v) in pairs(mask)
        !v && continue
        res[1, ci] = c[1]
        res[2, ci] = c[2]
        ci += 1
    end
    return res
end

"""
    groundtruth(ph::StaticPhantom, data::AbstractArray, angles::Vector;
                startmotion=1, threshold=Inf, verbose=false)

Construct a prediction of a phantom motion data given a rotation information,
and return a tensor of original and predicted values for the masked volume of
the phantom `ph`, slice indices, and mask index map for translation to the original
phantom coordinate space.
"""
function groundtruth(ph::StaticPhantom, data::AbstractArray, angles::Vector;
    startmotion=1, threshold=Inf, verbose=false, flipangles=false)
    # get motion angles
    motionangles = @view angles[startmotion:end]
    if flipangles
        motionangles .= -motionangles
    end
    nrots = length(motionangles)

    # take mask of a first slice
    maskidx = mask(ph, 1; threshold) |> maskindex
    ncoords = size(maskidx, 2)
    #ncoords = sum(maskmtx)

    # # get coordinate map
    # cmap = findall(maskmtx)

    # get valid slices
    validslices = ph.sliceinfo[:, 1] .> 0
    nslices = sum(validslices)
    # construct predictions
    res = zeros(nrots, ncoords, nslices, 2)
    for (kk, z) in pairs(findall(validslices))
        # get a ellipse's params
        origin, a, b = getellipse(ph, z)
        γ = findinitialrotation(ph, z)
        for (ii, α) in pairs(motionangles), (jj, ci) in pairs(eachcol(maskidx))
            i, j = ci
            # Coordinate transformation
            coord = ellipserot(α, γ, a, b) * ([i, j, z] .- origin) .+ origin
            # interpolate intensities
            pred = ph.interpolation(coord...)
            res[ii, jj, kk, 1] = pred
            res[ii, jj, kk, 2] = data[i, j, z, ii+startmotion-1]
        end
        verbose && println("Processed slice: $z")
    end
    # get slice identifiers
    sliceids = Int.(ph.sliceinfo[validslices, 2])
    # return predictions, slice ids & coordinate map
    GroundTruth(res, sliceids, maskidx)
end

"""
    serialize(filepath::String, gt::GroundTruth)
    serialize(filepath::String, gt::GroundTruthCleaned)

Serialize a phantom ground truth data object `gt` into HDF5-formatted file.

Example:

```julia
julia> BDTools.serialize("gt.h5", gt)
```
"""
function serialize(filepath::String, gt::GroundTruth)
    HDF5.h5open(filepath, "w") do io
        g = HDF5.create_group(io, "GroundTruth")
        dset = HDF5.create_dataset(g, "data", eltype(gt.data), size(gt.data))
        HDF5.write(dset, gt.data)
        dset = HDF5.create_dataset(g, "sliceindex", eltype(gt.sliceindex), size(gt.sliceindex))
        HDF5.write(dset, gt.sliceindex)
        dset = HDF5.create_dataset(g, "maskindex", eltype(gt.maskindex), size(gt.maskindex))
        HDF5.write(dset, gt.maskindex)
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
    deserialize(filepath::String, ::Type{GroundTruth})
    deserialize(filepath::String, ::Type{GroundTruthCleaned})

Deserialize a phantom ground truth data from a HDF5-formatted file, and
construct `GroundTruth` object.

Example:

```julia
julia> gt = BDTools.deserialize("gt.h5", GroundTruth)
GroundTruth: (598, 359, 9, 2)
```
"""
function deserialize(filepath::String, ::Type{GroundTruth})
    HDF5.h5open(filepath, "r") do io
        GroundTruth(
            io["GroundTruth/data"] |> read,
            io["GroundTruth/sliceindex"] |> read,
            io["GroundTruth/maskindex"] |> read
        )
    end
end

function deserialize(filepath::String, ::Type{GroundTruthCleaned})
    HDF5.h5open(filepath, "r") do io
        GroundTruthCleaned(
            io["GroundTruth/data"] |> read
        )
    end
end

# Denoiser CNN
include("denoiser.jl")
using .Denoiser
using .Denoiser: Denoiser

end # module BDTools