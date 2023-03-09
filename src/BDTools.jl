module BDTools

using Optim: Optim
using Interpolations

export StaticPhantom, staticphantom

include("segmentation.jl")
include("ellipse.jl")
include("utils.jl")

"""
A static phantom model
"""
struct StaticPhantom
    data::Array{Float64, 3}
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
centers(ph::StaticPhantom) = ph.ellipses[:,1:2]

"""
    predictcenter(ph::StaticPhantom, t::Real)

Return a center coordinate of a phantom slice given a parametric value `t`.
"""
predictcenter(ph::StaticPhantom, t::Real) = ph.center + ph.centersdir*t

"""
    fittedcenters(ph::StaticPhantom)

Return a projection of fitted ellipses' center to a center axis.
"""
fittedcenters(ph::StaticPhantom) = mapslices(c->fittedcenter(ph, vec(c)), centers(ph), dims=2)

"""
    fittedcenter(ph::StaticPhantom)

Return a projection of fitted ellipses' center to a center axis.
"""
function fittedcenter(ph::StaticPhantom, p::Vector)
    p0 = ph.center
    dir = ph.centersdir
    p0+(p-p0)'*dir*dir
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
        threshfn = ci -> (ci.I[1]-o[1])^2/a^2+(ci.I[2]-o[2])^2/b^2 > threshold
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
    start = fittedcenter(ph, cs[z,:])
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
function findinitialrotation(ph::StaticPhantom, z::Int; intensity = 0.01)
    # get a phantom slice
    @assert 1 <= z <= size(ph)[3] "Invalid slice index"
    img = @view ph.data[:,:,z]

    # mask all points outside the internal region of a phantom
    maskidx = mask(ph, z)
    masked = @view img[maskidx]

    # threshold intensity to get a collection of darkest points
    mini, maxi = extrema(masked)
    irng = maxi-mini
    mininidx = findall(masked .< mini+irng*intensity)

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
            coords = map(α->ellipserot(α, γ, a, b)*(p.-ori).+ori, 0.0:0.1:π)
            sim = map(c->ph.interpolation(c...), coords)
            res += abs(-(extrema(sim)...))
        end
        res
    end

    res = Optim.optimize(f, 0.0, float(π))
    Optim.minimizer(res)
end

"""
    staticphantom(ph::Array{Float64, 4}, sliceinfo::Matrix{Int}; staticslices=1:200)

Construct a static phantom object from phantom time series and slice motion data.
"""
function staticphantom(ph::Array{Float64, 4}, sliceinfo::Matrix{Int};
                       staticslices=1:200, interpolationtype = BSpline(Quadratic()))
    sno = size(sliceinfo, 1)
    @assert size(ph, 3) == sno "Incompatible data dimensions"

    # volume averaged over static slices
    staticph = dropdims(mean(view(ph,:,:,:,staticslices), dims=4), dims=4);

    # skip slices
    useslices = findall(.!iszero, sliceinfo[:,1])

    # keep slices segmentation
    segments = Array{SegmentedImage}(undef, sno)

    # get ellipse parameters over averaged static volume (z-slices)
    ellipses = zeros(sno, 9)
    for i in 1:sno
        img = view(staticph,:,:,i)
        segments[i] = seg = segment3(img)
        ellipses[i,:] = fitellipse(img, seg, edge(seg); verbose=false)
    end

    # construct phantom center axis
    centers = fitline(ellipses[useslices, 1:2]')

    # construct interpolation object
    phantom_itp = phantominterp(staticph; interpolationtype)

    # Form phantom object
    StaticPhantom(staticph, ellipses, sliceinfo, vec(centers.μ), centers.v, phantom_itp, segments)
end

"""
    predict(ph::StaticPhantom, data::AbstractArray, angles::Vector;
            startmotion=1, threshold=Inf, verbose=false)

Construct a prediction of a phantom motion data given a rotation information
"""
function predict(ph::StaticPhantom, data::AbstractArray, angles::Vector;
                 startmotion=1, threshold=Inf, verbose=false)
    # get motion angles
    motionangles = @view angles[startmotion:end]
    nrots = length(motionangles)

    # take mask of a first slice
    maskmtx = mask(ph, 1; threshold)
    ncoords = sum(maskmtx)

    # get coordinate map
    cmap = findall(maskmtx)

    # get valid slices
    validslices = ph.sliceinfo[:,1] .> 0
    nslices = sum(validslices)
    # construct predictions
    res = zeros(nrots, ncoords, nslices, 2)
    for (kk, z) in pairs(findall(validslices))
        # get a ellipse's params
        origin, a, b = getellipse(ph, z)
        γ = findinitialrotation(ph, z)
        for (ii, α) in pairs(motionangles), (jj,ci) in pairs(cmap)
            i, j = ci.I
            # Coordinate transformation
            coord = ellipserot(α, γ, a, b)*([i,j,z].-origin).+origin
            # interpolate intensities
            pred = ph.interpolation(coord...)
            res[ii, jj, kk, 1] = pred
            res[ii, jj, kk, 2] = data[i,j,z,ii+startmotion-1]
        end
        verbose && println("Processed slice: $z")
    end
    # get slice identifiers
    sliceids = Int.(ph.sliceinfo[validslices,2])
    # return predictions, slice ids & coordinate map
    res, sliceids, cmap
end

end # module BDTools