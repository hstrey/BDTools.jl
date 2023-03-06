module BDTools

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
end

Base.show(io::IO, ph::StaticPhantom) = print(io, "Phantom: $(size(ph.data))")

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
    staticphantom(ph::Array{Float64, 4}, sliceinfo::Matrix{Int}; staticslices=1:200)

Construct a static phantom object from phantom time series and slice motion data.
"""
function staticphantom(ph::Array{Float64, 4}, sliceinfo::Matrix{Int}; staticslices=1:200)
    sno = size(sliceinfo, 1)
    @assert size(ph, 3) == sno "Incompatible data dimensions"

    # volume averaged over static slices
    staticph = dropdims(mean(view(ph,:,:,:,staticslices), dims=4), dims=4);

    # skip slices
    useslices = findall(.!iszero, sliceinfo[:,1])

    # get ellipse parameters over averaged static volume (z-slices)
    ellipses = zeros(sno, 9)
    for i in 1:sno
        ellipses[i,:] = getellipse(view(staticph,:,:,i), verbose=false)
    end

    # construct phantom center axis
    centers = fitline(ellipses[useslices, 1:2]')

    # construct interpolation object
    phantom_itp = phantominterp(staticph)

    # Form phantom object
    StaticPhantom(staticph, ellipses, sliceinfo, vec(centers.μ), centers.v, phantom_itp)
end

end # module BDTools