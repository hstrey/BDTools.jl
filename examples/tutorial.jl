using BDTools
using Plots
using NIfTI
using DelimitedFiles

# ## Loading data
#
# First, we load phantom data from a NIfTI formatted file.
#
const DATA_DIR = "../PhantomData"
phantom_ts = niread(joinpath(DATA_DIR, "BFC_time_series.nii"));
sz = size(phantom_ts)

# Next, we load rotation angles from rotation data,
angles, firstrotidx = BDTools.getangles(joinpath(DATA_DIR, "epi",  "log.csv"))

# and slice motion information.
sliceinfo, _ = readdlm(joinpath(DATA_DIR, "slices.csv"), ',', Int, header=true)

# ## Construct static phantom
#
# Use `staticphantom` function to construct a static phantom object
# by providing phantom data time series and slice motion info.
# Resulting object contains an ellipse fit for each slice of a static phantom.
#
sph = staticphantom(convert(Array, phantom_ts), sliceinfo);

# ### Show phantom center axis
#
# Using phantom fitted ellipse parameters, we construct a phantom center axis (z-axis),
# and fit ellipse centers on this axis.
#
let ecs = BDTools.centers(sph), rng=collect(-1.:0.15:1.)
    # show original data
    p = scatter(ecs[:,1], ecs[:,2], label="centers", legend=:topleft)
    # show predicted phantom center axis
    cc = map(t->BDTools.predictcenter(sph, t), rng)
    plot!(p, map(first, cc), map(last, cc), label="axis")
    # project slice centers to a fitted center axis
    xy = BDTools.fittedcenters(sph)
    scatter!(p, xy[:,1], xy[:,2], label="fitted")
end

# ## Construct ground truth dataset
#

let α = deg2rad(30),
    yc = staticEs[2, sliceId2],   # slice y-center
    origin = [centersFit.fnx(yc), yc, 1],
    (a, b) = staticEs[3:4,sliceId2]

    # Coordinate transformation
    coords = [BDTools.ellipserot(α, γ, a, b)*([i,j,sliceId2].-origin).+origin for i in 1:sz[1], j in 1:sz[2]]
    # interpolate intensities
    sim = map(c->phantom_itp(c...), coords)
    # generate image
    gen = Gray.(sim |> genimg)
    # show averaged image
    ave = Gray.(staticimgs[:,:,sliceId2] |> genimg)
    pave = plot(ave, aspect_ratio=1.0, axis=nothing, framestyle=:none, title="img z=$sliceId2", size=(300,350))
    # show generated image
    pgen = plot(gen, aspect_ratio=1.0, axis=nothing, framestyle=:none, title="generated at $alpha_grad", legend=:none)
    hline!(pgen, [h], color=:red)
    vline!(pgen, [v], color=:green)
    plot(pave, pgen)
end




