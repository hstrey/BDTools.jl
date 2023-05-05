# Tutorial

Load libraries required for this tutorial.

```julia
using BDTools
using Plots
using NIfTI
using DelimitedFiles
```

## Loading data

First, we load phantom data from a NIfTI formatted file.

```julia
const DATA_DIR = "../PhantomData"
phantom_ts = niread(joinpath(DATA_DIR, "BFC_time_series.nii"));
sz = size(phantom_ts)
```

Next, we load rotation angles from rotation data,

```julia
angles, firstrotidx = BDTools.getangles(joinpath(DATA_DIR, "epi",  "log.csv"))
```

and slice motion information.

```julia
sliceinfo, _ = readdlm(joinpath(DATA_DIR, "slices.csv"), ',', Int, header=true)
```

## Construct static phantom

Use `staticphantom` function to construct a static phantom object
by providing phantom data time series and slice motion info.
Resulting object contains an ellipse fit for each slice of a static phantom.

```julia
sph = staticphantom(convert(Array, phantom_ts), sliceinfo);
```

### Show phantom center axis

Using phantom fitted ellipse parameters, we construct a phantom center axis (z-axis),
and fit ellipse centers on this axis.

```julia
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
```

## Construct ground truth dataset

We can construct a ground truth data at any rotation angle.
Providing a rotation angle `α` and a slice coordinate `z`, we generate
a rotated phantom.

```julia
let α = deg2rad(10), z = 3
    # get ellipse parameters at slice z
    origin, a, b = BDTools.getellipse(sph, z)
    # get a ellipse's initial rotation angle
    γ = BDTools.findinitialrotation(sph, z)

    # Coordinate transformation
    coords = [BDTools.ellipserot(α, γ, a, b)*([i,j,z].-origin).+origin for i in 1:sz[1], j in 1:sz[2]]
    # interpolate intensities
    sim = map(c->sph.interpolation(c...), coords)
    # generate image
    gen = BDTools.Images.Colors.Gray.(sim |> BDTools.genimg)
    # show averaged image
    ave = BDTools.Images.Colors.Gray.(sph.data[:,:,z] |> BDTools.genimg)
    pave = plot(ave, aspect_ratio=1.0, axis=nothing, framestyle=:none, title="Slice $z", size=(300,350))
    # show generated image
    pgen = plot(gen, aspect_ratio=1.0, axis=nothing, framestyle=:none, title="Rotated at $(rad2deg(α))°", legend=:none)
    plot(pave, pgen)
end
```

## Generate rotated predictions

For a rotation information, we can generate a predictions of rotated phantoms.

```julia
gt = groundtruth(sph, phantom_ts, angles; startmotion=firstrotidx, threshold=.95)
```

and plot prediction and original data

```julia
let x = 42, y = 52, z = 3 # get coordinates
    cidx = gt[x, y] # get a masked coordinate index
    cidx === nothing && return

    # plot data
    plot(gt[x, y, z], label="prediction", title="Intensity (x=$x, y=$y, z=$z)")
    plot!(gt[x, y, z, true], label="original")
end
```

A generated phantom ground truth data can be serialized into HDF5 file format.
Use [`BDTools.serialize`](@ref) function to save data as follows:

```julia
BDTools.serialize("gt.h5", gt)
```

for reading serialized phantom ground truth data, use [`BDTools.deserialize`](@ref) function
that accepts file name and [`GroundTruth`](@ref) type as parameters.

```julia
gt = BDTools.deserialize("gt.h5", GroundTruth)
```
