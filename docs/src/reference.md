```@meta
DocTestSetup = :(using BDTools)
```

## Static Phantom

```@docs
BDTools.StaticPhantom
BDTools.staticphantom
BDTools.findinitialrotation(::StaticPhantom, ::Int)
BDTools.mask(::StaticPhantom, ::Int)
BDTools.GroundTruth
BDTools.groundtruth
Base.getindex(::GroundTruth, ::Int, ::Int)
Base.getindex(::GroundTruth, ::Int, ::Int, ::Int, ::Bool)
```

### I/O
```@docs
BDTools.serialize
BDTools.deserialize
```

## Ellipse Fitting

```@docs
BDTools.fitellipse
BDTools.fitellipsedirect
BDTools.canonical
```

## Utils

```@docs
BDTools.getellipse
BDTools.ellipserot
BDTools.rotate
BDTools.rotatevoxel
BDTools.simulated_coordinates
BDTools.phantominterp
BDTools.fitline
BDTools.getangles
BDTools.maskindex
```
## QualityMeasures

```@docs
BDTools.st_snr
BDTools.mul_noise
```
