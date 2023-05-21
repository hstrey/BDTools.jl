using Test
using BDTools
using NIfTI

@testset "Segmentation" begin
    img = zeros(5, 5)

    simg = BDTools.segment3(img)
    @test length(simg.segment_labels) >= 3

    edg = BDTools.edge(simg)
    @test sum(edg) > 1

    @test size(BDTools.getcoordinates(ones(5, 3, 5))) == (3, 75)
end

@testset "Ellipse detection" begin
    # 45 degree rotated ellipse
    img = zeros(6, 6, 5)
    for i in 1:4, j in 1:4
        i == 4 && j == 1 && continue
        i == 1 && j == 4 && continue
        img[i+1, j+1, :] .= 1
    end

    # check direct fitting method
    coords = hcat(([i.I...] for i in findall(img[:, :, 1] .> 0))...)
    elps = BDTools.canonical(BDTools.fitellipsedirect(coords))
    @test elps[1] ≈ 3.5
    @test elps[2] ≈ 3.5
    @test elps[6] ≈ 45

    # check curve fitting method
    elps = BDTools.getellipse(img[:, :, 1], verbose=false)
    @test elps[1] ≈ 3.89 atol = 0.1
    @test elps[2] ≈ 2.9 atol = 0.1
    @test rad2deg(elps[5]) ≈ 10.7 atol = 0.1

    # check curve fitting method
    elps = BDTools.getellipse3d(img, verbose=false)
    @test elps[1] ≈ 3.89 atol = 0.1
    @test elps[2] ≈ 2.9 atol = 0.1
    @test rad2deg(elps[5]) ≈ 10.7 atol = 0.1
end

@testset "B-field Correction" begin
    tempdir = mktempdir()
	array_size = (84, 84, 16)
	circle_radius = 10
    # Euclidean distance between two points
	distance(x1, y1, x2, y2) = sqrt((x2 - x1)^2 + (y2 - y1)^2)

    # Create an example phantom of random numbers from 0 to 1000 of size = array_size
	phantom = Float32.(rand(0:1000, array_size...))
	phantom_path = joinpath(tempdir, "image.nii")
    niwrite(phantom_path, NIVolume(phantom))

    # Create a circular mask of size = array_size with radius = circle_radius
	mask = Float64.([
	    distance(i, j, array_size[1] / 2, array_size[2] / 2) <= circle_radius ? 1 : 0
	    for i in 1:array_size[1], j in 1:array_size[2], k in 1:array_size[3]
	])
    mask_path = joinpath(tempdir, "mask.nii")
    niwrite(mask_path, NIVolume(mask))

    # Run b-field correction on the example phantom and mask
    input_image, mask, bfield, corrected_image = BDTools.bfield_correction(phantom_path, mask_path)
   	@test size(input_image) == size(corrected_image)
    @test length(size(input_image)) == 3
end