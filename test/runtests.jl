using Revise
using Test
using BDTools


@testset "Segmentation" begin
    img = zeros(5,5)

    simg = BDTools.segment3(img)
    @test length(simg.segment_labels) >= 3

    edg = BDTools.edge(simg)
    @test sum(edg) > 1

    @test size(BDTools.getcoordinates(ones(5,3,5))) == (3,75)
end

@testset "Ellipse detection" begin
    # 45 degree rotated ellipese
    img = zeros(6,6,5)
    for i in 1:4, j in 1:4
        i == 4 && j == 1 && continue
        i == 1 && j == 4 && continue
        img[i+1, j+1, :] .= 1
    end

    # check direct fitting method
    coords = hcat(([i.I...] for i in findall(img[:,:,1] .> 0))...)
    elps = BDTools.canonical(BDTools.fitellipsedirect(coords))
    @test elps[1] ≈ 3.5
    @test elps[2] ≈ 3.5
    @test elps[6] ≈ 45

    # check curve fitting method
    elps = BDTools.getellipse3d(img, verbose=false)
    @test elps[1] ≈ 3.89 atol=0.1
    @test elps[2] ≈ 2.9 atol=0.1
    @test rad2deg(elps[5]) ≈ 10.7 atol=0.1
end