using Images
using ImageSegmentation

genimg(p::AbstractArray) = p ./ maximum(p)

function genunitimg!(img, arr)
    minval, maxval = extrema(arr)
    diff = maxval-minval
    broadcast!(x-> (x-minval)/diff, img, arr)
end

"""
    getcoordinates(img::AbstractArray; threshold=0) -> Matrix{Int64}

Return a matrix of indices for elements, which values are grater then `threshold`,
in an input array `img`.
"""
function getcoordinates(img::AbstractArray; threshold=0)
    hcat(([i.I...] for i in findall(img .> threshold))...)
end

"""
    segment3(img::AbstractArray{T})-> SegmentedImage

Segment an BD image on three segments: outer region, outer & inner cylinders
"""
function segment3(img::AbstractArray; s2offset=10)
    # outer boundary segmentation of the image
    sz = size(img)
    out_seeds = [(CartesianIndex(1,1), 1),
                 (CartesianIndex(sz[1],sz[2]), 1),
                 (CartesianIndex(sz[1]>>1,sz[2]>>1), 2)]
    outsegs = seeded_region_growing(img, out_seeds)

    # inner boundary segmentation of the image
    idxs = findall(labels_map(outsegs) .== 2)
    in_seeds = [(idxs[1], 1), (idxs[s2offset], 1), (idxs[end-s2offset], 1), (idxs[end], 1),
                (CartesianIndex(sz[1]>>1,sz[2]>>1), 2)]
    insegs = seeded_region_growing(img, in_seeds)

    # combine segments
    push!(outsegs.segment_labels, 3)
    outsegs.segment_means[3] = segment_mean(insegs, 2)
    outsegs.segment_pixel_count[2] -= insegs.segment_pixel_count[2]
    outsegs.segment_pixel_count[3] = insegs.segment_pixel_count[2]
    outsegs.image_indexmap[insegs.image_indexmap .== 2] .= 3
    return outsegs
end

"""
    edge(segs::SegmentedImage) -> Matrix{Bool}

Construc an edge mask for a segmented image.
"""
function edge(segs::SegmentedImage; segmentid=3, upper=80, lower=20)
    # get inner region
    inner = labels_map(segs) .== segmentid
    # get an inner edge
    img_edges = canny(inner, (Percentile(upper), Percentile(lower)), 2);
    # remove any points outside of outer boundary
    for ci in findall(labels_map(segs) .== 1)
        img_edges[ci] = 0
    end
    img_edges
end
