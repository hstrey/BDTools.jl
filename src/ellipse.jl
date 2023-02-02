using LsqFit: curve_fit
using LinearAlgebra

"""
    fitellipse(xy::AbstractMatrix)

Perform an ellipse fitting from a collection of 2D coordinates `xy`.

Implemented from "Direct least squares fitting of ellipses", Fitzgibbon, 1996.
"""
function fitellipse(xy::AbstractMatrix)
    # design matrix
    D = let x = xy[1,:], y = xy[2,:]
        [x.*x x.*y y.*y x y ones(size(xy,2))]
    end
    # scatter matrix
    S = D' * D
    # constraint matrix
    C = zeros(6,6)
    C[1, 3] = 2
    C[2, 2] = -1
    C[3, 1] = 2
    # solve eigensystem
    F = eigen(inv(S) * C)
    F.vectors[:, findmax(F.values) |> last]
end


"""
    fitellipsedirect(xy::AbstractMatrix)

Perform an ellipse fitting from a collection of 2D coordinates `xy`.

Implemented from "Numerically stable direct least squares fitting of ellipses", Halir, 1998.
"""
function fitellipsedirect(xy::AbstractMatrix)
    D1, D2 = let x = xy[1,:], y = xy[2,:]
        # quadratic part of the design matrix
        [x.*x x.*y y.*y],
        # linear part of the design matrix
        [x y ones(size(xy,2))]
    end
    # quadratic part of the scatter matrix
    S1 = D1' * D1
    # combined part of the scatter matrix
    S2 = D1' * D2
    # linear part of the scatter matrix
    S3 = D2' * D2

    T = -inv(S3) * S2' # for getting a2 from a1
    M = S1 + S2 * T # reduced scatter matrix
    M = [M[3, :]./2 -M[2, :] M[1, :]./2] # premultiply by inv(C1)

    # solve eigensystem
    F = eigen(M)
    evec = real.(F.vectors)
    cond = 4 * evec[1, :] .* evec[3, :] - evec[2, :].^2 # evaluate a’Ca
    a1 = evec[:, findall(cond .> 0)] # eigenvector for min. pos. eigenvalue
    [a1; T * a1] |> vec
end

"""
    canonical(p::Vector)

Convert ellipse parameters `p` (6-element vector) in general form into canonical:
center coordinates, axes & rotation angle in radians and
"""
function canonical(params::Vector)
    (A,B,C,D,E,F) = params
    c1 = B^2-4*A*C
    c2 = 2(A*E^2 + C*D^2 - B*D*E + c1*F)
    c3 = sqrt((A-C)^2+B^2)
    a = -sqrt(c2*(A+C+c3))/c1
    b = -sqrt(c2*(A+C-c3))/c1
    x = (2*C*D-B*E)/(c1)
    y = (2*A*E-B*D)/(c1)
    θ = (C - A - c3)/B |> atan
    x, y, a, b, θ, rad2deg(θ)
end

"""
    gaussellipse3d(xyz, p)

Estimate intensity difference for a Gaussian ellipse at each point of 3D tensor `xyz`
given the ellipse parameters in `p`.
"""
function gaussellipse3d(xyz, p)
    x0,y0,rx,ry,θ,α,A,bg,σ = p #unpack parameters
	ct, st = cos(θ), sin(θ)
	c1, c2 = ct*cos(α), ct*sin(α)
	# xc = z->z.*cos(θ).*cos(α).+x0
	# yc = z->z.*cos(θ).*sin(α).+y0
	# dx = map(r->r[1]-r[3]*c1+x0, eachrow(xyz))
	# dy = map(r->r[2]-r[3]*c2+y0, eachrow(xyz))
	# dx = view(xyz, :, 1) .- xc(view(xyz, :, 3))
    # dy = view(xyz, :, 2) .- yc(view(xyz, :, 3))
	dxy = hcat(([i-k*c1+x0, j-k*c2+y0] for (i,j,k) in eachcol(xyz))...)
	dx = view(dxy, 1, :)
    dy = view(dxy, 2, :)
    return bg .- A * exp.( -(1 .-sqrt.(( dx .* ct .+ dy .* st ).^2/rx^2+(dx .* st .- dy .* ct).^2/ry^2)).^2/σ^2)
end

"""
    prepare_initial_point(p::Vector) -> Vector

Correct an initial solution for ellipse fitting optimization problem given
a parameter vector `p`: x-coordinate of an ellipse center, y-coordinate of an ellipse center,
long axis, short axis, an ellipse rotation angle in XY-plane, an ellipse rotation angle in XZ-plane,
coordinate intensity value, background intensity, variance.
"""
function prepare_initial_point(p::Vector)
    x0, y0, a, b, θ, α, A, bg, σ = p
    # flip axes so a > b
    dont_flip = a > b
    a, b = dont_flip ? (a,b) : (b,a)
    # if the axes were flipped, then rotate ellipse by π/2
    θ = dont_flip ? θ : (θ - π/2)
    #  make sure that angle within [0;π]
    θ %= π
    θ = θ < 0 ? π + θ : θ
    # form initial solution
    [x0, y0, a, b, θ, α, (A < 0 ? 0.0 : A), bg, abs(σ)]
end

"""
    fitellipse3d(imgs::AbstractArray, mask::BitArray, edge::Matrix{Int};
                 verbose=true, keepinitialonerror=true, secondfit=true) -> Vector

Perform fitting of an ellipse given an image tensor `imgs`, `mask` for fitting coordinates,
and `edge` elements of an ellipse shape. Returns a vector with fitted ellipse parameters.

If `secondfit` is `true`, then an additional fitting is performed after the result of
first fitting is corrected.
"""
function fitellipse3d(imgs::AbstractArray, mask::BitArray, edge::Matrix{Int};
                     verbose=true, keepinitialonerror=true, secondfit=true)
    r,c,h = size(imgs)
    # find ellipse in edge
    coords = hcat(([i.I...] for i in findall(edge.>0))...)
    E1 = fitellipsedirect(coords) |> canonical
    verbose && @debug "Initial fit" E1

    # get all points within the outer boundary
    idxs = findall(mask)
    coords2 = hcat(([ci.I...] for ci in idxs)...)
    z = @view imgs[idxs]

    # refine ellipse parameters using outer segment points
    # and edge ellipse estimate
    mmz = extrema(z)
    p0 = prepare_initial_point([E1[1:5]..., π/40, mmz..., 0.01])
    fit = curve_fit(gaussellipse3d, coords2, z, p0, autodiff=:forwarddiff)
    E = fit.param
    verbose && @debug "LSQ fit 1" p0 E

    # refine again
    lb = [c/2-3.0, r/2-3.0, 0.0, 0.0, -π, -π/2,   0,   0, 0.001]
    ub = [c/2+3.0, r/2+3.0, Inf, Inf,  π,  π/2, Inf, Inf, 1.000]
    p0 = prepare_initial_point(E)
    E = try
        if secondfit
            fit = curve_fit(gaussellipse3d, coords2, z, p0, lower=lb, upper=ub, autodiff=:forwarddiff)
            fit.param
        else
            p0
        end
    catch ex
        p0 = prepare_initial_point([E1[1:5]..., 0, mmz[1], mmz[2]/2, 0.01])
        if keepinitialonerror
            p0
        else
            @debug "Error. Trying with constraints." p0
            fit = curve_fit(gaussellipse3d, coords2, z, p0, lower=lb, upper=ub)
            fit.param
        end
    end
    verbose && secondfit && @debug "LSQ fit 2" p0 E

    return E
end

"""
    getellipse3d(imgs::AbstractArray; verbose=true, secondfit=true) -> Vector

Returns a vector with fitted ellipse parameters from a 3D image tensor.
"""
function getellipse3d(imgs::AbstractArray; verbose=true, secondfit=true)
    segs = segment3.(eachslice(imgs, dims=3))
    mask = cat(labels_map.(segs)..., dims=3) .!= 1
    fitellipse3d(imgs, mask, sum(edge.(segs)); verbose, secondfit)
end