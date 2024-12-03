### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ ce42a378-9cdf-11ee-114c-5d772b3e1943
begin
	using Pkg
	Pkg.activate(temp = true)

	Pkg.add(url = "https://github.com/hstrey/BDTools.jl")
	Pkg.add.(["Plots", "DataFrames", "Distributions", "Turing", "Statistics", "StatsBase"])

	using BDTools
	using Distributions
	using Turing
	using Plots
	using DataFrames
	using Statistics
	using StatsBase
end

# ╔═╡ 58bbb316-36f1-49a2-b4a0-a4cdb356597b
Pkg.add("StatsPlots")

# ╔═╡ 6efbaf5e-6438-44e3-b2ef-c551a777e48e
Pkg.add("Polynomials")

# ╔═╡ 23a9d18b-6eb0-4ba0-88a8-1f5c53848c92
using StatsPlots

# ╔═╡ 9120f285-8251-4009-a757-21d38aa8b11a
using Polynomials

# ╔═╡ 4aa4d29d-6636-41eb-9d32-6efbb040fd4f
gt = BDTools.deserialize("gt_clean.h5", GroundTruthCleaned)

# ╔═╡ c7087ebe-464b-4237-bec5-0a84cb28b96f
ori = gt.data[:,:,1]

# ╔═╡ 96dee18e-e3cb-4c75-8629-46f13f17a513
sim = gt.data[:,:,2]

# ╔═╡ 758168e6-af1e-4fed-8254-a1bb11c63820
ori_mean = mean(ori, dims=1)

# ╔═╡ 9f8e44d5-d737-422a-a30c-4e334e2f6552
ori_norm = (ori .- ori_mean)

# ╔═╡ f2f9d160-4924-4740-bac9-a3762a24029d
sim_mean = mean(sim, dims=1)

# ╔═╡ b59235eb-6001-496d-be8a-ec22ca1f7147
sim_mean[1,1]

# ╔═╡ 9b842333-2b6c-4424-8592-748cae0846ca
sim_norm = (sim .- sim_mean)

# ╔═╡ 05749d94-a3c5-4fa5-9cc5-e3177fe2cd26
sim_std = std(sim, dims=1)

# ╔═╡ f6e08db9-95e3-464c-8db7-afa52dadf469
sim_norm_vec = vec(sim_norm) ./ std(vec(sim_norm))

# ╔═╡ 1e2208dd-11da-4c7d-a8c7-9a9a30c69b40
norm_const = std(vec(sim_norm))

# ╔═╡ d51566dd-16d7-4089-a007-416b8e255390
ori_norm_vec = vec(ori_norm) ./ std(vec(sim_norm))

# ╔═╡ 7f91773b-df7b-4e68-98c2-6c27d3699cde
"""
    noise_model(pred_ts, orig_ts)

Turing probabilistic model for signal with added white and multiplicative noise

"""
@model function noise_model(pred_ts, orig_ts)
    ampl ~ Uniform(0, 5)
	σ ~ Uniform(0,20)
    orig_ts ~ MvNormal(pred_ts, sqrt.(σ^2 .+ ampl^2 .* pred_ts .^ 2))
end

# ╔═╡ 0c01ddd4-b159-4782-8a8e-c91850d0017b
mymodel = noise_model(sim_norm_vec, ori_norm_vec)

# ╔═╡ 66272f86-cd13-4434-a59a-c6e02d1ff9cf
chain = Turing.sample(mymodel, NUTS(0.65), 2000)

# ╔═╡ 0e0fea7e-2432-4c08-8995-1e2b9c092fc6
plot(chain)

# ╔═╡ 31b9dbca-1d92-420a-870f-7e641e397726
sigma = vec(chain[:σ])

# ╔═╡ 203c0a19-3bc9-4710-96bf-bdcd8d50b63b
mean_sigma = mean(sigma)

# ╔═╡ 061153e3-9a9e-40bf-8483-0bf9afd9ba0f
amplitude = vec(chain[:ampl])

# ╔═╡ 9e30431e-978b-4410-be92-cd85c12c1492
ampl_mean = mean(amplitude)

# ╔═╡ ed245c2e-a3b0-4bbb-aa6c-e37ae13bf63c
per_signal = vec(sim_std ./mean(vec(sim_mean))) .*100

# ╔═╡ a01528d4-d880-47dc-a71b-05ede029249e
histogram(vec(per_signal))

# ╔═╡ a1abd71f-de82-4d10-ac0c-b9f12c567cf3
sim_std_dist = vec(sim_std)/norm_const

# ╔═╡ d1373b74-b65e-435f-b422-abc355addcf6
perc_mult = sqrt.(ampl_mean^2 .* sim_std_dist.^2 ./ (mean_sigma^2 .* norm_const^2 .+ ampl_mean^2 .* sim_std_dist.^2))

# ╔═╡ 3eb22cdc-f4b7-4b97-8dd6-8fa04dcfd897
histogram(perc_mult)

# ╔═╡ 68c6089c-3b6f-485c-92a9-cfd5fda40858
mean(vec(sim_mean))

# ╔═╡ 48c2e68f-a445-431d-9221-3ed056175253
theory_sim_std_dist = vec(per_signal) .* mean(vec(sim_mean)) ./ 100 ./ norm_const

# ╔═╡ c6564224-5737-4d9b-91ec-b013f55bfa93
theory_per_mult = sqrt.(ampl_mean^2 .* theory_sim_std_dist.^2 ./ (mean_sigma^2 .* norm_const^2 .+ ampl_mean^2 .* theory_sim_std_dist.^2))

# ╔═╡ 20ec677b-97b8-4359-8ce1-5e79d194b390
begin
	scatter(vec(per_signal),perc_mult)
	scatter!(vec(per_signal),theory_per_mult)
end

# ╔═╡ 08c44b57-f9a3-434c-802c-052aee8c3392
100*100

# ╔═╡ Cell order:
# ╠═ce42a378-9cdf-11ee-114c-5d772b3e1943
# ╠═58bbb316-36f1-49a2-b4a0-a4cdb356597b
# ╠═23a9d18b-6eb0-4ba0-88a8-1f5c53848c92
# ╠═6efbaf5e-6438-44e3-b2ef-c551a777e48e
# ╠═9120f285-8251-4009-a757-21d38aa8b11a
# ╠═4aa4d29d-6636-41eb-9d32-6efbb040fd4f
# ╠═c7087ebe-464b-4237-bec5-0a84cb28b96f
# ╠═96dee18e-e3cb-4c75-8629-46f13f17a513
# ╠═758168e6-af1e-4fed-8254-a1bb11c63820
# ╠═9f8e44d5-d737-422a-a30c-4e334e2f6552
# ╠═f2f9d160-4924-4740-bac9-a3762a24029d
# ╠═b59235eb-6001-496d-be8a-ec22ca1f7147
# ╠═9b842333-2b6c-4424-8592-748cae0846ca
# ╠═05749d94-a3c5-4fa5-9cc5-e3177fe2cd26
# ╠═f6e08db9-95e3-464c-8db7-afa52dadf469
# ╠═1e2208dd-11da-4c7d-a8c7-9a9a30c69b40
# ╠═d51566dd-16d7-4089-a007-416b8e255390
# ╠═7f91773b-df7b-4e68-98c2-6c27d3699cde
# ╠═0c01ddd4-b159-4782-8a8e-c91850d0017b
# ╠═66272f86-cd13-4434-a59a-c6e02d1ff9cf
# ╠═0e0fea7e-2432-4c08-8995-1e2b9c092fc6
# ╠═31b9dbca-1d92-420a-870f-7e641e397726
# ╠═203c0a19-3bc9-4710-96bf-bdcd8d50b63b
# ╠═061153e3-9a9e-40bf-8483-0bf9afd9ba0f
# ╠═9e30431e-978b-4410-be92-cd85c12c1492
# ╠═ed245c2e-a3b0-4bbb-aa6c-e37ae13bf63c
# ╠═a01528d4-d880-47dc-a71b-05ede029249e
# ╠═d1373b74-b65e-435f-b422-abc355addcf6
# ╠═a1abd71f-de82-4d10-ac0c-b9f12c567cf3
# ╠═3eb22cdc-f4b7-4b97-8dd6-8fa04dcfd897
# ╠═20ec677b-97b8-4359-8ce1-5e79d194b390
# ╠═68c6089c-3b6f-485c-92a9-cfd5fda40858
# ╠═48c2e68f-a445-431d-9221-3ed056175253
# ╠═c6564224-5737-4d9b-91ec-b013f55bfa93
# ╠═08c44b57-f9a3-434c-802c-052aee8c3392
