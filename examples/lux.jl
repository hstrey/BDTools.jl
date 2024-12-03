# ╔═╡ ae7c03f6-8b1c-4d26-aa28-9438a747986a
md"""
## Package imports
"""

# ╔═╡ 9dc705a4-3c49-4db7-ae2c-29e37a800f60
using PlutoUI

# ╔═╡ 7ed6c051-a65b-41ac-860a-85a16f905e80
using Lux

# ╔═╡ 240fc0fd-2503-4fa7-b378-7362ec929ed1
using LuxCUDA, LuxAMDGPU, Metal

# ╔═╡ 1142a0e4-3ab6-422c-9c54-9365302ece3a
using Zygote

# ╔═╡ d4c325db-80cd-460d-be19-a96f1a2e8876
using Random

# ╔═╡ d2e09982-a252-4c21-b313-929042f7affc
using Statistics

# ╔═╡ 5a361a24-a581-4821-89ee-4dc4e771bc34
using MLUtils

# ╔═╡ 71525259-217c-428e-ad9f-76a359d74872
using Optimisers

# ╔═╡ d4f90012-26c3-4a17-bd86-0746ed6a67ec
using ComponentArrays

# ╔═╡ 3ec7945a-e8b3-4c08-8d74-ca96dee4674e
TableOfContents()

# ╔═╡ 1720f534-1f31-43fb-ac00-95cc7c1a60dd
md"""
## Set GPU/CPU device
"""

# ╔═╡ f1337bab-f681-43a3-8d9f-c4c6e07865cb
if LuxCUDA.functional()
	@info "Using CUDA"
	dev = gpu_device()
elseif LuxAMDGPU.functional()
	@info "Using AMD"
	dev = gpu_device()
elseif Metal.functional()
	@info "Convolutions not yet supported on Metal. Using CPU"
	dev = cpu_device()
else
    @info "No GPU is available. Using CPU."
	dev = cpu_device()
end

# ╔═╡ 2f442280-ec1b-426f-aad0-ba78f102963d
md"""
# Lux Example
"""

# ╔═╡ 161893ae-1427-4101-b402-4a63cf24a39e
begin
	rng = Random.default_rng()
	Random.seed!(rng, 0)
end

# ╔═╡ a6ac3c9b-8180-4380-9c07-873a50a17253
md"""
## Data
"""

# ╔═╡ c11e185f-8d90-4410-b9c9-479f8cb0a413
function dataloader(img_size, batch_size)
	xs = rand(Float32, img_size...)
	ys = rand(Float32, img_size...)
	
	train_loader = DataLoader((xs, ys); batchsize = batch_size, shuffle = true)
end

# ╔═╡ fab3813b-362b-48ae-93ab-8c1659550cdc
train_data = dataloader((6000, 1, 128), 8)

# ╔═╡ bec57c14-c6cd-4020-8432-25270d147bc2
md"""
## Model
"""

# ╔═╡ 24eba775-4b8b-4847-a63d-c6ed887f773a
function create_denoiser()
    return Chain(
        Conv((9,), 1=>18, sigmoid, pad=SamePad()),
        [Chain(Conv((9,), 18=>18, pad=SamePad()), BatchNorm(18, sigmoid)) for _ in 1:6]...,
        Dropout(0.2),
        Conv((1,), 18=>1, pad=SamePad())
    )
end

# ╔═╡ c22615c0-745c-480c-9091-f918e431579d
model = create_denoiser()

# ╔═╡ 42a2ebc8-371a-4056-949f-1ae8fb74a6bc
begin
	# Initialize parameters and states
	ps, st = Lux.setup(rng, model)
	ps = ps |> ComponentArray
end

# ╔═╡ 52c22099-6a73-4bf9-a08c-6f43cde919d2
md"""
## Loss Function & Optimizer
"""

# ╔═╡ 1653b2cc-ca22-4fe0-a6d4-c5b7751d3ba6
begin
	function negr2(y_pred, y_true)
	    SS_tot = sum(abs2, y_true .- mean(y_true))
	    SS_res = sum(abs2, y_true .- y_pred)
	    r2 = 1 - SS_res / (SS_tot + eps())
	    return -r2
	end
	
	loss_function(ps, x, y) = negr2(model(x, ps, st)[1], y)
end

# ╔═╡ 7930dbbc-2469-4b05-8fb2-c6e50ec70c14
begin
	opt = Optimisers.Adam(1e-5)
	opt_state = Optimisers.setup(opt, ps)
end

# ╔═╡ 3516c02d-1935-4f21-940f-efc5cb52ee08
md"""
## Train
"""

# ╔═╡ 85655ee0-a54b-4d88-9be9-c0edc47628d1
function train(model, train_data, opt_state, ps, st; epochs=3)
    losses = Float64[]

    ps = ps |> dev
    st = st |> dev

    for epoch in 1:epochs
		@info epoch
        for (x, y) in train_data
			x = x |> dev
            y = y |> dev
            gs = gradient(loss_function, ps, x, y)[1]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)
        end

        # Calculate and record the average loss for this epoch
        avg_loss = mean(data -> loss_function(ps, data[1], data[2]), train_data)
        push!(losses, avg_loss)
    end

    return losses
end

# ╔═╡ 6385f97e-ab1b-4359-94f5-83ba32072f1d
losses = train(model, train_data, opt_state, ps, st)