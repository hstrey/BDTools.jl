using Statistics
using Turing
using LinearAlgebra

"""
    noise_model(pred_ts, orig_ts)

Turing probabilistic model for signal with added white and multiplicative noise

"""
@model function noise_model(pred_ts, orig_ts)
    σ ~ Uniform(0, 20)
    ampl ~ Uniform(0, 5)
    orig_ts ~ MvNormal(pred_ts, sqrt.(σ^2 .+ ampl^2 .* pred_ts .^ 2))
end

"""
mul_noise(pred_ts, orig_ts)

Calculates the amplitudes of white and multiplicative noise using HMC sampling
Expects normalized pred_ts (mean=0, std=1) and orig_ts (mean=0, std using pred_ts)

"""
function mul_noise(pred_ts, orig_ts)
    mymodel = noise_model(pts_norm, ots_norm)
    chain = Turing.sample(mymodel, NUTS(0.65), 1000)
    sigma = chain[:σ]
    amplitude = chain[:ampl]
    return mean(sigma), std(sigma), mean(amplitude), std(amplitude)
end

"""
    st_snr(pred_ts, orig_ts)

Calculates the standardized signal-to-noise ratio: Power(signal)/Power(noise)

"""
function st_snr(pred_ts, orig_ts)
    pts_std = std(pred_ts, dims=1)
    noise_std = std(orig_ts .- pred_ts, dims=1)
    return mean(pts_std .^ 2 ./ noise_std .^2), std(pts_std .^ 2 ./ noise_std .^2)
end
