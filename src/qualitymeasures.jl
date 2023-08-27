using Statistics
using Turing
using LinearAlgebra

@model function noise_model(pred_ts, orig_ts)
    σ ~ Uniform(0, 20)
    ampl ~ Uniform(0, 5)
    orig_ts ~ MvNormal(pred_ts, sqrt.(σ^2 .+ ampl^2 .* pred_ts .^ 2))
end

"""
mul_noise(pred_ts, orig_ts)

Calculates the amplitudes of white and multiplicative noise using HMC sampling

"""
function mul_noise(pred_ts, orig_ts)
    pts_norm = pred_ts .- mean(pred_ts)
    ots_norm = orig_ts .- mean(orig_ts)
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
    pts_norm = pred_ts .- mean(pred_ts)
    ots_norm = orig_ts .- mean(orig_ts)
    return std(pts_norm)^2 / std(ots_norm .- pts_norm)^2
end
