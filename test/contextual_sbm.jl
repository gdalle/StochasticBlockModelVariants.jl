using Random: default_rng
using StochasticBlockModelVariants
using Test

rng = default_rng()

# Parameters

d = 5.0
λ = 2.0
μ = 2.0
N = 3 * 10^3
P = 3 * 10^2

init_std = 1e-3
iterations = 10

# Sampling

csbm = ContextualSBM(; d, λ, μ, N, P)

(; latents, observations) = rand(rng, csbm);

# Inference

(; storage, next_storage, temp_storage) = init_amp(rng; observations, csbm, init_std);

storage_history = run_amp(rng; observations, csbm, init_std, iterations);

@test effective_snr(csbm) > 1

overlap_history = [evaluate_amp(; storage, latents) for storage in storage_history]

@test_broken last(overlap_history) > 0.5
