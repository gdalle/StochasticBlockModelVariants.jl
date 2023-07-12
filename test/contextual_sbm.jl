using Random: default_rng
using Statistics
using StochasticBlockModelVariants
using Test

rng = default_rng()

function test_recovery(; d, λ, μ, N, P, init_std=1e-3, iterations=20)
    csbm = ContextualSBM(; d, λ, μ, N, P)
    @assert effective_snr(csbm) > 1
    (; latents, observations) = rand(rng, csbm)
    storage_history = run_amp(rng; observations, csbm, init_std, iterations)
    overlap_history = [evaluate_amp(; storage, latents) for storage in storage_history]
    @test last(overlap_history) > 10 * first(overlap_history)
    @test last(overlap_history) > 0.5
    return nothing
end

test_recovery(; d=5.0, λ=2.0, μ=2.0, N=10^3, P=10^3)  # AMP-BP
test_recovery(; d=5.0, λ=0.0, μ=2.0, N=10^3, P=10^3)  # AMP
test_recovery(; d=5.0, λ=2.0, μ=0.0, N=10^3, P=10^3)  # BP
