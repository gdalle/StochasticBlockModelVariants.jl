using Random: default_rng
using Statistics
using StochasticBlockModelVariants
using Test

rng = default_rng()

function test_recovery(; N, P, d, λ, μ, ρ, init_std=1e-3, iterations=20)
    csbm = ContextualSBM(; N, P, d, λ, μ, ρ)
    @assert effective_snr(csbm) > 1
    (; latents, observations) = rand(rng, csbm)
    storage_history = run_amp(rng; observations, csbm, init_std, iterations)
    overlap_history = [evaluate_amp(; storage, latents) for storage in storage_history]
    @test last(overlap_history) > 10 * first(overlap_history)
    @test last(overlap_history) > 0.5
    return nothing
end

test_recovery(; N=10^3, P=10^3, d=5, λ=2, μ=2, ρ=0.0)  # AMP-BP
test_recovery(; N=10^3, P=10^3, d=5, λ=0, μ=2, ρ=0.0)  # AMP
test_recovery(; N=10^3, P=10^3, d=5, λ=2, μ=0, ρ=0.0)  # BP
