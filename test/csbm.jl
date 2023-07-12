using JET
using Random: default_rng
using Statistics
using StochasticBlockModelVariants
using Test

rng = default_rng()

function test_recovery(csbm::ContextualSBM)
    @assert effective_snr(csbm) > 1
    (; latents, observations) = rand(rng, csbm)
    storage_history = run_amp(rng; observations, csbm)
    overlap_history = [evaluate_amp(; storage, latents) for storage in storage_history]
    @test last(overlap_history) > 10 * first(overlap_history)
    @test last(overlap_history) > 0.5
    return nothing
end

function test_jet(csbm::ContextualSBM)
    (; observations) = rand(rng, csbm)
    @test_opt target_modules = (StochasticBlockModelVariants,) run_amp(
        rng; observations, csbm, iterations=2
    )
    @test_call target_modules = (StochasticBlockModelVariants,) run_amp(
        rng; observations, csbm, iterations=2
    )
    return nothing
end

function test_allocations(csbm::ContextualSBM)
    (; observations) = rand(rng, csbm)
    (; storage, next_storage, temp_storage) = init_amp(
        rng; observations, csbm, init_std=1e-3
    )
    alloc = @allocated update_amp!(next_storage, temp_storage; storage, observations, csbm)
    @test alloc == 0
    return nothing
end

@testset "Correct code" begin
    test_recovery(ContextualSBM(; N=10^3, P=10^3, d=5, λ=2, μ=2, ρ=0.0))  # AMP-BP
    test_recovery(ContextualSBM(; N=10^3, P=10^3, d=5, λ=0, μ=2, ρ=0.0))  # AMP
    test_recovery(ContextualSBM(; N=10^3, P=10^3, d=5, λ=2, μ=0, ρ=0.0))  # BP
end

@testset "Good code" begin
    test_jet(ContextualSBM(; N=10^3, P=10^3, d=5, λ=2, μ=2, ρ=0.0))
    test_allocations(ContextualSBM(; N=10^3, P=10^3, d=5, λ=2, μ=2, ρ=0.0))
end
