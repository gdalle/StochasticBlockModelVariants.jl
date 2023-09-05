using JET
using Random: default_rng
using Statistics
using StochasticBlockModelVariants
using Test

function test_recovery(csbm::CSBM; test_u=true, test_v=true)
    rng = default_rng()
    @assert effective_snr(csbm) > 1
    (; latents, observations) = rand(rng, csbm)
    (; u, v) = latents
    (; û_history, v̂_history, converged) = run_amp(rng, observations, csbm)
    @test converged
    first_q = overlaps(; u, v, û=û_history[:, begin], v̂=v̂_history[:, begin])
    last_q = overlaps(; u, v, û=û_history[:, end], v̂=v̂_history[:, end])
    if test_u
        @test last_q.qᵤ >= first_q.qᵤ
        @test last_q.qᵤ > 0.5
    end
    if test_v
        @test last_q.qᵥ >= first_q.qᵥ
        @test last_q.qᵥ > 0.5
    end
    return nothing
end

function test_jet(csbm::CSBM)
    rng = default_rng()
    (; observations) = rand(rng, csbm)
    @test_opt target_modules = (StochasticBlockModelVariants,) run_amp(
        rng, observations, csbm; max_iterations=2
    )
    @test_call target_modules = (StochasticBlockModelVariants,) run_amp(
        rng, observations, csbm; max_iterations=2
    )
    return nothing
end

function test_allocations(csbm::CSBM)
    rng = default_rng()
    (; observations) = rand(rng, csbm)
    (; marginals, next_marginals) = init_amp(rng, observations, csbm; init_std=1e-3)
    alloc = @allocated update_amp!(next_marginals, marginals, observations, csbm)
    @test alloc == 0
    return nothing
end

@testset "Correct code" begin
    test_recovery(CSBM(; N=10^3, P=10^3, d=5, λ=2, μ=2, ρ=0.0))  # AMP-BP
    test_recovery(CSBM(; N=10^3, P=10^3, d=5, λ=0, μ=2, ρ=0.0))  # AMP
    test_recovery(CSBM(; N=10^3, P=10^3, d=5, λ=2, μ=0, ρ=0.0); test_v=false)  # BP
    test_recovery(CSBM(; N=10^2, P=10^2, d=5, λ=2, μ=2, ρ=0.5))  # semi-supervised
end

@testset "Good code" begin
    test_jet(CSBM(; N=10^3, P=10^3, d=5, λ=2, μ=2, ρ=0.0))
    test_allocations(CSBM(; N=10^3, P=10^3, d=5, λ=2, μ=2, ρ=0.0))
end
