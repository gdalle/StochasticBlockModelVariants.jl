using JET
using Random: default_rng
using Statistics
using StochasticBlockModelVariants
using Test

function test_jet(sbm::AbstractSBM)
    rng = default_rng()
    (; observations) = rand(rng, sbm)
    @test_opt target_modules = (StochasticBlockModelVariants,) run_amp(
        rng, observations, sbm; max_iterations=2
    )
    @test_call target_modules = (StochasticBlockModelVariants,) run_amp(
        rng, observations, sbm; max_iterations=2
    )
    return nothing
end

test_jet(CSBM(; N=10^3, P=10^3, d=5, λ=2, μ=2, ρ=0.0))
test_jet(GLMSBM(; N=10^3, M=10^3, c=5, λ=2, ρ=0.0, Pʷ=GaussianWeightPrior()))
