using JET
using Random: default_rng
using Statistics
using StochasticBlockModelVariants
using Test

function test_allocations(sbm::AbstractSBM)
    rng = default_rng()
    (; observations) = rand(rng, sbm)
    (; marginals, next_marginals) = init_amp(rng, observations, sbm; init_std=1e-3)
    alloc = @allocated update_amp!(next_marginals, marginals, observations, sbm)
    @test alloc == 0
    return nothing
end

test_allocations(CSBM(; N=10^3, P=10^3, d=5, λ=2, μ=2, ρ=0.0))
@test_skip test_allocations(
    GLMSBM(; N=10^3, M=10^3, c=5, λ=2, ρ=0.0, Pʷ=GaussianWeightPrior())
)
