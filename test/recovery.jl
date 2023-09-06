using JET
using Random: default_rng
using Statistics
using StochasticBlockModelVariants
using Test

function test_recovery(sbm::AbstractSBM; test_discrete=true, test_continuous=true)
    rng = default_rng()
    q_dis, q_cont, converged = evaluate_amp(rng, sbm)
    # @test converged  # TODO: toggle
    if test_discrete
        @test q_dis > 0.5
    end
    if test_continuous
        @test q_cont > 0.5
    end
    return nothing
end

test_recovery(CSBM(; N=10^3, P=10^3, d=5, λ=2, μ=2, ρ=0.0))  # AMP-BP
test_recovery(CSBM(; N=10^3, P=10^3, d=5, λ=0, μ=2, ρ=0.0))  # AMP
test_recovery(CSBM(; N=10^3, P=10^3, d=5, λ=2, μ=0, ρ=0.0); test_continuous=false)  # BP
test_recovery(CSBM(; N=10^2, P=10^2, d=5, λ=2, μ=2, ρ=0.5))  # semi-supervised

test_recovery(GLMSBM(; N=10^3, M=10^3, c=5, λ=2, ρ=0.0, Pʷ=GaussianWeightPrior()))  # AMP-BP
