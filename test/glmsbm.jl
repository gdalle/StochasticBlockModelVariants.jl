using JET
using Random: default_rng
using Statistics
using StochasticBlockModelVariants
using StochasticBlockModelVariants: GaussianWeightPrior
using Test

rng = default_rng()

glmsbm = GLMSBM(; N=10^4, M=10^4, c=5, λ=2, ρ=0.0, Pʷ=GaussianWeightPrior{Float64}())

(; latents, observations) = rand(rng, glmsbm)

run_amp(rng, observations, glmsbm; max_iterations=2)
