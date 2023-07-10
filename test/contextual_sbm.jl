using StochasticBlockModelVariants

csbm = ContextualSBM(; d=3, λ=1, μ=2.0, N=10, P=20)

(; latents, obs) = rand(csbm)
