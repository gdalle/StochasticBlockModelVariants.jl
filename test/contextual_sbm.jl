using StochasticBlockModelVariants

csbm = ContextualSBM(; d=3.0, λ=1.0, μ=2.0, N=10, P=20)

(; latents, observations) = rand(csbm)
