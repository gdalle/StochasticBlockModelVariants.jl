using Base.Threads
using CairoMakie
using LinearAlgebra
using Random: default_rng, rand!
using Statistics
using StochasticBlockModelVariants
using ProgressMeter

BLAS.set_num_threads(1)

function compute_fig1_glmsbm(; N, c, ρ, Pʷ, α_values, λ_values, trials)
    rng = default_rng()
    I, J, K = length(α_values), length(λ_values), trials
    qs_values = zeros(I, J, K)
    qw_values = zeros(I, J, K)
    converged_values = zeros(Bool, I, J, K)
    prog = Progress(I * J * K; desc="CSBM - Fig 1")
    for i in 1:I
        for j in 1:J
            α, λ = α_values[i], λ_values[j]
            M = ceil(Int, N / α)
            glmsbm = GLMSBM(; N, M, c, λ, ρ, Pʷ)
            F = rand(rng, glmsbm).observations.F
            for k in 1:K  # don't parallelize
                (; latents, observations) = rand!(rng, F, glmsbm)
                (qs, qw, converged) = evaluate_amp(rng, glmsbm, latents, observations)
                qs_values[i, j, k] = qs
                qw_values[i, j, k] = qw
                converged_values[i, j, k] = converged
                next!(prog)
            end
        end
    end
    return (; qs_values, qw_values)
end

function plot_fig1_glmsbm(res; α_values, λ_values)
    qs_values, qw_values = res
    f = Figure()
    ax1 = Axis(f[1, 1]; title="Communities", xlabel="λ", ylabel="q_s", limits=(0, 2, 0, 1))
    ax2 = Axis(f[1, 2]; title="Weights", xlabel="λ", ylabel="q_w", limits=(0, 2, 0, 1))
    for (i, α) in enumerate(α_values)
        qs_means = dropdims(mean(qs_values[i, :, :]; dims=2); dims=2)
        qs_stds = dropdims(std(qs_values[i, :, :]; dims=2); dims=2)
        lines!(ax1, λ_values, qs_means; label="α=$α")
        errorbars!(ax1, λ_values, qs_means, qs_stds; label=nothing)

        qw_means = dropdims(mean(qw_values[i, :, :]; dims=2); dims=2)
        qw_stds = dropdims(std(qw_values[i, :, :]; dims=2); dims=2)
        lines!(ax2, λ_values, qw_means; label="α=$α")
        errorbars!(ax2, λ_values, qw_means, qw_stds; label=nothing)
    end
    return f
end

N = 10^4
c = 5
ρ = 0.0
Pʷ = GaussianWeightPrior()
α_values = reverse([0.3, 1, 3, 10, 30])
α_values = reverse([0.3, 1])
λ_values = 0:0.1:2
trials = 1
res1 = compute_fig1_glmsbm(; N, c, ρ, Pʷ, α_values, λ_values, trials)  # kills Julia
plot_fig1_glmsbm(res1; α_values, λ_values)
