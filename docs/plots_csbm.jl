using Base.Threads
using CairoMakie
using LinearAlgebra
using Random: default_rng
using Statistics
using StochasticBlockModelVariants
using ProgressMeter

BLAS.set_num_threads(1)

function compute_fig1_csbm(; α, d, μ, ρ, N_values, λ_values, trials)
    rng = default_rng()
    I, J, K = length(N_values), length(λ_values), trials
    qu_values = zeros(I, J, K)
    qv_values = zeros(I, J, K)
    converged_values = zeros(Bool, I, J, K)
    prog = Progress(I * J * K; desc="CSBM - Fig 1")
    for i in 1:I
        for j in 1:J
            @threads for k in 1:K
                N, λ = N_values[i], λ_values[j]
                P = ceil(Int, N / α)
                csbm = CSBM(; N, P, d, μ, λ, ρ)
                (qu, qv, converged) = evaluate_amp(rng, csbm)
                qu_values[i, j, k] = qu
                qv_values[i, j, k] = qv
                converged_values[i, j, k] = converged
                next!(prog)
            end
        end
    end
    return (; qu_values, qv_values)
end

function plot_fig1_csbm(res; N_values, λ_values)
    qu_values, qv_values = res
    f = Figure()
    for (i, N) in enumerate(N_values)
        ax = Axis(f[i, 1]; title="N = $N", xlabel="λ", ylabel="q_u", limits=(0, 2, 0, 1))
        qu_means = dropdims(mean(qu_values[i, :, :]; dims=2); dims=2)
        qu_stds = dropdims(std(qu_values[i, :, :]; dims=2); dims=2)
        lines!(ax, λ_values, qu_means)
        errorbars!(ax, λ_values, qu_means, qu_stds)
    end
    return f
end

α = 10
d = 5
μ = 2
ρ = 0.0
N_values = reverse([3 * 10^3, 10^4, 3 * 10^4, 10^5])
N_values = reverse([3 * 10^3, 10^4])
λ_values = 0:0.1:2
trials = 10
res1 = compute_fig1_csbm(; α, d, μ, ρ, N_values, λ_values, trials)
plot_fig1_csbm(res1; N_values, λ_values)
