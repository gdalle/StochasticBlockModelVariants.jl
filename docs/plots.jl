using Base.Threads
using CairoMakie
using LinearAlgebra
using Random: default_rng
using Statistics
using StochasticBlockModelVariants
using ProgressMeter

BLAS.set_num_threads(1)

function compute_fig1(; N=3 * 10^3, P=N ÷ 10, d=5, μ=2, ρ=0.1, λ_values=0:0.1:2, trials=10)
    rng = default_rng()
    qᵤ_values = zeros(trials, length(λ_values))
    qᵥ_values = zeros(trials, length(λ_values))
    prog = Progress(trials * length(λ_values); desc="Fig 1 - trials")
    @threads for i in 1:trials
        for j in eachindex(λ_values)
            λ = λ_values[j]
            csbm = CSBM(; N, P, d, μ, λ, ρ)
            (; qᵤ, qᵥ) = evaluate_amp(rng; csbm)
            qᵤ_values[i, j] = qᵤ
            qᵥ_values[i, j] = qᵥ
            next!(prog)
        end
    end
    return (; λ_values, qᵤ_values, qᵥ_values)
end

function plot_fig1(res)
    f = Figure()
    ax = Axis(f[1, 1]; title="Fig 1", xlabel="λ", ylabel="qᵤ", limits=(0, 2, 0, 1))
    qᵤ_means = dropdims(mean(res.qᵤ_values; dims=1); dims=1)
    qᵤ_stds = dropdims(std(res.qᵤ_values; dims=1); dims=1)
    lines!(ax, λ_values, qᵤ_means)
    scatter!(ax, λ_values, qᵤ_means)
    errorbars!(ax, λ_values, qᵤ_means, qᵤ_stds)
    return f
end

res1 = compute_fig1()
plot_fig1(res1)
