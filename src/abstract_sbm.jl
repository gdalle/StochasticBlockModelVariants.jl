"""
$(TYPEDEF)

Abstract supertype for Stochastic Block Models with additional node features.

# Subtypes

- [`CSBM`](@ref)
- [`GLMSBM`](@ref)
"""
abstract type AbstractSBM end

"""
    length(sbm::AbstractSBM)

Return the number of nodes `N` in the graph.
"""
Base.length

"""
    rand(rng, sbm::AbstractSBM)

Sample observations consisting of a graph and continuous features, return a named tuple `(; latents, observations)`.
"""
Base.rand

"""
    nb_features(sbm::AbstractSBM)

Return the number of nodes `N` in the graph.
"""
function nb_features end

"""
    average_degree(sbm::AbstractSBM)

Return the average degre `d` of a node in the graph.
"""
function average_degree end

"""
    communities_snr(sbm::AbstractSBM)

Return the signal-to-noise ratio `λ` of the communities in the graph.
"""
function communities_snr end

"""
    affinities(sbm::AbstractSBM)

Return a named tuple `(; cᵢ, cₒ)` containing the affinities inside and outside of a community.
"""
function affinities(sbm::AbstractSBM)
    d = average_degree(sbm)
    λ = communities_snr(sbm)
    cᵢ = d + λ * sqrt(d)
    cₒ = d - λ * sqrt(d)
    return (; cᵢ, cₒ)
end

"""
    fraction_observed(sbm::AbstractSBM)

Return the fraction `ρ` of community assignments that are observed.
"""
function fraction_observed end

"""
    sample_graph(rng, sbm::AbstractSBM, communities::Vector{<:Integer})

Sample a graph `g` based on known community assignments.
"""
function sample_graph(rng::AbstractRNG, sbm::AbstractSBM, communities::Vector{<:Integer})
    N = length(sbm)
    (; cᵢ, cₒ) = affinities(sbm)
    Is, Js = Int[], Int[]
    for i in 1:N, j in (i + 1):N
        r = rand(rng)
        if (
            ((communities[i] == communities[j]) && (r < cᵢ / N)) ||
            ((communities[i] != communities[j]) && (r < cₒ / N))
        )
            push!(Is, i)
            push!(Is, j)
            push!(Js, j)
            push!(Js, i)
        end
    end
    Vs = fill(true, length(Is))
    A = sparse(Is, Js, Vs, N, N)
    g = SimpleWeightedGraph(A)
    return g
end

"""
    sample_mask(rng, sbm::AbstractSBM, communities::Vector{<:Integer})

Sample a vector `Ξ` (Xi) whose components are equal to the community assignments with probability `ρ` and equal to `missing` with probability `1-ρ`. 
"""
function sample_mask(rng::AbstractRNG, sbm::AbstractSBM, communities::Vector{<:Integer})
    N = length(sbm)
    ρ = fraction_observed(sbm)
    Ξ = Vector{Union{Missing,Int}}(undef, N)
    Ξ .= missing
    for i in 1:N
        if rand(rng) < ρ
            Ξ[i] = communities[i]
        end
    end
    return Ξ
end

"""
    run_amp(
        rng,
        observations,
        sbm::AbstractSBM;
        init_std,
        max_iterations,
        convergence_threshold,
        recent_past,
        damping,
        show_progress
    )

Run AMP-BP for `sbm` based on a set of `observations` to estimate the discrete communities and the continuous weights / centroids.

Return a tuple `(discrete_history, continuous_history, converged)` containing the history of latent variable marginals as well as a boolean convergence indicator.

# Keyword arguments

- `init_std`: Noise level used to initialize some messages
- `max_iterations`: Maximum number of iterations allowed
- `convergence_threshold`: Lower bound that the standard deviation of recent latent estimates has to reach for the algorithm to have converged
- `recent_past`: Number of instants taken into account when computing the recent standard deviation
- `damping`: Fraction in `[0, 1]` telling us how much the previous message is copied into the next message
- `show_progress`: Whether to display a progress bar with convergence statistics
"""
function run_amp end

"""
    evaluate_amp(rng, sbm; kwargs...)

Sample observations from `sbm` and [`run_amp`](@ref) on them before computing discrete and continuous overlaps with the true latent variables.

Return a tuple `(q_dis, q_cont, converged)` containing the final overlaps as well as a boolean convergence indicator.
"""
function evaluate_amp end
