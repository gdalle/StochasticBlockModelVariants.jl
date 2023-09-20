"""
$(TYPEDEF)

Abstract supertype for Stochastic Block Models with additional node features.

# Subtypes

- [`CSBM`](@ref)
- [`GLMSBM`](@ref)
"""
abstract type AbstractSBM end

"""
$(TYPEDEF)

Abstract supertype for SBM communities and latent weights / centroids.

# Subtypes

- [`LatentsCSBM`](@ref)
- [`LatentsGLMSBM`](@ref)
"""
abstract type AbstractLatents end

"""
$(TYPEDEF)

Abstract supertype for SBM graph and node feature observations.

# Subtypes

- [`ObservationsCSBM`](@ref)
- [`ObservationsGLMSBM`](@ref)
"""
abstract type AbstractObservations end

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

Return the number of features per node, called either `P` or `M` depending on the paper.
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
function run_amp(
    rng::AbstractRNG,
    observations,
    sbm::AbstractSBM;
    init_std=1e-3,
    max_iterations=200,
    convergence_threshold=1e-3,
    recent_past=5,
    damping=0.0,
    show_progress=false,
)
    N = length(sbm)
    P = nb_features(sbm)
    (; marginals, next_marginals) = init_amp(rng, observations, sbm; init_std)

    R = eltype(marginals)
    discrete_history = Matrix{R}(undef, N, max_iterations)
    continuous_history = Matrix{R}(undef, P, max_iterations)
    converged = false
    prog = Progress(max_iterations; desc="AMP-BP", enabled=show_progress)

    for t in 1:max_iterations
        update_amp!(next_marginals, marginals, observations, sbm)
        copy_damp!(marginals, next_marginals; damping=(t == 1 ? zero(damping) : damping))

        discrete_history[:, t] .= discrete_estimates(marginals)
        continuous_history[:, t] .= continuous_estimates(marginals)

        if t <= recent_past
            discrete_recent_std = typemax(R)
            continuous_recent_std = typemax(R)
        else
            discrete_recent_std = mean(
                std(view(discrete_history, :, (t - recent_past):t); dims=2)
            )
            continuous_recent_std = mean(
                std(view(continuous_history, :, (t - recent_past):t); dims=2)
            )
        end
        converged = (
            discrete_recent_std < convergence_threshold &&
            continuous_recent_std < convergence_threshold
        )
        if converged
            discrete_history = discrete_history[:, 1:t]
            continuous_history = continuous_history[:, 1:t]
            break
        else
            showvalues = [
                (:discrete_recent_std, discrete_recent_std),
                (:continuous_recent_std, continuous_recent_std),
                (:convergence_threshold, convergence_threshold),
            ]
            next!(prog; showvalues)
        end
    end

    return (; discrete_history, continuous_history, converged)
end

"""
    evaluate_amp(rng, sbm; kwargs...)

Sample observations from `sbm` and [`run_amp`](@ref) on them before computing discrete and continuous overlaps with the true latent variables.

Return a tuple `(q_dis, q_cont, converged)` containing the final overlaps as well as a boolean convergence indicator.
"""
function evaluate_amp(
    rng::AbstractRNG,
    sbm::AbstractSBM,
    latents::AbstractLatents,
    observations::AbstractObservations;
    kwargs...,
)
    (; discrete_history, continuous_history, converged) = run_amp(
        rng, observations, sbm; kwargs...
    )
    q_discrete = discrete_overlap(discrete_values(latents), discrete_history[:, end])
    q_continuous = continuous_overlap(
        continuous_values(latents), continuous_history[:, end]
    )
    return (; q_discrete, q_continuous, converged)
end

function evaluate_amp(rng::AbstractRNG, sbm::AbstractSBM; kwargs...)
    (; latents, observations) = rand(rng, sbm)
    return evaluate_amp(rng, sbm, latents, observations; kwargs...)
end

function semisupervised_loss_amp(
    rng::AbstractRNG, sbm::AbstractSBM, observations::AbstractObservations; kwargs...
)
    Ξ_backup = copy(observations.Ξ)
    observations.Ξ .= missing
    (; discrete_history, continuous_history, converged) = run_amp(
        rng, observations, sbm; kwargs...
    )
    observations.Ξ .= Ξ_backup
    discrete_est = discrete_history[:, end]
    L = sum(abs2, skipmissing(discrete_est - observations.Ξ))
    return L
end
