"""
$(TYPEDEF)

Abstract supertype for Stochastic Block Models with additional node features.
"""
abstract type AbstractSBM end

"""
    length(sbm::AbstractSBM)

Return the number of nodes `N` in the graph.
"""
Base.length

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
    affinities(sbm)

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
    fraction_observed(sbm)

Return the fraction `ρ` of community assignments that are observed.
"""
function fraction_observed end

"""
    sample_graph(rng, sbm, communities)

Sample a graph `g` from an SBM based on known community assignments.
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
    sample_mask(rng, sbm, communities)

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

prior(::Type{R}, u, Ξᵢ) where {R} = ismissing(Ξᵢ) ? one(R) / 2 : R(Ξᵢ == u)

sigmoid(x) = 1 / (1 + exp(-x))

freq_equalities(x, y) = mean(x[i] ≈ y[i] for i in eachindex(x, y))

function discrete_overlap(u, û)
    R = eltype(û)
    û_sign = sign.(û)
    û_sign[abs.(û) .< eps(R)] .= 1
    q̂ᵤ = max(freq_equalities(u, û_sign), freq_equalities(u, -û_sign))
    qᵤ = 2 * (q̂ᵤ - one(R) / 2)
    return qᵤ
end

function continuous_overlap(v, v̂)
    R = eltype(v)
    q̂ᵥ = max(abs(dot(v̂, v)), abs(dot(v̂, -v)))
    qᵥ = q̂ᵥ / (eps(R) + norm(v̂) * norm(v))
    return qᵥ
end
