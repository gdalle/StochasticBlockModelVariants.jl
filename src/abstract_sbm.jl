"""
    AbstractSBM

Abstract supertype for Stochastic Block Models with additional node features.
"""
abstract type AbstractSBM end

"""
    average_degree(sbm)

Return the average degre `d` of a node in the graph.
"""
function average_degree end

"""
    communities_snr(sbm)

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
