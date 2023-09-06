## Model

"""
$(TYPEDEF)

A generative model for graphs with node features, which combines a Generalized Linear Model with a Stochastic Block Model.

Reference: <https://arxiv.org/abs/2303.09995>

# Fields

$(TYPEDFIELDS)
"""
struct GLMSBM{R<:Real,D} <: AbstractSBM
    "graph size"
    N::Int
    "feature dimension"
    M::Int
    "average degree"
    c::R
    "SNR of the communities"
    λ::R
    "fraction of nodes revealed"
    ρ::R
    "prior on the weights"
    Pʷ::D

    function GLMSBM(; N::Integer, M::Integer, c::R1, λ::R2, ρ::R3, Pʷ::D) where {R1,R2,R3,D}
        R = promote_type(R1, R2, R3)
        return new{R,D}(N, M, c, λ, ρ, Pʷ)
    end
end

Base.length(glmsbm::GLMSBM) = glmsbm.N
nb_features(glmsbm::GLMSBM) = glmsbm.M
average_degree(glmsbm::GLMSBM) = glmsbm.c
communities_snr(glmsbm::GLMSBM) = glmsbm.λ
fraction_observed(glmsbm::GLMSBM) = glmsbm.ρ

## Latents

"""
$(TYPEDEF)

The hidden variables generated by sampling from a [`GLMSBM`](@ref).

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct LatentsGLMSBM{R<:Real}
    "community assignments, length `N`"
    s::Vector{Int}
    "feature weights, length `M`"
    w::Vector{R}
end

## Observations

"""
$(TYPEDEF)

The observations generated by sampling from a [`GLMSBM`](@ref).

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct ObservationsGLMSBM{R<:Real,G<:AbstractGraph{Int}}
    "undirected unweighted graph with `N` nodes (~ adjacency matrix `A`)"
    g::G
    "revealed communities `±1` for a fraction `ρ` of nodes and `missing` for the rest, ngth `N`"
    Ξ::Vector{Union{Int,Missing}}
    "feature matrix, size `(N, M)`"
    F::Matrix{R}
end

## Simulation

"""
    rand(rng, glmsbm)

Sample from a [`GLMSBM`](@ref) and return a named tuple `(; latents, observations)`.
"""
function Base.rand(rng::AbstractRNG, glmsbm::GLMSBM)
    (; N, M, ρ, Pʷ) = glmsbm

    F = randn(rng, N, M) ./ sqrt(M)

    w = [rand(Pʷ) for l in 1:M]
    s = round.(Int, sign.(F * w))

    g = sample_graph(rng, glmsbm, s)
    Ξ = sample_mask(rng, glmsbm, s)

    latents = LatentsGLMSBM(; w, s)
    observations = ObservationsGLMSBM(; g, Ξ, F)
    return (; latents, observations)
end

## Weight priors

struct RademacherWeightPrior{R} end
struct GaussianWeightPrior{R} end

RademacherWeightPrior() = RademacherWeightPrior{Float64}()
GaussianWeightPrior() = GaussianWeightPrior{Float64}()

function Base.rand(rng::AbstractRNG, ::RademacherWeightPrior{R}) where {R}
    return rand(rng, (-one(R), +one(R)))
end

Base.rand(rng::AbstractRNG, ::GaussianWeightPrior{R}) where {R} = randn(rng, R)

fₐ(::RademacherWeightPrior, Λ, Γ) = tanh(Γ)
fᵥ(::RademacherWeightPrior, Λ, Γ) = inv(abs2(cosh(Γ)))

fₐ(::GaussianWeightPrior, Λ, Γ) = Γ / (Λ + 1)
fᵥ(::GaussianWeightPrior, Λ, Γ) = 1 / (Λ + 1)

function gₒ(ω, χ₊, V)
    Znn = (1 + (2χ₊ - 1) * erf(ω / sqrt(2V))) / 2
    return inv(sqrt(2π * V)) * (2χ₊ - 1) * exp(-ω^2 / (2V)) / Znn
end