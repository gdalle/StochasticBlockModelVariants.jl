## Model

"""
    GLMSBM

A generative model for graphs with node features, which combines a Generalized Linear Model with a Stochastic Block Model.

Reference: <https://arxiv.org/abs/2303.09995>

# Fields

- `N`: graph size
- `M`: feature dimension
- `c`: average degree
- `λ`: SNR of the communities
- `ρ`: fraction of nodes revealed
- `Pʷ`: prior on the weights
"""
struct GLMSBM{R<:Real,D} <: AbstractSBM
    N::Int
    M::Int
    c::R
    λ::R
    ρ::R
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
    LatentsGLMSBM

The hidden variables generated by sampling from a [`GLMSBM`](@ref).

# Fields

- `s::Vector`: community assignments, length `N`
- `w::Vector`: feature weights, length `M`
"""
@kwdef struct LatentsGLMSBM{R<:Real}
    s::Vector{Int}
    w::Vector{R}
end

## Observations

"""
    ObservationsGLMSBM

The observations generated by sampling from a [`GLMSBM`](@ref).

# Fields
- `g::AbstractGraph`: undirected unweighted graph with `N` nodes (~ adjacency matrix `A`)
- `Ξ::Vector`: revealed communities `±1` for a fraction `ρ` of nodes and `missing` for the rest, length `N`
- `F::Matrix`: feature matrix, size `(M, N)`
"""
@kwdef struct ObservationsGLMSBM{R<:Real,G<:AbstractGraph{Int}}
    g::G
    Ξ::Vector{Union{Int,Missing}}
    F::Matrix{R}
end

## Simulation

"""
    rand(rng, glmsbm)

Sample from a [`GLMSBM`](@ref) and return a named tuple `(; latents, observations)`.
"""
function Base.rand(rng::AbstractRNG, glmsbm::GLMSBM)
    (; N, M, ρ, Pʷ) = glmsbm

    F = randn(rng, M, N) ./ sqrt(M)

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

function Base.rand(rng::AbstractRNG, ::RademacherWeightPrior{R}) where {R}
    return rand(rng, (-one(R), +one(R)))
end

Base.rand(rng::AbstractRNG, ::GaussianWeightPrior{R}) where {R} = randn(rng, R)

fₐ(::RademacherWeightPrior{R}, Λ, Γ) where {R} = tanh(Γ)
fᵥ(::RademacherWeightPrior{R}, Λ, Γ) where {R} = inv(abs2(cosh(Γ)))

fₐ(::GaussianWeightPrior{R}, Λ, Γ) where {R} = Γ / (Λ + 1)
fᵥ(::GaussianWeightPrior{R}, Λ, Γ) where {R} = 1 / (Λ + 1)
