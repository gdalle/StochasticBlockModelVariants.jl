## Model

"""
    ContextualSBM

# Fields

- `d`: average degree
- `λ`: SNR of the graph
- `μ`: SNR of the features
- `N`: graph size
- `P`: feature dimension
"""
@kwdef struct ContextualSBM{R<:Real}
    d::R
    λ::R
    μ::R
    N::Int
    P::Int
end

function affinities(csbm::ContextualSBM)
    (; d, λ) = csbm
    cᵢ = d + λ * sqrt(d)
    cₒ = d - λ * sqrt(d)
    return (; cᵢ, cₒ)
end

## Latents

"""
    ContextualSBMLatents

# Fields

- `u::Vector`: community assignments, length `N`
- `v::Vector`: feature centroids, length `P`
"""
@kwdef struct ContextualSBMLatents{R<:Real}
    u::Vector{Int}
    v::Vector{R}
end

## Observations

"""
    ContextualSBMObservations

# Fields

- `A::AbstractMatrix`: adjacency matrix, size `(N, N)`
- `B::Matrix`: feature matrix, size `(P, N)`
- `G::AbstractGraph`: unweighted graph generated from `A`
"""
@kwdef struct ContextualSBMObservations{R<:Real,AM<:AbstractMatrix{Bool},AG<:AbstractGraph}
    A::AM
    B::Matrix{R}
    G::AG
end

## Simulation

"""
    rand(rng, csbm)

Simulate a contextual SBM, return a named tuple `(; latents, observations)`.
"""
function Base.rand(rng::AbstractRNG, csbm::ContextualSBM)
    (; μ, N, P) = csbm
    (; cᵢ, cₒ) = affinities(csbm)

    u = rand(rng, (-1, +1), N)
    v = randn(rng, P)

    Is, Js = Int[], Int[]
    for i in 1:N, j in 1:i
        r = rand(rng)
        if (u[i] == u[j] && r < cᵢ / N) || (u[i] != u[j] && r < cₒ / N)
            push!(Is, i)
            push!(Js, j)
        end
    end
    Vs = fill(true, length(Is))
    A = Symmetric(sparse(Is, Js, Vs, N, N))
    G = SimpleWeightedGraph(A)

    Z = randn(rng, P, N)
    B = similar(Z)
    for α in 1:P, i in 1:N
        B[α, i] = sqrt(μ / N) * v[α] * u[i] * Z[α, i]
    end

    latents = ContextualSBMLatents(; u, v)
    observations = ContextualSBMObservations(; A, B, G)
    return (; latents, observations)
end

Base.rand(csbm::ContextualSBM) = Base.rand(default_rng(), csbm)

## Message-passing

"""
    ContextualSBMMessages

# Fields

- `û::Vector`: posterior mean of `u`, length `N`
- `v̂::Vector`: posterior mean of `v`, length `P`
- `σᵤ::Vector`: posterior variance of `u`, length `N`
- `σᵥ::Ref`: posterior variance of `v`
- `û_no_feat::Vector`: posterior mean of `u` if there were no features, length `N` (aka `Bᵥ`)
- `v̂_no_graph::Vector`: posterior mean of `v` if there were no graph, length `P` (aka `Bᵤ`)
- `σᵥ_no_feat::Ref`: posterior variance of `v` if there were no features (aka `Aᵤ`)
- `h::Vector`: global external field, length `2`
- `h̃::Matrix`: individual external field, size `(2, P)`
- `χ₊_msg::AbstractMatrix`: message about the marginal distribution of `u`, size `(N, N)`
- `χ₊::Vector`: marginal distribution of `u`, length `N`
"""
@kwdef struct ContextualSBMMessages{R<:Real,M<:AbstractMatrix{R}}
    û::Vector{R}
    v̂::Vector{R}
    σᵤ::Vector{R}
    σᵥ::Base.RefValue{R}
    û_no_feat::Vector{R}
    v̂_no_graph::Vector{R}
    σᵥ_no_feat::Base.RefValue{R}
    h::Vector{R}
    h̃::Matrix{R}
    χ₊_msg::M
    χ₊::Vector{R}
end

function amp_bp_update!(
    next_msg::ContextualSBMMessages,
    msg::ContextualSBMMessages,
    observations::ContextualSBMObservations,
    csbm::ContextualSBM,
)
    (; d, λ, μ, N, P) = csbm
    (; A, B, G) = observations
    (; û, v̂, σᵤ, σᵥ) = msg
    (; û_no_feat, v̂_no_graph, σᵥ_no_graph) = msg
    (; h, h̃, χ₊_msg, χ₊) = msg
    (; next_û, next_v̂, next_σᵤ, next_σᵥ) = next_msg
    (; next_û_no_feat, next_v̂_no_graph, next_σᵥ_no_graph) = next_msg
    (; next_h, next_h̃, next_χ₊_msg, next_χ₊) = next_msg

    for i in 1:N
        σᵤ[i] = 1 - abs2(û[i])
    end

    # AMP estimation of v
    next_σᵥ_no_graph = (μ / N) * sum(abs2, û)
    @views for α in 1:P
        next_v̂_no_graph[α] = sqrt(μ / N) * dot(B[α, :], û) - (μ / N) * dot(σᵤ, v̂)
    end
    for α in 1:P
        next_v̂[α] = next_v̂_no_graph[α] / (1 + σᵥ_no_graph[])
    end
    next_σᵥ[] = 1 / (1 + σᵥ_no_graph[])

    # BP estimation of u
    @views for i in 1:N
        next_û_no_feat[i] =
            sqrt(μ / N) * dot(B[:, i], next_v̂) - (μ / (N / P)) * σᵥ[] * û[i]
    end

    # Estimation of the field h

    return nothing
end
