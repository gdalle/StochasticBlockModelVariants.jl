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

function effective_snr(csbm::ContextualSBM)
    (; λ, μ, N, P) = csbm
    return λ^2 + μ^2 / (N / P)
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

    r = rand(rng, N, N)
    Is, Js = Int[], Int[]
    for i in 1:N, j in 1:i
        if ((u[i] == u[j]) && (r[i, j] < cᵢ / N)) || ((u[i] != u[j]) && (r[i, j] < cₒ / N))
            push!(Is, i)
            push!(Js, j)
        end
    end
    Vs = fill(true, length(Is))
    A = Symmetric(sparse(Is, Js, Vs, N, N))
    G = SimpleWeightedGraph(A)

    Z = randn(rng, P, N)
    B = similar(Z)
    for i in 1:N, α in 1:P
        B[α, i] = sqrt(μ / N) * v[α] * u[i] + Z[α, i]
    end

    latents = ContextualSBMLatents(; u, v)
    observations = ContextualSBMObservations(; A, B, G)
    return (; latents, observations)
end

## Message-passing

"""
    AMPStorage

# Fields

- `û::Vector`: posterior mean of `u`, length `N`
- `v̂::Vector`: posterior mean of `v`, length `P`
- `χ₊e::AbstractMatrix`: message about the marginal distribution of `u`, size `(N, N)`
"""
@kwdef struct AMPStorage{R<:Real,M<:AbstractMatrix{R}}
    û::Vector{R}
    v̂::Vector{R}
    χ₊e::M
end

function Base.:(==)(storage1::AMPStorage, storage2::AMPStorage)
    return (
        storage1.û == storage2.û &&
        storage1.v̂ == storage2.v̂ &&
        storage1.χ₊e == storage2.χ₊e
    )
end

function Base.copy(storage::AMPStorage)
    return AMPStorage(; û=copy(storage.û), v̂=copy(storage.v̂), χ₊e=copy(storage.χ₊e))
end

function Base.copy!(storage_dest::AMPStorage, storage_source::AMPStorage)
    storage_dest.û .= storage_source.û
    storage_dest.v̂ .= storage_source.v̂
    storage_dest.χ₊e .= storage_source.χ₊e
    return storage_dest
end

"""
    AMPTempStorage

# Fields

- `û_no_feat::Vector`: posterior mean of `u` if there were no features, length `N` (aka `Bᵥ`)
- `v̂_no_graph::Vector`: posterior mean of `v` if there were no graph, length `P` (aka `Bᵤ`)
- `h̃₊::Vector`: individual external field for `u=1`, length `N`
- `h̃₋::Vector`: individual external field for `u=-1`, length `N`
- `χ₊::Vector`: marginal probability of `u=1`, length `N`
"""
@kwdef struct AMPTempStorage{R<:Real}
    û_no_feat::Vector{R}
    v̂_no_graph::Vector{R}
    h̃₊::Vector{R}
    h̃₋::Vector{R}
    χ₊::Vector{R}
end

function init_amp(
    rng::AbstractRNG;
    observations::ContextualSBMObservations,
    csbm::ContextualSBM{R},
    init_std::Real,
) where {R}
    (; N, P) = csbm
    (; G) = observations

    û = init_std .* randn(rng, R, N)
    v̂ = init_std .* randn(rng, R, P)
    Is, Js = Int[], Int[]
    for i in 1:N, j in neighbors(G, i)
        push!(Is, i)
        push!(Js, j)
    end
    Vs = (one(R) / 2) .+ init_std .* randn(rng, R, length(Is))
    χ₊e = sparse(Is, Js, Vs, N, N)
    storage = AMPStorage(; û, v̂, χ₊e)
    next_storage = copy(storage)

    û_no_feat = zeros(R, N)
    v̂_no_graph = zeros(R, P)
    h̃₊ = zeros(R, N)
    h̃₋ = zeros(R, N)
    χ₊ = zeros(R, N)
    temp_storage = AMPTempStorage(; û_no_feat, v̂_no_graph, h̃₊, h̃₋, χ₊)

    return (; storage, next_storage, temp_storage)
end

function update_amp!(
    next_storage::AMPStorage,
    temp_storage::AMPTempStorage;
    storage::AMPStorage,
    observations::ContextualSBMObservations,
    csbm::ContextualSBM,
)
    (; d, λ, μ, N, P) = csbm
    (; B, G) = observations
    (; cᵢ, cₒ) = affinities(csbm)

    ûᵗ, v̂ᵗ, χ₊eᵗ = storage.û, storage.v̂, storage.χ₊e
    ûᵗ⁺¹, v̂ᵗ⁺¹, χ₊eᵗ⁺¹ = next_storage.û, next_storage.v̂, next_storage.χ₊e
    (; û_no_feat, v̂_no_graph, h̃₊, h̃₋, χ₊) = temp_storage

    # AMP estimation of v
    σᵥ_no_graph = (μ / N) * sum(ûᵗ[i]^2 for i in 1:N)
    for α in 1:P
        v̂_no_graph[α] = (
            sqrt(μ / N) * sum(B[α, i] * ûᵗ[i] for i in 1:N) -
            (μ / N) * sum((1 - ûᵗ[i]^2) * v̂ᵗ[α] for i in 1:N)
        )
        v̂ᵗ⁺¹[α] = v̂_no_graph[α] / (1 + σᵥ_no_graph)
    end
    σᵥ = 1 / (1 + σᵥ_no_graph)

    # BP estimation of u
    for i in 1:N
        û_no_feat[i] = (
            sqrt(μ / N) * sum(B[α, i] * v̂ᵗ⁺¹[α] for α in 1:P) - (μ / (N / P)) * σᵥ * ûᵗ[i]
        )
    end

    # Estimation of the field h
    h₊ = (1 / N) * sum(cᵢ * (1 + ûᵗ[i]) / 2 + cₒ * (1 - ûᵗ[i]) / 2 for i in 1:N)
    h₋ = (1 / N) * sum(cₒ * (1 + ûᵗ[i]) / 2 + cᵢ * (1 - ûᵗ[i]) / 2 for i in 1:N)
    for i in 1:N
        h̃₊[i] = -h₊ + û_no_feat[i]
        h̃₋[i] = -h₋ - û_no_feat[i]
    end

    # BP update of the messages
    for i in 1:N, j in neighbors(G, i)
        s_ij = h̃₊[i] - h̃₋[i]
        for k in neighbors(G, i)
            k != j || continue
            num = (cₒ + 2λ * sqrt(d) * χ₊eᵗ[k, i])
            den = (cᵢ - 2λ * sqrt(d) * χ₊eᵗ[k, i])
            s_ij += log(num / den)
        end
        χ₊eᵗ⁺¹[i, j] = sigmoid(s_ij)
    end

    # BP update of the marginals
    for i in 1:N
        s_i = h̃₊[i] - h̃₋[i]
        for k in neighbors(G, i)
            num = (cₒ + 2λ * sqrt(d) * χ₊eᵗ[k, i])
            den = (cᵢ - 2λ * sqrt(d) * χ₊eᵗ[k, i])
            s_i += log(num / den)
        end
        χ₊[i] = sigmoid(s_i)
    end

    # BP estimation of u
    for i in 1:N
        ûᵗ⁺¹[i] = 2χ₊[i] - 1
    end

    return nothing
end

function run_amp(
    rng::AbstractRNG;
    observations::ContextualSBMObservations,
    csbm::ContextualSBM,
    init_std::Real,
    iterations::Integer,
)
    (; storage, next_storage, temp_storage) = init_amp(rng; observations, csbm, init_std)
    storage_history = [copy(storage)]
    @showprogress "AMP-BP" for iter in 1:iterations
        update_amp!(next_storage, temp_storage; storage, observations, csbm)
        copy!(storage, next_storage)
        push!(storage_history, copy(storage))
    end
    return storage_history
end

function evaluate_amp(; storage::AMPStorage, latents::ContextualSBMLatents)
    u = latents.u
    N = length(u)
    û = 2 .* Int.(storage.û .> 0) .- 1
    q̂ᵤ = (1 / N) * max(count_equalities(û, u), count_equalities(û, -u))
    qᵤ = 2 * (q̂ᵤ - 0.5)
    return qᵤ
end
