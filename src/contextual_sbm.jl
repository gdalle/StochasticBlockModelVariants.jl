struct ContextualSBM{R<:Real}
    d::R
    λ::R
    μ::R
    N::Int
    P::Int

    function ContextualSBM(; d::R1, λ::R2, μ::R3, N, P) where {R1,R2,R3}
        R = promote_type(R1, R2, R3)
        return new{R}(d, λ, μ, N, P)
    end
end

@kwdef struct ContextualSBMLatents{R<:Real}
    u::Vector{Int}  # (N,)
    v::Vector{R}  # (P,)
end

@kwdef struct ContextualSBMObservations{R<:Real}
    A::Symmetric{Bool,SparseMatrixCSC{Bool,Int}}  # (N, N)
    G::SimpleWeightedGraph{Int,Bool}
    B::Matrix{R}  # (P, N)
end

@kwdef struct ContextualSBMMessages
    # From variables to factors
    χ_node_node
    χ_node_feat
    χ_feat_node
    # From factors to variables
    ψ_node_node
    ψ_node_feat
    ψ_feat_node
end

const CSBM = ContextualSBM
const CSBML = ContextualSBMLatents
const CSBMO = ContextualSBMObservations

function affinities(csbm::CSBM)
    (; d, λ) = csbm
    cᵢ = d + λ * sqrt(d)
    cₒ = d - λ * sqrt(d)
    return (; cᵢ, cₒ)
end

nb_nodes(csbm::CSBM) = csbm.N
nb_nodes(latents::CSBML) = length(latents.u)
nb_nodes(obs::CSBMO) = size(obs.A, 1)

nb_features(csbm::CSBM) = csbm.P
nb_features(latents::CSBML) = length(latents.v)
nb_features(obs::CSBMO) = size(obs.B, 1)

function Base.rand(rng::AbstractRNG, csbm::CSBM)
    N, P = nb_nodes(csbm), nb_features(csbm)
    μ = csbm.μ
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

    latents = CSBML(; u, v)
    obs = CSBMO(; A, G, B)
    return (; latents, obs)
end

Base.rand(csbm::CSBM) = Base.rand(default_rng(), csbm)
