## Storage

"""
    AMPStorage

# Fields

- `û::Vector`: posterior mean of `u`, length `N`
- `v̂::Vector`: posterior mean of `v`, length `P`
- `χ₊e::Dict`: messages about the marginal distribution of `u`, size `(N, N)`
"""
@kwdef struct AMPStorage{R<:Real}
    û::Vector{R}
    v̂::Vector{R}
    χ₊e::Dict{Tuple{Int,Int},R}
end

function Base.copy(storage::AMPStorage)
    return AMPStorage(; û=copy(storage.û), v̂=copy(storage.v̂), χ₊e=copy(storage.χ₊e))
end

function Base.copy!(storage_dest::AMPStorage, storage_source::AMPStorage)
    storage_dest.û .= storage_source.û
    storage_dest.v̂ .= storage_source.v̂
    copy!(storage_dest.χ₊e, storage_source.χ₊e)
    return storage_dest
end

"""
    AMPTempStorage

# Fields

- `û_no_feat::Vector`: posterior mean of `u` if there were no features, length `N` (aka `Bᵥ`)
- `v̂_no_comm::Vector`: posterior mean of `v` if there were no communities, length `P` (aka `Bᵤ`)
- `h̃₊::Vector`: individual external field for `u=1`, length `N`
- `h̃₋::Vector`: individual external field for `u=-1`, length `N`
- `χ₊::Vector`: marginal probability of `u=1`, length `N`
"""
@kwdef struct AMPTempStorage{R<:Real}
    û_no_feat::Vector{R}
    v̂_no_comm::Vector{R}
    h̃₊::Vector{R}
    h̃₋::Vector{R}
    χ₊::Vector{R}
end

function Base.copy(temp_storage::AMPTempStorage)
    return AMPTempStorage(;
        û_no_feat=copy(temp_storage.û_no_feat),
        v̂_no_comm=copy(temp_storage.v̂_no_comm),
        h̃₊=copy(temp_storage.h̃₊),
        h̃₋=copy(temp_storage.h̃₋),
        χ₊=copy(temp_storage.χ₊),
    )
end

## Message-passing

function init_amp(
    rng::AbstractRNG;
    observations::ContextualSBMObservations{R},
    csbm::ContextualSBM{R},
    init_std,
) where {R}
    (; N, P) = csbm
    (; g, Ξ) = observations

    û = prior₊.(R, Ξ) + init_std .* randn(rng, R, N)
    v̂ = 2 .* prior₊.(R, Ξ) .- one(R) .+ init_std .* randn(rng, R, P)
    χ₊e = Dict{Tuple{Int,Int},R}()
    for i in 1:N, j in neighbors(g, i)
        χ₊e[i, j] = (one(R) / 2) + init_std * randn(rng, R)
    end
    storage = AMPStorage(; û, v̂, χ₊e)
    next_storage = copy(storage)

    û_no_feat = zeros(R, N)
    v̂_no_comm = zeros(R, P)
    h̃₊ = zeros(R, N)
    h̃₋ = zeros(R, N)
    χ₊ = zeros(R, N)
    temp_storage = AMPTempStorage(; û_no_feat, v̂_no_comm, h̃₊, h̃₋, χ₊)

    return (; storage, next_storage, temp_storage)
end

prior₊(::Type{R}, Ξᵢ) where {R} = Ξᵢ == 0 ? one(R) / 2 : R(Ξᵢ == 1)
prior₋(::Type{R}, Ξᵢ) where {R} = Ξᵢ == 0 ? one(R) / 2 : R(Ξᵢ == -1)

function update_amp!(
    next_storage::AMPStorage{R},
    temp_storage::AMPTempStorage{R};
    storage::AMPStorage{R},
    observations::ContextualSBMObservations{R},
    csbm::ContextualSBM{R},
) where {R}
    (; d, λ, μ, N, P) = csbm
    (; g, B, Ξ) = observations
    (; cᵢ, cₒ) = affinities(csbm)

    ûᵗ, v̂ᵗ, χ₊eᵗ = storage.û, storage.v̂, storage.χ₊e
    ûᵗ⁺¹, v̂ᵗ⁺¹, χ₊eᵗ⁺¹ = next_storage.û, next_storage.v̂, next_storage.χ₊e
    (; û_no_feat, v̂_no_comm, h̃₊, h̃₋, χ₊) = temp_storage

    ûₜ_sum = sum(ûᵗ)
    ûₜ_sum2 = sum(abs2, ûᵗ)

    # AMP estimation of v
    σᵥ_no_comm = (μ / N) * ûₜ_sum2
    mul!(v̂_no_comm, B, ûᵗ)
    v̂_no_comm .*= sqrt(μ / N)
    v̂_no_comm .-= (μ / N) .* v̂ᵗ .* (N - ûₜ_sum2)
    v̂ᵗ⁺¹ .= v̂_no_comm ./ (one(R) + σᵥ_no_comm)
    σᵥ = one(R) / (one(R) + σᵥ_no_comm)

    # BP estimation of u
    mul!(û_no_feat, B', v̂ᵗ⁺¹)
    û_no_feat .*= sqrt(μ / N)
    û_no_feat .-= (μ / (N / P)) .* σᵥ .* ûᵗ

    # Estimation of the field h
    h₊ = (one(R) / 2N) * (cᵢ * (N + ûₜ_sum) + cₒ * (N - ûₜ_sum))
    h₋ = (one(R) / 2N) * (cₒ * (N + ûₜ_sum) + cᵢ * (N - ûₜ_sum))
    h̃₊ .= -h₊ .+ log.(prior₊.(R, Ξ)) .+ û_no_feat
    h̃₋ .= -h₋ .+ log.(prior₋.(R, Ξ)) .- û_no_feat

    # BP update of the marginals
    for i in 1:N
        s_i = h̃₊[i] - h̃₋[i]
        for k in neighbors(g, i)
            common = 2λ * sqrt(d) * χ₊eᵗ[k, i]
            s_i += log((cₒ + common) / (cᵢ - common))
        end
        χ₊[i] = s_i
    end

    # BP update of the messages
    for i in 1:N, j in neighbors(g, i)
        common = 2λ * sqrt(d) * χ₊eᵗ[j, i]
        s_ij = log((cₒ + common) / (cᵢ - common))
        χ₊eᵗ⁺¹[i, j] = χ₊[i] - s_ij
    end

    # Sigmoidize probabilities
    χ₊ .= sigmoid.(χ₊)
    for (key, val) in pairs(χ₊eᵗ⁺¹)
        χ₊eᵗ⁺¹[key] = sigmoid(val)
    end

    # BP estimation of u
    ûᵗ⁺¹ .= 2 .* χ₊ .- one(R)

    return nothing
end

function run_amp(
    rng::AbstractRNG;
    observations::ContextualSBMObservations,
    csbm::ContextualSBM,
    init_std::Real=1e-3,
    iterations::Integer=10,
    show_progress=false,
)
    (; storage, next_storage, temp_storage) = init_amp(rng; observations, csbm, init_std)
    storage_history = [copy(storage)]
    prog = Progress(iterations; desc="AMP-BP", enabled=show_progress)
    for _ in 1:iterations
        update_amp!(next_storage, temp_storage; storage, observations, csbm)
        copy!(storage, next_storage)
        push!(storage_history, copy(storage))
        next!(prog)
    end
    return storage_history
end

function overlap(; storage::AMPStorage, latents::ContextualSBMLatents)
    û = sign.(storage.û)
    @assert all(abs.(û) .> 0.5)
    u = latents.u
    q̂ᵤ = (1 / length(û)) * max(count_equalities(û, u), count_equalities(û, -u))
    qᵤ = 2 * (q̂ᵤ - 0.5)
    return qᵤ
end
