## Marginals

"""
    AMPMarginalsCSBM

# Fields

- `û::Vector`: posterior mean of `u`, length `N`
- `v̂::Vector`: posterior mean of `v`, length `P`
- `û_no_feat::Vector`: posterior mean of `u` if there were no features, length `N` (aka `Bᵥ`)
- `v̂_no_comm::Vector`: posterior mean of `v` if there were no communities, length `P` (aka `Bᵤ`)
- `h̃₊::Vector`: individual external field for `u=1`, length `N`
- `h̃₋::Vector`: individual external field for `u=-1`, length `N`
- `χ₊e::Dict`: messages about the marginal distribution of `u`, size `(N, N)`
- `χ₊::Vector`: marginal probability of `u=1`, length `N`
"""
@kwdef struct AMPMarginalsCSBM{R<:Real}
    û::Vector{R}
    v̂::Vector{R}
    û_no_feat::Vector{R}
    v̂_no_comm::Vector{R}
    h̃₊::Vector{R}
    h̃₋::Vector{R}
    χ₊e::Dict{Tuple{Int,Int},R}
    χ₊::Vector{R}
end

function Base.copy(marginals::AMPMarginalsCSBM)
    return AMPMarginalsCSBM(;
        û=copy(marginals.û),
        v̂=copy(marginals.v̂),
        û_no_feat=copy(marginals.û_no_feat),
        v̂_no_comm=copy(marginals.v̂_no_comm),
        h̃₊=copy(marginals.h̃₊),
        h̃₋=copy(marginals.h̃₋),
        χ₊e=copy(marginals.χ₊e),
        χ₊=copy(marginals.χ₊),
    )
end

function Base.copy!(marginals_dest::AMPMarginalsCSBM, marginals_source::AMPMarginalsCSBM)
    copy!(marginals_dest.û, marginals_source.û),
    copy!(marginals_dest.v̂, marginals_source.v̂),
    copy!(marginals_dest.û_no_feat, marginals_source.û_no_feat),
    copy!(marginals_dest.v̂_no_comm, marginals_source.v̂_no_comm),
    copy!(marginals_dest.h̃₊, marginals_source.h̃₊),
    copy!(marginals_dest.h̃₋, marginals_source.h̃₋),
    copy!(marginals_dest.χ₊e, marginals_source.χ₊e),
    copy!(marginals_dest.χ₊, marginals_source.χ₊),
    return marginals_dest
end

## Message-passing

function init_amp(
    rng::AbstractRNG; observations::ObservationsCSBM{R}, csbm::CSBM{R}, init_std
) where {R}
    (; N, P) = csbm
    (; g, Ξ) = observations

    û = 2 .* prior₊.(R, Ξ) .- one(R) .+ init_std .* randn(rng, R, N)
    v̂ = init_std .* randn(rng, R, P)
    
    û_no_feat = zeros(R, N)
    v̂_no_comm = zeros(R, P)
    
    h̃₊ = zeros(R, N)
    h̃₋ = zeros(R, N)
    
    χ₊e = Dict{Tuple{Int,Int},R}()
    for i in 1:N, j in neighbors(g, i)
        χ₊e[i, j] = prior₊(R, Ξ[i]) + init_std * randn(rng, R)
    end
    χ₊ = zeros(R, N)
    
    marginals = AMPMarginalsCSBM(; û, v̂, û_no_feat, v̂_no_comm, h̃₊, h̃₋, χ₊e, χ₊)
    next_marginals = copy(marginals)
    return (; marginals, next_marginals)
end

function update_amp!(
    next_marginals::AMPMarginalsCSBM{R};
    marginals::AMPMarginalsCSBM{R},
    observations::ObservationsCSBM{R},
    csbm::CSBM{R},
) where {R}
    (; d, λ, μ, N, P) = csbm
    (; g, B, Ξ) = observations
    (; cᵢ, cₒ) = affinities(csbm)

    ûᵗ, v̂ᵗ, χ₊eᵗ = marginals.û, marginals.v̂, marginals.χ₊e
    ûᵗ⁺¹, v̂ᵗ⁺¹, χ₊eᵗ⁺¹ = next_marginals.û, next_marginals.v̂, next_marginals.χ₊e
    (; û_no_feat, v̂_no_comm, h̃₊, h̃₋, χ₊) = next_marginals

    ûₜ_sum = sum(ûᵗ)
    ûₜ_sum2 = sum(abs2, ûᵗ)

    # CSBMAMP estimation of v
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
    observations::ObservationsCSBM{R},
    csbm::CSBM{R},
    init_std=1e-3,
    max_iterations=200,
    convergence_threshold=1e-3,
    recent_past=10,
    show_progress=false,
) where {R}
    (; N, P) = csbm
    (; marginals, next_marginals) = init_amp(rng; observations, csbm, init_std)

    û_history = Matrix{R}(undef, N, max_iterations)
    v̂_history = Matrix{R}(undef, P, max_iterations)
    converged = false
    prog = Progress(max_iterations; desc="AMP-BP for CSBM", enabled=show_progress)

    for t in 1:max_iterations
        update_amp!(next_marginals; marginals, observations, csbm)
        copy!(marginals, next_marginals)

        û_history[:, t] .= marginals.û
        v̂_history[:, t] .= marginals.v̂

        if t <= recent_past
            û_recent_std = typemax(R)
            v̂_recent_std = typemax(R)
        else
            û_recent_std = maximum(std(view(û_history, :, (t - recent_past):t); dims=2))
            v̂_recent_std = maximum(std(view(v̂_history, :, (t - recent_past):t); dims=2))
        end
        converged = (
            û_recent_std < convergence_threshold && v̂_recent_std < convergence_threshold
        )
        if converged
            û_history = û_history[:, 1:t]
            v̂_history = v̂_history[:, 1:t]
            break
        else
            showvalues = [
                (:û_recent_std, û_recent_std),
                (:v̂_recent_std, v̂_recent_std),
                (:convergence_threshold, convergence_threshold),
            ]
            next!(prog; showvalues)
        end
    end

    return (; û_history, v̂_history, converged)
end

function evaluate_amp(rng::AbstractRNG; csbm::CSBM, kwargs...)
    (; latents, observations) = rand(rng, csbm)
    (; û_history, v̂_history, converged) = run_amp(rng; observations, csbm, kwargs...)
    (; qᵤ, qᵥ) = overlaps(;
        u=latents.u, v=latents.v, û=û_history[:, end], v̂=v̂_history[:, end]
    )
    return (; qᵤ, qᵥ)
end
