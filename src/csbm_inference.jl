## Marginals

"""
$(TYPEDEF)

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct MarginalsCSBM{R}
    "posterior mean of `u`, length `N`"
    û::Vector{R}
    "posterior mean of `v`, length `P`"
    v̂::Vector{R}
    "posterior mean of `u` if there were no features, length `N` (aka `Bᵥ`)"
    û_no_feat::Vector{R}
    "posterior mean of `v` if there were no communities, length `P` (aka `Bᵤ`)"
    v̂_no_comm::Vector{R}
    "individual external field for `u=1`, length `N`"
    h̃₊::Vector{R}
    "individual external field for `u=-1`, length `N`"
    h̃₋::Vector{R}
    "messages about the marginal distribution of `u`, size `(N, N)`"
    χe₊::Dict{Tuple{Int,Int},R}
    "marginal probability of `u=1`, length `N`"
    χ₊::Vector{R}
end

Base.eltype(::MarginalsCSBM{R}) where {R} = R

## Message-passing

function init_amp(
    rng::AbstractRNG, observations::ObservationsCSBM{R1}, csbm::CSBM{R2}; init_std::R3
) where {R1,R2,R3}
    R = promote_type(R1, R2, R3)
    (; N, P) = csbm
    (; g, Ξ) = observations

    û = 2 .* prior.(R, 1, Ξ) .- one(R) .+ init_std .* randn(rng, R, N)
    v̂ = init_std .* randn(rng, R, P)

    û_no_feat = zeros(R, N)
    v̂_no_comm = zeros(R, P)

    h̃₊ = zeros(R, N)
    h̃₋ = zeros(R, N)

    χe₊ = Dict{Tuple{Int,Int},R}()
    for i in 1:N, j in neighbors(g, i)
        χe₊[i, j] = prior(R, 1, Ξ[i]) + init_std * randn(rng, R)
    end
    χ₊ = zeros(R, N)

    marginals = MarginalsCSBM(; û, v̂, û_no_feat, v̂_no_comm, h̃₊, h̃₋, χe₊, χ₊)
    next_marginals = deepcopy(marginals)
    return (; marginals, next_marginals)
end

function update_amp!(
    next_marginals::MarginalsCSBM{R},
    marginals::MarginalsCSBM{R},
    observations::ObservationsCSBM,
    csbm::CSBM,
) where {R}
    (; d, λ, μ, N, P) = csbm
    (; g, Ξ, B) = observations
    (; cᵢ, cₒ) = affinities(csbm)

    ûᵗ, v̂ᵗ, χe₊ᵗ = marginals.û, marginals.v̂, marginals.χe₊
    ûᵗ⁺¹, v̂ᵗ⁺¹, χe₊ᵗ⁺¹ = next_marginals.û, next_marginals.v̂, next_marginals.χe₊
    (; û_no_feat, v̂_no_comm, h̃₊, h̃₋, χ₊) = next_marginals

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
    h̃₊ .= -h₊ .+ log.(prior.(R, +1, Ξ)) .+ û_no_feat
    h̃₋ .= -h₋ .+ log.(prior.(R, -1, Ξ)) .- û_no_feat

    # BP update of the marginals
    for i in 1:N
        s_i = h̃₊[i] - h̃₋[i]
        for k in neighbors(g, i)
            common = 2λ * sqrt(d) * χe₊ᵗ[k, i]
            s_i += log((cₒ + common) / (cᵢ - common))
        end
        χ₊[i] = s_i
    end

    # BP update of the messages
    for i in 1:N, j in neighbors(g, i)
        common = 2λ * sqrt(d) * χe₊ᵗ[j, i]
        s_ij = log((cₒ + common) / (cᵢ - common))
        χe₊ᵗ⁺¹[i, j] = χ₊[i] - s_ij
    end

    # Sigmoidize probabilities
    χ₊ .= sigmoid.(χ₊)
    for (key, val) in pairs(χe₊ᵗ⁺¹)
        χe₊ᵗ⁺¹[key] = sigmoid(val)
    end

    # BP estimation of u
    ûᵗ⁺¹ .= 2 .* χ₊ .- one(R)

    return nothing
end

function run_amp(
    rng::AbstractRNG,
    observations::ObservationsCSBM,
    csbm::CSBM;
    init_std=1e-3,
    max_iterations=200,
    convergence_threshold=1e-3,
    recent_past=5,
    damping=0.0,
    show_progress=false,
)
    (; N, P) = csbm
    (; marginals, next_marginals) = init_amp(rng, observations, csbm; init_std)

    R = eltype(marginals)
    û_history = Matrix{R}(undef, N, max_iterations)
    v̂_history = Matrix{R}(undef, P, max_iterations)
    converged = false
    prog = Progress(max_iterations; desc="AMP-BP for CSBM", enabled=show_progress)

    for t in 1:max_iterations
        update_amp!(next_marginals, marginals, observations, csbm)
        copy_damp!(marginals, next_marginals; damping=(t == 1 ? zero(damping) : damping))

        û_history[:, t] .= marginals.û
        v̂_history[:, t] .= marginals.v̂

        if t <= recent_past
            û_recent_std = typemax(R)
            v̂_recent_std = typemax(R)
        else
            û_recent_std = mean(std(view(û_history, :, (t - recent_past):t); dims=2))
            v̂_recent_std = mean(std(view(v̂_history, :, (t - recent_past):t); dims=2))
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

    return (û_history, v̂_history, converged)
end

function evaluate_amp(rng::AbstractRNG, csbm::CSBM; kwargs...)
    (; latents, observations) = rand(rng, csbm)
    (û_history, v̂_history, converged) = run_amp(rng, observations, csbm; kwargs...)
    qu = discrete_overlap(latents.u, û_history[:, end])
    qv = continuous_overlap(latents.v, v̂_history[:, end])
    return (qu, qv, converged)
end
