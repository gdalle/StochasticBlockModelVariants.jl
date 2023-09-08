## Marginals and storage

"""
$(TYPEDEF)

# Fields

$(TYPEDFIELDS)
"""
@kwdef struct MarginalsGLMSBM{R<:Real}
    "length `N`"
    ŝ::Vector{R}
    "length `M`"
    ŵ::Vector{R}
    "length `M`, alias `a`"
    v::Vector{R}
    "length `M`"
    Γ::Vector{R}
    "length `N`"
    ω::Vector{R}
    "length `N`"
    gₒ::Vector{R}
    "size `(2, N)`"
    ψl::Matrix{R}
    "size `(2, N, N)`"
    χe::Dict{Tuple{Int,Int,Int},R}
    "size `(2, N)`"
    χl::Matrix{R}
    "size `(2, N)`"
    χ::Matrix{R}
end

Base.eltype(::MarginalsGLMSBM{R}) where {R} = R

## Message-passing

ind(s) = mod(s, 3)  # sends 1 to 1 and -1 to 2

function init_amp(
    rng::AbstractRNG, observations::ObservationsGLMSBM{R1}, glmsbm::GLMSBM{R2}; init_std::R3
) where {R1,R2,R3}
    R = promote_type(R1, R2, R3)
    (; N, M) = glmsbm
    (; g, Ξ) = observations

    ŝ = zeros(R, N)
    ŵ = init_std .* R.(randn(rng, M))
    v = ones(R, M)
    Γ = zeros(R, M)

    ω = zeros(R, N)
    gₒ = zeros(R, N)

    ψl = ones(R, 2, N) / 2
    χe = Dict{Tuple{Int,Int,Int},R}()
    for μ in 1:N, ν in neighbors(g, μ)
        p = one(R) / 2 + init_std * R(randn(rng))
        χe[1, μ, ν] = clamp(p, zero(R), one(R))
        χe[2, μ, ν] = 1 - χe[1, μ, ν]
    end
    χl = ones(R, 2, N) / 2
    χ = ones(R, 2, N) / 2

    marginals = MarginalsGLMSBM(; ŝ, ŵ, v, Γ, ω, gₒ, ψl, χl, χe, χ)
    next_marginals = deepcopy(marginals)
    return (; marginals, next_marginals)
end

function update_amp!(
    next_marginals::MarginalsGLMSBM{R},
    marginals::MarginalsGLMSBM{R},
    observations::ObservationsGLMSBM,
    glmsbm::GLMSBM,
) where {R}
    (; N, M, c, λ, Pʷ) = glmsbm
    (; g, Ξ, F) = observations
    (; cᵢ, cₒ) = affinities(glmsbm)
    C = [cᵢ cₒ; cₒ cᵢ]

    ŝᵗ = marginals.ŝ
    ŵᵗ = marginals.ŵ
    vᵗ = marginals.v
    Γᵗ = marginals.Γ
    ωᵗ = marginals.ω
    gₒᵗ = marginals.gₒ
    ψlᵗ = marginals.ψl
    χeᵗ = marginals.χe
    χlᵗ = marginals.χl
    χᵗ = marginals.χ

    ŝᵗ⁺¹ = next_marginals.ŝ
    ŵᵗ⁺¹ = next_marginals.ŵ
    vᵗ⁺¹ = next_marginals.v
    Γᵗ⁺¹ = next_marginals.Γ
    ωᵗ⁺¹ = next_marginals.ω
    gₒᵗ⁺¹ = next_marginals.gₒ
    ψlᵗ⁺¹ = next_marginals.ψl
    χeᵗ⁺¹ = next_marginals.χe
    χlᵗ⁺¹ = next_marginals.χl
    χᵗ⁺¹ = next_marginals.χ

    # AMP update of ω, V
    Vᵗ⁺¹ = sum(vᵗ) / M
    # mul!(ωᵗ⁺¹, F, ŵᵗ)
    # ωᵗ⁺¹ .-= Vᵗ⁺¹ .* gₒᵗ
    for μ in 1:N
        ωᵗ⁺¹[μ] = sum(F[μ, l] * ŵᵗ[l] for l in 1:M) - Vᵗ⁺¹ * gₒᵗ[μ]
    end

    # AMP update of ψl, gₒ, μ, Λ, Γ
    # for s in (-1, 1)
    #     @views ψlᵗ⁺¹[ind(s), :] .= (one(R) .+ s .* erf.(ωᵗ⁺¹ ./ sqrt(2Vᵗ⁺¹))) ./ 2
    # end
    # @views gₒᵗ⁺¹ = gₒ.(ωᵗ⁺¹, χlᵗ[1, :], Ref(Vᵗ⁺¹))  # TODO: toggle
    # @views gₒᵗ⁺¹ = gₒ.(ωᵗ⁺¹, prior.(R, 1, Ξ), Ref(Vᵗ⁺¹))
    for μ in 1:N
        gₒᵗ⁺¹[μ] = gₒ(ωᵗ⁺¹[μ], prior(R, 1, Ξ[μ]), Vᵗ⁺¹)
    end
    Λᵗ⁺¹ = sum(abs2, gₒᵗ⁺¹) / M
    # mul!(Γᵗ⁺¹, F', gₒᵗ⁺¹)
    # Γᵗ⁺¹ .+= Λᵗ⁺¹ .* ŵᵗ
    for l in 1:M
        Γᵗ⁺¹[l] = Λᵗ⁺¹ * ŵᵗ[l] + sum(F[μ, l] * gₒᵗ⁺¹[μ] for μ in 1:N)
    end

    # AMP update of the estimated marginals a, v
    ŵᵗ⁺¹ .= fₐ.(Ref(Pʷ), Λᵗ⁺¹, Γᵗ⁺¹)
    vᵗ⁺¹ .= fᵥ.(Ref(Pʷ), Λᵗ⁺¹, Γᵗ⁺¹)

    # # BP update of the field h
    # hᵗ⁺¹ = Vector{R}(undef, 2)
    # for s in (-1, 1)
    #     hᵗ⁺¹[ind(s)] =
    #         sum(C[ind(s), ind(sμ)] * χᵗ[ind(sμ), μ] for μ in 1:N for sμ in (-1, 1)) / N
    # end

    # # BP update of the messages χe and of the marginals χ
    # for μ in 1:N
    #     for sμ in (-1, 1)
    #         χᵗ⁺¹[ind(sμ), μ] = prior(R, sμ, Ξ[μ]) * exp(-hᵗ⁺¹[ind(sμ)]) * ψlᵗ⁺¹[ind(sμ), μ]
    #         for η in neighbors(g, μ)
    #             χᵗ⁺¹[ind(sμ), μ] *= sum(
    #                 C[ind(sη), ind(sμ)] * χeᵗ[ind(sη), η, μ] for sη in (-1, 1)
    #             )
    #         end
    #     end
    #     @views χᵗ⁺¹[:, μ] ./= sum(χᵗ⁺¹[:, μ])
    # end
    # for μ in 1:N, ν in neighbors(g, μ)
    #     for sμ in (-1, 1)
    #         extra_factor = sum(C[ind(sν), ind(sμ)] * χeᵗ[ind(sν), ν, μ] for sν in (-1, 1))
    #         χeᵗ⁺¹[ind(sμ), μ, ν] = χᵗ⁺¹[ind(sμ), μ] / extra_factor
    #     end
    #     normalization = χeᵗ⁺¹[1, μ, ν] + χeᵗ⁺¹[2, μ, ν]
    #     χeᵗ⁺¹[1, μ, ν] /= normalization
    #     χeᵗ⁺¹[2, μ, ν] /= normalization
    # end

    # # BP update of the SBM-to-GLM messages χl
    # for μ in 1:N
    #     for sμ in (-1, 1)
    #         χlᵗ⁺¹[ind(sμ), μ] = prior(R, sμ, Ξ[μ]) * exp(-hᵗ⁺¹[ind(sμ)])
    #         for η in neighbors(g, μ)
    #             χlᵗ⁺¹[ind(sμ), μ] *= sum(
    #                 C[ind(sη), ind(sμ)] * χeᵗ[ind(sη), η, μ] for sη in (-1, 1)
    #             )
    #         end
    #     end
    #     @views χlᵗ⁺¹[:, μ] ./= sum(χlᵗ⁺¹[:, μ])
    # end

    # @views ŝᵗ⁺¹ .= 2 .* χᵗ⁺¹[1, :] .- one(R)

    return nothing
end

function run_amp(
    rng::AbstractRNG,
    observations::ObservationsGLMSBM,
    glmsbm::GLMSBM;
    init_std=1e-3,
    max_iterations=100,
    convergence_threshold=1e-3,
    recent_past=10,
    damping=0.5,
    show_progress=false,
)
    (; N, M) = glmsbm
    (; marginals, next_marginals) = init_amp(rng, observations, glmsbm; init_std)

    R = eltype(marginals)
    ŝ_history = Matrix{R}(undef, N, max_iterations)
    ŵ_history = Matrix{R}(undef, M, max_iterations)
    converged = false
    prog = Progress(max_iterations; desc="AMP-BP for GLM-SBM", enabled=show_progress)

    for t in 1:max_iterations
        update_amp!(next_marginals, marginals, observations, glmsbm)
        copy_damp!(marginals, next_marginals; damping=(t == 1 ? zero(damping) : damping))

        ŝ_history[:, t] .= marginals.ŝ
        ŵ_history[:, t] .= marginals.ŵ

        if t <= recent_past
            ŝ_recent_std = typemax(R)
            ŵ_recent_std = typemax(R)
        else
            ŝ_recent_std = mean(std(view(ŝ_history, :, (t - recent_past):t); dims=2))
            ŵ_recent_std = mean(std(view(ŵ_history, :, (t - recent_past):t); dims=2))
        end
        converged = (
            ŝ_recent_std < convergence_threshold && ŵ_recent_std < convergence_threshold
        )
        if converged
            ŝ_history = ŝ_history[:, 1:t]
            ŵ_history = ŵ_history[:, 1:t]
            break
        else
            showvalues = [
                (:ŝ_recent_std, ŝ_recent_std),
                (:ŵ_recent_std, ŵ_recent_std),
                (:convergence_threshold, convergence_threshold),
            ]
            next!(prog; showvalues)
        end
    end

    return (; ŝ_history, ŵ_history, converged)
end

function evaluate_amp(rng::AbstractRNG, glmsbm::GLMSBM; kwargs...)
    (; latents, observations) = rand(rng, glmsbm)
    (; ŝ_history, ŵ_history, converged) = run_amp(rng, observations, glmsbm; kwargs...)
    q_dis = discrete_overlap(latents.s, ŝ_history[:, end])
    q_cont = continuous_overlap(latents.w, ŵ_history[:, end])
    return (; q_dis, q_cont, converged)
end
