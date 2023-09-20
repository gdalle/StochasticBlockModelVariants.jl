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
    "length 2"
    h::Vector{R}
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

discrete_estimates(marginals::MarginalsGLMSBM) = marginals.ŝ
continuous_estimates(marginals::MarginalsGLMSBM) = marginals.ŵ

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
    h = zeros(R, 2)

    ψl = ones(R, 2, N) / 2
    χe = Dict{Tuple{Int,Int,Int},R}()
    for μ in 1:N, ν in neighbors(g, μ)
        p = one(R) / 2 + init_std * R(randn(rng))
        χe[1, μ, ν] = clamp(p, zero(R), one(R))
        χe[2, μ, ν] = 1 - χe[1, μ, ν]
    end
    χl = ones(R, 2, N) / 2
    χ = ones(R, 2, N) / 2

    marginals = MarginalsGLMSBM(; ŝ, ŵ, v, Γ, ω, gₒ, h, ψl, χl, χe, χ)
    next_marginals = deepcopy(marginals)
    return (; marginals, next_marginals)
end

function update_amp!(
    next_marginals::MarginalsGLMSBM{R},
    marginals::MarginalsGLMSBM{R},
    observations::ObservationsGLMSBM,
    glmsbm::GLMSBM,
) where {R}
    (; N, M, Pʷ) = glmsbm
    (; g, Ξ, F) = observations
    (; cᵢ, cₒ) = affinities(glmsbm)
    C = ((cᵢ, cₒ), (cₒ, cᵢ))

    ŵᵗ = marginals.ŵ
    vᵗ = marginals.v
    gₒᵗ = marginals.gₒ
    χeᵗ = marginals.χe
    χlᵗ = marginals.χl
    χᵗ = marginals.χ

    ŝᵗ⁺¹ = next_marginals.ŝ
    ŵᵗ⁺¹ = next_marginals.ŵ
    vᵗ⁺¹ = next_marginals.v
    Γᵗ⁺¹ = next_marginals.Γ
    ωᵗ⁺¹ = next_marginals.ω
    gₒᵗ⁺¹ = next_marginals.gₒ
    hᵗ⁺¹ = next_marginals.h
    ψlᵗ⁺¹ = next_marginals.ψl
    χeᵗ⁺¹ = next_marginals.χe
    χlᵗ⁺¹ = next_marginals.χl
    χᵗ⁺¹ = next_marginals.χ

    # AMP update of ω, V
    Vᵗ⁺¹ = sum(vᵗ) / M
    mul!(ωᵗ⁺¹, F, ŵᵗ)
    ωᵗ⁺¹ .-= Vᵗ⁺¹ .* gₒᵗ

    # AMP update of ψl, gₒ, μ, Λ, Γ
    for s in (-1, 1)
        @views ψlᵗ⁺¹[ind(s), :] .= (one(R) .+ s .* erf.(ωᵗ⁺¹ ./ sqrt(2Vᵗ⁺¹))) ./ 2
    end
    @views gₒᵗ⁺¹ .= gₒ.(ωᵗ⁺¹, χlᵗ[1, :], Ref(Vᵗ⁺¹))
    Λᵗ⁺¹ = sum(abs2, gₒᵗ⁺¹) / M
    mul!(Γᵗ⁺¹, F', gₒᵗ⁺¹)
    Γᵗ⁺¹ .+= Λᵗ⁺¹ .* ŵᵗ

    # AMP update of the estimated marginals a, v
    ŵᵗ⁺¹ .= fₐ.(Ref(Pʷ), Λᵗ⁺¹, Γᵗ⁺¹)
    vᵗ⁺¹ .= fᵥ.(Ref(Pʷ), Λᵗ⁺¹, Γᵗ⁺¹)

    # # BP update of the field h
    for s in (-1, 1)
        hᵗ⁺¹[ind(s)] =
            sum(C[ind(s)][ind(sμ)] * χᵗ[ind(sμ), μ] for μ in 1:N for sμ in (-1, 1)) / N
    end

    # # BP update of the messages χe and of the marginals χ
    for μ in 1:N
        for sμ in (-1, 1)
            χᵗ⁺¹[ind(sμ), μ] = prior(R, sμ, Ξ[μ]) * exp(-hᵗ⁺¹[ind(sμ)]) * ψlᵗ⁺¹[ind(sμ), μ]
            for η in neighbors(g, μ)
                χᵗ⁺¹[ind(sμ), μ] *= sum(
                    C[ind(sη)][ind(sμ)] * χeᵗ[ind(sη), η, μ] for sη in (-1, 1)
                )
            end
        end
        @views χᵗ⁺¹[:, μ] ./= sum(χᵗ⁺¹[:, μ])
    end
    for μ in 1:N, ν in neighbors(g, μ)
        for sμ in (-1, 1)
            extra_factor = sum(C[ind(sν)][ind(sμ)] * χeᵗ[ind(sν), ν, μ] for sν in (-1, 1))
            χeᵗ⁺¹[ind(sμ), μ, ν] = χᵗ⁺¹[ind(sμ), μ] / extra_factor
        end
        normalization = χeᵗ⁺¹[1, μ, ν] + χeᵗ⁺¹[2, μ, ν]
        χeᵗ⁺¹[1, μ, ν] /= normalization
        χeᵗ⁺¹[2, μ, ν] /= normalization
    end

    # # BP update of the SBM-to-GLM messages χl
    for μ in 1:N
        for sμ in (-1, 1)
            χlᵗ⁺¹[ind(sμ), μ] = prior(R, sμ, Ξ[μ]) * exp(-hᵗ⁺¹[ind(sμ)])
            for η in neighbors(g, μ)
                χlᵗ⁺¹[ind(sμ), μ] *= sum(
                    C[ind(sη)][ind(sμ)] * χeᵗ[ind(sη), η, μ] for sη in (-1, 1)
                )
            end
        end
        @views χlᵗ⁺¹[:, μ] ./= sum(χlᵗ⁺¹[:, μ])
    end

    @views ŝᵗ⁺¹ .= 2 .* χᵗ⁺¹[1, :] .- one(R)

    return nothing
end
