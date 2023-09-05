## Marginals and storage

"""
    MarginalsGLMSBM

# Fields

- `a::Vector`: length `M`
- `v::Vector`: length `M`
- `Γ::Vector`: length `M`
- `ω::Vector`: length `N`
- `gₒ::Vector`: length `N`
- `ψl::Vector{PlusMinusMeasure}`: length `N`
- `χe::Dict{Tuple{Int,Int},PlusMinusMeasure}`: size `(N, N)`
- `χl::Vector{PlusMinusMeasure}`: length `N`
- `χ::Vector{PlusMinusMeasure}`: length `N`

"""
@kwdef struct MarginalsGLMSBM{R<:Real}
    a::Vector{R}
    v::Vector{R}
    Γ::Vector{R}
    ω::Vector{R}
    gₒ::Vector{R}
    ψl::Vector{PlusMinusMeasure{R}}
    χe::Dict{Tuple{Int,Int},PlusMinusMeasure{R}}
    χl::Vector{PlusMinusMeasure{R}}
    χ::Vector{PlusMinusMeasure{R}}
end

function Base.copy(marginals::MarginalsGLMSBM)
    return MarginalsGLMSBM(;
        a=copy(marginals.a),
        v=copy(marginals.v),
        Γ=copy(marginals.Γ),
        ω=copy(marginals.ω),
        gₒ=copy(marginals.gₒ),
        ψl=copy(marginals.ψl),
        χe=copy(marginals.χe),
        χl=copy(marginals.χl),
        χ=copy(marginals.χ),
    )
end

function Base.copy!(marginals_dest::MarginalsGLMSBM, marginals_source::MarginalsGLMSBM)
    copy!(marginals_dest.a, marginals_source.a)
    copy!(marginals_dest.v, marginals_source.v)
    copy!(marginals_dest.Γ, marginals_source.Γ)
    copy!(marginals_dest.ω, marginals_source.ω)
    copy!(marginals_dest.gₒ, marginals_source.gₒ)
    copy!(marginals_dest.ψl, marginals_source.ψl)
    copy!(marginals_dest.χe, marginals_source.χe)
    copy!(marginals_dest.χl, marginals_source.χl)
    copy!(marginals_dest.χ, marginals_source.χ)
    return marginals_dest
end

## Message-passing

function init_amp(
    rng::AbstractRNG; observations::ObservationsGLMSBM{R}, glmsbm::GLMSBM{R}, init_std
) where {R}
    (; N, M) = glmsbm
    (; g, Ξ) = observations

    a = init_std .* randn(rng, R, M)
    v = ones(R, M)
    Γ = zeros(R, M)

    ω = zeros(R, N)
    gₒ = zeros(R, N)

    ψl = [PlusMinusMeasure(one(R) / 2) for μ in 1:N]
    χe = Dict{Tuple{Int,Int},PlusMinusMeasure{R}}()
    for μ in 1:N, ν in neighbors(g, μ)
        p = one(R) / 2 + init_std * randn(rng, R)
        χe[μ, ν] = PlusMinusMeasure(clamp(p, zero(R), one(R)))
    end
    χl = [PlusMinusMeasure(one(R) / 2) for μ in 1:N]
    χ = [PlusMinusMeasure(one(R) / 2) for μ in 1:N]

    marginals = MarginalsGLMSBM(; a, v, Γ, ω, gₒ, ψl, χl, χe, χ)
    next_marginals = copy(marginals)
    return (; marginals, next_marginals)
end

function update_amp!(
    next_marginals::MarginalsGLMSBM{R};
    marginals::MarginalsGLMSBM{R},
    observations::ObservationsGLMSBM{R},
    glmsbm::GLMSBM{R},
) where {R}
    (; N, M, c, λ, Pʷ) = glmsbm
    (; g, Ξ, F) = observations
    (; cᵢ, cₒ) = affinities(glmsbm)
    C = AffinityMatrix(cᵢ, cₒ)

    (aᵗ, vᵗ, Γᵗ, ωᵗ, gₒᵗ, ψlᵗ, χeᵗ, χlᵗ, χᵗ) = (
        marginals.a,
        marginals.v,
        marginals.Γ,
        marginals.ω,
        marginals.gₒ,
        marginals.ψl,
        marginals.χe,
        marginals.χl,
        marginals.χ,
    )

    (aᵗ⁺¹, vᵗ⁺¹, Γᵗ⁺¹, ωᵗ⁺¹, gₒᵗ⁺¹, ψlᵗ⁺¹, χeᵗ⁺¹, χlᵗ⁺¹, χᵗ⁺¹) = (
        next_marginals.a,
        next_marginals.v,
        next_marginals.Γ,
        next_marginals.ω,
        next_marginals.gₒ,
        next_marginals.ψl,
        next_marginals.χe,
        next_marginals.χl,
        next_marginals.χ,
    )

    # AMP update of ω, V
    Vᵗ⁺¹ = sum(vᵗ) / M
    mul!(ωᵗ⁺¹, F, aᵗ)
    ωᵗ⁺¹ .-= Vᵗ⁺¹ .* gₒᵗ

    # AMP update of ψl, gₒ, μ, Λ, Γ
    for μ in 1:N, sμ in (-1, 1)
        ψlᵗ⁺¹[μ][sμ] = (one(R) + sμ * erf(ωᵗ⁺¹[μ] / sqrt(2Vᵗ⁺¹))) / 2
    end
    gₒᵗ⁺¹ = gₒ.(ωᵗ⁺¹, χlᵗ, Ref(Vᵗ⁺¹))
    Λᵗ⁺¹ = sum(abs2, gₒᵗ⁺¹) / M
    mul!(Γᵗ⁺¹, F, gₒᵗ⁺¹)
    Γᵗ⁺¹ .+= Λᵗ⁺¹ .* aᵗ

    # AMP update of the estimated marginals a, v
    aᵗ⁺¹ .= fₐ.(Ref(Pʷ), Ref(Λᵗ⁺¹), Γᵗ⁺¹)
    vᵗ⁺¹ .= fᵥ.(Ref(Pʷ), Ref(Λᵗ⁺¹), Γᵗ⁺¹)

    # BP update of the field h
    hᵗ⁺¹ = PlusMinusMeasure(zero(R), zero(R))
    for s in (-1, 1)
        hᵗ⁺¹[s] = sum(C[s, sμ] * χᵗ[μ][sμ] for μ in 1:N for sμ in (-1, 1))
    end

    # BP update of the messages χe and of the marginals χ
    for μ in 1:N, ν in neighbors(g, μ)
        for sμ in (-1, 1)
            χeᵗ⁺¹[μ, ν][sμ] = prior(R, sμ, Ξ[μ]) * exp(-hᵗ⁺¹[sμ]) * ψlᵗ⁺¹[μ][sμ]
            for η in neighbors(g, μ)
                if η != ν
                    χeᵗ⁺¹[μ, ν][sμ] *= sum(C[sη, sμ] * χeᵗ[η, μ][sη] for sη in (-1, 1))
                end
            end
        end
        normalize!(χeᵗ⁺¹[μ, ν])
    end
    for μ in 1:N
        for sμ in (-1, 1)
            χᵗ⁺¹[μ][sμ] = prior(R, sμ, Ξ[μ]) * exp(-hᵗ⁺¹[sμ]) * ψlᵗ⁺¹[μ][sμ]
            for η in neighbors(g, μ)
                χᵗ⁺¹[μ][sμ] *= sum(C[sη, sμ] * χeᵗ[η, μ][sη] for sη in (-1, 1))
            end
        end
        normalize!(χᵗ⁺¹[μ])
    end

    # BP update of the SBM-to-GLM messages χl
    for μ in 1:N
        for sμ in (-1, 1)
            χlᵗ⁺¹[μ][sμ] = prior(R, sμ, Ξ[μ]) * exp(-hᵗ⁺¹[sμ])
            for η in neighbors(g, μ)
                χlᵗ⁺¹[μ][sμ] *= sum(C[sη, sμ] * χeᵗ[η, μ][sη] for sη in (-1, 1))
            end
        end
        normalize!(χlᵗ⁺¹[μ])
    end

    return nothing
end

function run_amp(
    rng::AbstractRNG,
    observations::ObservationsGLMSBM{R},
    glmsbm::GLMSBM{R};
    init_std=1e-3,
    max_iterations=200,
    convergence_threshold=1e-3,
    recent_past=10,
    show_progress=false,
) where {R}
    (; marginals, next_marginals) = init_amp(rng; observations, glmsbm, init_std)
    converged = false
    prog = Progress(max_iterations; desc="AMP-BP for GLM-SBM", enabled=show_progress)
    for t in 1:max_iterations
        update_amp!(next_marginals; marginals, observations, glmsbm)
        copy!(marginals, next_marginals)
        next!(prog)
    end
    return marginals
end
