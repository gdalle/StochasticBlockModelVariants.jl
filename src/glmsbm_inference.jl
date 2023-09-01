## Marginals and storage

"""
    MarginalsGLMSBM

# Fields

- `a::Vector{R}`: length `M`
- `v::Vector{R}`: length `M`
- `Γ::Vector{R}`: length `M`
- `ω::Vector{R}`: length `N`
- `gₒ::Vector{R}`: length `N`
- `ψl₊::Vector{R}`: length `N`
- `χl₊::Vector{R}`: length `N`
- `χe₊::Dict{Tuple{Int,Int},R}`: size `(N, N)`
- `χ₊::Vector{R}`: length `N`

"""
@kwdef struct MarginalsGLMSBM{R<:Real}
    a::Vector{R}
    v::Vector{R}
    Γ::Vector{R}
    ω::Vector{R}
    gₒ::Vector{R}
    ψl₊::Vector{R}
    ψl₋::Vector{R}
    χl₊::Vector{R}
    χl₋::Vector{R}
    χe₊::Dict{Tuple{Int,Int},R}
    χe₋::Dict{Tuple{Int,Int},R}
    χ₊::Vector{R}
    χ₋::Vector{R}
end

function Base.copy(marginals::MarginalsGLMSBM)
    return MarginalsCSBM(;
        a=copy(marginals.a),
        v=copy(marginals.v),
        Γ=copy(marginals.Γ),
        ω=copy(marginals.ω),
        gₒ=copy(marginals.gₒ),
        ψl₊=copy(marginals.ψl₊),
        ψl₋=copy(marginals.ψl₋),
        χl₊=copy(marginals.χl₊),
        χl₋=copy(marginals.χl₋),
        χe₊=copy(marginals.χe₊),
        χe₋=copy(marginals.χe₋),
        χ₊=copy(marginals.χ₊),
        χ₋=copy(marginals.χ₋),
    )
end

function Base.copy!(marginals_dest::MarginalsGLMSBM, marginals_source::MarginalsGLMSBM)
    copy!(marginals_dest.a, marginals_source.a)
    copy!(marginals_dest.v, marginals_source.v)
    copy!(marginals_dest.Γ, marginals_source.Γ)
    copy!(marginals_dest.ω, marginals_source.ω)
    copy!(marginals_dest.gₒ, marginals_source.gₒ)
    copy!(marginals_dest.ψl₊, marginals_source.ψl₊)
    copy!(marginals_dest.ψl₋, marginals_source.ψl₋)
    copy!(marginals_dest.χl₊, marginals_source.χl₊)
    copy!(marginals_dest.χl₋, marginals_source.χl₋)
    copy!(marginals_dest.χe₊, marginals_source.χe₊)
    copy!(marginals_dest.χe₋, marginals_source.χe₋)
    copy!(marginals_dest.χ₊, marginals_source.χ₊)
    copy!(marginals_dest.χ₋, marginals_source.χ₋)
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

    χl₊ = ones(R, N) / 2
    χe₊ = Dict{Tuple{Int,Int},R}()
    for μ in 1:N, ν in neighbors(g, μ)
        χe₊[μ, ν] = one(R) / 2 + init_std * randn(rng, R)
    end
    χ₊ = zeros(R, N)

    marginals = MarginalsGLMSBM(; a, v, Γ, ω, gₒ, ψl₊, ψl₋, χl₊, χl₋, χe₊, χe₋, χ₊, χ₋)
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

    (aᵗ, vᵗ, Γᵗ, ωᵗ, gₒᵗ, ψl₊ᵗ, ψl₋ᵗ, χl₊ᵗ, χl₋ᵗ, χe₊ᵗ, χe₋ᵗ, χ₊ᵗ, χ₋ᵗ) = (
        marginals.a,
        marginals.v,
        marginals.Γ,
        marginals.ω,
        marginals.gₒ,
        marginals.ψl₊,
        marginals.ψl₋,
        marginals.χl₊,
        marginals.χl₋,
        marginals.χe₊,
        marginals.χe₋,
        marginals.χ₊,
        marginals.χ₋,
    )
    (aᵗ⁺¹, vᵗ⁺¹, Γᵗ⁺¹, ωᵗ⁺¹, gₒᵗ⁺¹, ψl₊ᵗ⁺¹, ψl₋ᵗ⁺¹, χl₊ᵗ⁺¹, χl₋ᵗ⁺¹, χe₊ᵗ⁺¹, χe₋ᵗ⁺¹, χ₊ᵗ⁺¹, χ₋ᵗ⁺¹) = (
        next_marginals.a,
        next_marginals.v,
        next_marginals.Γ,
        next_marginals.ω,
        next_marginals.gₒ,
        next_marginals.ψl₊,
        next_marginals.ψl₋,
        next_marginals.χl₊,
        next_marginals.χl₋,
        next_marginals.χe₊,
        next_marginals.χe₋,
        next_marginals.χ₊,
        next_marginals.χ₋,
    )

    # AMP update of ω, V
    Vᵗ⁺¹ = sum(vᵗ) / M
    mul!(ωᵗ⁺¹, F, aᵗ)
    ωᵗ⁺¹ .-= Vᵗ⁺¹ .* gₒᵗ

    # AMP update of ψl, gₒ, μ, Λ, Γ
    ψl₊ .= missing  # TODO: fix
    gₒᵗ⁺¹ = gₒ.(ωᵗ⁺¹, χl₊ᵗ, χl₋ᵗ, Ref(Vᵗ⁺¹))
    Λᵗ⁺¹ = sum(abs2, gₒᵗ⁺¹) / M
    mul!(Γᵗ⁺¹, F, gₒᵗ⁺¹)
    Γᵗ⁺¹ .+= Λᵗ⁺¹ .* aᵗ

    # AMP update of the estimated marginals a, v
    aᵗ⁺¹ .= fₐ.(Ref(Λᵗ⁺¹), Γᵗ⁺¹)
    vᵗ⁺¹ .= fᵥ.(Ref(Λᵗ⁺¹), Γᵗ⁺¹)

    # BP update of the field h
    h₊ᵗ⁺¹
    h₋ᵗ⁺¹

    # BP update of the messages χe and of the marginals χ
    χe₊ᵗ⁺¹
    χe₋ᵗ⁺¹
    χ₊ᵗ⁺¹
    χ₋ᵗ⁺¹

    # BP update of the SBM-to-GLM messages χl
    χl₊ᵗ⁺¹
    χl₋ᵗ⁺¹

    return nothing
end
