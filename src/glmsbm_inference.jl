## Marginals and storage

"""
    AMPMarginalsGLMSBM

# Fields

"""
@kwdef struct AMPMarginalsGLMSBM{R<:Real}
    a::Vector{R}
    v::Vector{R}
    ω::Vector{R}
    Γ::Vector{R}
    gₒ::Vector{R}
    ψ₊l::Vector{R}
    χ₊l::Vector{R}
    χ₊e::Dict{Tuple{Int,Int},R}
    χ₊::Dict{Tuple{Int,Int},R}
end

function init_amp(
    rng::AbstractRNG; observations::ObservationsGLMSBM{R}, glmsbm::GLMSBM{R}, init_std
) where {R}
    (; N, M) = glmsbm
    (; g, Ξ) = observations

    a = init_std .* randn(rng, R, M)
    v = ones(R, M)
    
    gₒ = zeros(R, N)
    
    χ₊e = Dict{Tuple{Int,Int},R}()
    for μ in 1:N, ν in neighbors(g, μ)

    end

    return (; marginals, next_marginals)
end
