sigmoid(x) = 1 / (1 + exp(-x))

freq_equalities(x, y) = mean(x[i] ≈ y[i] for i in eachindex(x, y))

function overlaps(;
    u::Vector{<:Integer}, v::Vector{R}, û::Vector{R}, v̂::Vector{R}
) where {R}
    û .= sign.(û)
    û[abs.(û) .< eps(R)] .= one(R)

    q̂ᵤ = max(freq_equalities(û, u), freq_equalities(û, -u))
    qᵤ = 2 * (q̂ᵤ - one(R) / 2)

    q̂ᵥ = max(abs(dot(v̂, v)), abs(dot(v̂, -v)))
    qᵥ = q̂ᵥ / (eps(R) + norm(v̂) * norm(v))

    return (; qᵤ, qᵥ)
end

mutable struct PlusMinusMeasure{R}
    p₊::R
    p₋::R
end

PlusMinusMeasure(p) = PlusMinusMeasure(p, one(typeof(p)) - p)

Base.copy(pmm::PlusMinusMeasure) = PlusMinusMeasure(pmm.p₊, pmm.p₋)

function Base.getindex(pmm::PlusMinusMeasure, i)
    if i == 1
        return pmm.p₊
    else
        return pmm.p₋
    end
end

function Base.setindex!(pmm::PlusMinusMeasure, v, i)
    if i == 1
        pmm.p₊ = v
    else
        pmm.p₋ = v
    end
end

function LinearAlgebra.normalize!(pmm::PlusMinusMeasure)
    s = pmm.p₊ + pmm.p₋
    pmm.p₊ /= s
    return pmm.p₋ /= s
end
