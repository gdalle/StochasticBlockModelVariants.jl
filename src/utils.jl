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
