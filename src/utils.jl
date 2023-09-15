
prior(::Type{R}, u, Ξᵢ) where {R} = ismissing(Ξᵢ) ? one(R) / 2 : R(Ξᵢ == u)

sigmoid(x) = 1 / (1 + exp(-x))

freq_equalities(x, y) = mean(x[i] ≈ y[i] for i in eachindex(x, y))

"""
    discrete_overlap(x_true, x_est)

Compute the alignment between two ±1-valued vectors, up to global sign switch.
"""
function discrete_overlap(u, û)
    R = eltype(û)
    û_sign = sign.(û)
    û_sign[abs.(û) .< eps()] .= 1
    q̂ᵤ = max(freq_equalities(u, û_sign), freq_equalities(u, -û_sign))
    qᵤ = 2 * (q̂ᵤ - one(R) / 2)
    return qᵤ
end

"""
    continuous_overlap(x_true, x_est)

Compute the alignment between two real-valued vectors, up to global sign switch.
"""
function continuous_overlap(v, v̂)
    R = eltype(v)
    q̂ᵥ = max(dot(v̂, v), dot(v̂, -v))
    qᵥ = q̂ᵥ / (eps(R) + norm(v̂) * norm(v))
    return qᵥ
end

function copy_damp!(dest::T, source::T; damping=0) where {T}
    for n in fieldnames(T)
        x_dest = getfield(dest, n)
        x_source = getfield(source, n)
        for k in eachindex(x_dest)
            x_dest[k] = x_source[k] * (1 - damping) + x_dest[k] * damping
        end
    end
end
