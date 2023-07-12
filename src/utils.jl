sigmoid(x) = 1 / (1 + exp(-x))

freq_equalities(x, y) = mean(x[i] ≈ y[i] for i in eachindex(x, y))
