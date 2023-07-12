sigmoid(x) = 1 / (1 + exp(-x))

count_equalities(x, y) = sum(x[i] ≈ y[i] for i in eachindex(x, y))
