@compile_workload begin
    rng = default_rng()
    for sbm in [
        CSBM(; N=10^2, P=10^2, d=5, λ=2, μ=2, ρ=0.0),
        GLMSBM(; N=10^2, M=10^2, c=5, λ=2, ρ=0.0, Pʷ=GaussianWeightPrior()),
        GLMSBM(; N=10^2, M=10^2, c=5, λ=2, ρ=0.1, Pʷ=RademacherWeightPrior()),
    ]
        evaluate_amp(rng, sbm)
    end
end
