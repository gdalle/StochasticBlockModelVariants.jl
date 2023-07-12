module StochasticBlockModelVariants

using Graphs: AbstractGraph, neighbors
using LinearAlgebra: dot, mul!, norm
using PrecompileTools: @compile_workload
using ProgressMeter: Progress, next!
using Random: AbstractRNG, default_rng
using SimpleWeightedGraphs: SimpleWeightedGraph
using Statistics: mean, std
using SparseArrays: SparseMatrixCSC, sparse, findnz

export ContextualSBM, ContextualSBMLatents, ContextualSBMObservations
export affinities, effective_snr
export overlaps
export init_amp, update_amp!, run_amp, evaluate_amp

include("utils.jl")
include("csbm.jl")
include("csbm_inference.jl")

@compile_workload begin
    rng = default_rng()
    csbm = ContextualSBM(; N=10^2, P=10^2, d=5, λ=2, μ=2, ρ=0.1)
    (; observations) = rand(rng, csbm)
    run_amp(rng; observations, csbm, max_iterations=2)
end

end
