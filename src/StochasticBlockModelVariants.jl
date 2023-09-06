"""
    StochasticBlockModelVariants

A package for inference in SBMs with node features using message-passing algorithms.

# Exports

$(EXPORTS)
"""
module StochasticBlockModelVariants

using DensityInterface: logdensityof, densityof
using DocStringExtensions
using Graphs: AbstractGraph, neighbors
using Infiltrator
using LinearAlgebra: LinearAlgebra, dot, mul!, norm, normalize!
using PrecompileTools: @compile_workload
using ProgressMeter: Progress, next!
using Random: AbstractRNG, default_rng
using SimpleWeightedGraphs: SimpleWeightedGraph
using SpecialFunctions: erf
using Statistics: mean, std
using SparseArrays: SparseMatrixCSC, sparse, findnz

export nb_features, average_degree, communities_snr, affinities, features_snr, effective_snr
export CSBM, LatentsCSBM, ObservationsCSBM
export GLMSBM, LatentsGLMSBM, ObservationsGLMSBM
export overlaps
export init_amp, update_amp!, run_amp, evaluate_amp

include("abstract_sbm.jl")
include("csbm.jl")
include("glmsbm.jl")
include("csbm_inference.jl")
include("glmsbm_inference.jl")

# @compile_workload begin
#     rng = default_rng()
#     csbm = CSBM(; N=10^2, P=10^2, d=5, λ=2, μ=2, ρ=0.1)
#     (; observations) = rand(rng, csbm)
#     run_amp(rng; observations, csbm, max_iterations=2)
# end

end
