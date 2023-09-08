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
using LinearAlgebra: LinearAlgebra, dot, mul!, norm, normalize!
using PrecompileTools: @compile_workload
using ProgressMeter: Progress, next!
using Random: AbstractRNG, default_rng
using SimpleWeightedGraphs: SimpleWeightedGraph
using SpecialFunctions: erf
using Statistics: mean, std
using SparseArrays: SparseMatrixCSC, sparse, findnz

export AbstractSBM
export nb_features, average_degree, communities_snr, affinities, features_snr, effective_snr
export CSBM, LatentsCSBM, ObservationsCSBM
export GLMSBM, LatentsGLMSBM, ObservationsGLMSBM
export GaussianWeightPrior, RademacherWeightPrior
export init_amp, update_amp!, run_amp, evaluate_amp
export discrete_overlap, continuous_overlap

include("utils.jl")
include("abstract_sbm.jl")
include("csbm.jl")
include("glmsbm.jl")
include("csbm_inference.jl")
include("glmsbm_inference.jl")

end
