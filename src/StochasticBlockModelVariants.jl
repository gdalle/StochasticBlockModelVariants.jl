module StochasticBlockModelVariants

using Base: RefValue
using Graphs: AbstractGraph, has_edge, neighbors
using LinearAlgebra: Symmetric, dot
using ProgressMeter: @showprogress
using Random: AbstractRNG, default_rng
using SimpleWeightedGraphs: SimpleWeightedGraph, SimpleWeightedDiGraph
using Statistics: mean
using SparseArrays: SparseMatrixCSC, sparse, findnz

export ContextualSBM, ContextualSBMLatents, ContextualSBMObservations
export affinities, effective_snr
export init_amp, update_amp!, run_amp, evaluate_amp

include("utils.jl")
include("contextual_sbm.jl")

end
