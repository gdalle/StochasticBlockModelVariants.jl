module StochasticBlockModelVariants

using Graphs: AbstractGraph, neighbors
using LinearAlgebra: dot, mul!
using ProgressMeter: @showprogress
using Random: AbstractRNG, default_rng
using SimpleWeightedGraphs: SimpleWeightedGraph
using Statistics: mean
using SparseArrays: SparseMatrixCSC, sparse, findnz

export ContextualSBM, ContextualSBMLatents, ContextualSBMObservations
export affinities, effective_snr
export init_amp, update_amp!, run_amp, evaluate_amp

include("utils.jl")
include("csbm.jl")
include("csbm_inference.jl")

end
