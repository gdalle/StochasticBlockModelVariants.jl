module StochasticBlockModelVariants

using Base
using Graphs: AbstractGraph
using LinearAlgebra: Symmetric, dot
using Random: AbstractRNG, default_rng
using SimpleWeightedGraphs: SimpleWeightedGraph
using SparseArrays: SparseMatrixCSC, sparse

export ContextualSBM, ContextualSBMLatents, ContextualSBMObservations

include("contextual_sbm.jl")

end
