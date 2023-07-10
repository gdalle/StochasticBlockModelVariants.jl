module StochasticBlockModelVariants

using SimpleWeightedGraphs: SimpleWeightedGraph
using LinearAlgebra: Symmetric
using Random: AbstractRNG, default_rng
using SparseArrays: SparseMatrixCSC, sparse

export ContextualSBM, ContextualSBMLatents, ContextualSBMObservations

include("contextual_sbm.jl")

end
