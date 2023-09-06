var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = StochasticBlockModelVariants","category":"page"},{"location":"#StochasticBlockModelVariants","page":"Home","title":"StochasticBlockModelVariants","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for StochasticBlockModelVariants.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [StochasticBlockModelVariants]","category":"page"},{"location":"#StochasticBlockModelVariants.StochasticBlockModelVariants","page":"Home","title":"StochasticBlockModelVariants.StochasticBlockModelVariants","text":"StochasticBlockModelVariants\n\nA package for inference in SBMs with node features using message-passing algorithms.\n\nExports\n\nAbstractSBM\nCSBM\nGLMSBM\nGaussianWeightPrior\nLatentsCSBM\nLatentsGLMSBM\nObservationsCSBM\nObservationsGLMSBM\nRademacherWeightPrior\naffinities\naverage_degree\ncommunities_snr\neffective_snr\nevaluate_amp\nfeatures_snr\ninit_amp\nnb_features\nrun_amp\nupdate_amp!\n\n\n\n\n\n","category":"module"},{"location":"#StochasticBlockModelVariants.AbstractSBM","page":"Home","title":"StochasticBlockModelVariants.AbstractSBM","text":"abstract type AbstractSBM\n\nAbstract supertype for Stochastic Block Models with additional node features.\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.CSBM","page":"Home","title":"StochasticBlockModelVariants.CSBM","text":"struct CSBM{R} <: AbstractSBM\n\nA generative model for graphs with node features, which combines a Stochastic Block Model with a mixture of Gaussians.\n\nReference: https://arxiv.org/abs/2306.07948\n\nFields\n\nN::Int64: graph size\nP::Int64: feature dimension\nd::Any: average degree\nλ::Any: SNR of the communities\nμ::Any: SNR of the features\nρ::Any: fraction of node assignments observed\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.GLMSBM","page":"Home","title":"StochasticBlockModelVariants.GLMSBM","text":"struct GLMSBM{R<:Real, D} <: AbstractSBM\n\nA generative model for graphs with node features, which combines a Generalized Linear Model with a Stochastic Block Model.\n\nReference: https://arxiv.org/abs/2303.09995\n\nFields\n\nN::Int64: graph size\nM::Int64: feature dimension\nc::Real: average degree\nλ::Real: SNR of the communities\nρ::Real: fraction of nodes revealed\nPʷ::Any: prior on the weights\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.LatentsCSBM","page":"Home","title":"StochasticBlockModelVariants.LatentsCSBM","text":"struct LatentsCSBM{R<:Real}\n\nThe hidden variables generated by sampling from a CSBM.\n\nFields\n\nu::Vector{Int64}: community assignments, length N\nv::Vector{R} where R<:Real: feature centroids, length P\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.LatentsGLMSBM","page":"Home","title":"StochasticBlockModelVariants.LatentsGLMSBM","text":"struct LatentsGLMSBM{R<:Real}\n\nThe hidden variables generated by sampling from a GLMSBM.\n\nFields\n\ns::Vector{Int64}: community assignments, length N\nw::Vector{R} where R<:Real: feature weights, length M\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.MarginalsCSBM","page":"Home","title":"StochasticBlockModelVariants.MarginalsCSBM","text":"struct MarginalsCSBM{R}\n\nFields\n\nû::Vector: posterior mean of u, length N\nv̂::Vector: posterior mean of v, length P\nû_no_feat::Vector: posterior mean of u if there were no features, length N (aka Bᵥ)\nv̂_no_comm::Vector: posterior mean of v if there were no communities, length P (aka Bᵤ)\nh̃₊::Vector: individual external field for u=1, length N\nh̃₋::Vector: individual external field for u=-1, length N\nχe₊::Dict{Tuple{Int64, Int64}}: messages about the marginal distribution of u, size (N, N)\nχ₊::Vector: marginal probability of u=1, length N\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.MarginalsGLMSBM","page":"Home","title":"StochasticBlockModelVariants.MarginalsGLMSBM","text":"struct MarginalsGLMSBM{R<:Real}\n\nFields\n\nŝ::Vector{R} where R<:Real: length N\nŵ::Vector{R} where R<:Real: length M\nv::Vector{R} where R<:Real: length M, alias a\nΓ::Vector{R} where R<:Real: length M\nω::Vector{R} where R<:Real: length N\ngₒ::Vector{R} where R<:Real: length N\nψl::Matrix{R} where R<:Real: size (2, N)\nχe::Dict{Tuple{Int64, Int64, Int64}, R} where R<:Real: size (2, N, N)\nχl::Matrix{R} where R<:Real: size (2, N)\nχ::Matrix{R} where R<:Real: size (2, N)\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.ObservationsCSBM","page":"Home","title":"StochasticBlockModelVariants.ObservationsCSBM","text":"struct ObservationsCSBM{R<:Real, G<:Graphs.AbstractGraph{Int64}}\n\nThe observations generated by sampling from a CSBM.\n\nFields\n\ng::Graphs.AbstractGraph{Int64}: undirected unweighted graph with N nodes (~ adjacency matrix A)\nΞ::Vector{Union{Missing, Int64}}: revealed communities ±1 for a fraction ρ of nodes and missing for the rest, length N\nB::Matrix{R} where R<:Real: feature matrix, size (P, N)\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.ObservationsGLMSBM","page":"Home","title":"StochasticBlockModelVariants.ObservationsGLMSBM","text":"struct ObservationsGLMSBM{R<:Real, G<:Graphs.AbstractGraph{Int64}}\n\nThe observations generated by sampling from a GLMSBM.\n\nFields\n\ng::Graphs.AbstractGraph{Int64}: undirected unweighted graph with N nodes (~ adjacency matrix A)\nΞ::Vector{Union{Missing, Int64}}: revealed communities ±1 for a fraction ρ of nodes and missing for the rest, ngth N\nF::Matrix{R} where R<:Real: feature matrix, size (N, M)\n\n\n\n\n\n","category":"type"},{"location":"#Base.length","page":"Home","title":"Base.length","text":"length(sbm::AbstractSBM)\n\nReturn the number of nodes N in the graph.\n\n\n\n\n\n","category":"function"},{"location":"#Base.rand-Tuple{Random.AbstractRNG, CSBM}","page":"Home","title":"Base.rand","text":"rand(rng, csbm)\n\nSample from a CSBM and return a named tuple (; latents, observations).\n\n\n\n\n\n","category":"method"},{"location":"#Base.rand-Tuple{Random.AbstractRNG, GLMSBM}","page":"Home","title":"Base.rand","text":"rand(rng, glmsbm)\n\nSample from a GLMSBM and return a named tuple (; latents, observations).\n\n\n\n\n\n","category":"method"},{"location":"#StochasticBlockModelVariants.affinities-Tuple{AbstractSBM}","page":"Home","title":"StochasticBlockModelVariants.affinities","text":"affinities(sbm)\n\nReturn a named tuple (; cᵢ, cₒ) containing the affinities inside and outside of a community.\n\n\n\n\n\n","category":"method"},{"location":"#StochasticBlockModelVariants.average_degree","page":"Home","title":"StochasticBlockModelVariants.average_degree","text":"average_degree(sbm::AbstractSBM)\n\nReturn the average degre d of a node in the graph.\n\n\n\n\n\n","category":"function"},{"location":"#StochasticBlockModelVariants.communities_snr","page":"Home","title":"StochasticBlockModelVariants.communities_snr","text":"communities_snr(sbm::AbstractSBM)\n\nReturn the signal-to-noise ratio λ of the communities in the graph.\n\n\n\n\n\n","category":"function"},{"location":"#StochasticBlockModelVariants.effective_snr-Tuple{CSBM}","page":"Home","title":"StochasticBlockModelVariants.effective_snr","text":"effective_snr(csbm)\n\nCompute the effective SNR λ² + μ² / (N/P).\n\n\n\n\n\n","category":"method"},{"location":"#StochasticBlockModelVariants.fraction_observed","page":"Home","title":"StochasticBlockModelVariants.fraction_observed","text":"fraction_observed(sbm)\n\nReturn the fraction ρ of community assignments that are observed.\n\n\n\n\n\n","category":"function"},{"location":"#StochasticBlockModelVariants.nb_features","page":"Home","title":"StochasticBlockModelVariants.nb_features","text":"nb_features(sbm::AbstractSBM)\n\nReturn the number of nodes N in the graph.\n\n\n\n\n\n","category":"function"},{"location":"#StochasticBlockModelVariants.sample_graph-Tuple{Random.AbstractRNG, AbstractSBM, Vector{<:Integer}}","page":"Home","title":"StochasticBlockModelVariants.sample_graph","text":"sample_graph(rng, sbm, communities)\n\nSample a graph g from an SBM based on known community assignments.\n\n\n\n\n\n","category":"method"},{"location":"#StochasticBlockModelVariants.sample_mask-Tuple{Random.AbstractRNG, AbstractSBM, Vector{<:Integer}}","page":"Home","title":"StochasticBlockModelVariants.sample_mask","text":"sample_mask(rng, sbm, communities)\n\nSample a vector Ξ (Xi) whose components are equal to the community assignments with probability ρ and equal to missing with probability 1-ρ. \n\n\n\n\n\n","category":"method"}]
}
