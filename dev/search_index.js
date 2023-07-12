var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = StochasticBlockModelVariants","category":"page"},{"location":"#StochasticBlockModelVariants","page":"Home","title":"StochasticBlockModelVariants","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for StochasticBlockModelVariants.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [StochasticBlockModelVariants]","category":"page"},{"location":"#StochasticBlockModelVariants.AMPMarginals","page":"Home","title":"StochasticBlockModelVariants.AMPMarginals","text":"AMPMarginals\n\nFields\n\nû::Vector: posterior mean of u, length N\nv̂::Vector: posterior mean of v, length P\nχ₊e::Dict: messages about the marginal distribution of u, size (N, N)\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.AMPStorage","page":"Home","title":"StochasticBlockModelVariants.AMPStorage","text":"AMPStorage\n\nFields\n\nû_no_feat::Vector: posterior mean of u if there were no features, length N (aka Bᵥ)\nv̂_no_comm::Vector: posterior mean of v if there were no communities, length P (aka Bᵤ)\nh̃₊::Vector: individual external field for u=1, length N\nh̃₋::Vector: individual external field for u=-1, length N\nχ₊::Vector: marginal probability of u=1, length N\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.ContextualSBM","page":"Home","title":"StochasticBlockModelVariants.ContextualSBM","text":"ContextualSBM\n\nA generative model for graphs with node features, which combines the Stochastic Block Model with a mixture of Gaussians.\n\nReference: https://arxiv.org/abs/2306.07948\n\nFields\n\nN: graph size\nP: feature dimension\nd: average degree\nλ: SNR of the communities\nμ: SNR of the features\nρ: fraction of nodes revealed\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.ContextualSBMLatents","page":"Home","title":"StochasticBlockModelVariants.ContextualSBMLatents","text":"ContextualSBMLatents\n\nThe hidden variables generated by sampling from a ContextualSBM.\n\nFields\n\nu::Vector: community assignments, length N\nv::Vector: feature centroids, length P\n\n\n\n\n\n","category":"type"},{"location":"#StochasticBlockModelVariants.ContextualSBMObservations","page":"Home","title":"StochasticBlockModelVariants.ContextualSBMObservations","text":"ContextualSBMObservations\n\nThe observations generated by sampling from a ContextualSBM.\n\nFields\n\nA::AbstractMatrix: symmetric boolean adjacency matrix, size (N, N)\ng::AbstractGraph: undirected unweighted graph generated from A\nB::Matrix: feature matrix, size (P, N)\nΞ::Vector: revealed communities ±1 for a fraction of nodes and 0 for the rest, length N\n\n\n\n\n\n","category":"type"},{"location":"#Base.rand-Tuple{Random.AbstractRNG, ContextualSBM}","page":"Home","title":"Base.rand","text":"rand(rng, csbm)\n\nSample from a ContextualSBM and return a named tuple (; latents, observations).\n\n\n\n\n\n","category":"method"},{"location":"#StochasticBlockModelVariants.affinities-Tuple{ContextualSBM}","page":"Home","title":"StochasticBlockModelVariants.affinities","text":"affinities(csbm)\n\nReturn a named tuple (; cᵢ, cₒ) containing the affinities inside and outside of a community.\n\n\n\n\n\n","category":"method"},{"location":"#StochasticBlockModelVariants.effective_snr-Tuple{ContextualSBM}","page":"Home","title":"StochasticBlockModelVariants.effective_snr","text":"effective_snr(csbm)\n\nCompute the effective SNR λ² + μ² / (N/P).\n\n\n\n\n\n","category":"method"}]
}