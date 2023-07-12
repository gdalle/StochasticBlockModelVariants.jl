using Base.Threads
using CairoMakie
using LinearAlgebra
using Random: default_rng
using StochasticBlockModelVariants
using ProgressMeter

BLAS.set_num_threads(1)
