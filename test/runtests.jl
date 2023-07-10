using Aqua
using Documenter
using JET
using JuliaFormatter
using StochasticBlockModelVariants
using Test

@testset verbose = true "StochasticBlockModelVariants.jl" begin
    @testset "Code quality" begin
        if VERSION >= v"1.9"
            Aqua.test_all(StochasticBlockModelVariants; ambiguities=false)
        end
    end

    @testset "Code formatting" begin
        @test JuliaFormatter.format(
            StochasticBlockModelVariants; verbose=false, overwrite=false
        )
    end

    @testset "Code linting" begin
        if VERSION >= v"1.9"
            JET.test_package(StochasticBlockModelVariants; target_defined_modules=true)
        end
    end

    @testset "Doctests" begin
        doctest(StochasticBlockModelVariants)
    end
end
