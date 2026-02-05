using Test
using Lux, ComponentArrays, Random

include("../src/TumorModels.jl")
using .TumorModels

@testset "UDE Architecture Tests" begin
    # Test 1: Dimensions
    model = create_ude_architecture()
    rng = Random.default_rng()
    p, st = Lux.setup(rng, model)
    
    x_input = [0.5, 1.0] # Tumor=0.5, Drug=1.0
    y_output, st_new = model(x_input, p, st)
    
    @test length(y_output) == 1
    @test isa(y_output[1], Float32) || isa(y_output[1], Float64)
    
    println("Architecture Dimension Check: PASSED")
end

@testset "Drug Concentration Physics" begin
    # Check pulse logic
    @test TumorModels.drug_concentration(0.0) == 0.0
    @test TumorModels.drug_concentration(10.0) > 0.0
    println("Physics Check: PASSED")
end