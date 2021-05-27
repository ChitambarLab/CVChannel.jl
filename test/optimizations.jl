using Test
using CVChannel

@testset "./src/optimizations.jl" begin

maxEntState = 0.5 * [1 0 0 1 ; 0 0 0 0 ; 0 0 0 0 ; 1 0 0 1]
maxMixState = 0.25 * [1 0 0 0 ; 0 1 0 0 ; 0 0 1 0 ; 0 0 0 1]

@testset "eaCVPrimal" begin
    @test isapprox(eaCVPrimal(maxMixState,2,2)[1] ,1/2, atol=1e-6)
    @test isapprox(eaCVPrimal(maxEntState,2,2)[1], 2, atol = 1e-6)
    #We don't test second output on maxMix because the optimal optimizer set is too big
    @test isapprox(eaCVPrimal(maxEntState,2,2)[2], 2*maxEntState, atol = 1e-6)
end

@testset "eaCVDual" begin
    @test isapprox(eaCVDual(maxMixState,2,2)[1] ,1/2, atol=1e-6)
    @test isapprox(eaCVDual(maxEntState,2,2)[1], 2, atol = 1e-6)
    @test isapprox(eaCVDual(maxMixState,2,2)[2],1/4*[1 0 ; 0 1], atol = 1e-6)
    @test isapprox(eaCVDual(maxEntState,2,2)[2],[1 0 ; 0 1], atol = 1e-6)
end

@testset "pptCVPrimal" begin
    @test isapprox(pptCVPrimal(maxMixState,2,2)[1] ,1/2, atol=1e-6)
    @test isapprox(pptCVPrimal(maxEntState,2,2)[1], 1, atol = 1e-6)
    #Again we don't test second output on maxMix
    #One can determine the optimizer is what I give below, but the numerics gets only close-ish, hence atol is big
    @test isapprox(pptCVPrimal(maxEntState,2,2)[2],[2/3 0 0 1/3 ; 0 1/3 0 0 ; 0 0 1/3 0 ; 1/3 0 0 2/3], atol=10^(-2.4))
end

@testset "pptCVDual" begin
    @test isapprox(pptCVDual(maxMixState,2,2)[1] ,1/2, atol=1e-6)
    @test isapprox(pptCVDual(maxEntState,2,2)[1], 1, atol = 1e-6)
    @test isapprox(pptCVDual(maxMixState,2,2)[2],1/4*[1 0 ; 0 1], atol = 1e-6)
    @test isapprox(pptCVDual(maxEntState,2,2)[2],1/2*[1 0 ; 0 1], atol = 1e-6)
end
end
