using Test
using CVChannel

@testset "./src/CVChannel.jl" begin
    maxEntState = 0.5 * [1 0 0 1 ; 0 0 0 0 ; 0 0 0 0 ; 1 0 0 1]
    maxMixState = 0.25 * [1 0 0 0 ; 0 1 0 0 ; 0 0 1 0 ; 0 0 0 1]
    @testset "isPPT" begin
        @test isPPT(maxMixState,2,[2,2])
        @test !isPPT(maxEntState,2,[2,2])
    end
    @testset "EntropyTests" begin
        @testset "minEntropyPrimal" begin
            @test isapprox(minEntropyPrimal(maxMixState,2,2)[1] ,1/2, atol=1e-6)
            @test isapprox(minEntropyPrimal(maxEntState,2,2)[1], 2, atol = 1e-6)
            #We don't test second output on maxMix because the optimal optimizer set is too big
            @test isapprox(minEntropyPrimal(maxEntState,2,2)[2], 2*maxEntState, atol = 1e-6)
        end
        @testset "minEntropyDual" begin
            @test isapprox(minEntropyDual(maxMixState,2,2)[1] ,1/2, atol=1e-6)
            @test isapprox(minEntropyDual(maxEntState,2,2)[1], 2, atol = 1e-6)
            @test isapprox(minEntropyDual(maxMixState,2,2)[2],1/4*[1 0 ; 0 1], atol = 1e-6)
            @test isapprox(minEntropyDual(maxEntState,2,2)[2],[1 0 ; 0 1], atol = 1e-6)
        end
        @testset "minEntropyPPTPrimal" begin
            @test isapprox(minEntropyPPTPrimal(maxMixState,2,2)[1] ,1/2, atol=1e-6)
            @test isapprox(minEntropyPPTPrimal(maxEntState,2,2)[1], 1, atol = 1e-6)
            #Again we don't test second output on maxMix
            #One can determine the optimizer is what I give below, but the numerics gets only close-ish, hence atol is big
            @test isapprox(minEntropyPPTPrimal(maxEntState,2,2)[2],[2/3 0 0 1/3 ; 0 1/3 0 0 ; 0 0 1/3 0 ; 1/3 0 0 2/3], atol=10^(-2.4))
        end
        @testset "minEntropyPPTDual" begin
            @test isapprox(minEntropyPPTDual(maxMixState,2,2)[1] ,1/2, atol=1e-6)
            @test isapprox(minEntropyPPTDual(maxEntState,2,2)[1], 1, atol = 1e-6)
            @test isapprox(minEntropyPPTDual(maxMixState,2,2)[2],1/4*[1 0 ; 0 1], atol = 1e-6)
            @test isapprox(minEntropyPPTDual(maxEntState,2,2)[2],1/2*[1 0 ; 0 1], atol = 1e-6)
        end
    end
end
