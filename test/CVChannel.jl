using Test
using CVChannel

@testset "./src/CVChannel.jl" begin
    maxEntState = 0.5 * [1 0 0 1 ; 0 0 0 0 ; 0 0 0 0 ; 1 0 0 1]
    maxMixState = 0.25 * [1 0 0 0 ; 0 1 0 0 ; 0 0 1 0 ; 0 0 0 1]
    checkQubit = [2/3 1/4*1im ; -1/4*1im 1/3]
    function depolChan(X)
        return 1/2*[1 0 ; 0 1]
    end
    function identChan(X)
        return X
    end
    @testset "GeneralFunctions" begin
        @testset "isPPT" begin
            @test isPPT(maxMixState,2,[2,2])
            @test !isPPT(maxEntState,2,[2,2])
        end
        @testset "swapOperator" begin
            @test isapprox(swapOperator(2),[1 0 0 0 ; 0 0 1 0 ; 0 1 0 0 ; 0 0 0 1], atol = 1e-6)
            swap31 = [1 0 0 0 0 0 0 0 0 ; 0 0 0 1 0 0 0 0 0 ; 0 0 0 0 0 0 1 0 0]
            swap32 = [0 1 0 0 0 0 0 0 0 ; 0 0 0 0 1 0 0 0 0 ; 0 0 0 0 0 0 0 1 0]
            swap33 = [0 0 1 0 0 0 0 0 0 ; 0 0 0 0 0 1 0 0 0 ; 0 0 0 0 0 0 0 0 1]
            swap3 = [swap31;swap32;swap33]
            @test isapprox(swapOperator(3),swap3, atol =1e-6)
        end
        @testset "choi" begin
            @test isapprox(choi(identChan,2),2*maxEntState, atol = 1e-6)
            @test isapprox(choi(depolChan,2),1/2*[1 0 1 0 ; 0 1 0 1 ; 1 0 1 0 ; 0 1 0 1], atol = 1e-6)
        end
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
    @testset "ChannelTests" begin
        @testset "qDepolarizingChannel" begin
            @test isapprox(qDepolarizingChannel(maxEntState,0),maxEntState, atol=1e-6)
            @test isapprox(qDepolarizingChannel(maxEntState,1),maxMixState, atol=1e-6)
        end
        @testset "dephrasureChannel" begin
            @test isapprox(dephrasureChannel(checkQubit,0,0),[2/3 1/4*1im 0; -1/4*1im 1/3 0; 0 0 0], atol = 1e-6)
            @test isapprox(dephrasureChannel(checkQubit,1/2,0),[2/3 0 0; 0 1/3 0; 0 0 0], atol = 1e-6)
            @test isapprox(dephrasureChannel(checkQubit,0,1/2),[2/6 1/8*1im 0; -1/8*1im 1/6 0; 0 0 1/2], atol = 1e-6)
            @test isapprox(dephrasureChannel(checkQubit,1/2,1/2),[2/6 0 0; 0 1/6 0; 0 0 1/2], atol = 1e-6)
        end
        @testset "wernerHolevoChannel" begin
            @test isapprox(wernerHolevoChannel(checkQubit,1), (1/3*[5/3 -1/4*1im ; 1/4*1im 4/3]), atol = 1e-6)
            @test isapprox(wernerHolevoChannel(checkQubit,0), [1/3 1/4*1im ; -1/4*1im 2/3], atol = 1e-6)
            @test isapprox(wernerHolevoChannel(checkQubit,1/2), [4/9 1/12*1im ; -1/12*1im 5/9], atol = 1e-6)
        end
    end
    @testset "StateTests" begin
        @testset "wernerStates" begin
            @test isapprox(wernerState(2,1),[1/3 0 0 0 ; 0 1/6 1/6 0 ; 0 1/6 1/6 0 ; 0 0 0 1/3], atol = 1e-6)
            @test isapprox(wernerState(2,0),[0 0 0 0 ; 0 1/2 -1/2 0 ; 0 -1/2 1/2 0 ; 0 0 0 0], atol = 1e-6)
            @test isapprox(wernerState(2,1/2),[1/6 0 0 0 ; 0 1/3 -1/6 0 ; 0 -1/6 1/3 0 ; 0 0 0 1/6], atol = 1e-6)
        end
    end
end
