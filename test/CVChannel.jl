using Test
using CVChannel

@testset "./src/CVChannel.jl" begin
    maxEntState = 0.5 * [1 0 0 1 ; 0 0 0 0 ; 0 0 0 0 ; 1 0 0 1]
    maxMixState = 0.25 * [1 0 0 0 ; 0 1 0 0 ; 0 0 1 0 ; 0 0 0 1]
    @test isPPT(maxMixState,2,[2,2])
    @test !isPPT(maxEntState,2,[2,2])
    @test (abs(minEnt(maxMixState,2,2)[1]-1/2) <= 1e-6) == true
    @test (abs(minEnt(maxMixState,2,2,false)[1]-1/2) <= 1e-6) == true
    @test (abs(minEnt(maxEntState,2,2)[1]- 2) <= 1e-6) == true
    @test (abs(minEntPPT(maxMixState,2,2,false)[1]-1/2) <= 1e-6) == true
    @test (abs(minEntPPT(maxEntState,2,2,false)[1]-1) <= 1e-6) == true
end
