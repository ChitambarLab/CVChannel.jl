using Test
using CVChannel

@testset "./src/CVChannel.jl" begin
    @test isPPT([1 0 0 0 ; 0  1 0 0 ; 0 0 1 0 ; 0 0 0 1],2,[2,2])
    @test !isPPT([1 0 0 1 ; 0 0 0 0; 0 0 0 0 ; 1 0 0 1],2,[2,2])
end
