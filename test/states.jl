using Test
using CVChannel

@testset "./src/states.jl" begin

@testset "wernerStates" begin
    @test isapprox(wernerState(2,1),[1/3 0 0 0 ; 0 1/6 1/6 0 ; 0 1/6 1/6 0 ; 0 0 0 1/3], atol = 1e-6)
    @test isapprox(wernerState(2,0),[0 0 0 0 ; 0 1/2 -1/2 0 ; 0 -1/2 1/2 0 ; 0 0 0 0], atol = 1e-6)
    @test isapprox(wernerState(2,1/2),[1/6 0 0 0 ; 0 1/3 -1/6 0 ; 0 -1/6 1/3 0 ; 0 0 0 1/6], atol = 1e-6)

    @testset "errors" begin
        @test_throws DomainError wernerState(1, 0.5)
        @test_throws DomainError wernerState(2, -.1)
        @test_throws DomainError wernerState(2, 1.1)
    end
end

end
