using Test
using CVChannel

@testset "./src/channels.jl" begin

maxEntState = 0.5 * [1 0 0 1 ; 0 0 0 0 ; 0 0 0 0 ; 1 0 0 1]
maxMixState = 0.25 * [1 0 0 0 ; 0 1 0 0 ; 0 0 1 0 ; 0 0 0 1]

@testset "choi" begin
    depolChan(X) = 1/2*[1 0 ; 0 1]
    identChan(X) = X
    @test isapprox(choi(identChan,2,2),2*maxEntState, atol = 1e-6)
    @test isapprox(choi(depolChan,2,2),1/2*[1 0 1 0 ; 0 1 0 1 ; 1 0 1 0 ; 0 1 0 1], atol = 1e-6)
end

@testset "depolarizingChannel" begin
    @test isapprox(depolarizingChannel(maxEntState,0),maxEntState, atol=1e-6)
    @test isapprox(depolarizingChannel(maxEntState,1),maxMixState, atol=1e-6)

    @testset "errors" begin
        @test_throws DomainError depolarizingChannel(maxEntState,1.1)
        @test_throws DomainError depolarizingChannel(maxEntState,-.1)
        @test_throws DomainError depolarizingChannel([1 0 0;0 0 0],0.5)
    end
end

@testset "dephrasureChannel" begin
    checkQubit = [2/3 1/4*1im ; -1/4*1im 1/3]
    @test isapprox(dephrasureChannel(checkQubit,0,0),[2/3 1/4*1im 0; -1/4*1im 1/3 0; 0 0 0], atol = 1e-6)
    @test isapprox(dephrasureChannel(checkQubit,1/2,0),[2/3 0 0; 0 1/3 0; 0 0 0], atol = 1e-6)
    @test isapprox(dephrasureChannel(checkQubit,0,1/2),[2/6 1/8*1im 0; -1/8*1im 1/6 0; 0 0 1/2], atol = 1e-6)
    @test isapprox(dephrasureChannel(checkQubit,1/2,1/2),[2/6 0 0; 0 1/6 0; 0 0 1/2], atol = 1e-6)

    @testset "errors" begin
        @test_throws DomainError dephrasureChannel([1 0 0;0 0 0;0 0 0], 1, 1)
        @test_throws DomainError dephrasureChannel([1 0;0 0], 1.1, 1)
        @test_throws DomainError dephrasureChannel([1 0;0 0], -.1, 1)
        @test_throws DomainError dephrasureChannel([1 0;0 0], 1, 1.1)
        @test_throws DomainError dephrasureChannel([1 0;0 0], 1, -.1)
    end
end

@testset "wernerHolevoChannel" begin
    checkQubit = [2/3 1/4*1im ; -1/4*1im 1/3]
    @test isapprox(wernerHolevoChannel(checkQubit,1), (1/3*[5/3 -1/4*1im ; 1/4*1im 4/3]), atol = 1e-6)
    @test isapprox(wernerHolevoChannel(checkQubit,0), [1/3 1/4*1im ; -1/4*1im 2/3], atol = 1e-6)
    @test isapprox(wernerHolevoChannel(checkQubit,1/2), [4/9 1/12*1im ; -1/12*1im 5/9], atol = 1e-6)

    @testset "errors" begin
        @test_throws DomainError wernerHolevoChannel([1 0;0 0;0 0], 1)
        @test_throws DomainError wernerHolevoChannel([1 0;0 0], 1.1)
        @test_throws DomainError wernerHolevoChannel([1 0;0 0],-.1)
    end
end

end
