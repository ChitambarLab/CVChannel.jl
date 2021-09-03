using Test, LinearAlgebra
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

@testset "is_choi_matrix" begin
    @test is_choi_matrix(Matrix(I, 6, 6), 2, 3)
    @test !is_choi_matrix(Matrix(I, 4, 4), 2, 3)
end

@testset "Choi" begin
    @testset "function instantiation" begin
        depolChan(X) = 1/2*[1 0; 0 1]
        depolChoi = Choi(depolChan, 2, 2)

        @test depolChoi isa Choi{Float64}
        @test depolChoi.JN isa Matrix{Float64}
        @test depolChoi.in_dim == 2
        @test depolChoi.out_dim == 2
    end

    @testset "matrix instantiation" begin
        JN = [1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1]
        choi_channel = Choi(JN, 2, 2)

        @test choi_channel isa Choi{Int}
        @test choi_channel.JN isa Matrix{Int}
        @test choi_channel.JN == JN
        @test choi_channel.in_dim == 2
        @test choi_channel.out_dim == 2
    end

    @testset "DomainError" begin
        @test_throws DomainError Choi(Matrix(I, 4, 4), 2, 3)
    end
end

@testset "parChoi" begin
    @testset "subsystem system swap" begin
        chan = Choi(2*maxEntState, 2, 2)
        par_chan = parChoi(chan, chan)

        @test par_chan isa Choi{Float64}
        @test par_chan.in_dim == 4
        @test par_chan.out_dim == 4

        @test par_chan.JN == [
            1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
            1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1
        ]
    end

    @testset "subsystem dims" begin
        dephrasure_chan(ρ) = dephrasureChannel(ρ, 0 ,0)
        identity_chan(ρ) = ρ

        deph_chan = Choi(dephrasure_chan, 2,3)
        id_chan = Choi(identity_chan, 2, 2)

        par_choi = parChoi(deph_chan, id_chan)

        @test par_choi isa Choi{Complex{Float64}}
        @test par_choi.in_dim == 4
        @test par_choi.out_dim == 6
        @test size(par_choi.JN) == (24,24)
    end
end

@testset "isometricChannel" begin
    kraus_set1 = Vector{Matrix{Complex{Float64}}}(undef, 2)
    kraus_set1[1] = [1 0 ; 0 0 ; 0 0]
    kraus_set1[2] = 1/sqrt(2)*[0 0 ; 0 1 ; 0 1]
    @test isapprox(isometricChannel(kraus_set1),[1 0 ; 0 0 ; 0 0 ; 0 1/sqrt(2) ; 0 0 ; 0 1/sqrt(2)], atol=1e-6)

    kraus_set2 = Vector{Matrix{Complex{Float64}}}(undef, 1)
    kraus_set2[1] = [1 0 ; 0 -1]
    @test isapprox(isometricChannel(kraus_set2),[1 0 ; 0 -1], atol=1e-6)
end

@testset "complementaryChannel" begin
    @testset "First Map" begin
        kraus_set1 = Vector{Matrix{Complex{Float64}}}(undef, 2)
        kraus_set1[1] = [1. 0 ; 0 0 ; 0 0]
        kraus_set1[2] = 1/sqrt(2)*[0 0 ; 0 1 ; 0 1]
        comp_kraus = complementaryChannel(kraus_set1)
        @test comp_kraus[1] == [1. 0 ; 0 0]
        @test comp_kraus[2] == 1/sqrt(2)*[0 0; 0 1]
        @test comp_kraus[3] == 1/sqrt(2)*[0 0; 0 1]
    end
    @testset "Unitary Map" begin
        kraus_set2 = Vector{Matrix{Complex{Float64}}}(undef, 1)
        kraus_set2[1] = [1 0 ; 0 -1]
        comp_kraus = complementaryChannel(kraus_set2)
        @test comp_kraus[1] == [1 0 ; 0 0]
        @test comp_kraus[2] == [0 1 ; 0 0]
    end
end

@testset "krausAction" begin
    #By linearity this should suffice for checking it
    #works on multiple dimensions
    ρ1 = [1 0 ; 0 0]
    ρ2 = [0 0 ; 0 1]
    ρ3 = [1 0 0 ; 0 0 0 ; 0 0 0]
    ρ4 = [0 0 0 ; 0 1 0 ; 0 0 0]
    ρ5 = [0 0 0 ; 0 0 0 ; 0 0 1]
    @testset "Identity Channel" begin
        kraus_set1 = Vector{Matrix{Complex{Float64}}}(undef, 1)
        kraus_set1[1] = [1 0 ; 0 1]
        @test krausAction(kraus_set1,ρ1) == ρ1
        @test krausAction(kraus_set1,ρ2) == ρ2
        kraus_set1 = Vector{Matrix{Complex{Float64}}}(undef, 1)
        kraus_set1[1] = [1 0 0 ; 0 1 0 ; 0 0 1]
        @test krausAction(kraus_set1,ρ3) == ρ3
        @test krausAction(kraus_set1,ρ4) == ρ4
        @test krausAction(kraus_set1,ρ5) == ρ5
    end
    @testset "Completely Depolarizing Channel" begin
        #This uses that applying the Discrete-Weyl operators uniformly
        #to a qudit turns it into the maximally mixed state
        kraus_set1 = Vector{Matrix{Complex{Float64}}}(undef, 4)
        kraus_set1[1] = 1/sqrt(4)*discreteWeylOperator(0,0,2)
        kraus_set1[2] = 1/sqrt(4)*discreteWeylOperator(0,1,2)
        kraus_set1[3] = 1/sqrt(4)*discreteWeylOperator(1,0,2)
        kraus_set1[4] = 1/sqrt(4)*discreteWeylOperator(1,1,2)
        @test krausAction(kraus_set1,ρ1) == 1/2*[1 0 ; 0 1]
        @test krausAction(kraus_set1,ρ2) == 1/2*[1 0 ; 0 1]
        kraus_set1 = Vector{Matrix{Complex{Float64}}}(undef, 9)
        kraus_ctr = 1
        for i = 0:2;
            for j = 0:2;
                kraus_set1[kraus_ctr] = 1/sqrt(9)*discreteWeylOperator(i,j,3)
                kraus_ctr += 1
            end
        end
        @test isapprox(krausAction(kraus_set1,ρ3),1/3*[1 0 0; 0 1 0 ; 0 0 1], atol = 1e-6)
        @test isapprox(krausAction(kraus_set1,ρ4),1/3*[1 0 0; 0 1 0 ; 0 0 1], atol = 1e-6)
        @test isapprox(krausAction(kraus_set1,ρ5),1/3*[1 0 0; 0 1 0 ; 0 0 1], atol = 1e-6)
    end
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

@testset "siddhuChannel" begin
    @testset "verify channel definition" begin
        for s in [0:0.1:0.5;]
            sidchan(X) = siddhuChannel(X,s)
            testchan = Choi(sidchan,3,3)
            α = 1-s
            γ = sqrt(s)
            β = sqrt(1-s)
            @test isapprox(testchan.JN,
                [s 0 0 0 0 γ 0 0 0;
                0 α 0 0 0 0 0 0 β;
                0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 ;
                γ 0 0 0 0 1 0 0 0 ;
                0 0 0 0 0 0 0 0 0 ;
                0 0 0 0 0 0 0 0 0 ;
                0 β 0 0 0 0 0 0 1
                ],
                atol = 1e-6
                )
        end
    end

    @testset "errors" begin
        @test_throws DomainError siddhuChannel([1 0 ; 0 0; 0 0], 0.2)
        @test_throws DomainError siddhuChannel([1 0 ; 0 0], 0.3)
        @test_throws DomainError siddhuChannel([1 0 0 ; 0 0 0 ; 0 0 0], 7)
    end
end

@testset "generalizedSiddhu" begin
    @testset "verify channel definition" begin
        for s in [0:0.1:0.5;]
            for μ in [0:0.1:1;]
                genSidChan(X) = generalizedSiddhu(X,s,μ)
                testchan = Choi(genSidChan,3,3)
                α, β, γ, δ = 1-s, 1-μ, sqrt(s*(1-μ)), sqrt(s*μ)
                ϵ, ζ, η = sqrt(μ*(1-s)), sqrt((1-μ)*(1-s)), sqrt(μ*(1-μ))
                target = [s 0 0 0 γ 0 0 0 δ ;
                          0 α 0 0 0 ϵ ζ 0 0;
                          0 0 0 0 0 0 0 0 0 ;
                          0 0 0 0 0 0 0 0 0 ;
                          γ 0 0 0 β 0 0 0 η ;
                          0 ϵ 0 0 0 μ η 0 0 ;
                          0 ζ 0 0 0 η β 0 0 ;
                          0 0 0 0 0 0 0 0 0 ;
                          δ 0 0 0 η 0 0 0 μ]
                @test isapprox(testchan.JN, target, atol = 1e-6)
            end
        end
    end
    @testset "errors" begin
        @test_throws DomainError generalizedSiddhu([1 0 ; 0 0; 0 0], 0.2,1)
        @test_throws DomainError generalizedSiddhu([1 0 ; 0 0], 0.3,1)
        @test_throws DomainError generalizedSiddhu([1 0 0 ; 0 0 0 ; 0 0 0], 7,1)
        @test_throws DomainError generalizedSiddhu([1 0 0 ; 0 0 0 ; 0 0 0], 0.2,7)
    end
end

@testset "GADChannel" begin
    @testset "Verify Channel Definition" begin
        #By linearity we just need to check the computational basis
        zero_state, one_state = [1 0 ; 0 0], [0 0 ; 0 1]
        scan_range = [0:0.1:1;]
        for p in scan_range
            for n in scan_range
                @test isapprox(GADChannel(zero_state,p,n),[1-p*n 0 ; 0 p*n], atol = 1e-6)
                @test isapprox(GADChannel(one_state,p,n),[(1-n)*p 0 ; 0 1-p+p*n], atol = 1e-6)
            end
        end
    end

    @testset "errors" begin
        @test_throws DomainError GADChannel([1 0 ; 0 0; 0 0], 0.2,0.4)
        @test_throws DomainError GADChannel([1 0 0 ; 0 0 0; 0 0 0], 0.4, 0.2)
        @test_throws DomainError GADChannel([1 0 0 ; 0 0 0 ; 0 0 0], 0.3,1.2)
        @test_throws DomainError GADChannel([1 0 0 ; 0 0 0 ; 0 0 0], 1.5,0.1)
    end
end
end
