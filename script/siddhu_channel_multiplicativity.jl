using LinearAlgebra
using CVChannel
using Convex
using Test
using DelimitedFiles

"""
This script invesetigates the channel which we call
the Siddhu channel. It is defined in (9) of
https://arxiv.org/pdf/2003.10367.pdf. It is
parameterized by s∈[0,1/2]
"""

println("First we define the channel function.")
function siddhuChannel(ρ :: Matrix{<:Number}, s :: Union{Int,Float64})
    K0 = [sqrt(s) 0 0 ; 0 0 0 ; 0 1 0]
    K1 = [0 0 0 ; sqrt(1-s) 0 0 ; 0 0 1]
    return K0*ρ*K0' + K1*ρ*K1'
end

@testset "verify Siddhu channel" begin
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


println("Next we look at the parallel communication value")
println("Since we know cv(N) is at least 2 always, all we need to do is check that cvPPT(N ⊗ N) = 4 always.")
@testset "Verify that cv is 2 and cv(N ⊗ N) is multiplicative"
    scan_range =[0:0.01:0.5;]
    is_mult = true
    ctr = 1
    for s_id in scan_range
        ctr % 10 == 0 ? println("Now on ", ctr, " of ", length(scan_range), " points.") : nothing
        #To save time I don't check the single copy cv
        # parallel channel uses
        sidchan(X) = siddhuChannel(X,s)
        sid_chan= Choi(sidchan,3,3)
        par_dims = [sid_chan.in_dim, sid_chan.out_dim, sid_chan.in_dim, sid_chan.out_dim]
        par_JN = permuteSubsystems(kron(sid_chan.JN, sid_chan.JN), [1,3,2,4], par_dims)
        par_in_dim = sid_chan.in_dim * sid_chan.in_dim
        par_out_dim = sid_chan.out_dim * sid_chan.out_dim
        par_cv, = pptCVDual(par_JN, 9, 9)
        !isapprox(par_cv,4,atol=1e-6) ? is_mult = false : nothing
        ctr += 1
    end
    @test is_mult
end

println("Next we consider the symmetric amplitude damping channel.")
println("First we define the channel function.")
#We just define the generalized qubit amplitude damping channel https://arxiv.org/abs/2107.13486
#and consider specific cases
function GADChannel(ρ :: Matrix{<:Number}, p :: Union{Int,Float64}, n :: Union{Int,Float64})
    K0 = sqrt(1-n)*[1 0 ; 0 sqrt(1-p)]
    K1 = sqrt(p*(1-n))*[0 1 ; 0 0]
    K2 = sqrt(n)*[sqrt(1-p) 0 ; 0 1]
    K3 = sqrt(p*n)*[0 0 ; 1 0]
    return K0*ρ*K0' + K1*ρ*K1' + K2*ρ*K2' + K3*ρ*K3'
end

n = 0.1
p = 0.2
K0 = sqrt(1-n)*[1 0 ; 0 sqrt(1-p)]
K1 = sqrt(p*(1-n))*[0 1 ; 0 0]
K2 = sqrt(n)*[sqrt(1-p) 0 ; 0 1]
K3 = sqrt(p*n)*[0 0 ; 1 0]
_state = [0 1 ; 0 0]
print(K3*_state*K3')

out = GADChannel(zero_state, 0.3, 0.2)

println("Then we verify the channel.")
@testset "Verify the GDAC" begin
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

gadChan(ρ) = GADChannel(ρ,0.1,0.2)
test = Choi(gadChan,2,2)
