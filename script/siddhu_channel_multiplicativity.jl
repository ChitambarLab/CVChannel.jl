using LinearAlgebra
using CVChannel
using Convex
using Test
using DelimitedFiles

"""
This script invesetigates the channel which we call
the Siddhu channel when tensored with itself.
It is defined in (9) of https://arxiv.org/abs/2003.10367.
It is parameterized by s∈[0,1/2]
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
@testset "Verify that cv is 2 and cv(N ⊗ N) is multiplicative" begin
    scan_range =[0:0.01:0.5;]
    is_mult = true
    ctr = 1
    println("Now starting to scan over s.")
    for s_id in scan_range
        ctr % 10 == 0 ? println("Now on ", ctr, " of ", length(scan_range), " points.") : nothing
        #To save time I don't check the single copy cv
        # parallel channel uses
        sidchan(X) = siddhuChannel(X,s_id)
        sid_chan= Choi(sidchan,3,3)
        par_cv, = pptCV(parChoi(sid_chan, sid_chan), :dual)
        !isapprox(par_cv,4,atol=3e-6) ? is_mult = false : nothing
        ctr += 1
    end
    @test is_mult
end
