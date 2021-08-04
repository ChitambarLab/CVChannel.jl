using LinearAlgebra
using CVChannel
using Convex
using Test
using DelimitedFiles
"""
This script verifies the Siddhu channel when
tensored with the depolarizing channel is multiplicative.
The Siddhu channel is defined in (9) of https://arxiv.org/abs/2003.10367.
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

print("\nFinally, we look at the communication value of running the")
print(" Siddhu and the d-dimensional depolarizing channel in parallel.")
println("We start with d = 2.")
@testset "Verifying Multiplicativity of Siddhu Channel with d = 2 Depolarzing Channel" begin
    d = 2
    sid_scan_range =[0:0.1:0.5;]
    depol_scan_range = [0:0.1:1;]
    sid_ctr, q_ctr = 1,1
    max_val = 0
    is_mult = true
    for s_id in sid_scan_range
        println("---Now scanning for s=", s_id, ".-----")
        for q_id in depol_scan_range
            target_val = 2*(d*(1-q_id)+q_id)

            depolqChan(X) = depolarizingChannel(X,q_id)
            sidchan(X) = siddhuChannel(X,s_id)

            depolq_chan = Choi(depolqChan,d,d)
            sid_chan= Choi(sidchan,3,3)

            par_dims = [sid_chan.in_dim, sid_chan.out_dim, depolq_chan.in_dim, depolq_chan.out_dim]
            par_JN = permuteSubsystems(kron(sid_chan.JN, depolq_chan.JN), [1,3,2,4], par_dims)
            par_in_dim = sid_chan.in_dim * depolq_chan.in_dim
            par_out_dim = sid_chan.out_dim * depolq_chan.out_dim
            par_cv, = pptCVDual(par_JN, par_in_dim, par_out_dim)

            non_mult = par_cv - target_val
            !isapprox(non_mult,0,atol=3e-6) ? is_mult = false : nothing
            max_val < abs(non_mult) ? max_val = abs(non_mult) : nothing

            q_ctr += 1
        end
        sid_ctr += 1
        q_ctr = 1
    end

    print(max_val)
    @test is_mult
end

@testset "Verifying Multiplicativity of Siddhu Channel with d = 3 Depolarzing Channel" begin
    d = 3
    sid_scan_range =[0:0.1:0.5;]
    depol_scan_range = [0:0.1:1;]
    sid_ctr, q_ctr = 1,1
    max_val = 0
    is_mult = true
    for s_id in sid_scan_range
        println("---Now scanning for s=", s_id, ".-----")
        for q_id in depol_scan_range
            target_val = 2*(d*(1-q_id)+q_id)

            depolqChan(X) = depolarizingChannel(X,q_id)
            sidchan(X) = siddhuChannel(X,s_id)

            depolq_chan = Choi(depolqChan,d,d)
            sid_chan= Choi(sidchan,3,3)

            par_dims = [sid_chan.in_dim, sid_chan.out_dim, depolq_chan.in_dim, depolq_chan.out_dim]
            par_JN = permuteSubsystems(kron(sid_chan.JN, depolq_chan.JN), [1,3,2,4], par_dims)
            par_in_dim = sid_chan.in_dim * depolq_chan.in_dim
            par_out_dim = sid_chan.out_dim * depolq_chan.out_dim
            par_cv, = pptCVDual(par_JN, par_in_dim, par_out_dim)

            non_mult = par_cv - target_val
            !isapprox(non_mult,0,atol=3e-6) ? is_mult = false : nothing
            max_val < abs(non_mult) ? max_val = abs(non_mult) : nothing

            q_ctr += 1
        end
        sid_ctr += 1
        q_ctr = 1
    end

    print(max_val)
    @test is_mult
end
