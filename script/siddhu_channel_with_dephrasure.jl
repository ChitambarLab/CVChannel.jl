using LinearAlgebra
using CVChannel
using Convex
using Test
using DelimitedFiles

"""
This script invesetigates the channel which we call
the Siddhu channel when tensored with the dephrasure
channel.
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

print("\n Finally, we look at the communication value of running the")
print(" Siddhu and dephrasure channel in parallel. Note as a special")
print(" case this considers the erasure channel.")
@testset "Verifying Multiplicativity of Siddhu Channel with Dephrasure Channel" begin
    sid_scan_range =[0:0.1:0.5;]
    dephrasure_scan_range = [0:0.1:1;]
    sid_ctr, p_ctr, q_ctr = 1,1,1
    max_val = 0
    is_mult = true
    for s_id in sid_scan_range
        println("---Now scanning for s =", s_id, ".-----")
        for q_id in dephrasure_scan_range
            println("Now scanning for q =", q_id,".")
            target_val = 2*(2-q_id)
            for p_id in dephrasure_scan_range
                dephrasurepq(X) = dephrasureChannel(X,p_id,q_id)
                sidchan(X) = siddhuChannel(X,s_id)

                dephr_chan = Choi(dephrasurepq,2,3)
                sid_chan= Choi(sidchan,3,3)

                par_dims = [sid_chan.in_dim, sid_chan.out_dim, dephr_chan.in_dim, dephr_chan.out_dim]
                par_JN = permuteSubsystems(kron(sid_chan.JN, dephr_chan.JN), [1,3,2,4], par_dims)
                par_in_dim = sid_chan.in_dim * dephr_chan.in_dim
                par_out_dim = sid_chan.out_dim * dephr_chan.out_dim
                par_cv, = pptCVDual(par_JN, par_in_dim, par_out_dim)

                non_mult = par_cv - target_val
                !isapprox(non_mult,0,atol=3e-6) ? is_mult = false : nothing
                max_val < non_mult ? max_val = non_mult : nothing

                p_ctr += 1
            end
            q_ctr += 1
            p_ctr = 1
        end
        sid_ctr += 1
        q_ctr = 1
    end

    print(max_val)
    @test is_mult
end
