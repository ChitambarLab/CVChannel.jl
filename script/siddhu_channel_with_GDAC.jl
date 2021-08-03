using LinearAlgebra
using CVChannel
using Convex
using Test
using DelimitedFiles

"""
This script invesetigates the channel which we call
the Siddhu channel when tensored with the generalized
qubit amplitude damping channel (GADC).
The Siddhu channel is defined in (9) of https://arxiv.org/abs/2003.10367.
It is parameterized by s∈[0,1/2]
The GADC is defined in https://arxiv.org/abs/2107.13486
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

println("First we define the generalized qubit amplitude damping chanenl.")
function GADChannel(ρ :: Matrix{<:Number}, p :: Union{Int,Float64}, n :: Union{Int,Float64})
    K0 = sqrt(1-n)*[1 0 ; 0 sqrt(1-p)]
    K1 = sqrt(p*(1-n))*[0 1 ; 0 0]
    K2 = sqrt(n)*[sqrt(1-p) 0 ; 0 1]
    K3 = sqrt(p*n)*[0 0 ; 1 0]
    return K0*ρ*K0' + K1*ρ*K1' + K2*ρ*K2' + K3*ρ*K3'
end

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

print("\n Finally, we look at the communication value of running the")
print(" Siddhu and GDAC channel in parallel.")
@testset "Verifying Multiplicativity of Siddhu Channel with GDAC" begin
    sid_scan_range =[0:0.1:0.5;]
    gad_scan_range = [0:0.1:1;]
    sid_ctr, gad_ctr_1, gad_ctr_2 = 1,1,1
    max_val = 0
    is_mult = true
    for s_id in sid_scan_range
        println("---Now scanning for s =", s_id, ".-----")
        for p_id in gad_scan_range
            println("Now scanning for p =", p_id,".")
            target_val = 2*(1+sqrt(1-p_id))
            for n_id in gad_scan_range
                symgadchan(X) = GADChannel(X,p_id,n_id)
                sidchan(X) = siddhuChannel(X,s_id)

                gad_chan = Choi(symgadchan,2,2)
                sid_chan= Choi(sidchan,3,3)

                par_dims = [sid_chan.in_dim, sid_chan.out_dim, gad_chan.in_dim, gad_chan.out_dim]
                par_JN = permuteSubsystems(kron(sid_chan.JN, gad_chan.JN), [1,3,2,4], par_dims)
                par_in_dim = sid_chan.in_dim * gad_chan.in_dim
                par_out_dim = sid_chan.out_dim * gad_chan.out_dim
                par_cv, = pptCVDual(par_JN, par_in_dim, par_out_dim)

                non_mult = par_cv - target_val
                !isapprox(non_mult,0,atol=5e-6) ? is_mult = false : nothing
                max_val < non_mult ? max_val = non_mult : nothing

                gad_ctr_2 += 1
            end
            gad_ctr_1 += 1
            gad_ctr_2 = 1
        end
        sid_ctr += 1
        gad_ctr_1 = 1
    end

    print(max_val)
    @test is_mult
end
