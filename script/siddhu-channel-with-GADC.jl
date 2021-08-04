using CVChannel
using Test

"""
This script investigates the tensor product between the Siddhu channel
and the generalized amplitude damping channel (GADC).
The GADC is defined in https://arxiv.org/abs/2107.13486
"""

print("\nHere we verify that the Siddhu channel is multiplicative with GADC.")
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

                par_cv, = pptCV(parChoi(sid_chan, gad_chan), :dual)

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
    @test is_mult
end
