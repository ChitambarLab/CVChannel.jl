using CVChannel
using Test

"""
This script shows the Siddhu and dephrasure channels tensored
admit multiplicative communication value.
"""

print("\n Here we verify the multiplicativity of the Siddhu and dephrasure")
print(" channel when ran in parallel. Note as a special")
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

                par_cv, = pptCV(parChoi(sid_chan, dephr_chan), :dual)

                non_mult = par_cv - target_val
                !isapprox(non_mult,0,atol=4e-6) ? is_mult = false : nothing
                max_val < non_mult ? max_val = non_mult : nothing

                p_ctr += 1
            end
            q_ctr += 1
            p_ctr = 1
        end
        sid_ctr += 1
        q_ctr = 1
    end
    @test is_mult
end
