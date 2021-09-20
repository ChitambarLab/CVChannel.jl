using CVChannel
using Test
"""
This script verifies the Siddhu channel when
tensored with the depolarizing channel is multiplicative.
"""

print("\nHere, we look at the communication value of running the")
print(" Siddhu and the d-dimensional depolarizing channel in parallel for d = {2,3}.")
@testset "Verifying Multiplicativity of Siddhu Channel with d∈[2,3] Depolarizing Channel" begin
    for d in [2:3;]
        println("~~~~Now scanning for d = ",d)
        s_scan_range =[0:0.1:0.5;]
        q_scan_range = [0:0.1:1;]
        max_val = 0
        is_mult = true
        for s in s_scan_range
            println("---Now scanning for s = ", s, "-----")
            for q in q_scan_range
                target_val = 2*(d*(1-q)+q)

                depolqChan(X) = depolarizingChannel(X,q)
                sidchan(X) = siddhuChannel(X,s)

                depolq_chan = Choi(depolqChan,d,d)
                sid_chan= Choi(sidchan,3,3)

                par_cv, = pptCV(parChoi(sid_chan, depolq_chan), :dual)

                non_mult = par_cv - target_val
                !isapprox(non_mult,0,atol=3e-6) ? is_mult = false : nothing
                max_val < abs(non_mult) ? max_val = abs(non_mult) : nothing
            end
        end
        @test is_mult
    end
end
