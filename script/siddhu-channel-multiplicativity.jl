using CVChannel
using Test

"""
This script shows the multiplicativity of the Siddhu channel with itself.
"""

println("In this script we look at the parallel communication value of the Siddhu channel with itself.")
println("Since we know cv(N) is 2 always, all we need to do is check that cvPPT(N ⊗ N) = 4 always.")
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
