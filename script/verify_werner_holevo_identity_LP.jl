using CVChannel
using Test
"""
This script verifies that the LP for determining the cvPPT agrees with
the pre-established SDP method for dimensions small enough for the SDP
to be able to run. This is not part of the test files because it added
30 seconds to running the tests even though it is only three small cases,
though it takes less than 10 seconds if you have the solvers initialized.
"""
println("We now verify the LP and SDP agree on three settings.")
@testset "verify WHIDLP" begin
    conditions = [[2,2,0],[3,2,0],[2,3,0.25]]
    println("\nHere we initialize the identity channel and the solver. (One moment please...)")
    identChan(X) = X
    eaCVDual(choi(identChan,2,2),2,2)
    println("All done initializing! We now run the tests.")
    for cond in conditions
        wern_choi = cond[1]*wernerState(Int(cond[1]),(1-cond[3]))
        #because parameterization is backwards between LP and WernerState
        ident_choi = choi(identChan,Int(cond[2]),Int(cond[2]))
        kron_par_choi = kron(wern_choi,ident_choi)
        par_choi = permuteSubsystems(kron_par_choi,[1,3,2,4],convert(Array{Int64},[cond[1],cond[1],cond[2],cond[2]]))
        par_cv = pptCVPrimal(par_choi,Int(cond[1]*cond[2]),Int(cond[1]*cond[2]))
        par_cv_LP = WHIDLP(Int(cond[1]),Int(cond[2]),cond[3])
        @test isapprox(par_cv[1],par_cv_LP, atol=1e-6)
    end
end
println("Goodbye!")
