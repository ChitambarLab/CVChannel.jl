using LinearAlgebra
using CVChannel
using Test

"""
This verifies the multiplicativity of the [dephrasure channel](https://arxiv.org/abs/1806.08327)

The dephrasure channel is parameterized by two parameters. We scan over these parameters
and check that it is multiplicative at each point.
...
"""

println("\nVerifying the multiplicativity of channel value for dephrasure channel...")

@testset "Verifying Multiplicativity of Dephrasure Channel" begin
    println("We are going to check the multiplicativity when p,q in {0,0.1,...0.9,1}.")
    p_vals = [0.0:0.1:1;]; q_vals = [0.0:0.1:1;];
    pCtr = 0; qCtr = 0;
    max_diff = 0;
    for q_id in q_vals
        print("\nStarting q= ", q_id,". \n")
        orig_cv = 2 - q_id
        for p_id in p_vals
            #Get Data
            dephrasurepq(ρ) = dephrasureChannel(ρ,p_id,q_id)
            orig_choi = choi(dephrasurepq,2,3);
            par_choi = permuteSubsystems(kron(orig_choi,orig_choi),[1,3,2,4],[2,3,2,3]);

            #This guarantees upper bound which is important since the claim
            #is cv = cv_ppt
            par_cv, opt1, opt2 = pptCVDual(par_choi,4,9);

            diff = par_cv - orig_cv^2
            abs(diff) > max_diff ? max_diff = abs(diff) : nothing

            #Iterate
            pCtr = pCtr + 1
        end
        qCtr += 1
        pCtr = 0
    end

    println("The difference between single copy channel value and parallel is upper bounded by:", max_diff)
    @test max_diff <= 5e-6
end
