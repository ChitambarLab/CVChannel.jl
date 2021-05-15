using LinearAlgebra
using CVChannel
using Convex
using Test

println("\nVerifying the multiplicativity of channel value for dephrasure channel...")

@testset "Verifying Multiplicativity of Dephrasure Channel" begin
    println("We are going to check the multiplicativity when p,q in {0,0.1,...0.9,1}.")
    p_vals = [0.0:0.1:1;]; q_vals = [0.0:0.1:1;];
    pCtr = 0; qCtr = 0;
    pq_array = zeros(11,11);
    cv_array = zeros(11,11);
    cv2_array = zeros(11,11);
    for p_id in p_vals
        print("\nStarting p= ", p_id,". \n")
        for q_id in q_vals
            #Get Data
            dephrasurepq(ρ) = dephrasureChannel(ρ,p_id,q_id)
            orig_choi = choi(dephrasurepq,2,3);
            val1, opt1 = pptCVDual(orig_choi,2,3);
            parallel_choi = permuteSubsystems(kron(orig_choi,orig_choi),[1,3,2,4],[2,3,2,3]);
            val2, opt2 = pptCVPrimal(parallel_choi,4,9);

            #Store Data
            cv_array[pCtr+1,qCtr+1] = val1;
            cv2_array[pCtr+1,qCtr+1] = val2;

            #Iterate
            qCtr = qCtr + 1;
        end
        pCtr = pCtr + 1; qCtr = 0;
    end
    diff_array = cv2_array - cv_array.^2;
    for i in [1:1:11;]
        for j in [1:1:11;]
            if abs(diff_array[i,j]) <= 1e-5
                diff_array[i,j] = 0
            end
        end
    end
    println("The difference between single copy channel value and parallel is approximately:")
    show(stdout, "text/plain", diff_array)
    println("\n")
    @test diff_array == zeros(11,11)
end
