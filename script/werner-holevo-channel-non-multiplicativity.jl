using LinearAlgebra
using CVChannel
using Convex
using Test

#This script looks at the werner-holevo channels
println("\nFirst we convince ourselves that the choi states of the Werner-Holevo (WH)")
println("channels are the werner states multiplied by the dimension, so that we can")
println("just use the werner states.")

@testset "Verifying werner states are choi states of WH channels" begin
    wernerHolevoChannel_0(ρ) = wernerHolevoChannel(ρ,0)
    wernerHolevoChannel_1(ρ) = wernerHolevoChannel(ρ,1)
    wernerHolevoChannel_05(ρ) = wernerHolevoChannel(ρ,0.5)
    @test 3*wernerState(3,0) == choi(wernerHolevoChannel_0,3,3)
    @test 3*wernerState(3,1) == choi(wernerHolevoChannel_1,3,3)
    @test 3*(0.5*wernerState(3,0)+0.5*wernerState(3,1)) == choi(wernerHolevoChannel_05,3,3)
end

println("\nWe begin with the non-multiplicativity of the Werner-Holevo channel")
println("with p=0, whose Choi state is the anti-symmetric projector, over the")
println("PPT cone for d = 3.")

@testset "Non-multiplicativity of Anti-Symmetric Projector" begin
    orig_choi = 3*wernerState(3,0)
    test1 = minEntropyPPTDual(orig_choi,3,3)

    kron_par_choi = kron(orig_choi,orig_choi)
    par_choi = permuteSubsystems(kron_par_choi,[1,3,2,4],[3,3,3,3])
    test2 = minEntropyPPTPrimal(par_choi,9,9)
    @test test1[1]^2 != test2[2]
end

println("\nFinally we will see the Werner-Holevo channels are multiplicative")
println("for p less than 0.3, by searching over p in {0,0.01,...,0.3}.")

@testset "Non-multiplicativity of WH channels for p<0.3" begin
    p_vals = [0:0.01:0.3;];
    ctr = 1;
    results = zeros(31,4)
    println("\nBeginning to get the data.")
    for p_id in p_vals
        if (p_id != 0)&&(Int(floor(p_id*100)) % 5 == 0)
            println("\nNow evaluating for p = ", p_id)
        end
        orig_choi = 3*wernerState(3,p_id)
        test1 = minEntropyPPTDual(orig_choi,3,3)
        kron_par_choi = kron(orig_choi,orig_choi)
        par_choi = permuteSubsystems(kron_par_choi,[1,3,2,4],[3,3,3,3])
        test2 = minEntropyPPTPrimal(par_choi,9,9)
        results[ctr,:] = [p_id, test1[1], test2[1], test2[1]-test1[1]^2]
        ctr = ctr + 1
    end
    println("\ncolumn labels at bottom")
    show(stdout, "text/plain", results)
    println("\n p     cv(N)   cv(N^2)  diff")
    println("\nThus by looking at the diff column, we see what we were verifying.")
    @test (results[:,4] .> 0) == ones(31)

    #I leave this here in case one wants it, but the print out
    #is sufficient and I don't see why one would want this if running
    #from the command prompt
    #using Plots
    #using LaTeXStrings
    #x = [0:0.01:0.3;];
    #y = results[:,2:4];
    #title_str = "Multiplicativity of Holevo-Werner Channel";
    #label_str = [L"cv(\mathcal{N})" L"cv \left(\mathcal{N}^{\otimes 2} \right)" L"cv \left(\mathcal{N}^{\otimes 2} \right) - cv(\mathcal{N})"];
    #plot(x,
    #     y,
    #     xlims = (0,0.31),
    #     title = title_str,
    #     label= label_str,
    #     lw=1)
end
