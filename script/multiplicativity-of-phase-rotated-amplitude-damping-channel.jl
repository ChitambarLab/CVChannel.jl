using LinearAlgebra
using CVChannel
using Convex
using Test

println("First we define the channel and verify it works.")
function phaseRotAmpDamp(ρ :: Matrix{<:Number}, ε :: Union{Int,Float64}) :: Matrix{<:Number}
    K0 = [1 0 ; 0 sqrt(1-ε)]
    K1 = 1/sqrt(2)*[1 exp(1im*pi/4) ; exp(-1im*pi/4) 1]*[0 sqrt(ε) ; 0 0]

    return K0*ρ*K0' + K1*ρ*K1'
end

@testset "Verify the phase rotated Amplitude Damping Channel Works" begin
    ρ1 = [3/4 0 ; 0 1/4]
    ρ2 = 1/2*[1 -1im ; 1im 1]
    ε1 = 0.3; ε2 = 0.7;

    target1(ε) = [
        3/4+1/8*ε  1/8*exp(1im*pi/4)*ε
        1/8*exp(-1im*pi/4)*ε 1/4-1/8*ε
    ]

    target2(ε) = [
        1/2+1/4*ε -1im/2*sqrt(1-ε)+1/4*exp(1im*pi/4)*ε
        1im/2*sqrt(1-ε)+1/4*exp(-1im*pi/4)*ε 1/2-1/4*ε
    ]

    @test isapprox(phaseRotAmpDamp(ρ1,ε1),target1(ε1), atol=1e-6)
    @test isapprox(phaseRotAmpDamp(ρ1,ε2),target1(ε2), atol=1e-6)
    @test isapprox(phaseRotAmpDamp(ρ2,ε1),target2(ε1), atol=1e-6)
    @test isapprox(phaseRotAmpDamp(ρ2,ε2),target2(ε2), atol=1e-6)
end

println("Now we are going to verify the multiplicativity of the channel")
@testset "Verify Multiplicativity of the Channel" begin
    ε_vals = [0:0.01:1;];
    ctr = 1;
    results = zeros(size(ε_vals)[1],4)
    println("\nBeginning to get the data...")
    println("\nPlease note it starts slow calling the solver.")
    for ε_id in ε_vals
        if (ε_id != 0)&&(Int(floor(ε_id*100)) % 5 == 0)
            println("\nNow evaluating for ε = ", ε_id)
        end
        phaseRotAmpDampVarep(ρ) = phaseRotAmpDamp(ρ,ε_id)
        orig_choi = choi(phaseRotAmpDampVarep,2,2)
        test1 = minEntropyPPTDual(orig_choi,2,2)
        kron_par_choi = kron(orig_choi,orig_choi)
        par_choi = permuteSubsystems(kron_par_choi,[1,3,2,4],[2,2,2,2])
        test2 = minEntropyPPTPrimal(par_choi,4,4)
        results[ctr,:] = [ε_id, test1[1], test2[1], test2[1]-test1[1]^2]
        ctr = ctr + 1
    end
    println("\ncolumn labels at bottom")
    show(stdout, "text/plain", results)
    println("\n p     cv(N)   cv(N^2)  diff")
    println("\nThus by looking at the diff column, we see what we were verifying")
    println("up to a numerical accuracy of 5e-6.")
    @test all(result -> result > 0 || isapprox(result, 0, atol=5e-6), results[:,4])
end
