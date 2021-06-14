using LinearAlgebra
using CVChannel
using Convex
using Test
"""
In this script we investigate a family of co-positive maps.
Specifically, we investigate the set of states presented in
Eqn. (82) of https://arxiv.org/abs/1004.1655

Note we know co-positive maps have multiplicativity (cite Theorem/proposition
when we have a write-up, I guess), so the only way to check this at all is
implementing the DPS hierarchy.
"""

println("\nFirst we define the set of bound entangled states we will use.")
println("\nWe note they can be scaled to channels as they are bell diagonal.")

#We need this variation on the bellUnitary function because the authors
#decided to do something non-standard to generate them in a different
#order, I guess
function bellUnitaryVar(m :: Int64, n :: Int64, d :: Int64) :: Matrix
    #There probably are better names for this function, but yeah
    if m < 0 || n < 0
        throw(DomainError((n,m), "Make sure m,n ∈ [0,1,...,d-1]."))
    elseif m >= d || n >= d
        throw(DomainError((n,m), "Make sure m,n < d."))
    end
    λ = exp(2*pi*1im / d)
    U = zeros(ComplexF64,d,d)
    for k = 0 : d-1
        U[((k+n) % d) + 1,k+1] = λ^(k*m)
    end
    return U
end

function boundBell(ε :: Union{Int,Float64}) :: Matrix
    if ε <= 0
        throw(DomainError(ε, "Make sure ε > 0."))
    end
    #This function generates the states defined in (82)
    id_mat = Matrix(1I, 3, 3)
    bell = 1/3*vec(id_mat)*vec(id_mat)';
    #Π operators defined in (40) and implicitly using (35)
    Π1 = zeros(9,9); Π2 = zeros(9,9)
    for i = 0:2
        Π1 = Π1 + kron(id_mat,bellUnitaryVar(i,1,3))*bell*kron(id_mat,bellUnitaryVar(i,1,3)')
        Π2 = Π2 + kron(id_mat,bellUnitaryVar(i,2,3))*bell*kron(id_mat,bellUnitaryVar(i,2,3)')
    end
    #They don't say to normalize it, but otherwise it isn't a state
    Π1 = 1/tr(Π1) * Π1 ; Π2 = 1/tr(Π2) * Π2
    N_ε = 1/(1+ε+ε^(-1))

    return N_ε*(bell + ε*Π1 + ε^(-1)*Π2)
end

println("\nAs a sanity check, we first see they are PPT states.")
println("\nBecause of numerical error, we can't just call the isPPT function")
@testset "Verifying class of states are PPT" begin
    ε_checks = [0.1,0.25,0.5,0.75,1,1.5,2,3,5]
    fail_state = false
    for ε_id in ε_checks
        test_state = boundBell(ε_id)
        realvals = real.(eigvals(partialtranspose(test_state,2,[3,3])))
        imvals = imag.(eigvals(partialtranspose(test_state,2,[3,3])))
        if !isapprox(tr(test_state),1,atol = 1e-9)
            fail_state = true
            break
        elseif !all(imvals -> imvals < 1e-10, imvals)
            fail_state = true
            break
        elseif !all(realvals -> realvals > 0 || isapprox(realvals,0,atol=1e-8),realvals)
            fail_state = true
            break
        else
        end
    end
    @test !fail_state
end

println("\nAs a second sanity check, we make sure they are multiplicative with eachother.")
println("\nThis also gives us a chance to initialize the solver.")
@testset "Verifying PPT multiplicativity" begin
    test_state1 = 3*boundBell(0.25)
    test_state2 = 3*boundBell(1.5)
    println("\nInitializing solver, please wait...")
    cv1 = pptCVDual(test_state1,3,3)
    println("\nAll initialized. The rest should be rather quick.")
    cv2 = pptCVDual(test_state2,3,3)
    state11 = permuteSubsystems(kron(test_state1,test_state1),[1,3,2,4],[3,3,3,3])
    state12 = permuteSubsystems(kron(test_state1,test_state2),[1,3,2,4],[3,3,3,3])
    state22 = permuteSubsystems(kron(test_state2,test_state2),[1,3,2,4],[3,3,3,3])
    cv11 = pptCVPrimal(state11,9,9)
    cv12 = pptCVPrimal(state12,9,9)
    cv22 = pptCVPrimal(state22,9,9)
    @test isapprox(cv11[1] - cv1[1]^2, 0, atol = 1e-6)
    @test isapprox(cv12[1] - cv1[1]*cv2[1], 0, atol = 5e-6)
    @test isapprox(cv22[1] - cv2[1]^2, 0, atol = 5e-6)
end

println("\nThis motivates attempting to use the DPS hierarchy, but we will now")
println("see that we run out of memory to do that.")
@testset "No memory for DPS hierarchy" begin
    test_state1 = 3*boundBell(0.25)
    cv1 = pptCVDual(test_state1,3,3)
    state11 = permuteSubsystems(kron(test_state1,test_state1),[1,3,2,4],[3,3,3,3])
    out_of_mem = false
    try
        cv11 = twoSymCVPrimal(state11,9,9)
    catch
        out_of_mem = true
    end
    @test out_of_mem
end
