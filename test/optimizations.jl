using Test, LinearAlgebra
using CVChannel

@testset "./src/optimizations.jl" begin

maxEntState = 0.5 * [1 0 0 1 ; 0 0 0 0 ; 0 0 0 0 ; 1 0 0 1]
maxMixState = 0.25 * [1 0 0 0 ; 0 1 0 0 ; 0 0 1 0 ; 0 0 0 1]

max_ent_choi = Choi(2*maxEntState, 2, 2)
max_mix_choi = Choi(2*maxMixState, 2, 2)

@testset "eaCV" begin
    primal_opt_tuple = eaCV(max_ent_choi)
    @test length(primal_opt_tuple) == 2
    @test primal_opt_tuple[1] ≈ 4 atol=1e-6
    @test primal_opt_tuple[2] ≈ 2*maxEntState atol=1e-6

    dual_opt_tuple = eaCV(max_ent_choi, :dual)
    @test length(dual_opt_tuple) == 2
    @test dual_opt_tuple[1] ≈ 4 atol=1e-6
    @test dual_opt_tuple[2] ≈ [2 0;0 2]
end

@testset "eaCVPrimal" begin
    @test isapprox(eaCVPrimal(maxMixState,2,2)[1] ,1/2, atol=1e-6)
    @test isapprox(eaCVPrimal(maxEntState,2,2)[1], 2, atol = 1e-6)
    #We don't test second output on maxMix because the optimal optimizer set is too big
    @test isapprox(eaCVPrimal(maxEntState,2,2)[2], 2*maxEntState, atol = 1e-6)
end

@testset "eaCVDual" begin
    @test isapprox(eaCVDual(maxMixState,2,2)[1] ,1/2, atol=1e-6)
    @test isapprox(eaCVDual(maxEntState,2,2)[1], 2, atol = 1e-6)
    @test isapprox(eaCVDual(maxMixState,2,2)[2],1/4*[1 0 ; 0 1], atol = 1e-6)
    @test isapprox(eaCVDual(maxEntState,2,2)[2],[1 0 ; 0 1], atol = 1e-6)
end

@testset "pptCV" begin
    primal_opt_tuple = pptCV(max_ent_choi)
    @test length(primal_opt_tuple) == 2
    @test primal_opt_tuple[1] ≈ 2 atol=1e-6
    @test primal_opt_tuple[2] ≈ [2/3 0 0 1/3 ; 0 1/3 0 0 ; 0 0 1/3 0 ; 1/3 0 0 2/3] atol=1e-2

    dual_opt_tuple = pptCV(max_ent_choi, :dual)
    @test length(dual_opt_tuple) == 3
    @test dual_opt_tuple[1] ≈ 2 atol=1e-6
    @test dual_opt_tuple[2] ≈ [1 0;0 1] atol=1e-6
    @test dual_opt_tuple[3] ≈ [0 0 0 0;0 1 -1 0;0 -1 1 0;0 0 0 0] atol=1e-6
end

@testset "pptCVPrimal" begin
    @test isapprox(pptCVPrimal(maxMixState,2,2)[1] ,1/2, atol=1e-6)
    @test isapprox(pptCVPrimal(maxEntState,2,2)[1], 1, atol = 1e-6)
    #Again we don't test second output on maxMix
    #One can determine the optimizer is what I give below, but the numerics gets only close-ish, hence atol is big
    @test isapprox(pptCVPrimal(maxEntState,2,2)[2],[2/3 0 0 1/3 ; 0 1/3 0 0 ; 0 0 1/3 0 ; 1/3 0 0 2/3], atol=10^(-2.4))
end

@testset "pptCVDual" begin
    @test isapprox(pptCVDual(maxMixState,2,2)[1] ,1/2, atol=1e-6)
    @test isapprox(pptCVDual(maxEntState,2,2)[1], 1, atol = 1e-6)
    @test isapprox(pptCVDual(maxMixState,2,2)[2],1/4*[1 0 ; 0 1], atol = 1e-6)
    @test isapprox(pptCVDual(maxEntState,2,2)[2],1/2*[1 0 ; 0 1], atol = 1e-6)
end

@testset "pptCVMultiplicativity" begin
    @testset "Choi operator arguments" begin
        ident3_choi = Choi(vec(Matrix{Int}(I,3,3))*vec(Matrix{Int}(I,3,3))',3,3)
        v1 = pptCVMultiplicativity(max_ent_choi, ident3_choi)
        v2 = pptCVMultiplicativity(max_ent_choi, ident3_choi, singular_method = :dual)
        v3 = pptCVMultiplicativity(max_ent_choi, ident3_choi, parallel_method = :primal)
        @test_throws MethodError pptCVMultiplicativity(max_ent_choi, ident3_choi, singular_method = :cat)
        @test_throws TypeError pptCVMultiplicativity(max_ent_choi, ident3_choi, parallel_method = true)
        #Check single copy
        @test isapprox(v1[1],2, atol=1e-6)
        @test isapprox(v1[2],3, atol=1e-6)
        @test isapprox(v2[1],2, atol=1e-6)
        @test isapprox(v2[2],3, atol=1e-6)
        @test isapprox(v3[1],2, atol=1e-6)
        @test isapprox(v3[2],3, atol=1e-6)
        @test v1[1] != v2[1]
        @test v1[2] != v2[2]
        #Check parallel answer
        @test v1[3]==v2[3]
        @test v1[3]!=v3[3]
        @test v2[3]!=v3[3]
        @test isapprox(v1[3],v2[3], atol=1e-5)
        @test isapprox(v1[3],v3[3], atol=1e-5)
        @test isapprox(v1[3],6, atol=1e-5)
    end
    @testset "matrix and dims arguments" begin
        choi_ident3 = vec([1 0 0 ; 0 1 0 ; 0 0 1])*vec([1 0 0 ; 0 1 0 ; 0 0 1])'
        v1 = pptCVMultiplicativity(2*maxEntState,2,2,choi_ident3,3,3)
        v2 = pptCVMultiplicativity(2*maxEntState,2,2,choi_ident3,3,3,step1isdual = true)
        v3 = pptCVMultiplicativity(2*maxEntState,2,2,choi_ident3,3,3,step2isprimal = true)
        @test_throws TypeError pptCVMultiplicativity(2*maxEntState,2,2,choi_ident3,3,3,step2isprimal = "cat")
        @test_throws TypeError pptCVMultiplicativity(2*maxEntState,2,2,choi_ident3,3,3,step1isdual = 12)
        #Check single copy
        @test isapprox(v1[1],2, atol=1e-6)
        @test isapprox(v1[2],3, atol=1e-6)
        @test isapprox(v2[1],2, atol=1e-6)
        @test isapprox(v2[2],3, atol=1e-6)
        @test isapprox(v3[1],2, atol=1e-6)
        @test isapprox(v3[2],3, atol=1e-6)
        @test v1[1] != v2[1]
        @test v1[2] != v2[2]
        #Check parallel answer
        @test v1[3]==v2[3]
        @test v1[3]!=v3[3]
        @test v2[3]!=v3[3]
        @test isapprox(v1[3],v2[3], atol=1e-5)
        @test isapprox(v1[3],v3[3], atol=1e-5)
        @test isapprox(v1[3],6, atol=1e-5)
    end
end

@testset "WHIDLP" begin
    identChan(X) = X
    conditions = [[2,2,0],[3,2,0],[2,3,0.25]]
    for (d1f,d2f,λ) in conditions
        d1 = Int(d1f)
        d2 = Int(d2f)

        wern_choi = d1*wernerState(d1, λ)
        ident_choi = choi(identChan, d2, d2)
        kron_par_choi = kron(wern_choi,ident_choi)
        par_choi = permuteSubsystems(kron_par_choi, [1,3,2,4], [d1,d1,d2,d2])
        par_cv = pptCVPrimal(par_choi, d1*d2, d1*d2)
        par_cv_LP = WHIDLP(d1, d2, λ)
        @test isapprox(par_cv[1], par_cv_LP, atol=1e-5)
    end
end

@testset "generalWHLPConstraints" begin
    #Domain Error Checks
    @test_throws DomainError generalWHLPConstraints(3,3,ones(4))
    @test_throws DomainError generalWHLPConstraints(2,4,[1.,6])

    #Generating Constraints Checks
    n,d,λ = 1,2,1
    A,B,g,a = generalWHLPConstraints(n,d,λ*ones(n))
    @testset "n,d,λ = 1,2,1" begin
        @test A == [1 1 ; 1 -1]
        @test B == [1 0 ; 1 d]
        @test a == reshape([1.;(2λ-1)],:,1)
        @test g == [d 1]
    end

    λ = 0.8
    A,B,g,a = generalWHLPConstraints(n,d,λ*ones(n))
    @testset "n,d,λ = 1,2,0.8" begin
        @test A == [1 1 ; 1 -1]
        @test B == [1 0 ; 1 d]
        @test a == reshape([1.;(2λ-1)],:,1)
        @test g == [d 1]
    end

    d = 4
    A,B,g,a = generalWHLPConstraints(n,d,λ*ones(n))
    @testset "n,d,λ = 1,4,0.8" begin
        @test A == [1 1 ; 1 -1]
        @test B == [1 0 ; 1 d]
        @test a == reshape([1.;(2λ-1)],:,1)
        @test g == [d 1]
    end

    n,d,λ = 2,3,1
    A,B,g,a = generalWHLPConstraints(n,d,λ*ones(n))
    targA = [1 1 1 1 ; 1 -1 1 -1 ; 1 1 -1 -1 ; 1 -1 -1 1]
    targB = [1. 0 0 0 ; 1 d 0 0 ; 1 0 d 0 ; 1 d d d^2]
    targa = reshape([1. 1 1 1],:,1)
    targg = [d^2 d d 1]
    @testset "n,d,λ = 2,3,1" begin
        @test A == targA
        @test B == targB
        @test a == targa
        @test g == targg
    end
end

@testset "wernerHolevoCVPPT" begin
    n,d,λ = 1,3,0
    A,B,g,a = generalWHLPConstraints(n,d,λ*ones(n))
    lp_cv_ppt, opt = wernerHolevoCVPPT(n,d,A,B,g,a)
    orig_choi = 3*wernerState(3,0)
    test1 = pptCVDual(orig_choi,3,3)
    @test isapprox(test1[1], lp_cv_ppt, atol = 1e-6)

    λ = 0.2
    A,B,g,a = generalWHLPConstraints(n,d,λ*ones(n))
    lp_cv_ppt, opt = wernerHolevoCVPPT(n,d,A,B,g,a)
    choi_2 = 3*wernerState(3,λ)
    test2 = pptCVDual(choi_2,3,3)
    @test isapprox(test2[1], lp_cv_ppt, atol = 1e-6)

    n,λ = 2,0
    A,B,g,a = generalWHLPConstraints(n,d,λ*ones(n))
    lp_cv_ppt, opt = wernerHolevoCVPPT(n,d,A,B,g,a)
    kron_par_choi = kron(orig_choi,orig_choi)
    par_choi = permuteSubsystems(kron_par_choi,[1,3,2,4],[3,3,3,3])
    test3 = pptCVPrimal(par_choi,9,9)
    @test isapprox(test3[1], lp_cv_ppt, atol = 1e-5)
end

@testset "twoSymCVPrimal" begin
    #We can't really test a lot with this since we don't really know much
    #Here we verify that it gives answers we know it should give even over
    #the separable cone
    test1 = 3*wernerState(3,0)
    choi1 = Choi(test1, 3, 3)
    cv1 = pptCVPrimal(test1,3,3)
    cv1_two_sym = twoSymCVPrimal(test1,3,3)
    choi_cv1_two_sym = twoSymCVPrimal(choi1)
    @test isapprox(cv1[1], cv1_two_sym[1], atol=1e-6)
    @test isapprox(cv1[1], choi_cv1_two_sym[1], atol=1e-6)

    test2 = vec([1 0 0 ; 0 1 0 ; 0 0 1])*vec([1 0 0 ; 0 1 0 ; 0 0 1])'
    choi2 = Choi(test2, 3, 3)
    cv2 = pptCVPrimal(test2,3,3)
    cv2_two_sym = twoSymCVPrimal(test2,3,3)
    choi_cv2_two_sym = twoSymCVPrimal(choi2)
    @test isapprox(cv2[1], cv2_two_sym[1], atol=2e-6)
    @test isapprox(cv2[1], choi_cv2_two_sym[1], atol=2e-6)
end
end
