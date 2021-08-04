using LinearAlgebra
using CVChannel
using Convex
using Test
using DelimitedFiles

"""
This script verifies the multiplicativity of the depolarizing channel
with itself for small dimensions.
"""
@testset "Verifying Multiplicativity for d=3" begin
    println("Here we verify multiplicativity of the depolarizing channel with itself for d=3.")
    d = 3
    scan_range = [0:0.05:1;]
    is_mult = true
    max_val = 0
    for q_id in scan_range
        println("Now checking q =", q_id, ".")
        depolqChan(X) = depolarizingChannel(X,q_id)
        depolq_chan = Choi(depolqChan,d,d)

        par_dims = d*[1,1,1,1]
        par_JN = permuteSubsystems(kron(depolq_chan.JN, depolq_chan.JN), [1,3,2,4], par_dims)
        par_in_dim = d^2
        par_out_dim = d^2
        par_cv, = pptCVDual(par_JN, d^2, d^2)

        target = (d*(1-q_id)+q_id)^2
        !isapprox(par_cv,target,atol=3e-6) ? is_mult = false : nothing
        max_val < abs(par_cv-target) ? max_val = abs(par_cv-target) : nothing
    end
    @test is_mult
end

"""
This verifies multiplicativity for d=4 but it takes forever
and doesn't matter for running in parallel with other channels
because it is too big.

@testset "Verifying Multiplicativity for d=4" begin
    d = 4
    scan_range = [0.75:0.05:1;]
    is_mult = true
    max_val = 2.264111881089e-6
    for q_id in scan_range
        println("Now checking q =", q_id, ".")
        depolqChan(X) = depolarizingChannel(X,q_id)
        depolq_chan = Choi(depolqChan,d,d)

        par_dims = d*[1,1,1,1]
        par_JN = permuteSubsystems(kron(depolq_chan.JN, depolq_chan.JN), [1,3,2,4], par_dims)
        par_in_dim = d^2
        par_out_dim = d^2
        par_cv, = pptCVDual(par_JN, d^2, d^2)

        target = (d*(1-q_id)+q_id)^2
        !isapprox(par_cv,target,atol=3e-6) ? is_mult = false : nothing
        max_val < abs(par_cv-target) ? max_val = abs(par_cv-target) : nothing
    end
    @test is_mult
end

"""
