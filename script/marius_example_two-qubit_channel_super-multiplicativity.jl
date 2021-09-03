using Base.Iterators
using CVChannel
using LinearAlgebra
using QBase
using Random
using Test

"""
In a previous note, Marius Junge described a channel that has a super-multiplicative
communication value (CV). This channel maps a 4-dimension Hilbert
space to a 5-dimension Hilbert space.
In this script, we first certify the CV of the singular use of this channel to be 2.
Then, we demonstrate using the see-saw method that the channel can achieve the
theoretical maximum CV of 5.
"""

"""
The channel is described by 5 operators which act upon a quantum state ρ to construct
the evolved state. Note that these operators do not for a typical kraus channel as
they are not complete.
"""
function marius_example_channel_operators()
    k1 = kron(σx, σI)
    k2 = kron(σy, σI)
    k3 = kron(σz, σx)
    k4 = kron(σz, σy)
    k5 = kron(σz, σz)

    [k1,k2,k3,k4,k5]
end

"""
The singular channel is implemented with the following method. The input `ρ`
must be an 4x4 matrix.
"""
function marius_example_channel(ρ :: AbstractMatrix)
    c = marius_example_channel_operators()
    basis = computational_basis_vectors(5)

     return sum( r -> sum( s -> tr(c[r]'*ρ*c[s])*basis[r]*basis[s]', 1:5), 1:5)/5
end

"""
The product of two instances of the channel is considered. The input `ρ` must be
a 16x16 matrix.
"""
function par_marius_example_channel(ρ :: AbstractMatrix)
    k_ops = marius_example_channel_operators()
    k_prod_ops = collect(flatten(map(k_a -> map(k_b -> kron(k_a, k_b), k_ops), k_ops)))
    basis = computational_basis_vectors(25)

    return sum( r -> sum( s -> tr(k_prod_ops[r]'*ρ*k_prod_ops[s])*basis[r]*basis[s]', 1:25), 1:25)/25
end

"""
Claim: the communication value of a single channel use of `marius_example_channel`
does not exceed 2.

The following tests certify this communication value using the primal and dual of
the PPT relaxation. Then we show that the PPT outer approximation is tight with
the CV using the see-saw method.
"""

println("\nVerifying communication value of a single channel use")

@time @testset "communication value of single channel use" begin

    choi_chan = Choi(marius_example_channel, 4, 5)
    kraus_ops = map(i -> reshape(choi_chan.JN[:,i],(5,4)) , 1:20)
    @test sum(k -> k' * k , kraus_ops) ≈ I

    @testset "PPT relaxation CV" begin
        cv_primal, = pptCV(choi_chan)
        cv_dual, = pptCV(choi_chan, :dual)

        @test cv_primal ≈ 2
        @test isapprox(cv_dual,2,atol=1e-6)
    end

    @testset "see-saw optimziation" begin
        Random.seed!(42)
        init_states = haarStates(4,4)
        cv_tuple4, = seesawCV(init_states, kraus_ops, 5)

        @test cv_tuple4[1] ≈ 2 atol=1e-3
        @test all(ρ -> is_density_matrix(ρ, atol=1e-6), cv_tuple4[2])
        @test is_povm(cv_tuple4[3], atol=1e-6)

        init_states = haarStates(8,4)
        cv_tuple8, = seesawCV(init_states, kraus_ops, 5)

        @test cv_tuple8[1] ≈ 2 atol=1e-4
        @test all(ρ -> is_density_matrix(ρ, atol=1e-6), cv_tuple8[2])
        @test is_povm(cv_tuple8[3], atol=2e-6)

        init_states = haarStates(16,4)
        cv_tuple16, = seesawCV(init_states, kraus_ops, 5)

        @test cv_tuple16[1] ≈ 2 atol=2e-6
        @test all(ρ -> is_density_matrix(ρ, atol=2e-6), cv_tuple16[2])
        @test is_povm(cv_tuple16[3], atol=1e-6)
    end
end

"""
Claim: The product channel `marius_example_channel_prod` has a communication value
exceeding 4.

The singular use of the channel is certified to be 2. We showed this above using
the primal and dual form of the PPT relaxation to find a communication value of 2
and this was demonstrated to be tight with the CV using the see-saw method.
Hence any communication value found to be greater than 4 is a expression of
super-multiplicativity. We demonstrate that the see-saw method can find the
theoretical maximum which is known to be 5.
"""

println("\nVerifying super-multiplivity of communication value of the product channel...")

@time @testset "communication value of parallel channel uses" begin

    par_choi_chan = Choi(par_marius_example_channel, 16, 25)
    par_kraus_ops = map(i -> reshape(par_choi_chan.JN[:,i],(25,16)) , 1:400)
    @test sum(k -> k' * k , par_kraus_ops) ≈ I

    @testset "see-saw verification of super-multiplicativity" begin
        println("performing see-saw optimization of parallel channel")
        Random.seed!(314)
        par_init_states = haarStates(32,16)
        cv_tuple32, = seesawCV(par_init_states, par_kraus_ops, 10, verbose=true)

        @test cv_tuple32[1] ≈ 5 atol=2e-2
        @test all(ρ -> is_density_matrix(ρ, atol=1e-6), cv_tuple32[2])
        @test is_povm(cv_tuple32[3], atol=1e-6)
    end
end
