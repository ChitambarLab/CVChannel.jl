using LinearAlgebra
using QBase
using Test

@testset "qutrit example" begin

# swap operator
F_33 = [
    1 0 0 0 0 0 0 0 0;
    0 0 0 1 0 0 0 0 0;
    0 0 0 0 0 0 1 0 0;
    0 1 0 0 0 0 0 0 0;
    0 0 0 0 1 0 0 0 0;
    0 0 0 0 0 0 0 1 0;
    0 0 1 0 0 0 0 0 0;
    0 0 0 0 0 1 0 0 0;
    0 0 0 0 0 0 0 0 1;
]

# 3x3 identity operator
I3 = Matrix(I,3,3)

# swap 2,3 for 4 qubit state
I3_F33_I3 = kron(I3,F_33,I3)

# proposed choi operator
J_N = 1/2*(Matrix(I,9,9)-F_33)

@testset "singular channel has cv(N) = 1.5" begin
    comp_states_3 = computational_basis_states(3)

    @testset "computational basis is bad choice" begin
        σAB_no_perm = sum(kron.(comp_states_3,comp_states_3))

        @test partial_trace(σAB_no_perm,[3,3],1) == I  # correct partial trace
        @test tr(σAB_no_perm*J_N) == 0
    end

    @testset "cyclic permutation on Bob's state is good choice" begin
        σAB_perm = sum(kron.(comp_states_3,comp_states_3[[2,3,1]]))

        @test partial_trace(σAB_perm, [3,3], 1) == I
        @test tr(σAB_perm*J_N) == 1.5
    end
end

@testset "product of choi matrices requires swap on σAABB" begin
    bell_states3 = generalized_bell_states(3)
    bell_prod_states = kron.(bell_states3,bell_states3)

    σAABB = sum(bell_prod_states)

    @testset "σAABB (no swap) does not achieve cv(N⊗N) = 3" begin
        @test partial_trace(σAABB,[9,9],1) ≈ I
        @test tr(sum(bell_prod_states)*kron(J_N,J_N)) ≈ 1.5
    end

    σABAB = I3_F33_I3*σAABB*I3_F33_I3'

    @testset "σABAB (swap) achievs cv(N⊗N) = 3" begin
        @test partial_trace(σABAB, [9,9],1) ≈ I
        @test tr(σABAB*kron(J_N,J_N)) ≈ 3
    end
end
