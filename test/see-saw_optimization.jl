using Test
using LinearAlgebra
using QBase
using CVChannel

@testset "see-saw_optimization.jl" begin

id_kraus_ops = [[1 0;0 1]]
bit_flip_kraus_ops = [[1 0;0 1]*sqrt(0.7), [0 1;1 0]*sqrt(0.3)]

x_states = [[0.5 0.5;0.5 0.5], [0.5 -0.5;-0.5 0.5]]
z_states = [[1 0;0 0],[0 0;0 1]]
trine_states = trine_qubit_states()

@testset "fixedStateCV" begin
    @testset "identity chanel" begin
        z_cv, z_povm = fixedStateCV(z_states, id_kraus_ops)

        @test z_cv ≈ 2
        @test is_povm(z_povm)
        @test isapprox(z_povm, z_states, atol=1e-7)

        trine_cv, trine_povm = fixedStateCV(trine_states, id_kraus_ops)

        @test trine_cv ≈ 2
        @test is_povm(trine_povm)
        @test isapprox(trine_povm, 2/3*trine_states, atol=1e-7)
    end

    @testset "bit-flip channel" begin
        x_cv, x_povm = fixedStateCV(x_states, bit_flip_kraus_ops)

        @test x_cv ≈ 2 atol = 1e-7
        @test is_povm(x_povm)
        @test isapprox(x_povm, x_states, atol=1e-7)

        z_cv, z_povm = fixedStateCV(z_states, bit_flip_kraus_ops)

        @test z_cv ≈ 1.4 atol = 1e-7
        @test is_povm(z_povm)
        @test isapprox(z_povm, z_states, atol=1e-7)
    end
end

@testset "fixedMeasurementCV" begin
    @testset "identity channel" begin
        z_cv, opt_z_states = fixedMeasurementCV(z_states, id_kraus_ops)

        @test z_cv ≈ 2 atol = 1e-6
        @test all(ρ -> is_density_matrix(ρ, atol=1e-6), opt_z_states)
        @test isapprox(opt_z_states, z_states, atol=1e-6)

        trine_povm = 2/3 * trine_states
        trine_cv, opt_trine_states = fixedMeasurementCV(trine_povm, id_kraus_ops)

        @test trine_cv ≈ 2 atol = 1e-6
        @test all(ρ -> is_density_matrix(ρ, atol=1e-6), opt_trine_states)
        @test isapprox(opt_trine_states, trine_states, atol=1e-6)
    end

    @testset "bit-flip channel" begin
        x_cv, opt_x_states = fixedStateCV(x_states, bit_flip_kraus_ops)

        @test x_cv ≈ 2 atol = 1e-7
        @test all(ρ -> is_density_matrix(ρ), opt_x_states)
        @test isapprox(opt_x_states, x_states, atol=1e-7)

        z_cv, opt_z_states = fixedStateCV(z_states, bit_flip_kraus_ops)

        @test z_cv ≈ 1.4 atol = 1e-7
        @test all(ρ -> is_density_matrix(ρ), opt_z_states)
        @test isapprox(opt_z_states, z_states, atol=1e-7)
    end
end

end
