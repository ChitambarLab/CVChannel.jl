using Test
using LinearAlgebra
using QBase
using Base.Iterators
using Suppressor
using CVChannel

@testset "./src/see-saw_optimization.jl" begin

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

@testset "seesawCV" begin
    d = 3
    anti_sym_choi = wernerState(d, 0) * d
    anti_sym_kraus_ops = map(i -> reshape(anti_sym_choi[:,i], (d,d)), 1:d^2)
    par_anti_sym_kraus_ops = collect(flatten(
        map(k1 -> map(k2 -> kron(k1,k2), anti_sym_kraus_ops), anti_sym_kraus_ops)
    ))

    # verify kraus operators are trace-preserving
    @test sum(k -> k * k', anti_sym_kraus_ops) == I
    @test sum(k -> k * k', par_anti_sym_kraus_ops) == I

    @testset "singular qutrit anti-symmetric Werner-Holevo channel" begin
        init_states = [
            [1/2 0 0;0 1/3 0;0 0 1/6],[1/6 0 0;0 1/2 0;0 0 1/3],[1/3 0 0;0 1/6 0;0 0 1/2]
        ]

        num_steps = 3
        max_cv_tuple, cvs, opt_ensembles, opt_povms = seesawCV(
            init_states, anti_sym_kraus_ops, num_steps
        )

        @test max_cv_tuple[1] ≈ 1.5
        @test all(ρ -> is_density_matrix(ρ), max_cv_tuple[2])
        @test is_povm(max_cv_tuple[3])
        @test length(max_cv_tuple) == 3

        @test length(cvs) == 2 * num_steps
        @test max(cvs...) ≈ 1.5
        @test min(cvs...) ≈ 1.25 atol=1e-5

        @test length(opt_ensembles) == num_steps
        @test all(states -> all(ρ -> is_density_matrix(ρ, atol=1e-6), states), opt_ensembles)

        @test length(opt_povms) == num_steps
        @test all(Π -> is_povm(Π, atol=1e-6), opt_povms)
    end

    @testset "parallel qutrit anti-symmetric Werner-Holevo channel" begin
        z_basis_states = computational_basis_states(9)
        max_mix_state = Matrix{Float64}(I, (9,9))/9
        init_states = map(ρ -> ρ/16 + 15/16 * max_mix_state, z_basis_states)

        num_steps = 2
        max_cv_tuple, cvs, opt_ensembles, opt_povms = seesawCV(
            init_states, par_anti_sym_kraus_ops, num_steps
        )

        @test max_cv_tuple[1] ≈ 2.25 atol=1e-6
        @test all(ρ -> is_density_matrix(ρ, atol=1e-6), max_cv_tuple[2])
        @test is_povm(max_cv_tuple[3], atol=1e-6)
        @test length(max_cv_tuple) == 3

        @test length(cvs) == 2 * num_steps
        @test max(cvs...) ≈ 2.25 atol=1e-6
        @test min(cvs...) ≈ 1.078 atol=1e-3

        @test length(opt_ensembles) == num_steps
        @test all(states -> all(ρ -> is_density_matrix(ρ, atol=1e-6), states), opt_ensembles)

        @test length(opt_povms) == num_steps
        @test all(Π -> is_povm(Π, atol=1e-6), opt_povms)
    end

    @testset "verbose printout" begin
        init_states = [
            [1/2 0 0;0 1/3 0;0 0 1/6],[1/6 0 0;0 1/2 0;0 0 1/3],[1/3 0 0;0 1/6 0;0 0 1/2]
        ]

        num_steps = 1
        empty_printout = @capture_out seesawCV(
            init_states, anti_sym_kraus_ops, num_steps
        )
        @test empty_printout == ""

        printout = @capture_out seesawCV(
            init_states, anti_sym_kraus_ops, num_steps, verbose=true
        )

        regex_match = r"^i = 1\nfixed_state_cv = \d\.\d+\nfixed_povm_cv = \d\.\d+\nmax_cv = \d\.\d+\n$"
        @test occursin(regex_match, printout)
    end
end

end
