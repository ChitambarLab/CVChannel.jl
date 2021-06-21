using Test
using CVChannel

"""
This script verifies that the channels formed from axisymmetric states are multiplicative
for qubits and qutrits.
"""

println("Verifying qubit multiplicativity of axisymmetric channels")
@testset "qubit multiplicativity of axisymmetric channels" begin
    y_step = 0.1
    x_step = 0.1

    d = 2
    y_bounds = CVChannel._axisymmetric_y_bounds(d)

    it = 0
    y_range = y_bounds[1]:y_step:y_bounds[2]
    for y in y_range
        it = it + 1
        println("verifying y-slice $it of $(length(y_range))")
        x_constraints = CVChannel._axisymmetric_x_constraints(d,y)
        for x in x_constraints[1]:x_step:x_constraints[2]
            @testset "(x,y) = ($x,$y)" begin
                ρ_axi = axisymmetricState(d,x,y)

                J_N = d*ρ_axi
                (opt_cv_N, opt_σAB_N) = pptCVDual(J_N, d, d)

                J_NN = permuteSubsystems(kron(J_N,J_N), [1,3,2,4], [d,d,d,d])
                (opt_cv_NN, opt_σAB_NN) = pptCVPrimal(J_NN, d^2, d^2)

                multiplicativity = opt_cv_NN - opt_cv_N^2

                @test multiplicativity ≈ 0 atol=2e-5
            end
        end
    end
end

println("Verifying qutrit multiplicativity of axisymmetric channels")
@time @testset "qutrit multiplicativity of axisymmetric channels" begin
    y_step = 0.1
    x_step = 0.1

    d = 3

    y_bounds = CVChannel._axisymmetric_y_bounds(d)

    y_range = y_bounds[1]:y_step:y_bounds[2]
    it = 0
    for y in y_range
        it = it + 1
        println("verifying y-slice $it of $(length(y_range))")
        x_constraints = CVChannel._axisymmetric_x_constraints(d,y)
        for x in x_constraints[1]:x_step:x_constraints[2]
            @testset "(x,y) = ($x,$y)" begin
                ρ_axi = axisymmetricState(d,x,y)

                J_N = d*ρ_axi
                (opt_cv_N, opt_σAB_N) = pptCVDual(J_N, d, d)

                J_NN = permuteSubsystems(kron(J_N,J_N), [1,3,2,4], [d,d,d,d])
                (opt_cv_NN, opt_σAB_NN) = pptCVPrimal(J_NN, d^2, d^2)

                multiplicativity = opt_cv_NN - opt_cv_N^2

                @test multiplicativity ≈ 0 atol=2e-5
            end
        end
    end
end
