using Test, LinearAlgebra, Convex, Suppressor
using CVChannel

# Check whether a Mosek license exists in the test ENV
const has_MOSEK_LICENSE = hasMOSEKLicense()

@testset "./src/optimizer_interface.jl" begin

@testset "qsolve!()" begin
    function _state_optimization()
        Π = [0.5 0.5;0.5 0.5]
        ρ = HermitianSemidefinite(2)

        objective = real(tr(Π*ρ))
        constraints = [
            real(tr(ρ)) == 1,
            imag(tr(ρ)) == 0
        ]

        return ρ, objective, constraints
    end

    # NOTE: These test should really be testing which optimizer is running, but
    # Suppressor is not able to capture the outputs for some unknown reason.
    @testset "runs with SCS backend" begin
        (ρ, objective, constraints) = _state_optimization()
        problem = maximize(objective, constraints)

        # unable to capture output from SCS with suppressor
        # testing in default quiet mode
        qsolve!(problem)

        @test problem.optval ≈ 1
        @test ρ.value ≈ [0.5 0.5;0.5 0.5]
    end

    if has_MOSEK_LICENSE
        @testset "runs with mosek backend" begin
            (ρ, objective, constraints) = _state_optimization()
            problem = maximize(objective, constraints)

            mosek_out = @capture_out qsolve!(problem, quiet=false, use_mosek=true)

            @test occursin(r"Problem\n\s*Name\s*:\s*\n\s*Objective sense\s*:\s*max\s*\n", mosek_out)
            @test problem.optval ≈ 1
            @test isapprox(ρ.value, [0.5 0.5;0.5 0.5], atol=1e-5)
        end
    else
        @warn "MOSEK tests skipped because a license was not found."
    end
end

@testset "hasMOSEKLicense()" begin
    test_dir = joinpath(@__DIR__,"files")

    # WARNING: Modifying ENV is a good way to shoot yourself in the foot.
    # making a copy of the environment variables
    env_copy = copy(ENV)

    # restores the environment variables to original values
    _restoreENV() = begin
        ENV["MOSEKLM_LICENSE_FILE"] = haskey(env_copy, "MOSEKLM_LICENSE_FILE") ? env_copy["MOSEKLM_LICENSE_FILE"] : ""
        ENV["HOME"] = haskey(env_copy, "HOME") ? env_copy["HOME"] : ""
        ENV["PROFILE"] = haskey(env_copy, "PROFILE") ? env_copy["PROFILE"] : ""
    end

    @testset "modifying ENV to be different license states" begin
        ENV["MOSEKLM_LICENSE_FILE"] = ""
        ENV["HOME"] = ""
        ENV["PROFILE"] = ""
        @test !hasMOSEKLicense()

        ENV["MOSEKLM_LICENSE_FILE"] = joinpath(test_dir,"mosek","mosek.lic")
        @test hasMOSEKLicense()

        ENV["MOSEKLM_LICENSE_FILE"] = ""
        ENV["HOME"] = test_dir
        @test hasMOSEKLicense()

        ENV["HOME"] = ""
        ENV["PROFILE"] = test_dir
        @test hasMOSEKLicense()

        _restoreENV()
    end

    # NOTE: This must run last!
    # In case any tests failed, making sure environment variables are properly restored
    _restoreENV()
end

end
