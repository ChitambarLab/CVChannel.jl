using Test, LinearAlgebra, Convex, Suppressor
using CVChannel

@testset "./src/optimizer_interface.jl" begin

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

@testset "hasMOSEKLicense()" begin
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

@testset "useSCS() and useMOSEK()" begin
    ENV["MOSEKLM_LICENSE_FILE"] = ""
    ENV["HOME"] = test_dir
    ENV["PROFILE"] = ""
    @test hasMOSEKLicense()

    local return_val    # needed to probe inside @capture_err scope
    err = @capture_err return_val = useSCS()
    @test occursin(r"Warning: Using SCS backend\.", err)
    @test !CVChannel._USE_MOSEK
    @test !return_val

    err = @capture_err return_val = useMOSEK()
    @test occursin(r"Warning: Using MOSEK backend\.", err)
    @test CVChannel._USE_MOSEK
    @test return_val

    # repeating SCS test now that we know the value of CVChannel._USE_MOSEK
    err = @capture_err return_val = useSCS()
    @test occursin(r"Warning: Using SCS backend\.", err)
    @test !CVChannel._USE_MOSEK
    @test !return_val

    ENV["HOME"] = @__DIR__
    @test !hasMOSEKLicense()

    err = @capture_err return_val = useMOSEK()
    @test occursin(r"Warning: No MOSEK license found\. Using SCS backend\.", err)
    @test !return_val
    @test !CVChannel._USE_MOSEK

    _restoreENV()
end

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
        @suppress_err useSCS()
        @test !CVChannel._USE_MOSEK

        (ρ, objective, constraints) = _state_optimization()
        problem = maximize(objective, constraints)

        qsolve!(problem)

        @test problem.optval ≈ 1
        @test ρ.value ≈ [0.5 0.5;0.5 0.5]
    end

    @testset "runs with mosek backend" begin
        ENV["MOSEKLM_LICENSE_FILE"] = joinpath(test_dir,"mosek","mosek.lic")
        @suppress_err useMOSEK()
        @test CVChannel._USE_MOSEK

        (ρ, objective, constraints) = _state_optimization()
        problem = maximize(objective, constraints)

        qsolve!(problem)

        @test problem.optval ≈ 1
        @test isapprox(ρ.value, [0.5 0.5;0.5 0.5], atol=1e-5)

        @suppress_err useSCS()
        _restoreENV()
    end
end

# NOTE: This must run last!
# In case any tests failed, making sure environment variables are properly restored
_restoreENV()

end
