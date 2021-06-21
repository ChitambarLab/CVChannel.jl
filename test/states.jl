using Test
using LinearAlgebra
using QBase
using CVChannel

@testset "./src/states.jl" begin

@testset "wernerState" begin
    @test isapprox(wernerState(2,1),[1/3 0 0 0 ; 0 1/6 1/6 0 ; 0 1/6 1/6 0 ; 0 0 0 1/3], atol = 1e-6)
    @test isapprox(wernerState(2,0),[0 0 0 0 ; 0 1/2 -1/2 0 ; 0 -1/2 1/2 0 ; 0 0 0 0], atol = 1e-6)
    @test isapprox(wernerState(2,1/2),[1/6 0 0 0 ; 0 1/3 -1/6 0 ; 0 -1/6 1/3 0 ; 0 0 0 1/6], atol = 1e-6)

    @testset "errors" begin
        @test_throws DomainError wernerState(1, 0.5)
        @test_throws DomainError wernerState(2, -.1)
        @test_throws DomainError wernerState(2, 1.1)
    end
end

@testset "axisymmetricState" begin
    @testset "x=0., y=0. is maximally mixed" begin
        x = 0.
        y = 0.
        for d in 2:8
            d = 2
            ρ_axi = axisymmetricState(d,x,y)
            @test ρ_axi == Matrix(I, (d^2, d^2))/d^2
        end
    end

    @testset "bell state is 'upper right corner'" begin
        ϵ = 1e-8

        for d in 2:8
            bell_state = generalized_bell_states(d)[1]

            x = CVChannel._axisymmetric_x_bounds(d)[2] - ϵ
            y = CVChannel._axisymmetric_y_bounds(d)[2]

            ρ_axi = axisymmetricState(d, x, y)
            @test ρ_axi ≈ bell_state
        end
    end

    @testset "light scan over range " begin
        y_step = 0.1
        x_step = 0.1
        for d in 2:5
            y_bounds = CVChannel._axisymmetric_y_bounds(d)
            for y in y_bounds[1]:y_step:y_bounds[2]
                x_constraints = CVChannel._axisymmetric_x_constraints(d,y)
                for x in x_constraints[1]:x_step:x_constraints[2]
                    ρ_axi = axisymmetricState(d,x,y)
                    @test is_density_matrix(ρ_axi)
                end
            end
        end
    end

    @testset "domain errors" begin
        @test_throws DomainError axisymmetricState(1,0.,0.)
        @test_throws DomainError axisymmetricState(2,0.,2.)
        @test_throws DomainError axisymmetricState(2,2.,0.)
        @test_throws DomainError axisymmetricState(2,1(2*sqrt(2))+1e-5,0.)
    end
end

end
