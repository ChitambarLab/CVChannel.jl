using Test
using LinearAlgebra
using QBase
using Random
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
        x = 0
        y = 0
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
        y_step = 0.5
        x_step = 0.5
        for d in 2:4
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

@testset "haarStates" begin
    @testset "verifying states, n = $n" for n in 1:5
        @testset "d = $d" for d in 1:5
            rand_states = haarStates(n, d)

            @test length(rand_states) == n
            @test all(ρ -> size(ρ) == (d,d), rand_states)
            @test all(ρ -> is_density_matrix(ρ), rand_states)
        end
    end

    @testset "seeded random states verification" begin
        Random.seed!(666)
        rand_states = haarStates(4,2)
        match_states = [
            [
                0.9068588653033508 + 0.0im -0.2824120726122391 - 0.06862423017367089im;
                -0.2824120726122391 + 0.06862423017367089im 0.09314113469664888 + 0.0im
            ],
            [
                0.3718976279148742 + 0.0im -0.3176606642535559 + 0.36425469750183553im;
                -0.3176606642535559 - 0.36425469750183553im 0.628102372085126 + 0.0im
            ],
            [
                0.4049640512580946 + 0.0im -0.1688517083280211 - 0.46093087230238206im;
                -0.1688517083280211 + 0.46093087230238206im 0.5950359487419055 + 0.0im
            ],
            [
                0.17921064730550684 + 0.0im -0.3426052023877003 + 0.17238290661991693im;
                -0.3426052023877003 - 0.17238290661991693im 0.8207893526944933 + 0.0im
            ]
        ]

        @test all(i -> rand_states[i] ≈ match_states[i], 1:4)
    end
end

end
