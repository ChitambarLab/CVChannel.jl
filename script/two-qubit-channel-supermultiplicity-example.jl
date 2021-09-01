using LinearAlgebra: tr
using Test
using Base.Iterators: flatten
using QBase: States, Unitaries
using BellScenario: LocalSignaling, BellGame, Nonlocality
using SignalingDimension: maximum_likelihood_facet

# TODO: revisit this script to drop Signaling Dimension and BellScenario

"""
In a previous note, Marius Junge described a channel that has a supermultiplicative
communication value. This channel maps a 4-dimension Hilbert
space to a 5-dimension Hilbert space and its kraus operators are constructed with
the following method.
"""
function marius_example_kraus_operators()
    σ = Unitaries.paulis
    Id = [1 0;0 1]

    k1 = kron(σ[1], Id)
    k2 = kron(σ[2], Id)
    k3 = kron(σ[3], σ[1])
    k4 = kron(σ[3], σ[2])
    k5 = kron(σ[3], σ[3])

    [k1,k2,k3,k4,k5]
end

"""
The singular channel is implemented with the following method. The input `ρ_set` must be an
array of 4-dimensional quantum states.
"""
function marius_example_channel(ρ_set :: Vector{<:States.AbstractDensityMatrix}) :: Vector{States.DensityMatrix}
    c = marius_example_kraus_operators()
    basis = States.basis_kets(5)

    channel(ρ) = States.DensityMatrix(
        sum( r -> sum( s -> tr(c[r]'*ρ*c[s])*basis[r]*basis[s]', 1:5), 1:5)/5
    )

    channel.(ρ_set)
end

"""
The product of two instances of the channel is considered. The input `ρ_set` must be an
array of 16-dimensional quantum states.
"""
function marius_example_channel_prod(ρ_set :: Vector{<:States.AbstractDensityMatrix}) :: Vector{States.DensityMatrix}
    k_ops = marius_example_kraus_operators()
    k_prod_ops = collect(flatten(map(k_a -> map(k_b -> kron(k_a, k_b), k_ops), k_ops)))
    basis = States.basis_kets(25)

    channel(ρ) = States.DensityMatrix(
        sum( r -> sum( s -> tr(k_prod_ops[r]'*ρ*k_prod_ops[s])*basis[r]*basis[s]', 1:25), 1:25)/25
    )

    channel.(ρ_set)
end

"""
    generalized_bell_states_im(dim :: Int64) :: Vector{States.DensityMatrix}

Constructs the generalized Bell basis with a phase for arbitrary `dim ≥ 2`.
"""
function generalized_bell_states_im(dim :: Int64) :: Vector{States.DensityMatrix}
    if !(dim ≥ 2)
        throw(DomainError(dim, "hilbert space dimension must satisfy `dim ≥ 2`"))
    end

    basis = States.basis_kets(dim)

    kets = map(
        c -> map(
            p -> States.Ket(sum(
                j -> exp(im*π*p*j/(dim))*kron(basis[j+1],basis[mod(j + c, dim) + 1]),
                0:dim-1
            )/sqrt(dim)),
        0:dim-1),
    0:dim-1)

    States.pure_state.(collect(flatten(kets)))
end

"""
    bloch_antipodal_states()

Constructs three pairs of antipodal states where each pair lies along x-, y-, and
z-axes of bloch sphere.
"""
function bloch_antipodal_states()
    ρ_bloch_x1 = States.bloch_qubit(1,0,0)
    ρ_bloch_x2 = States.bloch_qubit(-1,0,0)
    ρ_bloch_y1 = States.bloch_qubit(0,1,0)
    ρ_bloch_y2 = States.bloch_qubit(0,-1,0)
    ρ_bloch_z1 = States.bloch_qubit(0,0,1)
    ρ_bloch_z2 = States.bloch_qubit(0,0,-1)

    [ρ_bloch_x1, ρ_bloch_x2, ρ_bloch_y1, ρ_bloch_y2, ρ_bloch_z1, ρ_bloch_z2]
end

# """
# Claim: the communication value of a single channel use of `marius_example_channel`
# does not exceed 2.
#
# The following tests demonstrate examples that support this claim.
# """

println("\nVerifying communication value of a single channel use...")

@time @testset "communication value of single channel use" begin
    @testset "computational basis states" begin
        states = marius_example_channel(States.basis_states(4))

        X = 4
        Y = 5
        d = 5

        scenario = LocalSignaling(X,Y,d)
        G_ML = BellGame([1 0 0 0;1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1], d)

        opt_dict = Nonlocality.optimize_measurement(scenario, G_ML, states)
        cv = d + opt_dict["violation"]

        @test isapprox(cv, 1.80, atol=1e-5)
    end

    @testset "bell states" begin
        states = marius_example_channel(States.bell_states)

        X = 4
        Y = 5
        d = 5

        scenario = LocalSignaling(X,Y,d)
        G_ML = BellGame([1 0 0 0;1 0 0 0;0 1 0 0;0 0 1 0;0 0 0 1], d)

        opt_dict = Nonlocality.optimize_measurement(scenario, G_ML, states)
        cv = d + opt_dict["violation"]

        @test isapprox(cv, 1.8, atol=1e-5)
    end

    @testset "computational basis extended with bell states" begin
        states = marius_example_channel([
            States.basis_states(4)...,
            States.bell_states...
        ])

        X = 8
        Y = 8
        d = 5

        scenario = LocalSignaling(X,Y,d)
        G_ML = maximum_likelihood_facet(Y, d)

        opt_dict = Nonlocality.optimize_measurement(scenario, G_ML, states)
        cv = d + opt_dict["violation"]

        @test isapprox(cv, 1.8, atol=1e-5)
    end

    @testset "bloch states" begin
        bloch_prod_states = kron.(bloch_antipodal_states(), bloch_antipodal_states())
        states = marius_example_channel(bloch_prod_states)

        X = 6
        Y = 6
        d = 5

        scenario = LocalSignaling(X,Y,d)
        G_ML = maximum_likelihood_facet(Y, d)

        opt_dict = Nonlocality.optimize_measurement(scenario, G_ML, states)
        cv = d + opt_dict["violation"]

        @test isapprox(cv, 2, atol=1e-5)
    end
end

# """
# Claim: The product channel `marius_example_channel_prod` has a communication value
# exceeding 4.
#
# If the singular use of the channel has mamximum communication value of 2, then
# the following tests verify supermultiplicity
# """

println("\nVerifying supermultiplicity of communication value of the product channel...")

@time @testset "communication value of double channel use" begin
    @testset "real + im generalized bell states do not show supermultiplicity" begin
        im_bell_states = generalized_bell_states_im(4)
        bell_states = States.generalized_bell_states(4)

        states = marius_example_channel_prod([bell_states..., im_bell_states...])

        X = 32
        Y = 32
        d = 25

        scenario = LocalSignaling(X,Y,d)
        G_ML = maximum_likelihood_facet(Y, d)

        opt_dict = Nonlocality.optimize_measurement(scenario, G_ML, states)
        cv = d + opt_dict["violation"]

        @test isapprox(cv, 3.09568, atol=1e-5)
    end

    @testset "product combinations of bloch antipodal states is multiplicative" begin
        bloch_prod_states = kron.(bloch_antipodal_states(), bloch_antipodal_states())
        bloch_combo_states = collect( flatten(
            map( ρa -> map( ρb -> kron(ρa, ρb), bloch_prod_states), bloch_prod_states)
        ))

        states = marius_example_channel_prod(bloch_combo_states)

        X = 36
        Y = 36
        d = 25

        scenario = LocalSignaling(X,Y,d)
        G_ML = maximum_likelihood_facet(Y, d)

        opt_dict = Nonlocality.optimize_measurement(scenario, G_ML, states)
        cv = d + opt_dict["violation"]

        @test isapprox(cv, 4, atol=1e-5)
    end

    println("Verifying supermultiplicity example, this may take a few minutes...")
    @time @testset "supermultiplicity of antipodal bloch states and generalized bell states" begin
        bloch_prod_states = kron.(bloch_antipodal_states(), bloch_antipodal_states())
        bloch_combo_states = collect( flatten(
            map( ρa -> map( ρb -> kron(ρa, ρb), bloch_prod_states), bloch_prod_states)
        ))
        bell_states = States.generalized_bell_states(4)

        states = marius_example_channel_prod([bloch_combo_states..., bell_states...])

        X = 52
        Y = 52
        d = 25

        scenario = LocalSignaling(X,Y,d)
        G_ML = maximum_likelihood_facet(Y, d)

        @time opt_dict = Nonlocality.optimize_measurement(scenario, G_ML, states)
        cv = d + opt_dict["violation"]

        @test isapprox(cv, 4.10539, atol=1e-6)
    end
end
