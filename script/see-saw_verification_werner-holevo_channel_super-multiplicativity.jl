using Base.Iterators
using CVChannel
using LinearAlgebra
using QBase
using Random
using Test

"""
This script verifies that the see-saw method can reproduce the known super-multiplicativity
of the anti-symmetric Werner-Holevo channel.
We demonstrate the see-saw method's ability to scale to dimensions beyond what can be found
using the PPT relaxation on the Choi operator approach to the communication value (CV).
The verification procedure below takes about 10-20 minutes to complete.
"""

@time @testset "see-saw verification of anti-symmetric werner-holevo channel super-multiplicativiity" for d in 2:6
@time @testset "(d = $d)" begin
    Random.seed!(111) # seeding random numbers for reproducibility

    println("\nd = ", d)
    anti_sym_choi = wernerState(d, 0) * d
    nonzero_ids = filter(i -> anti_sym_choi[:,i] != zeros(d^2), 1:d^2)
    anti_sym_kraus_ops = map(i -> reshape(anti_sym_choi[:,i], (d,d)), nonzero_ids)*sqrt((d-1)/2)
    par_anti_sym_kraus_ops = collect(flatten(
        map(k1 -> map(k2 -> kron(k1,k2), anti_sym_kraus_ops), anti_sym_kraus_ops)
    ))

    @test sum(k -> k' * k, anti_sym_kraus_ops) ≈ I
    @test choi(anti_sym_kraus_ops) ≈ anti_sym_choi
    @test sum(k -> k' * k, par_anti_sym_kraus_ops) ≈ I

    # warning: this computation is slow for d=6 and larger
    @test choi(par_anti_sym_kraus_ops) ≈ parChoi(
        Choi(anti_sym_choi, d, d), Choi(anti_sym_choi, d, d)
    ).JN

    init_states = haarStates(d, d)
    max_cv_tuple, = seesawCV(init_states, anti_sym_kraus_ops, 3)

    @test max_cv_tuple[1] ≈ d/(d-1) atol=1e-6
    @test all(ρ -> is_density_matrix(ρ, atol=1e-6), max_cv_tuple[2])
    @test is_povm(max_cv_tuple[3], atol=1e-6)

    par_init_states = haarStates(d^2, d^2)
    par_max_cv_tuple, = seesawCV(
        par_init_states,
        par_anti_sym_kraus_ops,
        5,
    )

    @test par_max_cv_tuple[1] ≈ 2*d/(d-1) atol=5e-2
    @test all(ρ -> is_density_matrix(ρ, atol=1e-5), par_max_cv_tuple[2])
    @test is_povm(par_max_cv_tuple[3], atol=1e-5)

    println("max_cv : ", max_cv_tuple[1], ", match : ", d/(d-1))
    println("par_max_cv : ", par_max_cv_tuple[1], ", match : ", 2*d/(d-1), "\n")
end
end
