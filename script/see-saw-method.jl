using Convex, SCS
using CVChannel
using QBase
using LinearAlgebra


function optimize_measurement(ρ_states::Vector{<:AbstractMatrix}, kraus_ops :: Vector{<:AbstractMatrix}) :: Dict
    d = size(kraus_ops[1],1)
    n = length(ρ_states)

    # add povm variables and constraints
    Π_vars = map(i -> HermitianSemidefinite(d), 1:n)
    constraints = (sum(map(Π_y -> real(Π_y), Π_vars)) == Matrix{Float64}(I, d, d))
    constraints += (sum(map(Π_y -> imag(Π_y), Π_vars)) == zeros(Float64, d, d))


    # apply channel here
    ρ_states_ev = map(ρ ->  sum(k -> k*ρ *k' , kraus_ops), ρ_states)

    # add the objective
    objective = maximize(real(sum(tr.(Π_vars .* ρ_states_ev))), constraints)

    # optimize model
    solve!(objective, SCS.Optimizer(verbose=0))

    # parse/return results
    score = objective.optval
    opt_povm = map(Πy -> Πy.value, Π_vars)

    Dict(
        "score" => score,
        "opt_povm" => opt_povm,
        "states" => ρ_states,
        "states_ev" => ρ_states_ev,
    )
end

function optimize_states(povm::Vector{<:AbstractMatrix}, kraus_ops :: Vector{<:AbstractMatrix}) :: Dict
    d = size(kraus_ops[1],2)
    n = length(povm)

    # add povm variables and constraints
    ρ_vars = map(i -> HermitianSemidefinite(d), 1:n)
    constraints = map(ρ_x -> tr(ρ_x) == 1, ρ_vars)

    # apply channel here
    povm_ev = map(Π ->  sum(k -> k'*Π *k , kraus_ops), povm)

    # add the objective
    objective = maximize(real(sum(tr.(povm_ev.* ρ_vars))), constraints)

    # optimize model
    solve!(objective, SCS.Optimizer(verbose=0))

    # parse/return results
    score = objective.optval
    opt_states = map(ρx -> ρx.value, ρ_vars)

    Dict(
        "score" => score,
        "opt_states" => opt_states,
        "povm" => povm,
        "povm_ev" => povm_ev,
    )
end

function see_saw_optimization(init_states, kraus_ops, num_steps) :: Dict
    states = init_states
    povm = []
    scores = zeros(Float64, length(states))

    for i in 1:num_steps

        povm_opt_dict = optimize_measurement(states, kraus_ops)
        povm = povm_opt_dict["opt_povm"]

        states_opt_dict = optimize_states(povm, kraus_ops)
        states = states_opt_dict["opt_states"]

        push!(scores, states_opt_dict["score"])

        println("i = ", i)
        println("score = ", states_opt_dict["score"])
    end

    return Dict(
        "states" => states,
        "povm" => povm,
        "scores" => scores
    )
end

function haar_states(n, d)
    ρ0 = zeros(Float64, d, d)
    ρ0[1,1] = 1.0

    unitaries = map(i -> random_unitary(d), 1:n)

    map(U -> U*ρ0*U', unitaries)
end

kraus_ops1 = [[1 0;0 1]*sqrt(0.7), [0 1;1 0]*sqrt(0.3)]

init_states = haar_states(3,2)

opt_dict = see_saw_optimization(init_states, kraus_ops1, 10)

# testing werner holevo results
d = 7
werner_choi = wernerState(d, 0)*d
werner_kraus = map(i -> reshape(werner_choi[:,i], (d,d)), 1:d^2)*sqrt((d-1)/2)

sum(k -> k*k', werner_kraus)

init_states = haar_states(d,d)

opt_dict = see_saw_optimization(init_states, werner_kraus, 10)

naive_werner_kraus_prod = Vector{Array{Float64,2}}(undef,0)
for k1 in werner_kraus, k2 in werner_kraus
    push!(naive_werner_kraus_prod, kron(k1,k2))
end

prod_init_states = haar_states(d^2,d^2)

opt_dict = see_saw_optimization(prod_init_states, naive_werner_kraus_prod, 10)



anti_sym_WH(ρ) = wernerHolevoChannel(ρ, 0)
anti_sym_WH_choi(d) = real.(choi(anti_sym_WH, d, d))

anti_sym_WH_choi(4)
