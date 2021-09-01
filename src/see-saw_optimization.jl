"""
    fixedStateCV(
        states :: Vector{<:AbstractMatrix},
        kraus_ops :: Vector{<:AbstractMatrix}
    ) :: Tuple{Float64, Vector{Matrix{ComplexF64}}}

For a fixed ensemble of `states` and quantum channel described by `kraus_ops`,
the communication value (CV) is evaluated by maximizing over all POVM measurments.
This optimization is expressed in primal form as the semi-definite program:

```math
\\begin{matrix}
    & \\max_{\\{\\Pi_x\\}_{x}} \\sum_x \\text{Tr}\\left[\\Pi_x\\mathcal{N}(\\rho_x)\\right] \\\\
    & \\\\
    & \\text{s.t.} \\quad \\sum_x\\Pi_x = \\mathbb{I} \\;\\; \\text{and} \\;\\; \\Pi_x \\geq 0
\\end{matrix}
```

where each state ``\\rho_x`` satisfies ``\\text{Tr}[\\rho_x] = 1`` and ``\\rho_x \\geq 0``.
The channel ``\\mathcal{N}`` is applied to each state as
``\\mathcal{N}(\\rho_x) = \\sum_j k \\rho_x k^\\dagger``.

# Returns

A `Tuple` whose first element is the evaluated CV and second element is the
optimal POVM measurement that achieve the maximal CV.
"""
function fixedStateCV(
    states :: Vector{<:AbstractMatrix},
    kraus_ops :: Vector{<:AbstractMatrix}
) :: Tuple{Float64, Vector{Matrix{ComplexF64}}}
    d = size(kraus_ops[1],1)
    n = length(states)

    # add povm variables and constraints
    povm_vars = map(i -> HermitianSemidefinite(d), 1:n)
    constraints = sum(map(Πy -> real(Πy), povm_vars)) == Matrix{Float64}(I, d, d)
    constraints += sum(map(Πy -> imag(Πy), povm_vars)) == zeros(Float64, d, d)

    # apply channel to states
    evolved_states = map(ρ ->  sum(k -> k * ρ * k' , kraus_ops), states)

    # maximize CV over POVM measurements
    objective = maximize(real(sum(tr.(povm_vars .* evolved_states))), constraints)
    qsolve!(objective)

    cv = objective.optval
    opt_povm = map(Πy -> Πy.value, povm_vars)

    return cv, opt_povm
end

"""
    fixedMeasurementCV(
        povm :: Vector{<:AbstractMatrix},
        kraus_ops :: Vector{<:AbstractMatrix}
    ) :: Tuple{Float64, Vector{Matrix{ComplexF64}}}

For a fixed `povm` measurement and quantum channel described by `kraus_ops`, the
communication value (CV) is evaluated by maximizing over all state encodings.
This optimization is expressed in primal form as the semi-definite program:

```math
\\begin{matrix}
    & \\max_{\\{\\rho_x\\}_{x}} \\sum_x \\text{Tr}\\left[\\mathcal{N}^\\dagger(\\Pi_x)\\rho_x\\right] \\\\
    & \\\\
    & \\text{s.t.} \\quad \\text{Tr}[\\rho_x] = 1 \\;\\; \\text{and} \\;\\; \\rho_x \\geq 0
\\end{matrix}
```

where ``\\{\\Pi_y \\}_y`` is a POVM and the adjoint channel
``\\mathcal{N}^\\dagger`` is applied to each POVM element as
``\\mathcal{N}^\\dagger (\\Pi_y) = \\sum_j k^\\dagger \\Pi_y k``.

# Returns

A `Tuple` whose first element is the evaluated CV and second element is the
optimal set of states that achieve the maximal CV.
"""
function fixedMeasurementCV(
    povm :: Vector{<:AbstractMatrix},
    kraus_ops :: Vector{<:AbstractMatrix}
) :: Tuple{Float64, Vector{Matrix{ComplexF64}}}
    d = size(kraus_ops[1],2)
    n = length(povm)

    # add state variables and constraints
    state_vars = map(i -> HermitianSemidefinite(d), 1:n)
    constraints = map(ρx -> tr(ρx) == 1, state_vars)

    # apply adjoint channel to POVM
    evolved_povm = map(Π ->  sum(k -> k' * Π * k , kraus_ops), povm)

    # maximize CV over state encodings
    objective = maximize(real(sum(tr.(evolved_povm .* state_vars))), constraints)
    qsolve!(objective)

    cv = objective.optval
    opt_states = map(ρx -> ρx.value, state_vars)

    return cv, opt_states
end

"""
    seesawCV(
        init_states :: Vector{<:AbstractMatrix},
        kraus_ops :: Vector{<:AbstractMatrix},
        num_steps :: Int64;
        verbose :: Bool = false
    )

Performs the see-saw optimization technique to maximize the communication
value (CV) of the channel described by `kraus_ops` over all states and measurements.
This iterative and variational optimization technique combines coordinate ascent
maximization with semi-definite programming.
The number of iterations is determined by `num_steps` where each iteration
consists of a two-step procedure:

1. The POVM measurement is optimized with respect to a fixed state ensemble
   using the [`fixedStateCV`](@ref) function.
2. The state ensemble is optimized with respect to a fixed povm state using
   the [`fixedMeasurementCV`](@ref) function.

This procedure is initialized with `init_states` and after many iterations, a
local maximum of the CV is found.
The `verbose` keyword argument can be used to print out the CV evaluated in each
step.

The see-saw method was first introduced in the field of quantum nonlocality by
[https://arxiv.org/abs/quant-ph/0102024](https://arxiv.org/abs/quant-ph/0102024)
and has been developed further by several other works including
[https://arxiv.org/abs/quant-ph/0604045](https://arxiv.org/abs/quant-ph/0604045),
[https://arxiv.org/abs/quant-ph/0508210](https://arxiv.org/abs/quant-ph/0508210),
and [https://arxiv.org/abs/1006.3032](https://arxiv.org/abs/1006.3032).
We note that our implementation is quite distinct from previous works.
This technique is typically applied to space-like separated Bell scenarios and
often restricted to optimization over projective measurements.


# Returns

A `Tuple` containing the following data in order:

1. `max_cv_tuple :: Tuple`, `(max_cv, opt_states, opt_povm)` A 3-tuple containing the maximal
   communication value and the optimal states/povm that achieve this value.
2. `cvs :: Vector{Float64}`, A list of each evaluated CV. Since states and measurements
   are optimized in each iteration, we have `length(cvs) == 2 * num_steps`.
3. `opt_ensembles :: Vector{Vector{Matrix{ComplexF64}}}`, A list of state ensembles
   optimized in each step, where `length(opt_ensembles) == num_steps`.
4. `opt_povms :: Vector{Vector{Matrix{ComplexF64}}}`, A list of POVM measurements
   optimized in each step, where `length(opt_povms) == num_steps`.

!!! warning "Local Optima"
    This function is not guaranteed to find global optima.
"""
function seesawCV(
    init_states :: Vector{<:AbstractMatrix},
    kraus_ops :: Vector{<:AbstractMatrix},
    num_steps :: Int64;
    verbose :: Bool = false
) :: Tuple
    opt_ensembles = Vector{Vector{Matrix{ComplexF64}}}(undef, num_steps)
    opt_povms = Vector{Vector{Matrix{ComplexF64}}}(undef, num_steps)
    cvs = zeros(Float64, 2*num_steps)
    cv_id = 1

    max_cv_tuple = (1, [], [])

    for i in 1:num_steps
        # maximizing CV over POVMs
        opt_states = (i == 1) ? init_states : opt_ensembles[i-1]

        fixed_state_cv, opt_povm = fixedStateCV(opt_states, kraus_ops)

        cvs[cv_id] = fixed_state_cv
        cv_id += 1
        opt_povms[i] = opt_povm

        if fixed_state_cv > max_cv_tuple[1]
            max_cv_tuple = (fixed_state_cv, opt_states, opt_povm)
        end

        # maximizing CV over states
        fixed_povm_cv, opt_states = fixedMeasurementCV(opt_povm, kraus_ops)

        cvs[cv_id] = fixed_povm_cv
        cv_id += 1
        opt_ensembles[i] = opt_states

        if fixed_povm_cv > max_cv_tuple[1]
            max_cv_tuple = (fixed_povm_cv, opt_states, opt_povm)
        end

        if verbose
            println("i = ", i)
            println("fixed_state_cv = ", fixed_state_cv)
            println("fixed_povm_cv = ", fixed_povm_cv)
            println("max_cv = ", max_cv_tuple[1])
        end
    end

    return max_cv_tuple, cvs, opt_ensembles, opt_povms
end
