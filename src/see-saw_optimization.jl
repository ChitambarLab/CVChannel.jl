"""
    fixedStateCV(
        states :: Vector{<:AbstractMatrix},
        kraus_ops :: Vector{<:AbstractMatrix}
    ) :: Tuple{Float64, Vector{Matrix{ComplexF64}}}

For a fixed ensemble of `states` and quantum channel described by `kraus_ops`,
the communication value (CV) is evaluated by maximizing over all POVM measurements.
This optimization is expressed in primal form as the semidefinite program:

```math
\\begin{matrix}
    \\text{fixedStateCV}(\\mathcal{N})&= \\max_{\\{\\Pi_x\\}_{x}} \\sum_x \\text{Tr}\\left[\\Pi_x\\mathcal{N}(\\rho_x)\\right] \\\\
    & \\\\
    & \\text{s.t.} \\quad \\sum_x\\Pi_x = \\mathbb{I} \\;\\; \\text{and} \\;\\; \\Pi_x \\geq 0
\\end{matrix}
```

where each state ``\\rho_x`` satisfies ``\\text{Tr}[\\rho_x] = 1`` and ``\\rho_x \\geq 0``.
The channel ``\\mathcal{N}`` is applied to each state as
``\\mathcal{N}(\\rho_x) = \\sum_j k \\rho_x k^\\dagger``.

# Returns

A `Tuple`, `(cv, opt_povm)` where `cv` is the evaluated communication value and
`opt_povm` is the optimal POVM measurement.
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
communication value (CV) and optimal state encodings are computed.
The fixed measurement CV is evaluated as

```math
\\text{fixedMeasurementCV}(\\mathcal{N}) = \\sum_y ||\\mathcal{N}^{\\dagger}(\\Pi_y)||_{\\infty}
```

where ``||\\mathcal{N}^{\\dagger}(\\Pi_y)||_{\\infty}`` is the largest eigenvalue
of the POVM element ``\\Pi_y`` evolved by the adjoint channel,
``\\mathcal{N}^{\\dagger}(\\Pi_y) = \\sum_j k^{\\dagger}_j \\Pi_y k_j``.
The states which maximize the CV are simply the eigenvectors corresponding to the
largest eigenvalue of each respective POVM element.

# Returns

A `Tuple`, `(cv, opt_states)` where `cv` is the communication value and
`opt_states` is the set of optimal states.
"""
function fixedMeasurementCV(
    povm :: Vector{<:AbstractMatrix},
    kraus_ops :: Vector{<:AbstractMatrix}
) :: Tuple{Float64, Vector{Matrix{ComplexF64}}}
    d = size(kraus_ops[1],2)
    n = length(povm)

    # apply adjoint channel to POVM
    evolved_povm = map(Π ->  sum(k -> k' * Π * k , kraus_ops), povm)


    opt_states = Vector{Matrix{ComplexF64}}(undef, n)
    cv = 0
    for i in 1:n
        povm_el = evolved_povm[i]
        vals, vecs = eigen(povm_el)

        opt_states[i] = vecs[:,end] * vecs[:,end]'
        cv += real(vals[end])
    end

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
This iterative and biconvex optimization technique combines coordinate ascent
maximization with semidefinite programming.
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

The see-saw method has shown success in similar encoding/decoding
optimization problems in quantum information, *e.g.*,
[https://arxiv.org/abs/quant-ph/0307138v2](https://arxiv.org/abs/quant-ph/0307138v2)
and [https://arxiv.org/abs/quant-ph/0606078v1](https://arxiv.org/abs/quant-ph/0606078v1).
We note that our implementation is quite distinct from previous works, however,
the core iterative approach remains the same.

# Returns

A `Tuple` containing the following data in order:

1. `max_cv_tuple :: Tuple`, `(max_cv, opt_states, opt_povm)` A 3-tuple containing the maximal
   communication value and the optimal states/POVM that achieve this value.
2. `cvs :: Vector{Float64}`, A list of each evaluated CV. Since states and measurements
   are optimized in each iteration, we have `length(cvs) == 2 * num_steps`.
3. `opt_ensembles :: Vector{Vector{Matrix{ComplexF64}}}`, A list of state ensembles
   optimized in each step, where `length(opt_ensembles) == num_steps`.
4. `opt_povms :: Vector{Vector{Matrix{ComplexF64}}}`, A list of POVM measurements
   optimized in each step, where `length(opt_povms) == num_steps`.

!!! warning "Optimum Not Guaranteed"
    This function is not guaranteed to find a global or local optima. However,
    `seesawCV` will always provide a lower bound on the communication value.
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
