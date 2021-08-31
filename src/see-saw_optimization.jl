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
