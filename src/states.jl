"""
    wernerState(dim :: Int, p ::Union{Int,Float64}) :: Matrix{Float64}
This function constructs the Werner states,
```math
    \\sigma_{d,p} = p \\frac{\\Pi_{0}}{d+1 \\choose 2} + (1-p) \\frac{\\Pi_{1}}{d \\choose 2}
```
where ``p \\in [0,1]`` and ``\\Pi_0, \\Pi_1`` are the projectors onto the symmetric and anti-symmetric
subspaces respectively. They can be determined by
```math
    \\Pi_0 = \\frac{1}{2} (I_{A} \\otimes I_{B} + \\mathbb{F}) \\hspace{1cm} \\Pi_1 = \\frac{1}{2}(I_{A} \\otimes I_{B} - \\mathbb{F})
```
where ``\\mathbb{F}`` is the swap operator.

A `DomainError` is thrown if:
* `d ≤ 1`
* `p` is not in range `0 ≤ p ≤ 1`
"""
function wernerState(d :: Int, p ::Union{Int,Float64}) :: Matrix{Float64}
    if d ≤ 1
        throw(DomainError(d,"wernerState requires the local dimension is two or greater."))
    elseif !(0 ≤ p ≤ 1)
        throw(DomainError(p,"wernerState requires p ∈ [0,1]."))
    end
    swap = swapOperator(d)
    Π0 = (I + swap)/2
    Π1 = (I - swap)/2
    return p * Π0 / binomial(d+1,2) + (1-p) * Π1 / binomial(d,2)
end

"""
    axisymmetricState(
        d :: Int64,
        x :: Union{Int,Float64},
        y :: Union{Int,Float64}
    ) :: Matrix{Float64}

Construct the axisymmetric state ``\\rho^{\\text{axi}}`` as described in section
IV.C. of [this paper](https://arxiv.org/pdf/1505.01833.pdf).
This state is a bipartite quantum state with each subspace having dimension `d`.
The diagonal of ``\\rho^{\\text{axi}}`` is parameterized as

```math
    \\rho^{\\text{axi}}_{jj,jj} = \\frac{1}{d^2}+a, \\quad \\text{and}\\quad \\rho^{\\text{axi}}_{jk,jk} = \\frac{1}{d^2}- \\frac{a}{d-1} \\; (j\\neq k)
```

while the off-diagonals are expressed as

```math
    \\rho^{\\text{axi}}_{jj,kk} = b,
```

where ``a = y\\frac{d-1}{d}`` and ``b = \\frac{x}{\\sqrt{d(d-1)}}``.
All remaining elements of ``\\rho^{\\text{axi}}`` are zero.
Inputs `x` and `y` of the `axisymmetricState` function parameterize ``\\rho^{\\text{axi}}``
and are constrained as:
* ``-\\frac{1}{d\\sqrt{d-1}} \\leq y \\leq \\frac{\\sqrt{d-1}}{d}``
* ``-\\frac{1}{\\sqrt{d(d-1)}}\\leq x \\leq \\sqrt{\\frac{d-1}{d}}``
* ``-\\frac{1}{\\sqrt{d}}\\left( y + \\frac{1}{d\\sqrt{d-1}}\\right) \\leq x \\leq \\frac{d-1}{\\sqrt{d}}\\left(y + \\frac{1}{d\\sqrt{d-1}} \\right)``

If any of the constraints above do not hold, a `DomainError` is thrown.
"""
function axisymmetricState(
    d :: Int64,
    x :: Union{Int,Float64},
    y :: Union{Int,Float64}
) :: Matrix{Float64}
    y_bounds = _axisymmetric_y_bounds(d)
    x_bounds = _axisymmetric_x_bounds(d)
    x_constraints = _axisymmetric_x_constraints(d,y)

    if !(y_bounds[1] ≤ y ≤ y_bounds[2])
        throw(DomainError(y, "input `y` does not satisfy `$(y_bounds[1]) ≤ y ≤ $(y_bounds[2])`."))
    elseif !(x_bounds[1] ≤ x ≤ x_bounds[2])
        throw(DomainError(x, "input `x` does not satisfy `$(x_bounds[1]) ≤ x ≤ $(x_bounds[2])`."))
    elseif !(x_constraints[1] ≤ x ≤ x_constraints[2])
        throw(DomainError(x, "input `x` does not satisfy `$(x_constraints[1]) ≤ x ≤ $(x_constraints[2])`."))
    elseif d ≤ 1
        throw(DomainError(d, "input `d` must satisfy `d ≥ 2`."))
    end

    a = y*sqrt(d-1)/d
    b = x/sqrt(d*(d-1))

    ρ_axi = zeros(Float64, d, d, d, d)
    for j in 1:d, k in 1:d
        if j == k
            ρ_axi[j,j,j,j] = a + 1/d^2
        else
            ρ_axi[j,k,j,k] = 1/d^2 - a/(d-1)
            ρ_axi[j,j,k,k] = b
        end
    end

    return reshape(ρ_axi, d^2, d^2)
end

# Helper for axisymmetricState, implements Eq. (42a) of https://arxiv.org/pdf/1505.01833.pdf
function _axisymmetric_y_bounds(d :: Int64) :: Tuple{Float64, Float64}
    return (-1/(d*sqrt(d-1)), sqrt(d-1)/d)
end

# Helper for axisymmetricState, implements Eq. (42b) of https://arxiv.org/pdf/1505.01833.pdf
function _axisymmetric_x_bounds(d :: Int64) :: Tuple{Float64, Float64}
    return (-1/sqrt(d*(d-1)), sqrt((d-1)/d))
end

# Helper for axisymmetricState, implements Eq. (43) of https://arxiv.org/pdf/1505.01833.pdf
function _axisymmetric_x_constraints(d,y) :: Tuple{Float64, Float64}
    lower = -1/sqrt(d)*(y + 1/(d*sqrt(d-1)))
    upper = (d-1)/sqrt(d)*(y+1/(d*sqrt(d-1)))

    return (lower, upper)
end
