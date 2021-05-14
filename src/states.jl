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
