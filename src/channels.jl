"""
    choi(𝒩 :: Function, Σ :: Int, Λ :: Int) :: Matrix{ComplexF64}

This function returns the Choi state of a channel 𝒩. It does this using that

```math
        J(\\mathcal{N}) = \\sum_{a,b \\in \\Sigma} E_{a,b} \\otimes \\mathcal{N}(E_{a,b}) ,
```

where ``\\Sigma`` is the finite alphabet indexing the input space and ``E_{a,b}``
is a square matrix of dimension ``\\Sigma`` with a ``1`` in the ``(a,b)`` entry
and a ``0`` everywhere else. The input ``\\Lambda`` is the output dimension.
Note this assumes you have a function that calculates
``\\mathcal{N}(X)`` for arbitrary input ``X``. As many of the functions for channels
in this module have multiple parameters, please note that if you have a channel function
`𝒩(ρ, p, q)` that calculates ``\\mathcal{N}_{p,q}(\\rho)``, you can declare a function
`𝒩_xy(ρ) = 𝒩(ρ,x,y)` for fixed `(x,y)` and then call, `choi(𝒩_xy, Σ)`.
"""
function choi(𝒩 :: Function, Σ :: Int, Λ :: Int) :: Matrix{ComplexF64}
    eab_matrix = zeros(Σ,Σ)
    choi_matrix = zeros(Σ*Λ,Σ*Λ)
    for i in 1 : Σ
        for j in 1 : Σ
            eab_matrix[i,j] = 1
            choi_matrix += kron(eab_matrix,𝒩(eab_matrix))
            eab_matrix[i,j] = 0
        end
    end
    return choi_matrix
end

"""
    depolarizingChannel(ρ :: Matrix{Float64}, q :: Union{Int,Float64}) :: Matrix{ComplexF64}

This calculates the action of the depolarizing channel,

```math
\\Delta_{q}(\\rho) = (1-q)\\rho + q \\text{Tr}(\\rho) \\frac{1}{d} I_{AB} ,
```

where ``q \\in [0,1].``
Note these channels are the channels covariant with respect to the unitary group.

A `DomainError` is thrown if:
* Matrix `ρ` is not square
* Input `q` does not satisfy `0 ≤ q ≤ 1`
"""
function depolarizingChannel(ρ :: Matrix{<:Number}, q :: Union{Int,Float64}) :: Matrix{ComplexF64}
    dim = size(ρ,1)
    if !isequal(dim,size(ρ,2))
        throw(DomainError(ρ, "the input ρ is not a square matrix"))
    elseif !(0 ≤ q ≤ 1)
        throw(DomainError(q, "depolarizingChannel requires q ∈ [0,1]."))
    end

    return (1-q)*ρ  + q*(tr(ρ))*(1/(dim))*I
end

"""
    dephrasureChannel(
        ρ :: Matrix{<:Number},
        p :: Union{Int,Float64},
        q :: Union{Int,Float64}
    ) :: Matrix{ComplexF64}

This function calculates the action of the [dephrasureChannel](https://arxiv.org/abs/1806.08327),

```math
\\mathcal{N}_{p,q}( \\rho) := (1-q)((1-p) \\rho + pZ \\rho Z) + q \\text{Tr}( \\rho) |e\\rangle \\langle e|,
```

where ``p,q \\in [0,1]``, ``Z`` is the Pauli-Z matrix, and ``|e\\rangle\\langle e|``
an  error  flag orthogonal to the  Hilbert space of input state ``\\rho``.

A `DomainError` is thrown if:
* Matrix `ρ` is not `2x2`
* Inputs `p` or `q` do not satisdy `0 ≤ p,q ≤ 1`
"""
function dephrasureChannel(ρ :: Matrix{<:Number},p :: Union{Int,Float64}, q :: Union{Int,Float64}) :: Matrix{ComplexF64}
    if ((size(ρ,1)!=2)||(size(ρ,2)!=2))
        throw(DomainError(ρ, "the input ρ is not a qubit"))
    elseif !(0 ≤ q ≤ 1)
        throw(DomainError(q, "dephrasureChannel requires q ∈ [0,1]."))
    elseif !(0 ≤ p ≤ 1)
        throw(DomainError(p, "dephrasureChannel requires p ∈ [0,1]."))
    end
    pauli_Z = [1 0 ; 0 -1]
    output_ρ = zeros(ComplexF64,3,3)
    output_ρ[1:2,1:2]=(1-q)*((1-p)*ρ + p*pauli_Z*ρ*pauli_Z)
    output_ρ[3,3] = q*tr(ρ)
    return output_ρ
end

"""
    wernerHolevoChannel(ρ :: Matrix{<:Number}, p :: Union{Int,Float64}) :: Matrix{ComplexF64}

This function calculates the action of the [generalized Werner-Holevo channels](https://arxiv.org/abs/1406.7142)

```math
    \\mathcal{W}^{d,p}(ρ) = p \\mathcal{W}^{d,0}(ρ) + (1-p) \\mathcal{W}^{d,1}(ρ)
```

where ``p \\in [0,1]``. This means these are convex combinations of the original [Werner-Holevo channels](https://arxiv.org/abs/quant-ph/0203003)
which are defined as

```math
    \\mathcal{W}^{d,0}(ρ) = \\frac{1}{d+1}((\\text{Tr}ρ)I_{d} +ρ^{T}) \\hspace{1cm}
    \\mathcal{W}^{d,1}(ρ) = \\frac{1}{d-1}((\\text{Tr}ρ)I_{d} -ρ^{T}) .
```

Note the Choi matrices of these generalized channels are the (unnormalized) Werner states.

A `DomainError` is thrown if:
* Matrix `ρ` is not square
* `p` is not in  range `0 ≤ p ≤ 1`
"""
function wernerHolevoChannel(ρ :: Matrix{<:Number}, p :: Union{Int,Float64}) :: Matrix{ComplexF64}
    if !isequal(size(ρ)...)
        throw(DomainError(ρ, "the input ρ is not a square matrix"))
    elseif !(0 ≤ p ≤ 1)
        throw(DomainError(p, "wernerHolevoChannel requires p ∈ [0,1]."))
    end
    dim = size(ρ,1)
    term_1 = 1/(dim+1) * (tr(ρ)*I + transpose(ρ))
    term_2 = 1/(dim-1) * (tr(ρ)*I - transpose(ρ))
    return p *term_1 + (1-p)*term_2
end
