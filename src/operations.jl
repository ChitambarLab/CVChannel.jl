"""
    isPPT(x, sys :: Int, dims :: Vector) :: Bool
This function returns true if the input state x is PPT
with respect to the (sys)th system. False otherwise.
dims is a vector of the sizes of the subsystems.
"""
function isPPT(x,sys::Int,dims::Vector) :: Bool
    #We don't make sure the inputs are good because the partialtranspose function
    #will do that for us

    #Get PPT of x
    xPPT = partialtranspose(x,sys,dims)

    #Check if it is PSD
    if eigmin(xPPT) >= 0 || isapprox(0,eigmin(xPPT))
        return true
    else
        return false
    end
end

"""
    swapOperator(dim :: Int) :: Matrix{Float64}
This function is the swap operator ``\\mathbb{F}`` which is defined by the action
```math
\\mathbb{F}(u \\otimes v) = v \\otimes u \\hspace{5mm} u,v \\in \\mathcal{H}_{A} .
```
The function uses that ``\\mathbb{F} = \\sum_{a,b \\in \\Sigma} E_{a,b} \\otimes E_{b,a}``
where ``E_{a,b}`` is a square matrix of dimension ``\\Sigma`` with a one in the ``(a,b)``
entry and a ``0`` everywhere else.
"""
function swapOperator(dim :: Int) :: Matrix{Float64}
    swap_operator = zeros(dim^2,dim^2)

    for col_id in 1:dim^2
        # factoring col_id into Alice and Bob subsystem ids
        a_id = floor(Int64, (col_id-1)/dim)
        b_id = (col_id-1) % dim

        # swapping Alice and Bob subsystem ids to construct row_id
        row_id = b_id*dim + a_id + 1
        swap_operator[row_id, col_id] = 1
    end

    return swap_operator
end

"""
    permuteSubsystems(
        ρ:: Vector,
        perm::Vector{Int64},
        dims::Vector{Int64}
    ) :: Vector
This function returns the vector with the subsystems permuted. For example, given three
subspaces ``A,B,C``, and the permutation ``\\pi`` defined by ``(A,B,C) \\xrightarrow[]{\\pi} (C,A,B),``
the function implements the process:
```math
    |e_{i}\\rangle_{A} |e_j \\rangle_{B} |e_k \\rangle_{C} \\xrightarrow[]{\\pi}
    |e_{k} \\rangle_{C} |e_{i}\\rangle_{A}  |e_{j} \\rangle_{B}  ,
```
by re-indexing the vector, permuting the indices appropriately, and converting it
back into a vector.
"""
function permuteSubsystems(ρ:: Vector, perm::Vector{Int64},dims::Vector{Int64}) :: Vector
    #This is almost identical to Tony Cubitt's implementation of this function https://www.dr-qubit.org/matlab.html
    #This is largely because Julia does reshape column-wise as Matlab does
    orig_shape = size(ρ)
    num_subsys = length(perm)
    #Note certain things get reversed. This is because Julia does reshape column-wise
    permTup = Tuple((num_subsys+1) .- reverse(perm)) #Reshape requires tuples
    dimTup = Tuple(reverse(dims))
    result = reshape(permutedims(reshape(ρ,dimTup),permTup),orig_shape)
end
"""
    permuteSubsystems(
        ρ:: Matrix,
        perm::Vector{Int64},
        dims::Vector{Int64}
    ) :: Matrix

This function returns the matrix with the subsystems permuted. It is a generalization of the vector code.
For example, given three subspaces ``A,B,C``, and the permutation ``\\pi`` defined by ``(A,B,C) \\xrightarrow[]{\\pi} (C,A,B),``
the function implements the process:
```math
|e_{i}\\rangle\\langle e_{i}|_{A}\\otimes |e_j \\rangle\\langle e_j|_{B}\\otimes |e_k \\rangle\\langle e_k|_{C} \\xrightarrow[]{\\pi}
|e_{k} \\rangle\\langle e_{k}|_{C}\\otimes |e_{i}\\rangle\\langle e_{i}|_{A} \\otimes  |e_{j} \\rangle\\langle e_{j}|_{B} ,
```
by re-indexing the matrix, permuting the indices, and reconstructing the matrix.
Both bra and ket indices receive the same permutation.
"""
function permuteSubsystems(ρ:: Matrix,perm::Vector{Int64},dims::Vector{Int64}) :: Matrix
    #This is almost identical to Tony Cubitt's implementation of this function https://www.dr-qubit.org/matlab.html
    #This is largely because Julia does reshape column-wise as Matlab does
    if !isequal(size(ρ)...)
            throw(DomainError(ρ, "the input ρ is not a square matrix or a pure state"))
    end
    orig_shape = size(ρ)
    num_subsys = length(perm)
    #For the matrix version we do the same thing we just have twice as many indices
    #because we keep track of bras and kets
    permPrime = (num_subsys+1) .- reverse(perm)
    permTup = Tuple([permPrime num_subsys .+ permPrime])
    dimTup = Tuple([reverse(dims) reverse(dims)])
    result = reshape(permutedims(reshape(ρ,dimTup),permTup),orig_shape)

    return result
end

"""
    shiftOperator(d::Int64) :: Matrix

This function returns the operator that shifts the computational basis mod d
for complex Euclidean space of dimension d. That is, it returns the operator ``S``
defined by the action
```math
    S|e_{k}\\rangle = |e_{k+1}\\rangle (mod d)
```
"""

function shiftOperator(d::Int64) :: Matrix
    shift = zeros(1,d)
    shift[1,d] = 1
    row = zeros(1,d)
    for i = 1: (d-1)
        row[1,i] = 1
        shift = vcat(shift,row)
        row[1,i] = 0
    end
    return shift
end

"""
    bellUnitary(m :: Int64, n :: Int64, d :: Int64)

This function returns the (m,n)^th unitary for generating the generalized
Bell basis. They are defined by their action on the computational basis:
```math
    U_{n,m}|e_{k}\\rangle = e^{2 \\pi mk i / d} |e_{k+n}\\rangle
```
These are actually the Weyl Operator basis. See https://arxiv.org/abs/0901.4729
for further details.
"""
function bellUnitary(n :: Int64, m :: Int64, d :: Int64) :: Matrix
    #There probably are better names for this function, but yeah
    if m < 0 || n < 0
        throw(DomainError((n,m), "Make sure m,n ∈ [0,1,...,d-1]."))
    elseif m >= d || n >= d
        throw(DomainError((n,m), "Make sure m,n < d."))
    end
    λ = exp(2*pi*1im / d)
    U = zeros(ComplexF64,d,d)
    for k = 0 : d-1
        U[k+1,((k+m) % d) + 1] = λ^(k*n)
    end
    return U
end
