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
    S|k\\rangle = |k+1 \\mod d \\rangle
```
"""
function shiftOperator(d::Int64) :: Matrix
    id = Matrix(I,d,d)
    shift = id[[end,1:(end-1)...],:]

    return shift
end

"""
    discreteWeylOperator(m :: Int64, n :: Int64, d :: Int64)

This function returns the (m,n)^th unitary for generating the generalized
Bell basis. They are defined by their action on the computational basis:
```math
    U_{m,n}|e_{k}\\rangle = e^{2 \\pi mk i / d} |e_{k+n}\\rangle
```
These are the discrete Weyl Operator basis. See
[Section 3 of this paper](https://arxiv.org/abs/1004.1655) for further details.

!!! warning
    Different works define the discrete Weyl operators differently such that
    the phase or ordering may differ. Please check your reference.
"""
function discreteWeylOperator(m :: Int64, n :: Int64, d :: Int64) :: Matrix
    if m < 0 || n < 0
        throw(DomainError((n,m), "Make sure m,n ∈ [0,1,...,d-1]."))
    elseif m >= d || n >= d
        throw(DomainError((n,m), "Make sure m,n < d."))
    end
    λ = exp(2*pi*1im / d)
    U = zeros(ComplexF64,d,d)
    for k = 0 : d-1
        U[((k+n) % d) + 1,k+1] = λ^(k*m)
    end
    return U
end

"""
    isometricChannel(kraus_ops :: Vector{Any}) :: Matrix

This function builds the isometric representation ``V`` of a
channel 𝒩: A → B from the Kraus operators ``\\{K_{i}\\}``.
It does this by calculating
```math
    V = \\sum_{i} K_{i} \\otimes |i\\rangle
```
"""
function isometricRep(kraus_ops :: Vector{Any}) :: Matrix
    dim_B , dim_A = size(kraus_ops[1])
    dim_E = length(kraus_ops)
    V, e_vec = zeros(dim_B*dim_E,dim_A), zeros(dim_E)
    for i in [1:dim_E;]
        e_vec[i] = 1
        V += kron(kraus_ops[i],e_vec)
        e_vec[i] = 0
    end
    return V
end

"""
    complementaryChannel(kraus_ops :: Vector{Any}) :: Vector{Any}

This function takes a set of Kraus operators for a channel ``\\mathcal{N}_{A \\to B}``
and returns a set of Kraus operators for the complementary channel,
``\\mathcal{N}_{A \\to E}``. It does this by generating the Kraus operators of
the isometric representation of the channel followed by partial trace on the
``B`` space.
"""
function complementaryChannel(kraus_ops :: Vector{Any}) :: Vector{Any}
    V = isometricRep(kraus_ops)
    dim_B , dim_A = size(kraus_ops[1])
    dim_E = length(kraus_ops)

    comp_kraus = Any[]
    b_vec, e_id = zeros(dim_B), Matrix(1I,dim_E,dim_E)
    for i in [1:dim_B;]
        b_vec[i] = 1
        push!(comp_kraus, kron(b_vec,e_id)'*V)
        b_vec[i] = 0
    end

    return comp_kraus
end
