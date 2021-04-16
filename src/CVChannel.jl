module CVChannel

using Convex
using SCS
using MosekTools
using LinearAlgebra

export isPPT, minEntropyPrimal, minEntropyDual, minEntropyPPTPrimal, minEntropyPPTDual
export swapOperator, depolarizingChannel, dephrasureChannel, wernerHolevoChannel, wernerState
export choi, permuteSubsystems
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
    minEntropyPrimal(
        ρ :: Matrix{<:Number},
        dimA :: Int,
        dimB :: Int
    ) :: Tuple{Float64, Matrix{ComplexF64}}

This function solves the SDP
```math
\\min \\{ \\langle \\rho, X \\rangle :  \\text{Tr}_{A}(X) = I_{B} , X \\succeq 0 \\}
```
and returns the optimal value and the optimizer, X.
This is the SDP corresponding to the min-entropy.
To determine the min-entropy, take ``-\\log_{2}`` of the objective value.
(See [Section 6.1 of this reference](https://arxiv.org/abs/1504.00233) for further details about
the min-entropy). Note: we label the primal as the maximization problem unlike
in the above reference.
"""
function minEntropyPrimal(ρ :: AbstractArray, dimA :: Int, dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
    X = HermitianSemidefinite(dimA*dimB)
    objective = real(tr(ρ' * X))
    constraint = partialtrace(X, 1, [dimA,dimB]) == Matrix{Float64}(I,dimB,dimB)
    problem = maximize(objective,constraint)
    #solve!(problem, () -> Mosek.Optimizer(QUIET = true))
    solve!(problem, SCS.Optimizer(verbose=0))
    return problem.optval, X.value
end
"""
    minEntropyDual(
        ρ :: Matrix{<:Number},
        dimA :: Int,
        dimB :: Int
    ) :: Tuple{Float64, Matrix{ComplexF64}}

This function solves the SDP
```math
\\min \\{ \\text{Tr}(Y) :  I_{A} \\otimes Y \\succeq \\rho, Y \\in \\text{Herm}(B) \\}
```
and returns the optimal value and the optimizer, Y. This is the dual problem for the SDP for the min-entropy. To determine
the min-entropy, take ``-\\log_{2}`` of the objective value.
(See [Section 6.1 of this reference](https://arxiv.org/abs/1504.00233) for further details about
the min-entropy). Note: we label the primal as the maximization problem unlike
in the above reference.
"""
function minEntropyDual(ρ :: AbstractArray, dimA :: Int, dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
    identMat = Matrix{Float64}(I, dimA, dimA)
    Y = HermitianSemidefinite(dimB)
    objective = real(tr(Y))
    constraint = [kron(identMat , Y) ⪰ ρ]
    problem = minimize(objective,constraint)
    #solve!(problem, () -> Mosek.Optimizer(QUIET = true))
    solve!(problem, SCS.Optimizer(verbose=0))
    return problem.optval, Y.value
end
"""
    minEntropyPPTPrimal(
        ρ :: Matrix{<:Number},
        dimA :: Int,
        dimB :: Int
    ) :: Tuple{Float64,  Matrix{ComplexF64}}

This function solves the SDP
```math
\\min \\{ \\langle \\rho, X \\rangle :  \\text{Tr}_{A}(X) = I_{B} , \\Gamma(X) \\succeq 0, X \\succeq 0 \\}
```
where ``\\Gamma( \\cdot)`` is the partial transpose with respect to the second system,
and returns the optimal value and the optimizer, X.
This is the dual problem for the SDP for the min-entropy restricted to the PPT cone.
This has various interpretations. Note: we label the primal as the maximization problem.
"""
function minEntropyPPTPrimal(ρ :: AbstractArray, dimA :: Int, dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
    X = HermitianSemidefinite(dimA*dimB)
    objective = real(tr(ρ' * X))
    constraints = [partialtrace(X, 1, [dimA,dimB]) == Matrix{Float64}(I,dimB,dimB),
                   partialtranspose(X,2,[dimA,dimB]) ⪰ 0]
    problem = maximize(objective,constraints)
    #solve!(problem, () -> Mosek.Optimizer(QUIET = true))
    solve!(problem, SCS.Optimizer(verbose=0))
    return problem.optval, X.value
end
"""
    minEntropyPPTDual(
        ρ :: Matrix{<:Number},
        dimA :: Int,
        dimB :: Int
    ) :: Tuple{Float64,  Matrix{ComplexF64}, Matrix{ComplexF64}}

This function solves the SDP
```math
\\min \\{ \\text{Tr}(Y_{1}) : I_{A} \\otimes Y_{1} - \\Gamma(Y_{2}) \\succeq \\rho, Y_{2} \\succeq 0, Y_{1} \\in \\text{Herm}(B) \\}
```
where `` \\Gamma( \\cdot)`` is the partial transpose with respect to the second system,
and returns the optimal value and optimizer, ``(Y_1 , Y_2 )``.
This is the dual problem for the SDP for the min-entropy restricted to the PPT cone.
This has various interpretations. Note: we label the primal as the maximization problem.
"""
function minEntropyPPTDual(ρ :: AbstractArray, dimA :: Int, dimB :: Int, dual=true :: Bool) :: Tuple{Float64,  Matrix{ComplexF64}, Matrix{ComplexF64}}
    identMat = Matrix{Float64}(I, dimA, dimA)
    Y1 = ComplexVariable(dimB,dimB)
    Y2 = HermitianSemidefinite(dimA*dimB)
    objective = real(tr(Y1))
    constraints = [kron(identMat,Y1) - partialtranspose(Y2, 2 , [dimA,dimB]) ⪰ ρ,
                   Y1' - Y1 == zeros(dimB,dimB)] #Forces Hermiticity
    problem = minimize(objective,constraints)
    #solve!(problem, () -> Mosek.Optimizer(QUIET = true))
    solve!(problem, SCS.Optimizer(verbose=0))
    return problem.optval, Y1.value, Y2.value
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
    choi(𝒩 :: Function, Σ :: Int) :: Matrix{ComplexF64}
This function returns the Choi state of a channel 𝒩. It does this using that
```math
        J(\\mathcal{N}) = \\sum_{a,b \\in \\Sigma} E_{a,b} \\otimes \\mathcal{N}(E_{a,b}) ,
```
where ``\\Sigma`` is the finite alphabet indexing the input space and ``E_{a,b}``
is a square matrix of dimension ``\\Sigma`` with a ``1`` in the ``(a,b)`` entry
and a ``0`` everywhere else. Note this assumes you have a function that calculates
``\\mathcal{N}(X)`` for arbitrary input ``X``. As many of the functions for channels
in this module have multiple parameters, please note that if you have a channel function
`𝒩(ρ, p, q)` that calculates ``\\mathcal{N}_{p,q}(\\rho)``, you can declare a function
`𝒩_xy(ρ) = 𝒩(ρ,x,y)` for fixed `(x,y)` and then call, `choi(𝒩_xy, Σ)`.
"""
function choi(𝒩 :: Function, Σ :: Int) :: Matrix{ComplexF64}
    eab_matrix = zeros(Σ,Σ)
    choi_matrix = zeros(Σ^2,Σ^2)
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
    permuteSubsystems(ρ::Union{Vector,Matrix},perm::Vector{Int64},dim::Vector{Int64}) :: Union{Vector,Matrix}
This function returns the vector or matrix with the subsystems permuted. In principle this follows
the (generalization) of the mapping for re-ordering pure states:
    ```math
        |e_{i}\\rangle_{A} |e_j \\rangle_{B} |e_k \\rangle_{C} \\to [i,j,k] \\xrightarrow[]{\\pi} [\\pi(i),\\pi(j),\\pi)(k)]
        \\to |e_i\\rangle_{\\pi(A)} |e_j \\rangle_{\\pi(B)} |e_k \\rangle_{\\pi(C)}
    ```
    where ``\\pi`` is the permutation of the subsystems.
"""
function permuteSubsystems(ρ:: Union{Vector,Matrix},perm::Vector{Int64},dim::Vector{Int64}) :: Union{Vector,Matrix}
    orig_shape = size(ρ)
    num_subsys = length(perm)
    if isa(ρ,Vector)
        perm = Tuple(perm)
        dim = Tuple(dim)
        #This transpose_order permutedims is used because I can only think in `row-wise reshaping,' apparently.
        #This was the suggested work around on the discourse site for the julia language
        #If one wanted, they could figure out how to do the procedure column-wise,
        #which is how Julia is written, but I can't wrap my head around it at the moment
        transpose_order = Tuple([num_subsys:-1:1;])
        reshaped = permutedims(reshape(ρ,dim),transpose_order)
        reordered = permutedims(permutedims(reshaped,perm),transpose_order)
        result = reshape(reordered,orig_shape)

        return result
    else
        if !isequal(size(ρ)...)
            throw(DomainError(ρ, "the input ρ is not a square matrix or a pure state"))
        end
        #For the matrix version we do the same thing we just have twice as many indices
        #because we keep track of bras and kets
        perm = Tuple(vcat(perm,perm .+num_subsys))
        dim = Tuple(vcat(dim,dim))
        transpose_order = Tuple([num_subsys:-1:1;2*num_subsys:-1:num_subsys+1])
        reshaped = permutedims(reshape(ρ,dim),transpose_order)
        reordered = permutedims(permutedims(reshaped,perm),transpose_order)
        result = copy(transpose(reshape(reordered,orig_shape)))
        #Somehow I need this extra transpose in matrix version
        #The copy is to get rid of the data type transpose

        return result
    end
end
end
