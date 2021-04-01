module CVChannel

using Convex
using MosekTools
using LinearAlgebra

export isPPT, minEntropyPrimal, minEntropyDual, minEntropyPPTPrimal, minEntropyPPTDual
export swapOperator,qDepolarizingChannel, dephrasureChannel, wernerHolevoChannel, wernerState
export getChoi
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
    minEntropyPrimal(ρ,dimA :: Int ,dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
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
function minEntropyPrimal(ρ, dimA :: Int, dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
    X = HermitianSemidefinite(dimA*dimB)
    objective = real(tr(ρ' * X))
    constraint = partialtrace(X, 1, [dimA,dimB]) == Matrix{Float64}(I,dimB,dimB)
    problem = maximize(objective,constraint)
    solve!(problem, () -> Mosek.Optimizer(QUIET = true))
    return problem.optval, X.value
end
"""
    minEntropyDual(ρ,dimA :: Int ,dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
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
function minEntropyDual(ρ, dimA :: Int, dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
    identMat = Matrix{Float64}(I, dimA, dimA)
    Y = HermitianSemidefinite(dimB)
    objective = real(tr(Y))
    constraint = [kron(identMat , Y) ⪰ ρ]
    problem = minimize(objective,constraint)
    solve!(problem, () -> Mosek.Optimizer(QUIET = true))
    return problem.optval, Y.value
end
"""
    minEntropyPPTPrimal(ρ,dimA :: Int ,dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
This function solves the SDP
```math
\\min \\{ \\langle \\rho, X \\rangle :  \\text{Tr}_{A}(X) = I_{B} , \\Gamma(X) \\succeq 0, X \\succeq 0 \\}
```
where ``\\Gamma( \\cdot)`` is the partial transpose with respect to the second system,
and returns the optimal value and the optimizer, X.
This is the dual problem for the SDP for the min-entropy restricted to the PPT cone.
This has various interpretations. Note: we label the primal as the maximization problem.
"""
function minEntropyPPTPrimal(ρ, dimA :: Int, dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
    X = HermitianSemidefinite(dimA*dimB)
    objective = real(tr(ρ' * X))
    constraints = [partialtrace(X, 1, [dimA,dimB]) == Matrix{Float64}(I,dimB,dimB),
                   partialtranspose(X,2,[dimA,dimB]) ⪰ 0]
    problem = maximize(objective,constraints)
    solve!(problem, () -> Mosek.Optimizer(QUIET = true))
    return problem.optval, X.value
end
"""
    minEntropyPPTDual(ρ,dimA :: Int ,dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}, Matrix{ComplexF64}}
This function solves the SDP
```math
\\min \\{ \\text{Tr}(Y_{1}) : I_{A} \\otimes Y_{1} - \\Gamma(Y_{2}) \\succeq \\rho, Y_{2} \\succeq 0, Y_{1} \\in \\text{Herm}(B) \\}
```
where `` \\Gamma( \\cdot)`` is the partial transpose with respect to the second system,
and returns the optimal value and optimizer, ``(Y_1 , Y_2 )``.
This is the dual problem for the SDP for the min-entropy restricted to the PPT cone.
This has various interpretations. Note: we label the primal as the maximization problem.
"""
function minEntropyPPTDual(ρ, dimA :: Int, dimB :: Int, dual=true :: Bool) :: Tuple{Float64,  Matrix{ComplexF64}, Matrix{ComplexF64}}
    identMat = Matrix{Float64}(I, dimA, dimA)
    Y1 = ComplexVariable(dimB,dimB)
    Y2 = HermitianSemidefinite(dimA*dimB)
    objective = real(tr(Y1))
    constraints = [kron(identMat,Y1) - partialtranspose(Y2, 2 , [dimA,dimB]) ⪰ ρ,
                   Y1' - Y1 == zeros(dimB,dimB)] #Forces Hermiticity
    problem = minimize(objective,constraints)
    solve!(problem, () -> Mosek.Optimizer(QUIET = true))
    return problem.optval, Y1.value, Y2.value
end
"""
    swapOperator(dim :: Int) :: Matrix{Float64}
This function is the swap operator ``\\mathbb{F}`` which is defined by the action
```math
\\mathbb{F}(u \\otimes v) = v \\otimes u \\hspace{5mm} u,v \\in \\mathcal{H}_{A} .
```
The function uses that ``\\mathbb{F} = \\sum_{a,b \\in \\Sigma} E_{a,b} \\otimes E_{b,a}``.
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
    qDepolarizingChannel(ρ :: Matrix{Float64}, q :: Union{Int,Float64}) :: Matrix{ComplexF64}
This calculates the action of the depolarizing channel,
```math
\\Delta_{q}(\\rho) = (1-q)\\rho + q \\text{Tr}(\\rho) \\frac{1}{d} I_{AB} ,
```
Note these channels are the channels covariant with respect to the unitary group.
"""
function qDepolarizingChannel(ρ :: Matrix{Float64}, q :: Union{Int,Float64}) :: Matrix{ComplexF64}
    if(ndims(ρ)!=2)
        ErrorException("the input ρ does not have 2 dimensions")
    elseif(size(ρ,1)*size(ρ,2) != size(ρ,1)^2)
        ErrorException("the input ρ is not a square matrix")
    elseif(q>1||q<0)
        DomainError("qDepolarizingChannel requires q ∈ [0,1].")
    else
        identMat = Matrix{Float64}(I, size(ρ,1), size(ρ,1))
        return (1-q)*ρ  + q*(tr(ρ))*(1/(size(ρ,1)))*identMat
    end
end
"""
    dephrasureChannel(ρ,p :: Union{Int,Float64}, q :: Union{Int,Float64}) :: Matrix{ComplexF64}
This function calculates the action of the [dephrasureChannel](https://arxiv.org/abs/1806.08327),
```math
\\mathcal{N}_{p,q}( \\rho) := (1-q)((1-p) \\rho + pZ \\rho Z) + q \\text{Tr}( \\rho) |e\\rangle \\langle e|,
```
where ``Z`` is the Pauli-Z matrix.
"""
function dephrasureChannel(ρ,p :: Union{Int,Float64}, q :: Union{Int,Float64}) :: Matrix{ComplexF64}
    #This isn't merged with your previous package, so define here
    if(ndims(ρ)!=2)
        ErrorException("the input ρ does not have 2 dimensions")
    elseif((size(ρ,1)!=2)||(size(ρ,2)!=2))
        ErrorException("the input ρ is not a qubit")
    elseif(q>1||q<0)
        DomainError("dephrasureChannel requires q ∈ [0,1].")
    elseif(p>1||q<0)
        DomainError("dephrasureChannel requires p ∈ [0,1].")
    else
        pauli_Z = [1 0 ; 0 -1]
        output_ρ = convert(Matrix{ComplexF64},zeros(3,3))
        output_ρ[1:2,1:2]=(1-q)*((1-p)*ρ + p*pauli_Z*ρ*pauli_Z)
        output_ρ[3,3] = q*tr(ρ)
        return output_ρ
    end
end
"""
    wernerHolevoChannel(ρ, p :: Union{Int,Float64}) :: Matrix{ComplexF64}
This function calculates the action of the [generalized Werner-Holevo channels](https://arxiv.org/abs/1406.7142)
```math
    \\mathcal{W}^{d,p}(ρ) = p \\mathcal{W}^{d,0}(ρ) + (1-p) \\mathcal{W}^{d,1}(ρ)
```
which is a convex combination of the original [Werner-Holevo channels](https://arxiv.org/abs/quant-ph/0203003)
which are defined as
```math
    \\mathcal{W}^{d,0}(ρ) = \\frac{1}{d+1}((\\text{Tr}ρ)I_{d} +ρ^{T})
    \\mathcal{W}^{d,1}(ρ) = \\frac{1}{d-1}((\\text{Tr}ρ)I_{d} -ρ^{T}) .
```
Note the Choi matrices of these generalized channels are the (unnormalized) Werner states.
"""
function wernerHolevoChannel(ρ, p :: Union{Int,Float64}) :: Matrix{ComplexF64}
    if(size(ρ,1)*size(ρ,2) != size(ρ,1)^2)
        ErrorException("the input ρ is not a square matrix")
    elseif(p>1||p<0)
        DomainError("wernerHolevoChannel requires p ∈ [0,1].")
    else
        dim = size(ρ,1)
        identMat = Matrix{Float64}(I, dim, dim)
        term_1 = 1/(dim+1) * (tr(ρ)*identMat + transpose(ρ))
        term_2 = 1/(dim-1) * (tr(ρ)*identMat - transpose(ρ))
        return p *term_1 + (1-p)*term_2
    end
end
"""
    wernerState(dim :: Int, p ::Union{Int,Float64}) :: Matrix{Float64}
This function constructs the Werner states,
```math
    \\sigma_{d,p} = p \\frac{\\Pi_{0}}{d+1 \\choose 2} + (1-p) \\frac{\\Pi_{1}}{d \\choose 2}
```
where ``\\Pi_0`` and ``\\Pi_1`` are the projectors onto the symmetric and anti-symmetric
subspaces respectively. They can be determined by
```math
    \\Pi_0 = \\frac{I_{A} \\otimes I_{B} + \\mathbb{F}}{2} \\hspace{1cm} \\Pi_1 = \\frac{I_{A} \\otimes I_{B} - \\mathbb{F}}{2}
```
where ``\\mathbb{F}`` is the swap operator.
"""
function wernerState(d :: Int, p ::Union{Int,Float64}) :: Matrix{Float64}
    if d <= 1
        DomainError("wernerState requires the local dimension is two or greater.")
    elseif(p>1||p<0)
        DomainError("wernerState requires p ∈ [0,1].")
    else
        identMat = Matrix{Float64}(I, d^2, d^2)
        swap = swapOperator(d)
        Π0 = (identMat + swap)/2
        Π1 = (identMat - swap)/2
        return p * Π0 / binomial(d+1,2) + (1-p) * Π1 / binomial(d,2)
    end
end
"""
    getChoi(𝒩 :: Function, Σ :: Int) :: Matrix{ComplexF64}
This function returns the Choi state of a channel 𝒩. It does this using that
```math
        J(\\mathcal{N}) = \\sum_{a,b \\in \\Sigma} E_{a,b} \\otimes \\mathcal{N}(E_{a,b}) ,
```
where ``\\Sigma`` is the finite alphabet indexing the input space. Note this
assumes you have a function that calculates ``\\mathcal{N}(X)`` for arbitrary
input ``X``. As many of the functions for channels in this module have multiple
parameters, please note that if you have a function f(ρ,p,q) for calculating
``\\mathcal{N}_{p,q}(\\rho)``, you can declare a function g that calculates
``\\mathcal{N}_{x,y}(\\rho)`` for fixed ``(x,y)`` and then pass g to the getChoi
 function.
"""
function getChoi(𝒩, Σ) :: Matrix{ComplexF64}
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
end
