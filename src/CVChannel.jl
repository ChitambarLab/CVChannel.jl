module CVChannel

using Convex
using MosekTools
using LinearAlgebra

export isPPT, minEntropyPrimal, minEntropyDual, minEntropyPPTPrimal, minEntropyPPTDual
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

end
