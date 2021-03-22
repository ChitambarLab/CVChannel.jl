module CVChannel

using Convex
using MosekTools
using LinearAlgebra

export isPPT, minEnt, minEntPPT
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
    minEnt(x,dimA :: Int ,dimB :: Int, dual=true :: Boolean)

This function returns both the objective value of the
SDP for the min-entropy along with its optimizer. To determine
the min-entropy, take -log2 of the objective value.
(See https://arxiv.org/abs/1504.00233 for further details about
the min-entropy). It can run the dual or the primal (Note: we label
the primal as the maximization problem unlike in the above reference.)
"""
function minEnt(ρ, dimA :: Int, dimB :: Int, dual=true :: Bool)
    if dual == true
        identMat = Matrix{Float64}(I, dimA, dimA)
        Y = HermitianSemidefinite(dimB)
        objective = real(tr(Y))
        constraint = [kron(identMat , Y) ⪰ ρ]
        problem = minimize(objective,constraint)
        solve!(problem, () -> Mosek.Optimizer(QUIET = true))
        return problem.optval, Y.value
    else
        X = HermitianSemidefinite(dimA*dimB)
        objective = real(tr(ρ' * X))
        constraint = partialtrace(X, 1, [dimA,dimB]) == Matrix{Float64}(I,dimB,dimB)
        problem = maximize(objective,constraint)
        solve!(problem, () -> Mosek.Optimizer(QUIET = true))
        return problem.optval, X.value
    end
end
"""
    minEnt(x,dimA :: Int ,dimB :: Int, dual=true :: Boolean)

This function returns both the objective value of the
SDP for the min-entropy along with its optimizer when the optimizer is
restricted to the PPT cone. This has various interpretations. It can run
the dual or the primal. Notation follows Dropbox Note. (Note: we label
the primal as the maximization problem following Watrous.)
"""
function minEntPPT(ρ, dimA :: Int, dimB :: Int, dual=true :: Bool)
    if dual == true
        identMat = Matrix{Float64}(I, dimA, dimA)
        Y1 = ComplexVariable(dimB,dimB)
        Y2 = HermitianSemidefinite(dimA*dimB)
        objective = real(tr(Y1))
        constraints = [kron(identMat,Y1) - partialtranspose(Y2, 2 , [dimA,dimB]) ⪰ ρ,
                                Y1' - Y1 == zeros(dimB,dimB)] #Forces Hermiticity
        problem = minimize(objective,constraints)
        solve!(problem, () -> Mosek.Optimizer(QUIET = true))
        return problem.optval, Y1.value, Y2.value
    else
        X = HermitianSemidefinite(dimA*dimB)
        objective = real(tr(ρ' * X))
        constraints = [partialtrace(X, 1, [dimA,dimB]) == Matrix{Float64}(I,dimB,dimB),
                       partialtranspose(X,2,[dimA,dimB]) ⪰ 0]
        problem = maximize(objective,constraints)
        solve!(problem, () -> Mosek.Optimizer(QUIET = true))
        return problem.optval, X.value
    end
end

end
