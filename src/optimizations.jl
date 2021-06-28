"""
    eaCVPrimal(
        ρ :: Matrix{<:Number},
        dimA :: Int,
        dimB :: Int
    ) :: Tuple{Float64, Matrix{ComplexF64}}
This function solves the SDP
```math
\\max \\{ \\langle \\rho, X \\rangle :  \\text{Tr}_{A}(X) = I_{B} , X \\succeq 0 \\}
```
and returns the optimal value and the optimizer, X.
This is the primal problem of the SDP corresponding to the entanglement-assisted commmunication value.
It is related to the channel min-entropy ``H_{\\min}(A|B)_{\\mathcal{J}(\\mathcal{N})}`` by
``cv_{\\text{ea}}(\\mathcal{N}) = 2^{-H_{\\min}(A|B)_{\\mathcal{J}(\\mathcal{N})}}``.
(See [Section 6.1 of this reference](https://arxiv.org/abs/1504.00233) for further details about
the min-entropy.)
Note: we label the primal as the maximization problem.
"""
function eaCVPrimal(ρ :: AbstractArray, dimA :: Int, dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
    X = HermitianSemidefinite(dimA*dimB)
    objective = real(tr(ρ' * X))
    constraint = partialtrace(X, 1, [dimA,dimB]) == Matrix{Float64}(I,dimB,dimB)
    problem = maximize(objective,constraint)
    qsolve!(problem)
    return problem.optval, X.value
end

"""
    eaCVDual(
        ρ :: Matrix{<:Number},
        dimA :: Int,
        dimB :: Int
    ) :: Tuple{Float64, Matrix{ComplexF64}}
This function solves the SDP
```math
\\min \\{ \\text{Tr}(Y) :  I_{A} \\otimes Y \\succeq \\rho, Y \\in \\text{Herm}(B) \\}
```
and returns the optimal value and the optimizer, Y. This is the dual problem for the SDP
corresponding to the entanglement-assisted commmunication value.
It is related to the channel min-entropy ``H_{\\min}(A|B)_{\\mathcal{J}(\\mathcal{N})}`` by
``cv_{\\text{ea}}(\\mathcal{N}) = 2^{-H_{\\min}(A|B)_{\\mathcal{J}(\\mathcal{N})}}``.
(See [Section 6.1 of this reference](https://arxiv.org/abs/1504.00233) for further details about
the min-entropy.)
Note: we label the primal as the maximization problem.
"""
function eaCVDual(ρ :: AbstractArray, dimA :: Int, dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
    identMat = Matrix{Float64}(I, dimA, dimA)
    Y = HermitianSemidefinite(dimB)
    objective = real(tr(Y))
    constraint = [kron(identMat , Y) ⪰ ρ]
    problem = minimize(objective,constraint)
    qsolve!(problem)
    return problem.optval, Y.value
end

"""
    pptCVPrimal(
        ρ :: Matrix{<:Number},
        dimA :: Int,
        dimB :: Int
    ) :: Tuple{Float64,  Matrix{ComplexF64}}
This function solves the SDP
```math
\\max \\{ \\langle \\rho, X \\rangle :  \\text{Tr}_{A}(X) = I_{B} , \\Gamma(X) \\succeq 0, X \\succeq 0 \\}
```
where ``\\Gamma( \\cdot)`` is the partial transpose with respect to the second system, and
returns the optimal value and the optimizer, X.
This is the primal problem for the SDP relaxation of the channel value. The relaxation is to the PPT cone.
This has various interpretations. Note: we label the primal as the maximization problem.
"""
function pptCVPrimal(ρ :: AbstractArray, dimA :: Int, dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
    X = HermitianSemidefinite(dimA*dimB)
    objective = real(tr(ρ' * X))
    constraints = [partialtrace(X, 1, [dimA,dimB]) == Matrix{Float64}(I,dimB,dimB),
                   partialtranspose(X,2,[dimA,dimB]) ⪰ 0]
    problem = maximize(objective,constraints)
    qsolve!(problem)
    return problem.optval, X.value
end

"""
    pptCVDual(
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
This is the dual problem for the SDP relaxation of the channel value. The relaxation is to the PPT cone.
This has various interpretations. Note: we label the primal as the maximization problem.
"""
function pptCVDual(ρ :: AbstractArray, dimA :: Int, dimB :: Int, dual=true :: Bool) :: Tuple{Float64,  Matrix{ComplexF64}, Matrix{ComplexF64}}
    identMat = Matrix{Float64}(I, dimA, dimA)
    Y1 = ComplexVariable(dimB,dimB)
    Y2 = HermitianSemidefinite(dimA*dimB)
    objective = real(tr(Y1))
    constraints = [kron(identMat,Y1) - partialtranspose(Y2, 2 , [dimA,dimB]) ⪰ ρ,
                   Y1' - Y1 == zeros(dimB,dimB)] #Forces Hermiticity
    problem = minimize(objective,constraints)
    qsolve!(problem)
    return problem.optval, Y1.value, Y2.value
end

"""
    twoSymCVPrimal(
        ρ :: Matrix{<:Number},
        dimA :: Int,
        dimB :: Int
    ) :: Tuple{Float64,  Matrix{ComplexF64}}
This function solves the SDP
```math
\\max \\{ \\langle \\rho, X \\rangle :  \\text{Tr}_{A}(X) = I_{B} ,
                                        \\Gamma^{B_{1}}(X) \\succeq 0,
                                        X = (I_{A} \\otimes \\mathbb{F}_{B})X(I_{A} \\otimes \\mathbb{F}_{B}),
                                        X \\succeq 0 \\}
```
where ``\\Gamma^{B_{1}}( \\cdot)`` is the partial transpose with respect to the second system,
and returns the optimal value and the optimizer, X. The conditions on X demand it is two-symmetric, i.e.
an element of the lowest level of the [DPS hierarchy.](https://arxiv.org/abs/quant-ph/0308032)
Note: we label the primal as the maximization problem.

!!! warning "Runs Out of Memory Easily"
    This function  will run out of memory for the tensor product of even qutrit to qutrit channels.
"""
function twoSymCVPrimal(ρ :: AbstractArray, dimA :: Int, dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
    X = HermitianSemidefinite(dimA*dimB^2)
    idA = Matrix(1I, dimA, dimA); idB = Matrix(1I, dimB, dimB)
    F = swapOperator(dimB)
    objective = real(tr(kron(ρ,idB)' * X))
    constraints = [partialtrace(partialtrace(X, 1, [dimA,dimB,dimB]),2,[dimB,dimB]) == Matrix{Float64}(I,dimB,dimB),
                   X - (kron(idA,F)*X*kron(idA,F)') == 0,
                   partialtranspose(X,2,[dimA,dimB,dimB]) ⪰ 0]
    problem = maximize(objective,constraints)
    qsolve!(problem)
    return problem.optval, X.value
end
#I don't write the dual program because it's too big to be useful anyways

"""
    threeSymCVPrimal(
        ρ :: Matrix{<:Number},
        dimA :: Int,
        dimB :: Int
    ) :: Tuple{Float64,  Matrix{ComplexF64}}
This function solves the SDP
```math
\\max \\{ \\langle \\rho, X \\rangle :  \\text{Tr}_{A}(X) = I_{B} ,
                                        \\Gamma^{B_{1}}(X) \\succeq 0,
                                        X = (I_{A} \\otimes W_{\\pi})X(I_{A} \\otimes W_{\\pi}^{*}) \\quad \\forall \\pi \\in \\mathcal{S_{3}},
                                        X \\succeq 0 \\}
```
where ``\\Gamma^{B_{1}}( \\cdot)`` is the partial transpose with respect to the second system,
``W_{\\pi}`` is the unitary that permutes the subspaces according to permutation ``\\pi``.
The function returns the optimal value and the optimizer, X. The conditions on X demand it is two-symmetric, i.e.
an element of the lowest level of the [DPS hierarchy.](https://arxiv.org/abs/quant-ph/0308032)
Note: we label the primal as the maximization problem.
"""
function threeSymCVPrimal(ρ :: AbstractArray, dimA :: Int, dimB :: Int) :: Tuple{Float64,  Matrix{ComplexF64}}
    """
    This is a helper function for threeSymCVPrimal because permuteSubsystems would need
    a reshape(::Variable, ::NTuple), but Convex only supports reshape(::Variable, ::Int, ::Int).
    Obviously it is not ideal.
    """
    function swapCons(X,dimA,dimB,t)
        idA,idB = Matrix(1I, dimA, dimA), Matrix(1I,dimB,dimB)
        if t == 1 #Permutation [1,2,4,3]
            return kron(idA,kron(idB,swapOperator(dimB)))*X*kron(idA,kron(idB,swapOperator(dimB)))'
        elseif t == 2 #Permutation [1,3,2,4]
            return kron(idA,kron(swapOperator(dimB),idB))*X*kron(idA,kron(swapOperator(dimB),idB))'
        elseif t == 3 #Permutation [1,3,4,2]
            Xp = kron(idA,kron(swapOperator(dimB),idB))*X*kron(idA,kron(swapOperator(dimB),idB))' #Swaps 3 and 2
            return kron(idA,kron(idB,swapOperator(dimB)))*Xp*kron(idA,kron(idB,swapOperator(dimB)))' #Swaps 2 and 4
        elseif t == 4 #Permutation [1,4,2,3]
            Xp = kron(idA,kron(idB,swapOperator(dimB)))*X*kron(idA,kron(idB,swapOperator(dimB)))' #Swaps 3 and 4
            X = kron(idA,kron(swapOperator(dimB),idB))*Xp*kron(idA,kron(swapOperator(dimB),idB))' #Swaps 2 and 4
        else #Permutation [1,4,3,2]
            Xp = kron(idA,kron(swapOperator(dimB),idB))*X*kron(idA,kron(swapOperator(dimB),idB))' #Swaps 3 and 2
            Xpp = kron(idA,kron(idB,swapOperator(dimB)))*Xp*kron(idA,kron(idB,swapOperator(dimB)))' #Swaps 2 and 4
            return kron(idA,kron(swapOperator(dimB),idB))*Xpp*kron(idA,kron(swapOperator(dimB),idB))' #Swaps 3 and 4
        end
    end
    X = HermitianSemidefinite(dimA*dimB^3)
    idB12 = Matrix(1I, dimB^2, dimB^2)
    dims = [dimA,dimB,dimB,dimB]
    objective = real(tr(kron(ρ,idB12)' * X))
    constraints = [partialtrace(partialtrace(X, 1, [dimA,dimB,dimB^2]),2,[dimB,dimB^2]) == Matrix{Float64}(I,dimB,dimB),
                   X - swapCons(X,dimA,dimB,1) == 0,
                   X - swapCons(X,dimA,dimB,2) == 0,
                   X - swapCons(X,dimA,dimB,3) == 0,
                   X - swapCons(X,dimA,dimB,4) == 0,
                   X - swapCons(X,dimA,dimB,5) == 0,
                   partialtranspose(X,2,dims) ⪰ 0]
    problem = maximize(objective,constraints)
    qsolve!(problem)
    return problem.optval, X.value
end
#I don't write the dual program because it's too big to be useful anyways
