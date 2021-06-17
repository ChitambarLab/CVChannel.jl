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
where ``\\Gamma( \\cdot)`` is the partial transpose with respect to the second system,
and returns the optimal value and the optimizer, X.
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
    WHIDLP(
        d1 :: Int64,
        d2 :: Int64,
        λ :: Union{Int,Float64}
    ) :: Float64

This function implements the linear program (LP) to determine the communication
value of the Werner-Holevo channel tensored with the identity channel, when the
problem is relaxed to optimizing over the PPT cone. (See cite for derivation).
d1,d2 are the input-output dimensions of the Werner-Holevo and identity channel
respectively. λ is the parameter defining the Werener-Holevo channel.
*Note it is the opposite of what we use for the rest of the code. This should be
rectified at some point when things are written up and the code is finalized*
"""
function WHIDLP(d1 :: Int64, d2 :: Int64, λ :: Union{Int,Float64} ) :: Float64
    #This is the vector of variables and we use alphabetical order x[1] = w, x[2] = x,...
    v = Variable(4)
    #Given the number of constraints we define the problem and add constraints
    objective = (v[1]+v[3]*d2+(1-2λ)*(v[2]+v[4]*d2))
    problem = maximize(objective)
    problem.constraints += [
        0 <= v[1]-v[2]+d2*v[3]-d2*v[4],
        0 <= v[1]-v[2],
        0 <= v[1]+v[2]+d2*v[3]+d2*v[4],
        0 <= v[1]+v[2],
        0 <= v[1] + d1*v[2] - v[3] - d1*v[4],
        0 <= v[1] - v[3],
        0 <= v[1] + d1*v[2] + v[3] + d1*v[4],
        0 <= v[1] + v[3],
        1 == d1*d2*v[1]+d2*v[2]+d1*v[3] + v[4]
    ]
    qsolve!(problem)

    return d1*d2*problem.optval
end
