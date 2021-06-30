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
!!! warning
    The `λ` parameter of `WHIDLP` relates to the mixing probability `p` used
    throughout as `p = (1 - λ)`. This will likely be rectified in a future version.
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

"""
    generalWHLPConstraints(
        n :: Int,
        d :: Int,
        λ_vec :: Union{Vector{Float64},Vector{Int64}}
    ) :: Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64},Matrix{Float64}}

This function returns the linear program constraints for calculating the PPT
communication value of the Werner-Holevo channels run in parallel for arbitrary
``n''. See (need to cite something) for derivation.

!!! warning
    It takes ``O(n2^{2n})`` steps to generate. If one wants a large dimension,
    we suggest you save the resulting constraints.
"""
function generalWHLPConstraints(n :: Int, d :: Int, λ_vec :: Union{Vector{Float64},Vector{Int64}}) :: Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64},Matrix{Float64}}
    #Quick sanity checks
    if n > 11
        println("WARNING: You are trying to generate constraints at a size where the time it will take is non-trivial.")
    elseif length(λ_vec) != n
        throw(DomainError(λ_vec, "λ_vec must have the length of n."))
    elseif !all(λ_vec -> λ_vec >= 0 && λ_vec <= 1, λ_vec)
        throw(DomainError(λ_vec, "λ_vec must contain values in [0,1]"))
    end

    #Note that the objective function will need to be scaled by d after the calculation
    A = zeros(2^n,2^n)
    B = zeros(2^n,2^n)
    g = zeros(1,2^n)
    #a = zeros(2^n,1)
    ζ = ones(2^n,1)
    for s in [0:2^n - 1;]
        #Trace condition
        s_string = digits(Int8, s, base=2, pad=n) |> reverse
        w_s = sum(s_string) #This is the hamming weight
        g[s+1] = d^(n-w_s) #This is d to the power of the hamming weight
        for j in [0:2^n - 1;]
            j_string = digits(Int8, j, base=2, pad=n) |> reverse
            #This is the positivity condition
            #This is the hamming weight of bit and of s and j
            w_sj = sum(digits(Int8,(s&j),base=2,pad=n))
            A[j+1,s+1] = WH_coeff_sign(w_sj)
            #This is for the objective function
            if j <= n -1
                ζ[s+1] = ζ[s+1]*WH_lambda_coeff(j+1,s_string[j+1],λ_vec)
            end
            #This whole loop is the ppt constraint
            B_nonzero = true
            for i in [0:n-1;]
                #This is for the ppt constraint
                if s_string[i+1] == 1 && j_string[i+1] == 0
                    B_nonzero = false
                end
            end
            if B_nonzero
                B[j+1,s+1] = d^(w_s) #d^(-1. *(n - w_s)) original scaling
            else
                B[j+1,s+1] = 0
            end
        end
    end
    a = A*ζ
    return A, B, g, a
end
#These are helper functions
function WH_coeff_sign(x)
    if isodd(x)
        return -1
    else
        return 1
    end
end
function WH_lambda_coeff(i,bit,λ_vec)
    if bit == 0
        return λ_vec[i]
    else
        return (1-λ_vec[i])
    end
end

"""
    wernerHolevoCVPPT(
        n :: Int64
        d :: Int64,
        A :: Matrix{Float64},
        B :: Matrix{Float64},
        g :: Matrix{Float64},
        a :: Matrix{Float64}
    ):: Tuple{Float64, Matrix{Float64}}

This function evaluates the linear program for the PPT relaxation of the communication
value of the Werner-Holevo channel using the form that [`generalWHLPConstraints`](@ref)
outputs. It returns the cvPPT value and the optimizer.

!!! warning
    For ``n \\geq 10`` the solver may be slow.
"""
function wernerHolevoCVPPT(n :: Int64, d :: Int64, A :: Matrix{Float64}, B :: Matrix{Float64}, g :: Matrix{Float64}, a :: Matrix{Float64}) :: Tuple{Float64, Matrix{Float64}}
    v = Variable(2^n)
    objective = a' * v
    problem = maximize(objective)
    problem.constraints += [
        A * v >= 0 ; B * v >= 0 ; g * v == 1
    ]
    qsolve!(problem)
    cvPPT = d^n * problem.optval
    return cvPPT, v.value
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
