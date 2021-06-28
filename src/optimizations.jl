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
    generalWHLPConstraints(n :: Int, d :: Int, λ_vec :: Vector{Float64})
    :: Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64},Matrix{Float64}}

This function returns the linear program constraints for calculating the PPT
communication value of the Werner-Holevo channels run in parallel for arbitrary
``n''. See (need to cite something) for derivation.

!!! warning
    It takes ``O(n2^{2n})`` steps to generate. If one wants a large dimension,
    we suggest you save the resulting constraints.
"""
function generalWHLPConstraints(n :: Int, d :: Int, λ_vec :: Vector{Float64}) :: Tuple{Matrix{Float64},Matrix{Float64},Matrix{Float64},Matrix{Float64}}
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
