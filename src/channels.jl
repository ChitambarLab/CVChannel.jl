"""
    is_choi_matrix(JN :: AbstractMatrix, dimA :: Int, dimB :: Int) :: Bool

Returns `true` if the supplied matrix `JN` is a Choi operator.
This function returns `false` if
* `size(JN) != (dimA * dimB, dimA * dimB)`
"""
function is_choi_matrix(JN :: AbstractMatrix, dimA :: Int, dimB :: Int) :: Bool
    JN_dim = dimA*dimB
    if size(JN) != (JN_dim, JN_dim)
        return false
    end

    return true
end

"""
    Choi( JN :: AbstractMatrix, in_dim :: Int, out_dim :: Int ) :: Choi{<:Number}

    Choi( N :: Function, in_dim :: Int, out_dim :: Int ) :: Choi{ComplexF64}

    Choi( kraus_ops :: Vector{<:AbstractMatrix} ) :: Choi{ComplexF64}

Constructs the Choi matrix representation of a quantum channel.
If either a function `N` or set of kraus operators is provided as input, the
Choi matrix is constructed with the [`choi`](@ref) method.

The `Choi` type contains the fields:
* `JN :: Matrix{<:Number}` - The choi matrix.
* `in_dim :: Int` - The channel's input dimension.
* `out_dim :: Int` - The Channel's output dimension.

A `DomainError` is thrown if [`is_choi_matrix`](@ref) returns `false`.
"""
struct Choi{T<:Number}
    JN :: Matrix{T}
    in_dim :: Int
    out_dim :: Int
    Choi(
        JN :: AbstractMatrix,
        in_dim :: Int,
        out_dim :: Int
    ) = is_choi_matrix(JN, in_dim, out_dim) ? new{eltype(JN)}(JN, in_dim, out_dim) : throw(
        DomainError(JN, "The Choi matrix dimensions are not valid.")
    )
    Choi(
        N :: Function,
        in_dim :: Int,
        out_dim :: Int
    ) = Choi( choi(N, in_dim, out_dim), in_dim, out_dim)
    Choi(
        kraus_ops :: Vector{<:AbstractMatrix},
    ) = Choi(choi(kraus_ops), size(kraus_ops[1])...)
end

# print out matrix forms when Choi types are displayed
function show(io::IO, mime::MIME{Symbol("text/plain")}, choi_op :: Choi)
    summary(io, choi_op)
    print(io, "\nin_dim : ", choi_op.in_dim, ", out_dim : ", choi_op.out_dim)
    print(io, "\nJN : ")
    show(io, mime, choi_op.JN)
end

"""
    parChoi(chan1 :: Choi, chan2 :: Choi) :: Choi

Returns the tensor product of two [`Choi`](@ref) matrices

```math
    J^{A:B}_{\\mathcal{N}}\\otimes J^{A':B'}_{\\mathcal{M}} \\to
    J^{AA':BB'}_{\\mathcal{N}\\otimes\\mathcal{M}}
```

where ``J^{A:B}_{\\mathcal{N}}`` and ``J^{A':B'}_{\\mathcal{M}}`` are the Choi
matrices for `chan1` and `chan2` respectively.
Note the implicit swap between systems ``B \\leftrightarrow A'``.
"""
function parChoi(chan1 :: Choi, chan2 :: Choi) :: Choi
    par_dims = [chan1.in_dim, chan1.out_dim, chan2.in_dim, chan2.out_dim]
    par_JN = permuteSubsystems(kron(chan1.JN, chan2.JN), [1,3,2,4], par_dims)
    par_in_dim = chan1.in_dim * chan2.in_dim
    par_out_dim = chan1.out_dim * chan2.out_dim

    return Choi(par_JN, par_in_dim, par_out_dim)
end

"""
    choi(𝒩 :: Function, Σ :: Int, Λ :: Int) :: Matrix{ComplexF64}
    choi(kraus_ops :: Vector{<:AbstractMatrix}) :: Matrix{ComplexF64}

This function returns the Choi state of a channel represented either as a generic
function `𝒩` or a set of Kraus operators `kraus_ops`. It does this using that

```math
        J(\\mathcal{N}) = \\sum_{a,b \\in \\Sigma} E_{a,b} \\otimes \\mathcal{N}(E_{a,b}) ,
```

where ``\\Sigma`` is the finite alphabet indexing the input space and ``E_{a,b}``
is a square matrix of dimension ``\\Sigma`` with a ``1`` in the ``(a,b)`` entry
and a ``0`` everywhere else. The input ``\\Lambda`` is the output dimension.
Note this assumes you have a function that calculates
``\\mathcal{N}(X)`` for arbitrary input ``X``. As many of the functions for channels
in this module have multiple parameters, please note that if you have a channel function
`𝒩(ρ, p, q)` that calculates ``\\mathcal{N}_{p,q}(\\rho)``, you can declare a function
`𝒩_xy(ρ) = 𝒩(ρ,x,y)` for fixed `(x,y)` and then call, `choi(𝒩_xy, Σ)`.
"""
function choi(𝒩 :: Function, Σ :: Int, Λ :: Int) :: Matrix{ComplexF64}
    eab_matrix = zeros(ComplexF64, Σ,Σ)
    choi_matrix = zeros(ComplexF64, Σ*Λ,Σ*Λ)
    for i in 1:Σ
        row_ids = ((i-1) * Λ + 1):((i-1) * Λ + Λ)
        for j in 1:Σ
            col_ids = ((j-1) * Λ + 1):((j-1) * Λ + Λ)

            eab_matrix[i,j] = 1
            choi_matrix[row_ids,col_ids] += 𝒩(eab_matrix)
            eab_matrix[i,j] = 0
        end
    end
    return choi_matrix
end
function choi(kraus_ops :: Vector{<:AbstractMatrix}) :: Matrix{ComplexF64}
    kraus_channel(ρ) = krausAction(kraus_ops, ρ)
    dim_out, dim_in = size(kraus_ops[1])

    choi(kraus_channel, dim_in, dim_out)
end

"""
    isometricChannel(kraus_ops :: Vector) :: Matrix

This function builds the isometric representation ``V`` of a
channel ``\\mathcal{N}: A \\to B`` from the Kraus operators ``\\{K_{i}\\}``.
It does this by calculating
```math
    V = \\sum_{i} K_{i} \\otimes |i\\rangle
```
"""
function isometricChannel(kraus_ops :: Vector) :: Matrix
    dimB , dimA = size(kraus_ops[1])
    dimE = length(kraus_ops)
    V, e_vec = zeros(dimE*dimB,dimA), zeros(dimE)
    for i in 0:dimE-1;
        V[1+i*dimB:(i+1)*dimB,1:dimA] = kraus_ops[i+1]
    end
    label_vec = [1:dimB*dimE;]
    perms_ids_vec = permuteSubsystems(label_vec,[2,1],[dimE,dimB])
    V = V[perms_ids_vec,:]

    return V
end

"""
    complementaryChannel(kraus_ops :: Vector) :: Vector

This function takes a set of Kraus operators for a channel ``\\mathcal{N}_{A \\to B}``
and returns a set of Kraus operators for the complementary channel,
``\\mathcal{N}^{c}_{A \\to E}``. It does this by generating the Kraus operators of
the isometric representation of the channel followed by partial trace on the
``B`` space.
!!! info
    If ``\\mathcal{N}_{A \\to B}`` is already isometric, the code lets dimE=2
    so that functions are well behaved.
"""
function complementaryChannel(kraus_ops :: Vector) :: Vector
    V = isometricChannel(kraus_ops)
    dimB , dimA = size(kraus_ops[1])
    dimE = length(kraus_ops)

    comp_kraus = Vector{Matrix{Complex{Float64}}}(undef, dimB)
    if dimE == 1
        #If dimE = 1, the complementary channel is the replacer channel
        #To get it to work with other functions, we let dimE=2 so it isn't scalar
        b_vec = zeros(dimB)
        for i in [1:dimB;]
            b_vec[i] = 1
            comp_kraus[i] = kron([1;0],b_vec')
            b_vec[i] = 0
        end
    else
        b_vec, e_id = zeros(dimB), Matrix(1I,dimE,dimE)
        for i in [1:dimB;]
            b_vec[i] = 1
            comp_kraus[i] = kron(b_vec,e_id)'*V
            b_vec[i] = 0
        end
    end

    return comp_kraus
end

"""
    krausAction(kraus_ops :: Vector, X)

This function takes a set of Kraus operators for a channel
``\\mathcal{N}_{A \\to B}`` and returns the output of the channel
for input ``X``. That is, given input ``X`` it returns
```math
    \\mathcal{N}_{A \\to B}(X) = \\sum_{i} K_{i} X K_{i}^{\\ast}
```
"""
function krausAction(kraus_ops :: Vector, X)
    dimB, dimA = size(kraus_ops[1])
    out = zeros(dimB,dimB)
    for i = [1:length(kraus_ops);]
        out += kraus_ops[i]*X*kraus_ops[i]'
    end
    return out
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
#The aligned below makes the documenter space the bmatrix properly
"""
    siddhuChannel(ρ :: Matrix{<:Number}, s :: Union{Int,Float64}) :: Matrix{<:Number}

This function calculates the action of the Siddhu channel ``N_{s}`` which is defined by Kraus operators:
```math
    \\begin{aligned}
    K_{0} = \\begin{bmatrix} \\sqrt{s} & 0 & 0 \\\\ 0 & 0 & 0 \\\\ 0 & 1 & 0 \\end{bmatrix}
    \\hspace{5mm}
    K_{1} = \\begin{bmatrix} 0 & 0 & 0 \\\\ \\sqrt{1-s} & 0 & 0 \\\\ 0 & 0 & 1 \\end{bmatrix} ,
    \\end{aligned}
```
where ``s \\in [0,1/2]``.
This channel was introduced in Equation 9 of [this paper](https://arxiv.org/abs/2003.10367).
"""
function siddhuChannel(ρ :: Matrix{<:Number}, s :: Union{Int,Float64}) :: Matrix{<:Number}
    if !isequal(size(ρ)...)
        throw(DomainError(ρ, "the input ρ is not a square matrix"))
    elseif size(ρ)[1] != 3
        throw(DomainError(ρ, "The input must be a qutrit operator."))
    elseif !(0 ≤ s ≤ 1/2)
        throw(DomainError(s, "siddhuChannel requires s ∈ [0,1/2]."))
    end

    K0 = [sqrt(s) 0 0 ; 0 0 0 ; 0 1 0]
    K1 = [0 0 0 ; sqrt(1-s) 0 0 ; 0 0 1]
    return K0*ρ*K0' + K1*ρ*K1'
end

"""
    generalizedSiddhu(
          ρ :: Matrix{<:Number},
          s :: Union{Int,Float64},
          μ :: Union{Int,Float64}
    ) :: Matrix{<:Number}

This function calculates the action of the generalized Siddhu channel
``\\mathcal{N}_{s,\\mu}`` on the qutrit state ``\\rho``.
The action of the channel is defined by Kraus operators:
```math
    \\begin{aligned}
    K_{0} = \\begin{bmatrix}
                \\sqrt{s} & 0 & 0 \\\\
                0 & \\sqrt{1-\\mu} & 0 \\\\
                0 & 0 & \\sqrt{\\mu}
             \\end{bmatrix}
    \\hspace{5mm}
    K_{1} = \\begin{bmatrix}
                0 & 0 & \\sqrt{1-\\mu} \\\\
                \\sqrt{1-s} & 0 & 0 \\\\
                0 & \\sqrt{\\mu} & 0
            \\end{bmatrix} ,
    \\end{aligned}
```
where ``s \\in [0,1/2]``, ``\\mu \\in [0,1]``.
This channel was introduced by Leditzky *et al.* (cite when on arxiv).

A `DomainError` is thrown if:
* Matrix `ρ` is not square
* `ρ` is not a 3-dimensional matrix
* `s` is not in range `0 ≤ s ≤ 1/2`
* `μ` is not in range `0 ≤ μ ≤ 1`
"""
function generalizedSiddhu(ρ :: Matrix{<:Number}, s :: Union{Int,Float64}, μ :: Union{Int,Float64}) :: Matrix{<:Number}
    if !isequal(size(ρ)...)
        throw(DomainError(ρ, "the input ρ is not a square matrix"))
    elseif size(ρ)[1] != 3
        throw(DomainError(ρ, "The input must be a qutrit operator."))
    elseif !(0 ≤ s ≤ 1/2)
        throw(DomainError(s, "generalizedSiddhu requires s ∈ [0,1/2]."))
    elseif !(0 ≤ μ ≤ 1)
        throw(DomainError(μ, "generalizedSiddhu requires μ ∈ [0,1]."))
    end

    K0 = [sqrt(s) 0 0 ; 0 sqrt(1-μ) 0 ; 0 0 sqrt(μ)]
    K1 = [0 0 sqrt(1-μ) ; sqrt(1-s) 0 0 ; 0 sqrt(μ) 0]
    return K0*ρ*K0' + K1*ρ*K1'
end

"""
    GADChannel(
        ρ :: Matrix{<:Number},
        p :: Union{Int,Float64},
        n :: Union{Int,Float64}
    ) :: Matrix{<:Number}

This function calculates the action of the generalized (qubit) amplitude damping channel ``\\mathcal{A}_{p,n}``
which is defined by Kraus operators:
```math
    \\begin{aligned}
    K_{0} =& \\sqrt{1-n} \\begin{bmatrix} 1 & 0 \\\\ 0 & \\sqrt{1-p} \\end{bmatrix}
    \\hspace{5mm}
    K_{1} =& \\sqrt{p(1-n)} \\begin{bmatrix} 0 & 1 \\\\ 0 & 0 \\end{bmatrix}  \\\\
    K_{2} =& \\sqrt{n} \\begin{bmatrix} \\sqrt{1-p} & 0 \\\\ 0 & 1 \\end{bmatrix}
    \\hspace{5mm}
    K_{3} =& \\sqrt{pn} \\begin{bmatrix} 0 & 0 \\\\ 1 & 0 \\end{bmatrix}
    \\end{aligned}
```
where ``p,n \\in [0,1]``.
This channel may be found in Section 3 of [this paper](https://arxiv.org/abs/2107.13486).
"""
function GADChannel(ρ :: Matrix{<:Number}, p :: Union{Int,Float64}, n :: Union{Int,Float64}) :: Matrix{<:Number}
    if !isequal(size(ρ)...)
        throw(DomainError(ρ, "the input ρ is not a square matrix"))
    elseif size(ρ)[1] != 2
        throw(DomainError(ρ, "The input must be a qubit operator."))
    elseif !(0 ≤ p ≤ 1)
        throw(DomainError(p, "siddhuChannel requires p ∈ [0,1]."))
    elseif !(0 ≤ n ≤ 1)
        throw(DomainError(n, "siddhuChannel requires n ∈ [0,1]."))
    end

    K0 = sqrt(1-n)*[1 0 ; 0 sqrt(1-p)]
    K1 = sqrt(p*(1-n))*[0 1 ; 0 0]
    K2 = sqrt(n)*[sqrt(1-p) 0 ; 0 1]
    K3 = sqrt(p*n)*[0 0 ; 1 0]
    return K0*ρ*K0' + K1*ρ*K1' + K2*ρ*K2' + K3*ρ*K3'
end
