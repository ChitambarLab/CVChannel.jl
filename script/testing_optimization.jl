using Convex, SCS
using LinearAlgebra
using CVChannel

function cvkPrimal(k :: Integer, kraus_ops :: Vector{Any})
    dimB , dimA = size(kraus_ops[1])
    length(kraus_ops) == 1 ? dimE = 2 : dimE = length(kraus_ops)
    idX = Matrix(1I,k,k)
    maxEnt = 1/k*vec(idX)*vec(idX)'

    #Declare variables
    σ = HermitianSemidefinite(k*dimA*dimE)
    ρ = HermitianSemidefinite(k^2*dimA^2)
    X = ComplexVariable(k^2*dimA*dimE, k^2*dimA*dimE)

    #Begin problem
    objective = 1/2*real(tr(X) + tr(X'))
    problem = maximize(objective)

    #Add constraints
    problem.constraints += [tr(ρ) == 1]
    problem.constraints += [tr(σ) == 1]
    problem.constraints += [partialtrace(partialtrace(partialtrace(ρ,1,[k,k,dimA,dimA]),1,[k,dimA,dimA]),1,[dimA,dimA])==partialtrace(partialtrace(partialtrace(ρ,1,[k,k,dimA,dimA]),1,[k,dimA,dimA]),2,[dimA,dimA])]
    problem.constraints += partialtrace(partialtrace(ρ,4,[k,k,dimA,dimA]),3,[k,k,dimA]) == maxEnt
    problem.constraints += [_pmap(k,kraus_ops,ρ) X ; X' kron(idX,σ)] ⪰ 0
    #Solve the problem, return value
    solve!(problem,SCS.Optimizer)
    cvk = k*(problem.optval)^2
    return cvk, problem.optval, ρ.value, σ.value, X.value
end

function _pmap(k :: Integer ,kraus_ops :: Vector{Any}, ρ)
    dimB , dimA = size(kraus_ops[1])
    idX2, idA = Matrix(1I,k^2,k^2), Matrix(1I,dimA,dimA)
    comp_kraus = complementaryChannel(kraus_ops)
    big_kraus = Any[]
    for i in [1:length(comp_kraus);]
        push!(big_kraus, kron(idX2,kron(idA,comp_kraus[i])))
    end
    return krausAction(big_kraus,ρ)
end

####Tets for cvkPrimal####
testkraus = Any[]
#Identity channel (Need at least two kraus operators)
idKraus = Matrix(1I,3,3)
push!(testkraus,idKraus)
push!(testkraus,idKraus)
push!(testkraus,idKraus)
#Replacer Channel
#push!(testkraus,[1 0; 0 0])
#push!(testkraus,[0 1; 0 0])
cvk, optval, ρ, σ, X = cvkPrimal(3,testkraus)

t = _pmap(2,testkraus,ρ)

####Tests for _prim_map####
testkraus = Any[]
#push!(testkraus,[1 0; 0 1])
push!(testkraus,1/sqrt(2)*[1 0 ; 0 -1])
push!(testkraus,1/sqrt(2)*[0 1; 1 0])
H = [1 0 ; 0 0 ; 0 0 ; 0 1 ; 1/2 0 ; 0 1/2]
#H = [1 2 ; 3 4 ; 5 6 ; 7 8 ; 9 10 ; 11 12]

dimB , dimA = size(testkraus[1])
comp_kraus = complementaryChannel(testkraus)
t = krausAction(comp_kraus,H[1:dimA,1:dimA])
test = _prim_map(2,H,testkraus)

function cvkPrimalFancy(k :: Integer, kraus_ops :: Vector{Any})
    dimB , dimA = size(kraus_ops[1])
    length(kraus_ops) == 1 ? dimE = 2 : dimE = length(kraus_ops)
    idX = Matrix(1I,k,k)

    #Declare variables
    σ = HermitianSemidefinite(k*dimE)
    #all Hs stuck on top of eachother
    H = ComplexVariable(Int(dimA*k*(k+1)/2),dimA)
    X = ComplexVariable(k^2*dimE, k^2*dimE)

    #Begin problem
    objective = real(tr(X) + tr(X'))
    problem = maximize(objective)

    #Add constraints
    problem.constraints += [tr(_prim_map(k,H,kraus_ops)) == 1]
    problem.constraints += [tr(σ) == 1]
    #Note that _prim_map, σ is positive by this positivity constraint
    problem.constraints += [[ _prim_map(k,H,kraus_ops) X ; X' kron(idX, σ)] ⪰ 0]
    for i in [1:Int(k*(k+1)/2);] #Hermitian constraints
        problem.constraints += H[Int((i-1)*dimA+1):Int(i*dimA),1:dimA] == H[Int((i-1)*dimA+1):Int(i*dimA),1:dimA]'
    end

    #Solve the problem, return value
    solve!(problem,SCS.Optimizer)
    cvk = k*(problem.optval/2)^2
    return cvk, problem.optval, H.value, σ.value, X.value
end


function classicalcoh_map(k :: Int, dimA :: Int, X)
    dimA2 = Int(dimA^2)
    idA2 = Matrix(1I,dimA2,dimA2)
    baseProj = Any[]
    evec = zeros(k)
    for i in [1:k;]
        evec[i] = 1
        push!(baseProj,kron(kron(evec*evec',evec*evec'),idA2))
        evec[i] = 0
    end

    out = zeros(k^2*dimA^2,k^2*dimA^2)
    for i in [1:k;]
        for j in [1:k;]
            out += baseProj[i]*X*baseProj[j]
        end
    end
    return out
end

k, dimA = 3,2
clcoh_map(X) = classicalcoh_map(k,dimA,X)
t = choi(clcoh_map,k^2*dimA^2,k^2*dimA^2)
