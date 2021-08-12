using Convex, SCS
using LinearAlgebra

#Do not type the output
function testChan(X :: Variable)
    krausOps = Any[]
    push!(krausOps,sqrt(1/2)*[1 0 ; 0 -1])
    push!(krausOps,sqrt(1/2)*[0 1 ; 1 0])
    out = zeros(2,2)
    for i in [1:length(krausOps);]
        out += krausOps[i]*X*krausOps[i]'
    end
    return out
end

X = HermitianSemidefinite(2)
objective = real(tr(X' * [0 1 ; 1 0]))
problem = maximize(objective)
problem.constraints += [tr(X) == 1]
problem.constraints += [testChan(X) == X]
solve!(problem,SCS.Optimizer)
α, X = problem.optval, X.value
