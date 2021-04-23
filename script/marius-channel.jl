using LinearAlgebra
using CVChannel
using Convex
using Test

#To import less
function basisElm(i::Int,j::Int,dim::Int)::Matrix
    B = zeros(dim,dim)
    B[i,j] = 1
    return B
end

function mariusChannel(ρ :: Matrix{<:Number}) :: Matrix{<:Number}
    σX = [0 1 ; 1 0]; σY = [0 -1im ; 1im 0]; σZ = [1 0; 0 -1];
    id = [1 0 ; 0 1];
    c1 = kron(σX,id); c2 = kron(σY,id);
    c3 = kron(σZ,σX); c4 = kron(σZ,σY); c5 = kron(σZ,σZ);
    baseOps = [c1,c2,c3,c4,c5]

    ρout = zeros(5,5)
    for i in 1:5
        for j in 1:5
            ρout = ρout + 1/5*tr(baseOps[i]*ρ*baseOps[j])*basisElm(i,j,5)
        end
    end

    return ρout
end

orig_choi = choi(mariusChannel,4,5)
test1 = minEntropyPPTDual(orig_choi,4,5)
