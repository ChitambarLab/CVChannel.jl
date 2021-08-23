using CVChannel
using Test
using DelimitedFiles
"""
This script understands the communication value of the
generalized Siddhu channel.
"""

#This is if you want multiplicativity
s_range = [0:0.1:0.5;]
μ_range = [0:0.1:1;]
cv_table = zeros(length(s_range),length(μ_range))
par_cv_table = zeros(length(s_range),length(μ_range))
non_mult_table = zeros(length(s_range),length(μ_range))
s_ctr, μ_ctr = 1,1
for s_id in s_range
    println("---Now scanning for s=",s_id,"---")
    for μ_id in μ_range
        println("μ=",μ_id)
        genSidChan(X) = generalizedSiddhu(X,s_id,μ_id)
        gensid_chan = Choi(genSidChan,3,3)
        #To save time we only calculate cv once since its same channel
        cv, = pptCV(gensid_chan, :dual)
        par_choi = parChoi(gensid_chan,gensid_chan)
        parcv, = pptCV(par_choi, :primal)
        cv_table[s_ctr,μ_ctr] =cv
        par_cv_table[s_ctr,μ_ctr] = parcv
        non_mult = parcv - cv^2
        isapprox(non_mult,0,atol=1e-6) ?
            non_mult_table[s_ctr,μ_ctr] = 0 :
                non_mult_table[s_ctr,μ_ctr] = non_mult
        μ_ctr += 1
    end
    s_ctr += 1
    μ_ctr = 1
end

s_range = [0:0.1:0.5;]
μ_range = [0:0.1:1;]
data_table_ppt_cv = zeros(length(s_range),length(μ_range))
data_table_2sym_cv = zeros(length(s_range),length(μ_range))
data_table_3sym_cv = zeros(length(s_range),length(μ_range))
s_ctr, μ_ctr = 1,1
for s_id in s_range
    println("---Now scanning for s=",s_id,"---")
    for μ_id in μ_range
        println("μ=",μ_id)
        genSidChan(X) = generalizedSiddhu(X,s_id,μ_id)
        gensid_chan = Choi(genSidChan,3,3)
        cv_ppt, = pptCV(gensid_chan, :dual)
        #cv_2sym, = twoSymCVPrimal(gensid_chan)
        cv_3sym, = threeSymCVPrimal(gensid_chan)
        data_table_ppt_cv[s_ctr,μ_ctr] = cv_ppt
        #data_table_2sym_cv[s_ctr,μ_ctr] = cv_2sym
        data_table_3sym_cv[s_ctr,μ_ctr] = cv_3sym
        μ_ctr += 1
    end
    s_ctr += 1
    μ_ctr = 1
end

file_name = readline()
file_to_open = string(file_name,".csv")
writedlm(file_to_open, data_table_3sym_cv, ',')

using LinearAlgebra
using Convex

function genSiddhuParallel(ρ :: Matrix{<:Number}, s :: Union{Int,Float64}, μ :: Union{Int,Float64}) :: Matrix{<:Number}
    K0 = [sqrt(s) 0 0 ; 0 sqrt(1-μ) 0 ; 0 0 sqrt(μ)]
    K1 = [0 0 sqrt(1-μ) ; sqrt(1-s) 0 0 ; 0 sqrt(μ) 0]
    p1 = kron(K0,K0)*ρ*kron(K0,K0)' + kron(K0,K1)*ρ*kron(K0,K1)'
    p2 = kron(K1,K0)*ρ*kron(K1,K0)' + kron(K1,K1)*ρ*kron(K1,K1)'
    return p1 + p2
end

s,μ = 0.1, 0.5
genSidChan(X) = generalizedSiddhu(X,s,μ)
genSidParChan(X) = genSiddhuParallel(X,s,μ)
out1 = genSidChan([1 0 0 ; 0 0 0 ; 0 0 0])
out2 = genSidChan([0 0 0 ; 0 1 0 ; 0 0 0])
out3 = genSidChan([0 0 0 ; 0 0 0 ; 0 0 1])
gensid_chan = Choi(genSidChan,3,3)
cv, = pptCV(gensid_chan, :dual)
par_choi = parChoi(gensid_chan,gensid_chan)
max_ent = 1/3*vec([1 0 0 ; 0 1 0 ; 0 0 1])*vec([1 0 0 ; 0 1 0 ; 0 0 1])'
