using CVChannel
using Test
"""
This script understands the communication value of the
generalized Siddhu channel.
"""

s_range = [0:0.1:0.5;]
μ_range = [0:0.1:1;]
data_table = zeros(length(s_range),length(μ_range))
s_ctr, μ_ctr = 1,1
for s_id in s_range
    println("---Now scanning for s=",s_id,"---")
    for μ_id in μ_range
        println("μ=",μ_id)
        genSidChan(X) = generalizedSiddhu(X,s_id,μ_id)
        gensid_chan = Choi(genSidChan,3,3)
        cvppt, = pptCVDual(gensid_chan.JN,3,3)
        data_table[s_ctr,μ_ctr] = cvppt
        μ_ctr += 1
    end
    s_ctr += 1
    μ_ctr = 1
end
