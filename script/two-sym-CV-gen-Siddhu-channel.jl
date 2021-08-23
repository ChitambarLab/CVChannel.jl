using CVChannel
using Test
using DelimitedFiles
"""
This script looks at the communication value of the
generalized Siddhu channel using SDP relaxations.
Specifically, it shows that 2-sym cv is effectively
the same as cv PPT, and that they are loose at least
some of the time. It also shows multiplicativity over PPT cone.
"""
@testset "Investigate generalized Siddhu channel" begin
    println("We calculate cvPPT, 2symCV, and cvPPT of the channel ran in parallel.")
    println("We then show that 2symCV provides no improvement to cvPPT and that cvPPT")
    println("is effectively multiplicative for the gen Siddhu with itself.")
    s_range = [0:0.1:0.5;]
    μ_range = [0:0.1:1;]
    cv_table = zeros(length(s_range),length(μ_range))
    par_cv_table = zeros(length(s_range),length(μ_range))
    twosym_cv_table = zeros(length(s_range),length(μ_range))
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
            cv_2sym, = twoSymCVPrimal(gensid_chan)
            cv_table[s_ctr,μ_ctr] =cv
            par_cv_table[s_ctr,μ_ctr] = parcv
            twosym_cv_table[s_ctr,μ_ctr] = cv_2sym
            non_mult_table[s_ctr,μ_ctr] = parcv - cv^2
            μ_ctr += 1
        end
        s_ctr += 1
        μ_ctr = 1
    end

    println("First we verify that cvPPT is effectively multiplicative over the whole range.")
    @test all(non_mult_table -> non_mult_table < 2e-5 , non_mult_table[:,:])

    println("Next we verify that cvPPT is approximately 2symCV over the whole range.")
    diff = twosym_cv_table - cv_table
    @test all(diff -> diff < 3e-6, diff[:,:])

    println("Given this we just look at the 2symCV of the channel.")

    info_vec = Vector{Union{Nothing,String}}(nothing, length(s_range)+1)
    label_vec = hcat("s↓ μ:",μ_range')
    info_vec[1] = "INFO:"
    info_vec[2] = "Generated by two-sym-CV-gen-Siddhu-channel.jl"
    info_vec[3] = "s_range = " * string(s_range)
    info_vec[4] = "μ_range = " * string(μ_range)
    data_to_save = hcat(vcat(label_vec,hcat(s_range,twosym_cv_table)),info_vec)

    println("Here is the 2symCV of the channel:")
    show(stdout, "text/plain", data_to_save)
    println("\nPlease name the file you'd like to write these results to: \n")
    file_name = readline()
    file_to_open = string(file_name,".csv")
    writedlm(file_to_open, data_to_save, ',')
end