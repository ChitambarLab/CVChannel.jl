using CVChannel
using Test
using DelimitedFiles

"""
This obtains data about the Werner-Holevo channel when tensored
with the dephrasure channel
"""

print("\nThis script shows that there is non-multiplicativity of")
print(" cvPPT for the Werner-Holevo channel tensored with the dephrasure channel.")
@testset "Werner-Holevo with Dephrasure is Non-Multiplicative" begin
    λ_range = [0:0.05:0.3;]
    p_range = [0,0.1,0.2,0.8,0.9,1]
    q_range = [0:0.1:1;]
    λ_ctr, p_ctr, q_ctr = 1,1,1
    data_table = zeros(length(λ_range),length(p_range),length(q_range))
    for λ in λ_range
        println("---Now scanning λ=",λ,"---")
        A,B,g,a = generalWHLPConstraints(1,3,λ*ones(1))
        cvWH, v1 =wernerHolevoCVPPT(1,3,A,B,g,a)
        whChan(X) = wernerHolevoChannel(X,λ)
        wh_chan = Choi(whChan,3,3)
        for q in q_range
            println("Now scanning q=",q,".")
            target_val = (2-q)*cvWH
            for p in p_range
                dephrasurepq(X) = dephrasureChannel(X,p,q)
                dephr_chan = Choi(dephrasurepq,2,3)

                par_cv, = pptCV(parChoi(wh_chan, dephr_chan), :dual)

                non_mult = par_cv - target_val
                isapprox(non_mult,0,atol=3e-6) ?
                    data_table[λ_ctr,p_ctr,q_ctr] = 0 :
                        data_table[λ_ctr,p_ctr,q_ctr] = non_mult

                p_ctr += 1
            end
            q_ctr += 1
            p_ctr = 1
        end
        λ_ctr += 1
        q_ctr = 1
    end

    println("\n Here is a subset of the data.")
    println("Non-multiplicativity for λ=0")
    show(stdout, "text/plain", data_table[1,:,:])
    println("Non-multiplicativity for λ=0.1")
    show(stdout, "text/plain", data_table[3,:,:])
    println("Non-multiplicativity for λ=0.2")
    show(stdout, "text/plain", data_table[5,:,:])
    println("Note that the non-multiplicativity is symmetric about p.")
    println("Also note that as λ increases, the non-multiplicativity decreases.")

    header, p_label = hcat("p|q:",[0:0.1:1;]'), p_range
    data_to_save = vcat(header,hcat(p_label, data_table[1,:,:]))
    for i in [2:length(λ_range);]
        data_to_save = vcat(data_to_save,hcat(p_label,data_table[i,:,:]))
    end
    info_vec = Vector{Union{Nothing,String}}(nothing, size(data_to_save)[1])
    info_vec[1] = "INFO:"
    info_vec[2] = "Generated by werner-holevo-with-dephrasure.jl"
    info_vec[3] = "λ increases as table goes down"
    info_vec[4] = "λ_range ="*string(λ_range)
    info_vec[5] = "p_range="*string(p_range)
    info_vec[6] = "q_range="*string(q_range)
    data_to_save = hcat(data_to_save,info_vec)

    println("Please name the file you'd like to write the results to: \n")
    file_name = readline()
    file_to_open = string(file_name,".csv")
    writedlm(file_to_open, data_to_save, ',')

    #This is sufficient for checking that there is non-multiplicativity somewhere
    @test data_table[1:1:1] > [0.1]
end
