using LinearAlgebra
using CVChannel
using Test
using DelimitedFiles

"""
In this script we look at the PPT relaxation of the communication value of the
Werner-Holevo channel for multiple copies and its relation to the parameters
λ and the number of copies.
"""

@testset "PPTcv in dimension" begin
    println("Here we are going to generate data about the PPT relaxation of cv.")
    println("Our interest is largely in how it scales in d.")
    println("We scan over for n ≤ 4, λ in [0,0.25,0.5], and d in [3,10,30,100,250,500,750]")

    d_list = [3,10,30,100,250,500,750]
    n_list = [1:4;]
    λ_list = [0,0.25,0.5]
    data_table = zeros(length(n_list),length(d_list),length(λ_list))
    λ_ctr = 1
    for λ_val in λ_list
        println("Now evaluating for λ = ", λ_val)
        for n_val in n_list
            println("Now evaluating for n = ", n_val)
            d_ctr = 1
            for d_val in d_list
                A,B,g,a = generalWHLPConstraints(n_val,d_val,λ_val*ones(n_val))
                cv_ppt, opt_var = wernerHolevoCVPPT(n_val,d_val,A,B,g,a)
                data_table[n_val,d_ctr,λ_ctr] = cv_ppt
                d_ctr += 1
            end
        end
        λ_ctr +=1
    end

    header = hcat(["n\\dim:"], d_list')
    data_table_0 = vcat(header,hcat(n_list,data_table[:,:,1]))
    data_table_1 = vcat(header,hcat(n_list,data_table[:,:,2]))
    data_table_2 = vcat(header,hcat(n_list,data_table[:,:,3]))
    println("Here are the results. Note that each converges to a value as d increases.")
    println("\nλ=", λ_list[1],":")
    show(stdout, "text/plain", data_table_0)
    println("\nλ=", λ_list[2],":")
    show(stdout, "text/plain", data_table_1)
    println("\nλ=", λ_list[3],":")
    show(stdout, "text/plain", data_table_2)

    ell = size(data_table_0)[2]
    data_to_save = vcat(data_table_0,zeros(ell)')
    data_to_save = vcat(data_to_save,vcat(data_table_1,zeros(ell)'))
    data_to_save = vcat(data_to_save,data_table_2)
    info_vec = Vector{Union{Nothing,String}}(nothing, size(data_to_save)[1])
    info_vec[1] = "INFO:"
    info_vec[2] = "Generated by multi-copy-werner-holevo-dimension-scaling.jl"
    info_vec[3] = "Tables descend as lambda increases"
    info_vec[4] = "d_list" * string(d_list)
    info_vec[5] = "n_list" * string(n_list)
    info_vec[6] = "lambda_list" * string(λ_list)
    data_to_save = hcat(data_to_save,info_vec)

    print("\nPlease name the file you'd like to write the results to: \n")
    file_name = readline()
    file_to_open = string(file_name,".csv")
    writedlm(file_to_open, data_to_save, ',')

    print("\nWe verify that the values converge for all lambda for")
    print(" n<4, because for n=4, the LP doesn't behave well (see printed data). \n")
    @test isapprox(data_table_0[2:4,7],data_table_0[2:4,8],atol=1e-2)
    @test isapprox(data_table_1[2:4,7],data_table_1[2:4,8],atol=1e-2)
    @test isapprox(data_table_2[2:4,7],data_table_2[2:4,8],atol=1e-2)
end
