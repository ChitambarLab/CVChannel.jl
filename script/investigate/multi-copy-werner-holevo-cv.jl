using LinearAlgebra
using CVChannel
using Test
using DelimitedFiles

"""
In this script we look at the PPT relaxation of the communication value of the
Werner-Holevo channel for multiple copies and its relation to the parameters
λ and the number of copies.
"""

println("Here we look at how the PPT relaxation of cv scales in number of copies.")
println("We do this for n ≤ 8 and λ in [0,0.1,...,1].") #It takes a lot more time if n = 9 & λ != 0
println("This also allows us to generate our non-multiplicativity results.")
@testset "Generate cv as function of n and lambda" begin
    d = 3
    n_list = [1:8;]
    λ_list = [0:0.01:1;]
    data_table = zeros(length(n_list),length(λ_list))
    for n_val in n_list
        println("Now obtaining values for n = ", n_val)
        λ_ctr = 1
        for λ_val in λ_list
            if λ_ctr == Int(ceil(length(λ_list)/2))
                println("Now halfway through this value of n.")
            end
            A,B,g,a = generalWHLPConstraints(n_val,d,λ_val*ones(n_val))
            cv_ppt, opt_var = wernerHolevoCVPPT(n_val,d,A,B,g,a)
            data_table[n_val,λ_ctr] = cv_ppt
            λ_ctr += 1
        end
    end
    #This generates the table about multiplicativity
    mult_table =[
        (data_table[2,:] - data_table[1,:].^2)';
        (data_table[4,:] - data_table[2,:].^2)';
        (data_table[8,:] - data_table[4,:].^2)';
    ]
    λ_header = zeros(1,length(λ_list))
    spacing = ""
    for i in 1:length(λ_list)
        λ_header[i] = λ_list[i]
        spacing = [spacing ""]
    end
    header_p_1 = hcat("n\\lambda:", λ_header)
    header_p_2 = hcat("lambda:", λ_header)
    col_inf_p_2 = ["cv(N^2) - cv(N)^2";"cv(N^4) - cv(N^2)^2";"cv(N^8) - cv(N^4)^2"]
    data_to_save_p_1 = vcat(hcat(n_list, data_table),spacing)
    data_to_save_p_2 = vcat(header_p_2,hcat(col_inf_p_2,mult_table))
    data_to_save = vcat(header_p_1,vcat(data_to_save_p_1,data_to_save_p_2))
    println("Here are the results:")
    show(stdout, "text/plain", data_to_save)

    info_vec = Vector{Union{Nothing,String}}(nothing, size(data_to_save)[1])
    info_vec[1] = "INFO:"
    info_vec[2] = "Generated by multi-copy-werner-holevo-cv.jl"
    info_vec[3] = "d = " * string(d)
    info_vec[4] = "n_list = " * string(n_list)
    info_vec[5] = "lambda_list = " * string(λ_list)
    data_to_save = hcat(data_to_save,info_vec)
    print("\nPlease name the file you'd like to write the results to: \n")
    file_name = readline()
    file_to_open = string(file_name,".csv")
    writedlm(file_to_open, data_to_save, ',')

    @test isapprox(data_table[2,1], 3, atol = 1e-6)
end
