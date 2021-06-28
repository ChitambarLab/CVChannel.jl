using LinearAlgebra
using CVChannel
using Convex
using Test

"""
In this script we look at the PPT relaxation of the communication value of the
Werner-Holevo channel for multiple copies. This is feasible by using the
generalized LP.
"""

println("First we look at how the PPT relaxation of cv scales in number of copies.")
println("We do this for n ≤ 8 and λ in [0,0.1,...,1].") #It takes a lot more time if n = 9 & λ != 0
println("This also allows us to generate our non-multiplicativity results.")
d = 3
n_list = [1:8;]
λ_list = [0:0.1:1;]
data_table = zeros(length(n_list),length(λ_list))
for n_val in n_list
    println("Now obtaining values for n = ", n_val)
    λ_ctr = 1
    for λ_val in λ_list
        if λ_ctr == Int(ceil(length(λ_list)/2))
            println("Now halfway through this value of n.")
        end
        A,B,g,a = generalWHLPConstraints(n_val,d,λ_val*ones(n))
        cv_ppt, opt_var = wernerHolevoCVPPT(n,d,A,B,g,a)
        data_table[n_val,λ_ctr] = cv_ppt
        λ_ctr += 1
    end
end


cv_ppt, opt_var = wernerHolevoCVPPT(n,d,A,B,g,a)

println("Second, we look at how the PPT relaxation of cv scales in dimension.")
println("We do this for n ≤ 8, λ in [0,0.25,0.5,0.75], d in [10,10^2,...,10^7]")
p_list = [1:7;]
n_list = [1:8;]
λ_list = [0:0.25:0.75;]
λ_ctr = 1
for λ_val in λ_list
    for n_val in n_list
        for p_val in p_list
            d = 10^(p_val)
        end
    end
end
