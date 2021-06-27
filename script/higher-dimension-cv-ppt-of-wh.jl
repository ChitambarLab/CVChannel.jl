using LinearAlgebra
using CVChannel
using Test

"""
This function calculates the PPT communication value of the Werner-Holevo
channel in higher dimensions. It does this by generating the constraints
as matrices.
"""

n = 2 #This is the number of states you are tensoring
d = 3 #This is the dimension of your Werner-Holevo channels


#These are helper functions
function coeff_sign(x)
    if isodd(x)
        return -1
    else
        return 1
    end
end
#For simplicity, we assume its the same channel n fold times. The function is
#written so it could be altered for the more general case easily, by hard-coding
#the lambda vector into the function and then using i to pick which lambda
function lambda_coeff(i,bit)
    λ = 0.9
    λ_vec = λ*ones(4)
    if bit == 0
        return λ_vec[i]
    else
        return (1-λ_vec[i])
    end
end

function generalLPConstraints()
    #Note that the objective function will need to be scaled by d after the calculation
    A = zeros(2^n,2^n)
    B = zeros(2^n,2^n)
    g = zeros(1,2^n)
    #a = zeros(2^n,1)
    ζ = ones(2^n,1)
    for s in [0:2^n - 1;]
        #Trace condition
        s_string = digits(Int8, s, base=2, pad=n) |> reverse
        w_s = sum(s_string) #This is the hamming weight
        g[s+1] = d^(n-w_s) #This is d to the power of the hamming weight
        for j in [0:2^n - 1;]
            j_string = digits(Int8, j, base=2, pad=n) |> reverse
            #This is the positivity condition
            #This is the hamming weight of bit and of s and j
            w_sj = sum(digits(Int8,(s&j),base=2,pad=n))
            A[j+1,s+1] = coeff_sign(w_sj)
            #This is for the objective function
            if j <= n -1
                ζ[s+1] = ζ[s+1]*lambda_coeff(j+1,s_string[j+1])
            end
            #This whole loop is the ppt constraint
            B_nonzero = true
            for i in [0:n-1;]
                #This is for the ppt constraint
                if s_string[i+1] == 1 && j_string[i+1] == 0
                    B_nonzero = false
                end
                #println(ζ)
            end
            if B_nonzero
                B[j+1,s+1] = d^(w_s) #d^(-1. *(n - w_s)) original scaling
            else
                B[j+1,s+1] = 0
            end
        end
    end
    #println(A)
    #println(ζ)
    a = A*ζ
    return A, B, g, a
end

A,B,g,a = generalLPConstraints()
