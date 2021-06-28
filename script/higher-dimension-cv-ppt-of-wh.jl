using LinearAlgebra
using CVChannel
using Test

"""
This function allows you to calculate the PPT communication value of the Werner-Holevo
channel in higher dimensions. It does this by generating the constraints
as matrices.
"""
#These are helper functions
function coeff_sign(x)
    if isodd(x)
        return -1
    else
        return 1
    end
end
function lambda_coeff(i,bit,λ_vec)
    if bit == 0
        return λ_vec[i]
    else
        return (1-λ_vec[i])
    end
end

function generalWHLPConstraints(n,d,λ_vec)
    #Quick sanity checks
    if n > 11
        println("WARNING: You are trying to generate constraints at a size where the time it will take is non-trivial.")
    elseif length(λ_vec) != n
        throw(DomainError(λ_vec, "λ_vec must have the length of n."))
    elseif !all(λ_vec -> λ_vec >= 0 && λ_vec <= 1, λ_vec)
        throw(DomainError(λ_vec, "λ_vec must contain values in [0,1]"))
    end

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
                ζ[s+1] = ζ[s+1]*lambda_coeff(j+1,s_string[j+1],λ_vec)
            end
            #This whole loop is the ppt constraint
            B_nonzero = true
            for i in [0:n-1;]
                #This is for the ppt constraint
                if s_string[i+1] == 1 && j_string[i+1] == 0
                    B_nonzero = false
                end
            end
            if B_nonzero
                B[j+1,s+1] = d^(w_s) #d^(-1. *(n - w_s)) original scaling
            else
                B[j+1,s+1] = 0
            end
        end
    end
    a = A*ζ
    return A, B, g, a
end


n = 2 #This is the number of states you are tensoring
d = 3 #This is the dimension of your Werner-Holevo channels
λ_vec = 1*ones(1,n)
A,B,g,a = generalWHLPConstraints(n,d,λ_vec)
