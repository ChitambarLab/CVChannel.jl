using LinearAlgebra
using CVChannel
using Convex
using Test
using DelimitedFiles

"""
This code first verifies the non-multiplicativity of the Werner-Holevo over the PPT
cone when tensored with the identity. This is different than over the separable cone.
We verify this using an LP that we scan over.
Second, this code investigates the noise robustness of the Werner-Holevo channel
"""
#This script looks at multiplicativity of WH with identity over PPT cone
println("\nFirst we convince ourselves that the choi states of the Werner-Holevo (WH)")
println("channels are the werner states multiplied by the dimension, so that we can")
println("just use the werner states.")

@testset "Verifying werner states are choi states of WH channels" begin
    wernerHolevoChannel_0(ρ) = wernerHolevoChannel(ρ,0)
    wernerHolevoChannel_1(ρ) = wernerHolevoChannel(ρ,1)
    wernerHolevoChannel_05(ρ) = wernerHolevoChannel(ρ,0.5)
    @test 3*wernerState(3,0) == choi(wernerHolevoChannel_0,3,3)
    @test 3*wernerState(3,1) == choi(wernerHolevoChannel_1,3,3)
    @test 3*(0.5*wernerState(3,0)+0.5*wernerState(3,1)) == choi(wernerHolevoChannel_05,3,3)
end

println("\nHere we initialize the identity channel and the solver. (One moment please...)")
identChan(X) = X
eaCVDual(choi(identChan,2,2),2,2)

println("\nNow we look at tensoring the Werner-Holevo channel with the identity.")
println("\nWhat makes this interesting is it is non-multiplcative over the separable")
println("cone, but over the PPT it is multiplicative. This exemplifies a separation.\n\n")

""" I left this here but I don't know why anyone would want it
println("First we use the SDP, but we also have an LP for it. The SDP is just for the curious. \n")
println("Otherwise, you can just skip it, since it's slower. Would you like to skip SDP? (y/n)")

function wantToRunSDP()
    prop_response = false
    while !prop_response
        user_response = readline()
        if user_response=="y"
            return true
        elseif user_response == "n"
            return false
        else
            print("Please respond with `y` or `n`. \n ")
        end
    end
end

usr_choice = wantToRunSDP()

if !usr_choice
    for wern_dim = 2:4
        for ident_dim = 2:4
            if wern_dim == 4 && ident_dim == 4

            else
                wern_choi = wern_dim*wernerState(wern_dim,0)
                wern_cv = pptCVDual(wern_choi,wern_dim,wern_dim)
                ident_choi = choi(identChan,ident_dim,ident_dim)
                ident_cv = ident_dim

                kron_par_choi = kron(wern_choi,ident_choi)
                par_choi = permuteSubsystems(kron_par_choi,[1,3,2,4],[wern_dim,wern_dim,ident_dim,ident_dim])
                par_cv = pptCVPrimal(par_choi,wern_dim*ident_dim,wern_dim*ident_dim)

                diff_val = par_cv[1] - wern_cv[1]*ident_cv
                print("\n Werner Dim= ", wern_dim, ". id Dim = ", ident_dim, ". Multiplicativity = ", diff_val, ".")
            end
        end
    end
end
"""

function nonMultWHID()
    println("\nNow we look at the Werner-Holevo channels tensored with the identity")
    println("\nfor dimension [2,8] for both channels for values of p in [0.4,0.5,...,1]")

    table_vals = zeros(343,7)
    λ_vals = [1,0.9,0.8,0.7,0.6,0.5,0.4];
    λ_ctr = 1;
    table_ctr = 1;
    for λ in λ_vals
        print("\nNow evaluating for Werner-Holevo channel with parameter = ", round(1-λ, digits = 2))
        #Note the LP is parameterized backwards with respect to how our SDP is
        for d1 = 2:8
            println("\nNow evaluating for d = ", d1, "...")
            #Here we get the communication value of the Werner-Holevo channel itself using an
            wern_choi = d1*wernerState(d1,(1-λ))
            wern_cv = pptCVDual(wern_choi,d1,d1)
            cv_WH = wern_cv[1]

            for d2 = 2:8
                #This is the vector of variables and we use alphabetical order x[1] = w, x[2] = x,...
                v = Variable(4)
                #Given the number of constraints we define the problem and add constraints
                objective = (v[1]+v[3]*d2+(1-2λ)*(v[2]+v[4]*d2))
                problem = maximize(objective)
                problem.constraints += [0 <= v[1]-v[2]+d2*v[3]-d2*v[4],
                                0 <= v[1]-v[2],
                                0 <= v[1]+v[2]+d2*v[3]+d2*v[4],
                                0 <= v[1]+v[2],
                                0 <= v[1] + d1*v[2] - v[3] - d1*v[4],
                                0 <= v[1] - v[3],
                                0 <= v[1] + d1*v[2] + v[3] + d1*v[4],
                                0 <= v[1] + v[3],
                                1 == d1*d2*v[1]+d2*v[2]+d1*v[3] + v[4]
                                ]
                                qsolve!(problem)
                table_vals[table_ctr,:] = [1-λ, #parameter
                                            d1,  #WH dimension
                                            d2,  #identity channel dimension
                                            cv_WH,  #comm val of WH
                                            d2,     #comm val of id (redundant)
                                            d1*d2*problem.optval, #comm val of WH ⊗ id
                                            d1*d2*problem.optval - cv_WH*d2]; #Non-multiplicativity over PPT
                table_ctr = table_ctr + 1;
            end
        end
        λ_ctr = λ_ctr + 1
    end

    table_vals = vcat(["parameter" "WH Dim" "id Dim" "cv(WH)" "cv(id)" "cv(WH o id)" "Non-mult"],table_vals)

    print("\nPlease name the file you'd like to write the results to: \n")
    file_name = readline()
    file_to_open = string(file_name,".csv")
    writedlm(file_to_open, table_vals, ',')
    if all(table_vals -> table_vals < 1e-4 , table_vals[2:end,7])
        print("\nSomething went awry. Please check your results.")
        return false
    else
        print("\nIf you check your results, you will see there is non-multiplicativity for p >= 0.5.")
        return true
    end
end

@testset "non-multiplicativity" begin
    @test nonMultWHID()
end

print("\nThis completes our investigation of the non-multiplicativity of Werner with the identity over PPT cone.")

#I guess we could cut these into two scripts. They were unified because they came up in discussion at the same time,
#and use the same channel

function noisyWHnonMult()
    print("\n\nOur final investigation of the Werner-Holevo channel is its non-multiplicativity's robustness to noise.")
    print("\nTo do this we just take a linear combination of the Werner-Holevo channel with the maximally")
    print("depolarizing channel. We consider d=3 and p & mixing param in [0,0.05,...,0.4]")
    print("\nNOTE: This will take a few minutes.")

    dim = 3
    p_vals = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    mix_vals = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    depol_choi = Matrix(1.0*I,dim^2,dim^2)
    len = length(p_vals); wid = length(mix_vals)
    mult_table = zeros(len,wid)
    cv_table = zeros(len,wid)
    p_ctr = 0; mix_ctr = 1;
    for p in p_vals
        print("\nNow evaluating for p = ", p)
        wern_choi = dim*wernerState(dim,p)
        p_ctr = p_ctr + 1
        for mix_amt in mix_vals
            if isapprox(mix_amt,0.2, atol = 1e-6)
                print("\nNow halfway through this value of p.")
            end
            mix_choi = (1-mix_amt)*wern_choi + mix_amt*depol_choi #This uses linearity of Choi mapping
            mix_cv = pptCVDual(mix_choi,dim,dim)

            kron_par_choi = kron(mix_choi,mix_choi)
            par_choi = permuteSubsystems(kron_par_choi,[1,3,2,4],dim*[1,1,1,1])
            par_cv = pptCVPrimal(par_choi,dim^2,dim^2)

            mult_table[p_ctr,mix_ctr] = (par_cv[1] - mix_cv[1]^2)
            cv_table[p_ctr,mix_ctr] = mix_cv[1]

            mix_ctr = mix_ctr + 1
        end
        mix_ctr = 1
    end

    data_to_save = vcat(vcat(mult_table, zeros(1,wid)),cv_table)
    print("\nPlease name the file you'd like to write the results to: \n")
    file_name = readline()
    file_to_open = string(file_name,".csv")
    writedlm(file_to_open, data_to_save, ',')
    if all(mult_table -> mult_table < 1e-4 , mult_table)
        print("\nSomething went awry. Please check your results.")
        return false
    else
        print("\nIf you check your results, you will see there is non-multiplicative regime.")
        print(" The first table is the non-multiplicativity results. The second the original cv values.")
        print(" The rows are for fixed value of p in increasing amount of mixture left to right.")
        print(" The rows are in descending value of p. \n")
        return true
    end
end


@testset "non-Multiplicativity with Noise" begin
    @test noisyWHnonMult()
end

print("\nGoodbye!")
