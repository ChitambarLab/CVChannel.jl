using LinearAlgebra
using CVChannel
using Test
using DelimitedFiles


"""
A qubit to qubit channel can be parameterized (up to a unitary)
by 6 parameters (t1,t2,t3,λ1,λ2,λ3). It is only non-unital if
it satisfies certain properties (Generalized Fujiwara-Algoet conditions).
These parameters may be shown all have absolute value bounded by 1.

In this script, we scan over the non-unital qubit to qubit channels
using the generalized FA conditions and check for multiplicativity for
such channels. You can hardcode the range you go over. You choose the
stepsize during the run. All data from the run is saved to a csv file
that you choose. The code tells you if non-multiplicativity was detected
regardless, in case you don't care about the specific data.
"""

#This function checks if the generalized Fujiwara-Algoet conditions hold.
#Assumes the input is of ordering (t1,t2,t3,λ1,λ2,λ3) and that it is not the
#case that t1=t2=t3=0 (i.e. it needs to be non-unital)
#See Theorem 4 of https://arxiv.org/abs/1306.0495
function genFAConditons(t_vec :: Vector, λ_vec :: Vector) :: Bool
    t = norm(t_vec); u = t_vec / t;

    q0 = (1 + λ_vec[1] + λ_vec[2] + λ_vec[3])/4;
    q1 = (1 + λ_vec[1] - λ_vec[2] - λ_vec[3])/4;
    q2 = (1 - λ_vec[1] + λ_vec[2] - λ_vec[3])/4;
    q3 = (1 - λ_vec[1] - λ_vec[2] - λ_vec[3])/4;
    q_vec = [q0,q1,q2,q3];
    q = 256 * prod(q_vec);                          #Eqn. (32)
    r = 1 - sum(λ_vec.^2) + 2*sum((λ_vec .* u).^2)  #Eqn. (31)

    if findfirst(x -> x < 0, q_vec) !== nothing
        return false
    elseif r^2 - q  < 0
        return false
    elseif t^2 - r + sqrt(r^2 - q) > 0              #Eqn. (30)
        return false
    else
        return true
    end
end

print("\nFirst we verify our function for generating the Choi matrix from the")
print(" 6 parameters does what it ought.")
#This function converts the parameters into the Choi matrix    #Eqn. (27)
#Assumes the input is of ordering (t1,t2,t3,λ1,λ2,λ3)
function tMatrixChoi(t :: Vector, λ :: Vector) :: Matrix{<:Number}
    choi_matrix = [
        0.5*(1+λ[3]+t[3])    0                   0.5*(t[1]+1im*t[2]) 0.5*(λ[1]+λ[2])
        0                    0.5*(1-λ[3]+t[3])   0.5*(λ[1]-λ[2])     0.5*(t[1]+1im*t[2])
        1/2*(t[1]-1im*t[2])  0.5*(λ[1]-λ[2])     0.5*(1-λ[3]-t[3])   0
        1/2*(λ[1]+λ[2])      0.5*(t[1]-1im*t[2]) 0                   1/2*(1+λ[3]-t[3])
    ]
    return choi_matrix
end

@testset "Verifying tMatrixChoi" begin
    t = [0,0,0]; λ=[0,0,0];
    target = [1/2 0 0 0 ; 0 1/2 0 0  ; 0 0 1/2 0 ; 0 0 0 1/2]
    @test tMatrixChoi(t,λ) == target

    t = [0,0,0]; λ=[1/3,1/4,1/2];
    target = [3/4 0 0 7/24 ; 0 1/4 1/24 0 ; 0 1/24 1/4 0 ; 7/24 0 0 3/4]
    @test isapprox(tMatrixChoi(t,λ),target, atol= 1e-9)

    t = [1/4,1/3,1/2]; λ=[0,0,0];
    target = [3/4 0 0.5*(1/4+1im*1/3) 0 ;
            0 3/4 0 0.5*(1/4+1im*1/3) ;
            0.5*(1/4-1im*1/3) 0 1/4 0 ;
            0 0.5*(1/4-1im*1/3) 0 1/4 ]
    @test isapprox(tMatrixChoi(t,λ),target, atol=1e-9)

    t = [1/4,1/3,1/2]; λ=[1/3,1/4,1/2];
    target = [ 1 0 1/8+1im/6 7/24;
               0 1/2 1/24 1/8+1im/6;
               1/8-1im/6 1/24 0 0;
               7/24 1/8-1im/6 0 1/2]
    @test isapprox(tMatrixChoi(t,λ),target,atol=1e-9)
end

#Here you set the range you want to run over
function stepSize() :: Float64
    print("\nNow one decides how to scan over the space.")
    print("\nIf you want to alter the step-size from non-uniform, you must hard-code it in the script. \n")
    print("\nPlease enter your choice of stepsize in (10^(-5),1): \n")
    acceptable_input = false
    while !acceptable_input
        step_size = parse(Float64,readline())
        if step_size ≥ 1 || step_size ≤ 1e-5
            print("\nPlease enter a number in (10^(-5),1): ")
        else
            return step_size
        end
    end
end

step_size = stepSize()

step_sizes = step_size*[1,1,1,1,1,1]
#|λ_i | ≤ 1 for all i ; See between (36) and (37)
#From this, it's easy to at least numerically see that |t_i| ≤ 1 ∀i
min_vals = [-1,-1,-1,-1,-1,-1]
max_vals = [1,1,1,1,1,1]

function verifyScanningRange(step_sizes :: Vector, min_vals :: Vector, max_vals :: Vector) :: Bool
    print("\n\n To be safe, we check how many points we'll need to evaluate.")
    total_num_of_scan_points = 1;
    for i in 1:6
        total_num_of_scan_points = total_num_of_scan_points * length(min_vals[i]:step_sizes[i]:max_vals[i])
    end

    print("\n Warning: if you check this range for evaluable_points, you will check ", total_num_of_scan_points," points. Do you wish to continue? (y/n) \n")
    proper_response = false
    while !proper_response
        user_response = readline()
        if user_response=="y"
            return true
        elseif user_response == "n"
            print("\n See you later!")
            return false
        else
            print("Please respond with `y` or `n`. \n ")
        end
    end
end

want_to_scan_range = verifyScanningRange(step_sizes, min_vals, max_vals)

if want_to_scan_range
    function numOfEvaluations(step_sizes :: Vector, min_vals :: Vector ,max_vals ::Vector) :: Matrix{Float16}
        #Here we check how many points you'll actually be running the SDP for
        #if you check this range. We also return what those parameters are
        nonUnitalParams = Float16[0 0 0 0 0 0]
        for it1 in min_vals[1]:step_sizes[1]:max_vals[1]
            for it2 in min_vals[2]:step_sizes[2]:max_vals[2]
                for it3 in min_vals[3]:step_sizes[3]:max_vals[3]
                    for it4 in min_vals[4]:step_sizes[4]:max_vals[4]
                        for it5 in min_vals[5]:step_sizes[5]:max_vals[5]
                            for it6 in min_vals[6]:step_sizes[6]:max_vals[6]
                                t = [it1,it2,it3];
                                λ = [it4,it5,it6]
                                if genFAConditons(t,λ) && !(it1==it2==it3==0)
                                    x = hcat(t,λ)
                                    newRow = copy(transpose(convert(Matrix{Float16},reshape(x,6,1))))
                                    nonUnitalParams = vcat(nonUnitalParams,newRow)
                                end
                            end
                        end
                    end
                end
            end
        end
        nonUnitalParams = nonUnitalParams[setdiff(1:end, 1), :]
        return nonUnitalParams
    end

    nonUnitalParams = numOfEvaluations(step_sizes, min_vals,max_vals)
else
    "\nSee you later!"
    exit()
end

function verifyProceeding(evaluable_points :: Int64) :: Bool
    print("Warning: if you run over this range, you will check ", evaluable_points," points. Do you wish to continue? (y/n) \n")
    proper_response = false
    while !proper_response
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

evaluable_points = size(nonUnitalParams)[1]
want_to_proceed = verifyProceeding(evaluable_points)

#We write all the data to a csv, but we have the computer keep track
#of if non-multiplcativity has been detected
function scanRange(nonUnitalParams) :: Tuple{String,Bool}
    evaluable_points = size(nonUnitalParams)[1]
    diffMatrix = zeros(Float16,evaluable_points,1)
    nonMultDetected = false
    print("Please name the file you'd like to write to: \n")
    file_name = readline()
    file_to_open = string(file_name,".csv")
    evaluable_points = size(nonUnitalParams)[1]
    println("\nRunning the SDP solver for the first time. It will run faster after the first point.")
    for ctr = 1 : evaluable_points
        if (ctr % 5 == 0)
            println("\nNow evaluating evaluable point ", ctr," of ", evaluable_points,".")
        end

        t = nonUnitalParams[ctr,1:3]; λ = nonUnitalParams[ctr,4:6]
        orig_choi = tMatrixChoi(t,λ)
        test1 = pptCVDual(orig_choi,2,2)
        kron_par_choi = kron(orig_choi,orig_choi)
        par_choi = permuteSubsystems(kron_par_choi,[1,3,2,4],[2,2,2,2])
        test2 = pptCVPrimal(par_choi,4,4)
        diffMatrix[ctr] = test2[1] - test1[1]^2
        if abs(diffMatrix[ctr]) > 1e-5 && diffMatrix[ctr] > 0
            if eigmin(orig_choi) >= 0 #This should always be true, but I don't trust numerical error, so it's a final sanity check
                nonMultDetected = true
            end
        end
    end

    data_to_save = hcat(nonUnitalParams,diffMatrix)
    writedlm(file_to_open, data_to_save, ',')
    return file_to_open, nonMultDetected
end

if want_to_proceed
    @testset "Multiplicative Over Range" begin
    (file_to_open,non_mult_detected)=scanRange(nonUnitalParams)
    print("\nIf it was multiplicative over the range, the test passes.")
    @test !non_mult_detected
    end
    print("\nAll of the data from the run was saved to your file, if you'd like to look at it.")
else
    print("\nSee you later!")
end
