using LinearAlgebra
using CVChannel
using Convex
using Test


"""
A qubit to qubit channel can be parameterized (up to a unitary)
by 6 parameters (t1,t2,t3,λ1,λ2,λ3). It is only non-unital if
it satisfies certain properties (Generalized Fujiwara-Algoet conditions).
"""

#This function checks if the generalized Fujiwara-Algoet conditions hold.
#Assumes the input is of ordering (t1,t2,t3,λ1,λ2,λ3) and that it is not the
#case that t1=t2=t3=0 (i.e. it needs to be non-unital)
#See Theorem 4 of https://arxiv.org/abs/1306.0495
function genFAConditons(v :: Vector) :: Bool
    t_vec = v[1:3]; λ_vec = v[4:6];
    t = norm(t_vec); u = t_vec / t;

    q0 = (1 + λ_vec[1] + λ_vec[2] + λ_vec[3])/4;
    q1 = (1 + λ_vec[1] - λ_vec[2] - λ_vec[3])/4;
    q2 = (1 - λ_vec[1] + λ_vec[2] - λ_vec[3])/4;
    q3 = (1 - λ_vec[1] - λ_vec[2] - λ_vec[3])/4;
    q_vec = [q0,q1,q2,q3];
    q = 256 * prod(q_vec);                          #Eqn. (32)
    r = 1 - sum(λ_vec.^2) + 2*sum((λ_vec .* u).^2)  #Eqn. (31)

    for i in 1:4
        if q_vec[i] < 0
            return false
        end
    end
    if r^2 - q  < 0
        return false
    elseif t^2 - r + sqrt(r^2 - q) > 0              #Eqn. (30)
        return false
    else
        return true
    end
end

print("\nFirst we verify our function for generating the Choi matrix from the")
print(" 6 parameters does what it ought.")
#This function converts the parameters into the Choi matrix
#Assumes the input is of ordering (t1,t2,t3,λ1,λ2,λ3)
function tMatrixChoi(v :: Vector) :: Matrix{<:Number}
    t = v[1:3]; λ = v[4:6];
    choi_matrix = [
        0.5*(1+λ[3]+t[3])    0                   0.5*(t[1]+1im*t[2]) 0.5*(λ[1]+λ[2])
        0                    0.5*(1-λ[3]+t[3])   0.5*(λ[1]-λ[2])     0.5*(t[1]+1im*t[2])
        1/2*(t[1]-1im*t[2])  0.5*(λ[1]-λ[2])     0.5*(1-λ[3]-t[3])   0
        1/2*(λ[1]+λ[2])      0.5*(t[1]-1im*t[2]) 0                   1/2*(1+λ[3]-t[3])
    ]
    return choi_matrix
end

@testset "Verifying tMatrixChoi" begin
    x = [0,0,0,0,0,0]
    target = [1/2 0 0 0 ; 0 1/2 0 0  ; 0 0 1/2 0 ; 0 0 0 1/2]
    @test tMatrixChoi(x) == target

    x = [0,0,0,1/3,1/4,1/2]
    target = [3/4 0 0 7/24 ; 0 1/4 1/24 0 ; 0 1/24 1/4 0 ; 7/24 0 0 3/4]
    @test isapprox(tMatrixChoi(x),target, atol= 1e-9)

    x = [1/4,1/3,1/2,0,0,0]
    target = [3/4 0 0.5*(1/4+1im*1/3) 0 ;
            0 3/4 0 0.5*(1/4+1im*1/3) ;
            0.5*(1/4-1im*1/3) 0 1/4 0 ;
            0 0.5*(1/4-1im*1/3) 0 1/4 ]
    @test isapprox(tMatrixChoi(x),target, atol=1e-9)

    x = [1/4,1/3,1/2,1/3,1/4,1/2]
    target = [ 1 0 1/8+1im/6 7/24;
               0 1/2 1/24 1/8+1im/6;
               1/8-1im/6 1/24 0 0;
               7/24 1/8-1im/6 0 1/2]
    @test isapprox(tMatrixChoi(x),target,atol=1e-9)
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
#From this, it's easy to at least numerically check that |t_i| ≤ 1 ∀i
min_vals = [-1,-1,-1,-1,-1,-1]
max_vals = [1,1,1,1,1,1]


function verifyScanningRange(step_sizes :: Vector, min_vals :: Vector, max_vals :: Vector) :: Bool
    print("\n\n To be safe, we check how many points we'll need to evaluate.")
    total_num_of_scan_points = 1;
    for i in 1:1:6
        total_num_of_scan_points = total_num_of_scan_points * length(min_vals[i]:step_sizes[i]:max_vals[i])
    end

    print("\n Warning: if you check this range for evaluatable_points, you will check ", total_num_of_scan_points," points. Do you wish to continue? (y/n) \n")
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
    function numOfEvaluations(step_sizes :: Vector, min_vals :: Vector ,max_vals ::Vector) :: Int64
        #Here we make you check you are certain about how many points you'll
        #actually be running the SDP for if you check this range
        evaluatable_points = 0;
        for it1 in min_vals[1]:step_sizes[1]:max_vals[1]
            for it2 in min_vals[2]:step_sizes[2]:max_vals[2]
                for it3 in min_vals[3]:step_sizes[3]:max_vals[3]
                    for it4 in min_vals[4]:step_sizes[4]:max_vals[4]
                        for it5 in min_vals[5]:step_sizes[5]:max_vals[5]
                            for it6 in min_vals[6]:step_sizes[6]:max_vals[6]
                                x = [it1,it2,it3,it4,it5,it6]
                                if genFAConditons(x) && !(it1==it2==it3==0)
                                    evaluatable_points += 1
                                end
                            end
                        end
                    end
                end
            end
        end
        return evaluatable_points
    end

    evaluatable_points = numOfEvaluations(step_sizes, min_vals,max_vals)
else
    "\nSee you later!"
    exit()
end

function verifyProceeding(evaluatable_points :: Int64) :: Bool
    print("Warning: if you run over this range, you will check ", evaluatable_points," points. Do you wish to continue? (y/n) \n")
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

want_to_proceed = verifyProceeding(evaluatable_points)


#Since the code is fast to run for a single point but slow to run over many,
#the theory is to save the inputs for verification rather than save the
#optimizers or anything. We're going to use a text file instead of a csv
#so that we don't need extra packages
function scanRange(step_sizes :: Vector, min_vals :: Vector ,max_vals ::Vector, evaluatable_points :: Int64) :: String
    print("Please name the file you'd like to write to: \n")
    file_name = readline()
    file_to_open = string(file_name,".txt")
    open(file_to_open, "w") do f
        write(f,"The ordering is: t1 t2 t3 λ1 λ2 λ3 \n \n")
    end
    print(evaluatable_points)
    counter = 0 #This is so people have an idea of how the run is going
    for it1 in min_vals[1]:step_sizes[1]:max_vals[1]
        for it2 in min_vals[2]:step_sizes[2]:max_vals[2]
            for it3 in min_vals[3]:step_sizes[3]:max_vals[3]
                for it4 in min_vals[4]:step_sizes[4]:max_vals[4]
                    for it5 in min_vals[5]:step_sizes[5]:max_vals[5]
                        for it6 in min_vals[6]:step_sizes[6]:max_vals[6]
                            x = [it1,it2,it3,it4,it5,it6]
                            if genFAConditons(x) && !(it1==it2==it3==0)
                                counter += 1
                                if (counter % 5 == 0)
                                    println("\nNow evaluating evaluatable point ", counter," of ", evaluatable_points,".")
                                elseif counter == 1
                                    println("\nRunning the SDP solver for the first time. It will run faster after the first point.")
                                end

                                orig_choi = tMatrixChoi(x)
                                if eigmin(orig_choi) >= 0
                                    test1 = pptCVDual(orig_choi,2,2)
                                    kron_par_choi = kron(orig_choi,orig_choi)
                                    par_choi = permuteSubsystems(kron_par_choi,[1,3,2,4],[2,2,2,2])
                                    test2 = pptCVPrimal(par_choi,4,4)
                                    if abs(test2[1] - test1[1]^2) > 1e-5 && test2[1] - test1[1]^2 > 0
                                        open(file_to_open, "w") do f
                                        write(f, "", string(x[1]), " ", string(x[2]), " ", string(x[3]), " ", string(x[4]), " ", string(x[5]), " ", string(x[6]), " \n")
                                    end
                                end
                            end
                            end
                        end
                    end
                end
            end
        end
    end

    return file_to_open
end

if want_to_proceed
    @testset "Multiplicative Over Range" begin
    file_to_open=scanRange(step_sizes,min_vals,max_vals,evaluatable_points)
    f = open(file_to_open)
    lines = readlines(f)
    close(f)
    print("\nIf there are no entries in your file, it was multiplicative over the range.")
    print("\nWe check this for you. \n")
    @test lines[2] == " "
    end
else
    print("\nSee you later!")
end
