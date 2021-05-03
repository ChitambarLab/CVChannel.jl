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
#Assumes the input is of ordering (t1,t2,t3,λ1,λ2,λ3)
#See Theorem 4 of https://arxiv.org/abs/1306.0495
function genFAConditons(v :: Vector) :: Bool
    t_vec = v[1:3]; λ_vec = v[4:6];
    t = norm(t_vec); u = t_vec / t;
    q0 = (1 + sum(λ_vec))/4;
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
    elseif t^2 - r + sqrt(r^2 - q) > 0
        return false
    else
        return true
    end
end

#This function converts the parameters into the Choi matrix
#Assumes the input is of ordering (t1,t2,t3,λ1,λ2,λ3)
function tMatrixChoi((v :: Vector) :: Matrix{<:Number}
end

println("To speed up the search we first see the range of elements")
AcceptableParam = [0 0 0 0 0 0]
stepSizes = 0.5*[1,1,1,1,1,1]
minVals = -5*[1,1,1,1,1,1]
maxVals = 5*[1,1,1,1,1,1]
for it1 in minVals[1]:stepSizes[1]:maxVals[1]
    for it2 in minVals[2]:stepSizes[2]:maxVals[2]
        for it3 in minVals[3]:stepSizes[3]:maxVals[3]
            for it4 in minVals[4]:stepSizes[4]:maxVals[4]
                for it5 in minVals[5]:stepSizes[5]:maxVals[5]
                    for it6 in minVals[6]:stepSizes[6]:maxVals[6]
                        x = [it1 it2 it3 it4 it5 it6]
                        if genFAConditons(vec(x))
                            AcceptableParam = [AcceptableParam ; x]
                        end
                    end
                end
            end
        end
    end
end

maxes = [findmax(AcceptableParam[:,1])
         findmax(AcceptableParam[:,2])
         findmax(AcceptableParam[:,3])
         findmax(AcceptableParam[:,4])
         findmax(AcceptableParam[:,5])
         findmax(AcceptableParam[:,6])
        ]

mins = [findmin(AcceptableParam[:,1])
         findmin(AcceptableParam[:,2])
         findmin(AcceptableParam[:,3])
         findmin(AcceptableParam[:,4])
         findmin(AcceptableParam[:,5])
         findmin(AcceptableParam[:,6])
        ]
