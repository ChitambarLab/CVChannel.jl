using LinearAlgebra
using CVChannel
using Convex
using Test

#todo: more in depth doc string
#This script looks at the werner-holevo channels interaction with noise, non-PPT multiplicativity, etc.
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

println("\nHere we initialize the identity channel and the solver.")
identChan(X) = X
eaCVDual(choi(identChan,2,2),2,2)

println("\nNow we look at tensoring the antisymmetric projector with the identity\n")
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

#This is Eric's cv_PPT linear program
#Pick your dimensions. d is dim of Werner-Holevo, dp is dim of idenity
function analytic_soln(d,dp,λ)
    grad_neg = false
    if (-2+dp+d*dp)/(-2+2dp+d*dp) < 0
        grad_neg = true
    end
    if grad_neg
        print("here")
        return (2*d*dp*(1-λ))/(1+d)
    elseif d >= dp
        return (2(d*(1-λ)-dp*(1-2λ)+d*dp*λ))/((1+λ)*(d*dp+d-2))
    else
        return (dp*(d*(1-2λ)-dp+(2*(1-dp)*λ)))/(d*dp+d-2)
    end
end

table_vals = zeros(14,14,11)
λ_vals = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
λ_ctr = 1
for λ in λ_vals
    for d1 = 2:15
        for d2 = 2:15
            #This is the vector of variables and we use alphabetical order x[1] = w, x[2] = x,...
            v = Variable(4)
            #Given the number of constraints we define the problem and add constraints
            objective = d1*d2*(v[1]+v[3]*d2+(1-2λ)*(v[2]+v[4]*d2))
            problem1 = maximize(objective)
            problem1.constraints += [0 <= v[1]-v[2]+d2*v[3]-d2*v[4],
                            0 <= v[1]-v[2],
                            0 <= v[1]+v[2]+d2*v[3]+d2*v[4],
                            0 <= v[1]+v[2],
                            0 <= v[1] + d1*v[2] - v[3] - d1*v[4],
                            0 <= v[1] - v[3],
                            0 <= v[1] + d1*v[2] + v[3] + d1*v[4],
                            0 <= v[1] + v[3],
                            1 == d1*d2*v[1]+d2*v[2]+d1*v[3] + v[4]
                            ]
                            qsolve!(problem1)
            table_vals[d1-1,d2-1,λ_ctr] = problem1.optval
        end
    end
    λ_ctr = λ_ctr + 1
end
print(v.value)

d1 = 3
d2 = 10
λ = 0.51
objective = d1*d2*(v[1]+v[3]*d2+(1-2λ)*(v[2]+v[4]*d2))
problem1 = maximize(objective)
problem1.constraints += [0 <= v[1]-v[2]+d2*v[3]-d2*v[4],
                0 <= v[1]-v[2],
                0 <= v[1]+v[2]+d2*v[3]+d2*v[4],
                0 <= v[1]+v[2],
                0 <= v[1] + d1*v[2] - v[3] - d1*v[4],
                0 <= v[1] - v[3],
                0 <= v[1] + d1*v[2] + v[3] + d1*v[4],
                0 <= v[1] + v[3],
                1 == d1*d2*v[1]+d2*v[2]+d1*v[3] + v[4]
                ]
qsolve!(problem1)
problem1.optval

wern_dim = 3
ident_dim = 10
wern_choi = wern_dim*wernerState(wern_dim,1)
wern_cv = pptCVDual(wern_choi,wern_dim,wern_dim)
ident_choi = choi(identChan,ident_dim,ident_dim)
ident_cv = ident_dim

kron_par_choi = kron(wern_choi,ident_choi)
par_choi = permuteSubsystems(kron_par_choi,[1,3,2,4],[wern_dim,wern_dim,ident_dim,ident_dim])
par_cv = pptCVPrimal(par_choi,wern_dim*ident_dim,wern_dim*ident_dim)
