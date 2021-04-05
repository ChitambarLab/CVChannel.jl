using LinearAlgebra
using CVChannel
using Convex

#This script looks at the werner-holevo channels
"""
First we convince ourselves that the choi states
of the WH channel are the werner states multiplied
by the dimension (This can be commented out once you
are convinced)
"""
wernerHolevoChannel_0(ρ) = wernerHolevoChannel(ρ,0)
wernerHolevoChannel_1(ρ) = wernerHolevoChannel(ρ,1)
wernerHolevoChannel_05(ρ) = wernerHolevoChannel(ρ,0.5)
test1 = choi(wernerHolevoChannel_0,3)
test2 = 3*wernerState(3,0)
test3 = choi(wernerHolevoChannel_1,3)
test4 = 3*wernerState(3,1)
test5 = 3*(0.5*wernerState(3,0)+0.5*wernerState(3,1))
test6 = choi(wernerHolevoChannel_05,3)
test1 == test2
test3 == test4
test5 == test6
#By linearity, we know it works for p between 0 and 1

"""
Given the above block, rather than call choi every time
we'll just evaluate the minEntropyPPPT on the scaled
choi states
"""

"""
First we check if the non-multiplicativity of the anti-symmetric
projector (Holevo-Channel with p=0) holds over the PPT cone for d=3
"""
origChoi = 3*wernerState(3,0)
test1 = minEntropyPPTDual(origChoi,3,3)

kronParChoi = kron(origChoi,origChoi)
swaparoo = kron(kron(Matrix{Float64}(I,3,3),swapOperator(3)),Matrix{Float64}(I,3,3))
parChoi = swaparoo*kronParChoi*swaparoo
time1 = time()
test2 = minEntropyPPTPrimal(parChoi,9,9)
time2 = time()
