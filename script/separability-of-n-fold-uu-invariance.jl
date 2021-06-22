using CVChannel
using LinearAlgebra
using Convex
using Test

"""
Here we look at the separability of the PPT space of the state space
defined by U_1 U_1 ⊗ V_2 V_2 ⊗ ... ⊗ W_n W_n where each set of unitaries
U_i U_i acts on Hilbert spaces H_{A_{i}}H_{B_{i}} and the dimension of each
Hilbert space is the same. This is just a generalization of Example 7 of
https://arxiv.org/abs/quant-ph/0010095
"""

println("For each space Alice-Bob space, the extreme points of the state space
        are given by the extreme points of the Werner-Holevo Channels:")
println("Π0 = (d(d+1))^(-1)(I+F)    Π1 = (d(d-1))^(-1)(I-F)")
d = 3
i_mat = Matrix(1.0I, d^2, d^2)
f_mat = swapOperator(d)
phi_p = partialtranspose(f_mat,2,[d,d])
ext_points = [
              (1/(d*(d+1)))*(i_mat + f_mat),
              (1/(d*(d-1)))*(i_mat - f_mat)
              ]

println("Therefore the space can be parameterized by expectations of tensor
        products of I and F")
n = 3
observables = []
for i in [1:2^n-1;] #We don't need the all identities expectation
  bit_string = digits(Int8, i, base=2, pad=n) |> reverse
  op = (f_mat)^(bit_string[1])
  for j in [2:n;]
    op = kron(op, (f_mat)^(bit_string[j]))
  end
  push!(observables, op)
end

opPPT = [kron(phi_p,i_mat),kron(i_mat,phi_p),kron(phi_p,phi_p)]
tr(kron(ext_points[1],ext_points[1])*opPPT[1])
tr(kron(ext_points[1],ext_points[1])*opPPT[2])
tr(kron(ext_points[1],ext_points[1])*opPPT[3])

tr(kron(ext_points[1],ext_points[2])*opPPT[1])
tr(kron(ext_points[1],ext_points[2])*opPPT[2])
tr(kron(ext_points[1],ext_points[2])*opPPT[3])

tr(kron(ext_points[2],ext_points[1])*opPPT[1])
tr(kron(ext_points[2],ext_points[1])*opPPT[2])
tr(kron(ext_points[2],ext_points[1])*opPPT[3])

tr(kron(ext_points[2],ext_points[2])*opPPT[1])
tr(kron(ext_points[2],ext_points[2])*opPPT[2])
tr(kron(ext_points[2],ext_points[2])*opPPT[3])

tr(kron(i_mat,i_mat)*kron(f_mat,i_mat))
tr(kron(i_mat,f_mat)*kron(f_mat,i_mat))
tr(kron(f_mat,i_mat)*kron(f_mat,i_mat))
tr(kron(f_mat,f_mat)*kron(f_mat,i_mat))
tr(f_mat)
tr(i_mat)
