using LinearAlgebra
using CVChannel
using Convex
using Test


"""
A qubit to qubit channel can be parameterized (up to a unitary)
"""
λ = [λ1,λ2,λ3]
q0 = (1 + sum[λ])/4
q1 = (1 + λ[1] - λ[2] - λ[3])/4
q2 = (1 - λ[1] + λ[2] - λ[3])/4
q3 = (1 - λ[1] - λ[2] - λ[3])/4
q = [q0,q1,q2,q3]
