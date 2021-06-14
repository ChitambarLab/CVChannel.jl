using LinearAlgebra
using CVChannel
using Convex
using Test
"""
In this script we investigate a family of co-positive maps.
Specifically, we investigate the set of states presented in
Eqn. (82) of https://arxiv.org/abs/1004.1655

Note we know co-positive maps have multiplicativity (cite Theorem/proposition
when we have a write-up, I guess), so the only way to check this at all is
implementing the DPS hierarchy.
"""

println("\nFirst we define the set of bound entangled states we will use.")
println("\nWe note they can be scaled to channels as they are bell diagonal.")

function boundBell(ε :: :: Union{Int,Float64})


println("\nAs a sanity check, we first see they are multiplicative over the PPT cone.")
