using Test, SafeTestsets

println("importing CVChannel")
@time using CVChannel

@time @safetestset "./test/CVChannel.jl" begin include("CVChannel.jl") end
