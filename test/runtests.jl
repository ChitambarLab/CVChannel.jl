using Test, SafeTestsets

include("../mosek_license.jl")

@testset "checking for MOSEK license" begin
    @test haskey(ENV,"MOSEKLM_LICENSE_FILE")
    @test isfile(ENV["MOSEKLM_LICENSE_FILE"])
end

println("importing CVChannel")
@time using CVChannel

@time @safetestset "./test/CVChannel.jl" begin include("CVChannel.jl") end
