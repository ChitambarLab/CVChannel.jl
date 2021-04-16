using Test, SafeTestsets

include("../mosek_license.jl")

@testset "checking for MOSEK license" begin
    @test haskey(ENV,"MOSEKLM_LICENSE_FILE")
    @test isfile(ENV["MOSEKLM_LICENSE_FILE"])
end

println("importing CVChannel")
@time using CVChannel

@testset "run tests" begin

println("testing ./src/CVChannel.jl")
@time @safetestset "./test/CVChannel.jl" begin include("CVChannel.jl") end

println("testing ./src/optimizer_interface.jl")
@time @safetestset "./test/optimizer_interface.jl" begin include("optimizer_interface.jl") end

end
