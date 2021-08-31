using Test, SafeTestsets

println("importing CVChannel")
@time using CVChannel

println("running ./test/CVChannel.jl")
@time begin # timing block

@testset "running ./test/runtests.jl" begin

    println("testing ./src/operations.jl")
    @time @safetestset "./test/operations.jl" begin include("operations.jl") end

    println("testing ./src/channels.jl")
    @time @safetestset "./test/channels.jl" begin include("channels.jl") end

    println("testing ./states.jl")
    @time @safetestset "./test/states.jl" begin include("states.jl") end

    println("testing ./src/optimizations.jl")
    @time @safetestset "./test/optimizations.jl" begin include("optimizations.jl") end

    println("testing ./src/optimizer_interface.jl")
    @time @safetestset "./test/optimizer_interface.jl" begin include("optimizer_interface.jl") end

    println("testing ./src/see-saw_optimization.jl")
    @time @safetestset "./test/see-saw_optimization.jl" begin include("see-saw_optimization.jl") end
end

println("\ntotal elapsed time :")
end # timing block
