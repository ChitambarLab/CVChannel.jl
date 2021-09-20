using CVChannel
using Test

println("This function verifies the 3-sym CV SDP works.")
println("It does this by making sure it gets the right answer in cases that we don't need 3sym to be exact.")
println("It is in its own script to not slow down any other script, since it takes a while.")
@testset "3SymCV Works" begin
    println("Tests starting.")
    test1 = 3*wernerState(3,0)
    cv1 = pptCVPrimal(test1,3,3)
    cv1_3_sym = threeSymCVPrimal(test1,3,3)
    println("Test 1 done.")
    @test isapprox(cv1[1],cv1_3_sym[1], atol=1e-6)
    test2 = vec([1 0 0 ; 0 1 0 ; 0 0 1])*vec([1 0 0 ; 0 1 0 ; 0 0 1])'
    cv2 = pptCVPrimal(test2,3,3)
    cv2_3_sym = threeSymCVPrimal(test2,3,3)
    @test isapprox(cv2[1],cv2_3_sym[1], atol=2e-6)
end
