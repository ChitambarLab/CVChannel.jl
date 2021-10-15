using Convex, SCS
using LinearAlgebra
using CVChannel
using Test

"""
This script verifies that the SDP relaxation of classical-coherent
optimization is too loose to be useful. As the optimization problems themselves
aren't useful outside of this script, they are all contained within this
script. Derivations may be found in the conic approach note.
"""

println("First we declare the functions.")
function cvkPrimal(k :: Integer, kraus_ops :: Vector)
    dimB, dimA = size(kraus_ops[1])
    dimE = (length(kraus_ops) == 1) ? 2 : length(kraus_ops)
    idX = Matrix(1I,k,k)

    #Declare variables
    σ = HermitianSemidefinite(k*dimE)
    #all Hs stuck on top of eachother
    H = ComplexVariable(Int(dimA*k*(k+1)/2),dimA)
    X = ComplexVariable(k^2*dimE, k^2*dimE)
    #Begin problem
    objective = real(tr(X) + tr(X'))
    problem = maximize(objective)
    #Add constraints
    problem.constraints += [tr(_prim_map(k,H,kraus_ops)) == 1]
    problem.constraints += [tr(σ) == 1]
    #Note that _prim_map, σ is positive by this positivity constraint
    problem.constraints += [[ _prim_map(k,H,kraus_ops) X ; X' kron(idX, σ)] ⪰ 0]
    for i in [1:Int(k*(k+1)/2);] #Hermitian constraints
        problem.constraints += H[Int((i-1)*dimA+1):Int(i*dimA),1:dimA] == H[Int((i-1)*dimA+1):Int(i*dimA),1:dimA]'
    end
    #Solve the problem, return value
    solve!(problem,SCS.Optimizer(verbose=false))
    cvk = k*(problem.optval/2)^2
    return cvk, problem.optval, H.value, σ.value, X.value
end
function _prim_map(k :: Integer, H :: Union{Variable,Matrix{<:Number}}, kraus_ops::Vector)
    #This constructs the output of (id_{XX'} ⊗ 𝒩_{c})(P), which we call out_p
    dimB, dimA = size(kraus_ops[1])
    dimE = (length(kraus_ops) == 1) ? 2 : length(kraus_ops)
    comp_kraus = complementaryChannel(kraus_ops)

    out_p = zeros(k^2*dimE,k^2*dimE)
    xxp = zeros(k^2,k^2)
    for i in 1:k;
        for j in i:k;
            xxp[i^2,j^2] = 1
            #The index is by our bijection
            ind = Int((i-1)*k- (i-1)*(i-2)/2 + (j-i) + 1);
            if i == j
                out_p += kron(xxp,krausAction(comp_kraus,H[((ind-1)*dimA+1):ind*dimA,1:dimA]))
            else
                out_p += kron(xxp,krausAction(comp_kraus,H[((ind-1)*dimA+1):ind*dimA,1:dimA]))
                out_p += kron(xxp,krausAction(comp_kraus,H[((ind-1)*dimA+1):ind*dimA,1:dimA]))'
            end
            xxp[i^2,j^2] = 0
        end
    end
    return out_p / k
end

function cvkDual(k :: Integer, kraus_ops :: Vector)
    length(kraus_ops) == 1 ? dimE = 2 : dimE = length(kraus_ops)
    idXE = Matrix(1I,k*dimE,k*dimE)
    idXXE = Matrix(1I,k^2*dimE,k^2*dimE)
    #Get adjoint map of complementary channel
    comp_kraus = complementaryChannel(kraus_ops)
    adj_comp_kraus = Vector{Matrix{Complex{Float64}}}(undef, length(comp_kraus))
    for i in 1:length(comp_kraus);
        adj_comp_kraus[i] = comp_kraus[i]'
    end
    #Declare variables
    z1, z2 = Variable(), Variable()
    Ytilde = HermitianSemidefinite(k^2*dimE,k^2*dimE)
    W = HermitianSemidefinite(k^2*dimE,k^2*dimE)
    #Begin problem
    objective = z2
    problem = minimize(objective)
    #Add constraints
    problem.constraints += [[(Ytilde+z1*idXXE) -idXXE ; -idXXE W] ⪰ 0]
    problem.constraints += [z2*idXE ⪰ partialtrace(W, 1, [k,k,dimE])]
    for i in [1:k;]
        for j in [i:k;]
            problem.constraints += [_dual_map_const(Ytilde,adj_comp_kraus,i,j,k,dimE) == 0]
        end
    end
    #Solve the problem, return value
    solve!(problem,SCS.Optimizer(verbose=false))
    cvk = k*((z1.value+z2.value)/2)^2
    zs = [z1.value,z2.value]
    return cvk, problem.optval, zs, Ytilde.value, W.value
end
#Helper function
function _dual_map_const(Ytilde::Variable, adj_comp_kraus::Vector, i::Integer, j::Integer, k::Integer, dimE::Integer)
    i_vec, j_vec = zeros(k), zeros(k)
    idE = Matrix(1I,dimE,dimE)
    i_vec[i], j_vec[j] = 1,1
    i_op,j_op = kron(i_vec,i_vec,idE), kron(j_vec,j_vec,idE)
    total_op = i_op'*Ytilde*j_op - j_op'*Ytilde*i_op
    return krausAction(adj_comp_kraus,total_op)
end


@testset "SDP relaxation not useful" begin
    println("Now we show they break down quickly using the identity and replacer channels.")
    #Identity Channel
    id_kraus_chan = Vector{Matrix{Complex{Float64}}}(undef, 1)
    id_kraus_chan[1] = Matrix(1I,3,3)

    #Replacer channel
    rep_kraus_chan = Vector{Matrix{Complex{Float64}}}(undef, 3)
    rep_kraus_chan[1] = [1 0 0; 0 0 0]
    rep_kraus_chan[2] = [0 1 0; 0 0 0]
    rep_kraus_chan[3] = [0 0 1; 0 0 0]

    println("First we look at the primal problem. We see it breaks down almost immediately.")
    @testset "Primal Problem" begin
        println("Getting data")
        cv1prim_id, = cvkPrimal(1,id_kraus_chan)
        println("Primal for cv k=1 for Identity channel is: ", cv1prim_id)
        cv2prim_id, = cvkPrimal(2,id_kraus_chan)
        println("Primal for cv k=2 for Identity channel is: ", cv2prim_id)
        cv3prim_id, = cvkPrimal(3,id_kraus_chan)
        println("Primal for cv k=3 for Identity channel is: ", cv3prim_id)
        cv1prim_rep, = cvkPrimal(1,rep_kraus_chan)
        println("Primal for cv k=1 for Replacer channel is: ", cv1prim_rep)
        cv2prim_rep, = cvkPrimal(2,rep_kraus_chan)
        println("Primal for cv k=2 for Replacer channel is: ", cv2prim_rep)
        @testset "k=1,2 tight for identity channel" begin
            @test isapprox(cv1prim_id,1,atol=1e-6)
            @test isapprox(cv2prim_id,2,atol=1e-4)
        end
        @testset "k=3+ not tight for identity channel" begin
            @test !isapprox(cv3prim_id,3,atol=1e-6)
            @test isapprox(cv3prim_id,6.000182, atol=2e-4)
        end
        @testset "k=1 tight for replacer channel" begin
            @test isapprox(cv1prim_rep,1,atol=1e-6)
        end
        @testset "k=2+ not tight for replacer channel" begin
            @test !isapprox(cv2prim_rep,1,atol=1e-6)
        end
    end
    println("We now see the same issue happens with the dual problem.")
    println("However, here we see there is a huge gap of the order ~1e10.")
    println("This is probably a mix of:")
    println("1) the fidelity function not being numerically ideal in general,")
    println("2) the complicated structure we demand of the variable Y,")
    println("3) possibly a lack of strong duality.")
    @testset "Dual Problem" begin
        println("Getting data and presenting value")
        cv1dual_id, = cvkDual(1,id_kraus_chan)
        println("Dual for cv k=1 for Identity channel is: ", cv1dual_id)
        cv2dual_id, = cvkDual(2,id_kraus_chan)
        println("Dual for cv k=2 for Identity channel is: ", cv2dual_id)
        cv3dual_id, = cvkDual(3,id_kraus_chan)
        println("Dual for cv k=3 for Identity channel is: ", cv3dual_id)
        cv1dual_rep, = cvkDual(1,rep_kraus_chan)
        println("Dual for cv k=1 for Replacement channel is: ", cv1dual_rep)
        cv2dual_rep, = cvkDual(2,rep_kraus_chan)
        println("Dual for cv k=2 for Replacement channel is: ", cv2dual_rep)
        @testset "k=1,2,3 not tight for identity channel" begin
            @test !isapprox(cv1dual_id,1,atol=1e-6)
            @test !isapprox(cv2dual_id,2,atol=1e-6)
            @test !isapprox(cv3dual_id,3,atol=1e-6)
        end
        @testset "k=1,2 not tight for replacer channel" begin
            @test !isapprox(cv1dual_rep, 1, atol=1e-6)
            @test !isapprox(cv2dual_rep, 1, atol=1e-6)
        end
    end
end
