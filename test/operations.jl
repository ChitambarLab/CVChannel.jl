using Test, LinearAlgebra
using CVChannel

@testset "./src/operations.jl" begin

@testset "isPPT" begin
    @test isPPT( Matrix(I,4,4), 2, [2,2] )
    @test !isPPT( [1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1], 2, [2,2] )
end

@testset "swapOperator" begin
    @test isapprox(swapOperator(2),[1 0 0 0 ; 0 0 1 0 ; 0 1 0 0 ; 0 0 0 1], atol = 1e-6)
    swap3 = [
        1 0 0 0 0 0 0 0 0 ;
        0 0 0 1 0 0 0 0 0 ;
        0 0 0 0 0 0 1 0 0 ;
        0 1 0 0 0 0 0 0 0 ;
        0 0 0 0 1 0 0 0 0 ;
        0 0 0 0 0 0 0 1 0 ;
        0 0 1 0 0 0 0 0 0 ;
        0 0 0 0 0 1 0 0 0 ;
        0 0 0 0 0 0 0 0 1
        ]
    @test isapprox(swapOperator(3),swap3, atol =1e-6)
end

@testset "permuteSubsystems" begin
    ρ = [1:1:4;];
    @test permuteSubsystems(ρ,[2,1],[2,2]) == [1;3;2;4]
    ρ = [1:1:8;]; target = [1;3;5;7;2;4;6;8];
    @test permuteSubsystems(ρ,[3,1,2],[2,2,2])==target
    ρ = copy(reshape([1:1:16;],(4,4))');
    target = [ 1 3 2 4 ; 9 11 10 12 ; 5 7 6 8; 13 15 14 16];
    @test permuteSubsystems(ρ,[2,1],[2,2])==target
    ρ = copy(reshape([1:1:64;],(8,8))')
    target = [
        1 5 2 6 3 7 4 8;
        33 37 34 38 35 39 36 40;
        9 13 10 14 11 15 12 16;
        41 45 42 46 43 47 44 48;
        17 21 18 22 19 23 20 24;
        49 53 50 54 51 55 52 56;
        25 29 26 30 27 31 28 32;
        57 61 58 62 59 63 60 64;
    ]
    @test permuteSubsystems(ρ,[2,3,1],[2,2,2])==target
    ρ = copy(reshape([1:1:144;],(12,12))')
    target = [
    1    3    5    7    9   11    2    4    6    8   10   12
    25   27   29   31   33   35   26   28   30   32   34   36
    49   51   53   55   57   59   50   52   54   56   58   60
    73   75   77   79   81   83   74   76   78   80   82   84
    97   99  101  103  105  107   98  100  102  104  106  108
    121  123  125  127  129  131  122  124  126  128  130  132
    13   15   17   19   21   23   14   16   18   20   22   24
    37   39   41   43   45   47   38   40   42   44   46   48
    61   63   65   67   69   71   62   64   66   68   70   72
    85   87   89   91   93   95   86   88   90   92   94   96
    109  111  113  115  117  119  110  112  114  116  118  120
    133  135  137  139  141  143  134  136  138  140  142  144
    ]
    @test permuteSubsystems(ρ,[3,1,2],[2,3,2]) == target
    @test_throws DomainError permuteSubsystems([1 2 3; 4 5 6], [2,3,1],[1,2,3])
end

end
