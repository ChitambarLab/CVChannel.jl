using Test
using CVChannel

"""
This script verifies that the entanglement-assisted communication value for
axisymmetric channels is bound as `eaCV(J_N) ≤ d * CV(J_N)`, where `d` is the
Hilbert space dimension of the channel's output. PPT axisymmetric states are also
separable, therefore, we can compute the communication value with `pptCVPrimal`.
"""

"""
    _verify_eaCV_bound_for_axisymmetric_channels(
        d::Int64,
        x_step::Float64,
        y_step::Float64
    )

Scans across the `d`-dimensional axisymmetric channel parameters evaluating both
the entanglement-assisted CV (`eaCVPrimal`) and the CV (`pptCVPrimal`).
The two axisymmetric parameters x and y are scanned across with step-sizes `x_step`
and `y_step` respectively.
For each channel, it is verified that the entanglement-assisted CV is bound by 
`eaCV ≦ d * CV`.
"""
function _verify_eaCV_bound_for_axisymmetric_channels(d::Int64, x_step::Float64, y_step::Float64)
    y_bounds = CVChannel._axisymmetric_y_bounds(d)

    it = 0
    y_range = y_bounds[1]:y_step:y_bounds[2]
    for y in y_range
        it = it + 1
        println("(d = $d) verifying y-slice $it of $(length(y_range))")
        x_constraints = CVChannel._axisymmetric_x_constraints(d,y)
        for x in x_constraints[1]:x_step:x_constraints[2]
            @testset "(x,y) = ($x,$y)" begin
                ρ_axi = axisymmetricState(d,x,y)

                J_N = d*ρ_axi
                (ea_cv_N, ea_σAB_N) = eaCVPrimal(J_N, d, d)
                (ppt_cv_N, ppt_σAB_N) = pptCVPrimal(J_N, d, d)

                @test ea_cv_N ≤ d * ppt_cv_N || ea_cv_N ≈ d * ppt_cv_N
            end
        end
    end
end

println("Verifying bounds on entanglement-assisted CV")
@testset "Coarse-grained verification of entanglement-assisted CV bounds (d = $d)" for d in 2:8
    y_step = 0.1
    x_step = 0.2

    _verify_eaCV_bound_for_axisymmetric_channels(d,x_step,y_step)
end

@testset "Fine-grained verification of entanglement-assisted CV bounds (d = $d)" for d in 2:4
    y_step = 0.01
    x_step = 0.01

    _verify_eaCV_bound_for_axisymmetric_channels(d,x_step,y_step)
end
