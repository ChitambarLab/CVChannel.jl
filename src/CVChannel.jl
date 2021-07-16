"""
Both quantum and classical channels can be used for classical communication, however,
there are subtle differences in their respective performance.
This software package seeks to compare classical and quantum channels using the
communication value as a device-independent quantifier of channel performance.
The *communication value* ``(\\text{cv})`` of a classical channel ``\\mathbf{P} : [n] \\to [n']``
with transition probabilities ``P(y|x)`` is defined as:
```math
\\text{cv}(\\mathbf{P}) = \\sum_{y\\in[n]'} \\max_{x\\in[n]} P(y|x).
```
For a quantum channel ``\\mathcal{N} \\in \\text{CPTP}(A\\to B)``, the ``[n]\\to [n']``
communication value is defined as:
```math
\\text{cv}^{n \\to n'}(\\mathcal{N})=
    \\max_{\\{\\Pi_y\\}_{y=1}^{n'}, \\{\\rho_x \\}_{x=1}^n}
    \\{\\text{cv}(\\mathbf{P}) \\; | \\; P(y|x) = \\text{Tr}[\\Pi_y\\mathcal{N}(\\rho_x)] \\}.
```
*Features:*
* Tools for evaluating the communication value of quantum and classical channels.
* A numerical analysis of the super-multiplicativity of the communction value of
    quantum channels.
"""
module CVChannel

using Convex, SCS, MosekTools
using LinearAlgebra

export isPPT, swapOperator, permuteSubsystems, shiftOperator, discreteWeylOperator
include("operations.jl")

export wernerState, axisymmetricState
include("states.jl")

export choi, depolarizingChannel, dephrasureChannel, wernerHolevoChannel
include("channels.jl")

export qsolve!, hasMOSEKLicense
include("optimizer_interface.jl")

export eaCVPrimal, eaCVDual, pptCVPrimal, pptCVDual, pptMultiplicativity
export WHIDLP, generalWHLPConstraints, wernerHolevoCVPPT
export twoSymCVPrimal, threeSymCVPrimal
include("optimizations.jl")

end
