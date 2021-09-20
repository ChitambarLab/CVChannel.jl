"""
*A numerics library for evaluating the communication value of a quantum channel.*

The communication value (CV) quantifies the performance of single-copy classical
communication.

# Features:
* Convex optimization methods for certifying the communication value of a quantum channel.
* Tools for certifying the non-multiplicativity of the communication value for
  quantum channels.

# Contents:

* A formal introduction to the communication value is found in the
  [CV Background](@ref) section.
* Documentation for our optimization methods is found in the
  [CV Optimizations](@ref) section.
* Documentation for our multiplicativity analysis tools are found in the
  [CV Multiplicativity](@ref) section.
* Supporting methods are found in the Utilities section.
"""
module CVChannel

using Convex, SCS, MosekTools
using LinearAlgebra
using QBase

import Base: show

export isPPT, swapOperator, permuteSubsystems, shiftOperator
export discreteWeylOperator
include("operations.jl")

export wernerState, axisymmetricState, haarStates
include("states.jl")

export choi, is_choi_matrix, Choi, parChoi
export isometricChannel, complementaryChannel, krausAction
export depolarizingChannel, dephrasureChannel
export siddhuChannel, generalizedSiddhu
export wernerHolevoChannel, GADChannel
include("channels.jl")

export qsolve!, hasMOSEKLicense
include("optimizer_interface.jl")

export eaCV, eaCVPrimal, eaCVDual
export pptCV, pptCVPrimal, pptCVDual, pptCVMultiplicativity
export WHIDLP, generalWHLPConstraints, wernerHolevoCVPPT
export twoSymCVPrimal, threeSymCVPrimal
include("optimizations.jl")

export fixedStateCV, fixedMeasurementCV, seesawCV
include("see-saw_optimization.jl")

end
