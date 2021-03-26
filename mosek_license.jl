# Setting the environment to point to the local mosek license.
# This script must be included before `using CVChannel`
ENV["MOSEKLM_LICENSE_FILE"] = joinpath(@__DIR__,"mosek.lic")
