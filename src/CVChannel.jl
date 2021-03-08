using Convex: partialtranspose, eigmin

"""
This function returns true if the input state x is PPT
with respect to the (sys)th system. False otherwise.
dims is a vector of the sizes of the subsystems.
"""

function isPPT(x,sys::Int,dims::Vector)
    #We don't make sure the inputs are good because the partialtranspose function
    #will do that for us

    #Get PPT of x
    xPPT = partialtranspose(x,sys,dims)

    #Check if it is PSD
    if eigmin(xPPT) >= 0 || isapprox(0,eigmin(xPPT))
        return true
    else
        return false
    end
end
