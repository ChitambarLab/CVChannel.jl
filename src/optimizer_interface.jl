"""
    hasMOSEKLicense() :: Bool

Returns `true` if a MOSEK license is found locally.
The `ENV` configuration is used to search for  a MOSEK license in the following order:
* `ENV[MOSEKLMM_LICENSE_FILE]` a direct filepath to the MOSEK license.
* Filepath `ENV["HOME"]/mosek/mosek.lic`
* Filepath `ENV["PROFILE"]/mosek/mosek.lic`

!!! warning "Expired MOSEK License"
    The method `hasMOSEKLicense` will return `true` if an expired MOSEK license
    is found. An error will be thrown if MOSEK is used with an expired license.
    To obtain a MOSEK license please visit [https://www.mosek.com/license/request/?i=acp](https://www.mosek.com/license/request/?i=acp).
"""
function hasMOSEKLicense() :: Bool
    has_license = false
    if haskey(ENV,"MOSEKLM_LICENSE_FILE") && isfile(ENV["MOSEKLM_LICENSE_FILE"])
        has_license = true
    elseif haskey(ENV,"HOME") && isfile(joinpath(ENV["HOME"],"mosek","mosek.lic"))
        has_license = true
    elseif haskey(ENV,"PROFILE") && isfile(joinpath(ENV["PROFILE"],"mosek","mosek.lic"))
        has_license = true
    end

    return has_license
end

# "private" variable for keeping track of whether to use MOSEK.
_USE_MOSEK = false

"""
    useSCS() :: Bool

Sets SCS as the active backend solver and returns the value of the
`_USE_MOSEK` state variable.
"""
function useSCS() :: Bool
    @warn "Using SCS backend."
    global _USE_MOSEK = false
end

"""
    useMOSEK() :: Bool

Sets MOSEK as the active backend solver and returns the value  of the
`_USE_MOSEK` state variable.
Note that the  [`hasMOSEKLicense`](@ref) method must return `true` to use MOSEK.
"""
function useMOSEK() :: Bool
    if hasMOSEKLicense()
        @warn "Using MOSEK backend."
        global _USE_MOSEK = true
    else
        @warn "No MOSEK license found. Using SCS backend."
        global _USE_MOSEK = false
    end
end

"""
    qsolve!( problem :: Problem; cvx_kwargs... )

A wrapper for `Convex.solve!(...)` that selects and configures the backend.
The supported backends are SCS (default) and MOSEK (license required).
The [`useSCS`](@ref) or [`useMOSEK`](@ref) methods declare the backend.

!!! warning "Mutability"
    The selected backend is mutable and determined by the internal
    variable `_USE_MOSEK`. This value should only be set by the [`useSCS`](@ref)
    or [`useMOSEK`](@ref) interfaces.

The name `qsolve!` is chosen to avoid a namespace conflict with the `Convex.solve!`.
This method is a featured utility for simplifying the interface with optimization
software.

The `cvx_kwargs...` are keyword arguments passed to `Convex.solve!`:
* `check_vexity :: Bool = true`
* `verbose :: Bool = true`
* `warmstart :: Bool = false`
* `silent_solver :: Bool = true`

For details, see [Convex.jl docs](https://jump.dev/Convex.jl/stable/reference/#Convex.solve!).
"""
function qsolve!(problem::Problem; quiet::Bool=true, cvx_kwargs...)
    optimizer = _USE_MOSEK ? Mosek.Optimizer(QUIET=quiet) : SCS.Optimizer(verbose=!quiet,eps=1e-6)
    solve!(problem, () -> optimizer; cvx_kwargs... )
end
