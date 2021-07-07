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

"""
    qsolve!( problem :: Problem; qsolve_kwargs..., cvx_kwargs... )

A wrapper for `Convex.solve!(...)` that selects and configures the backend.
The supported backends are SCS (default) and MOSEK (license required).
The name `qsolve!` is chosen to avoid a namespace conflict with the `Convex.solve!`.
This method is a featured utility for simplifying the interface with optimization
software.

`qsolve_kargs`:
* `quiet :: Bool = true` - If true, solve messages are suppressed.
* `use_mosek :: Bool = false` - If true, MOSEK is used instead of SCS (default).

The `cvx_kwargs...` are keyword arguments passed to `Convex.solve!`:
* `check_vexity :: Bool = true`
* `verbose :: Bool = true`
* `warmstart :: Bool = false`
* `silent_solver :: Bool = true`

For details, see [Convex.jl docs](https://jump.dev/Convex.jl/stable/reference/#Convex.solve!).
"""
function qsolve!(problem::Problem; quiet::Bool=true, use_mosek::Bool=false, cvx_kwargs...)
    optimizer = (use_mosek && hasMOSEKLicense()) ? Mosek.Optimizer(QUIET=quiet) : SCS.Optimizer(verbose=!quiet,eps=1e-6)
    solve!(problem, () -> optimizer; cvx_kwargs... )
end
