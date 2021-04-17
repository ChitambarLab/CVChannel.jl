```@meta
CurrentModule = CVChannel
```

# Optimizer Interface

The CVChannel package supports SCS (default) and MOSEK backends for optimizations.
SCS is open-source whereas MOSEK requires a license (free for academic institutions).
To obtain a MOSEK license, please visit [https://www.mosek.com/license/request/?i=acp](https://www.mosek.com/license/request/?i=acp).
For convenience, all optimization problems can be run through the [`qsolve!`](@ref) method
which abstracts the backend from the optimization.

```@docs
qsolve!
useSCS
useMOSEK
hasMOSEKLicense
```
