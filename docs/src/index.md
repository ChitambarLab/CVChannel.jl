```@meta
CurrentModule = CVChannel
```
# CVChannel.jl

This Julia package and numerical analysis support the findings in
[The Communication Value of a Quantum Channel](https://arxiv.org/abs/2109.11144).

```@docs
CVChannel
```

## Quick Start

1. Install Julia: [https://julialang.org/downloads/](https://julialang.org/downloads/)
2. Add the CVChannel.jl package from the Julia command prompt:

```julia
julia> using Pkg; Pkg.add("CVChannel")
```

To import the CVChannel.jl, run `using CVChannel` in the desired Julia file or
workspace.

## Numerical Analysis

The numerical analysis in this work investigates (non-)multiplicativity of the
communication value over a wide range of quantum channels.

### Scripts

This project uses scripts to verify and investigate the (non-)multiplicativity
of different quantum channels.
Scripts are found in the
[`./script`](https://github.com/ChitambarLab/CVChannel.jl/tree/main/script)
directory and are categorized into two directories as:
* [`./script/verify`](https://github.com/ChitambarLab/CVChannel.jl/tree/main/script/verify) - assert a numerical fact or result.
* [`./script/investigate`](https://github.com/ChitambarLab/CVChannel.jl/tree/main/script/investigate) - collects data for analysis.
Instructions for running scripts can be found in the [`README.md`](https://github.com/ChitambarLab/CVChannel.jl/blob/main/README.md#scripts)

### Notebooks

Our analysis uses Jupyter notebooks for figures and demonstrations.
Jupyter notebooks are found in the
[`./notebook`](https://github.com/ChitambarLab/CVChannel.jl/tree/main/notebook)
directory.
Notebooks are most conveniently viewed on github, however, instructions for
running notebooks are found in
[`README.md`](https://github.com/ChitambarLab/CVChannel.jl/blob/main/README.md#notebooks).

## Citing

To cite this software please see [CITATION.bib](https://github.com/ChitambarLab/CVChannel.jl/blob/main/CITATION.bib) or [Zenodo](https://zenodo.org/badge/latestdoi/344167841).

## Contributing

If you are interested in contributing to this software, please reach out to the authors.
Development instructions are found in the
[README.md](https://github.com/ChitambarLab/CVChannel.jl/blob/main/README.md#development).

## Licensing

CVChannel.jl is released under the [MIT License](https://github.com/ChitambarLab/CVChannel.jl/blob/main/LICENSE).

## Acknowledgments

Development of CVChannel.jl was made possible by the advisory of Eric Chitambar
and support from the Electrical and Computer Engineering and Physics departments
at the University of Illinois at Urbana-Champaign.

## Funding

This project is funded by NSF Award # 2016136.
