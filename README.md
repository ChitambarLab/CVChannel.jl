# CVChannel.jl

*A numerics library for evaluating the communication value of a quantum channel.*

[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://ChitambarLab.github.io/CVChannel.jl/dev)[![Test Status](https://github.com/ChitambarLab/CVChannel.jl/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/ChitambarLab/CVChannel.jl/actions/workflows/run_tests.yml)[![codecov](https://codecov.io/gh/chitambarlab/CVChannel.jl/branch/main/graph/badge.svg?token=a4zdwZxA58)](https://codecov.io/gh/chitambarlab/CVChannel.jl)[![DOI](https://zenodo.org/badge/344167841.svg)](https://zenodo.org/badge/latestdoi/344167841)

The communication value (CV) quantifies the performance of single-copy classical
communication.

## Features:
* Convex optimization methods bounding the communication value of a quantum channel.
* Tools for certifying the non-multiplicativity of the communication value for
  quantum channels.

This Julia package and numerical analysis support the findings in
[The Communication Value of a Quantum Channel](https://arxiv.org/abs/2109.11144).

## Quick Start

1. Install Julia: [https://julialang.org/downloads/](https://julialang.org/downloads/)
2. Add the CVChannel.jl package from the Julia command prompt:

```julia
julia> using Pkg; Pkg.add("CVChannel")
```

To import the CVChannel.jl, run `using CVChannel` in the desired Julia file or
workspace.

## SDP Solvers

This project optimizes the communication value of quantum channels using
semidefinite programming via [Convex.jl](https://jump.dev/Convex.jl/stable/).

### SCS

By default, CVChannel.jl uses [SCS](https://github.com/cvxgrp/scs)
to solve semidefinite programs.
SCS is an open-source numerical solver easily accessible through Julia.

### Mosek

If desired, [MOSEK](https://www.mosek.com/) can be used instead.
However, a MOSEK license is required and can be downloaded at
[https://www.mosek.com/products/academic-licenses/](https://www.mosek.com/products/academic-licenses/).
The license is free for personal and institutional academic use and once downloaded,
should be saved at `$HOME/mosek/mosek.lic`.

## Citing

To cite this software please see [CITATION.bib](https://github.com/ChitambarLab/CVChannel.jl/blob/main/CITATION.bib) or [![DOI](https://zenodo.org/badge/344167841.svg)](https://zenodo.org/badge/latestdoi/344167841).

## Development

It is recommended that you use [Revise.jl](https://timholy.github.io/Revise.jl/stable/)
so that your saved changes are reflected in the working codebase used by Julia.
Please refer to the Revise.jl documentation for more details.

To open the CVChannel.jl package for development,
* Enter package mode from the Julia REPL by entering `]`.
* Run `(@vx.x) pkg> develop --local .`

This tells Julia to run code from the local version of the CVChannel.jl module
rather than from the github repository.
Thus, `using CVChannel` will load the local version that you are modifying.

### Scripts

To run a script:
* `$ julia --proj=./script ./script/path/to/script.jl`

### Notebooks

Jupyter notebooks are found in the `./notebook` directory and  are written
either with Python or Julia.
If you are committing changes to a notebook, make sure you restart the kernel and
run all cells before committing.
To run or develop notebooks, perform the following steps:

#### Julia Notebooks

1. Navigate to the `./notebook` directory `$ cd ./notebook`
2. Run `$ julia --project=. -e "using IJulia; notebook(dir=pwd())"`

At this point, the Jupyter Notebook interface will launch in your default web
browser and you can then edit, create, or run the project notebooks.

#### Python Notebooks

1. Create the `CVChannel.jl-notebook` Conda environment with `$ conda env create -f python_environment.yml`.
2. Activate the `CVChannel.jl-notebook` environment with `$ conda activate CVChannel.jl-notebook`.
3. Launch the Jupyter notebook server with `$ jupyter-notebook`.

At this point, the Jupyter Notebook interface will launch in your default web
browser and you can then edit, create, or run the project notebooks.

### Tests

All tests can be run in two ways:
* From the command prompt run `$ julia --proj=./test --color=yes test/runtests.jl`.
* From package mode run `(@vx.x) pkg> test CVChannel`.

To run a single test, you can either run the test from the command line like a script,
or you can run it within your text editor.

### Documentation

Please refer to the [Documenter.jl](https://juliadocs.github.io/Documenter.jl/stable/)
documentation for details on how to write and build Julia documentation.

To verify that content renders properly, you'll need to build the documentation
locally by running the `./docs/make.jl` script.
* `$ julia --color=yes ./docs/make.jl`

This will build the HTML for the docs webpage.
To view the webpage, spawn a local http server:
the constructed website in your browser by spawning a local http server.
1. Navigate to the `./docs/build/` directory.
2. Run `$ python -m http.server --bind  localhost`
3. Then copy/paste the returned url into your web browser.

For this to work properly, the `python` command should run python version 3 or greater.

### Dependencies

Project dependencies are listed in `Project.toml`. To load the project environment
in the `julia>` REPL type `]` to enter `Pkg` mode. Then run
* `pkg> activate .`

Each directory `./docs`, `./test`, `./script`, and `./notebook` has its own
`Project.toml` which specifies the dependencies for the code in that directory
These dependencies are independent from the `CVChannel` dependencies specified
in `./Project.toml` file in the root directory.

See [Pkg.jl](https://julialang.github.io/Pkg.jl/v1/) for details about how to
add or update packages.
