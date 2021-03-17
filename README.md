# CV-Channel
A numerical analysis of the communication values accessible to quantum and classical channels.

## Development

To develop new code it is recommended that you use [Revise.jl](https://timholy.github.io/Revise.jl/stable/).
After launching the Julia command prompt run `using Revise`.
This will allow your saved changes to be reflected in the working codebase used by
Julia.
Therefore, you will generally not need to restart Julia to test your code changes.
Please refer to the Revise.jl documentation for more details.

To develop CVChannel.jl code enter package mode by entering `]` then run
* `(@vx.x) pkg> develop --local .` to tell Julia to run code from the local version of the CVChannel.jl module.

Now when you run `using CVChannel` it will load the local version that you are modifying.

## Run Tests

All tests can be run in two ways:
* From the command prompt run `$ julia --proj=./test --color=yes test/runtests.jl`.
* From package mode run `(@vx.x) pkg> test CVChannel`.

To run a single test,  you can either run the test file like a script from the command line,
or you can run it within your text editor.

## Run Scripts

To run a script:
* `$ julia --proj=./script ./script/path/to/script.jl`

## Dependencies

Project dependencies are listed in `Project.toml`. To load the project environment
in the `julia>` REPL type `]` to enter `Pkg` mode. Then run
* `pkg> activate .`

Each directory `./docs`, `./test`, and `./script` has its own `Project.toml` which
specifies the dependencies of the code in that directory. These dependencies are
independent of those in the main `Project.toml` file which specifies the `CVChannel`
module dependencies.

To add or updated packages in the the environment see [Pkg.jl](https://julialang.github.io/Pkg.jl/v1/)
for more details.
