using Documenter
using CVChannel

makedocs(;
    modules=[CVChannel],
    authors="Brian Doolittle <brian.d.doolittle@gmail.com> and Ian George <igeorge3@illinois.edu>",
    repo="https://github.com/ChitambarLab/cv-channel/blob/{commit}{path}#L{line}",
    sitename="CVChannel.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ChitambarLab.github.io/cv-channel",
        assets=String["assets/custom.css"],
    ),
    pages=[
        "Home" => "index.md",
        "CV Background" => "background.md",
        "CV Optimizations" => "communication_value.md",
        "CV Multiplicativity" => "cv_multiplicativity.md",
        "Utilities" => [
            "Channels, States, and Operations" => "channel_states.md",
            "Optimization Backends" => "optimizer_interface.md",
        ],
    ],
)

deploydocs(
    repo="github.com/ChitambarLab/cv-channel",
    devbranch = "main",
)
