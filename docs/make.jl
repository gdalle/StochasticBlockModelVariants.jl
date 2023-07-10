using StochasticBlockModelVariants
using Documenter

DocMeta.setdocmeta!(
    StochasticBlockModelVariants,
    :DocTestSetup,
    :(using StochasticBlockModelVariants);
    recursive=true,
)

makedocs(;
    modules=[StochasticBlockModelVariants],
    authors="Guillaume Dalle <22795598+gdalle@users.noreply.github.com> and contributors",
    repo="https://github.com/gdalle/StochasticBlockModelVariants.jl/blob/{commit}{path}#{line}",
    sitename="StochasticBlockModelVariants.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://gdalle.github.io/StochasticBlockModelVariants.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/gdalle/StochasticBlockModelVariants.jl", devbranch="main")
