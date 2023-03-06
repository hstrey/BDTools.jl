using Documenter
using BDTools

if haskey(ENV, "GITHUB_ACTIONS")
    ENV["JULIA_DEBUG"] = "Documenter"
end

deployconfig = Documenter.auto_detect_deploy_system()
Documenter.post_status(deployconfig; type="pending", repo="github.com/hstrey/BDTools.jl.git")
using Literate
# using Plots # to not capture precompilation output

# generate examples
EXAMPLE = joinpath(@__DIR__, "..", "examples", "tutorial.jl")
OUTPUT = joinpath(@__DIR__, "src/generated")
Literate.markdown(EXAMPLE, OUTPUT)

makedocs(
    format = Documenter.HTML(
        # assets = ["assets/custom.css", "assets/favicon.ico"],
        prettyurls = true, # haskey(ENV, "GITHUB_ACTIONS"),
        canonical = "https://hstrey.github.io/BDTools.jl/stable",
    ),
    modules = [BDTools],
    sitename = "BDTools.jl",
    pages = Any[
        "Home" => "index.md",
        "Tutorial" => "generated/tutorial.md",
        "API Reference" => "reference.md",
    ]
)

deploydocs(
    repo = "github.com/hstrey/BDTools.jl.git",
    push_preview = true,
    versions = ["v#.#", "dev" => "dev"],
    deploy_config = deployconfig,
)