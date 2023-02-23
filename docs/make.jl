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
EXAMPLE = joinpath(@__DIR__, "..", "examples", "example.jl")
OUTPUT = joinpath(@__DIR__, "src/generated")

# function preprocess(str)
#     str = replace(str, "x = 123" => "y = 321"; count=1)
#     return str
# end
# Literate.markdown(EXAMPLE, OUTPUT, preprocess = preprocess)
# # Literate.notebook(EXAMPLE, OUTPUT, preprocess = preprocess)
# # Literate.script(EXAMPLE, OUTPUT, preprocess = preprocess)

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
        "API Reference" => "reference.md",
        # "Examples" => [
        #     "overview.md",
        #     "generated/example.md"
        # ],
    ]
)

# deploydocs(
#     repo = "github.com/hstrey/BDTools.jl.git",
#     push_preview = true,
#     versions = ["v#.#", "dev" => "dev"],
#     deploy_config = deployconfig,
# )