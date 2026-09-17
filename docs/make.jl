using Documenter
using TinyHiGHS

DocMeta.setdocmeta!(TinyHiGHS, :DocTestSetup, :(using TinyHiGHS); recursive=true)

makedocs(
    modules = [TinyHiGHS],
    authors = "Laurent Plagne and contributors",
    sitename = "TinyHiGHS.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://LaurentPlagne.github.io/TinyHIGHS.jl",
        edit_link = "main",
        assets = ["assets/mermaid.js"],
    ),
    pages = [
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "System Architecture" => "architecture.md",
        "Evolution & Optimizations" => "porting_optimizations.md",
        "Benchmarks" => "benchmarks.md",
        "API Reference" => "api.md",
    ],
    warnonly = true,
    checkdocs = :none,
)

deploydocs(
    repo = "github.com/LaurentPlagne/TinyHIGHS.jl.git",
    devbranch = "main",
    push_preview = true,
)
