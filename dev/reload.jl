# Rechargement des sources de `TinyHiGHS` en session interactive (Kaimon).
#
#   include("julia_simplex/dev/reload.jl")   # une fois par session
#   reload_julia_simplex()                   # après chaque édition de src/
#
# Pourquoi un rechargement manuel : Revise suit les fichiers déjà inclus, mais
# **pas un fichier nouveau** ni une redéfinition de `struct`. Après avoir ajouté
# un fichier ou touché à un type, `Base.include` dans l'ordre de
# `src/TinyHiGHS.jl` évite de redémarrer la session. L'ordre est lu dans
# `TinyHiGHS.jl` : ne pas le recopier ici — une liste dupliquée a oublié
# `primal.jl` et produit une méthode liée à un `SimplexEngine` périmé
# (cf. `docs/pieges/julia-langage.md`, Revise et les `struct`).
#
# Reprise d'un chantier : voir `docs/architecture/portage-julia-simplexe-highs.md`
# §3.1 (état du jalon, prochain périmètre, commandes de tests).

const JULIA_SIMPLEX_ROOT = abspath(joinpath(@__DIR__, ".."))

# Import au niveau supérieur (pas dans une fonction : la liaison créée par
# `using` serait d'un monde plus récent que l'appelant).
if !isdefined(Main, :TinyHiGHS)
    JULIA_SIMPLEX_ROOT ∈ LOAD_PATH || push!(LOAD_PATH, JULIA_SIMPLEX_ROOT)
    @eval Main using TinyHiGHS
end

"""Ordre d'`include` de `src/TinyHiGHS.jl`, résolu en chemins absolus."""
function simplex_source_files()
    main = joinpath(JULIA_SIMPLEX_ROOT, "src", "TinyHiGHS.jl")
    files = String[]
    for line ∈ eachline(main)
        m = match(r"^include\(\"([^\"]+)\"\)", strip(line))
        m === nothing && continue
        push!(files, joinpath(JULIA_SIMPLEX_ROOT, "src", m.captures[1]))
    end
    return files
end

"""
    reload_julia_simplex()

Charge `julia_simplex` si besoin (`LOAD_PATH`), puis ré-inclut toutes les
sources dans l'ordre du module. Rend le module.

L'import de `TinyHiGHS` est fait au niveau supérieur du script (une liaison
créée par `using` depuis une fonction serait d'un monde plus récent que
l'appelant : « binding may be too new »).
"""
function reload_julia_simplex()
    module_ = getfield(Main, :TinyHiGHS)
    for f ∈ simplex_source_files()
        Base.include(module_, f)
    end
    return module_
end
