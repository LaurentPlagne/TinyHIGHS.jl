# Message proposé à Oscar Dowson

Bonjour Oscar,

J’ai préparé une reproduction autonome des benchmarks de la PR
[#3301](https://github.com/ERGO-Code/HiGHS/pull/3301), ainsi que les tests de
non-régression associés. Le dépôt TinyHiGHS.jl contient le port Julia, les
séquences anonymisées et le replay C++ utilisé pour comparer HiGHS officiel et
la version patchée.

Pour reproduire les résultats :

```bash
git clone https://github.com/LaurentPlagne/TinyHIGHS.jl.git
cd TinyHIGHS.jl

# 1. Replay C++ comparant votre libhighs installée :
./contrib_highs/run_bench_cpp.sh

# 2. Benchmark complet à trois voies en Julia :
julia --project=. bench/bench_3way.jl
```

Le script C++ détecte automatiquement l’artefact HiGHS fourni par Julia. Pour
tester une compilation locale, définir `HIGHS_INSTALL` ou `HIGHS_DIR` avant de
lancer le replay. Le benchmark Julia continue à produire les résultats
TinyHiGHS même lorsqu’un binaire C++ de comparaison n’est pas disponible, et
signale explicitement toute divergence d’objectif.

Bien cordialement,

Laurent
