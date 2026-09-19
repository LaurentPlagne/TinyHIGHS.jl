# Proposition de Pull Request pour HiGHS C++ : Inversion SIMD et Substitution Branchless des Pivots Diagonaux (`HFactor`)

Ce document fournit la feuille de route complète et les instructions pas-à-pas pour préparer, tester et soumettre la Pull Request dans le dépôt **HiGHS** (`/Users/laurentplagne/Projects/HIGHS`).

---

## 1. Contexte & Motivation Micro-Architecturale

### Le problème de la division flottante
Dans le simplexe révisé de HiGHS, chaque itération appelle intensivement les résolutions triangulaires creuses **FTRAN** ($U x = b$) et **BTRAN** ($U^T x = b$) via `HFactor`.
Pour chaque terme diagonal $U_{ii}$, l'algorithme d'origine effectue une division scalaire :
```cpp
pivot_multiplier /= u_pivot_value[i_logic];
```
Sur les architectures x86-64 et ARM64 modernes, l'instruction de division flottante (`FDIV` / `vdivsd`) présente une **latence de 10 à 15 cycles processeur** et ne peut pas être pipelinée avec le même débit que l'addition ou la multiplication.

### L'écueil du court-circuit conditionnel (*Branch Misprediction*)
Un test naïf `if (pivot == 1.0) ... else if (pivot == -1.0) ... else /=` fonctionne très bien sur des problèmes de graphes purs où plus de 90 % des pivots sont unitaires.
**Cependant, sur des matrices mixtes ou générales** (mélange de contraintes combinatoires et de contraintes continues réelles), les pivots alternent de manière imprévisible :
* Le prédicteur de branchement matériel (TAGE) subit des échecs répétés (*branch mispredictions*).
* Chaque vidage de pipeline sur un processeur moderne (14 à 20 étages) coûte **15 à 20 cycles**.
* **Résultat mesuré en laboratoire micro-architectural** : le temps de passe triangulaire explose de **3.0 µs à 7.0 µs (+130 % de régression !)**.

### La solution : Pré-inversion SIMD + Substitution Branchless
1. **Pendant la factorisation LU (`buildFinish`)** :
   Les pivots diagonaux $U_{ii}$ forment un tableau dense contigu de taille $m$. Le calcul de leurs inverses $D^{-1} = 1.0 / U_{ii}$ est vectorisé par paquets **SIMD** (AVX-512 : 8 doubles/cycle, AVX2 : 4 doubles/cycle, ARM Neon : 2 doubles/cycle).
2. **Pendant la mise à jour de Forrest-Tomlin (`updateFT`)** :
   Lorsqu'un pivot est ajouté en fin de tableau, son inverse est ajouté immédiatement : `u_pivot_inv_value.push_back(1.0 / new_pivot)`.
3. **Dans FTRAN, BTRAN et `solveHyper`** :
   La substitution devient **strictement sans branchement (*branchless*)** :
   ```cpp
   pivot_multiplier *= u_pivot_inv_value[i_logic];
   ```
   * **Latence fixe de 3 à 4 cycles** (contre 10–15 cycles pour `FDIV`).
   * **Zéro branchement** : aucune pénalité de prédiction possible.
   * **Immunité totale** : vitesse optimale et stable sur tous les types de problèmes (réseaux, mixtes, continus).

---

## 2. Emplacement des Fichiers Prêts à l'Emploi

Dans le présent dépôt TinyHiGHS, vous disposez déjà de :
* Le patch unifié Git : [`contrib_highs/patches/0002-hfactor-simd-branchless-pivot-inverses.patch`](patches/0002-hfactor-simd-branchless-pivot-inverses.patch)
* Le banc d'essai micro-architectural reproductible : [`bench/bench_pivot_strategies.jl`](../bench/bench_pivot_strategies.jl)

---

## 3. Guide pas-à-pas pour la session HiGHS (`/Users/laurentplagne/Projects/HIGHS`)

Lorsque vous démarrez votre session de développement sur le répertoire
`/Users/laurentplagne/Projects/HiGHS` :

### Étape 1 : Créer une branche propre
```bash
cd /Users/laurentplagne/Projects/HiGHS
git checkout latest
git pull origin latest
git checkout -b perf/simd-branchless-pivots
```

### Étape 2 : Appliquer le patch préparé
```bash
git apply /Users/laurentplagne/Projects/fusion_corrections/TinyHiGHS.jl/contrib_highs/patches/0002-hfactor-simd-branchless-pivot-inverses.patch
```
*(Optionnel si vous préférez vérifier le diff avant d'appliquer : `git apply --check ...`)*

### Étape 3 : Compiler et exécuter les tests Catch2
```bash
cd /Users/laurentplagne/Projects/HiGHS
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DALL_TESTS=ON
cmake --build build --parallel

# Exécution de la suite de tests unitaires
ctest --test-dir build --output-on-failure
```

### Étape 4 : Commit et Push sur votre fork
```bash
git add highs/util/HFactor.h highs/util/HFactor.cpp
git commit -m "HFactor: SIMD pre-inversion and branchless substitution for diagonal pivots"
git push -u myfork perf/simd-branchless-pivots
```

---

## 4. Modèle de Message pour la Pull Request GitHub

Voici le texte prêt à copier-coller pour l'ouverture de la PR sur `ERGO-Code/HiGHS` :

### Titre suggéré
```text
perf(HFactor): vectorized SIMD pre-inversion and branchless diagonal pivot substitution
```

### Description suggérée
```markdown
### Summary of Changes
This PR optimizes the triangular substitution passes in `HFactor` (`ftranU`, `btranU`, and `solveHyper`) by replacing the per-pivot scalar floating-point division (`FDIV`) with a branchless multiplication using pre-inverted diagonal pivots:
1. **Vectorized SIMD Pre-Inversion**: In `HFactor::buildFinish`, pivot inverses are computed in a single contiguous pass using portable vector intrinsics (AVX-512, AVX, and ARM Neon with compiler auto-vectorization fallback).
2. **Dynamic Updates**: In `HFactor::updateFT`, reciprocal pivots are pushed directly alongside new diagonal elements.
3. **Branchless Substitution**: `pivot_multiplier /= u_pivot_value[i]` is replaced by `pivot_multiplier *= u_pivot_inv_value[i]`.

### Motivation & Micro-Architectural Rationale
- **Latency reduction**: Floating-point division (`vdivsd` / `FDIV`) has a latency of 10–15 clock cycles. In contrast, floating-point multiplication (`vmulsd` / `FMUL`) has a latency of only 3–4 cycles and high pipeline throughput.
- **Elimination of branch misprediction hazard**: While checking `if (pivot == 1.0)` benefits pure network problems, on mixed problems (alternating between network constraints and general continuous constraints), branch target prediction (TAGE) suffers frequent mispredictions (15–20 cycle penalty per flush). In micro-benchmarks on mixed instances, branching degraded FTRAN runtime by +130% (from 3.0 µs to 7.0 µs).
- **Universality**: The branchless reciprocal approach removes the branch across **all** LP instance types; its numerical contract for arbitrary pivots is one ULP rather than bit-for-bit identity.

### Numerical Equivalence
- For unit pivots ($\pm 1.0$) and powers of two, multiplication by the precomputed reciprocal is bit-for-bit exact in IEEE-754 arithmetic.
- For arbitrary pivots, differences remain within one ULP on the tested corpus; the default branching strategy remains available for strict bit-for-bit oracle comparisons.
```
