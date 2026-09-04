# Appendix D.1 in Lean 4

An exercise and an independent check. Formalizes the parts of Appendix D.1 that are
*arithmetic* rather than *protocol-theoretic* — the parts that quantify over all poses, all
square roots, and all ring elements, which `code/randomized_implementations.py` cannot check
because SymPy can only evaluate instances.

Nothing here is cited by the thesis. It is not a deliverable; it is a second opinion.

## What is proved

1899 lines across five files, no `sorry`, no axioms beyond Lean's three (`propext`,
`Classical.choice`, `Quot.sound` — asserted for 54 headline results by
`LeanExactness/Check.lean`).

### `AppendixD1.lean` — Lemmas 5 and 6

**Part I — Lemma 5 (rotation-proof witness).**

| Lean name | Statement |
| --- | --- |
| `det_mem_of_pose` | if some orthogonal `R` carries `B` into a subring `K`, then `det B ∈ K` |
| `witness_mem_of_exists_pose` | the same in vector form: a witness outside `K` in *one* pose bars *every* pose |
| `gram_det_eq_witness_sq` | `det Γ = (det B)²` — the witness enters squared, as the discriminant |
| `gram_det_change_of_basis` | changing triple by `C` multiplies `det Γ` by `(det C)²` |

Stated over an arbitrary `Subring ℝ`, which is strictly more general than the thesis's `K_ℝ`
and makes the proofs shorter.

**Part IV — Lemma 5's last clause.** Stated over a `Subfield ℝ`, since it is the one step
that needs division.

| Lean name | Statement |
| --- | --- |
| `inv_entries_mem` | a matrix over a subfield has its inverse over that subfield |
| `det_eq_mul_of_gram_mem` | `det W = c · det B` with `c ∈ K`, given the Gram and cross inner products lie in `K` |
| `det_mem_iff_of_gram_mem` | hence two independent triples are in `K` together or out together |
| `witness_mem_iff_of_vertex_inner_mem` | **any independent triple decides**, over a solid's vertex family |
| `witness_notMem_of_witness_notMem` | the contrapositive, the direction D.1 uses |

The hypothesis is that every pairwise inner product of vertices lies in `K` — *pose-free*, since
inner products are rotation-invariant, which is what makes the conclusion a statement about the
solid. The change of basis is `C = M Γ⁻¹` with `Γ = B Bᵀ` and `M = W Bᵀ`, both assembled from
inner products alone; `Γ⁻¹` is where a subring will not do and `KRsubfield` is spent.

**Part II — Lemma 6 (membership in `K_ℝ`).**

| Lean name | Statement |
| --- | --- |
| `sqrt3_notMem_KR` | `√3 ∉ ℚ(√2,√5)` — bars the tetrahedron, cube, dodecahedron |
| `sqrt_notMem_KR` | **general criterion**: `√n ∉ K_ℝ` unless one of `n, 2n, 5n, 10n` is a perfect square |
| `sqrt6/7/15/30_notMem_KR` | instances of it, by `decide` |
| `eq_of_squarefree_of_isSquare_mul` | two squarefree naturals whose product is a square are equal |
| `sqrt_mem_KR_iff_of_squarefree` | **the classification**: for squarefree `d > 1`, `√d ∈ K_ℝ ↔ d ∈ {2,5,10}` |
| `descent` | a square in `K_ℝ` lying in `ℚ(√5)` is, up to a factor 2, a square in `ℚ(√5)` |
| `Q5.sq_eq_rat` | the descent's shadow one floor down: a rational square in `ℚ(√5)` is `a²` or `5b²` |
| `norm_of_sq` | a square in `ℚ(√5)` has a rational square for its norm |
| `icosahedral_witness_not_sq` | `2/25(5−√5)` is not a square in `K_ℝ` (norms `16/125`, `4/125`) |
| `icosahedral_normalizer_not_sq` | `(5+√5)/2` is not a square in `K_ℝ` (norms `5`, `5/4`) |

The descent and Lemma 6's first sentence are elementary — write `β = p + q√2` over the base,
square, read off `pq = 0` — which is the route the printed proofs take, so the formalization
tracks them step for step. **No Galois theory is used anywhere**, in Lean or in print, and
`ℚ(√5)` is built by hand as a field (`Q5.add/mul/neg/sub/inv/div`, coefficient uniqueness in
`Q5.ext`).

**Part III — the two composed.** `KRsubring` / `KRsubfield` present `K_ℝ` as a subring and a
subfield of `ℝ`, which lets Part I be applied to Part II. The field structure then gives
`mul_notMem_KR` and `div_notMem_KR`: a nonzero `K_ℝ`-multiple of an outsider, or of an
outsider's reciprocal, is an outsider. Those are what the solids actually need — see below.

### `Solids.lean` — the five solids, at their atlas coordinates

Everything in Part III was *conditional* on a witness value taken on faith from Table
`tab:povm-exactness`; nothing named a Platonic solid. This file writes the atlas vertices
down, computes the witness in Lean, and derives `no_pose_tetrahedron`, `no_pose_cube`,
`no_pose_dodecahedron`, `no_pose_icosahedron` unconditionally. Three things fell out:

- **The `√3` composed theorem did not reach any solid.** `no_pose_of_witness_eq_sqrt3` wants
  a witness of `√3` exactly, and no solid has one — all three `√3` solids give `±4√3/9`. The
  repair is `no_pose_of_witness_mem_sqrt3_coset`, quantifying over the `K_ℝ`-coset. (The
  icosahedral theorem needed no repair: its witness enters squared, and `2/25(5−√5)` is the
  value on the nose. `witness_ico_sq` proves it.)
- **The alignment sweep needs no sweep.** Appendix D.1 argues Corollary 7 vertex by vertex,
  and `randomized_implementations.py` runs it that way, 44 vertices. But the atlas presents
  each solid as `K_ℝ` coordinates over *one* normalizer, and `vertex_notMem_of_normalizer`
  disposes of all 44 in three lines. The sweep's content is a fact about the solid's *radius*.
  The octahedron is the solid whose normalizer is `1` — the appendix's asymmetry, restated.
- **The repo's two vertex orderings disagree, and the witness's sign rides on that.** Table
  E.1 / `povm_*.npz` and the randomized suite's `symbolic_solids()` (`code/randomized_core.py`)
  number the tetrahedron, cube and icosahedron differently — same sets, same pose — so the two
  give opposite signs, `±4√3/9` for the tetrahedron and the icosahedron the other way round.
  The verdict never moves (the min-poly column is sign-blind), but the caption sends the reader
  to Table E.1 for `v₁, v₂, v₃`, so the printed sign has to be reproducible from there:
  `det_witness` runs on `atlas_vertices()`, an `(a,b,c)` column carries the indices, and the
  caption states what no choice of triple moves, the `K_ℝ`-coset (the dodecahedron's 960
  spanning triples realize four different minimal polynomials, so the coset is the only
  triple-blind invariant). The generator asserts the signed value, so a reordering trips a
  check. `witness_tetU` and `witness_tetU'` prove both signs, and `no_pose_tetrahedron'` shows
  the verdict is the same either way — Lemma 5's "any independent triple decides", made
  concrete on the one disagreement the repo contains.

### `Weights.lean` — Theorem 4's arithmetic core

The *second* obstruction, the one on weights, which bars every deterministic protocol for all
five solids and is what forces the octahedron's coin.

| Lean name | Statement |
| --- | --- |
| `Rring` | `ℛ = ℤ[φ, i, √2, ½]` as a `ℤ`-subalgebra of `ℂ` |
| `Rring_le_halvedIntegral` | every element of `ℛ` is an algebraic integer over a power of two |
| `isDyadic_of_rat_mem_Rring` | **`ℛ ∩ ℚ ⊆ ℤ[½]`** |
| `rat_mem_Rring_of_isDyadic` | the converse, so the intersection is an equality |
| `isDyadic_two_div_iff` | `2/V ∈ ℤ[½] ↔ V` is a power of two |
| `weight_obstruction` | the two composed |
| `octahedron/icosahedron/dodecahedron_weight_notMem` | `⅓`, `⅙`, `⅒` are not in `ℛ` |
| `tetrahedron/cube_weight_mem` | `½` and `¼` **are** — those two fail the *other* test |
| `KCsubfield` | `K = ℚ(√2,√5,i)`, as the split `z = Re z + i Im z` over `K_ℝ` |
| `Rring_le_KCsubfield` | `ℛ ⊆ K`, generator by generator |
| `fracRring_eq_KC` | **`Frac ℛ = K`** — `√5 = 2φ − 1` is what closes the other inclusion |
| `mem_fracRring_iff_KR` | **`Frac ℛ ∩ ℝ = K_ℝ`** |

The decisive step is the one the appendix uses once and decisively: a rational
algebraic integer is an ordinary integer. Here that is mathlib's `IsIntegrallyClosed ℤ`.

**The last four rows identify the ring with the field.** `AppendixD1.lean` defines `KR` and
this file defines `Rring`; without a theorem joining them the two halves would prove things
about two objects only the prose identifies.
Every membership verdict in Appendix D.1 is a test against `ℚ(√2,√5)`, and
`mem_fracRring_iff_KR` is what says that is the right field to test against. Writing `K` as
`{z | Re z ∈ K_ℝ ∧ Im z ∈ K_ℝ}` is what keeps the proof short: `{1, i}` being a `K_ℝ`-basis is
then not a hypothesis but `Complex.ext`, and the only closure with any content is the inverse,
where `Complex.inv_re` divides by `normSq` and `KRsubfield` rather than `KRsubring` is spent.
`Subfield.closure` supplies the fraction field and the rationals; `√5 = 2φ − 1` supplies the
inclusion that would otherwise leave `Frac ℛ` too small.

Note the boundary this respects. Theorem 4 as *stated* quantifies over protocols and stays
out, for the reason below. Its second sentence — "`ℛ ∩ ℚ = ℤ[½]`, so a transitive covariant
POVM on `V` outcomes admits a deterministic implementation only if `V` is a power of
two" — is a claim about a subring of `ℂ`, needs no model of anything, and is on exactly the
footing Lemma 5 is. The boundary has not moved; it has been drawn where the mathematics is
rather than where the theorem number is.

## What is *not* proved, and deliberately so

**Lemma 2 (branch decomposition), Theorem 1, and Theorem 4's protocol half.** These quantify
over *protocols*. Mathlib has no circuit model, no POVMs, no channels, so formalizing them
means first writing the protocol model down — and the residual risk in Appendix D.1 is the
model's *adequacy* (does it cover repeat-until-success, heralding, catalytic ancillas?), not
the inferences drawn from it. Lean would certify the wrong half. The thesis itself records one
such gap already, at the comment on Theorem 4: the model never states "each fork has finitely
many outcomes".

**The general lesson.** Lean's coverage here is *anti-correlated* with the actual uncertainty.
Everything above is a statement about fields and rings, and every one of them was already
secure. What the exercise turned up was not a false theorem but three places where the
formalization did not reach what it appeared to reach (the `√3` coset), where the argument was
doing more work than it needed (the 44-vertex sweep), and where two artifacts in the repo
disagree about a printed sign.

Two steps that could pass for gaps are covered, and the shape of Parts II and IV is what
covers them.

*Lemma 5's last clause* is Part IV: sixty lines whose only idea is that `Matrix.adjugate_apply`
writes each adjugate entry as a determinant over `A`, `0` and `1`. What makes the clause bite
for a solid is `C` having entries in `K_ℝ`. `Sanity.lean` spends it to derive the tetrahedron's
second verdict from the first, with no second determinant computed — a proof where exhibiting
`no_pose_tetrahedron` against `no_pose_tetrahedron'` would only be a witness.

*Lemma 6's classification* rests on one fact about ℕ — two squarefree numbers whose product is
a square are equal. That is `eq_of_squarefree_of_isSquare_mul`, proved through
`Nat.factorization`: a square carries every prime to an even power, a squarefree number carries
each to at most one, and two numbers that are each `0` or `1` and sum to an even number are
equal. It is the sentence the slow walk spells out
(`extras/math/exactness-slow-walk.tex`), formalized.

## Sanity controls

`LeanExactness/Sanity.lean` exists because a formalization that compiles can still be vacuous.
It checks that `KR` is inhabited by what it should be; that the `¬ ∃ β, KR β ∧ α = β²` shape is
genuinely refutable (`3 + 2√2 = (1+√2)²` is exhibited as a square in `K_ℝ`); that both branches
of `descent` are reachable; that the four norms match the four the slow walk prints, value for
value; that Lemma 5 does *not* fire on the octahedron, whose witness is `1 ∈ K_ℝ`; that the
`no_pose_*` hypothesis shape is *satisfiable* (the identity carries the octahedron's triple
into `K_ℝ³`) so the four theorems refute something real; that the alignment lemma fires on
actual atlas vertices, including one with a zero coordinate, and has no purchase on the
octahedron; that `Rring` contains its generators; that `IsDyadic` is both satisfiable and
refutable; and that the weight test *passes* for the tetrahedron and cube, so it is a test with
two sides rather than one that rejects everything; and, for the ring-field identification,
that both sides of `Frac ℛ ∩ ℝ = K_ℝ` are exercised — `√10` in, `√3` out, the latter *by Lemma
6*, which is what the identification is for — and that the `∩ ℝ` removes something, `i` lying
in `Frac ℛ` and being no real's image.

It also controls the classification and Lemma 5's last clause. For the classification, that
**both its hypotheses are load-bearing**, by counterexample: drop `1 < d` and `d = 1` breaks
the `iff`; drop squarefreeness and `d = 8` does, `√8 = 2√2` lying in `K_ℝ` while
`8 ∉ {2,5,10}`. And that the classification reproduces `sqrt3_notMem_KR`, the one instance the
appendix needs. For Lemma 5's last clause, the tetrahedron is run through it: its four vertices
as one family, `tetU`'s and `tetU'`'s orderings as two triples drawn from it, every pairwise
inner product shown to lie in `K_ℝ` — the normalizer enters squared, as `1/3`, which is exactly
why the `√3` that bars the witness does not reach the hypothesis — and then the verdict for the
second triple obtained from the first *by the theorem*, agreeing with `Solids.lean`'s
independent hand computation. Both sides of that `iff` fall on the "outside `K_ℝ`" branch, the
branch Appendix D.1 actually uses, so the control does not pass for the trivial reason.

The last of these is the point of running two obstructions: the tetrahedron and cube fail on
directions and pass on weights, the icosahedron and dodecahedron fail both, and the octahedron
passes on directions and fails on weights. Neither test alone convicts all five.

## Setup

Toolchain is `leanprover/lean4:v4.33.0-rc2` with mathlib pinned in `lake-manifest.json`.
`lake` ships with `elan`, which puts it in `~/.elan/bin`; prefix the commands with that path if
it is not on yours.

```bash
cd extras/lean-exactness
lake exe cache get   # mathlib .olean cache, ~7.8 GB, only needed after a clean
lake build           # builds all six files; prints the axiom check
```

A clean build of this project's own files takes about nine seconds once the mathlib cache is
present; the cache download is the only slow step. `.lake/` is gitignored (7.8 GB), and so are
the template's `.github/` workflows, inert in a subdirectory — what is tracked is the five
`.lean` files, `LeanExactness.lean`, `lakefile.toml`, `lake-manifest.json`, `lean-toolchain`
and this README.

For interactive work, VS Code's `lean4` extension picks the toolchain up from
`lean-toolchain` automatically.
