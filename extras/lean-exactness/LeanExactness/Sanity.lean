/-
# Sanity controls

A formalization that compiles can still be vacuous, or can state something weaker than intended.
These `example`s are the guard: each must compile, and together they show that

  * `KR` is inhabited by the things it should be (so `√3 ∉ KR` is not "nothing is in `KR`");
  * the `¬ ∃ β, KR β ∧ α = β²` shape is genuinely *refutable* — there are `α` for which the
    corresponding statement is false, exhibited;
  * both branches of `descent` are reachable, so neither is dead code;
  * the four norms agree, value for value, with the four the slow walk prints
    (`extras/math/exactness-slow-walk.tex`): `16/125`, `4/125`, `5`, `5/4`.
-/
import LeanExactness.AppendixD1
import LeanExactness.Solids
import LeanExactness.Weights

namespace AppendixD1.Sanity

open Real Matrix AppendixD1

/-! ### `K_ℝ` is inhabited, and by the right things -/

example : Q5 (5 : ℝ) := ⟨5, 0, by norm_num⟩
example : Q5 (√5) := ⟨0, 1, by norm_num⟩
example : KR (1 : ℝ) := KR.ofQ5 ⟨1, 0, by norm_num⟩
example : KR (√2) := ⟨0, 1, ⟨0, 0, by norm_num⟩, ⟨1, 0, by norm_num⟩, by ring⟩
example : KR (√5) := KR.ofQ5 ⟨0, 1, by norm_num⟩
example : KR (√10) := ⟨0, √5, ⟨0, 0, by norm_num⟩, ⟨0, 1, by norm_num⟩, by
  rw [sqrt10_eq]; ring⟩

/-- Both five-fold surds lie in `ℚ(√5)`, hence in `K_ℝ`: the theorems about them are about
numbers `K_ℝ` actually contains — what fails is only their being *squares*. -/
example : KR ((2 / 25 : ℝ) * (5 - √5)) := KR.ofQ5 ⟨2 / 5, -2 / 25, by push_cast; ring⟩
example : KR ((1 / 2 : ℝ) * (5 + √5)) := KR.ofQ5 ⟨5 / 2, 1 / 2, by push_cast; ring⟩

/-! ### The `¬ ∃ β, KR β ∧ α = β²` shape is refutable

`3 + 2√2 = (1 + √2)²` is a square in `K_ℝ`, so the two theorems of Lemma 6 are not instances
of a schema that holds for everything. -/

example : ∃ β, KR β ∧ (3 + 2 * √2 : ℝ) = β ^ 2 := by
  refine ⟨1 + √2, ⟨1, 1, ⟨1, 0, by norm_num⟩, ⟨1, 0, by norm_num⟩, by ring⟩, ?_⟩
  linear_combination -sq_sqrt2

/-! ### Both branches of `descent` are reachable -/

/-- `α = p²` branch: `9 = 3²`, with `3 ∈ ℚ(√5)`. -/
example : ∃ p, Q5 p ∧ ((9 : ℝ) = p ^ 2 ∨ (9 : ℝ) = 2 * p ^ 2) :=
  descent (show Q5 (9 : ℝ) from ⟨9, 0, by norm_num⟩)
    (KR.ofQ5 (show Q5 (3 : ℝ) from ⟨3, 0, by norm_num⟩)) (by norm_num)

/-- `α = 2p²` branch: `2 = 2·1²`, reached via `β = √2`, which is *not* in `ℚ(√5)`. -/
example : ∃ p, Q5 p ∧ ((2 : ℝ) = p ^ 2 ∨ (2 : ℝ) = 2 * p ^ 2) :=
  descent Q5.two ⟨0, 1, ⟨0, 0, by norm_num⟩, ⟨1, 0, by norm_num⟩, by ring⟩
    (Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)).symm

/-! ### The four norms, against the four the slow walk prints -/

example : N (2 / 5) (-2 / 25) = 16 / 125 := by norm_num [N]
example : N (1 / 5) (-1 / 25) = 4 / 125 := by norm_num [N]
example : N (5 / 2) (1 / 2) = 5 := by norm_num [N]
example : N (5 / 4) (1 / 4) = 5 / 4 := by norm_num [N]

/-! ### Lemma 5 does not fire on the octahedron

The octahedron in atlas orientation has the Pauli axes for vertices, witness `1 ∈ K_ℝ` — so
`no_pose_of_witness_eq_sqrt3` has nothing to say about it, which is the asymmetry the whole
appendix turns on. -/

example : witness ![1, 0, 0] ![0, 1, 0] ![0, 0, 1] = 1 := by
  simp [witness, Matrix.det_fin_three]

example : KR (witness ![1, 0, 0] ![0, 1, 0] ![0, 0, 1]) := by
  rw [show witness ![1, 0, 0] ![0, 1, 0] ![0, 0, 1] = 1 by
    simp [witness, Matrix.det_fin_three]]
  exact KR.ofQ5 ⟨1, 0, by norm_num⟩

/-- And the identity rotation is an admissible pose, so the hypotheses of the Part III
theorems are satisfiable — they are not vacuously true for want of any `R`. -/
example : (1 : Matrix (Fin 3) (Fin 3) ℝ)ᵀ * 1 = 1 := by simp

/-! ### The `no_pose_*` theorems refute something satisfiable

The four solid-level theorems of `Solids.lean` conclude `False` from "some rotation carries
this triple into `K_ℝ³`".  That is only worth proving if the hypothesis shape is satisfiable in
general — and it is, at the octahedron, whose atlas triple the *identity* already carries into
`K_ℝ³`.  This is the positive control for the whole file. -/

theorem KR_zero : KR (0:ℝ) := KR.ofQ5 ⟨0, 0, by norm_num⟩
theorem KR_one : KR (1:ℝ) := KR.ofQ5 ⟨1, 0, by norm_num⟩

example : ∀ j, KR ((1 : Matrix (Fin 3) (Fin 3) ℝ).mulVec ![1, 0, 0] j) := by
  intro j
  rw [Matrix.one_mulVec]
  fin_cases j
  · exact KR_one
  · exact KR_zero
  · exact KR_zero

example : ∀ j, KR ((1 : Matrix (Fin 3) (Fin 3) ℝ).mulVec ![0, 0, 1] j) := by
  intro j
  rw [Matrix.one_mulVec]
  fin_cases j
  · exact KR_zero
  · exact KR_zero
  · exact KR_one

/-! ### The alignment lemma fires on real atlas vertices, and only where it should -/

open AppendixD1.Solids in
/-- Tetrahedron vertex 1, `(1,1,1)/√3`: coordinates in `K_ℝ`, not all zero — so
`no_alignment_sqrt3` applies and the vertex is barred. -/
example : ¬ (∀ i, KR ((![1, 1, 1] : Fin 3 → ℝ) i / √3)) :=
  no_alignment_sqrt3 (fun i => by fin_cases i <;> exact KR_one) ⟨0, by norm_num⟩

open AppendixD1.Solids in
/-- Dodecahedron vertex 9, `(0,σ,τ)/√3`: the zero coordinate is fine, the lemma needs only
*some* nonzero one. -/
example : ¬ (∀ i, KR ((![0, sigma, tau] : Fin 3 → ℝ) i / √3)) :=
  no_alignment_sqrt3
    (fun i => by fin_cases i; exacts [KR_zero, KR_sigma, KR_tau])
    ⟨2, by simpa using ne_of_gt tau_pos⟩

open AppendixD1.Solids in
/-- Icosahedron vertex 1, `(τ,1,0)/√(2+τ)`. -/
example : ¬ (∀ i, KR ((![tau, 1, 0] : Fin 3 → ℝ) i / rico)) :=
  no_alignment_ico
    (fun i => by fin_cases i; exacts [KR_tau, KR_one, KR_zero])
    ⟨0, by simpa using ne_of_gt tau_pos⟩

open AppendixD1.Solids in
/-- The contrast: the octahedron's normalizer is `1`, which *is* in `K_ℝ`, so
`vertex_notMem_of_normalizer` has no purchase — and indeed the vertex is in `K_ℝ³`. -/
example : ∀ i, KR ((![1, 0, 0] : Fin 3 → ℝ) i / 1) := by
  intro i
  fin_cases i <;> simpa using (by first | exact KR_one | exact KR_zero)

/-! ### The weight obstruction is not vacuous either

`Rring` must contain what it should, `IsDyadic` must be both satisfiable and refutable, and
the weight test must *pass* for two of the five solids — otherwise it would be a test that
rejects everything. -/

open AppendixD1.Weights in
example : (1 / 2 : ℂ) ∈ Rring := Algebra.subset_adjoin (by simp)

open AppendixD1.Weights in
example : Complex.I ∈ Rring := Algebra.subset_adjoin (by simp)

open AppendixD1.Weights in
example : phig ∈ Rring := Algebra.subset_adjoin (by simp)

open AppendixD1.Weights in
example : rt2 ∈ Rring := Algebra.subset_adjoin (by simp)

open AppendixD1.Weights in
/-- `IsDyadic` is satisfiable. -/
example : IsDyadic (3 / 8) := ⟨3, 3, by norm_num⟩

open AppendixD1.Weights in
/-- `IsDyadic` is refutable — so `octahedron_weight_notMem` is not an instance of a schema
that holds for everything. -/
example : ¬ IsDyadic (1 / 3) := by
  rintro ⟨k, m, hk⟩
  have h : (2:ℚ) ^ k = 3 * m := by field_simp at hk; linarith [hk]
  have hz : (2:ℤ) ^ k = 3 * m := by exact_mod_cast h
  have h3 : (3:ℤ) ∣ 2 ^ k := ⟨m, hz⟩
  have h3' : (3:ℕ) ∣ 2 ^ k := by exact_mod_cast h3
  have := (Nat.prime_three).dvd_of_dvd_pow h3'
  omega

open AppendixD1.Weights in
/-- The tetrahedron and cube pass the weight test, so it is a test with two sides. -/
example : ((2 / 4 : ℚ) : ℂ) ∈ Rring ∧ ((2 / 8 : ℚ) : ℂ) ∈ Rring :=
  ⟨tetrahedron_weight_mem, cube_weight_mem⟩

/-! ### The ring and the field are identified, and the identification is not vacuous

`mem_fracRring_iff_KR` says a real number lies in `Frac ℛ` exactly when it lies in `K_ℝ`.  A
statement of that shape is worth nothing if `Frac ℛ ∩ ℝ` were all of `ℝ`, or only `ℚ`, so both
sides are exercised: `√10` is in and `√3` is out — and `√3` is out *by Lemma 6*, which is the
point of connecting the two files at all.  The third control shows the `∩ ℝ` is not decoration:
`i` is in `Frac ℛ` and is no real number's image, so the intersection removes something. -/

open AppendixD1.Weights in
/-- In: `√10 = √2 · √5`, and both factors are ring elements. -/
example : ((√10 : ℝ) : ℂ) ∈ Subfield.closure (Rring : Set ℂ) :=
  ofReal_mem_fracRring ⟨0, √5, ⟨0, 0, by norm_num⟩, ⟨0, 1, by norm_num⟩,
    by rw [sqrt10_eq]; ring⟩

open AppendixD1.Weights in
/-- Out, and out *by Lemma 6*: `Frac ℛ` does not reach `√3`, which is exactly the verdict
Appendix D.1 spends on the tetrahedron, the cube and the dodecahedron. -/
example : ((√3 : ℝ) : ℂ) ∉ Subfield.closure (Rring : Set ℂ) :=
  fun h => sqrt3_notMem_KR ((mem_fracRring_iff_KR _).mp h)

open AppendixD1.Weights in
/-- The `∩ ℝ` is load-bearing: `i` lies in `Frac ℛ` and is not the image of any real. -/
example : Complex.I ∈ Subfield.closure (Rring : Set ℂ) ∧ ∀ x : ℝ, (x : ℂ) ≠ Complex.I :=
  ⟨Subfield.subset_closure I_mem_Rring, fun _ h => by simpa using congrArg Complex.im h⟩


/-! ### The two obstructions really are different tests

Every solid is convicted, but by different counts: the tetrahedron and cube fail on directions
and pass on weights; the icosahedron and dodecahedron fail both; the octahedron passes on
directions and fails on weights.  If either test alone sufficed, the appendix would not need
the ring *and* the field. -/

open AppendixD1.Weights in
example : ((2 / 6 : ℚ) : ℂ) ∉ Rring ∧ ((2 / 4 : ℚ) : ℂ) ∈ Rring :=
  ⟨octahedron_weight_notMem, tetrahedron_weight_mem⟩


/-! ### Lemma 6's classification: both hypotheses are load-bearing

`sqrt_mem_KR_iff_of_squarefree` reads "for squarefree `d > 1`". Neither qualifier is
decoration, and dropping either makes the `iff` false — exhibited, not asserted. -/

/-- Drop `1 < d`: `d = 1` gives `√1 = 1 ∈ K_ℝ` with `1 ∉ {2,5,10}`. -/
example : KR (√(1 : ℕ)) ∧ ¬ ((1 : ℕ) = 2 ∨ (1 : ℕ) = 5 ∨ (1 : ℕ) = 10) := by
  refine ⟨?_, by decide⟩
  rw [show ((1:ℕ):ℝ) = 1 by norm_num, Real.sqrt_one]
  exact KR.ofQ5 ⟨1, 0, by norm_num⟩

/-- Drop squarefreeness: `d = 8` gives `√8 = 2√2 ∈ K_ℝ` with `8 ∉ {2,5,10}`.  So the
classification really is a statement about squarefree `d`, and `eq_of_squarefree_of_isSquare_mul`
is where that hypothesis is spent. -/
example : KR (√(8 : ℕ)) ∧ ¬ ((8 : ℕ) = 2 ∨ (8 : ℕ) = 5 ∨ (8 : ℕ) = 10) := by
  refine ⟨?_, by decide⟩
  have h8 : ((8:ℕ):ℝ) = 2 ^ 2 * 2 := by norm_num
  rw [h8, Real.sqrt_mul (by positivity), Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 2)]
  exact ⟨0, 2, ⟨0, 0, by norm_num⟩, ⟨2, 0, by norm_num⟩, by ring⟩

/-- And the classification does reproduce `sqrt3_notMem_KR`, the one instance the appendix
needs — the check that the generalization did not drift. -/
example : ¬ KR (√3) := by
  have h : ¬ ((3:ℕ) = 2 ∨ (3:ℕ) = 5 ∨ (3:ℕ) = 10) := by decide
  intro hmem
  exact h ((sqrt_mem_KR_iff_of_squarefree (d := 3) (by decide +kernel) (by norm_num)).mp
    (by rwa [show ((3:ℕ):ℝ) = 3 by norm_num]))

/-! ### Lemma 5's last clause, run on the tetrahedron

`Solids.lean` carries the two vertex orderings the repo disagrees about, and derives the same
verdict from each by hand (`no_pose_tetrahedron` and `no_pose_tetrahedron'`).  That pair is a
*witness* for "any independent triple decides".  Here the general theorem is made to produce
it: the four vertices are given as one family, the two orderings become two triples drawn from
it, and `witness_notMem_of_witness_notMem` carries the verdict from one to the other with no
second computation. -/

/-- The tetrahedron's four vertices, before the `1/√3`.  Indices `0,1,2` are `tetU'`'s triple
and indices `0,3,2` are `tetU`'s — the same solid, the two orderings. -/
def tetU4 : Fin 4 → (Fin 3 → ℝ) := ![![1, 1, 1], ![1, -1, -1], ![-1, 1, -1], ![-1, -1, 1]]

/-- The same family at the atlas radius. -/
noncomputable def tetR : Fin 4 → (Fin 3 → ℝ) := fun a i => (1 / √3) * tetU4 a i

theorem tetU4_mem (a : Fin 4) (k : Fin 3) : tetU4 a k ∈ KRsubfield := by
  have h1 : (1 : ℝ) ∈ KRsubfield := one_mem _
  fin_cases a <;> fin_cases k <;> simp [tetU4, h1]

theorem inner_smul (c : ℝ) (u v : Fin 3 → ℝ) :
    (∑ k, (c * u k) * (c * v k)) = c ^ 2 * ∑ k, u k * v k := by
  simp only [Fin.sum_univ_three]; ring

theorem inv_sqrt3_sq : (1 / √3 : ℝ) ^ 2 = 1 / 3 := by
  have h3 : (√3 : ℝ) ^ 2 = 3 := by rw [pow_two]; exact Solids.sq_sqrt3
  rw [div_pow, one_pow, h3]

theorem tetR_inner (a b : Fin 4) :
    (∑ k, tetR a k * tetR b k) = (1 / 3 : ℝ) * ∑ k, tetU4 a k * tetU4 b k := by
  show (∑ k, ((1 / √3) * tetU4 a k) * ((1 / √3) * tetU4 b k)) = _
  rw [inner_smul, inv_sqrt3_sq]

/-- **The hypothesis is satisfied, and pose-free.**  Every pairwise inner product of the
tetrahedron's vertices lies in `K_ℝ` — the normalizer enters squared, as `1/3`, which is where
the `√3` that bars the *witness* does not reach. -/
theorem tetR_inner_mem (a b : Fin 4) : (∑ k, tetR a k * tetR b k) ∈ KRsubfield := by
  rw [tetR_inner]
  refine mul_mem ?_ (sum_mem fun k _ => mul_mem (tetU4_mem a k) (tetU4_mem b k))
  exact KR.ofQ5 ⟨1 / 3, 0, by push_cast; ring⟩

theorem witness_tetR_012 : witness (tetR 0) (tetR 1) (tetR 2) = (4 / 9 : ℝ) * √3 := by
  have h : witness (tetU4 0) (tetU4 1) (tetU4 2) = 4 := by
    simp [witness, tetU4, Matrix.det_fin_three]; norm_num
  show witness (fun i => (1 / √3) * tetU4 0 i) (fun i => (1 / √3) * tetU4 1 i)
    (fun i => (1 / √3) * tetU4 2 i) = (4 / 9 : ℝ) * √3
  exact Solids.sqrt3_solid_witness h

theorem witness_tetR_032 : witness (tetR 0) (tetR 3) (tetR 2) = (-4 / 9 : ℝ) * √3 := by
  have h : witness (tetU4 0) (tetU4 3) (tetU4 2) = -4 := by
    simp [witness, tetU4, Matrix.det_fin_three]; norm_num
  show witness (fun i => (1 / √3) * tetU4 0 i) (fun i => (1 / √3) * tetU4 3 i)
    (fun i => (1 / √3) * tetU4 2 i) = (-4 / 9 : ℝ) * √3
  exact Solids.sqrt3_solid_witness h

theorem witness_tetR_012_ne : witness (tetR 0) (tetR 1) (tetR 2) ≠ 0 := by
  rw [witness_tetR_012]
  have := Solids.sqrt3_pos
  positivity

theorem witness_tetR_032_ne : witness (tetR 0) (tetR 3) (tetR 2) ≠ 0 := by
  rw [witness_tetR_032]
  have h := Solids.sqrt3_pos
  intro hcon
  rw [mul_eq_zero] at hcon
  rcases hcon with h1 | h2
  · norm_num at h1
  · exact absurd h2 (ne_of_gt h)

/-- One ordering's witness is outside `K_ℝ`, computed directly — this is the hand step. -/
theorem witness_tetR_012_notMem : witness (tetR 0) (tetR 1) (tetR 2) ∉ KRsubfield := by
  rw [witness_tetR_012, mem_KRsubfield]
  exact smul_sqrt3_notMem_KR (KR.ofQ5 ⟨4 / 9, 0, by push_cast; ring⟩) (by norm_num)

/-- **And the other ordering follows with no second computation.**  This is Lemma 5's last
clause doing work: `Solids.lean` had to redo the determinant for `tetU'`; here the verdict
transfers from one triple to the other by the theorem alone. -/
theorem witness_tetR_032_notMem : witness (tetR 0) (tetR 3) (tetR 2) ∉ KRsubfield :=
  witness_notMem_of_witness_notMem tetR_inner_mem witness_tetR_012_ne witness_tetR_032_ne
    witness_tetR_012_notMem

/-- The transfer agrees with `Solids.lean`'s independent hand computation for `tetU`, which is
the check that the general clause did not drift from the concrete pair it generalizes. -/
example : witness (tetR 0) (tetR 3) (tetR 2) = (-4 / 9 : ℝ) * √3 := witness_tetR_032

/-- The clause is an `iff`, and it is not vacuously true by both sides being *in* `K_ℝ`: on
this solid both sides are *out*, which is the branch Appendix D.1 actually uses. -/
example : (witness (tetR 0) (tetR 1) (tetR 2) ∈ KRsubfield ↔
    witness (tetR 0) (tetR 3) (tetR 2) ∈ KRsubfield) ∧
    witness (tetR 0) (tetR 1) (tetR 2) ∉ KRsubfield :=
  ⟨(witness_mem_iff_of_vertex_inner_mem tetR_inner_mem witness_tetR_032_ne
      witness_tetR_012_ne), witness_tetR_012_notMem⟩

/-- The octahedron is the control on the other side: its witness is `1`, which *is* in `K_ℝ`,
so the transfer carries an affirmative verdict too and the clause is not a machine that only
ever says "outside". -/
example : (1 : ℝ) ∈ KRsubfield := one_mem _

end AppendixD1.Sanity
