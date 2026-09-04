/-
# The five solids, in atlas orientation

`AppendixD1.lean` proves Lemmas 5 and 6 and composes them, but every composed statement is
*conditional* on a witness value — `witness v₁ v₂ v₃ = √3`, `witness² = 2/25(5−√5)` — taken on
faith from Table `tab:povm-exactness`.  Nothing there names a Platonic solid.  This file closes
that loop: it writes the atlas vertices down, computes the witness in Lean, and derives the
verdict for each solid unconditionally.

Three things fell out of doing it.

**1. The `√3` composed theorem did not reach any solid.**  `no_pose_of_witness_eq_sqrt3` wants
a witness of `√3` exactly.  No solid has one: the tetrahedron, cube and dodecahedron all give
`±4√3/9` (`sqrt3_solid_witness` applied to `witness_tetU`, `witness_cubeU`, `witness_dodecU`
below, matching the table's `det[v₁ v₂ v₃]` column value for value).  The gap is closed by
`no_pose_of_witness_mem_sqrt3_coset`, which quantifies over the `K_ℝ`-coset and is what being a
*field* buys.  The icosahedral theorem needed no such repair — `2/25(5−√5)` is exactly the
value, because the witness enters that one squared.

**2. The alignment sweep needs no sweep.**  Appendix D.1 argues Corollary 7 vertex by vertex
("*every* vertex of the tetrahedron, cube, and dodecahedron has a coordinate of the form
`p/√3` with `p ∈ {1, σ, τ}`"), and `randomized_implementations.py` runs it that way, 44
vertices.  But the atlas presents each solid as integer-or-golden coordinates over *one*
normalizer, and `vertex_notMem_of_normalizer` below disposes of all 44 in three lines: if the
normalizer is outside `K_ℝ`, every vertex with a nonzero coordinate is too.  The sweep's real
content is a fact about the solid's *radius*, not about its vertices.  The octahedron is the
solid whose normalizer is `1`, which is the asymmetry the appendix turns on, restated.

**3. The two vertex orderings in the repo disagree, and the printed sign rides on that.**
Vertex data here is transcribed from `code/data/povm_atlas.tex` — Table E.1, the atlas a
reader of the thesis actually sees, and the ordering in `code/data/povm_*.npz`.  The suite's
own `symbolic_solids()` (`randomized_core.py`) numbers the tetrahedron, cube and icosahedron
differently.  Same solids, same pose, same vertex *sets* — different numbering.  So the two
orderings give the tetrahedron `−4√3/9` and `+4√3/9` (`witness_tetU` and `witness_tetU'`
below, both proved), and the icosahedron's sign goes the other way.  The verdict never moves —
the minimal-polynomial column is sign-blind — but the caption of Table `tab:povm-exactness`
sends the reader to that atlas for `v₁, v₂, v₃`, so the printed sign has to be reproducible
from it.

That is why the table prints the atlas's own triple.  `det_witness` runs on `atlas_vertices()`
(both in `randomized_core.py`), an `(a,b,c)` column carries the indices, and the caption states
what no choice of triple moves — the `K_ℝ`-coset; the dodecahedron's 960 spanning triples
realize four different minimal polynomials, so even that column is triple-dependent, and the
coset is the only triple-blind invariant.  The assert against `DET_DISPLAY`
(`randomized_fragments.py`) pins the signed value, so a reordering trips a check rather than
quietly rewriting the table.  The two orderings are exhibited side by side below, and the
verdict is the same from either.

For the icosahedron the witness triple is vertices 1, 2, **5**, the first three being coplanar.
-/
import LeanExactness.AppendixD1

namespace AppendixD1.Solids

open Real Matrix AppendixD1

/-! ## The golden ratio and the two normalizers -/

/-- `τ = φ`, the golden ratio.  (Conway–Smith notation, as the thesis uses.) -/
noncomputable def tau : ℝ := (1 + √5) / 2

theorem tau_pos : 0 < tau := by
  have : (0:ℝ) < √5 := Real.sqrt_pos.mpr (by norm_num)
  unfold tau; linarith

theorem KR_tau : KR tau := KR.ofQ5 ⟨1 / 2, 1 / 2, by unfold tau; push_cast; ring⟩

/-- `σ = 1/τ = τ − 1`, the inverse golden ratio. -/
noncomputable def sigma : ℝ := tau - 1

theorem KR_sigma : KR sigma := KR.ofQ5 ⟨-(1 / 2), 1 / 2, by unfold sigma tau; push_cast; ring⟩

theorem sq_sqrt3 : √3 * √3 = 3 := by rw [← sq]; exact Real.sq_sqrt (by norm_num)

theorem sqrt3_pos : (0:ℝ) < √3 := Real.sqrt_pos.mpr (by norm_num)

theorem sqrt3_ne_zero : √3 ≠ 0 := ne_of_gt sqrt3_pos

/-- The icosahedral normalizer, `2 + τ = (5 + √5)/2`. -/
theorem two_add_tau : 2 + tau = (1 / 2 : ℝ) * (5 + √5) := by unfold tau; ring

theorem two_add_tau_pos : 0 < 2 + tau := by linarith [tau_pos]

/-- `√(2+τ)`, what the atlas divides the icosahedron's vertices by. -/
noncomputable def rico : ℝ := √(2 + tau)

theorem rico_pos : 0 < rico := Real.sqrt_pos.mpr two_add_tau_pos

theorem rico_ne_zero : rico ≠ 0 := ne_of_gt rico_pos

theorem rico_sq : rico ^ 2 = 2 + tau := Real.sq_sqrt (le_of_lt two_add_tau_pos)

/-- The icosahedral normalizer is outside `K_ℝ` — Lemma 6's five-fold surd, read as
membership rather than as squareness. -/
theorem rico_notMem_KR : ¬ KR rico := by
  unfold rico
  rw [two_add_tau]
  exact sqrt_icosahedral_normalizer_notMem_KR

/-! ## Scaling: the witness is cubic in the radius -/

/-- Scaling all three vectors scales the witness by the cube.  This is what lets the atlas's
"components scaled by `1/√3`" header row be taken at face value. -/
theorem witness_smul (c : ℝ) (v₁ v₂ v₃ : Fin 3 → ℝ) :
    witness (fun i => c * v₁ i) (fun i => c * v₂ i) (fun i => c * v₃ i)
      = c ^ 3 * witness v₁ v₂ v₃ := by
  simp only [witness, Matrix.det_fin_three, Matrix.of_apply, Matrix.cons_val_zero,
    Matrix.cons_val_one, Matrix.head_cons, Matrix.cons_val_two, Matrix.tail_cons]
  ring

/-! ## The four inexact solids

Each solid's triple is written as `(1/r) •` an integer-or-golden triple, exactly as the atlas
prints it. -/

/-- Tetrahedron, atlas vertices 1, 2, 3, before the `1/√3`. -/
def tetU : Fin 3 → (Fin 3 → ℝ) := ![![1, 1, 1], ![-1, -1, 1], ![-1, 1, -1]]

/-- Cube, atlas vertices 1, 2, 3, before the `1/√3`. -/
def cubeU : Fin 3 → (Fin 3 → ℝ) := ![![-1, -1, -1], ![1, -1, -1], ![1, 1, -1]]

/-- Dodecahedron, atlas vertices 1, 2, 3, before the `1/√3`. -/
def dodecU : Fin 3 → (Fin 3 → ℝ) := ![![1, 1, 1], ![1, 1, -1], ![1, -1, 1]]

/-- The tetrahedron again, in the randomized suite's `symbolic_solids()` ordering
(`code/randomized_core.py`) — vertex 2 is `(1,−1,−1)` there, `(−1,−1,1)` in Table E.1.  This is
the triple whose witness is `+4√3/9`; Table `tab:povm-exactness` prints the atlas ordering's
`−4√3/9`. -/
def tetU' : Fin 3 → (Fin 3 → ℝ) := ![![1, 1, 1], ![1, -1, -1], ![-1, 1, -1]]

theorem witness_tetU : witness (tetU 0) (tetU 1) (tetU 2) = -4 := by
  simp [witness, tetU, Matrix.det_fin_three]; norm_num

/-- The other ordering gives `+4`, hence `+4√3/9`: the sign is a property of the numbering,
not of the solid. -/
theorem witness_tetU' : witness (tetU' 0) (tetU' 1) (tetU' 2) = 4 := by
  simp [witness, tetU', Matrix.det_fin_three]; norm_num

theorem witness_cubeU : witness (cubeU 0) (cubeU 1) (cubeU 2) = -4 := by
  simp [witness, cubeU, Matrix.det_fin_three]; norm_num

theorem witness_dodecU : witness (dodecU 0) (dodecU 1) (dodecU 2) = -4 := by
  simp [witness, dodecU, Matrix.det_fin_three]; norm_num

/-- A solid at radius `1/√3` with integer pre-scaling coordinates has witness `(c/9)√3`.  With
`c = ∓4` this is the `∓4√3/9` of Table `tab:povm-exactness`; the sign is the triple's
handedness, which field membership ignores. -/
theorem sqrt3_solid_witness {u₁ u₂ u₃ : Fin 3 → ℝ} {c : ℝ} (h : witness u₁ u₂ u₃ = c) :
    witness (fun i => (1 / √3) * u₁ i) (fun i => (1 / √3) * u₂ i) (fun i => (1 / √3) * u₃ i)
      = (c / 9) * √3 := by
  rw [witness_smul, h]
  have h3 : √3 ≠ 0 := sqrt3_ne_zero
  field_simp
  linear_combination (-(c * √3 ^ 2 + 3 * c)) * sq_sqrt3

/-- Icosahedron, atlas vertices 1, 2, **5**, before the `1/√(2+τ)`.  Vertices 1, 2, 3 are
coplanar (all have `z = 0`), so the table's "first spanning triple" skips to vertex 5. -/
noncomputable def icoU : Fin 3 → (Fin 3 → ℝ) := ![![tau, 1, 0], ![-tau, 1, 0], ![0, tau, 1]]

theorem witness_icoU : witness (icoU 0) (icoU 1) (icoU 2) = 2 * tau := by
  simp [witness, icoU, Matrix.det_fin_three]; ring

theorem tau_sq : tau ^ 2 = tau + 1 := by
  unfold tau
  linear_combination (1 / 4 : ℝ) * sq_sqrt5

/-- `(2+τ)³ = 25 + 10√5`. -/
theorem two_add_tau_cube : (2 + tau) ^ 3 = 25 + 10 * √5 := by
  unfold tau
  linear_combination ((√5 + 15) / 8) * sq_sqrt5

theorem rico_pow_six : rico ^ 6 = 25 + 10 * √5 := by
  rw [show (6 : ℕ) = 2 * 3 from rfl, pow_mul, rico_sq, two_add_tau_cube]

theorem four_tau_sq : 4 * tau ^ 2 = 6 + 2 * √5 := by unfold tau; linear_combination sq_sqrt5

/-- **The icosahedron's Gram determinant.**  `witness² = 4τ²/(2+τ)³ = 2/25(5−√5)` — exactly
the number `icosahedral_witness_not_sq` rejects, and exactly the square of the table's
`(1/5)√(10−2√5)`. -/
theorem witness_ico_sq :
    witness (fun i => (1 / rico) * icoU 0 i) (fun i => (1 / rico) * icoU 1 i)
        (fun i => (1 / rico) * icoU 2 i) ^ 2 = (2 / 25 : ℝ) * (5 - √5) := by
  rw [witness_smul, witness_icoU]
  have hden : (25 : ℝ) + 10 * √5 ≠ 0 := by positivity
  have hsplit : ((1 / rico) ^ 3 * (2 * tau)) ^ 2 = (4 * tau ^ 2) / rico ^ 6 := by
    field_simp [rico_ne_zero]; ring
  rw [hsplit, rico_pow_six, four_tau_sq, div_eq_iff hden]
  linear_combination (4 / 5 : ℝ) * sq_sqrt5

/-! ## The verdicts

Each is unconditional: no rotation whatsoever carries the solid's spanning triple into
`K_ℝ³`, so by Corollary 3 no protocol over any gate set of this thesis realizes the POVM. -/

/-- The shared verdict for the three `√3` solids, stated once over the pre-scaling witness.
Any nonzero `c` will do: the solid is barred by the *field*, not by the number `4`. -/
theorem no_pose_sqrt3_solid {u₁ u₂ u₃ : Fin 3 → ℝ} {c : ℚ} (hc : c ≠ 0)
    (hw : witness u₁ u₂ u₃ = (c : ℝ)) {R : Matrix (Fin 3) (Fin 3) ℝ} (hR : Rᵀ * R = 1)
    (h₁ : ∀ j, KR (R.mulVec (fun i => (1 / √3) * u₁ i) j))
    (h₂ : ∀ j, KR (R.mulVec (fun i => (1 / √3) * u₂ i) j))
    (h₃ : ∀ j, KR (R.mulVec (fun i => (1 / √3) * u₃ i) j)) : False :=
  no_pose_of_witness_mem_sqrt3_coset (q := ((c : ℝ)) / 9)
    (KR.ofQ5 ⟨c / 9, 0, by push_cast; ring⟩)
    (by simpa using hc) (sqrt3_solid_witness hw) hR h₁ h₂ h₃

theorem no_pose_tetrahedron {R : Matrix (Fin 3) (Fin 3) ℝ} (hR : Rᵀ * R = 1)
    (h : ∀ k j, KR (R.mulVec (fun i => (1 / √3) * tetU k i) j)) : False :=
  no_pose_sqrt3_solid (c := -4) (by norm_num) (by rw [witness_tetU]; norm_num)
    hR (h 0) (h 1) (h 2)

/-- The same solid in the *other* repo ordering, witness `+4√3/9`.  Same verdict — which is
Lemma 5's "any independent triple decides", made concrete on the one disagreement the repo
actually contains. -/
theorem no_pose_tetrahedron' {R : Matrix (Fin 3) (Fin 3) ℝ} (hR : Rᵀ * R = 1)
    (h : ∀ k j, KR (R.mulVec (fun i => (1 / √3) * tetU' k i) j)) : False :=
  no_pose_sqrt3_solid (c := 4) (by norm_num) (by rw [witness_tetU']; norm_num)
    hR (h 0) (h 1) (h 2)

theorem no_pose_cube {R : Matrix (Fin 3) (Fin 3) ℝ} (hR : Rᵀ * R = 1)
    (h : ∀ k j, KR (R.mulVec (fun i => (1 / √3) * cubeU k i) j)) : False :=
  no_pose_sqrt3_solid (c := -4) (by norm_num) (by rw [witness_cubeU]; norm_num)
    hR (h 0) (h 1) (h 2)

theorem no_pose_dodecahedron {R : Matrix (Fin 3) (Fin 3) ℝ} (hR : Rᵀ * R = 1)
    (h : ∀ k j, KR (R.mulVec (fun i => (1 / √3) * dodecU k i) j)) : False :=
  no_pose_sqrt3_solid (c := -4) (by norm_num) (by rw [witness_dodecU]; norm_num)
    hR (h 0) (h 1) (h 2)

theorem no_pose_icosahedron {R : Matrix (Fin 3) (Fin 3) ℝ} (hR : Rᵀ * R = 1)
    (h : ∀ k j, KR (R.mulVec (fun i => (1 / rico) * icoU k i) j)) : False :=
  no_pose_of_gram_eq_icosahedral witness_ico_sq hR (h 0) (h 1) (h 2)

/-! ## Corollary 7 (alignment): the sweep is about the radius

Appendix D.1 runs this vertex by vertex, and the sweep does have to be run — a single direction
in isolation has no rotation invariant to fail.  But it need not be run *vertex by
vertex*.  Every vertex of every solid in the atlas is `(a, b, c)/r` with `a, b, c ∈ K_ℝ` not
all zero, and the one lemma below kills all of them together. -/

/-- **The alignment sweep, in one lemma.**  A vertex given as `K_ℝ` coordinates over a
normalizer `r ∉ K_ℝ` cannot lie in `K_ℝ³`, whichever of its coordinates is the nonzero one. -/
theorem vertex_notMem_of_normalizer {r : ℝ} (hr0 : r ≠ 0) (hr : ¬ KR r) {a : Fin 3 → ℝ}
    (ha : ∀ i, KR (a i)) (hne : ∃ i, a i ≠ 0) : ¬ (∀ i, KR (a i / r)) := by
  obtain ⟨i, hi⟩ := hne
  exact fun h => div_notMem_KR (ha i) hi hr0 hr (h i)

/-- Tetrahedron, cube, dodecahedron: normalizer `√3`. -/
theorem no_alignment_sqrt3 {a : Fin 3 → ℝ} (ha : ∀ i, KR (a i)) (hne : ∃ i, a i ≠ 0) :
    ¬ (∀ i, KR (a i / √3)) :=
  vertex_notMem_of_normalizer sqrt3_ne_zero sqrt3_notMem_KR ha hne

/-- Icosahedron: normalizer `√(2+τ)`. -/
theorem no_alignment_ico {a : Fin 3 → ℝ} (ha : ∀ i, KR (a i)) (hne : ∃ i, a i ≠ 0) :
    ¬ (∀ i, KR (a i / rico)) :=
  vertex_notMem_of_normalizer rico_ne_zero rico_notMem_KR ha hne

/-- The atlas coordinate alphabet: every coordinate of every vertex of the four inexact solids
is one of `0, ±1, ±σ, ±τ`, and each of those lies in `K_ℝ`.  So `no_alignment_sqrt3` and
`no_alignment_ico` between them bar all 4 + 8 + 20 + 12 = 44 vertices. -/
theorem KR_atlas_alphabet :
    KR 0 ∧ KR 1 ∧ KR (-1 : ℝ) ∧ KR sigma ∧ KR (-sigma) ∧ KR tau ∧ KR (-tau) :=
  ⟨KR.ofQ5 ⟨0, 0, by norm_num⟩, KR.ofQ5 ⟨1, 0, by norm_num⟩,
    KR.ofQ5 ⟨-1, 0, by push_cast; ring⟩, KR_sigma, KR_sigma.neg, KR_tau, KR_tau.neg⟩

end AppendixD1.Solids
