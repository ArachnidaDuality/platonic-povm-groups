/-
# Appendix D.1, Lemmas 5 and 6, formalized

This file formalizes the two *field-arithmetic* statements of Appendix D.1 — the ones that
quantify over all poses and all square roots, which SymPy cannot check because it can only
evaluate instances.

  * **Lemma 5 (rotation-proof witness).**  `det [v_a v_b v_c]` is unmoved by rotation, so a
    witness outside `K_ℝ` in *one* pose bars *every* pose; and passing between spanning triples
    multiplies the Gram determinant by a square, so one triple decides for the solid.

  * **Lemma 6 (membership in `K_ℝ`).**  For squarefree `d > 1`, `√d ∈ K_ℝ = ℚ(√2,√5)` if and
    only if `d ∈ {2,5,10}` — in particular `√3 ∉ K_ℝ`; and neither of the two five-fold surds —
    the icosahedral witness `2/25(5−√5)` and the icosahedral normalizer `(5+√5)/2` — is a
    square in `K_ℝ`.

Deliberately NOT formalized: Lemma 2 (branch decomposition), Theorem 1, and the *protocol*
half of Theorem 4 (weights).  Those quantify over protocols, and formalizing them means first
defining the protocol model, which is where the residual mathematical risk actually lives.  A
Lean proof there would certify consequences of a definition rather than the definition's
adequacy.  Theorem 4's *arithmetic* half — `ℛ ∩ ℚ = ℤ[½]`, hence `V` a power of two — needs no
protocol model and is in `Weights.lean`.

Everything below is elementary and self-contained: the only mathlib inputs are basic matrix
determinant lemmas and the irrationality of `√n` for non-square `n`.

Two companion files carry this further.  `Solids.lean` instantiates the composed statements at
the actual atlas vertices of the five solids — without which everything here is conditional on
a witness value taken on faith from Table `tab:povm-exactness`.  `Weights.lean` formalizes the
second obstruction, the one on weights.

## Two steps that carry the statements to full strength

**Lemma 5's last clause** — "its square class is the same for every independent triple, so any
one of them decides" — needs more than the algebraic identity `gram_det_change_of_basis`: it
needs the change of basis `C` between two spanning triples to have entries in `K_ℝ`.  **Part
IV** proves that.  `C = M Γ⁻¹` is assembled from inner products alone, so the hypothesis is
*pose-free*; the inverse is what needs `K_ℝ` closed under division, which is why Part IV is
stated over a `Subfield` where Part I needs only a `Subring`.  The formalized results decide
*any* independent triple, not a given one.

**Lemma 6's classification** — the thesis states an `iff` over all squarefree `d > 1`, where
`sqrt_notMem_KR` gives only the criterion.  `sqrt_mem_KR_iff_of_squarefree` is the `iff`; the
ingredient it needs is one fact about ℕ, `eq_of_squarefree_of_isSquare_mul`.
-/
import Mathlib

namespace AppendixD1

open Real Matrix

/-! ## Part I — Lemma 5: the witness is rotation-proof -/

section Witness

variable {K : Subring ℝ}

/-- The determinant of a `3×3` real matrix with all entries in a subring `K` lies in `K`:
a determinant is sums of products of entries, and a subring is closed under both. -/
theorem det_mem_of_entries_mem {M : Matrix (Fin 3) (Fin 3) ℝ} (h : ∀ i j, M i j ∈ K) :
    M.det ∈ K := by
  rw [Matrix.det_fin_three]
  repeat' first
    | apply sub_mem
    | apply add_mem
    | apply mul_mem
    | apply h

/-- An orthogonal matrix has determinant `±1`. -/
theorem det_eq_one_or_neg_one {R : Matrix (Fin 3) (Fin 3) ℝ} (hR : Rᵀ * R = 1) :
    R.det = 1 ∨ R.det = -1 := by
  have h : R.det * R.det = 1 := by
    have h' := congrArg Matrix.det hR
    rwa [Matrix.det_mul, Matrix.det_transpose, Matrix.det_one] at h'
  exact mul_self_eq_one_iff.mp h

/-- **Lemma 5, pose-independence (matrix form).**  If *some* orthogonal `R` carries `B` to a
matrix with all entries in `K`, then `det B` already lies in `K` — in the pose at hand. -/
theorem det_mem_of_pose {B R : Matrix (Fin 3) (Fin 3) ℝ} (hR : Rᵀ * R = 1)
    (h : ∀ i j, (B * Rᵀ) i j ∈ K) : B.det ∈ K := by
  have hBR : (B * Rᵀ).det ∈ K := det_mem_of_entries_mem h
  rw [Matrix.det_mul, Matrix.det_transpose] at hBR
  rcases det_eq_one_or_neg_one hR with h1 | h1 <;> rw [h1] at hBR
  · simpa using hBR
  · simpa using neg_mem hBR

/-- The witness of three vectors: the determinant of the matrix carrying them as rows. -/
def witness (v₁ v₂ v₃ : Fin 3 → ℝ) : ℝ := (Matrix.of ![v₁, v₂, v₃]).det

/-- Rotating every vector is right multiplication by `Rᵀ` on the matrix of rows. -/
theorem rows_mulVec (R : Matrix (Fin 3) (Fin 3) ℝ) (v₁ v₂ v₃ : Fin 3 → ℝ) :
    (Matrix.of ![R.mulVec v₁, R.mulVec v₂, R.mulVec v₃]) = (Matrix.of ![v₁, v₂, v₃]) * Rᵀ := by
  ext i j
  fin_cases i <;>
    simp [Matrix.mul_apply, Matrix.mulVec, dotProduct, Matrix.transpose_apply, mul_comm]

/-- **Lemma 5, pose-independence (vector form).**  If some rotation carries `v₁,v₂,v₃` into
`K³`, then the witness determinant computed in the *original* pose already lies in `K`.
Contrapositive: a witness outside `K` in one pose bars every pose. -/
theorem witness_mem_of_exists_pose {v₁ v₂ v₃ : Fin 3 → ℝ} {R : Matrix (Fin 3) (Fin 3) ℝ}
    (hR : Rᵀ * R = 1) (h₁ : ∀ j, R.mulVec v₁ j ∈ K) (h₂ : ∀ j, R.mulVec v₂ j ∈ K)
    (h₃ : ∀ j, R.mulVec v₃ j ∈ K) : witness v₁ v₂ v₃ ∈ K := by
  apply det_mem_of_pose hR
  intro i j
  rw [← rows_mulVec]
  fin_cases i
  · simpa using h₁ j
  · simpa using h₂ j
  · simpa using h₃ j

/-- The Gram matrix of three vectors is `A Aᵀ` for `A` the matrix of rows. -/
theorem gram_apply (v₁ v₂ v₃ : Fin 3 → ℝ) (i j : Fin 3) :
    ((Matrix.of ![v₁, v₂, v₃]) * (Matrix.of ![v₁, v₂, v₃])ᵀ) i j =
      ∑ k, (![v₁, v₂, v₃] i) k * (![v₁, v₂, v₃] j) k := by
  simp [Matrix.mul_apply, Matrix.transpose_apply]

/-- **Lemma 5, the witness squares to the Gram determinant.** -/
theorem gram_det_eq_witness_sq (v₁ v₂ v₃ : Fin 3 → ℝ) :
    ((Matrix.of ![v₁, v₂, v₃]) * (Matrix.of ![v₁, v₂, v₃])ᵀ).det = witness v₁ v₂ v₃ ^ 2 := by
  rw [Matrix.det_mul, Matrix.det_transpose, witness, sq]

/-- **Lemma 5, the square class.**  Changing spanning triple by `C` multiplies the Gram
determinant by `(det C)²` — the same square class, hence the same verdict. -/
theorem gram_det_change_of_basis (B C : Matrix (Fin 3) (Fin 3) ℝ) :
    ((B * C)ᵀ * (B * C)).det = (Bᵀ * B).det * C.det ^ 2 := by
  simp only [Matrix.transpose_mul, Matrix.det_mul, Matrix.det_transpose]
  ring

end Witness

/-! ## Part II — Lemma 6: what `K_ℝ = ℚ(√2,√5)` holds

The argument runs in the tower `ℚ ⊂ ℚ(√5) ⊂ K_ℝ`, which is why `Q5` below is developed first
as a field in its own right.  Nothing here uses Galois theory: everything is done by writing
`β = p + q√2` over the base and reading off `pq = 0`, the route the thesis's printed proofs
run too. -/

/-- `Q5 x` says `x ∈ ℚ(√5)`. -/
def Q5 (x : ℝ) : Prop := ∃ a b : ℚ, x = a + b * √5

theorem sq_sqrt5 : √5 * √5 = 5 := by rw [← sq]; exact Real.sq_sqrt (by norm_num)

theorem sq_sqrt2 : √2 * √2 = 2 := by rw [← sq]; exact Real.sq_sqrt (by norm_num)

theorem sqrt10_eq : √10 = √2 * √5 := by
  rw [show (10 : ℝ) = 2 * 5 by norm_num, Real.sqrt_mul (by norm_num)]

theorem irrational_sqrt5 : Irrational (√5) := by norm_num

/-- No rational squares to `5`; the workhorse behind every non-square check below. -/
theorem rat_sq_ne_five (x : ℚ) : x ^ 2 ≠ 5 := by
  intro h
  refine irrational_sqrt5 ⟨|x|, ?_⟩
  push_cast
  rw [← Real.sqrt_sq_eq_abs]
  congr 1
  exact_mod_cast h

/-! ### `ℚ(√5)` is a field -/

theorem Q5.ofRat (q : ℚ) : Q5 q := ⟨q, 0, by simp⟩

theorem Q5.two : Q5 (2 : ℝ) := ⟨2, 0, by norm_num⟩

theorem Q5.three : Q5 (3 : ℝ) := ⟨3, 0, by norm_num⟩

theorem Q5.add {x y : ℝ} (hx : Q5 x) (hy : Q5 y) : Q5 (x + y) := by
  obtain ⟨a, b, rfl⟩ := hx; obtain ⟨c, d, rfl⟩ := hy
  exact ⟨a + c, b + d, by push_cast; ring⟩

theorem Q5.neg {x : ℝ} (hx : Q5 x) : Q5 (-x) := by
  obtain ⟨a, b, rfl⟩ := hx
  exact ⟨-a, -b, by push_cast; ring⟩

theorem Q5.sub {x y : ℝ} (hx : Q5 x) (hy : Q5 y) : Q5 (x - y) := by
  simpa [sub_eq_add_neg] using hx.add hy.neg

theorem Q5.mul {x y : ℝ} (hx : Q5 x) (hy : Q5 y) : Q5 (x * y) := by
  obtain ⟨a, b, rfl⟩ := hx; obtain ⟨c, d, rfl⟩ := hy
  refine ⟨a * c + 5 * (b * d), a * d + b * c, ?_⟩
  push_cast
  linear_combination ((b : ℝ) * d) * sq_sqrt5

/-- Coefficients over `ℚ(√5)` are unique — the one place `√5 ∉ ℚ` is used structurally. -/
theorem Q5.ext {a b c d : ℚ} (h : (a : ℝ) + b * √5 = (c : ℝ) + d * √5) : a = c ∧ b = d := by
  have key : (a : ℝ) - c = ((d : ℝ) - b) * √5 := by linarith
  have hbd : b = d := by
    by_contra hbd
    have hne : ((d : ℝ) - b) ≠ 0 := sub_ne_zero.mpr fun hh => hbd (by exact_mod_cast hh.symm)
    refine irrational_sqrt5 ⟨(a - c) / (d - b), ?_⟩
    push_cast
    rw [div_eq_iff hne]
    linear_combination key
  have hac : (a : ℝ) = c := by
    rw [hbd] at key
    simp only [sub_self, zero_mul] at key
    linarith
  exact ⟨by exact_mod_cast hac, hbd⟩

theorem Q5.inv {x : ℝ} (hx : Q5 x) : Q5 x⁻¹ := by
  obtain ⟨a, b, rfl⟩ := hx
  by_cases hx0 : (a : ℝ) + b * √5 = 0
  · rw [hx0]; simpa using Q5.ofRat 0
  · have hN : (a ^ 2 - 5 * b ^ 2 : ℚ) ≠ 0 := by
      intro hN
      by_cases hb : b = 0
      · subst hb
        norm_num at hN
        exact hx0 (by rw [hN]; norm_num)
      · exact rat_sq_ne_five (a / b) (by field_simp; linarith [hN])
    have hNR : ((a : ℝ) ^ 2 - 5 * (b : ℝ) ^ 2) ≠ 0 := by exact_mod_cast hN
    have key : ((a : ℝ) + b * √5) * ((a : ℝ) - b * √5) = (a : ℝ) ^ 2 - 5 * (b : ℝ) ^ 2 := by
      linear_combination (-(b : ℝ) ^ 2) * sq_sqrt5
    have hmul : ((a : ℝ) + b * √5) *
        (((a : ℝ) - b * √5) / ((a : ℝ) ^ 2 - 5 * (b : ℝ) ^ 2)) = 1 := by
      rw [← mul_div_assoc, key, div_self hNR]
    rw [inv_eq_of_mul_eq_one_right hmul]
    refine ⟨a / (a ^ 2 - 5 * b ^ 2), -b / (a ^ 2 - 5 * b ^ 2), ?_⟩
    push_cast
    ring

theorem Q5.div {x y : ℝ} (hx : Q5 x) (hy : Q5 y) : Q5 (x / y) := by
  rw [div_eq_mul_inv]; exact hx.mul hy.inv

/-! ### What `ℚ(√5)` does not hold -/

/-- If `√m`, `√n` and `√(mn)` are all irrational then `√n ∉ ℚ(√m)`: squaring `a + b√m` and
comparing rational parts forces `ab = 0`, and each surviving branch names a rational square root
of `n` or of `mn`. -/
theorem sq_ne_of_irrational {m n : ℕ} (hm : Irrational (√m)) (hn : Irrational (√n))
    (hmn : Irrational (√(m * n))) (a b : ℚ) : ((a : ℝ) + b * √m) ^ 2 ≠ n := by
  intro h
  have hsq : √(m : ℝ) * √(m : ℝ) = m := Real.mul_self_sqrt (by positivity)
  have hexp : (a : ℝ) ^ 2 + (b : ℝ) ^ 2 * m + 2 * a * b * √(m : ℝ) = n := by
    linear_combination h - (b : ℝ) ^ 2 * hsq
  by_cases hab : (a : ℝ) * b = 0
  · rcases mul_eq_zero.mp hab with ha | hb
    · -- `a = 0`: then `b²m = n`, so `(bm)² = mn`.
      have hb2 : (b : ℝ) ^ 2 * m = n := by
        rw [ha] at hexp; linear_combination hexp
      refine hmn ⟨|b * (m : ℚ)|, ?_⟩
      push_cast
      rw [show ((m : ℝ) * (n : ℝ)) = ((b : ℝ) * (m : ℝ)) ^ 2 from by
        linear_combination (-(m : ℝ)) * hb2, Real.sqrt_sq_eq_abs]
    · -- `b = 0`: then `a² = n`.
      have ha2 : (a : ℝ) ^ 2 = n := by
        rw [hb] at hexp; linear_combination hexp
      refine hn ⟨|a|, ?_⟩
      push_cast
      rw [← ha2, Real.sqrt_sq_eq_abs]
  · -- Otherwise `√m` is rational.
    have ha : (a : ℝ) ≠ 0 := fun h => hab (by rw [h]; ring)
    have hb : (b : ℝ) ≠ 0 := fun h => hab (by rw [h]; ring)
    have hd : (2 : ℝ) * a * b ≠ 0 := mul_ne_zero (mul_ne_zero two_ne_zero ha) hb
    refine hm ⟨((n : ℚ) - a ^ 2 - b ^ 2 * (m : ℚ)) / (2 * a * b), ?_⟩
    push_cast
    rw [div_eq_iff hd]
    linear_combination -hexp

/-- Package: `√n ∉ ℚ(√5)` whenever `√n` and `√(5n)` are irrational. -/
theorem notMem_Q5 {n : ℕ} (hn : Irrational (√n)) (h5n : Irrational (√(5 * n))) :
    ¬ Q5 (√n) := by
  rintro ⟨a, b, hab⟩
  have h5 : Irrational (√((5 : ℕ) : ℝ)) := by
    rw [show (((5 : ℕ)) : ℝ) = 5 by norm_num]; exact irrational_sqrt5
  have h5n' : Irrational (√(((5 : ℕ) : ℝ) * (n : ℝ))) := by
    rw [show (((5 : ℕ) : ℝ) * (n : ℝ)) = 5 * (n : ℝ) by norm_num]
    exact h5n
  refine sq_ne_of_irrational (m := 5) (n := n) h5 hn h5n' a b ?_
  rw [show (((5 : ℕ)) : ℝ) = 5 by norm_num, ← hab]
  exact Real.sq_sqrt (by positivity)

theorem sqrt2_notMem_Q5 : ¬ Q5 (√2) := by
  have := notMem_Q5 (n := 2) (by norm_num) (by norm_num)
  rwa [show ((2 : ℕ) : ℝ) = 2 by norm_num] at this

theorem sqrt3_notMem_Q5 : ¬ Q5 (√3) := by
  have := notMem_Q5 (n := 3) (by norm_num) (by norm_num)
  rwa [show ((3 : ℕ) : ℝ) = 3 by norm_num] at this

theorem sqrt6_notMem_Q5 : ¬ Q5 (√6) := by
  have := notMem_Q5 (n := 6) (by norm_num) (by norm_num)
  rwa [show ((6 : ℕ) : ℝ) = 6 by norm_num] at this

/-! ### `K_ℝ = ℚ(√2,√5)` -/

/-- `KR x` says `x ∈ ℚ(√2,√5)`, presented over the base `ℚ(√5)`. -/
def KR (x : ℝ) : Prop := ∃ p q : ℝ, Q5 p ∧ Q5 q ∧ x = p + q * √2

/-- The presentation in the thesis's basis: `a + b√2 + c√5 + d√10`. -/
theorem KR_iff (x : ℝ) : KR x ↔ ∃ a b c d : ℚ, x = a + b * √2 + c * √5 + d * √10 := by
  constructor
  · rintro ⟨p, q, ⟨a, c, rfl⟩, ⟨b, d, rfl⟩, rfl⟩
    exact ⟨a, b, c, d, by rw [sqrt10_eq]; ring⟩
  · rintro ⟨a, b, c, d, rfl⟩
    exact ⟨(a : ℝ) + c * √5, (b : ℝ) + d * √5, ⟨a, c, rfl⟩, ⟨b, d, rfl⟩,
      by rw [sqrt10_eq]; ring⟩

/-- **Lemma 6, first half.**  `√3 ∉ K_ℝ`.  This is what bars the tetrahedron, cube and
dodecahedron, whose vertices all carry a coordinate over `√3`. -/
theorem sqrt3_notMem_KR : ¬ KR (√3) := by
  rintro ⟨p, q, hp, hq, h⟩
  have expand : (p + q * √2) ^ 2 = 3 := by rw [← h]; exact Real.sq_sqrt (by norm_num)
  have key : p ^ 2 + 2 * q ^ 2 + 2 * p * q * √2 = 3 := by
    linear_combination expand - q ^ 2 * sq_sqrt2
  by_cases hpq : p * q = 0
  · rcases mul_eq_zero.mp hpq with h0 | h0
    · -- `p = 0`: then `2q² = 3`, so `(2q)² = 6` and `√6 ∈ ℚ(√5)`.
      refine sqrt6_notMem_Q5 ?_
      have h6 : (2 * q) ^ 2 = 6 := by rw [h0] at key; linear_combination 2 * key
      have : √6 = |2 * q| := by rw [← h6, Real.sqrt_sq_eq_abs]
      rw [this]
      rcases abs_choice (2 * q) with hc | hc <;> rw [hc]
      · exact (Q5.ofRat 2).mul hq
      · exact ((Q5.ofRat 2).mul hq).neg
    · -- `q = 0`: then `√3 ∈ ℚ(√5)`.
      exact sqrt3_notMem_Q5 (by rw [h, h0]; simpa using hp)
  · -- Otherwise `√2 ∈ ℚ(√5)`.
    refine sqrt2_notMem_Q5 ?_
    have hd : (2 : ℝ) * p * q ≠ 0 := by
      intro hcon
      exact hpq (by linear_combination hcon / 2)
    have hs : √2 = (3 - p ^ 2 - 2 * q ^ 2) / (2 * p * q) := by
      rw [eq_div_iff hd]; linear_combination key
    rw [hs]
    have h3 : Q5 (3 : ℝ) := Q5.three
    have h2 : Q5 (2 : ℝ) := Q5.two
    have hp2 : Q5 (p ^ 2) := by rw [sq]; exact hp.mul hp
    have hq2 : Q5 (q ^ 2) := by rw [sq]; exact hq.mul hq
    exact ((h3.sub hp2).sub (h2.mul hq2)).div ((h2.mul hp).mul hq)

/-! ### Lemma 6's first sentence, in general

`sqrt3_notMem_KR` is the one instance Appendix D.1 needs.  But the lemma as stated in the
thesis quantifies over *all* squarefree `d` — "for squarefree `d > 1`, `√d ∈ K_ℝ` if and only
if `d ∈ {2,5,10}`" — proved there by the same two-floor coefficient comparison as here.  Here
is the statement as a criterion: `√n` escapes `K_ℝ` unless one of `n`, `2n`, `5n`, `10n` is a
perfect square.  Those four numbers are `n` times the four squarefree parts `K_ℝ` can supply —
its three quadratic subfields plus `ℚ`, in another guise; and each hypothesis is decidable, so
any concrete `d` follows by `decide`. -/

theorem rat_sq_ne_natCast {n : ℕ} (h : ¬ IsSquare n) (x : ℚ) : x ^ 2 ≠ (n : ℚ) := by
  intro hx
  refine (irrational_sqrt_natCast_iff.mpr h) ⟨|x|, ?_⟩
  push_cast
  rw [← Real.sqrt_sq_eq_abs]
  congr 1
  exact_mod_cast hx

/-- A square in `ℚ(√5)` that happens to be rational is either a rational square or `5` times
one.  This is `descent`'s shadow one floor down, and it is what makes the criterion below run
in the tower rather than by counting subfields. -/
theorem Q5.sq_eq_rat {q : ℝ} (hq : Q5 q) {r : ℚ} (h : q ^ 2 = (r : ℝ)) :
    (∃ a : ℚ, a ^ 2 = r) ∨ (∃ b : ℚ, 5 * b ^ 2 = r) := by
  obtain ⟨a, b, rfl⟩ := hq
  have hexp : ((a ^ 2 + 5 * b ^ 2 : ℚ) : ℝ) + ((2 * a * b : ℚ) : ℝ) * √5
      = ((r : ℚ) : ℝ) + ((0 : ℚ) : ℝ) * √5 := by
    push_cast
    linear_combination h - (b : ℝ) ^ 2 * sq_sqrt5
  obtain ⟨hr, hab⟩ := Q5.ext hexp
  have hzero : a = 0 ∨ b = 0 := by
    rcases mul_eq_zero.mp (by linarith [hab] : (2 * a) * b = 0) with h' | h'
    · exact Or.inl (by linarith [h'])
    · exact Or.inr h'
  rcases hzero with h0 | h0
  · exact Or.inr ⟨b, by rw [h0] at hr; linarith [hr]⟩
  · exact Or.inl ⟨a, by rw [h0] at hr; linarith [hr]⟩

/-- **Lemma 6's first sentence, as a general criterion.**  `√n ∉ K_ℝ` unless one of `n`, `2n`,
`5n`, `10n` is a perfect square — equivalently, unless `n`'s squarefree part is `1`, `2`, `5`
or `10`. -/
theorem sqrt_notMem_KR {n : ℕ} (h1 : ¬ IsSquare n) (h2 : ¬ IsSquare (2 * n))
    (h5 : ¬ IsSquare (5 * n)) (h10 : ¬ IsSquare (10 * n)) : ¬ KR (√(n : ℝ)) := by
  rintro ⟨p, q, hp, hq, h⟩
  have expand : (p + q * √2) ^ 2 = (n : ℝ) := by rw [← h]; exact Real.sq_sqrt (by positivity)
  have key : p ^ 2 + 2 * q ^ 2 + 2 * p * q * √2 = (n : ℝ) := by
    linear_combination expand - q ^ 2 * sq_sqrt2
  by_cases hpq : p * q = 0
  · rcases mul_eq_zero.mp hpq with h0 | h0
    · -- `p = 0`: then `n = 2q²`, and `q` lives in `ℚ(√5)`.
      have hq2 : q ^ 2 = (((n : ℚ) / 2 : ℚ) : ℝ) := by
        rw [h0] at key; push_cast; linarith [key]
      rcases Q5.sq_eq_rat hq hq2 with ⟨a, ha⟩ | ⟨b, hb⟩
      · exact rat_sq_ne_natCast h2 (2 * a) (by push_cast; linarith [ha])
      · exact rat_sq_ne_natCast h10 (10 * b) (by push_cast; linarith [hb])
    · -- `q = 0`: then `n = p²`.
      have hp2 : p ^ 2 = (((n : ℚ)) : ℝ) := by
        rw [h0] at key; push_cast; linarith [key]
      rcases Q5.sq_eq_rat hp hp2 with ⟨a, ha⟩ | ⟨b, hb⟩
      · exact rat_sq_ne_natCast h1 a ha
      · exact rat_sq_ne_natCast h5 (5 * b) (by push_cast; linarith [hb])
  · -- otherwise `√2` would be in `ℚ(√5)`.
    refine sqrt2_notMem_Q5 ?_
    have hd : (2 : ℝ) * p * q ≠ 0 := by
      intro hcon; exact hpq (by linear_combination hcon / 2)
    have hs : √2 = ((n : ℝ) - p ^ 2 - 2 * q ^ 2) / (2 * p * q) := by
      rw [eq_div_iff hd]; linear_combination key
    rw [hs]
    have hn : Q5 ((n : ℝ)) := ⟨(n : ℚ), 0, by push_cast; ring⟩
    have hp2 : Q5 (p ^ 2) := by rw [sq]; exact hp.mul hp
    have hq2 : Q5 (q ^ 2) := by rw [sq]; exact hq.mul hq
    exact ((hn.sub hp2).sub (Q5.two.mul hq2)).div ((Q5.two.mul hp).mul hq)

/-- The criterion reproduces the hand proof of `sqrt3_notMem_KR`, which is the check that the
generalization did not drift. -/
theorem sqrt3_notMem_KR' : ¬ KR (√3) := by
  have := sqrt_notMem_KR (n := 3) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel)
  rwa [show ((3:ℕ):ℝ) = 3 by norm_num] at this

/-- And it settles the rest of the small squarefree numbers, which the hand proof does not:
`√6`, `√7`, `√15`, `√30` are all outside `K_ℝ`, while `√2`, `√5`, `√10` are inside
(`Sanity.lean`).  That is Lemma 6's "if and only if", on the range where it can be checked. -/
theorem sqrt6_notMem_KR : ¬ KR (√6) := by
  have := sqrt_notMem_KR (n := 6) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel)
  rwa [show ((6:ℕ):ℝ) = 6 by norm_num] at this

theorem sqrt7_notMem_KR : ¬ KR (√7) := by
  have := sqrt_notMem_KR (n := 7) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel)
  rwa [show ((7:ℕ):ℝ) = 7 by norm_num] at this

theorem sqrt15_notMem_KR : ¬ KR (√15) := by
  have := sqrt_notMem_KR (n := 15) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel)
  rwa [show ((15:ℕ):ℝ) = 15 by norm_num] at this

theorem sqrt30_notMem_KR : ¬ KR (√30) := by
  have := sqrt_notMem_KR (n := 30) (by decide +kernel) (by decide +kernel)
    (by decide +kernel) (by decide +kernel)
  rwa [show ((30:ℕ):ℝ) = 30 by norm_num] at this

/-! ### Lemma 6's first sentence, in full

The criterion above is not yet the thesis's sentence.  Appendix D.1 states a *classification* —
"for squarefree `d > 1`, `√d ∈ K_ℝ` if and only if `d ∈ {2,5,10}`" — and getting from the one to
the other is a fact about ℕ, which the printed sketch leaves implicit and
`extras/math/exactness-slow-walk.tex` names: a square carries every prime to an even power, so
two squarefree numbers multiply to a square only if they are equal; reading `d` itself as
`1 · d`, that leaves `d ∈ {1, 2, 5, 10}`.  Both sentences are below, in that order, and the
second is proved from the first exactly as the slow walk prints it. -/

/-- **Two squarefree naturals whose product is a perfect square are equal.**  The slow walk's
reason, formalized: a square carries every prime to an even power, a squarefree number carries
each prime to at most one, and two numbers that are each `0` or `1` and sum to an even number
are equal. -/
theorem eq_of_squarefree_of_isSquare_mul {m n : ℕ} (hm : Squarefree m) (hn : Squarefree n)
    (h : IsSquare (m * n)) : m = n := by
  have hm0 : m ≠ 0 := hm.ne_zero
  have hn0 : n ≠ 0 := hn.ne_zero
  obtain ⟨k, hk⟩ := h
  have hk0 : k ≠ 0 := by
    rintro rfl
    rw [mul_zero] at hk
    exact (mul_ne_zero hm0 hn0) hk
  refine Nat.factorization_inj hm0 hn0 (Finsupp.ext fun p => ?_)
  have hm1 : m.factorization p ≤ 1 := (Nat.squarefree_iff_factorization_le_one hm0).mp hm p
  have hn1 : n.factorization p ≤ 1 := (Nat.squarefree_iff_factorization_le_one hn0).mp hn p
  have hsum : m.factorization p + n.factorization p = k.factorization p + k.factorization p := by
    have := congrArg (fun f => f p) (congrArg Nat.factorization hk)
    simpa [Nat.factorization_mul hm0 hn0, Nat.factorization_mul hk0 hk0] using this
  omega

/-! The three that *are* inside.  `Sanity.lean` exhibits them as anonymous `example`s already;
they are named here because the classification's backward direction needs them. -/

theorem sqrt2_mem_KR : KR (√2) := ⟨0, 1, ⟨0, 0, by norm_num⟩, ⟨1, 0, by norm_num⟩, by ring⟩

theorem sqrt5_mem_KR : KR (√5) := ⟨√5, 0, ⟨0, 1, by norm_num⟩, ⟨0, 0, by norm_num⟩, by ring⟩

theorem sqrt10_mem_KR : KR (√10) :=
  ⟨0, √5, ⟨0, 0, by norm_num⟩, ⟨0, 1, by norm_num⟩, by rw [sqrt10_eq]; ring⟩

/-- **Lemma 6's first sentence, in full** — the thesis's `iff`, not just the criterion.
Both hypotheses are load-bearing and `Sanity.lean` shows it: drop squarefreeness and `d = 8`
is a counterexample (`√8 = 2√2 ∈ K_ℝ`, yet `8 ∉ {2,5,10}`); drop `1 < d` and `d = 1` is one. -/
theorem sqrt_mem_KR_iff_of_squarefree {d : ℕ} (hd : Squarefree d) (hd1 : 1 < d) :
    KR (√(d : ℝ)) ↔ d = 2 ∨ d = 5 ∨ d = 10 := by
  have sf2 : Squarefree 2 := Nat.squarefree_two
  have sf5 : Squarefree 5 := (show Nat.Prime 5 by norm_num).prime.squarefree
  have sf10 : Squarefree 10 := Nat.squarefree_mul_iff.mpr ⟨by norm_num, sf2, sf5⟩
  constructor
  · intro hmem
    by_contra hcon
    have h2 : d ≠ 2 := fun h => hcon (Or.inl h)
    have h5 : d ≠ 5 := fun h => hcon (Or.inr (Or.inl h))
    have h10 : d ≠ 10 := fun h => hcon (Or.inr (Or.inr h))
    refine sqrt_notMem_KR (n := d) ?_ ?_ ?_ ?_ hmem
    · intro hsq
      have : d = 1 := eq_of_squarefree_of_isSquare_mul hd squarefree_one (by simpa using hsq)
      omega
    · exact fun hsq => h2 (eq_of_squarefree_of_isSquare_mul sf2 hd hsq).symm
    · exact fun hsq => h5 (eq_of_squarefree_of_isSquare_mul sf5 hd hsq).symm
    · exact fun hsq => h10 (eq_of_squarefree_of_isSquare_mul sf10 hd hsq).symm
  · rintro (rfl | rfl | rfl)
    · simpa using sqrt2_mem_KR
    · simpa using sqrt5_mem_KR
    · simpa using sqrt10_mem_KR

/-! ### The descent, and the norm test

Write `β = p + q√2` over the base and square — the thesis's printed descent runs the same
way. -/

/-- **Lemma 6, the descent.**  If `α ∈ ℚ(√5)` is a square in `K_ℝ`, then `α` or `α/2` is
already a square in `ℚ(√5)`. -/
theorem descent {α β : ℝ} (hα : Q5 α) (hβ : KR β) (h : α = β ^ 2) :
    ∃ p, Q5 p ∧ (α = p ^ 2 ∨ α = 2 * p ^ 2) := by
  obtain ⟨p, q, hp, hq, rfl⟩ := hβ
  have hexp : α = p ^ 2 + 2 * q ^ 2 + 2 * p * q * √2 := by
    rw [h]; linear_combination q ^ 2 * sq_sqrt2
  by_cases hpq : p * q = 0
  · rcases mul_eq_zero.mp hpq with h0 | h0
    · exact ⟨q, hq, Or.inr (by rw [hexp, h0]; ring)⟩
    · exact ⟨p, hp, Or.inl (by rw [hexp, h0]; ring)⟩
  · exfalso
    refine sqrt2_notMem_Q5 ?_
    have hd : (2 : ℝ) * p * q ≠ 0 := by
      intro hcon
      exact hpq (by linear_combination hcon / 2)
    have hs : √2 = (α - p ^ 2 - 2 * q ^ 2) / (2 * p * q) := by
      rw [eq_div_iff hd]; linear_combination -hexp
    rw [hs]
    have h2 : Q5 (2 : ℝ) := Q5.two
    have hp2 : Q5 (p ^ 2) := by rw [sq]; exact hp.mul hp
    have hq2 : Q5 (q ^ 2) := by rw [sq]; exact hq.mul hq
    exact ((hα.sub hp2).sub (h2.mul hq2)).div ((h2.mul hp).mul hq)

/-- The norm of `ℚ(√5)`. -/
def N (a b : ℚ) : ℚ := a ^ 2 - 5 * b ^ 2

/-- A square in `ℚ(√5)` has a rational square for its norm.  This is the test the two
five-fold surds fail. -/
theorem norm_of_sq {a b c d : ℚ} (h : (a : ℝ) + b * √5 = ((c : ℝ) + d * √5) ^ 2) :
    N a b = (N c d) ^ 2 := by
  have hexp : (a : ℝ) + b * √5 = ((c ^ 2 + 5 * d ^ 2 : ℚ) : ℝ) + ((2 * c * d : ℚ) : ℝ) * √5 := by
    push_cast
    linear_combination h + (d : ℝ) ^ 2 * sq_sqrt5
  obtain ⟨ha, hb⟩ := Q5.ext hexp
  subst ha; subst hb
  simp only [N]
  ring

/-! ### The two five-fold surds -/

theorem rat_sq_ne_16_125 (x : ℚ) : x ^ 2 ≠ 16 / 125 := fun h =>
  rat_sq_ne_five (25 * x / 4) (by linear_combination (625 / 16 : ℚ) * h)

theorem rat_sq_ne_4_125 (x : ℚ) : x ^ 2 ≠ 4 / 125 := fun h =>
  rat_sq_ne_five (25 * x / 2) (by linear_combination (625 / 4 : ℚ) * h)

theorem rat_sq_ne_5_4 (x : ℚ) : x ^ 2 ≠ 5 / 4 := fun h =>
  rat_sq_ne_five (2 * x) (by linear_combination (4 : ℚ) * h)

/-- **Lemma 6, second half (icosahedron).**  The icosahedral witness `2/25(5 − √5)` is not a
square in `K_ℝ`.  Its norm is `16/125`, and that of its half is `4/125`; neither is a rational
square, because both have squarefree part `5`. -/
theorem icosahedral_witness_not_sq : ¬ ∃ β, KR β ∧ (2 / 25 : ℝ) * (5 - √5) = β ^ 2 := by
  rintro ⟨β, hβ, h⟩
  have hα : Q5 ((2 / 25 : ℝ) * (5 - √5)) := ⟨2 / 5, -2 / 25, by push_cast; ring⟩
  obtain ⟨p, hp, hcase⟩ := descent hα hβ h
  obtain ⟨c, d, rfl⟩ := hp
  rcases hcase with hc | hc
  · refine rat_sq_ne_16_125 (N c d) ?_
    have := norm_of_sq (a := 2 / 5) (b := -2 / 25) (c := c) (d := d) (by push_cast; linarith [hc])
    rw [← this]; simp only [N]; norm_num
  · refine rat_sq_ne_4_125 (N c d) ?_
    have := norm_of_sq (a := 1 / 5) (b := -1 / 25) (c := c) (d := d) (by push_cast; linarith [hc])
    rw [← this]; simp only [N]; norm_num

/-- **Lemma 6, second half (icosahedron, normalizer).**  The icosahedral normalizer `√(2+φ)`
squares to `(5 + √5)/2`, which is not a square in `K_ℝ`: norms `5` and `5/4`.

Named for the *icosahedron*: `2 + φ` is what normalizes the icosahedron's atlas vertices
(`povm_properties.py`'s `1/sqrt(2 + tau)`; the atlas header row reads "components scaled by
`1/√(2+τ)`"), and the thesis says so: Appendix D.1's sweep writes an icosahedron vertex as
`p/√(2+τ)` with `|p| ∈ {1, τ}`.  The dodecahedron's atlas normalizer is `√3`, and its witness
is the `√3` of `sqrt3_notMem_KR`. -/
theorem icosahedral_normalizer_not_sq : ¬ ∃ β, KR β ∧ (1 / 2 : ℝ) * (5 + √5) = β ^ 2 := by
  rintro ⟨β, hβ, h⟩
  have hα : Q5 ((1 / 2 : ℝ) * (5 + √5)) := ⟨5 / 2, 1 / 2, by push_cast; ring⟩
  obtain ⟨p, hp, hcase⟩ := descent hα hβ h
  obtain ⟨c, d, rfl⟩ := hp
  rcases hcase with hc | hc
  · refine rat_sq_ne_five (N c d) ?_
    have := norm_of_sq (a := 5 / 2) (b := 1 / 2) (c := c) (d := d) (by push_cast; linarith [hc])
    rw [← this]; simp only [N]; norm_num
  · refine rat_sq_ne_5_4 (N c d) ?_
    have := norm_of_sq (a := 5 / 4) (b := 1 / 4) (c := c) (d := d) (by push_cast; linarith [hc])
    rw [← this]; simp only [N]; norm_num

/-! ## Part III — the two parts composed

Lemma 5 is stated over an arbitrary subring; to apply it to `K_ℝ` we need `K_ℝ` as a subring,
which the closure lemmas above supply.  The composed statements are the ones Appendix D.1
actually uses: a witness (equivalently, a Gram determinant) outside `K_ℝ` bars *every* pose. -/

theorem KR.ofQ5 {x : ℝ} (h : Q5 x) : KR x := ⟨x, 0, h, ⟨0, 0, by norm_num⟩, by ring⟩

theorem KR.add {x y : ℝ} (hx : KR x) (hy : KR y) : KR (x + y) := by
  obtain ⟨p, q, hp, hq, rfl⟩ := hx; obtain ⟨p', q', hp', hq', rfl⟩ := hy
  exact ⟨p + p', q + q', hp.add hp', hq.add hq', by ring⟩

theorem KR.neg {x : ℝ} (hx : KR x) : KR (-x) := by
  obtain ⟨p, q, hp, hq, rfl⟩ := hx
  exact ⟨-p, -q, hp.neg, hq.neg, by ring⟩

theorem KR.mul {x y : ℝ} (hx : KR x) (hy : KR y) : KR (x * y) := by
  obtain ⟨p, q, hp, hq, rfl⟩ := hx; obtain ⟨p', q', hp', hq', rfl⟩ := hy
  refine ⟨p * p' + 2 * (q * q'), p * q' + q * p',
    (hp.mul hp').add (Q5.two.mul (hq.mul hq')), (hp.mul hq').add (hq.mul hp'), ?_⟩
  linear_combination (q * q') * sq_sqrt2

/-- `K_ℝ = ℚ(√2,√5)` as a subring of `ℝ`. -/
def KRsubring : Subring ℝ where
  carrier := {x | KR x}
  zero_mem' := KR.ofQ5 ⟨0, 0, by norm_num⟩
  one_mem' := KR.ofQ5 ⟨1, 0, by norm_num⟩
  add_mem' := KR.add
  mul_mem' := KR.mul
  neg_mem' := KR.neg

@[simp] theorem mem_KRsubring {x : ℝ} : x ∈ KRsubring ↔ KR x := Iff.rfl

theorem KR.inv {x : ℝ} (hx : KR x) : KR x⁻¹ := by
  obtain ⟨p, q, hp, hq, rfl⟩ := hx
  have hp2 : Q5 (p ^ 2) := by rw [sq]; exact hp.mul hp
  have hq2 : Q5 (q ^ 2) := by rw [sq]; exact hq.mul hq
  have hNQ : Q5 (p ^ 2 - 2 * q ^ 2) := hp2.sub (Q5.two.mul hq2)
  by_cases hx0 : p + q * √2 = 0
  · rw [hx0]; simpa using KR.ofQ5 (show Q5 (0 : ℝ) from ⟨0, 0, by norm_num⟩)
  · have hN : p ^ 2 - 2 * q ^ 2 ≠ 0 := by
      intro hN
      by_cases hq0 : q = 0
      · refine hx0 ?_
        have hp0 : p = 0 := by
          refine sq_eq_zero_iff.mp ?_
          rw [hq0] at hN; linarith
        rw [hp0, hq0]; ring
      · refine sqrt2_notMem_Q5 ?_
        have hpq : (p / q) ^ 2 = 2 := by field_simp; linarith
        rw [show √2 = |p / q| by rw [← hpq, Real.sqrt_sq_eq_abs]]
        rcases abs_choice (p / q) with hc | hc <;> rw [hc]
        · exact hp.div hq
        · exact (hp.div hq).neg
    have key : (p + q * √2) * (p - q * √2) = p ^ 2 - 2 * q ^ 2 := by
      linear_combination (-q ^ 2) * sq_sqrt2
    have hmul : (p + q * √2) * ((p - q * √2) / (p ^ 2 - 2 * q ^ 2)) = 1 := by
      rw [← mul_div_assoc, key, div_self hN]
    rw [inv_eq_of_mul_eq_one_right hmul]
    exact ⟨p / (p ^ 2 - 2 * q ^ 2), -q / (p ^ 2 - 2 * q ^ 2), hp.div hNQ, hq.neg.div hNQ, by ring⟩

/-- `K_ℝ` as a subfield of `ℝ` — which is what Appendix D.1 calls it.  Part I needs only the
subring structure, so it is stated over `KRsubring`; the field structure is what the coset
lemmas below and Part IV's change-of-basis step spend. -/
def KRsubfield : Subfield ℝ where
  toSubring := KRsubring
  inv_mem' := fun _ hx => KR.inv hx

@[simp] theorem mem_KRsubfield {x : ℝ} : x ∈ KRsubfield ↔ KR x := Iff.rfl

/-! ### The field structure in use: outsiders come in cosets

Lemma 6 bars two *specific* numbers, `√3` and the two five-fold surds.  What Appendix D.1
needs is that their `K_ℝ`-multiples are barred too, and that is exactly what being a field
buys: if `c ≠ 0` lies in `K_ℝ` then `c·r ∈ K_ℝ` would give `r = (c·r)/c ∈ K_ℝ`.

This is not pedantry.  **No solid's witness is `√3` on the nose.**  The tetrahedron, cube and
dodecahedron all give `±4√3/9` (Table `tab:povm-exactness`, reproduced in `Solids.lean`), and
the icosahedron's normalizer enters the alignment sweep as `1/√(2+φ)` and `τ/√(2+φ)`, never
bare.  A statement about `√3` alone reaches none of them. -/

/-- A nonzero `K_ℝ`-multiple of an outsider is an outsider. -/
theorem mul_notMem_KR {c r : ℝ} (hc : KR c) (hc0 : c ≠ 0) (hr : ¬ KR r) : ¬ KR (c * r) := by
  intro h
  refine hr ?_
  have : c * r * c⁻¹ = r := by field_simp
  rw [← this]
  exact h.mul (KR.inv hc)

/-- A nonzero `K_ℝ`-multiple of the *reciprocal* of an outsider is an outsider.  This is the
form the alignment sweep needs, the atlas presenting every vertex as coordinates *over* a
normalizer. -/
theorem div_notMem_KR {c r : ℝ} (hc : KR c) (hc0 : c ≠ 0) (hr0 : r ≠ 0) (hr : ¬ KR r) :
    ¬ KR (c / r) := by
  intro h
  refine hr ?_
  have hcr : c / r ≠ 0 := div_ne_zero hc0 hr0
  have : c / (c / r) = r := by field_simp
  rw [← this]
  exact hc.mul (KR.inv h)

/-- `√3` barred throughout its `K_ℝ`-coset — the form that reaches the three `√3` solids. -/
theorem smul_sqrt3_notMem_KR {q : ℝ} (hq : KR q) (hq0 : q ≠ 0) : ¬ KR (q * √3) :=
  mul_notMem_KR hq hq0 sqrt3_notMem_KR

/-- **The icosahedral normalizer itself is outside `K_ℝ`.**  `icosahedral_normalizer_not_sq`
says `(5+√5)/2` is not a *square* in `K_ℝ`; that is the same statement, read as membership. -/
theorem sqrt_icosahedral_normalizer_notMem_KR : ¬ KR (√((1 / 2 : ℝ) * (5 + √5))) := by
  intro h
  exact icosahedral_normalizer_not_sq ⟨_, h, (Real.sq_sqrt (by positivity)).symm⟩

/-- **Appendix D.1's verdict for the `√3` solids** — Lemmas 5 and 6 composed.  A vertex triple
whose witness determinant is any nonzero `K_ℝ`-multiple of `√3` cannot be carried into `K_ℝ³`
by *any* rotation. -/
theorem no_pose_of_witness_mem_sqrt3_coset {v₁ v₂ v₃ : Fin 3 → ℝ} {q : ℝ} (hq : KR q)
    (hq0 : q ≠ 0) (h : witness v₁ v₂ v₃ = q * √3)
    {R : Matrix (Fin 3) (Fin 3) ℝ} (hR : Rᵀ * R = 1)
    (h₁ : ∀ j, KR (R.mulVec v₁ j)) (h₂ : ∀ j, KR (R.mulVec v₂ j))
    (h₃ : ∀ j, KR (R.mulVec v₃ j)) : False := by
  have hw := witness_mem_of_exists_pose (K := KRsubring) hR h₁ h₂ h₃
  rw [h] at hw
  exact smul_sqrt3_notMem_KR hq hq0 hw

/-- The bare case, `q = 1`. -/
theorem no_pose_of_witness_eq_sqrt3 {v₁ v₂ v₃ : Fin 3 → ℝ} (h : witness v₁ v₂ v₃ = √3)
    {R : Matrix (Fin 3) (Fin 3) ℝ} (hR : Rᵀ * R = 1)
    (h₁ : ∀ j, KR (R.mulVec v₁ j)) (h₂ : ∀ j, KR (R.mulVec v₂ j))
    (h₃ : ∀ j, KR (R.mulVec v₃ j)) : False :=
  no_pose_of_witness_mem_sqrt3_coset (KR.ofQ5 ⟨1, 0, by norm_num⟩) one_ne_zero
    (by rw [h]; ring) hR h₁ h₂ h₃

/-- **Appendix D.1's verdict for the icosahedron.**  The witness enters squared — as the Gram
determinant, the discriminant of the solid's quadratic form — so the test is whether that
number is a *square* in `K_ℝ`.  For `2/25(5−√5)` it is not, in any pose. -/
theorem no_pose_of_gram_eq_icosahedral {v₁ v₂ v₃ : Fin 3 → ℝ}
    (h : witness v₁ v₂ v₃ ^ 2 = (2 / 25 : ℝ) * (5 - √5))
    {R : Matrix (Fin 3) (Fin 3) ℝ} (hR : Rᵀ * R = 1)
    (h₁ : ∀ j, KR (R.mulVec v₁ j)) (h₂ : ∀ j, KR (R.mulVec v₂ j))
    (h₃ : ∀ j, KR (R.mulVec v₃ j)) : False :=
  icosahedral_witness_not_sq
    ⟨witness v₁ v₂ v₃, witness_mem_of_exists_pose (K := KRsubring) hR h₁ h₂ h₃, h.symm⟩

/-! ## Part IV — Lemma 5's last clause: any independent triple decides

Part I proved the witness rotation-proof for a triple *given in advance*.  Lemma 5 claims more:
"its square class is the same for every independent triple, so any one of them decides".
`gram_det_change_of_basis` carried the algebraic half of that — passing between triples by `C`
multiplies the Gram determinant by `(det C)²` — and left the half that makes it bite for a
solid, namely that `C` has entries in `K_ℝ`.

Here is that half.  Write `B` for the matrix of the first triple's rows and `W` for the
second's.  The change of basis is `C = M Γ⁻¹`, where `Γ = B Bᵀ` is the first triple's Gram
matrix and `M = W Bᵀ` the cross inner products — so every entry of `C` is assembled from
**inner products of vertices and nothing else**.  Two things follow, and they are the point.

*The hypothesis is pose-free.*  Inner products are rotation-invariant, so "every pairwise inner
product of vertices lies in `K_ℝ`" is a statement about the solid, not about the pose it is
written in.  That is what lets the conclusion quantify over triples of a *vertex family*.

*Division is needed, and only here.*  `Γ⁻¹` is where `K_ℝ` must be closed under inverses, which
is why this part is stated over a `Subfield` where Part I needed only a `Subring`.  `KRsubfield`
supplies it; the entries of an inverse are controlled through `Matrix.adjugate_apply`, each
adjugate entry being itself a determinant of a matrix with entries drawn from `A`, `0` and `1`.

`Solids.lean` exhibits the clause on the one concrete pair of triples the repo contains — the
tetrahedron's `tetU` and `tetU'`, the two vertex orderings that disagree — via
`no_pose_tetrahedron` and `no_pose_tetrahedron'`.  That pair is a witness, not a proof; with
`witness_mem_iff_of_vertex_inner_mem` the clause is a theorem about every pair, and
`Sanity.lean` runs the tetrahedron through it to check the general statement reproduces the
hand-checked one. -/

section AnyTriple

variable {K : Subfield ℝ}

/-- `det_mem_of_entries_mem` restated over a subfield, which is what Part IV works in. -/
theorem det_mem_of_entries_mem' {A : Matrix (Fin 3) (Fin 3) ℝ} (hA : ∀ i j, A i j ∈ K) :
    A.det ∈ K :=
  det_mem_of_entries_mem (K := K.toSubring) hA

/-- A product of matrices over `K` is over `K`: entries are sums of products. -/
theorem mul_entries_mem {A B : Matrix (Fin 3) (Fin 3) ℝ}
    (hA : ∀ i j, A i j ∈ K) (hB : ∀ i j, B i j ∈ K) : ∀ i j, (A * B) i j ∈ K := by
  intro i j
  rw [Matrix.mul_apply]
  exact sum_mem fun k _ => mul_mem (hA i k) (hB k j)

/-- The adjugate of a matrix over `K` is over `K`.  Each entry is a determinant of `A` with one
row replaced by a standard basis vector, so its entries are drawn from `A`, `0` and `1`. -/
theorem adjugate_entries_mem {A : Matrix (Fin 3) (Fin 3) ℝ} (hA : ∀ i j, A i j ∈ K) :
    ∀ i j, A.adjugate i j ∈ K := by
  intro i j
  rw [Matrix.adjugate_apply]
  refine det_mem_of_entries_mem' ?_
  intro r c
  rw [Matrix.updateRow_apply]
  split
  · rcases eq_or_ne c i with rfl | h
    · simp
    · simp [h]
  · exact hA r c

/-- **The inverse of a matrix over a subfield is over that subfield.**  This is the step Part I
could not take: `A⁻¹ = (det A)⁻¹ • adj A`, and the leading scalar needs division.  No
nonsingularity hypothesis is required — a subfield contains `0⁻¹ = 0`, so the degenerate case
is true for the uninteresting reason. -/
theorem inv_entries_mem {A : Matrix (Fin 3) (Fin 3) ℝ} (hA : ∀ i j, A i j ∈ K) :
    ∀ i j, A⁻¹ i j ∈ K := by
  intro i j
  rw [Matrix.inv_def, Matrix.smul_apply, smul_eq_mul, Ring.inverse_eq_inv]
  exact mul_mem (inv_mem (det_mem_of_entries_mem' hA)) (adjugate_entries_mem hA i j)

/-- `(A Bᵀ) i j` is the inner product of `A`'s `i`th row with `B`'s `j`th. -/
theorem cross_entry (A B : Matrix (Fin 3) (Fin 3) ℝ) (i j : Fin 3) :
    (A * Bᵀ) i j = ∑ k, A i k * B j k := by
  simp [Matrix.mul_apply, Matrix.transpose_apply]

/-- **The change of basis lies in `K`.**  If the Gram matrix `B Bᵀ` and the cross inner products
`W Bᵀ` are over `K`, and the first triple is independent, then the second triple's determinant
is a `K`-multiple of the first's — the multiple being `det(M Γ⁻¹)`, the determinant of the
change of basis. -/
theorem det_eq_mul_of_gram_mem {B W : Matrix (Fin 3) (Fin 3) ℝ}
    (hG : ∀ i j, (B * Bᵀ) i j ∈ K) (hM : ∀ i j, (W * Bᵀ) i j ∈ K) (hB : B.det ≠ 0) :
    ∃ c ∈ K, W.det = c * B.det := by
  refine ⟨((W * Bᵀ) * (B * Bᵀ)⁻¹).det,
    det_mem_of_entries_mem' (mul_entries_mem hM (inv_entries_mem hG)), ?_⟩
  simp only [Matrix.det_mul, Matrix.det_transpose, Matrix.det_nonsing_inv, Ring.inverse_eq_inv]
  field_simp

/-- The same, as the equivalence Lemma 5 asserts: two independent triples are in `K` together
or out of it together.  The multiple is nonzero because both determinants are, and `K` being a
field carries the implication both ways. -/
theorem det_mem_iff_of_gram_mem {B W : Matrix (Fin 3) (Fin 3) ℝ}
    (hG : ∀ i j, (B * Bᵀ) i j ∈ K) (hM : ∀ i j, (W * Bᵀ) i j ∈ K)
    (hB : B.det ≠ 0) (hW : W.det ≠ 0) : W.det ∈ K ↔ B.det ∈ K := by
  obtain ⟨c, hc, hcW⟩ := det_eq_mul_of_gram_mem hG hM hB
  have hc0 : c ≠ 0 := by rintro rfl; rw [zero_mul] at hcW; exact hW hcW
  constructor
  · intro h
    have : B.det = W.det / c := by rw [hcW]; field_simp
    rw [this]; exact div_mem h hc
  · intro h; rw [hcW]; exact mul_mem hc h

/-- **Lemma 5's last clause: any independent triple decides.**  `V` is a solid's vertex family.
The hypothesis is that every pairwise inner product of vertices lies in `K` — rotation-invariant,
hence a statement about the solid and not about a pose.  The conclusion is that any two
independent triples of vertices give witnesses that lie in `K` together or not at all, so the
appendix may compute on whichever triple it likes. -/
theorem witness_mem_iff_of_vertex_inner_mem {ι : Type*} {V : ι → (Fin 3 → ℝ)}
    (hV : ∀ a b, (∑ k, V a k * V b k) ∈ K) {i₁ i₂ i₃ j₁ j₂ j₃ : ι}
    (hv : witness (V i₁) (V i₂) (V i₃) ≠ 0)
    (hw : witness (V j₁) (V j₂) (V j₃) ≠ 0) :
    witness (V j₁) (V j₂) (V j₃) ∈ K ↔ witness (V i₁) (V i₂) (V i₃) ∈ K := by
  have hG : ∀ i j, ((Matrix.of ![V i₁, V i₂, V i₃]) *
      (Matrix.of ![V i₁, V i₂, V i₃])ᵀ) i j ∈ K := by
    intro i j
    rw [cross_entry]
    fin_cases i <;> fin_cases j <;> simpa using hV _ _
  have hM : ∀ i j, ((Matrix.of ![V j₁, V j₂, V j₃]) *
      (Matrix.of ![V i₁, V i₂, V i₃])ᵀ) i j ∈ K := by
    intro i j
    rw [cross_entry]
    fin_cases i <;> fin_cases j <;> simpa using hV _ _
  exact det_mem_iff_of_gram_mem hG hM hv hw

/-- The contrapositive, which is the direction Appendix D.1 uses: one independent triple whose
witness is outside `K` bars *every* independent triple of the same family. -/
theorem witness_notMem_of_witness_notMem {ι : Type*} {V : ι → (Fin 3 → ℝ)}
    (hV : ∀ a b, (∑ k, V a k * V b k) ∈ K) {i₁ i₂ i₃ j₁ j₂ j₃ : ι}
    (hv : witness (V i₁) (V i₂) (V i₃) ≠ 0)
    (hw : witness (V j₁) (V j₂) (V j₃) ≠ 0)
    (h : witness (V i₁) (V i₂) (V i₃) ∉ K) : witness (V j₁) (V j₂) (V j₃) ∉ K :=
  fun hcon => h ((witness_mem_iff_of_vertex_inner_mem hV hv hw).mp hcon)

end AnyTriple

end AppendixD1
