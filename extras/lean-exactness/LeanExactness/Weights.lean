/-
# Theorem 4 (weights): the arithmetic core

Appendix D.1 runs *two* obstructions.  `AppendixD1.lean` and `Solids.lean` formalize the first
— the one on **directions**, which bars four solids by putting their vertices outside `K_ℝ`.
This file formalizes the arithmetic of the second, the one on **weights**, which bars every
deterministic protocol for all five and is what forces the octahedron's coin.

The two obstructions are not variants of one another.  The direction test lives in the *field*
`K_ℝ` and is blind to scale; the weight test lives in the *ring* `ℛ` and is exactly about the
denominator the field forgot — `⅓` and `1` are equally at home in `ℚ`, so no field sees the
weight obstruction, and this file is that arithmetic checked.

## Why this file exists, given that the README says Theorem 4 stays out

It does stay out — as a statement about protocols.  Theorem 4 says *a deterministic protocol
realizes only effects with `Tr E_k ∈ ℛ`*, and formalizing that needs the protocol model, which
is where the residual risk lives and which Lean would only certify against itself.  But the
theorem's *second* half is not about protocols at all:

> For the ring above `ℛ ∩ ℚ = ℤ[½]`, so a transitive covariant POVM on `V` outcomes
> admits a deterministic implementation only if `V` is a power of two.

That is a claim about a subring of `ℂ`, and it is the claim the octahedron's `⅓`, the
icosahedron's `⅙` and the dodecahedron's `⅒` actually run into.  It is checkable without any
model of anything, on exactly the footing Lemma 5 is: formalize the protocol-free core, leave
the protocol wrapper alone.  So the boundary is unchanged — it has just been drawn where the
mathematics is rather than where the theorem number is.

## What is proved

* `Rring` is `ℛ = ℤ[φ, i, √2, ½]` as a `ℤ`-subalgebra of `ℂ`, the ring holding every gate
  entry of every gate set of the thesis.
* `Rring_le_halvedIntegral` — every element of `ℛ` is an algebraic integer over a power of
  two.  (The generators `φ`, `i`, `√2` are algebraic integers; `½` supplies the denominator.)
* `isDyadic_of_rat_mem_Rring` — **`ℛ ∩ ℚ ⊆ ℤ[½]`**.  The decisive step is the classical one
  the appendix uses once and decisively: a rational algebraic integer is an integer.
  Here that is mathlib's `IsIntegrallyClosed ℤ`.
* `isDyadic_two_div_iff` — `2/V ∈ ℤ[½] ↔ V` is a power of two.
* `weight_obstruction` — the two composed, and then the five solids one at a time:
  the tetrahedron (`V = 4`) and cube (`V = 8`) pass and are barred by the *other* obstruction,
  while the octahedron (`6`), icosahedron (`12`) and dodecahedron (`20`) fail outright.
* `fracRring_eq_KC` and `mem_fracRring_iff_KR` — **`Frac ℛ = ℚ(√2,√5,i)` and
  `Frac ℛ ∩ ℝ = K_ℝ`**, the sentence that makes this file and `AppendixD1.lean` two halves of
  one appendix rather than two unrelated exercises.  See the section at the end.

Every membership verdict in Appendix D.1 is a test against `ℚ(√2,√5)`, so without that last
pair the formalization would prove things about two objects — the field `KR` of
`AppendixD1.lean` and the ring `Rring` here — that only the prose identifies.

## What is still not proved, and why

The step from a protocol to `Tr E_k = 2/V ∈ ℛ` — banning the coin, banning the discard,
bounding the rounds, and invoking covariance to make the weights equal.  That is the protocol
model, and it stays out for the reason the README gives.  Note the asymmetry this leaves: Lean
certifies that `⅓` is not available to the ring, and says nothing about whether a protocol is
confined to the ring.  As with Lemma 5, the formalized half is the half that was never in
doubt.
-/
import Mathlib
import LeanExactness.AppendixD1

namespace AppendixD1.Weights

open Polynomial Real AppendixD1

/-! ## The generators are algebraic integers -/

/-- The golden ratio, in `ℂ`. -/
noncomputable def phig : ℂ := (1 + √5) / 2

/-- `√2`, in `ℂ`. -/
noncomputable def rt2 : ℂ := (√2 : ℝ)

theorem isIntegral_rt2 : IsIntegral ℤ rt2 := by
  refine ⟨X ^ 2 - C 2, monic_X_pow_sub_C _ (by norm_num), ?_⟩
  simp only [eval₂_sub, eval₂_X_pow, eval₂_C, rt2]
  rw [show ((√2 : ℝ) : ℂ) ^ 2 = ((√2 ^ 2 : ℝ) : ℂ) by push_cast; ring,
    Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 2)]
  norm_num

theorem isIntegral_I : IsIntegral ℤ Complex.I := by
  refine ⟨X ^ 2 + C 1, by monicity!, ?_⟩
  simp [eval₂_add, Complex.I_sq]

theorem isIntegral_phig : IsIntegral ℤ phig := by
  refine ⟨X ^ 2 - X - C 1, by monicity!, ?_⟩
  simp only [eval₂_sub, eval₂_X_pow, eval₂_X, eval₂_C]
  have h5 : ((√5 : ℝ) : ℂ) ^ 2 = 5 := by
    rw [show ((√5 : ℝ) : ℂ) ^ 2 = ((√5 ^ 2 : ℝ) : ℂ) by push_cast; ring,
      Real.sq_sqrt (by norm_num : (0:ℝ) ≤ 5)]
    norm_num
  unfold phig; push_cast; linear_combination (1 / 4 : ℂ) * h5

theorem isIntegral_two_pow (k : ℕ) : IsIntegral ℤ ((2:ℂ) ^ k) := by
  rw [show ((2:ℂ)) ^ k = algebraMap ℤ ℂ (2 ^ k) by rw [map_pow]; norm_num]
  exact isIntegral_algebraMap

/-! ## `ℛ` sits inside the halved algebraic integers -/

/-- Numbers that become algebraic integers after multiplication by a power of two.  This is the
"algebraic integer over a power of two" of the appendix's refresher, made into a ring. -/
noncomputable def halvedIntegral : Subalgebra ℤ ℂ where
  carrier := {z | ∃ (k : ℕ) (x : ℂ), IsIntegral ℤ x ∧ z = x / 2 ^ k}
  mul_mem' := by
    rintro a b ⟨j, x, hx, rfl⟩ ⟨k, y, hy, rfl⟩
    exact ⟨j + k, x * y, hx.mul hy, by rw [pow_add]; ring⟩
  add_mem' := by
    rintro a b ⟨j, x, hx, rfl⟩ ⟨k, y, hy, rfl⟩
    refine ⟨j + k, 2 ^ k * x + 2 ^ j * y,
      ((isIntegral_two_pow k).mul hx).add ((isIntegral_two_pow j).mul hy), ?_⟩
    have h2 : (2:ℂ) ≠ 0 := two_ne_zero
    field_simp [pow_add]; ring
  algebraMap_mem' := fun n => ⟨0, algebraMap ℤ ℂ n, isIntegral_algebraMap, by simp⟩

/-- **`ℛ = ℤ[φ, i, √2, ½]`** — the ring the appendix tracks, holding the entries of every gate
of `2T`, Clifford, Clifford+`T`, `2I`, Clifford+`Φ` and their unions (and of CNOT, whose
entries are `0` and `1`). -/
noncomputable def Rring : Subalgebra ℤ ℂ := Algebra.adjoin ℤ {phig, Complex.I, rt2, 1 / 2}

/-- Every element of `ℛ` is an algebraic integer over a power of two. -/
theorem Rring_le_halvedIntegral : Rring ≤ halvedIntegral := by
  refine Algebra.adjoin_le ?_
  rintro z (rfl | rfl | rfl | rfl)
  · exact ⟨0, phig, isIntegral_phig, by simp⟩
  · exact ⟨0, Complex.I, isIntegral_I, by simp⟩
  · exact ⟨0, rt2, isIntegral_rt2, by simp⟩
  · exact ⟨1, 1, isIntegral_one, by norm_num⟩

/-! ## `ℛ ∩ ℚ = ℤ[½]` -/

/-- The dyadic rationals `ℤ[½]`: denominators powers of two, and nothing else. -/
def IsDyadic (q : ℚ) : Prop := ∃ (k : ℕ) (m : ℤ), q = m / 2 ^ k

/-- **`ℛ ∩ ℚ ⊆ ℤ[½]`.**  The one use of "a rational algebraic integer is an ordinary integer",
which is `IsIntegrallyClosed ℤ` here. -/
theorem isDyadic_of_rat_mem_Rring {q : ℚ} (h : ((q : ℚ) : ℂ) ∈ Rring) : IsDyadic q := by
  obtain ⟨k, x, hx, hxq⟩ := Rring_le_halvedIntegral h
  have hx' : x = ((2 ^ k * q : ℚ) : ℂ) := by
    have h2 : ((2:ℂ)) ^ k ≠ 0 := pow_ne_zero _ two_ne_zero
    field_simp at hxq
    push_cast
    linear_combination -hxq
  rw [hx'] at hx
  have hinj : Function.Injective (algebraMap ℚ ℂ) := (algebraMap ℚ ℂ).injective
  obtain ⟨m, hm⟩ := IsIntegrallyClosed.isIntegral_iff.mp ((isIntegral_algebraMap_iff hinj).mp hx)
  refine ⟨k, m, ?_⟩
  have hmq : (m : ℚ) = 2 ^ k * q := by exact_mod_cast hm
  field_simp
  linarith [hmq]

/-- The converse inclusion, so the intersection is an equality: `½ ∈ ℛ`, hence so is every
dyadic. -/
theorem rat_mem_Rring_of_isDyadic {q : ℚ} (h : IsDyadic q) : ((q : ℚ) : ℂ) ∈ Rring := by
  obtain ⟨k, m, rfl⟩ := h
  have hhalf : (1 / 2 : ℂ) ∈ Rring :=
    Algebra.subset_adjoin (by simp)
  have hm : ((m : ℚ) : ℂ) ∈ Rring := by
    have hcast : ((m : ℚ) : ℂ) = algebraMap ℤ ℂ m := by simp
    rw [hcast]; exact Rring.algebraMap_mem m
  have hsplit : (((m / 2 ^ k : ℚ)) : ℂ) = ((m : ℚ) : ℂ) * (1 / 2 : ℂ) ^ k := by
    push_cast
    rw [div_pow, one_pow]
    ring
  rw [hsplit]
  exact Rring.mul_mem hm (Rring.pow_mem hhalf k)

/-! ## `2/V` is dyadic exactly when `V` is a power of two -/

theorem isDyadic_two_div_iff {V : ℕ} (hV : 0 < V) :
    IsDyadic (2 / V) ↔ ∃ j : ℕ, V = 2 ^ j := by
  constructor
  · rintro ⟨k, m, hk⟩
    have hV0 : (V : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr hV.ne'
    have h2 : (2:ℚ) * 2 ^ k = m * V := by field_simp at hk; linarith [hk]
    have hz : (2:ℤ) * 2 ^ k = m * V := by exact_mod_cast h2
    have hdvd : (V : ℤ) ∣ 2 ^ (k + 1) := ⟨m, by rw [pow_succ]; linarith [hz]⟩
    have hn : V ∣ 2 ^ (k + 1) := by exact_mod_cast hdvd
    obtain ⟨j, _, hj⟩ := (Nat.dvd_prime_pow Nat.prime_two).mp hn
    exact ⟨j, hj⟩
  · rintro ⟨j, rfl⟩
    exact ⟨j, 2, by push_cast; ring⟩

/-- An odd prime factor rules a number out of being a power of two. -/
theorem not_pow_two_of_odd_prime_dvd {p V : ℕ} (hp : p.Prime) (hp2 : p ≠ 2) (hdvd : p ∣ V) :
    ¬ ∃ j : ℕ, V = 2 ^ j := by
  rintro ⟨j, rfl⟩
  exact hp2 ((Nat.prime_dvd_prime_iff_eq hp Nat.prime_two).mp (hp.dvd_of_dvd_pow hdvd))

/-! ## The verdict

`weight_obstruction` is Theorem 4's conclusion with its protocol hypothesis replaced by the
arithmetic consequence that hypothesis is used to establish. -/

/-- **Theorem 4, arithmetic half.**  If a transitive covariant POVM on `V` outcomes has
its common effect weight `2/V` inside `ℛ`, then `V` is a power of two. -/
theorem weight_obstruction {V : ℕ} (hV : 0 < V) (h : (((2 / V : ℚ)) : ℂ) ∈ Rring) :
    ∃ j : ℕ, V = 2 ^ j :=
  (isDyadic_two_div_iff hV).mp (isDyadic_of_rat_mem_Rring h)

/-- The octahedron: `V = 6`, weight `⅓`.  Not in `ℛ` — which is why its exact implementation
has to spend a coin, a discarded branch, or unbounded repetition. -/
theorem octahedron_weight_notMem : ((2 / 6 : ℚ) : ℂ) ∉ Rring := fun h =>
  not_pow_two_of_odd_prime_dvd (p := 3) (by norm_num) (by norm_num) (by norm_num)
    (weight_obstruction (by norm_num) h)

/-- The icosahedron: `V = 12`, weight `⅙`. -/
theorem icosahedron_weight_notMem : ((2 / 12 : ℚ) : ℂ) ∉ Rring := fun h =>
  not_pow_two_of_odd_prime_dvd (p := 3) (by norm_num) (by norm_num) (by norm_num)
    (weight_obstruction (by norm_num) h)

/-- The dodecahedron: `V = 20`, weight `⅒`. -/
theorem dodecahedron_weight_notMem : ((2 / 20 : ℚ) : ℂ) ∉ Rring := fun h =>
  not_pow_two_of_odd_prime_dvd (p := 5) (by norm_num) (by norm_num) (by norm_num)
    (weight_obstruction (by norm_num) h)

/-- The tetrahedron (`V = 4`, weight `½`) and the cube (`V = 8`, weight `¼`) **pass** the
weight test — their weights really are in `ℛ`.  They are barred by the *other* obstruction,
which is the point of running two: between them every solid is convicted of something. -/
theorem tetrahedron_weight_mem : ((2 / 4 : ℚ) : ℂ) ∈ Rring :=
  rat_mem_Rring_of_isDyadic ⟨1, 1, by norm_num⟩

theorem cube_weight_mem : ((2 / 8 : ℚ) : ℂ) ∈ Rring :=
  rat_mem_Rring_of_isDyadic ⟨2, 1, by norm_num⟩

/-! ## `Frac ℛ ∩ ℝ = K_ℝ` — the ring and the field, identified

`AppendixD1.lean` proves what it proves about `KR` and this file what it proves about `Rring`;
what says the two concern the same appendix is the identification here.  Appendix D.1 says it
in one sentence — *"Its field of fractions is `K = ℚ(√2,√5,i)`.  Its real subfield is
`K_ℝ = K ∩ ℝ = ℚ(√2,√5)`"* — and every membership verdict in the appendix, in Table D.2, in
Corollary 7's sweep and in `Solids.lean` is a test against the right-hand side.  That sentence
is the hinge on which the choice of test rests, and this section is that sentence.

`KCsubfield` is `K`, written the way the identification needs it: `z ∈ K` iff `Re z` and `Im z`
both lie in `K_ℝ`.  Written that way, `{1, i}` being a `K_ℝ`-basis of `K` is not a hypothesis to
discharge — the parts are *taken*, so a real `z` has `Im z = 0` and nothing else is needed.
What has to be earned is that the carrier is a subfield at all, where the inverse is the only
interesting closure (`Complex.inv_re` divides by `normSq`, which is why `KRsubfield` and not
`KRsubring` is spent), and that `ℛ`'s four generators lie in it.

`Frac ℛ` is `Subfield.closure ℛ`, and the identification runs both ways: `ℛ ⊆ K` closes the
closure inside `K`, while `√5 = 2φ − 1` together with `√2, i ∈ ℛ` puts `ℚ(√2,√5,i)` back inside
it.  `mem_fracRring_iff_KR` is then the appendix's sentence, and it is what licenses every test
the other files run. -/

/-- **`K = ℚ(√2,√5,i)`**, presented as the split `z = Re z + i Im z` over `K_ℝ`. -/
noncomputable def KCsubfield : Subfield ℂ where
  carrier := {z | z.re ∈ KRsubfield ∧ z.im ∈ KRsubfield}
  zero_mem' := by constructor <;> simp
  one_mem' := by constructor <;> simp
  add_mem' := fun hz hw => ⟨by simpa using add_mem hz.1 hw.1,
                            by simpa using add_mem hz.2 hw.2⟩
  neg_mem' := fun hz => ⟨by simpa using neg_mem hz.1, by simpa using neg_mem hz.2⟩
  mul_mem' := fun hz hw =>
    ⟨by simpa [Complex.mul_re] using sub_mem (mul_mem hz.1 hw.1) (mul_mem hz.2 hw.2),
     by simpa [Complex.mul_im] using add_mem (mul_mem hz.1 hw.2) (mul_mem hz.2 hw.1)⟩
  inv_mem' := fun z hz => by
    have hn : Complex.normSq z ∈ KRsubfield := by
      rw [Complex.normSq_apply]; exact add_mem (mul_mem hz.1 hz.1) (mul_mem hz.2 hz.2)
    exact ⟨by rw [Complex.inv_re]; exact div_mem hz.1 hn,
           by rw [Complex.inv_im]; exact div_mem (neg_mem hz.2) hn⟩

@[simp] theorem mem_KCsubfield {z : ℂ} : z ∈ KCsubfield ↔ KR z.re ∧ KR z.im := Iff.rfl

theorem KR_zero : KR (0 : ℝ) := KRsubfield.zero_mem

theorem KR_sqrt2 : KR (√2) := ⟨0, 1, ⟨0, 0, by norm_num⟩, ⟨1, 0, by norm_num⟩, by ring⟩

theorem phig_mem_KC : phig ∈ KCsubfield := by
  refine ⟨?_, ?_⟩
  · show KR phig.re
    rw [show phig.re = (1 + √5) / 2 from by unfold phig; simp]
    exact KR.ofQ5 ⟨1 / 2, 1 / 2, by push_cast; ring⟩
  · show KR phig.im
    rw [show phig.im = 0 from by unfold phig; simp]
    exact KR_zero

theorem rt2_mem_KC : rt2 ∈ KCsubfield := by
  refine ⟨?_, ?_⟩
  · show KR rt2.re
    rw [show rt2.re = √2 from by unfold rt2; simp]; exact KR_sqrt2
  · show KR rt2.im
    rw [show rt2.im = 0 from by unfold rt2; simp]; exact KR_zero

theorem rt2_mem_Rring : rt2 ∈ Rring := Algebra.subset_adjoin (by simp)

theorem I_mem_Rring : Complex.I ∈ Rring := Algebra.subset_adjoin (by simp)

/-- `√5` is *in* the ring, not merely in its fraction field: `√5 = 2φ − 1`.  This is the
inclusion that stops `Frac ℛ` from being smaller than `K`. -/
theorem sqrt5_mem_Rring : ((√5 : ℝ) : ℂ) ∈ Rring := by
  have hphi : phig ∈ Rring := Algebra.subset_adjoin (by simp)
  rw [show ((√5 : ℝ) : ℂ) = 2 * phig - 1 from by unfold phig; ring]
  exact sub_mem (mul_mem (by exact_mod_cast Rring.natCast_mem 2) hphi) Rring.one_mem

/-- `ℛ ⊆ K`: the four generators, one at a time. -/
theorem Rring_le_KCsubfield : (Rring : Set ℂ) ⊆ (KCsubfield : Set ℂ) := by
  show Rring ≤ subalgebraOfSubring KCsubfield.toSubring
  refine Algebra.adjoin_le ?_
  rintro z (rfl | rfl | rfl | rfl)
  · exact phig_mem_KC
  · exact ⟨by simp, by simp⟩
  · exact rt2_mem_KC
  · exact ⟨by simp, by simp⟩

/-- `K_ℝ ⊆ Frac ℛ`, in the basis `KR_iff` supplies: rationals come free with the subfield,
`√2` and `√5` are ring elements, and `√10` is their product. -/
theorem ofReal_mem_fracRring {x : ℝ} (hx : KR x) :
    (x : ℂ) ∈ Subfield.closure (Rring : Set ℂ) := by
  obtain ⟨a, b, c, d, hx'⟩ := (KR_iff x).mp hx
  have h2 : ((√2 : ℝ) : ℂ) ∈ Subfield.closure (Rring : Set ℂ) :=
    Subfield.subset_closure rt2_mem_Rring
  have h5 : ((√5 : ℝ) : ℂ) ∈ Subfield.closure (Rring : Set ℂ) :=
    Subfield.subset_closure sqrt5_mem_Rring
  rw [hx', sqrt10_eq]
  push_cast
  refine add_mem (add_mem (add_mem ?_ ?_) ?_) ?_
  · exact SubfieldClass.ratCast_mem _ a
  · exact mul_mem (SubfieldClass.ratCast_mem _ b) h2
  · exact mul_mem (SubfieldClass.ratCast_mem _ c) h5
  · exact mul_mem (SubfieldClass.ratCast_mem _ d) (mul_mem h2 h5)

/-- **`Frac ℛ = K = ℚ(√2,√5,i)`.**  One inclusion is `ℛ ⊆ K` closed up; the other splits
`z = Re z + i · Im z` and puts each half back through `ofReal_mem_fracRring`. -/
theorem fracRring_eq_KC : Subfield.closure (Rring : Set ℂ) = KCsubfield := by
  refine le_antisymm (Subfield.closure_le.mpr Rring_le_KCsubfield) ?_
  intro z hz
  rw [← Complex.re_add_im z]
  exact add_mem (ofReal_mem_fracRring hz.1)
    (mul_mem (ofReal_mem_fracRring hz.2) (Subfield.subset_closure I_mem_Rring))

/-- **`Frac ℛ ∩ ℝ = K_ℝ`** — the appendix's sentence, and the hinge every membership verdict
in Appendix D.1 turns on.  A real number is a quotient of gate-entry ring elements exactly when
it lies in `ℚ(√2,√5)`, so testing against that field is testing the right thing. -/
theorem mem_fracRring_iff_KR (x : ℝ) :
    (x : ℂ) ∈ Subfield.closure (Rring : Set ℂ) ↔ KR x := by
  rw [fracRring_eq_KC]
  exact ⟨fun h => by simpa using h.1, fun h => ⟨by simpa using h, by simp⟩⟩

end AppendixD1.Weights
