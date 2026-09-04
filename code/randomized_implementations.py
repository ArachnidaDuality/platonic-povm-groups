"""
Verification of the randomized-implementation claims for the Platonic-solid
POVMs.

The thesis distinguishes two protocols that both carried the name "randomized
implementation":

  R1  randomized-projective: draw g ~ G, apply U_g, apply one fixed alignment
      A (vertex axis -> zhat), read out in the Z basis. No ancillas; needs
      antipodal decomposability (tetrahedron excluded); estimator-channel
      factor T_zz. (The literature's "randomized measurements" primitive.)

  R2  twirled-native: draw g ~ G, apply U_g, then the native (Decker/Naimark)
      POVM, relabel in post-processing. Ancillas; works for all five solids,
      SIC included; estimator-channel factor tr(T)/3. (The literature's
      measurement/readout twirling.)

The suite freezes six code-backed findings behind that distinction. Each
finding's full account lives in the docstring of the module that proves it;
the index:

  Finding 1  (randomized_scalars)  The estimator-channel factor identifies
             the protocol: under a generic measurement-side affine noise
             r -> T r + t, R1's estimator channel is exactly depolarizing
             with kappa = T_zz, R2's with kappa = tr(T)/3, offsets
             vanishing in both and R2's scalar pose-free. A constant
             carried over from the OTHER protocol multiplies a weight-w
             Pauli term by exactly (kappa_run/kappa_cal)^w -- a BIAS no
             shot count removes.

  Finding 2  (randomized_scalars)  The SIC is not the price of the twirl;
             the ancilla is. R2 twirls the tetrahedral SIC to exactly
             depolarizing; R1 cannot even be defined for it.

  Finding 3  (randomized_obstruction)  The exactness obstruction, in two
             halves that between them convict all five solids: DIRECTION
             (any protocol, coin included) confines the Bloch vertices to
             K_R^3 and passes the octahedron alone; WEIGHT confines a
             deterministic dilation's effect traces to Z[1/2] and fails
             the octahedron too -- its exactness is bought entirely by
             the coin's 1/3.

  Finding 4  (randomized_twojobs)  Randomness does two jobs -- REALIZE the
             POVM and TWIRL the noise -- and R1's single draw must do
             both. The twirl bar is FLAT at T (order 12) for every solid;
             the realize bar CLIMBS 3 -> 4 -> 12 -> 60; the bars cross
             exactly at the icosahedron. The witness: the octahedron's
             coin {I, F, F+} is the one coin that is a group -- it
             realizes exactly and twirls nothing. Beyond groups, the
             flip-completed coin (the Klein layer drawn with it, applied
             AFTER the alignment) does both jobs exactly for every
             decomposable solid at the coin's Phi bill -- the
             dodecahedron's 0.4 per shot, the floor attained over group
             and non-group draws alike (check_flip_completion).

  Finding 5  (randomized_decker)  Decker's circuits are a construction,
             not a drop-in: the reorientation preserves outcome order for
             the tetrahedron alone, a skipped relabelling costs no bias
             but a 1/kappa^2 shot premium (1.54x to the dodecahedron's
             10621.86x), and the tail weight 27 <sum_a w_a^4> prices what
             the correction buys.

  Finding 6  (randomized_scalars)  Gate noise separates the protocols a
             second time, and the separation is an ORDER: over the 12-word
             T draw the twirled-native Z0 residual starts at gamma^2
             while the projective route on the same words stays linear --
             and the verdict needs BOTH the protocol and the draw.

Plus the implementation ledger: native (Naimark) ancilla counts against the
projective route's atlas circuits, and -- remark-level -- the frame
potentials showing 2T/2O/2I are exact unitary 2-/3-/5-designs (2I meets
t = 5 exactly and first fails at t = 6).

NOTATION, and it is a trap worth pinning. The suite's kappa is the
ESTIMATOR CHANNEL's multiplier, ideal 1 -- equivalently the overlap of the
believed measurement with the performed one, so kappa = 1 iff the belief is
exactly right and the shot premium of any misspecification is 1/kappa^2.
The thesis (Appendix F.3.1) and shadow_experiments.py both write eta for the
CALIBRATION SCALAR, whose noiseless value is 1/3. The two differ by the
canonical dual's factor 3:

    kappa = 3 eta,     kappa = 1  <->  eta = 1/3  (noiseless).

Premia, being ratios, are the same in either convention; a bare 1/kappa^2
carried into eta's units is not. The thesis prices the premium in Appendix
D.2's table caption and gives the kappa = 3 eta dictionary in Appendix F.3.2.

Everything is exact in the probabilistic sense: zero RNG, no sampling.
Every "randomized" quantity is a finite group average computed as an
explicit sum -- float64 linear algebra asserted at 1e-9 (the algebraic
numbers involved are well separated), SymPy exactly where field membership,
or a claim quantified over ARBITRARY noise, gate noise or state, IS the
claim. Inputs are the canonical atlas artifacts in code/data/, cross-checked
against independently constructed symbolic vertex sets.

Where a verdict is an IDENTITY -- which element a product is, whether two
vertices are antipodes, whether an orbit has four points -- it is decided a
second time by canonical form over a small algebraic number field, and the
two answers are required to agree. Those companions never replace the float
pipeline and never touch an emitted fragment; each is paired with a
value-for-value numeric agreement check, because a boolean exact test does
not test its own transcription.

MODULE MAP. main(), at the bottom, IS the run order, and the banners it
prints are the suite's Sections 0-5. This file keeps this charter, main()
and the __main__ guard; the bindings live in seven section-aligned sibling
modules:

  randomized_core.py         canonical data, float primitives, the symbolic
                             layer, atlas circuits -- shared by everything
  randomized_field.py        the number-field kit: exact fields, coercion,
                             exact channels, the reposed-twirl theorem
                             (tools only; its checks sit with their
                             sections)
  randomized_scalars.py      Sections 0-1: two protocols, two scalars
                             (findings 1 + 2 + 6)
  randomized_twojobs.py      Section 2: the two jobs of randomness
                             (finding 4), plus Section 5's design ladder
  randomized_obstruction.py  Section 3: the exactness obstruction
                             (finding 3)
  randomized_decker.py       Section 3 (cont.): Decker's circuits, outcome
                             order, tail weight (finding 5)
  randomized_fragments.py    Section 4's ledger and the six .tex fragments

The import graph is a DAG -- tools below checks, never the reverse: field
builds on core; scalars, twojobs and obstruction import core and field;
decker imports core alone; fragments imports core and decker.

Importing the module (or calling main()) writes nothing; running it as a
script additionally emits, once every check has passed, six thesis
fragments -- data/randomized_ledger.tex (the Section 5.2.3 implementation
ledger, transposed: solids as columns), data/randomized_ledger_appendix.tex
(the Appendix D coin-word table), data/randomized_exactness.tex (the
Appendix D det/min-poly table), data/randomized_labels.tex (the Appendix D
outcome-order table: Decker's permutations and the wrong-list prices), and
for Appendix F.3.3.1 data/randomized_sweep.tex (the two-bars sweep detail) and
for F.3.3.2 data/randomized_witness.tex (the C_3 witness -- a display, not a table).
Captions live in randomized_fragments.py, not in the .tex.

    cd code && uv run randomized_implementations.py

Exits 0 with a verification report if every claim checks out; raises
(non-zero exit) on the first failure.
"""

import numpy as np

from randomized_core import T_NOISE, t_NOISE
from randomized_scalars import (check_exact_scalars, check_gate_noise_residual,
                                check_canonical_data, check_two_protocols,
                                check_calibration_mismatch, check_alignment)
from randomized_twojobs import (check_exact_two_bars, check_coset_coin,
                                check_minimal_twirl, check_universal_twirl,
                                check_subgroup_sweep, check_coin_group,
                                check_wilkens_layers, check_flip_completion,
                                check_atlas_resources, check_unitary_designs)
from randomized_obstruction import (check_obstruction,
                                    check_no_exact_alignment, check_gate_axes,
                                    check_octahedron_exact,
                                    check_weight_obstruction,
                                    check_reorientation_obstruction)
from randomized_decker import (check_decker_outcome_order, check_tail_weight,
                              check_swept_third_moment)
from randomized_fragments import print_ledger, write_fragments

# Downstream re-exports -- explicit names, never a star.
# shadow_experiments.py's two-protocol study reads ri.T_NOISE, ri.t_NOISE,
# ri.channel_R1, ri.channel_R2 through a function-local import, so those four
# names must stay bound here -- T_NOISE and t_NOISE are already imported above
# for main()'s genericity asserts.
from randomized_core import channel_R1, channel_R2


# ---------------------------------------------------------------------------

def main():
    # the probe noise is generic: both candidate scalars distinct, offset and
    # anisotropy nonzero -- the depolarizing checks cannot pass by accident
    assert abs(T_NOISE[2, 2] - np.trace(T_NOISE) / 3) > 0.05
    assert np.linalg.norm(t_NOISE) > 0.05
    assert np.linalg.norm(T_NOISE - np.trace(T_NOISE) / 3 * np.eye(3)) > 0.05

    print("=== 0. canonical data ".ljust(74, "="))
    check_canonical_data()
    print()
    print("=== 1. two protocols, two scalars (findings 1 + 2 + 6) ".ljust(74, "="))
    check_two_protocols()
    print()
    check_exact_scalars()
    print()
    check_calibration_mismatch()
    print()
    check_gate_noise_residual()
    print()
    check_alignment()
    print()
    print("=== 2. the two jobs of randomness (finding 4) ".ljust(74, "="))
    check_coset_coin()
    print()
    check_minimal_twirl()
    print()
    check_subgroup_sweep()
    print()
    check_exact_two_bars()
    print()
    check_coin_group()
    print()
    check_wilkens_layers()
    print()
    check_flip_completion()
    print()
    check_universal_twirl()
    print()
    check_atlas_resources()
    print()
    print("=== 3. the exactness obstruction (finding 3) ".ljust(74, "="))
    check_obstruction()
    print()
    check_no_exact_alignment()
    print()
    check_gate_axes()
    print()
    check_octahedron_exact()
    print()
    check_weight_obstruction()
    print()
    check_reorientation_obstruction()
    print()
    check_decker_outcome_order()
    print()
    check_tail_weight()
    print()
    check_swept_third_moment()
    print()
    print("=== 4. the implementation ledger ".ljust(74, "="))
    print_ledger()
    print()
    print("=== 5. remark: the unitary-design ladder ".ljust(74, "="))
    check_unitary_designs()
    print()
    print("All randomized-implementation claims verified.")


if __name__ == "__main__":
    main()
    print()
    write_fragments()
