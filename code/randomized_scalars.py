"""Sections 0-1 of the randomized-implementation suite: the canonical-data
pins, the two protocols and their two scalars, the calibration-mismatch
law, the alignment step, and the gate-noise residual (findings 1 + 2 + 6).

randomized_implementations.py is the suite's entry point -- its docstring
carries the R1/R2 protocol definitions, the kappa/eta pin, the findings
index, the module map and the run instructions; `cd code && uv run
randomized_implementations.py` runs the whole suite. The findings proved
here:

  Finding 1  The estimator-channel factor identifies the protocol: under a
             generic measurement-side affine noise r -> T r + t, R1's
             estimator channel is exactly depolarizing with kappa = T_zz (the
             draw and the readout share an axis, so the noise is seen only
             along it), R2's with kappa = tr(T)/3 (nothing correlates with an
             axis, so the noise is seen whole). Offsets vanish in both.
             R2's scalar is also POSE-FREE, and that too is an identity
             rather than two poses that happened to agree: re-pose the
             measured POVM by an arbitrary 3x3 C and the channel comes back
             (tr(C^T C T)/3) Id_3. The theorem and why the measured vertices
             cannot reach the scalar are in exact_reposed_twirl_R2
             (randomized_field).

             One misspecification is NOT priced that way, and it turns on
             the same two scalars. A calibration is empirical: learned on
             the run it reconstructs, it absorbs whatever the apparatus
             did -- the reason a fixed (T, t) or a wrong list (finding 5)
             costs shots and never truth. A constant carried over from the
             OTHER protocol was never learned on the run being
             reconstructed, and that mistake lands on the other side of
             the ledger: the estimator divides by the believed constant
             once per touched site, so a weight-w Pauli term is multiplied
             by exactly (kappa_run/kappa_cal)^w -- a BIAS no shot count
             removes. check_calibration_mismatch proves the law on
             Appendix F's own estimator (shadow_experiments.py, lazily
             imported; write-free, nothing read from its npz) and pins
             the numbers behind it -- F.3.1's worked example, the
             dephasing swap's -6.49 against the true -5.23, and beside it
             the exact +0.7578 at depolarizing 0.1 under the noiseless
             dual and the probe's own +1.04 / -1.33.

  Finding 2  The SIC is not the price of the twirl; the ancilla is. R2 twirls
             the tetrahedral SIC to exactly depolarizing; R1 cannot even be
             defined for it (no antipodal vertex pairs).

  Finding 6  Gate noise separates the two protocols a second time, and the
             separation is an ORDER. Measurement-side noise is one channel
             shared by every draw, so Schur applies and finding 1 is exact.
             Per-GATE noise arrives inside the drawn circuit, correlated with
             g, and Schur does not apply. Expand the atlas's own words to
             first order in the damping gamma: every insertion reaches the
             estimator through its word PREFIX alone, the suffix cancelling
             against the post-processing (R_g^T S_i = P_i^T), so the
             first-order displacement m1 is a sum over prefixes. Over the
             12-word T draw that sum has no z component -- for every one of
             the 2^12 choices of atlas representative, in both compilations
             -- and the first-order matrix term is exactly -Id_3/3, which a
             scalar |0> calibration absorbs whole. Hence the twirled-native
             Z0 residual starts at gamma^2, coefficient (z-1)/12; what
             survives at first order is transverse and state-independent, an
             X0 slope of exactly 1/6. Those three constants are the BARE
             control row's, and that is the trap: Appendix F prints the
             DILATED row, where all three move. check_gate_noise_residual
             says which is which and where they move to; what the dilation
             does not move is the VANISHING itself. The projective route on
             the SAME words stays linear, (1-z)/4 for the octahedral draw.
             The verdict needs BOTH the protocol and the draw: run
             twirled-native over the O draw instead and Z0 is linear again,
             (1-z)/6, for all 2^24 representative choices -- while R1 over
             the SAME T words is linear too, so neither the protocol nor the
             draw suffices alone. Proved over a generic state and a generic
             dilation strength, so it covers Appendix F's gate-noise study
             (shadow_experiments.gate_noise_twirl) rather than reproducing
             it; nothing is read from that module's npz.
"""

import numpy as np
import sympy as sp
from sympy import I as sI
from sympy import Matrix, Rational, sqrt

from randomized_core import (DATA, SOLIDS, COVARIANCE, T_NOISE, t_NOISE, PAULI,
                             load_vertices, load_elements, load_rotations,
                             load_atlas, rotation_from_unitary, rot_key,
                             is_decomposable, alignment, channel_R1,
                             channel_R2, orbit_counts, symbolic_solids,
                             atlas_gates, bloch_matrix, exact_rotations)
from randomized_field import (FIELD_NAME, solid_field, to_field,
                              exact_vertices, exact_alignment,
                              exact_is_decomposable, exact_channel_R1, _probe,
                              exact_twirls, exact_channel_R2, exact_twirls_R2,
                              _exact_probe_noise, _as_float)

def check_exact_scalars():
    # Findings 1 + 2, exactly and generically. check_two_protocols decides both
    # scalars with allclose at the single probe T_NOISE; each is quantified here
    # over EVERY measurement-side (T, t), so T_NOISE stops being load-bearing.
    #
    # R1's half is not new mathematics -- rank_one_twirl already proves
    # M = (v.w) Id_3 over six free symbols in check_coin_group -- but it is
    # established here against channel_R1 itself rather than against a reduction
    # of it, and over the same probe basis as R2, so the two protocols are
    # finally checked by one argument instead of two.
    print(f"  {'solid':14s} {'grp':4s} {'field':16s} {'R2 -> tr(T)/3':>14s}"
          f" {'R1 -> T_zz':>18s}   exact vs float at T_NOISE")
    print("  " + "-" * 100)
    for solid in SOLIDS:
        K, g = solid_field(solid), COVARIANCE[solid]
        R = [to_field(M, K) for M in exact_rotations(g)]
        s = exact_vertices(solid, K)
        s_f, R_f = load_vertices(solid), load_rotations(g)
        assert exact_twirls_R2(s, R, K), solid

        # Transcription guard. The generic verdicts above are booleans, and a
        # slip inside these loops (a dropped b, a swapped index) moves numbers
        # while only sometimes flipping a boolean.
        # So require the exact channels to reproduce the float ones at T_NOISE,
        # which is a value-for-value test of the transcription itself.
        Te, te = _exact_probe_noise(K)
        d = np.abs(_as_float(exact_channel_R2(s, R, Te, te, K)[0], K)
                   - channel_R2(s_f, R_f, T_NOISE, t_NOISE)[0]).max()
        r1 = "n/a (not antipodal)"
        # which solids admit R1 at all is itself an identity question, so the
        # branch below is taken on field equality and the float verdict is
        # required to agree rather than trusted
        assert exact_is_decomposable(s, K) == is_decomposable(s_f), solid
        if exact_is_decomposable(s, K):
            assert exact_twirls(s, R, K), solid
            d = max(d, np.abs(_as_float(exact_channel_R1(s, R, Te, te, K)[0], K)
                              - channel_R1(s_f, R_f, T_NOISE, t_NOISE)[0]).max())
            r1 = "identity in T"
        assert d < 1e-12, (solid, d)
        print(f"  {solid:14s} {g:4s} {FIELD_NAME[solid]:16s} {'identity in T':>14s}"
              f" {r1:>18s}   max|diff| = {d:.1e}")
    # Two negative controls, so neither verdict can be passing vacuously.
    K = solid_field("octahedron")
    s = exact_vertices("octahedron", K)
    R = [to_field(M, K) for M in exact_rotations("O")]
    # (i) The two scalars are DIFFERENT identities, not two readings of one
    # number -- finding 1 itself. At the probe T = E_{0,0} we have tr T = 1 but
    # T_zz = 0, so R2 must return Id_3/3 exactly where R1 returns 0.
    P = _probe(K, entry=(0, 0))
    third = [[K.one / 3 if i == j else K.zero for j in range(3)] for i in range(3)]
    assert exact_channel_R2(s, R, *P, K)[0] == third
    assert exact_channel_R1(s, R, *P, K)[0] == [[K.zero] * 3 for _ in range(3)]
    # (ii) A reducible draw fails both tests outright: the C_3 coin of
    # check_coin_group, which realizes the octahedral POVM and twirls nothing.
    R_F = bloch_matrix(atlas_gates()["F"])
    C3 = [to_field(M, K) for M in (sp.eye(3), R_F, R_F * R_F)]
    assert not exact_twirls_R2(s, C3, K) and not exact_twirls(s, C3, K)
    print("  negative controls: at T = E_00 (tr T = 1, T_zz = 0) R2 gives Id_3/3 where")
    print("  R1 gives 0 -- finding 1 as an identity, not as a gap between two decimals;")
    print("  and the reducible C_3 coin fails both tests")

    print("[ok] both scalars are IDENTITIES in the noise, not values at a probe: R2 gives")
    print("     (tr T/3) Id_3 and zero offset on all five solids -- the tetrahedron")
    print("     included, where R1 is undefined -- and R1 gives T_zz Id_3 on the four")
    print("     antipodal ones. So neither protocol's scalar rests on T_NOISE any more,")
    print("     and the float<->float cross-check against shadow_experiments.py now has an")
    print("     exact anchor on this side of it")


# ---------------------------------------------------------------------------
# Gate noise: why the twirled-native residual starts at gamma^2
#
# check_exact_scalars settles MEASUREMENT-side noise, where one channel is
# shared by every draw and Schur applies.  Gate noise is the opposite case:
# each g is a circuit, damping arrives per elementary gate, so the noise is
# CORRELATED with g and Schur is unavailable.  Appendix F's gate-noise study
# (shadow_experiments.gate_noise_twirl) measures the consequence numerically
# and finds the twirled-native Z0 residual second order in gamma where the
# projective one is linear.  That is the claim this section proves.
#
# shadow_experiments.py is never lifted in place -- its npz keys are frozen --
# so this is a write-free COMPANION, and it reframes the claim: prove the
# theorem over a symbolic generic state, not over the TFIM ground state at
# seven values of gamma.  Generic in the state, in gamma, and in the modeled
# dilation strength; nothing is read from data/shadow_experiments.npz, so the
# two pillars stay independent.
# ---------------------------------------------------------------------------

# The BFS alphabets of the two atlases used here: 2T is <X, Z, F>, 2O adds H
# and S.  (2I adds Phi at depth <= 4; the theorem needs only the T and O
# draws, so it is left out rather than paid for.)  atlas_gates() is left
# alone deliberately -- four checks iterate it and would change if it grew.
NOISE_GATES = ("X", "Z", "F", "H", "S")

_NOISE_ROT = {}


def _noise_gate_rotations():
    """Exact SO(3) matrices of NOISE_GATES, memoized (bloch_matrix simplifies)."""
    if not _NOISE_ROT:
        G = dict(atlas_gates())
        G["H"] = (1 / sqrt(2)) * Matrix([[1, 1], [1, -1]])
        G["S"] = Matrix([[1, 0], [0, sI]])
        _NOISE_ROT.update({n: bloch_matrix(G[n]) for n in NOISE_GATES})
    return _NOISE_ROT


def _parse_word(seq):
    """Atlas sequence string -> [(gate, dagger), ...] in operator order.

    'F X' means F.X, so X acts first; a trailing dagger marks the SU(2)
    inverse.  main.py's convention, and the one shadow_experiments.py reads.
    """
    return [] if str(seq) == "I" else [(t.rstrip("†"), t.endswith("†"))
                                       for t in str(seq).split()]


def exact_word_rotation(tokens, ROT):
    """SO(3) matrix of a word, exactly -- leftmost token outermost."""
    M = sp.eye(3)
    for base, dag in tokens:
        M = M * (ROT[base].T if dag else ROT[base])
    return M


def exact_draw(g, mode="bfs"):
    """One representative word per SO(3) element of g, with its exact rotation.

    The +-q pair collapses by EXACT rotation equality: T's and O's rotations
    are integer signed permutations, so the key is a tuple of integers --
    canonical and hashable with no field needed.  Shallowest word wins, first
    atlas row breaks ties.  That is shadow_experiments.load_series's selection
    rule with its 9-decimal rounding key removed, and the rotations are built
    from the WORDS rather than lifted from floats, so no grid enters at all.
    Both directions are then checked against the atlas.
    """
    atlas = load_atlas(g)
    toks = [_parse_word(s) for s in atlas[f"{mode}_sequences"]]
    if mode == "bfs":                    # the stored depth IS the word length
        assert all(len(t) == d for t, d in zip(toks, atlas["bfs_depths"])), g
    ROT = _noise_gate_rotations()
    reps = {}
    for i, tk in enumerate(toks):
        k = tuple(exact_word_rotation(tk, ROT))
        if k not in reps or len(tk) < len(toks[reps[k]]):
            reps[k] = i
    idx = sorted(reps.values())
    Rs = [exact_word_rotation(toks[i], ROT) for i in idx]
    assert {tuple(M) for M in Rs} == {tuple(M) for M in exact_rotations(g)}, g
    for i, M in zip(idx, Rs):            # ... and each word IS its atlas row
        f = rotation_from_unitary(atlas["unitaries"][i])
        assert np.abs(np.array(M, dtype=float) - f).max() < 1e-9, (g, mode, i)
    return [toks[i] for i in idx], Rs


def gate_noise_channel(tokens, gam, ROT):
    """Affine Bloch channel of the word with amplitude damping after each gate.

    Damping is r -> diag(c, c, c^2) r + (0, 0, 1-c^2) with c = sqrt(1-gamma),
    exactly; shadow_experiments.chan_amp_damping is its float twin.  Tokens
    act rightmost-first, so the leftmost gate is applied last.
    """
    Tn, tn = sp.diag(sqrt(1 - gam), sqrt(1 - gam), 1 - gam), Matrix([0, 0, gam])
    T, t = sp.eye(3), sp.zeros(3, 1)
    for base, dag in reversed(tokens):
        R = ROT[base].T if dag else ROT[base]
        T, t = Tn * (R * T), Tn * (R * t) + tn
    return T, t


def _estimator_R1(Rs, chans, v):
    """R1's estimator channel under g-correlated noise, in eta units.

    The same average channel_R1 forms -- readout probability times snapshot
    vertex -- but with the noise INSIDE the word rather than after it.  Written
    in the shadow module's normalisation (ideal eta = 1/3, not kappa = 1); the
    calibrated residual below is a ratio and cannot see the convention.
    """
    n = len(Rs)
    M = sum((Matrix(3, 3, lambda a, b: (R.T * v)[a] * (T.T * v)[b])
             for R, (T, _) in zip(Rs, chans)), sp.zeros(3, 3)) / n
    m = sum(((R.T * v) * v.dot(t) for R, (_, t) in zip(Rs, chans)),
            sp.zeros(3, 1)) / n
    return M, m


def _estimator_R2(Rs, chans, TN=None, tN=None):
    """R2's estimator channel under the same g-correlated noise.

    The solid enters only through (1/V) sum_k n_k n_k^T, which is Id_3/3 for
    all five (each is at least a 2-design) -- so it contracts away and the
    channel is solid-independent, the fact the shadow study asserts row by row
    at 1e-12 and this makes structural.  (TN, tN) model the fixed dilation as
    one g-independent pre-measurement channel.
    """
    n = len(Rs)
    if TN is None:
        TN, tN = sp.eye(3), sp.zeros(3, 1)
    M = sum((R.T * TN * T for R, (T, _) in zip(Rs, chans)), sp.zeros(3, 3)) / (3 * n)
    m = sum((R.T * (TN * t + tN) for R, (_, t) in zip(Rs, chans)),
            sp.zeros(3, 1)) / (3 * n)
    return M, m


def calibrated_residual(M, m, r):
    """(M r + m)/eta - r: the bias surviving the scalar |0> calibration."""
    return (M * r + m) / (M[2, 2] + m[2]) - r


def _taylor(X, var, k):
    """Exact k-th Taylor coefficient of a matrix/vector in var at var = 0."""
    return X.applyfunc(lambda e: sp.simplify(sp.diff(e, var, k).subs(var, 0))
                       / sp.factorial(k))


def _prefix_sum(tokens, ROT):
    """sum over the word's prefixes of P^T zhat -- one word's share of m1.

    The derivation this checks: to first order the insertion after gate i
    contributes S_i zhat to t_g, and post-processing hits it with R_g^T =
    (S_i P_i)^T, so R_g^T S_i = P_i^T S_i^T S_i = P_i^T.  Every insertion
    reaches the estimator through its PREFIX alone; the suffix cancels.
    """
    tot = Matrix([0, 0, 0])
    P = sp.eye(3)
    for base, dag in reversed(tokens):
        P = (ROT[base].T if dag else ROT[base]) * P
        tot += P.T * Matrix([0, 0, 1])
    return tot


def _mz_reachable(g, mode):
    """{(m1) over EVERY choice of atlas representative}, by Minkowski sum.

    Each SO(3) element offers two atlas words (the +-q pair), so there are
    2^|G| representative sets -- 2^24 for O.  m1 is a SUM of independent
    per-element contributions, so the reachable set is a Minkowski sum and a
    running set computes it without enumerating the choices.
    """
    atlas = load_atlas(g)
    toks = [_parse_word(s) for s in atlas[f"{mode}_sequences"]]
    ROT = _noise_gate_rotations()
    by_rot = {}
    for i, tk in enumerate(toks):
        by_rot.setdefault(tuple(exact_word_rotation(tk, ROT)), []).append(i)
    assert all(len(v) == 2 for v in by_rot.values()), g   # exactly the +-q pair
    reach = {(0, 0, 0)}
    for k in sorted(by_rot, key=str):
        opts = {tuple(_prefix_sum(toks[i], ROT)) for i in by_rot[k]}
        reach = {tuple(a + b for a, b in zip(r, o)) for r in reach for o in opts}
    scale = Rational(1, 3 * len(by_rot))
    return {tuple(sp.nsimplify(c) * scale for c in r) for r in reach}


def _exact_seed(solid):
    """The alignment vertex v0, exactly -- via the field, never via nsimplify.

    A float alignment() lifted by nsimplify would be a normalising heuristic in
    front of an exact computation, which the exact layer never admits;
    exact_alignment already supplies the seed canonically, tie-break and all
    (the vertex order it breaks the tie on is load-bearing).
    """
    K = solid_field(solid)
    _, v = exact_alignment(exact_vertices(solid, K), K)
    return Matrix([K.to_sympy(c) for c in v])


def check_gate_noise_residual():
    # Appendix F's gate-noise finding, as a theorem: the twirled-native Z0
    # residual starts at gamma^2 while the projective one is linear. Proved
    # over a GENERIC state r = (x, y, z), so it is a statement about the
    # protocol rather than about the study's TFIM test vector.
    gam, dil = sp.symbols("gamma delta", nonnegative=True)
    r = Matrix(sp.symbols("x y z", real=True))
    ROT = _noise_gate_rotations()
    third = sp.eye(3) / 3

    draws, chans = {}, {}
    for g in ("T", "O"):
        toks, Rs = exact_draw(g)
        draws[g] = (toks, Rs)
        chans[g] = [gate_noise_channel(tk, gam, ROT) for tk in toks]

    # the dilation modeled generically: damping of arbitrary strength delta,
    # so the study's GAMMA_DIL = 0.05 is one point of a proved family
    TN = sp.diag(sqrt(1 - dil), sqrt(1 - dil), 1 - dil)

    rows = []
    for label, M, m in (
            ("R1 projective  O draw / octahedron",
             *_estimator_R1(draws["O"][1], chans["O"], Matrix([0, 0, 1]))),
            ("R1 projective  T draw / icosahedron",
             *_estimator_R1(draws["T"][1], chans["T"], _exact_seed("icosahedron"))),
            ("R2 native      T draw / bare",
             *_estimator_R2(draws["T"][1], chans["T"])),
            ("R2 native      T draw / dilation delta",
             *_estimator_R2(draws["T"][1], chans["T"], TN, Matrix([0, 0, dil]))),
            ("R2 native      O draw / bare (control)",
             *_estimator_R2(draws["O"][1], chans["O"])),
            # the like-for-like control of the row Appendix F prints: the O
            # words in front of the same modeled dilation (asserted after the
            # dilated T row's constants; appended last so rows[2], rows[3] hold)
            ("R2 native      O draw / dilation delta",
             *_estimator_R2(draws["O"][1], chans["O"], TN, Matrix([0, 0, dil])))):
        M0, m0 = _taylor(M, gam, 0), _taylor(m, gam, 0)
        M1, m1 = _taylor(M, gam, 1), _taylor(m, gam, 1)
        assert sp.simplify(M0 - M0[2, 2] * sp.eye(3)) == sp.zeros(3, 3), label
        assert m0 == sp.zeros(3, 1), label       # gamma = 0 twirls exactly
        c1 = _taylor(calibrated_residual(M, m, r), gam, 1)
        rows.append((label, M1, m1, c1, M, m))

    lin = {label: sp.simplify(c1[2]) for label, _, _, c1, _, _ in rows}
    print(f"  {'draw / protocol':38s} {'M1 diag':>7s} {'(m1)_z':>22s}"
          f"  Z0 residual")
    print("  " + "-" * 88)
    for label, M1, m1, _, _, _ in rows:
        diag = all(M1[a, b] == 0 for a in range(3) for b in range(3) if a != b)
        print(f"  {label:38s} {('yes' if diag else 'NO'):>7s} {str(m1[2]):>22s}"
              f"  {'SECOND ORDER' if lin[label] == 0 else 'linear'}")
    print("\n  d/dgamma of the |0>-calibrated Z0 residual, exact and for a generic state:")
    for label in lin:
        print(f"    {label:38s} {sp.collect(sp.expand(lin[label]), sqrt(5))}")

    # the two R2 rows on the T draw vanish identically in (x, y, z) -- and the
    # dilated one for EVERY delta, not just the study's 0.05
    assert lin["R2 native      T draw / bare"] == 0
    assert lin["R2 native      T draw / dilation delta"] == 0
    # ... and the three others do not, so the vanishing is not vacuous
    assert lin["R1 projective  O draw / octahedron"] == (1 - r[2]) / 4
    assert lin["R2 native      O draw / bare (control)"] == (1 - r[2]) / 6
    assert lin["R1 projective  T draw / icosahedron"].has(r[1])   # not even diagonal

    # the exact constants behind those verdicts, and the sharpness of gamma^2.
    # They are the BARE row's; the DILATED row Appendix F prints has its own,
    # asserted straight after. The VANISHING survives the dilation (asserted
    # above, over every delta); the constants do not.
    label_b, M1b, m1b, _, Mb, mb = rows[2]
    assert label_b.endswith("/ bare")      # not the dilated row Appendix F prints
    assert M1b == -third and m1b == Matrix([1, 1, 0]) / 18
    quad = _taylor(calibrated_residual(Mb, mb, r), gam, 2)
    assert sp.simplify(quad[2]) == (r[2] - 1) / 12          # zero only at |0>
    assert sp.simplify(_taylor(calibrated_residual(Mb, mb, r), gam, 1)[0]) == Rational(1, 6)
    print("\n  twirled-native on the T draw, bare row: M1 = -Id_3/3, m1 = (1,1,0)/18.")
    print("  M1 is SCALAR, so the scalar |0> calibration absorbs it entirely and the")
    print("  linear Z0 coefficient collapses to (m1)_z (1-z)/eta_0 -- which is 0. What")
    print("  survives at first order is transverse and state-independent: the X0 slope")
    print("  is (1/18)/(1/3) = 1/6 exactly. Z0 resumes at gamma^2 with coefficient")
    print("  (z-1)/12, vanishing only at z = 1, the calibration state itself.")

    # The same two constants for the DILATED row -- the one Appendix F prints,
    # so they belong in an assert rather than in a sentence about one. delta is
    # carried symbolically through the derivation, exactly as above, and fixed
    # at the study's 1/20 only here, where the constants live.
    label_d, _, _, _, Md, md = rows[3]
    assert label_d.endswith("T draw / dilation delta")
    res_d = calibrated_residual(Md.subs(dil, Rational(1, 20)),
                                md.subs(dil, Rational(1, 20)), r)
    quad_d, slope_d = _taylor(res_d, gam, 2)[2], _taylor(res_d, gam, 1)[0]
    assert sp.simplify(quad_d - (4 * sqrt(95) - 19) * (r[2] - 1) / 244) == 0
    assert sp.simplify(slope_d - (Rational(295, 488) - 15 * sqrt(95) / 244) * r[0]
                       - (4 * sqrt(95) - 19) / 122) == 0
    print("\n  the dilation moves both constants and neither verdict. At delta = 1/20")
    print("  the Z0 gamma^2 coefficient is (4 sqrt(95) - 19)(z - 1)/244, which is")
    print(f"  {float(quad_d.subs(r[2], 0)):.8f} at z = 0, and the X0 slope gains an x:")
    print(f"    {slope_d}")

    # F.3.4's mechanism sentence -- "to first order in gamma the 2T words'
    # estimator channel is a diagonal matrix, whose zz-entry the calibration
    # absorbs, plus an offset with no z-component, where the 2O words' offset
    # has one" -- on the DILATED row Appendix F prints, delta still symbolic,
    # so "for every state and modeled dilation strength" is the theorem and
    # not the study's 1/20. Exact identities, no tolerance.
    # (a) M1 of the dilated T row is DIAGONAL -- and diagonal is the word, not
    #     scalar: transverse entries 13 delta/72 - 11 sqrt(1-delta)/72 - 13/72
    #     against a zz-entry delta/9 - 2 sqrt(1-delta)/9 - 1/9, apart by
    #     5 sqrt(1-delta) (1 - sqrt(1-delta))/72 > 0 on all of (0, 1), so on
    #     [0, 1) it is scalar at delta = 0 alone (the bare row's -Id_3/3). The
    #     calibration divides the Z0 column by M[2,2] + m[2] and nothing else,
    #     so a diagonal M1 is exactly what it absorbs there; scalar was never
    #     what the vanishing needed.
    # (b) (m1)_z of that row is 0 identically in delta; the transverse entries
    #     are (1 - delta)/18, the dilation shrinking them without tilting.
    # (c) the bare O row's (m1)_z is 1/18 -- the z-component "the 2O words'
    #     offset has", and the reason its lin[...] above is (1 - z)/6.
    M1_dil, m1_dil = _taylor(Md, gam, 1), _taylor(md, gam, 1)
    assert all(M1_dil[a, b] == 0 for a in range(3) for b in range(3) if a != b)
    xx_dil = 13 * dil / 72 - 11 * sqrt(1 - dil) / 72 - Rational(13, 72)
    zz_dil = dil / 9 - 2 * sqrt(1 - dil) / 9 - Rational(1, 9)
    assert sp.simplify(M1_dil[0, 0] - xx_dil) == 0
    assert sp.simplify(M1_dil[1, 1] - xx_dil) == 0
    assert sp.simplify(M1_dil[2, 2] - zz_dil) == 0
    assert sp.simplify(xx_dil - zz_dil
                       - 5 * sqrt(1 - dil) * (1 - sqrt(1 - dil)) / 72) == 0
    assert sp.simplify(M1_dil.subs(dil, 0) + third) == sp.zeros(3, 3)
    assert m1_dil[2] == 0
    assert sp.simplify(m1_dil[0] - (1 - dil) / 18) == 0
    assert sp.simplify(m1_dil[1] - (1 - dil) / 18) == 0
    label_o, _, m1_o, _, _, _ = rows[4]
    assert label_o.endswith("O draw / bare (control)")
    assert m1_o[2] == Rational(1, 18)
    # (d) The sentence's other restoration, "taking the twirled-native average
    #     over 2O restores [the linear term]", was asserted above on the BARE O
    #     row alone. On the dilated O row -- the like-for-like control of the
    #     row F prints -- the linear Z0 coefficient is
    #     (1 - z)(1 - delta + sqrt(1-delta)) / (4 (1 - delta + 2 sqrt(1-delta)))
    #     exactly: (1 - z)/6 at delta = 0, and nonzero for every z < 1 and
    #     every delta in [0, 1), both factors positive there; at the study's
    #     1/20 on the maximally mixed state it reads sqrt(95)/122 + 21/244 =
    #     0.166. Its (m1)_z is (1 - delta + sqrt(1-delta))/36: the 2O offset
    #     keeps its z-component under the dilation; the 2T offset never had one.
    label_od, _, m1_od, _, _, _ = rows[5]
    assert label_od.endswith("O draw / dilation delta")
    lin_od = ((1 - r[2]) * (1 - dil + sqrt(1 - dil))
              / (4 * (1 - dil + 2 * sqrt(1 - dil))))
    assert sp.simplify(lin[label_od] - lin_od) == 0
    assert sp.simplify(lin_od.subs(dil, 0) - (1 - r[2]) / 6) == 0
    v_od = lin[label_od].subs({dil: Rational(1, 20), r[0]: 0, r[1]: 0, r[2]: 0})
    assert sp.simplify(v_od - (sqrt(95) / 122 + Rational(21, 244))) == 0
    assert float(v_od) > 0.1                      # 0.166: linear, not vanishing
    assert sp.simplify(m1_od[2] - (1 - dil + sqrt(1 - dil)) / 36) == 0
    print("\n  on the dilated T row M1 is DIAGONAL but not scalar -- transverse entries")
    print(f"    {xx_dil}")
    print(f"  against a zz-entry {zz_dil}, apart by")
    print("  5 sqrt(1-delta)(1 - sqrt(1-delta))/72 -- and its (m1)_z is 0 identically in")
    print("  delta (transverse (1-delta)/18). The bare O row's (m1)_z is 1/18, the dilated")
    print("  O row's (1 - delta + sqrt(1-delta))/36, whose linear Z0 coefficient")
    print("  (1-z)(1 - delta + sqrt(1-delta))/(4(1 - delta + 2 sqrt(1-delta))) reads")
    print(f"  {float(v_od):.6f} at delta = 1/20 on the maximally mixed state: linear again")

    # The mechanism, checked rather than asserted: R_g^T S_i = P_i^T, so every
    # insertion reaches the estimator through its prefix and m1 is a sum over
    # prefixes. It is the PREFIX MULTISET that decides the verdict.
    for g in ("T", "O"):
        toks, _ = draws[g]
        pre = sum((_prefix_sum(tk, ROT) for tk in toks), Matrix([0, 0, 0]))
        assert pre / (3 * len(toks)) == _taylor(
            _estimator_R2(draws[g][1], chans[g])[1], gam, 1), g

    # ... which is why the verdict is a fact about the DRAW, not the protocol.
    # Over every one of the 2^12 representative choices for T the z-share
    # cancels; over all 2^24 for O it never does, in either compilation.
    print(f"\n  {'draw':6s} {'mode':5s} {'#words':>6s} {'distinct m1':>12s}"
          f" {'(m1)_z over ALL 2^n representative choices':>46s}")
    print("  " + "-" * 82)
    for g, n in (("T", 12), ("O", 24)):
        for mode in ("bfs", "dij"):
            reach = _mz_reachable(g, mode)
            zs = sorted({m[2] for m in reach}, key=float)
            hit = sum(1 for m in reach if m[2] == 0)
            assert (hit == len(reach)) == (g == "T"), (g, mode)
            rng = f"{zs[0]}" if len(zs) == 1 else f"{zs[0]} ... {zs[-1]}"
            print(f"  {g:6s} {mode:5s} {n:>6d} {len(reach):>12d}"
                  f" {rng + (' (always 0)' if hit == len(reach) else ' (never 0)'):>46s}")

    # Transcription guard: the verdicts above are booleans and exact
    # rationals, so pair them with a value-for-value agreement against a float
    # recomputation of the same channels -- the shadow study's arithmetic.
    worst = 0.0
    for gv in (0.01, 0.05):
        for g in ("T", "O"):
            toks, Rs = draws[g]
            Rf = [np.array(M, dtype=float) for M in Rs]
            ROTf = {n: np.array(ROT[n], dtype=float) for n in NOISE_GATES}
            cf = []
            for tk in toks:
                Tn = np.diag([np.sqrt(1 - gv), np.sqrt(1 - gv), 1 - gv])
                tn = np.array([0.0, 0.0, gv])
                T, t = np.eye(3), np.zeros(3)
                for base, dag in reversed(tk):
                    R = ROTf[base].T if dag else ROTf[base]
                    T, t = Tn @ (R @ T), Tn @ (R @ t) + tn
                cf.append((T, t))
            Mf = np.mean([R.T @ T for R, (T, _) in zip(Rf, cf)], axis=0) / 3
            mf = np.mean([R.T @ t for R, (_, t) in zip(Rf, cf)], axis=0) / 3
            Me, me = _estimator_R2(Rs, chans[g])
            worst = max(worst,
                        np.abs(np.array(Me.subs(gam, gv).evalf(), dtype=float) - Mf).max(),
                        np.abs(np.array(me.subs(gam, gv).evalf(),
                                        dtype=float).ravel() - mf).max())
    assert worst < 1e-12, worst
    print(f"\n  transcription guard: the exact channels reproduce a float recomputation")
    print(f"  of the same words at gamma = 0.01 and 0.05, max |difference| = {worst:.1e}")

    print("[ok] the twirled-native Z0 residual is SECOND order in the per-gate damping")
    print("     and the projective one is linear -- for every state, and for every")
    print("     dilation strength, not at the study's seven gammas and one test vector.")
    print("     It needs BOTH, and each half is shown necessary above: R1 over the SAME")
    print("     T words is linear, and R2 over the O draw is linear again at (1-z)/6 --")
    print("     for all 2^24 choices of representative. What the twirled-native average")
    print("     buys is a SCALAR M1; what the T words add is a z-balanced prefix multiset")


# ---------------------------------------------------------------------------
# Section 0: pin the claims to the canonical data
# ---------------------------------------------------------------------------

def check_canonical_data():
    solids_sym = symbolic_solids()
    for solid in SOLIDS:
        s = load_vertices(solid)
        E = load_elements(solid)
        sym = np.array([[float(c) for c in v] for v in solids_sym[solid]])
        assert len(s) == len(sym)
        for v in s:                  # vertex sets agree as sets, with margin:
            near = np.sort(np.linalg.norm(sym - v, axis=1))[:2]
            assert near[0] < 1e-12 and near[1] > 0.1, (solid, v, near)
        V = len(s)                   # elements are (1/V)(I + n . sigma)
        expected = (np.eye(2)[None] + np.einsum("kn,nab->kab", s, PAULI)) / V
        d_E = np.abs(E - expected).max()
        assert d_E < 1e-12, (solid, d_E)
    print("[ok] npz vertices match the exact symbolic solids; elements = (1/V)(I + n.sigma)")

    gates_npz = np.load(DATA + "gates.npz", allow_pickle=True)
    names = [str(n) for n in gates_npz["names"]]
    for name, U in atlas_gates().items():
        A = np.array(U.evalf(), dtype=complex)
        B = gates_npz["su2"][names.index(name)]
        # |tr(A^dag B)| = 2 iff A = +-B: the ONLY guard that the symbolic gates
        # and gates.npz agree. Deviation is 0.0 exactly.
        d_tr = abs(abs(np.trace(A.conj().T @ B)) - 2)
        assert d_tr < 1e-12, f"gate {name} != +-gates.npz (|dev| = {d_tr:.2e})"
    print("[ok] symbolic gates match gates.npz projectively (X, Z, F, Phi)")

    for g in ("T", "O", "I"):
        R = load_rotations(g)
        U2 = load_atlas(g)["unitaries"]
        keys_R = {rot_key(Rg) for Rg in R}
        keys_U = {rot_key(rotation_from_unitary(U)) for U in U2}
        assert keys_R == keys_U and len(keys_R) == len(R)
    print("[ok] group_2X unitaries project onto exactly the group_X rotations (+-U pair up)")

    keys_T = {rot_key(Rg) for Rg in load_rotations("T")}
    for g in ("O", "I"):
        assert keys_T <= {rot_key(Rg) for Rg in load_rotations(g)}
    print("[ok] T < O and T < I as rotation groups (2T sits inside both 2O and 2I)")


# ---------------------------------------------------------------------------
# Section 1: findings 1 + 2 -- two protocols, two scalars
# ---------------------------------------------------------------------------

def check_two_protocols():
    kappa_R1 = T_NOISE[2, 2]
    kappa_R2 = np.trace(T_NOISE) / 3
    print(f"generic probe noise:  T_zz = {kappa_R1:.6f}   tr(T)/3 = {kappa_R2:.6f}")
    print(f"\n{'solid':14s} {'grp':4s} {'R1 (randomized-projective)':>27s}   {'R2 (twirled-native)':>21s}")
    print("-" * 72)
    for solid in SOLIDS:
        s = load_vertices(solid)
        R = load_rotations(COVARIANCE[solid])
        # R2 exists for every solid: unbiased, exactly depolarizing at tr(T)/3.
        # Worst M deviation over all five solids and both protocols is 1.3e-14
        # (dodecahedron, R2 at zero noise).
        M0, o0 = channel_R2(s, R, np.eye(3), np.zeros(3))
        d_M0 = np.abs(M0 - np.eye(3)).max()
        assert d_M0 < 1e-10, f"{solid}: R2 biased at zero noise (max |dev| = {d_M0:.2e})"
        assert np.allclose(o0, 0, atol=1e-9)
        M2, o2 = channel_R2(s, R, T_NOISE, t_NOISE)
        d_M2 = np.abs(M2 - kappa_R2 * np.eye(3)).max()
        assert d_M2 < 1e-10, \
            f"{solid}: R2 not depol at tr(T)/3 (max |dev| = {d_M2:.2e})"
        assert np.allclose(o2, 0, atol=1e-9)
        # R1 exists only for the antipodal solids
        if is_decomposable(s):
            hits = orbit_counts(s, R)    # the coin: uniform over the vertices
            assert hits.min() == hits.max() == 2 * len(R) // len(s)
            M0, o0 = channel_R1(s, R, np.eye(3), np.zeros(3))
            d_M0 = np.abs(M0 - np.eye(3)).max()
            assert d_M0 < 1e-10, \
                f"{solid}: R1 biased at zero noise (max |dev| = {d_M0:.2e})"
            assert np.allclose(o0, 0, atol=1e-9)
            M1, o1 = channel_R1(s, R, T_NOISE, t_NOISE)
            d_M1 = np.abs(M1 - kappa_R1 * np.eye(3)).max()
            assert d_M1 < 1e-10, \
                f"{solid}: R1 not depol at T_zz (max |dev| = {d_M1:.2e})"
            assert np.allclose(o1, 0, atol=1e-9)
            r1 = f"depol, kappa = {M1[0, 0]:.6f}"
        else:
            assert solid == "tetrahedron"
            r1 = "undefined (no antipodes)"
        r2 = f"depol, kappa = {M2[0, 0]:.6f}"
        print(f"{solid:14s} {COVARIANCE[solid]:4s} {r1:>27s}   {r2:>21s}")
    print("\n[ok] R1: exactly depolarizing at kappa = T_zz for the four antipodal solids;")
    print("[ok] R2: exactly depolarizing at kappa = tr(T)/3 for ALL FIVE (SIC included);")
    print("     offsets vanish, both estimators unbiased at zero noise")
    # The sharpness half of the laundering: with NO draw the channel is the
    # noise map ITSELF -- M = T and off = t exactly, by the frame condition
    # (3/V) sum n n^T = Id and sum n = 0 -- so a fixed coherent error
    # survives whole (an offset, and a tilt: bias FIRST order in the error)
    # until a draw is layered on. The laundering is the draw's doing, not
    # the POVM's. Asserted nowhere else; an identity, not a tolerance.
    # It doubles as the two-list theorem's irreducibility hypothesis failing
    # visibly: R = [Id] is reducible, and the offset that vanishes for every
    # irreducible draw and every belief list is here (3/V) B t -- t itself
    # in this no-mismatch case, where B = (V/3) Id.
    for solid in SOLIDS:
        s = load_vertices(solid)
        M, off = channel_R2(s, [np.eye(3)], T_NOISE, t_NOISE)
        d_M = np.abs(M - T_NOISE).max()          # worst over the five: 2.2e-16
        d_o = np.abs(off - t_NOISE).max()
        assert d_M < 1e-12, f"{solid}: undrawn channel != T (max |dev| = {d_M:.2e})"
        assert d_o < 1e-12, f"{solid}: undrawn offset != t (max |dev| = {d_o:.2e})"
    print("[ok] sharpness: with NO draw the channel is T itself, offset t, exactly")
    print("     (the frame condition) -- what the twirl removes is really there")


def check_calibration_mismatch():
    """The weight-w mismatch law behind Appendix F.3.1's mismatch paragraph,
    proved on that appendix's own estimator: a calibration constant carried
    across protocols is a BIAS, where every fixed misspecification of the
    measurement itself is a premium.

    The hinge is that a calibration is EMPIRICAL. Learned on the run it
    reconstructs, it absorbs whatever the apparatus actually did -- an
    inexact gate, a wrong list, any fixed (T, t) -- which is why those
    cost 1/kappa^(2w) in shots on a weight-w term and nothing in truth.
    A constant carried over from the other protocol was never learned on
    this run. The estimator divides by the believed constant once per
    touched site, so a weight-w Pauli term is multiplied by exactly
    (kappa_run/kappa_cal)^w, and no shot count moves it back -- the same
    law by which the appendix's uncalibrated dual shrinks by (1-p)^w.

    This is a write-free companion: shadow_experiments is imported lazily
    (its import of this module is function-local too, so there is no cycle;
    importing it runs nothing and writes nothing), and
    nothing is read from its npz -- every number is recomputed through its
    noisy_effects -> born_tensor -> exact_estimator_mean pipeline on its
    critical TFIM ground state, the code path that produced the appendix's
    tables. A twirled run of either protocol enters as its post-twirl
    effective measurement, the ideal effects shrunk by kappa_run; that is
    the same per-site OPERATOR, not merely the same channel --
    sum_k (s_ka / eta_cal) E~_k = (kappa_run / kappa_cal) sigma_a,
    asserted below -- which is why one scalar is all the reconstruction
    ever sees, whatever the solid measured.
    """
    import shadow_experiments as se

    rho = se.density(se.tfim_ground_state(se.G_CRIT))
    truth = {on: se.exact_value(rho, obs) for on, obs in se.OBSERVABLES.items()}
    e_true = truth["E_TFIM"]
    # the appendix's state, pinned against the Jordan-Wigner closed form --
    # the same independent witness shadow_experiments.main() runs
    assert abs(e_true - se.tfim_ground_energy_exact(se.N_QUBITS, se.G_CRIT)) \
        < 1e-9, e_true

    def estimates(s, kappa_run, kappa_cal):
        """Exact estimator means of a run whose channel multiplier is
        kappa_run, reconstructed believing kappa_cal (eta = kappa_cal/3)."""
        p = se.born_tensor(rho, se.noisy_effects(
            s, kappa_run * np.eye(3), np.zeros(3)))
        lut = se.lut_robust_canonical(s, kappa_cal / 3.0)
        return {on: se.exact_estimator_mean(p, obs, lut)
                for on, obs in se.OBSERVABLES.items()}

    def law(kappa_run, kappa_cal, obs):
        ratio = kappa_run / kappa_cal
        return sum(c * ratio ** len(sites)
                   * se.exact_value(rho, [(1.0, sites)]) for c, sites in obs)

    s_ico, s_sic = load_vertices("icosahedron"), load_vertices("tetrahedron")
    kappa_r1, kappa_r2 = T_NOISE[2, 2], np.trace(T_NOISE) / 3

    # the per-site operator identity behind everything below, at a generic
    # nonzero mismatch ratio
    E_eff = se.noisy_effects(s_ico, kappa_r1 * np.eye(3), np.zeros(3))
    single = np.einsum("ak,kij->aij", 3.0 * s_ico.T / kappa_r2, E_eff)
    d_op = np.abs(single - (kappa_r1 / kappa_r2) * PAULI).max()
    assert d_op < 1e-12, f"per-site operator identity fails ({d_op:.2e})"

    # F.3.1's two constants, taken from the protocol channels
    # rather than from the closed forms: dephasing 0.1 twirls to kappa = 1
    # exactly under R1 (T_zz = 1: removed outright) and to 13/15 under R2
    # (eta = 0.2889, the appendix's printed 0.289)
    T_d, t_d = se.chan_dephasing(0.1)
    R_I = load_rotations("I")
    M1, o1 = channel_R1(s_ico, R_I, T_d, t_d)
    d1 = max(np.abs(M1 - np.eye(3)).max(), np.abs(o1).max())
    assert d1 < 1e-10, f"R1 does not remove dephasing (max |dev| = {d1:.2e})"
    kappa_deph = np.trace(T_d) / 3
    M2, o2 = channel_R2(s_ico, R_I, T_d, t_d)
    d2 = max(np.abs(M2 - kappa_deph * np.eye(3)).max(), np.abs(o2).max())
    assert d2 < 1e-10, f"R2 not at tr(T)/3 on dephasing (max |dev| = {d2:.2e})"

    # F.3.1's printed decimals -- "(eta-hat = 0.289 and 0.311 at rate 0.1)",
    # dephasing then amplitude damping under twirled-native -- pinned as the
    # STRINGS the sentence prints, through the :.3f it rounds by, off the
    # channel's own scalar rather than the closed forms 13/45 and
    # (2 sqrt(0.9) + 0.9)/9 (0.28889 and 0.31082: neither within 3e-4 of a
    # rounding boundary, against channels exact to 1e-16). The maps are
    # Section 2.6's: dephasing diag(1-2p, 1-2p, 1), t = 0, from
    # shadow_experiments.chan_dephasing above; amplitude damping
    # diag(sqrt(1-g), sqrt(1-g), 1-g), t = (0, 0, g), built inline and required
    # to match its float twin chan_amp_damping (gate_noise_channel's docstring
    # names the twinning). R2 is solid-blind (finding 1), so the octahedron
    # under its O draw stands in for the icosahedron above. The sentence's R1
    # leg -- randomized-projective sees amplitude damping as T_zz = 1 - g, so
    # eta = (1-g)/3 exactly -- on the octahedron, whose alignment is the
    # identity, so the readout axis is Bloch z itself; 1e-12 against a
    # measured 1.1e-16, both rates.
    assert f"{M2[0, 0] / 3:.3f}" == "0.289", M2[0, 0] / 3
    # ...and the probe sentence beside them, which types the same channel's
    # diagonal and its two readings
    assert [f"{T_NOISE[i, i]:.2f}" for i in range(3)] == ["0.83", "0.71", "0.62"]
    assert f"{np.trace(T_NOISE) / 3:.2f}" == "0.72", np.trace(T_NOISE) / 3
    s_o6, R_O6 = load_vertices("octahedron"), load_rotations("O")
    eta_ad = {}
    for gam_ad in (0.1, 0.05):
        T_ad = np.diag([np.sqrt(1 - gam_ad), np.sqrt(1 - gam_ad), 1 - gam_ad])
        t_ad = np.array([0.0, 0.0, gam_ad])
        T_se, t_se = se.chan_amp_damping(gam_ad)
        assert np.abs(T_ad - T_se).max() < 1e-15 and np.abs(t_ad - t_se).max() < 1e-15
        M2a, o2a = channel_R2(s_o6, R_O6, T_ad, t_ad)
        d2a = max(np.abs(M2a - np.trace(T_ad) / 3 * np.eye(3)).max(), np.abs(o2a).max())
        assert d2a < 1e-10, f"R2 not at tr(T)/3 on damping {gam_ad} ({d2a:.2e})"
        eta_ad[gam_ad] = M2a[0, 0] / 3
        M1a, o1a = channel_R1(s_o6, R_O6, T_ad, t_ad)
        d1a = max(np.abs(M1a - (1 - gam_ad) * np.eye(3)).max(), np.abs(o1a).max())
        assert d1a < 1e-12, f"R1 not at (1-g) Id on damping {gam_ad} ({d1a:.2e})"
        assert abs(M1a[2, 2] / 3 - (1 - gam_ad) / 3) < 1e-12
    assert f"{eta_ad[0.1]:.3f}" == "0.311", eta_ad[0.1]

    # six directed pairs: Experiment 3's anchor both ways, F.3.1's
    # dephasing swap both ways, the probe's swap both ways
    pairs = (("depol 0.1 run, noiseless constant", 0.9, 1.0),
             ("noiseless run, depol 0.1 constant", 1.0, 0.9),
             ("R1 run, R2 constant (dephasing 0.1)", 1.0, kappa_deph),
             ("R2 run, R1 constant (dephasing 0.1)", kappa_deph, 1.0),
             ("R1 run, R2 constant (probe)", kappa_r1, kappa_r2),
             ("R2 run, R1 constant (probe)", kappa_r2, kappa_r1))
    e_est = {}
    worst_law = worst_solid = 0.0
    for label, k_run, k_cal in pairs:
        m_i = estimates(s_ico, k_run, k_cal)
        m_s = estimates(s_sic, k_run, k_cal)
        e_est[label] = m_i["E_TFIM"]
        for on, obs in se.OBSERVABLES.items():
            want = law(k_run, k_cal, obs)
            worst_law = max(worst_law, abs(m_i[on] - want), abs(m_s[on] - want))
            worst_solid = max(worst_solid, abs(m_i[on] - m_s[on]))
    assert worst_law < 1e-9, f"weight-w law fails (max |dev| = {worst_law:.2e})"
    assert worst_solid < 1e-12, \
        f"mismatch bias sees the solid (max |dev| = {worst_solid:.2e})"

    # the control that IS the separator: reconstructed with the constant of
    # the run itself, every observable is exact -- the empirical calibration
    # absorbed the noise, whatever it was
    worst_own = 0.0
    for k in (0.9, kappa_deph, kappa_r1, kappa_r2):
        m = estimates(s_ico, k, k)
        worst_own = max(worst_own, max(abs(m[on] - truth[on]) for on in truth))
    assert worst_own < 1e-9, \
        f"own-constant reconstruction biased (max |dev| = {worst_own:.2e})"

    # the appendix's worked example, pinned at the precision it prints, with
    # the three companion biases the check reports beside it
    bias_a = e_est["depol 0.1 run, noiseless constant"] - e_true
    assert abs(bias_a - 0.7578) < 5e-5, bias_a       # the depolarizing anchor
    e_swap = e_est["R1 run, R2 constant (dephasing 0.1)"]
    assert abs(e_swap + 6.49) < 5e-3, e_swap         # F.3.1's worked example
    assert abs(e_true + 5.23) < 5e-3, e_true
    assert abs((e_swap - e_true) + 1.27) < 5e-3, e_swap - e_true
    b_12 = e_est["R1 run, R2 constant (probe)"] - e_true
    b_21 = e_est["R2 run, R1 constant (probe)"] - e_true
    assert abs(b_12 - 1.04) < 5e-3 and abs(b_21 + 1.33) < 5e-3, (b_12, b_21)

    print("calibration constant mismatch (Appendix F.3.1's mismatch paragraph), on that")
    print("appendix's own exact estimator -- its TFIM ground state, "
          f"E = {e_true:.4f}:")
    print("  per-site operator identity sum_k (s_ka/eta_cal) E~_k = "
          "(kappa_run/kappa_cal)")
    print(f"  sigma_a: max |dev| = {d_op:.1e}")
    print("  weight-w law, term-exact on every observable, six directed "
          "pairs, SIC and")
    print(f"  icosahedron: max |dev| = {worst_law:.1e}; solid-blind at "
          f"{worst_solid:.1e}")
    print(f"  own constant -> exact (max |dev| = {worst_own:.1e}); the other "
          "protocol's -> a bias:")
    print(f"    depol 0.1 under the noiseless dual:  E_TFIM bias "
          f"{bias_a:+.4f}  (exact, not sampled)")
    print(f"    dephasing 0.1, R1 run / R2 constant (eta = "
          f"{kappa_deph / 3:.4f}):  E reads {e_swap:.4f}")
    print(f"      against the true {e_true:.4f} -- bias "
          f"{e_swap - e_true:+.4f}, F.3.1's worked example")
    print(f"    probe noise, the two directions:  {b_12:+.4f} / {b_21:+.4f}")
    print("  F.3.1's decimals, off the channels: twirled-native reads eta = "
          f"{M2[0, 0] / 3:.3f} (dephasing 0.1)")
    print(f"  and {eta_ad[0.1]:.3f} (amplitude damping 0.1); randomized-projective "
          "reads the damping at")
    print("  (1-g)/3 exactly, both rates")
    print("[ok] a constant carried across protocols is a bias, exactly")
    print("     (kappa_run/kappa_cal)^w per weight-w term; learned on the run")
    print("     itself it is no bias at all -- the separator between this "
          "check")
    print("     and every premium above")


def check_alignment():
    for solid in ("octahedron", "cube", "icosahedron", "dodecahedron"):
        s = load_vertices(solid)
        R = load_rotations(COVARIANCE[solid])
        A, v = alignment(s)
        if solid == "octahedron":
            d_I = np.abs(A - np.eye(3)).max()        # measured 0.0 exactly
            assert d_I < 1e-12, f"{solid}: A != I (max |dev| = {d_I:.2e})"
            print(f"  {solid:14s} v0 = zhat already: A = I")
        else:
            gap = min(np.linalg.norm(A - Rg) for Rg in R)
            assert gap > 1e-3, f"{solid}: A is a covariance-group element?!"
            print(f"  {solid:14s} A: v0 = {np.round(v, 4)} -> zhat;"
                  f"  min ||A - R_g|| = {gap:.3f}  (not in G)")
    print("[ok] A = I iff octahedron; otherwise A is no covariance-group element -- and")
    print("     no thesis gate set synthesizes it exactly (vertex-by-vertex, section 3)")
