"""Section 3 (cont.) of the randomized-implementation suite: Decker's five
circuits rebuilt from his formulas in his outcome order, what a skipped
relabelling costs, and the tail-weight functional (finding 5).

randomized_implementations.py is the suite's entry point -- its docstring
carries the R1/R2 protocol definitions, the kappa/eta pin, the findings
index, the module map and the run instructions; `cd code && uv run
randomized_implementations.py` runs the whole suite. The finding proved
here:

  Finding 5  Decker's circuits are a construction, not a drop-in, and the
             second correction is the expensive one. All five are rebuilt
             from his own formulas and shown to reproduce his printed vertex
             lists value for value IN HIS OUTCOME ORDER -- which he fixes
             twice, as the columns of M and as the numbered vertices of his
             figures (the Section 3 (cont.) banner has both). Against that
             datum: the reorientation of finding 3's companion
             (check_reorientation_obstruction) preserves outcome order
             for the TETRAHEDRON alone (1 of its 12; 0 of 24,
             24, 60, 60 elsewhere), so the other four owe a fixed
             permutation in post-processing. Skipping it costs no bias -- no
             relabelling can tilt a twirled estimator -- but shrinks the
             estimator by an overlap kappa, for a 1/kappa^2 shot premium
             that check_decker_outcome_order prices per solid: from 1.54x to
             the dodecahedron's 10621.86x, where kappa = -0.0097 leaves the
             channel indistinguishable from heavy depolarizing noise. The
             premium is exact for every state -- the calibrated single-site
             second moment is (9/kappa^2)(c0 + w.r), c0 = 1/3 and w = 0 --
             which is the ANCHORED draw's doing and not kappa's.

             The price is one instance of a theorem the suite asserts
             directly: channel_R2 takes a second vertex list (what the
             device measures, against what the estimator believes), and for
             an IRREDUCIBLE draw any fixed mismatch twirls to exactly
             tr(B T)/V Id with zero offset, B = sum_k b_k a_k^T pairing
             belief against device -- the frame overlap (1/V) sum_k
             b_k . a_k at T = Id, and Finding 1's tr(T)/3 when the two
             lists agree. Element by element over every reorientation coset
             member m = h R0, the induced labelling prices at
             kappa = tr(m T)/3, noiselessly tr(m)/3, with <kappa> = 0 on
             every solid and <kappa^2> = |T|_F^2/27, noiselessly 1/9 --
             so a single member is never
             "the" per-solid figure, and Table D.4's five (rotation AND
             labelling, against Decker's own order) land inside their coset
             ranges, the tetrahedron at its maximum ((1 + sqrt2)/3, the
             coherent-error law at 45 deg) and the cube at its minimum
             (-1/3). Sharpness: with NO draw the channel is the noise map
             itself, offset and all, exactly (check_two_protocols) -- both
             the laundering and the vanishing offset are the draw's doing,
             not the POVM's. Anchors, closed form, each run in BOTH noise
             settings: believing the antipodes prices at kappa = -1
             (-tr(T)/3 noisy), ANY SIC derangement at -1/3 (the Gram is
             constant off-diagonal -- a degeneracy the noise map lifts to
             nine distinct prices), and over ALL V! bijections
             E[kappa] = 0, E[kappa^2] = 1/(3(V-1)) -- exhaustive at
             V = 4, 6: a scramble is worse on a bigger POVM, and
             noiselessly worse in kind than a premium, kappa vanishing
             outright for 8 of the tetrahedron's 24 relabellings and 200
             of the octahedron's 720; a noise map turns that kill back
             into a (steep) tax, tr(B T) surviving where tr(B) = 0.

             What the correction BUYS -- the fourth moment -- has its own
             exact functional: the single-Pauli estimate's tail weight is
             27 <sum_a w_a^4> over the swept directions, a state-free,
             axis-free, draw-blind property of the posed vertex set
             (check_tail_weight: asserted against the protocol average at
             every pose and draw, the minimal T draw included). Two SOS
             identities pin its range to [9, 27], floor exactly the eight
             cube directions, ceiling exactly the six Pauli axes -- so
             the atlas pose is the tetrahedron's and cube's global
             OPTIMUM and the octahedron's global PESSIMUM, and D.2's
             unreoriented numbers follow exactly: 9 -> 15 for both cubic
             solids (equal in every common pose), 27 -> 12 for the
             octahedron, the 5-designs immovable at 81/5. The
             octahedron's own pose-minimum is 11, exact at the frame of
             signed (1/3, 2/3, 2/3) permutations and PROVED minimal in
             axis-angle, not merely searched, so Decker's 12 nearly
             attains what no pose of it beats. The same functional prices
             Appendix F.3.3's T-draw orbit POVMs: the dodecahedron's q
             splits into 1/3 on its cube eight and 7/9 on its twelve
             (sigma^4 + tau^4 = 7), tails 9 and 21, whose vertex-weighted
             mean is the solid's own 81/5.
"""

import itertools
import math

import numpy as np
import sympy as sp
from sympy import Matrix, Rational

from randomized_core import (SOLIDS, TAU_SYM, SIG_SYM, COVARIANCE, T_NOISE,
                             t_NOISE, load_vertices, load_rotations, rot_key,
                             alignment, channel_R2, REORIENT)

# ---------------------------------------------------------------------------
# Section 3 (cont.): Decker's outcome order, and what mislabelling costs
# ---------------------------------------------------------------------------
#
# check_reorientation_obstruction prices the ROTATION between Decker's pose and
# ours. A reorientation carries vertices to vertices but not indices to indices,
# so a second correction stands between his circuits and our atlas: a relabelling
# of outcomes. Pricing THAT needs a datum the rotation never sees -- his vertex
# list in HIS outcome order -- so the circuits are rebuilt here.
#
# The order is not ours to choose. Decker fixes it twice, and independently:
#
#   (i)  the POVM vectors ARE the columns of M, printed explicitly per solid
#        (Secs. 6-10), and outcome j is column j -- with the input embedded as
#        (psi_0, psi_1, 0, ..., 0) the circuit gives (Mtilde^dag v)_j =
#        conj(Mtilde_{0j}) psi_0 + conj(Mtilde_{1j}) psi_1 = <Psi_j | psi>. The
#        permutation Q in Mtilde = Q(A (x) F) acts on ROWS ("fixes the first
#        row and maps the (m+2)nd row to the second"), so it selects which two
#        rows are M without ever touching the outcome index.
#
#   (ii) each column is tied to a NUMBERED VERTEX of his figures ("the first
#        three vectors correspond to the upper three vertices 1-3"), anchored to
#        a coordinate ("vertex 1 is given by the vector (sqrt(2/3), 0, 1/sqrt3)").
#
# So the check below is self-certifying rather than merely self-consistent: it
# asserts the rebuilt circuit reproduces his printed columns VALUE FOR VALUE, in
# his order, and that column 1 lands on his stated anchor. A wrong convention
# fails it loudly instead of returning a plausible permutation -- which matters,
# because the headline "no reorientation preserves outcome order" is the GENERIC
# answer (the rotation group acts faithfully on vertices, so the coset-to-
# permutation map is injective and exactly 0 or 1 of any coset can preserve
# order). Only the tetrahedron's 1-of-12 is evidence on its own.
#
# The convention that has to be pinned is the Fourier one, and it bites: Decker
# defines F_m = sqrt(1/m)(omega^{jk}) with omega = exp(-2 pi i/m) (Sec. 4), but
# Sec. 7 uses its CONJUGATE for the cube. His printed Mtilde settles it -- the
# row (alpha, alpha i, -alpha, -alpha i) is alpha(1, w, w^2, w^3) with w = +i --
# and Sec. 8 confirms it in passing ("with omega = i as in the dihedral case").
# Same POVM either way; different outcome order. Appendix D defines F_m once,
# in the chapter intro, and names the cube's omega = +i there as the exception.

# Decker's parameters, verbatim from Secs. 6-10, unrescaled (|a|^2 + |b|^2 = 1).
_DA3, _DB3 = math.sqrt((3 + math.sqrt(3)) / 6), math.sqrt((3 - math.sqrt(3)) / 6)
_DP = math.sqrt(75 + 30 * math.sqrt(5)) / 30
_DM = math.sqrt(75 - 30 * math.sqrt(5)) / 30
_DAD, _DBD = math.sqrt(0.5 + _DP), math.sqrt(0.5 - _DP)
_DGD, _DDD = math.sqrt(0.5 + _DM), math.sqrt(0.5 - _DM)   # Sec. 9 (dodecahedron)
_DGI, _DDI = math.sqrt(0.5 - _DM), math.sqrt(0.5 + _DM)   # Sec. 10 (exchanged)

# sign of the exponent in F_m; +1 only for the cube (see the note above)
FOURIER_SIGN = {"tetrahedron": -1, "octahedron": -1, "cube": +1,
                "icosahedron": -1, "dodecahedron": -1}

# "vertex 1 is given by the vector ..." -- Secs. 6, 7, 8 and Secs. 9, 10
_ANCHOR_TOC = (math.sqrt(2 / 3), 0.0, 1 / math.sqrt(3))
_ANCHOR_ID = (math.sqrt((10 - 2 * math.sqrt(5)) / 15), 0.0,
              math.sqrt((5 + 2 * math.sqrt(5)) / 15))
DECKER_ANCHOR = {"tetrahedron": _ANCHOR_TOC, "octahedron": _ANCHOR_TOC,
                 "cube": _ANCHOR_TOC, "icosahedron": _ANCHOR_ID,
                 "dodecahedron": _ANCHOR_ID}


def decker_fourier(m, sign):
    """F_m = sqrt(1/m) (omega^{jk}), omega = exp(sign . 2 pi i/m)."""
    w = np.exp(sign * 2j * np.pi / m)
    return np.array([[w ** (j * k) for k in range(m)]
                     for j in range(m)]) / np.sqrt(m)


def _pad_block(F, dim):
    """F (+) I_{dim-m}: the padding that embeds an m-orbit into a register."""
    out = np.eye(dim, dtype=complex)
    out[:len(F), :len(F)] = F
    return out


def _cnot(nq, ctrl, targ):
    """Q^dag on nq qubits, as every Appendix D figure draws it."""
    P = np.zeros((2 ** nq, 2 ** nq))
    for i in range(2 ** nq):
        b = [(i >> (nq - 1 - p)) & 1 for p in range(nq)]
        if b[ctrl]:
            b[targ] ^= 1
        P[sum(v << (nq - 1 - p) for p, v in enumerate(b)), i] = 1
    return P


def bloch_of_ket(psi):
    """Bloch vector of the (unnormalized, non-zero) state vector psi."""
    rho = np.outer(np.asarray(psi, dtype=complex), np.conj(psi))
    rho = rho / np.trace(rho).real
    return np.array([2 * rho[0, 1].real, -2 * rho[0, 1].imag,
                     (rho[0, 0] - rho[1, 1]).real])


def decker_columns(solid):
    """The POVM vectors as Decker PRINTS them, in his own vertex numbering."""
    if solid == "tetrahedron":                                # Sec. 6, Line (2)
        return [(_DA3, _DB3), (_DA3, -_DB3),
                (_DB3, _DA3 * 1j), (_DB3, -_DA3 * 1j)]
    if solid == "cube":                                       # Sec. 7
        return [(_DA3, _DB3), (_DA3, _DB3 * 1j),
                (_DA3, -_DB3), (_DA3, -_DB3 * 1j),
                (_DB3, -_DA3), (_DB3, -_DA3 * 1j),
                (_DB3, _DA3), (_DB3, _DA3 * 1j)]
    m = 5 if solid == "dodecahedron" else 3
    om = np.exp(-2j * np.pi / m)
    if solid == "octahedron":                                 # Sec. 8, Eq. (4)
        return ([(_DA3, _DB3 * om ** j) for j in range(m)]
                + [(_DB3, -_DA3 * om ** j) for j in range(m)])
    g, d = (_DGI, _DDI) if solid == "icosahedron" else (_DGD, _DDD)
    return ([(_DAD, _DBD * om ** j) for j in range(m)]        # Secs. 9, 10,
            + [(_DBD, -_DAD * om ** j) for j in range(m)]     # Lines (6), (8)
            + [(g, d * om ** j) for j in range(m)]
            + [(d, -g * om ** j) for j in range(m)])


def decker_circuit(solid):
    """Rows of Mtilde^dag . iota -- the circuit exactly as Appendix D draws it.

    Row k is the (conjugated) POVM vector of computational outcome k; rows of
    norm zero are the outcomes the padding leaves unreachable.
    """
    s = FOURIER_SIGN[solid]
    if solid == "tetrahedron":                    # Fig. D.1
        a, b = math.sqrt((3 + math.sqrt(3)) / 12), math.sqrt((3 - math.sqrt(3)) / 12)
        UA, nq, na = np.sqrt(2) * np.array([[a, b], [b, -a]]), 2, 1
        head = (np.kron(np.eye(2), decker_fourier(2, s))
                @ np.diag([1, 1, 1, 1j]) @ np.kron(UA, np.eye(2)))
    elif solid == "octahedron":                   # Fig. D.2
        a, b = math.sqrt((3 + math.sqrt(3)) / 18), math.sqrt((3 - math.sqrt(3)) / 18)
        UA, nq, na = np.sqrt(3) * np.array([[a, b], [b, -a]]), 3, 1
        head = np.kron(UA, _pad_block(decker_fourier(3, s), 4).conj().T)
    elif solid == "cube":                         # Fig. D.3
        a, b = math.sqrt((3 + math.sqrt(3)) / 24), math.sqrt((3 - math.sqrt(3)) / 24)
        UA, nq, na = 2 * np.array([[a, b], [b, -a]]), 3, 1
        head = np.kron(UA, decker_fourier(4, s).conj().T)
    else:                                         # Figs. D.4, D.5
        icos = solid == "icosahedron"
        sc = math.sqrt(1 / 6) if icos else math.sqrt(1 / 10)
        g, d = (_DGI, _DDI) if icos else (_DGD, _DDD)
        a, b, g, d = _DAD * sc, _DBD * sc, g * sc, d * sc
        A = (np.sqrt(3) if icos else np.sqrt(5)) * np.array(
            [[a, b, g, d], [b, -a, d, -g], [g, -d, -a, b], [d, g, -b, -a]])
        d_iso = np.abs(A.conj().T @ A - np.eye(4)).max()
        assert d_iso < 1e-12, (solid, d_iso)
        nq, na = (4, 2) if icos else (5, 2)
        Fb = (_pad_block(decker_fourier(3, s), 4) if icos
              else _pad_block(decker_fourier(5, s), 8))
        head = np.kron(A.conj().T, Fb.conj().T)
    # Q^dag: target the last ancilla-register bit, control the data wire. This
    # is Decker's Q ("fixes the first row, maps the (m+2)nd to the second") for
    # every solid, and the CNOT each Appendix D caption spells out in kets.
    iota = np.zeros((2 ** nq, 2))
    iota[0, 0] = iota[1, 1] = 1
    return head @ _cnot(nq, ctrl=nq - 1, targ=na - 1).T @ iota


def decker_vertices(solid):
    """(V, 3) Bloch vertices in DECKER's outcome order, the live indices, and W.

    W (the circuit rows) comes back with them so that a caller wanting both the
    vertices and the rows they came from need not rebuild the circuit -- the
    kron, the 2^nq CNOT and the isometry check are not cheap at the dodecahedron.
    """
    W = decker_circuit(solid)
    live = [k for k in range(len(W)) if np.linalg.norm(W[k]) > 1e-9]
    return np.array([bloch_of_ket(np.conj(W[k])) for k in live]), live, W


def check_decker_outcome_order():
    KEPT = {"tetrahedron": 1, "octahedron": 0, "cube": 0,
            "icosahedron": 0, "dodecahedron": 0}
    LIVE = {"tetrahedron": 4, "octahedron": 6, "cube": 8,
            "icosahedron": 12, "dodecahedron": 20}
    verdict, kappas = {}, {}

    def second_moment(b, a, R_sm, T_sm, t_sm):
        """(c0, w) of the calibrated single-site SECOND moment under R2 --
        belief list b, device list a, draw R_sm, measurement-side noise (T, t).

        The calibrated single-shot estimate of sigma_alpha is 3 (R_g^T b_k)_alpha
        / kappa, drawn with probability (1/(|G| V))(1 + a_k . (T R_g r + t)), so
        E[o_alpha^2] = (9/kappa^2)(c0_alpha + w_alpha . r) with

            c0_alpha = (1/(V|G|)) sum_g sum_k (R_g^T b_k)_alpha^2 (1 + a_k . t),
            w_alpha  = (1/(V|G|)) sum_g sum_k (R_g^T b_k)_alpha^2  R_g^T T^T a_k.

        Matched and noiseless the moment is 3 per site, so the premium is
        exactly 1/kappa^2 for every state iff c0 = 1/3 and w = 0.  Returns c0
        as a (3,) and w as a (3, 3), alpha down the rows.
        """
        U_sm = np.einsum("gia,ki->gka", R_sm, b) ** 2      # (R_g^T b_k)_alpha^2
        W_sm = np.einsum("gib,ki->gkb", R_sm, a @ T_sm)    # (R_g^T T^T a_k)_beta
        norm_sm = len(b) * len(R_sm)
        return (np.einsum("gka,k->a", U_sm, 1 + a @ t_sm) / norm_sm,
                np.einsum("gka,gkb->ab", U_sm, W_sm) / norm_sm)

    sm_worst, d_ico = {}, None
    print(f"  {'solid':14s} {'F_m':>10s} {'live':>7s} {'his columns':>12s}"
          f" {'anchor':>7s} {'order kept':>11s} {'kappa':>10s} {'premium':>10s}")
    print("  " + "-" * 88)
    for solid in SOLIDS:
        d, live, W = decker_vertices(solid)
        cols = decker_columns(solid)
        n = load_vertices(solid)
        V = len(n)
        assert len(live) == len(cols) == V == LIVE[solid], solid
        assert live == sorted(live)              # padding never reorders
        # (i) his printed columns, value for value, in his order -- and (ii)
        # his stated anchor for vertex 1. This is what makes the check
        # self-certifying: a wrong Fourier convention dies here.
        for j in range(V):
            d_col = np.abs(d[j] - bloch_of_ket(cols[j])).max()
            assert d_col < 1e-10, (solid, j, d_col)
        d_anchor = np.abs(d[0] - DECKER_ANCHOR[solid]).max()
        assert d_anchor < 1e-10, (solid, d_anchor)
        # it really is a POVM, and the solid really is congruent to the atlas's
        d_pov = np.abs(sum(np.outer(np.conj(r), r) for r in W) - np.eye(2)).max()
        assert d_pov < 1e-12, (solid, d_pov)
        d_w = max(abs(np.linalg.norm(W[k]) ** 2 - 2 / V) for k in live)
        assert d_w < 1e-12, (solid, d_w)
        R = np.array(REORIENT[solid][1], dtype=float)
        rots = load_rotations(COVARIANCE[solid])

        def induced(RR):
            """The permutation RR induces on the atlas vertex list, with MARGIN.

            What is decided here is DISCRETE -- a permutation -- so a bare
            tolerance is the wrong guarantee: one wrong entry does not perturb
            the kappa below, it jumps it to a different value.  Measured
            over every match this check makes (all 12+24+24+60+60 coset members
            times every vertex, not only the accepted ones): worst accepted
            distance 7.5e-16, closest runner-up 0.7136 -- the solid's own
            minimum inter-vertex distance.  So the assertion is on the MARGIN,
            match inside 1e-9 and runner-up beyond 0.1, which leaves ~7e8 of
            slack on the reject side.  It also subsumes a rounding-grid vertex
            key: sorted(induced(R)) == range(V) is the same claim -- Table
            D.3's R carries THIS circuit's solid onto the atlas -- decided by
            separation rather than by an 8-decimal grid.
            """
            out = []
            for x in d:
                dist = np.linalg.norm(n - RR @ x, axis=1)
                near = np.sort(dist)[:2]
                assert near[0] < 1e-9 and near[1] > 0.1, (solid, near)
                out.append(int(np.argmin(dist)))
            return out
        # ...and Table D.3's R, derived independently there, carries THIS
        # circuit's solid onto the atlas -- the two derivations meet here
        perm = induced(R)
        assert sorted(perm) == list(range(V)), solid
        # D.2's other mismatch, priced there in words: run U_R but skip the
        # relabelling, so outcome j fires as atlas vertex perm[j] and is read
        # as atlas vertex j. The octahedron's key sends every vertex onto a
        # perpendicular one, so that kappa is exactly 0 -- killed rather than
        # taxed -- while the tetrahedron's key is the identity (its drawn R is
        # the one coset member that keeps outcome order), so its skip is free.
        skip = [float(n[j] @ n[perm[j]]) for j in range(V)]
        if solid == "octahedron":
            assert max(abs(x) for x in skip) < 1e-12, skip
        if solid == "tetrahedron":
            assert perm == list(range(V)), perm
        kept = sum(1 for g in rots if induced(g @ R) == list(range(V)))
        # the rotation group acts faithfully on vertices, so distinct coset
        # members induce distinct permutations: at most one can preserve order
        assert kept <= 1 and kept == KEPT[solid], (solid, kept)
        # kappa: the overlap of the believed list (the atlas, index for index)
        # with the actual one (his). Theorem: a fixed misspecification reaches
        # the estimator as this one scalar, so the bill is a 1/kappa^2 shot
        # premium and never a bias (thesis F.3.2).
        kappa = float(np.mean([n[k] @ d[k] for k in range(V)]))
        assert -1 <= kappa <= 1
        # ...and the PAIRING is fed through the two-list channel itself --
        # believe the atlas list, measure his -- which is what makes this
        # overlap the estimator-channel MULTIPLIER rather than a cosine
        # table: kappa Id noiseless, tr(B T)/V Id at the probe, offset 0.
        # (Watched: a belief/device swap fires the noisy assert on four of
        # the five -- the cube's B = n^T d is symmetric and stays blind.)
        M2, off2 = channel_R2(n, rots, np.eye(3), np.zeros(3), s_actual=d)
        assert np.abs(M2 - kappa * np.eye(3)).max() < 1e-10, solid
        assert np.allclose(off2, 0, atol=1e-9), solid
        kap_n = np.trace(n.T @ d @ T_NOISE) / V
        M2, off2 = channel_R2(n, rots, T_NOISE, t_NOISE, s_actual=d)
        assert np.abs(M2 - kap_n * np.eye(3)).max() < 1e-10, solid
        assert np.allclose(off2, 0, atol=1e-9), solid
        # F.3.2's premium sentence -- a fixed misspecification "multiplies the
        # single-shot second moment of a weight-w term by exactly 1/kappa^{2w}"
        # -- claims more than the two-list channel just asserted, which is the
        # FIRST moment. The second is priced per site by second_moment():
        # E[o^2] = (9/kappa^2)(c0 + w . r), and "exactly 1/kappa^2, for every
        # state" is c0 = 1/3 and w = 0 -- the weight-w power then follows from
        # the product dual, site by site. Asserted on the misspecification the
        # appendix prices, believe the atlas list and measure Decker's, under
        # both draws (the covariance group, and the atlas T draw that suffices
        # for the twirl) and both noise settings. 1e-12 against a measured
        # worst of 1.7e-15 on c0 and 1.9e-17 on w over the twenty cases; the
        # negative control after the table misses by 1.3e-2, ten orders away.
        dc0s, dws = [], []
        for R_sm in (rots, load_rotations("T")):
            for T_sm, t_sm in ((np.eye(3), np.zeros(3)), (T_NOISE, t_NOISE)):
                c0_sm, w_sm = second_moment(n, d, R_sm, T_sm, t_sm)
                dc0s.append(np.abs(c0_sm - 1 / 3).max())
                dws.append(np.abs(w_sm).max())
        assert max(dc0s) < 1e-12, (solid, max(dc0s))
        assert max(dws) < 1e-12, (solid, max(dws))
        sm_worst[solid] = (max(dc0s), max(dws))
        if solid == "icosahedron":
            d_ico = d                    # the negative control's device list
        # cross-check against the ROTATION-ONLY coset prices tr(h R)/3
        # (check_reorientation_obstruction's scan). This kappa is a rotation-
        # AND-labelling fact, so containment in that family's range is an
        # observation, not a law -- the antipodal anchor below prices at -1,
        # outside every coset range here -- but all five do land inside, and
        # two land ON an end. The tetrahedron's is the sharp one: its outcome
        # order survives (kept = 1), so its kappa IS a coset member's price,
        # the maximum. The cube reaches the minimum by a different route --
        # its eight per-vertex overlaps are not the tetrahedral angle at all
        # but a 4/4 split between -(1+sqrt2)/3 and (sqrt2-1)/3, and only
        # their MEAN is -1/3, the sqrt2's cancelling.
        cos_k = [np.trace(h @ R) / 3 for h in rots]
        assert min(cos_k) - 1e-12 < kappa < max(cos_k) + 1e-12, solid
        if solid == "tetrahedron":
            assert abs(kappa - max(cos_k)) < 1e-12
        if solid == "cube":
            assert abs(kappa - min(cos_k)) < 1e-12
            per = np.sort([n[k] @ d[k] for k in range(V)])
            assert np.abs(per[:4] + (1 + math.sqrt(2)) / 3).max() < 1e-12
            assert np.abs(per[4:] - (math.sqrt(2) - 1) / 3).max() < 1e-12
        verdict[solid], kappas[solid] = perm, kappa
        sign = "+" if FOURIER_SIGN[solid] > 0 else "-"
        print(f"  {solid:14s} {f'exp({sign}2pi i/m)':>10s} {len(live):3d}/{2 ** int(np.log2(len(W))):<3d}"
              f" {'reproduced':>12s} {'ok':>7s} {kept:>6d}/{len(rots):<4d}"
              f" {kappa:>+10.6f} {1 / kappa ** 2:>9.2f}x")
    assert abs(kappas["cube"] + 1 / 3) < 1e-12                # exactly -1/3
    # the tetrahedron's is (1 + sqrt2)/3 exactly: its surviving member is
    # R_z(45 deg), so its price is the coherent-error law (1 + 2 cos theta)/3
    # read at theta = 45 deg -- the identity tying this table to the
    # reorientation coset scan
    assert abs(kappas["tetrahedron"] - (1 + math.sqrt(2)) / 3) < 1e-12
    assert min(kappas, key=lambda s: abs(kappas[s])) == "dodecahedron"
    # F.3.2's prose literals, "weight-one premia from 1.54x to 10621.86x",
    # pinned as the strings the sentence prints, through the :.2f the table
    # rounds by. Table D.4 is generated from these same kappas
    # (randomized_fragments), so the TABLE cannot drift from them; the SENTENCE
    # can, and this is its pin -- with its "from ... to": the two are the
    # range's ends (the dodecahedron's minimum |kappa| is the line above).
    assert f"{1 / kappas['tetrahedron'] ** 2:.2f}" == "1.54", kappas["tetrahedron"]
    assert f"{1 / kappas['dodecahedron'] ** 2:.2f}" == "10621.86", kappas["dodecahedron"]
    assert max(kappas, key=lambda s: abs(kappas[s])) == "tetrahedron"
    print("[ok] all five circuits reproduce Decker's printed vertex lists value for")
    print("     value IN HIS OUTCOME ORDER, each anchored on his stated vertex 1.")
    print("     The cube alone needs omega = +i (Sec. 7 conjugates Sec. 4's F_m);")
    print("     under Sec. 4's convention it yields the same POVM misordered.")
    print("[ok] outcome order survives for the TETRAHEDRON alone -- 1 of its 12")
    print("     reorientations, the T-dagger already drawn in Figures 4.1 and D.1.")
    print("     For the other four it is 0 of 24, 24, 60, 60, so each owes the")
    print("     fixed permutation below, applied in classical post-processing:")
    print()
    for solid in SOLIDS:
        print(f"  {solid:14s} {verdict[solid]}")
    print()
    print("[ok] and skipping it is not free. Running his circuit against the atlas")
    print("     list is unbiased -- no relabelling can tilt a twirled estimator --")
    print("     but it shrinks by kappa, for a 1/kappa^2 shot premium: 1.54x, 9.65x,")
    print("     9.00x (kappa = -1/3 exactly), 73.60x, and 10621.86x. The three")
    print("     failure modes each get an exemplar, one per covariance group: the")
    print("     tetrahedron pays a premium and nothing more; the cube flips the")
    print("     sign; and the dodecahedron's kappa = -0.0097 leaves 1/kappa ill-")
    print("     conditioned and the channel indistinguishable from heavy")
    print("     depolarizing noise -- a labelling bug an experimenter would blame")
    print("     on the hardware. The modes nest rather than partition (the")
    print("     dodecahedron is negative too), so that is the worst mode each")
    print("     reaches; the octahedron and icosahedron are the interpolation.")
    print("     The pair worth reading together is the cube against the")
    print("     octahedron: |kappa| = 0.3333 vs 0.3220 -- near-equal by coincidence,")
    print("     not identity -- so 9.00x vs 9.65x, the same bill with opposite")
    print("     signs. The sign flip costs nothing in shots; the whole hazard is a")
    print("     negative estimator-channel factor discarded as an artefact.")
    # The premium's exactness is a property of the ANCHORED draw, not of kappa
    # -- the negative control for the second-moment pins in the loop. c0 = 1/3
    # needs only an irreducible draw (Schur puts every (R_g^T b_k)_alpha^2 at
    # 1/3 on average) and a centred device list (its t-term is mean(a) . t);
    # w is a CUBIC moment, E_g[(R_g^T b)_alpha^2 (R_g^T T^T a)_beta], and it
    # dies because T, O and I in the atlas pose all contain the three
    # coordinate half-turns, each monomial being odd in an axis one of them
    # negates. Conjugate T by an h in I \ T and the draw is still a group,
    # still irreducible, still inside I -- c0 stays 1/3 -- but its half-turns
    # are about h's axes, and on Decker's icosahedron list w comes back at
    # 0.0128 (0.0100 at the probe): a second moment that reads the state, which
    # no single premium prices. With the lists agreeing the same conjugate
    # draw gives w = 0 again, the antipodal list's own odd moments vanishing,
    # so what the pins exclude is the PAIRING of a mismatch with an un-anchored
    # draw. 1e-3 is a gap, not a boundary: fourteen orders above the accepted
    # 1.9e-17, one below the measured 1.3e-2.
    rots_T, rots_I = load_rotations("T"), load_rotations("I")
    keys_T = {rot_key(g) for g in rots_T}
    h_conj = next(g for g in rots_I if rot_key(g) not in keys_T)
    T_conj = np.array([h_conj @ g @ h_conj.T for g in rots_T])
    keys_c = {rot_key(g) for g in T_conj}
    assert len(keys_c) == 12 and keys_c != keys_T           # a DIFFERENT copy of T
    assert keys_c <= {rot_key(g) for g in rots_I}           # ... inside I
    assert all(rot_key(g1 @ g2) in keys_c for g1 in T_conj for g2 in T_conj)
    half_turns = [np.diag(v) for v in ([1., -1, -1], [-1., 1, -1], [-1., -1, 1])]
    for g_name in ("T", "O", "I"):
        assert all(any(np.allclose(g, P, atol=1e-12, rtol=0) for g in load_rotations(g_name))
                   for P in half_turns), g_name
    assert not any(np.allclose(g, P, atol=1e-12, rtol=0) for g in T_conj for P in half_turns)
    n_ico = load_vertices("icosahedron")
    c0_c, w_c = second_moment(n_ico, d_ico, T_conj, np.eye(3), np.zeros(3))
    assert np.abs(c0_c - 1 / 3).max() < 1e-12               # c0 does not see it
    assert np.abs(w_c).max() > 1e-3, np.abs(w_c).max()       # w does: 0.0128
    _, w_cp = second_moment(n_ico, d_ico, T_conj, T_NOISE, t_NOISE)
    # by value, not > 1e-3: w = 0 holds for any fixed map, so this is the only
    # assert that sees how T enters w (transposed 0.0114, dropped 0.0128)
    assert abs(np.abs(w_cp).max() - 0.0100) < 5e-4, np.abs(w_cp).max()
    _, w_cm = second_moment(n_ico, n_ico, T_conj, np.eye(3), np.zeros(3))
    assert np.abs(w_cm).max() < 1e-12                        # no mismatch, no w
    print()
    print("[ok] and the premium is EXACT, for every state: the calibrated single-site")
    print("     second moment is (9/kappa^2)(c0 + w . r) with c0 = 1/3 and w = 0 on all")
    print("     five solids, under the covariance draw and the T draw, at T = Id and at")
    print(f"     the probe -- worst |c0 - 1/3| = {max(v[0] for v in sm_worst.values()):.1e},"
          f" worst |w| = {max(v[1] for v in sm_worst.values()):.1e}. It is the")
    print("     anchored draw's doing, not kappa's: a T-conjugate inside I whose")
    print(f"     half-turns are not the coordinate ones leaves |w| = {np.abs(w_c).max():.4f} on")
    print("     Decker's icosahedron list (c0 still 1/3), a second moment that reads")
    print("     the state.")
    # Labels with no geometry behind them: two closed-form anchors and one
    # exact law, all zero-RNG, each anchor run in BOTH noise settings -- at
    # T = Id the two-list channel is belief/device-symmetric (tr B is), so
    # only the T_NOISE legs, priced independently at tr(B T)/V, can catch a
    # swapped wiring (the six 4-cycle derangements do; the antipodal
    # anchor's B = -(V/3) Id is symmetric and stays blind). Antipodal
    # relabel: believe the antipode list and the two-list channel is -Id
    # exactly (kappa = -1), -tr(T)/3 Id under noise. SIC derangement: the
    # tetrahedron's Gram is constant -1/3 off the diagonal, so ANY
    # fixed-point-free relabelling prices at kappa = -1/3 exactly -- run
    # over all 9 of them, not a representative; T_NOISE lifts the
    # degeneracy to nine DISTINCT prices (asserted). Scramble law,
    # exhaustive over ALL V! bijections: E[kappa] = 0 and E[kappa^2] =
    # 1/(3(V-1)), so a scrambled label is worse on a BIGGER POVM. Worse in
    # KIND, too -- noiselessly, and this is the part that does not survive
    # being read as a premium: 1/(3(V-1)) is a second moment, and at T = Id
    # E[1/kappa^2] does not exist, because kappa vanishes OUTRIGHT on a
    # large minority of bijections -- 8 of 24 at V = 4, 200 of 720 at
    # V = 6, where it is the single commonest outcome. There the noiseless
    # estimator channel is the ZERO map; a noise map revives it, the kill
    # being tr(B) = 0 while tr(B T) survives -- asserted below on one
    # killed bijection, taxed ~186x at the probe. Among the rest the median
    # premium is 9x at V = 4 but 36x at V = 6, against the 9 and 15 that
    # reading 3(V-1) as typical would predict -- right at V = 4 by
    # coincidence, wrong by 2.4x at V = 6. The unprinted residue beside
    # Table D.4, and what it says is that a scrambled label is a different
    # failure from a mispriced one.
    s6, rot6 = load_vertices("octahedron"), load_rotations("O")
    M, off = channel_R2(-s6, rot6, np.eye(3), np.zeros(3), s_actual=s6)
    d_M = np.abs(M + np.eye(3)).max()            # measured 2.1e-33
    assert d_M < 1e-10, f"antipodal relabel not -Id (max |dev| = {d_M:.2e})"
    assert np.allclose(off, 0, atol=1e-9)
    M, off = channel_R2(-s6, rot6, T_NOISE, t_NOISE, s_actual=s6)
    d_M = np.abs(M + (np.trace(T_NOISE) / 3) * np.eye(3)).max()
    assert d_M < 1e-10, f"noisy antipodal relabel not -tr(T)/3 Id ({d_M:.2e})"
    assert np.allclose(off, 0, atol=1e-9)
    s4, rot4 = load_vertices("tetrahedron"), load_rotations("T")
    noisy = []
    for p in itertools.permutations(range(4)):
        if any(p[k] == k for k in range(4)):
            continue
        M, off = channel_R2(s4[list(p)], rot4, np.eye(3), np.zeros(3),
                            s_actual=s4)
        d_M = np.abs(M + np.eye(3) / 3).max()    # worst over all 9: 5.6e-17
        assert d_M < 1e-10, f"SIC derangement {p} off -Id/3 ({d_M:.2e})"
        assert np.allclose(off, 0, atol=1e-9), p
        kap = np.trace(s4[list(p)].T @ s4 @ T_NOISE) / 4
        M, off = channel_R2(s4[list(p)], rot4, T_NOISE, t_NOISE, s_actual=s4)
        d_M = np.abs(M - kap * np.eye(3)).max()
        assert d_M < 1e-10, f"noisy SIC derangement {p} off tr(BT)/4 ({d_M:.2e})"
        assert np.allclose(off, 0, atol=1e-9), p
        noisy.append(round(float(kap), 12))
    assert len(set(noisy)) == 9                  # the degeneracy lifts whole
    # the kill is noiseless: a single-fixed-point bijection has kappa = 0
    p0 = (0, 2, 3, 1)
    kap0 = np.trace(s4[list(p0)].T @ s4 @ T_NOISE) / 4
    M, off = channel_R2(s4[list(p0)], rot4, np.eye(3), np.zeros(3),
                        s_actual=s4)
    assert np.abs(M).max() < 1e-10               # the ZERO map, at T = Id
    M, off = channel_R2(s4[list(p0)], rot4, T_NOISE, t_NOISE, s_actual=s4)
    assert abs(kap0) > 0.05                      # revived: taxed, not killed
    assert np.abs(M - kap0 * np.eye(3)).max() < 1e-10, p0
    assert np.allclose(off, 0, atol=1e-9), p0
    scramble = {}
    for s, law, n_zero in ((s4, 1 / 9, 8), (s6, 1 / 15, 200)):
        V = len(s)
        kk = np.array([np.mean([s[p[k]] @ s[k] for k in range(V)])
                       for p in itertools.permutations(range(V))])
        d_mean, d_msq = abs(kk.mean()), abs((kk ** 2).mean() - law)
        assert d_mean < 1e-12, (V, d_mean)
        assert d_msq < 1e-12, (V, d_msq)
        # the killed bijections are a COUNT, so this asserts on the margin the
        # way induced() does, not on a tolerance: exact zeros on one side, the
        # nearest live |kappa| (1/3 at V = 4, 1/6 at V = 6) on the other
        zero = np.abs(kk) < 1e-12
        assert int(zero.sum()) == n_zero, (V, int(zero.sum()))
        assert np.abs(kk[~zero]).min() > 0.1, V
        scramble[V] = (100 * zero.mean(), np.median(1 / kk[~zero] ** 2))
    print()
    print("[ok] anchors, closed form, both noise settings: believe the antipodes")
    print("     and the channel is -Id exactly (kappa = -1; -tr(T)/3 Id at the")
    print("     probe); on the SIC EVERY derangement prices at -1/3 exactly, all 9")
    print("     of them, the Gram being constant off-diagonal -- a degeneracy the")
    print("     probe noise lifts to nine DISTINCT prices, asserted. And")
    print("     over ALL V! label bijections, E[kappa] = 0 and E[kappa^2] =")
    print("     1/(3(V-1)) -- exhaustive at V = 4 (1/9) and V = 6 (1/15), so a")
    print("     scramble is worse on a bigger POVM. But that second moment is not")
    print("     a premium and must not be read as one: kappa is exactly ZERO for")
    print(f"     {scramble[4][0]:.1f}% of the tetrahedron's bijections and"
          f" {scramble[6][0]:.1f}% of the")
    print("     octahedron's -- at T = Id the estimator is killed outright, not")
    print("     taxed, and E[1/kappa^2] infinite. A noise map revives the kill")
    print("     into a steep tax -- tr(B T) survives where tr(B) = 0, one such")
    print(f"     bijection asserted at a {1 / kap0 ** 2:.0f}x premium at the probe. Among the")
    print(f"     survivors the median premium is {scramble[4][1]:.0f}x at V = 4 --"
          f" inverting 1/(3(V-1))")
    print(f"     happens to say 9 there too -- but {scramble[6][1]:.0f}x at V = 6,"
          f" 2.4x above the 15 it")
    print("     predicts. A scrambled label is a different failure from a")
    print("     mispriced one.")


def check_tail_weight():
    """The tail weight -- the single-Pauli estimate's fourth moment -- as an
    exact functional of the POSE, with every printed pose number pinned.

    The single-shot estimate is 3 x (sampled snapshot coordinate), so its
    fourth moment is 81 <w_a^4> over the swept snapshot directions; the
    frames paragraph (Section 5.2.1) and Appendix F.1 print the atlas-pose
    values, D.2's unreoriented paragraph moves them. Everything here is the
    exact protocol average asserted against a closed form. Three collapses
    make "the tail weight" one well-defined number per posed solid:
    STATE-FREE (the linear-in-state term averages to zero under every draw
    here -- asserted at a generic state, not just derived), AXIS-FREE (the
    coordinate 3-cycle in T equalizes the three Paulis), and DRAW-BLIND,
    by two different mechanisms: T and O preserve q = sum_a w_a^4
    pointwise (signed coordinate permutations), I does not -- yet a
    rotated 5-design is a 5-design, so the icosahedral solids read the
    sphere's own <q> = 3/5 in every pose. What remains is 27 <q> over the
    posed vertex set alone. Two SOS identities put q in [1/3, 1], floor
    exactly the eight cube directions, ceiling exactly the six Pauli axes,
    so every solid in every pose under every twirl lands in [9, 27]: the
    atlas pose is the tetrahedron's and cube's GLOBAL optimum and the
    octahedron's global PESSIMUM. Pinned below: D.2's 9 -> 15 and
    27 -> 12 (the 5-designs immovable at 81/5), the octahedron's own
    pose-minimum 11 (exact value, proved minimality), and Appendix
    F.3.3's T-draw orbit tails 9 and 21.
    """
    x, y, z = sp.symbols("x y z")
    q_sym = x**4 + y**4 + z**4
    S_sym = x**2 + y**2 + z**2
    # the range is two SOS identities, not a search: on the sphere S = 1,
    #   q - 1/3 = (x^2 - 1/3)^2 + (y^2 - 1/3)^2 + (z^2 - 1/3)^2 >= 0,
    #   1 - q   = 2 (x^2 y^2 + y^2 z^2 + z^2 x^2)               >= 0,
    # with equality iff every square vanishes: all w_a^2 = 1/3 (the eight
    # cube directions) at the floor, at most one nonzero coordinate (the
    # six Pauli axes) at the ceiling. Asserted as polynomial identities off
    # the sphere too, the slack carrying its (S - 1) multiplier:
    assert sp.expand(q_sym - Rational(1, 3)
                     - sum((w**2 - Rational(1, 3))**2 for w in (x, y, z))
                     - Rational(2, 3) * (S_sym - 1)) == 0
    assert sp.expand(1 - q_sym - 2 * (x**2 * y**2 + y**2 * z**2 + z**2 * x**2)
                     + (S_sym - 1) * (S_sym + 1)) == 0

    def q(w):
        return (np.asarray(w, float) ** 4).sum(axis=-1)

    def tail(verts):
        return 27 * float(q(verts).mean())

    def moment4(sl, rots, r):
        # E[o_a^4], a = x, y, z: readout probability times the fourth
        # power of the snapshot coordinate, averaged exactly over
        # draw x outcome. This is the R2 estimator with the belief moving
        # with the gate (snapshots and probabilities off the same posed
        # list) -- the sharper flavor, since R1's b^4 = 1 kills its state
        # term before any averaging; R1's orbit sweeps are priced in the
        # dodecahedron block below.
        V = len(sl)
        out = np.zeros(3)
        for Rg in rots:
            p = (1 + (sl @ (Rg @ r))) / V
            out += p @ (81 * (sl @ Rg) ** 4)
        return out / len(rots)

    def _rz(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def _ry(t):
        c, s = math.cos(t), math.sin(t)
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    C_GEN = _rz(0.3) @ _ry(0.7) @ _rz(1.1)     # a fixed generic pose probe
    R_GEN = np.array([0.24, -0.4, 0.56])       # a fixed generic state probe
    ATLAS_TAILS = {"tetrahedron": 9.0, "octahedron": 27.0, "cube": 9.0,
                   "icosahedron": 81 / 5, "dodecahedron": 81 / 5}
    DECKER_TAILS = {"tetrahedron": 15.0, "octahedron": 12.0, "cube": 15.0,
                    "icosahedron": 81 / 5, "dodecahedron": 81 / 5}
    rows = {}
    for solid in SOLIDS:
        n = load_vertices(solid)
        R = np.array(REORIENT[solid][1], dtype=float)
        # Decker's vertex SET is R^T (atlas set): check_decker_outcome_order
        # has already certified, vertex for vertex, that Table D.3's R
        # carries his rebuilt circuit's solid onto the atlas, and a tail
        # weight is order-blind, so the reorientation transpose stands in
        # for the circuits here (they are not cheap at the dodecahedron).
        d = n @ R
        gen = n @ C_GEN
        draws = ("T",) if COVARIANCE[solid] == "T" else ("T", COVARIANCE[solid])
        for verts in (n, d, gen):
            for g in draws:                      # the minimal twirl included
                for r in (np.zeros(3), R_GEN):
                    m = moment4(verts, load_rotations(g), r)
                    # measured worst over all 100 calls: axis spread
                    # 3.6e-15, deviation from the functional 2.5e-14
                    assert m.max() - m.min() < 1e-10, (solid, g)
                    assert abs(m.mean() - tail(verts)) < 1e-10, (solid, g)
            assert 9 - 1e-9 <= tail(verts) <= 27 + 1e-9, solid
        rows[solid] = (tail(n), tail(d), tail(gen))
        assert abs(tail(n) - ATLAS_TAILS[solid]) < 1e-10, solid
        assert abs(tail(d) - DECKER_TAILS[solid]) < 1e-10, solid
    # the generic pose lands the three cubic-axis solids strictly interior
    # (measured 16.115 / 16.327 / 16.115): the extremes are POSE facts
    for solid in ("tetrahedron", "octahedron", "cube"):
        assert 9.5 < rows[solid][2] < 26.5, solid
    # the tetrahedron prices as the cube in EVERY common pose: q is even
    # and the cube's eight vertices are the tetrahedron's four with their
    # antipodes, so the two vertex averages coincide identically
    assert all(abs(a - b) < 1e-12
               for a, b in zip(rows["tetrahedron"], rows["cube"]))
    # equality cases, attained vertex by vertex (measured 2.2e-16): the
    # atlas tetrahedron and cube sit at the SOS floor (every coordinate
    # +-1/sqrt3 -- the global optimum, and the only attaining points), the
    # atlas octahedron at the ceiling (the Pauli axes -- the pessimum)
    for solid, val in (("tetrahedron", 1 / 3), ("cube", 1 / 3),
                       ("octahedron", 1.0)):
        assert np.abs(q(load_vertices(solid)) - val).max() < 1e-12, solid
    # the icosahedral closed forms behind the immovable 81/5. Atlas
    # icosahedron: EVERY vertex has q = 3/5 exactly, (1 + tau^4)/(1 +
    # tau^2)^2 = 3/5 -- so even a sub-orbit sweep of it reads 81/5. Atlas
    # dodecahedron: q takes exactly two values, 1/3 on its cube eight and
    # 7/9 on its twelve (0, +-sigma, +-tau)/sqrt3 vertices -- the pretty
    # identity is sigma^4 + tau^4 = 7 -- and 27 x the vertex-weighted mean
    # is (8 x 9 + 12 x 21)/20 = 81/5. In Decker's pose neither list is
    # q-constant any more (spread asserted), yet both means hold at 3/5:
    # a rotated 5-design is a 5-design, the second collapse mechanism.
    assert sp.simplify((1 + TAU_SYM**4) / (1 + TAU_SYM**2)**2
                       - Rational(3, 5)) == 0
    assert sp.simplify(SIG_SYM**4 + TAU_SYM**4 - 7) == 0
    assert np.abs(q(load_vertices("icosahedron")) - 3 / 5).max() < 1e-12
    qd = np.sort(q(load_vertices("dodecahedron")))
    assert np.abs(qd[:8] - 1 / 3).max() < 1e-12
    assert np.abs(qd[8:] - 7 / 9).max() < 1e-12
    for solid in ("icosahedron", "dodecahedron"):
        dd = load_vertices(solid) @ np.array(REORIENT[solid][1], dtype=float)
        assert q(dd).max() - q(dd).min() > 0.05, solid
        assert abs(q(dd).mean() - 3 / 5) < 1e-12, solid
    # the first collapse mechanism, isolated: every T and O rotation
    # preserves q pointwise (measured 1.1e-15 over all 24 on a posed set),
    # where a generic I rotation moves a generic direction's q by ~0.3 --
    # the icosahedral draws really do need the design argument
    d_oct = load_vertices("octahedron") @ np.array(
        REORIENT["octahedron"][1], dtype=float)
    dev_O = max(np.abs(q(d_oct @ g) - q(d_oct)).max()
                for g in load_rotations("O"))
    assert dev_O < 1e-12, dev_O
    u = np.array([0.36, 0.48, 0.80])             # a generic unit direction
    dev_I = max(abs(q(g.T @ u) - q(u)) for g in load_rotations("I"))
    assert dev_I > 0.05, dev_I

    # the octahedron's own pose landscape: vertices +- the columns of C, so
    # the functional is f(C) = 9 sum_ac C_ac^4. The pose-minimum is 11, at
    # the frame whose columns are three of the signed permutations of
    # (1/3, 2/3, 2/3) -- EXACT there, per column 1/81 + 16/81 + 16/81 =
    # 11/27. Those 24 permutations ARE the O-orbit of the direction, but
    # they sit on 12 axes and no octahedron: the orbit names the pool the
    # three columns are drawn from, never the pose. Minimality is PROVED
    # below; the deterministic Euler-grid descent that follows (coarse
    # 40 x 20 x 40, three shrink rounds, zero RNG) lands at 11 + 1.0e-7 and
    # is kept as corroboration, and for its maximum: the coarse grid
    # contains the identity, so it recovers the atlas pose's 27 -- the
    # global pessimum as a pose-landscape fact. Decker's pose sits at
    # exactly 12: nearly attaining what no pose of the octahedron beats.
    C_STAR = np.array([[-1, 2, 2], [-2, 1, -2], [-2, -2, 1]]) / 3.0
    assert np.abs(C_STAR @ C_STAR.T - np.eye(3)).max() < 1e-15
    assert abs(np.linalg.det(C_STAR) - 1) < 1e-12
    M_STAR = Matrix([[-1, 2, 2], [-2, 1, -2], [-2, -2, 1]]) / 3
    assert 9 * sum(e**4 for e in M_STAR) == 11   # exact, in rationals
    assert abs(9 * float((C_STAR ** 4).sum()) - 11) < 1e-12

    # MINIMALITY, proved -- the grid below only corroborates it. In
    # axis-angle R = cI + s K_n + (1 - c) n n^T, P = sum_ij R_ij^4 is even
    # in s (so s^2 -> 1 - c^2 is lossless) and symmetric in u_i = n_i^2, so
    # P = P(c, e_2, e_3) at e_1 = 1, where dP/de_3 = 12 (1 - c)^3 (4c + 3)
    # is free of e_3: P is AFFINE in it, so at fixed (c, e_2) the minimum
    # sits at an endpoint of the feasible e_3-interval -- and no case split
    # is owed, an affine function bottoming out at an endpoint even where
    # its slope vanishes (c = 1, c = -3/4). Lagrange on {u >= 0, e_1 = 1,
    # e_2 fixed} gives u_j u_k = lam + mu (1 - u_i); times u_i, every u_i
    # is a root of mu t^2 - (lam + mu) t + e_3, so an endpoint has two u_i
    # equal or (the constraint binding) one u_i = 0. Both families come out
    # at 11/9 below: tail weight 9 x 11/9 = 11, attained at C_STAR above.
    cc, ss = sp.symbols("c s", real=True)
    nn = Matrix(sp.symbols("n1 n2 n3", real=True))
    Kn = Matrix([[0, -nn[2], nn[1]], [nn[2], 0, -nn[0]], [-nn[1], nn[0], 0]])
    Rax = cc * sp.eye(3) + ss * Kn + (1 - cc) * (nn * nn.T)
    Pax = sp.Poly(sp.expand(sum(e ** 4 for e in Rax)), ss).as_expr().subs(
        {ss ** 4: (1 - cc ** 2) ** 2, ss ** 2: 1 - cc ** 2})
    uu = sp.symbols("u1 u2 u3", nonnegative=True)
    for w, v in zip(nn, uu):
        Pax = sp.expand(Pax.subs(w ** 6, v ** 3).subs(w ** 4, v ** 2)
                        .subs(w ** 2, v))
    sym, rem, _ = sp.symmetrize(Pax, uu, formal=True)
    Pe = sp.expand(sym.subs(sp.Symbol("s1"), 1))     # n is a unit vector
    assert rem == 0 and sp.degree(Pe, sp.Symbol("s3")) == 1
    assert sp.expand(sp.diff(Pe, sp.Symbol("s3"))
                     - 12 * (1 - cc) ** 3 * (4 * cc + 3)) == 0
    aa = sp.symbols("a", real=True)

    def _rect_min(E, hi):
        """Exact min of the polynomial E(c, a) over [-1, 1] x [0, hi]: the
        four corners, the interior critical points, and each edge's own."""
        box = {cc: (-1, 1), aa: (0, hi)}
        vals = [E.subs({cc: p, aa: q}) for p in box[cc] for q in box[aa]]
        for so in sp.solve([E.diff(cc), E.diff(aa)], [cc, aa], dict=True):
            if (len(so) == 2 and all(v.is_real for v in so.values())
                    and all(box[k][0] <= v <= box[k][1]
                            for k, v in so.items())):
                vals.append(E.subs(so))
        for var, oth in ((cc, aa), (aa, cc)):
            for v0 in box[oth]:
                f = E.subs(oth, v0)
                vals += [f.subs(var, r) for r in sp.solve(f.diff(var), var)
                         if r.is_real and box[var][0] <= r <= box[var][1]]
        return min(sp.nsimplify(v) for v in vals)

    for uvec, hi in (((aa, aa, 1 - 2 * aa), Rational(1, 2)),  # two u_i equal
                     ((aa, 1 - aa, 0), 1)):                   # one u_i = 0
        assert _rect_min(sp.expand(Pax.subs(dict(zip(uu, uvec)))),
                         hi) == Rational(11, 9), uvec

    def _stack(ts, kind):
        c, s, z = np.cos(ts), np.sin(ts), np.zeros_like(ts)
        rows_ = ([[c, -s, z], [s, c, z], [z, z, z + 1]] if kind == "z"
                 else [[c, z, s], [z, z + 1, z], [-s, z, c]])
        return np.stack([np.stack(r, -1) for r in rows_], -2)

    al = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    be = np.linspace(0, np.pi, 20)
    ga = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    f_max = None
    for _ in range(4):
        C = np.einsum("aij,bjk,ckl->abcil",
                      _stack(al, "z"), _stack(be, "y"), _stack(ga, "z"))
        f = 9 * (C ** 4).sum((-2, -1))
        if f_max is None:
            f_max = float(f.max())
        i, j, k = np.unravel_index(np.argmin(f), f.shape)
        da, db, dg = al[1] - al[0], be[1] - be[0], ga[1] - ga[0]
        al = np.linspace(al[i] - da, al[i] + da, 21)
        be = np.linspace(be[j] - db, be[j] + db, 21)
        ga = np.linspace(ga[k] - dg, ga[k] + dg, 21)
    f_min = float(f.min())
    assert 11 - 1e-6 < f_min < 11 + 1e-5         # measured 11 + 1.0e-7
    assert abs(f_max - 27) < 1e-9                # Euler (0, 0, 0) is in-grid
    f_dec = 9 * float((np.array(REORIENT["octahedron"][1], dtype=float)
                       ** 4).sum())
    assert abs(f_dec - 12) < 1e-12               # measured 2.1e-15
    assert abs(f_dec - rows["octahedron"][1]) < 1e-12
    assert f_min < f_dec < rows["octahedron"][0]

    # Appendix F.3.3's footnote prices the T-draw's two dodecahedral orbit
    # POVMs at E[o^4] = 9 and 21 against the full solid's 81/5 -- the same
    # functional, read off the two q-values above. The alignment vertex
    # sits on the twelve-vertex family: its T-orbit is that family whole
    # (12 of the 20 vertices, 6 of the 10 axes, antipodally closed), q =
    # 7/9 throughout, tail 27 x 7/9 = 21. Any remaining vertex is on the
    # inscribed cube: T-orbit of size 4, antipodes completing the
    # eight-vertex POVM, q = 1/3, tail 9. R1's swept set IS the orbit, so
    # these are the R1 flavor of the identity, trivially state-free.
    s20, rT = load_vertices("dodecahedron"), load_rotations("T")
    _, v0 = alignment(s20)
    sw = np.array([g.T @ v0 for g in rT])
    m = 81 * (sw ** 4).mean(axis=0)
    assert m.max() - m.min() < 1e-10 and abs(m.mean() - 21) < 1e-10
    keys20 = {rot_key(v) for v in s20}
    orb = {rot_key(w) for w in sw}
    assert len(orb) == 12 and orb <= keys20      # 12 of 20, hit once each
    assert {rot_key(-w) for w in sw} == orb      # antipodally closed
    v1 = next(v for v in s20 if rot_key(v) not in orb)
    sw1 = np.array([g.T @ v1 for g in rT])
    m1 = 81 * (sw1 ** 4).mean(axis=0)
    assert m1.max() - m1.min() < 1e-10 and abs(m1.mean() - 9) < 1e-10
    orb1 = {rot_key(w) for w in sw1}
    povm1 = orb1 | {rot_key(-w) for w in sw1}
    assert len(orb1) == 4 and len(povm1) == 8    # the inscribed cube
    assert povm1 <= keys20 and len(orb | povm1) == 20
    assert (8 * 9 + 12 * 21) / 20 == 81 / 5      # the families' mean is F.1's

    print(f"  {'solid':14s} {'atlas':>8s} {'decker':>8s} {'generic':>9s}")
    print("  " + "-" * 44)
    for solid in SOLIDS:
        a, dd, gg = rows[solid]
        print(f"  {solid:14s} {a:8.3f} {dd:8.3f} {gg:9.4f}")
    print()
    print("[ok] the tail weight -- the single-Pauli estimate's fourth moment --")
    print("     is 27<w_x^4 + w_y^4 + w_z^4> over the swept directions, asserted")
    print("     as the exact protocol average at every pose, draw (T included),")
    print("     Pauli and state: state-free, axis-free, draw-blind. Two SOS")
    print("     identities put it in [9, 27], floor exactly the eight cube")
    print("     directions, ceiling exactly the six Pauli axes -- the atlas pose")
    print("     is the tetrahedron's and cube's GLOBAL optimum and the")
    print("     octahedron's global pessimum, and D.2's unreoriented numbers are")
    print("     pinned: 9 -> 15 (tetrahedron and cube, equal in every common")
    print("     pose), 27 -> 12 (octahedron), the 5-designs immovable at 81/5.")
    print("[ok] the octahedron's own pose landscape: minimum 11 -- exact at the")
    print("     frame of signed (1/3, 2/3, 2/3) permutations, and PROVED")
    print("     minimal: sum_ij R_ij^4 is affine in e_3(n^2), so Lagrange")
    print("     forces two endpoint families and both return 11/9. The grid")
    print(f"     descent corroborates at {f_min:.7f}, so Decker's 12 nearly")
    print("     attains what no pose of it beats; the grid's maximum is the")
    print("     atlas pose's own 27.")
    print("[ok] and F.3.3's T-draw orbit POVMs price by the same functional: the")
    print("     dodecahedron's q is 1/3 on its cube eight and 7/9 on its twelve")
    print("     (sigma^4 + tau^4 = 7), so the two realized POVMs read 9 and 21,")
    print("     and their vertex-weighted mean is the full solid's 81/5.")


def check_swept_third_moment():
    """D.2's second-moment claim, one moment order below the tail weight.

    The canonical dual reads a posed vertex set through its moments of order
    at most three (the site-cases identity of shadow_experiments.py's
    exact_backbone), so once centering fixes the first and the 2-design the
    second, the only thing left for a pose to move is the third-moment
    tensor S = <w w w> over the SWEPT directions. Four of the five solids
    are antipodal, so their S is zero in every pose and no draw is needed
    for it: the tetrahedron is the one solid with a third moment to lose.
    Decker's sweep loses it. His four vertices go to 24 distinct directions
    with S = 0 exactly -- a spherical 3-design, where the atlas sweep is
    only a 2-design (4 directions, S_xyz = sqrt(3)/9) -- which is why the
    variance moves for the tetrahedron alone, and moves by exactly the term
    S carries.

    It is the SWEEP, not the pose: Decker's raw four still carry a third
    moment, merely pointing elsewhere. And it is a law rather than a
    coincidence -- for a pose R_z(theta) the swept S_xyz is
    (sqrt(3)/9) cos 2theta, so a codimension-one set of poses zeroes it,
    45 degrees among them, and "Decker's pose alone" would be false. The
    promotion stops at three: the swept Decker tetrahedron's tail weight is
    15, not a 4-design's 81/5 (check_tail_weight), which is exactly why the
    fourth moment notices a pose the second does not.
    """
    def third(vs):
        return np.einsum("ka,kb,kc->abc", vs, vs, vs) / len(vs)

    def rz(th):
        c, s = math.cos(th), math.sin(th)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    rows = {}
    for solid in SOLIDS:
        n = load_vertices(solid)
        R = np.array(REORIENT[solid][1], dtype=float)
        rots = load_rotations(COVARIANCE[solid])
        for pose, base in (("atlas", n), ("decker", n @ R)):
            sw = np.vstack([base @ g for g in rots])
            S = third(sw)
            rows[solid, pose] = float(np.abs(S).max())
            # orders one and two first, so "third moments and nothing else"
            # is asserted, not assumed: centered, and a tight frame
            assert np.abs(sw.mean(axis=0)).max() < 1e-12, (solid, pose)
            assert np.abs(sw.T @ sw / len(sw)
                          - np.eye(3) / 3).max() < 1e-12, (solid, pose)
            if (solid, pose) == ("tetrahedron", "atlas"):
                assert abs(np.abs(S).max() - math.sqrt(3) / 9) < 1e-12
                assert abs(S[0, 1, 2] - math.sqrt(3) / 9) < 1e-12
            else:
                assert np.abs(S).max() < 1e-12, (solid, pose)
    # the sweep, not the pose (measured 0.192450 either way), and as a SET
    # and not merely a multiset: the 48 swept directions collapse to 24
    # distinct points hit twice each, where the atlas sweep collapses to the
    # tetrahedron's own 4 hit twelve times each
    n4 = load_vertices("tetrahedron")
    d4 = n4 @ np.array(REORIENT["tetrahedron"][1], dtype=float)
    assert np.abs(third(d4)).max() > 0.19
    rT = load_rotations("T")
    for base, want in ((n4, 4), (d4, 24)):
        sw = np.vstack([base @ g for g in rT])
        keys = [rot_key(w) for w in sw]
        assert len(set(keys)) == want
        assert all(keys.count(k) == len(sw) // want for k in set(keys))
    # the pose law behind the zero: T carries exactly one invariant in the
    # degree-3 harmonics, living in the m = +-2 plane, and R_z(45 deg) turns
    # that plane into the complement the T-average annihilates
    for th in (0.0, math.pi / 8, math.pi / 4, 0.9, 3 * math.pi / 4):
        sw = np.vstack([(n4 @ rz(th)) @ g for g in rT])
        assert abs((sw[:, 0] * sw[:, 1] * sw[:, 2]).mean()
                   - math.sqrt(3) / 9 * math.cos(2 * th)) < 1e-12, th

    print(f"  {'solid':14s} {'|S| atlas':>11s} {'|S| decker':>11s}")
    print("  " + "-" * 39)
    for solid in SOLIDS:
        print(f"  {solid:14s} {rows[solid, 'atlas']:11.2e}"
              f" {rows[solid, 'decker']:11.2e}")
    print()
    print("[ok] the swept third-moment tensor S = <w w w> vanishes for every")
    print("     solid in both poses but one -- the ATLAS tetrahedron, where")
    print("     S_xyz = sqrt(3)/9 -- the other four being antipodal. So the")
    print("     tetrahedron is the only solid whose variance a pose can move,")
    print("     and Decker's sweep is what moves it: his 4 vertices sweep to")
    print("     24 distinct directions with S = 0, a spherical 3-design where")
    print("     the atlas sweep is only a 2-design on 4 directions.")
    print("[ok] the sweep, not the pose (his raw four still carry a third")
    print("     moment), and a law, not a coincidence: the swept S_xyz of a")
    print("     pose R_z(theta) is (sqrt(3)/9) cos 2theta, so a codimension-one")
    print("     set of poses zeroes it. The promotion stops at three -- the")
    print("     tail weight is 15, not a 4-design's 81/5 -- which is why the")
    print("     fourth moment notices a pose the second does not.")
