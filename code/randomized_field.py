"""The randomized-implementation suite's number-field kit: the per-solid
fields and coercion, the exact channels, and the reposed-twirl theorem.
Tools only -- the checks that use them live with their sections
(randomized_scalars, randomized_twojobs, randomized_obstruction).

randomized_implementations.py is the suite's entry point -- its docstring
carries the R1/R2 protocol definitions, the kappa/eta pin, the findings
index, the module map and the run instructions -- and each finding's full
account lives in the docstring of the module that proves it;
`cd code && uv run randomized_implementations.py` runs the whole suite.
"""

import math

import numpy as np
import sympy as sp
from sympy import Rational, sqrt

from randomized_core import (TAU_SYM, COVARIANCE, T_NOISE, t_NOISE,
                             atlas_vertices, exact_rotations)

# ---------------------------------------------------------------------------
# The same two bars, exactly: every float decision above lifted to canonical
# form.  A companion, not a replacement -- the float pipeline stays in place
# and the two are asserted to agree, so nothing downstream (which keys on the
# float rot_key) moves and no emitted fragment changes.
#
# two_bars() rests on THREE tolerance-bearing mechanisms, not one: the subgroup
# lattice (rot_key's 9-decimal rounding grid), the realize test (argmin over
# vertex distances with a 1e-9 acceptance), and the twirl test (allclose
# against a single fixed decimal probe T_NOISE).  Lifting only the first would
# leave the headline counts -- twirl bar flat at 12, realize bar 3/4/12/60,
# crossing at the icosahedron -- still decided by tolerances.  All three move
# here.
#
# The twirl side is not merely made exact, it is made GENERIC: quantified over
# every measurement-side noise instead of evaluated at T_NOISE.  That is the
# reframing rank_one_twirl() already uses in check_coin_group -- free symbols,
# arbitrary v and w -- carried from the three full groups to every subgroup of
# the lattice.  It is what lets the sweep's zero/nonzero dichotomy be stated
# without reference to a probe.
# ---------------------------------------------------------------------------

# Rotation entries of T, O and I lie in Q(sqrt2, sqrt5) -- main.py's own field,
# the one _to_basis canonicalises.  Vertices need more, and each solid gets a
# separate small field rather than one global degree-16 field (150x faster,
# every field degree <= 4).  Each must ALSO contain the covariance group's
# entries, so the orbit b Rg^T v0 cannot leave it; that is checked by
# coercion, which raises rather than approximating.
FIELD_GENS = {"tetrahedron": (sqrt(3),), "octahedron": (), "cube": (sqrt(3),),
              "icosahedron": (sqrt(2 + TAU_SYM),),
              "dodecahedron": (sqrt(3), sqrt(5))}

FIELD_NAME = {"tetrahedron": "Q(sqrt3)", "octahedron": "Q", "cube": "Q(sqrt3)",
              "icosahedron": "Q(sqrt(2+tau))", "dodecahedron": "Q(sqrt3, sqrt5)"}

# Radial denominators -- the same table as povm_properties's _RADIAL: every
# solid's vertices are an integer-ish tuple over Q(sqrt5) over a single radius.
# Only exact_reposed_twirl_R2 uses them, and it is quadratic in the vertices,
# so clearing the radius costs it nothing and drops all five solids out of
# their own degree-4 fields into one degree-2 one.  Measured over the five:
# 60 s -> 22 s.
_RADIAL = {"tetrahedron": sqrt(3), "octahedron": Rational(1), "cube": sqrt(3),
           "icosahedron": sqrt(2 + TAU_SYM), "dodecahedron": sqrt(3)}

_POSE_FIELD = sp.QQ.algebraic_field(sqrt(5))


def solid_field(solid):
    """The solid's coefficient field as a SymPy domain (QQ when trivial)."""
    gens = FIELD_GENS[solid]
    return sp.QQ.algebraic_field(*gens) if gens else sp.QQ


def to_field(M, K):
    """Coerce a symbolic 3x3 into K as nested lists -- fail-loud, no simplify.

    from_sympy raises CoercionFailed outside K, so a mis-declared field or a
    re-posed solid cannot slip through as a wrong answer.  Deliberately no
    nsimplify/radsimp in front: every rotation entry and every vertex coerces
    raw, so nothing here rests on a normalising heuristic.
    """
    return [[K.from_sympy(M[i, j]) for j in range(3)] for i in range(3)]


def _mm(A, B, K):
    return [[sum((A[i][k] * B[k][j] for k in range(3)), K.zero) for j in range(3)]
            for i in range(3)]


def _mv(A, v, K):
    return [sum((A[i][k] * v[k] for k in range(3)), K.zero) for i in range(3)]


def _tr(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def _eye(K):
    return [[K.one if i == j else K.zero for j in range(3)] for i in range(3)]


def exact_subgroup_lattice(g):
    """subgroup_lattice(), with the rounding grid replaced by field equality.

    The lift is confined to one object.  Everything after the multiplication
    table -- pair closure, the completeness induction, the frozensets -- is
    integer index arithmetic, so making `mul` exact makes the whole lattice
    exact.  Field elements are canonical AND hashable, which is what a set
    dedup needs and what simplify(a-b)==0 cannot supply.
    """
    K = sp.QQ.algebraic_field(sqrt(2), sqrt(5))
    Rs = [to_field(R, K) for R in exact_rotations(g)]
    n = len(Rs)

    def key(A):
        return tuple(A[i][j] for i in range(3) for j in range(3))

    idx = {key(A): i for i, A in enumerate(Rs)}
    assert len(idx) == n, f"{g}: rotations not distinct in the field"
    mul = np.array([[idx[key(_mm(A, B, K))] for B in Rs] for A in Rs])

    def close(gens):
        S = set(gens)
        while True:
            new = set(mul[np.ix_(sorted(S), sorted(S))].ravel().tolist()) - S
            if not new:
                return frozenset(S)
            S |= new

    subs = {close((i, j)) for i in range(n) for j in range(i, n)}
    assert all(close((*S, h)) in subs
               for S in subs for h in range(n) if h not in S), "lattice incomplete"
    return mul, subs


_EXACT_LATTICE = {}


def exact_lattice(g):
    """exact_subgroup_lattice(g), memoized (four solids, three groups)."""
    if g not in _EXACT_LATTICE:
        _EXACT_LATTICE[g] = exact_subgroup_lattice(g)
    return _EXACT_LATTICE[g]


def exact_vertices(solid, K):
    """symbolic_solids()[solid] over K, permuted into povm_{solid}.npz order.

    Order is load-bearing, not cosmetic: alignment() takes v0 = argmax(s[:,2]),
    i.e. the FIRST vertex at the top latitude, and the top latitude is a tie on
    three of the four solids (cube 4, icosahedron 2, dodecahedron 2; only the
    octahedron has a unique v0).  A set-level correspondence would silently
    pick a different seed.  The permutation used is the one
    check_canonical_data() already asserts unique, not a new one.

    The npz is float64, so this is the one place the exact path must meet a
    float with no exact object on the other side to meet.  It cannot be removed
    -- only bounded, and the bound is enormous: the symbolic vertices
    reproduce the npz BIT-IDENTICALLY (max error 0.0) while the closest two
    distinct vertices of any solid are 0.71 apart.  So the assertion is on the
    MARGIN rather than on the threshold -- match inside 1e-12, runner-up beyond
    0.1 -- which leaves twelve orders of slack on each side and, unlike a bare
    threshold, cannot rot quietly.
    """
    return [[K.from_sympy(c) for c in v] for v in atlas_vertices(solid)]


def exact_alignment(s, K):
    """alignment(), radical-free and transcendental-free.

    The float version normalises the axis and goes through arccos/sin/cos.  It
    need not: with w = v0 x zhat left UNNORMALISED, |w| = sin(ang), so
    sin(ang) K_unit = K_w exactly, and (1 - cos(ang)) K_unit^2 = K_w^2/(1 + v_z)
    since |w|^2 = (1 - cos)(1 + cos).  So

        A = I + K_w + K_w^2 / (1 + v0_z),

    rational in v0's coordinates -- no square root enters and A stays in the
    solid's own field.  The octahedron's v0 = zhat gives K_w = 0, hence A = I,
    with no special case needed.
    """
    zs = [K.to_sympy(v[2]) for v in s]
    i0 = 0
    for i, z in enumerate(zs):                # np.argmax: FIRST strict maximum
        if sp.simplify(z - zs[i0]).is_positive:
            i0 = i
    v = s[i0]
    Kw = [[K.zero, K.zero, -v[0]], [K.zero, K.zero, -v[1]], [v[0], v[1], K.zero]]
    Kw2 = _mm(Kw, Kw, K)
    d = K.one + v[2]
    A = [[_eye(K)[i][j] + Kw[i][j] + Kw2[i][j] / d for j in range(3)]
         for i in range(3)]
    assert _mv(A, v, K) == [K.zero, K.zero, K.one], "A does not send v0 to zhat"
    return A, v


def exact_is_decomposable(s, K):
    """is_decomposable() over K -- antipodal pairing by field equality.

    This is the verdict that admits or bars R1 (finding 2): the tetrahedral SIC
    has no antipodal pairs, the other four solids are centrally symmetric.  It
    decides whether two vectors are negatives of each other, so it is an
    identity test, decided in canonical form rather than by a tolerance.
    """
    return all(any([-c for c in a] == list(u) for u in s) for a in s)


def exact_orbit_directions(R, v, K):
    """The DISTINCT directions R_g^T v, deduped by field equality.

    The float twin rounds to 9 decimals and calls np.unique; here two orbit
    points are the same point or they are not.  Used for the dodecahedron's
    inscribed-cube split, where the count itself is the claim.
    """
    out = []
    for Rg in R:
        w = _mv(_tr(Rg), v, K)
        if w not in out:
            out.append(w)
    return out


def exact_orbit_counts(s, R, K):
    """orbit_counts(), with argmin-plus-tolerance replaced by field equality."""
    _, v = exact_alignment(s, K)
    hits = [0] * len(s)
    for Rg in R:
        Rgt = _tr(Rg)
        for b in (K.one, -K.one):
            w = [b * c for c in _mv(Rgt, v, K)]
            hit = [k for k, u in enumerate(s) if list(u) == w]
            assert len(hit) == 1, "snapshot direction is not a vertex"
            hits[hit[0]] += 1
    return hits


def exact_channel_R1(s, R, T, t, K):
    """channel_R1() over K -- a literal transcription, same loop, same terms."""
    A, v = exact_alignment(s, K)
    M = [[K.zero] * 3 for _ in range(3)]
    off = [K.zero] * 3
    half = K.one / 2
    for Rg in R:
        TP = _mm(T, _mm(A, Rg, K), K)
        snap0 = _mv(_tr(Rg), v, K)
        for b in (K.one, -K.one):
            lin = [b * TP[2][j] * half for j in range(3)]
            const = (K.one + b * t[2]) * half
            snap = [3 * b * c for c in snap0]
            for i in range(3):
                off[i] += snap[i] * const
                for j in range(3):
                    M[i][j] += snap[i] * lin[j]
    n = K.convert(len(R))
    return [[M[i][j] / n for j in range(3)] for i in range(3)], [o / n for o in off]


def _probe(K, entry=None, t_at=None):
    """A basis probe: T with a single 1 at `entry`, t with a single 1 at `t_at`."""
    T = [[K.one if entry == (i, j) else K.zero for j in range(3)] for i in range(3)]
    return T, [K.one if t_at == a else K.zero for a in range(3)]


def exact_twirls(s, R, K):
    """Does R twirl for EVERY measurement-side noise (T, t) -- not just T_NOISE?

    M and off are LINEAR in (T, t): T enters exactly once, through
    (T @ pre)[2, :], and t exactly once, through (1 + b t_z)/2.  So the verdict
    on a spanning set of probes is the verdict on the whole probe space, and
    only row 2 of T and the single component t_z can enter at all.  Four probes
    span what survives: T = E_{2,k} for k = 0,1,2, and t = zhat.

    The condition itself is that M = T_zz I identically, so on the probes:
    E_{2,2} must give I and E_{2,0}, E_{2,1} must give 0.  The offset must
    vanish for every t, which on t = zhat reads sum_g Rg^T v0 = 0 (at t = 0 it
    vanishes for free, the two outcomes cancelling).

    exact_probe_span_ok() checks the span claim itself on the full group rather
    than leaving it resting on a reading of the code above.
    """
    Z = [[K.zero] * 3 for _ in range(3)]
    for k in range(3):
        M, _ = exact_channel_R1(s, R, *_probe(K, entry=(2, k)), K)
        if M != (_eye(K) if k == 2 else Z):
            return False
    _, off = exact_channel_R1(s, R, *_probe(K, t_at=2), K)
    return off == [K.zero] * 3


def exact_channel_R2(s, R, T, t, K):
    """channel_R2() over K -- a literal transcription of its one-list path,
    same loops, same terms. The two-list path (s_actual) has NO companion
    here: exact_reposed_twirl_R2 is the DIAGONAL case, belief = device =
    sC, priced at tr(C^T C T)/3 = tr(T)/3 for every rotation -- a pose,
    not a mismatch -- so the mismatch law tr(B T)/V rests on the float
    layer alone, at four float sites: the coset scan's noisy leg
    (check_reorientation_obstruction) and the anchors' T_NOISE legs,
    Table D.4 pairing and, for how T enters the two-list path rather than
    for the law itself, the second moment's probe leg (all three in
    check_decker_outcome_order). An exact two-list companion over
    Q(sqrt5)[C1, C2, T, t] would subsume the reposed theorem; it is
    deliberately not built -- the float sites above already guard the law,
    and an exact-layer addition is never free here: the entry module's
    charter requires every exact companion to be paired with a
    value-for-value numeric agreement check, so it would be a decision in
    its own right, not an extension of this function.
    """
    V = K.convert(len(s))
    M = [[K.zero] * 3 for _ in range(3)]
    off = [K.zero] * 3
    for Rg in R:
        TR = _mm(T, Rg, K)
        Rgt = _tr(Rg)
        for n in s:
            lin = [sum((n[a] * TR[a][j] for a in range(3)), K.zero) / V
                   for j in range(3)]
            const = (K.one + sum((n[a] * t[a] for a in range(3)), K.zero)) / V
            snap = [3 * c for c in _mv(Rgt, n, K)]
            for i in range(3):
                off[i] += snap[i] * const
                for j in range(3):
                    M[i][j] += snap[i] * lin[j]
    m = K.convert(len(R))
    return [[M[i][j] / m for j in range(3)] for i in range(3)], [o / m for o in off]


def exact_twirls_R2(s, R, K):
    """Is R2's channel (tr T/3) Id and unbiased for EVERY (T, t) -- not just T_NOISE?

    M is linear in T and does not involve t at all; off is AFFINE in t and does
    not involve T.  So nine probes decide M -- the claim M = (tr T/3) Id reads
    E_{i,i} -> Id/3 on the diagonal and E_{i,j} -> 0 off it -- and four decide
    off: t = 0 for the constant term, which must vanish on its own, then
    t = e_k for the linear part.

    Two differences from R1, both structural rather than incidental.  The WHOLE
    of T enters, through n^T T R_g summed over every vertex, so nothing reduces
    to a single row the way R1's readout does.  And there is no alignment: R2
    measures the native POVM, which is exactly why it admits the tetrahedron,
    where R1 is undefined.
    """
    Z = [[K.zero] * 3 for _ in range(3)]
    third = [[K.one / 3 if i == j else K.zero for j in range(3)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            M, off = exact_channel_R2(s, R, *_probe(K, entry=(i, j)), K)
            if M != (third if i == j else Z) or off != [K.zero] * 3:
                return False
    return all(exact_channel_R2(s, R, *_probe(K, t_at=k), K)[1] == [K.zero] * 3
               for k in range(3))


_POSE_RING = {}


def _pose_ring():
    """Q(sqrt5)[B, T, t]: 21 free symbols -- a 3x3 re-pose and a generic noise.

    Memoized like exact_lattice(): one ring serves all five solids, and
    building it is the sweep's only fixed cost.
    """
    if not _POSE_RING:
        names = ([f"B{i}{j}" for i in range(3) for j in range(3)]
                 + [f"T{i}{j}" for i in range(3) for j in range(3)]
                 + [f"t{k}" for k in range(3)])
        P = _POSE_FIELD.poly_ring(*names)
        g = [P.convert(sp.Symbol(n)) for n in names]
        _POSE_RING.update(
            P=P,
            B=[[g[3 * i + j] for j in range(3)] for i in range(3)],
            T=[[g[9 + 3 * i + j] for j in range(3)] for i in range(3)],
            t=g[18:])
    return _POSE_RING["P"], _POSE_RING["B"], _POSE_RING["T"], _POSE_RING["t"]


def exact_reposed_twirl_R2(solid, draw=None):
    """Is R2's scalar (tr T/3) at EVERY pose of the measured POVM, or only at
    the one pose check_reorientation_obstruction happens to try?

    That check asserts the twirled-native channel is (tr T/3) Id_3 with zero
    offset once the measured POVM is carried to Decker's pose.  It is a
    corollary; this is the theorem it is a corollary of.  For an ARBITRARY 3x3
    matrix C -- orthogonality is never used, so there is no parametrisation of
    SO(3) to get right and no corner of it to miss --

        M(sC) = (tr(C^T C T) / 3) Id_3,        off(sC) = 0,

    identically in C and in the noise (T, t).  A pose is a rotation, C^T C is
    then Id_3, and the theorem collapses to exactly what that assertion probes.

    Why the measured vertices cannot reach the scalar: they enter only through
    sum_k n_k n_k^T, which is (V/3) Id_3 for every atlas solid, and
    C^T ((V/3) Id_3) C = (V/3) C^T C; the offset needs sum_k n_k = 0 as well.
    Both moments are pose-covariant -- which is the file's own reason in code,
    that R2's twirl needs the DRAWN group irreducible rather than covariant for
    the measured POVM, so re-posing the POVM cannot touch it.

    No new transcription: exact_channel_R2 is run VERBATIM, its coefficient
    ring K taken to be a polynomial ring instead of a number field.  The loop
    check_exact_scalars already guards value-for-value against channel_R2 is
    the loop that proves this, and it is generic in K because every mechanism
    it uses -- K.zero, K.one, K.convert, exact division -- is.

    `draw` overrides the drawn group, which is the only hypothesis there is:
    the theorem needs it irreducible, so a reducible draw must fail at every
    pose exactly as it fails at one.

    Returns (verdict, evaluate).  evaluate(C, T, t) is the symbolic channel at
    float arguments: the verdict is a BOOLEAN, and a boolean exact check does
    not test its own transcription, so the caller pairs it
    with a value-for-value comparison against channel_R2.
    """
    P, B, T, t = _pose_ring()
    K, lam = _POSE_FIELD, _RADIAL[solid]
    # Clearing the radius rather than the compositum.  The free
    # matrix B then plays the part of C/lam, so lam itself never has to be
    # representable in Q(sqrt5) -- lam^2 does, and it enters in exactly one
    # place, next to the second moment it belongs to.
    s = [[P.convert(K.from_sympy(sp.expand(lam * c)), K) for c in v]
         for v in atlas_vertices(solid)]
    R = [[[P.convert(K.from_sympy(M[i, j]), K) for j in range(3)] for i in range(3)]
         for M in (exact_rotations(COVARIANCE[solid]) if draw is None else draw)]
    sB = [[sum((B[a][i] * n[a] for a in range(3)), P.zero) for i in range(3)]
          for n in s]
    M, off = exact_channel_R2(sB, R, T, t, P)          # the SAME loop, over K[.]

    lam2 = P.convert(K.from_sympy(sp.expand(lam ** 2)), K)
    A = [[lam2 * sum((B[k][i] * B[k][j] for k in range(3)), P.zero)
          for j in range(3)] for i in range(3)]                    # C^T C
    want = sum((sum((A[i][k] * T[k][i] for k in range(3)), P.zero)
                for i in range(3)), P.zero) / P.convert(3)         # tr(C^T C T)/3
    ok = (all(M[i][j] == (want if i == j else P.zero)
              for i in range(3) for j in range(3))
          and all(o == P.zero for o in off))

    def evaluate(C, Tn, tn):
        """The (M, off) this identity predicts, at float (C, T, t)."""
        vals = ([c / float(lam) for c in np.asarray(C, dtype=float).ravel()]
                + list(np.asarray(Tn, dtype=float).ravel())
                + list(np.asarray(tn, dtype=float)))

        def at(p):
            return sum(float(K.to_sympy(c))
                       * math.prod(vals[k] ** e for k, e in enumerate(mono))
                       for mono, c in p.terms())

        return (np.array([[at(M[i][j]) for j in range(3)] for i in range(3)]),
                np.array([at(o) for o in off]))

    return ok, evaluate


def exact_probe_span_ok(s, R, K):
    """The four probes really do span: nothing outside them can contribute.

    Sweeps all nine T = E_{i,j}.  Rows 0 and 1 must give M = 0 identically --
    that is what makes the twirl verdict depend on T through T_zz alone.
    """
    Z = [[K.zero] * 3 for _ in range(3)]
    return all(exact_channel_R1(s, R, *_probe(K, entry=(i, j)), K)[0] == Z
               for i in (0, 1) for j in range(3))


def exact_two_bars(solid):
    """two_bars()'s realize and twirl sets, with all three mechanisms lifted."""
    g = COVARIANCE[solid]
    K = solid_field(solid)
    R = [to_field(M, K) for M in exact_rotations(g)]   # coercion = the closure check
    s = exact_vertices(solid, K)
    _, subs = exact_lattice(g)
    assert exact_probe_span_ok(s, R, K), solid
    realize, twirl, irr = set(), set(), set()
    for S in subs:
        RS = [R[i] for i in sorted(S)]
        h = exact_orbit_counts(s, RS, K)
        if min(h) == max(h):
            realize.add(S)
        if exact_twirls(s, RS, K):
            twirl.add(S)
        # Schur, exactly: <chi, chi> = mean tr(R)^2 = 1
        tot = sum((sum((RS[m][i][i] for i in range(3)), K.zero) ** 2
                   for m in range(len(RS))), K.zero)
        if tot == K.convert(len(RS)):
            irr.add(S)
    return {"realize": realize, "twirl": twirl, "irr": irr,
            "bar_realize": min(map(len, realize)),
            "bar_twirl": min(map(len, twirl))}


def _exact_probe_noise(K):
    """T_NOISE / t_NOISE as exact rationals in K (two decimals, so exactly so)."""
    return ([[K.from_sympy(Rational(str(float(T_NOISE[i, j])))) for j in range(3)]
             for i in range(3)],
            [K.from_sympy(Rational(str(float(x)))) for x in t_NOISE])


def _as_float(M, K):
    return np.array([[float(K.to_sympy(M[i][j])) for j in range(3)] for i in range(3)])
