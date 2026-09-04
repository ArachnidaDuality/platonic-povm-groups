"""Shared foundations of the randomized-implementation suite: the canonical
npz data layer, the float primitives and estimator channels, the symbolic
layer, and the atlas circuits.

randomized_implementations.py is the suite's entry point -- its docstring
carries the R1/R2 protocol definitions, the kappa/eta pin, the findings
index, the module map and the run instructions -- and each finding's full
account lives in the docstring of the module that proves it;
`cd code && uv run randomized_implementations.py` runs the whole suite.
"""

import itertools
import math
from pathlib import Path

import numpy as np
import sympy as sp
from sympy import I as sI
from sympy import Matrix, Rational, sqrt

# the canonical quaternions, for the exact two-bars companion (proj_hash dedup)
from main import geometric_group

# A trailing-slash STRING, never a Path: use sites concatenate it, e.g.
# np.load(DATA + f"povm_{solid}.npz") or open(DATA + name, "w"); a Path raises
# TypeError at each. Note shadow_experiments.py's DATA is a Path: not ours.
DATA = f"{Path(__file__).parent / 'data'}/"

SOLIDS = ("tetrahedron", "octahedron", "cube", "icosahedron", "dodecahedron")

TAU_SYM = (1 + sqrt(5)) / 2          # golden ratio (Conway & Smith tau)
SIG_SYM = (sqrt(5) - 1) / 2          # inverse golden ratio sigma

# covariance rotation group of each solid's POVM
COVARIANCE = {"tetrahedron": "T", "octahedron": "O", "cube": "O",
              "icosahedron": "I", "dodecahedron": "I"}

# A fixed, generic measurement-side affine noise r -> T r + t, shared verbatim
# with shadow_experiments.py's two-protocol study, whose own channels are
# checked value for value against ours, up to the canonical dual's factor 3.
# Genericity is asserted in main() -- the two candidate scalars differ, the
# offset is nonzero, T is anisotropic -- so no depolarizing check can pass
# vacuously.
T_NOISE = np.array([[0.83, 0.06, -0.11],
                    [-0.04, 0.71, 0.09],
                    [0.12, -0.07, 0.62]])
t_NOISE = np.array([0.05, -0.03, 0.17])

PAULI = np.array([[[0, 1], [1, 0]],
                  [[0, -1j], [1j, 0]],
                  [[1, 0], [0, -1]]], dtype=complex)


# ---------------------------------------------------------------------------
# Canonical data (the symbolic pipeline's npz artifacts)
# ---------------------------------------------------------------------------

def load_vertices(solid):
    """Bloch vertices (V, 3) of the solid's POVM from the canonical npz."""
    return np.load(DATA + f"povm_{solid}.npz")["vertices"]


def load_elements(solid):
    """POVM elements (V, 2, 2) from the canonical npz."""
    return np.load(DATA + f"povm_{solid}.npz")["elements"]


def load_rotations(g):
    """SO(3) rotations (|G|, 3, 3) of the polyhedral group g in {T, O, I}."""
    return np.load(DATA + f"group_{g}.npz")["rotations"]


def load_atlas(g):
    """The binary group 2g's atlas: unitaries + synthesized circuits."""
    d = np.load(DATA + f"group_2{g}.npz")
    return {k: d[k] for k in d.files}


def rotation_from_unitary(U):
    """SO(3) rotation of a 2x2 unitary: R_ij = tr(sigma_i U sigma_j U+)/2."""
    return np.real(np.einsum("iab,bc,jcd,ad->ij", PAULI, U, PAULI, U.conj())) / 2


def rot_key(R, decimals=9):
    """Hashable rounded key of a rotation matrix: the float-grid identity trick.

    What binds is the distance from an exact entry to a rounding BOUNDARY, not
    the grid spacing.  T's and O's entries are {0, +-1} and land on the grid, so
    they get the full half-spacing, 5e-10.  I is the only tight group:
    tau/2 = 0.809016994|3749... and sigma/2 = 0.309016994|3749... sit 1.25e-10
    from a boundary -- four of I's nine entries, tied because the two differ by
    exactly 1/2, itself a grid multiple.  Measured error over every call site
    (raw npz, rotation_from_unitary, products R_i R_j, conjugates R_h R_i R_h^T)
    is at most 4.4e-16, so the margin is 2.8e5x and I is what binds it.

    exact_rotations is the companion that needs no grid at all; it is a second
    opinion on the float sweep, never a replacement for it.
    """
    return tuple(np.round(np.asarray(R), decimals).flatten().tolist())


# ---------------------------------------------------------------------------
# The two protocols as exact estimator channels
# ---------------------------------------------------------------------------

def is_decomposable(s):
    """Antipodal decomposability: every vertex has its antipode in the set."""
    return all(any(np.allclose(a, -b, atol=1e-9) for b in s) for a in s)


def design_strength(s, t_max=8):
    """Spherical t-design strength of a unit-vector set.

    t is the largest degree at which the vertex average of every monomial
    matches its average over the sphere. Derived here rather than read off
    Table 2.2 so that the ledger has a single generator; the values agree
    with povm_properties.py's independent computation (2, 3, 3, 5, 5).
    """
    def sphere_moment(e):
        if any(k % 2 for k in e):
            return 0.0                      # odd monomials integrate to zero
        num = math.prod(math.prod(range(k - 1, 0, -2)) or 1 for k in e)
        return num / (math.prod(range(sum(e) + 1, 0, -2)) or 1)

    for t in range(1, t_max + 1):
        for e in itertools.product(range(t + 1), repeat=3):
            if sum(e) == t and abs(float(np.mean(np.prod(s ** np.array(e), axis=1)))
                                   - sphere_moment(e)) > 1e-9:
                return t - 1
    return t_max


def alignment(s):
    """The fixed rotation A of R1, taking the vertex v0 nearest +z to zhat.

    Rodrigues' formula about the axis v0 x zhat. For the octahedron v0 is
    zhat itself and A = I -- the one solid whose R1 needs no alignment.
    """
    v = s[np.argmax(s[:, 2])]
    zhat = np.array([0.0, 0.0, 1.0])
    axis = np.cross(v, zhat)
    if np.linalg.norm(axis) < 1e-12:
        return np.eye(3), v
    axis /= np.linalg.norm(axis)
    ang = np.arccos(np.clip(v @ zhat, -1, 1))
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    A = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    # np.allclose/isclose keep rtol=1e-5 alongside any atol, so where the atol
    # has to be the whole bound the deviation is measured and compared
    # explicitly; allclose stands where the reference is 0, and in the float
    # searches, which the exact layer re-decides. Worst here 1.2e-16.
    d_A = np.abs(A @ v - zhat).max()
    assert d_A < 1e-12, f"alignment: A v0 != zhat, V = {len(s)} (max |dev| = {d_A:.2e})"
    return A, v


def channel_R1(s, R, T, t):
    """Exact estimator channel of R1 (randomized-projective) under noise
    r -> T r + t acting measurement-side.

    Protocol: draw g uniformly from the rotations R, apply U_g, apply the
    fixed alignment A (v0 -> zhat), read out Z. Snapshot = 3 b R_g^T v0
    (the canonical dual of a projective measurement). The outcome
    probability p(b) = (1 + b [T (A R_g r) + t]_z)/2 is linear in the input
    Bloch vector r, so E[snapshot] = M r + off with M and off computed
    below as exact group averages -- no sampling anywhere.
    """
    A, v = alignment(s)
    M = np.zeros((3, 3))
    off = np.zeros(3)
    for Rg in R:
        pre = A @ Rg
        for b in (+1, -1):
            lin = b * (T @ pre)[2, :] / 2.0          # coefficient of r in p(b)
            const = (1 + b * t[2]) / 2.0
            snap = 3.0 * b * (Rg.T @ v)
            M += np.outer(snap, lin)
            off += snap * const
    return M / len(R), off / len(R)


def channel_R2(s, R, T, t, s_actual=None):
    """Exact estimator channel of R2 (twirled-native) under the same noise.

    Protocol: draw g, apply U_g, measure the NATIVE (Naimark) POVM -- all V
    effects (1/V)(I + n_k . sigma) -- and relabel by g in post-processing.
    Snapshot = 3 R_g^T n_k; p(k) = (1/V)(1 + n_k . (T R_g r + t)). Again an
    exact average, all solids admitted (no decomposability needed).

    s_actual, if given, is the vertex list the device MEASURES, index for
    index, where s stays the list the estimator BELIEVES: snapshots are
    built from s, outcome probabilities from s_actual. This is the
    two-list channel of the price-of-inexactness theorem, which the thesis
    states in Appendix F.3.2. Both hypotheses matter.
    For an IRREDUCIBLE draw R any fixed mismatch twirls to exactly

        M = tr(B T)/V Id,  off = 0,   B = sum_k b_k a_k^T,

    b the believed vertex, a the measured one. Two readings of the same
    scalar: at T = Id it is the frame overlap (1/V) sum_k b_k . a_k, the
    kappa the callers price; with no mismatch B = (V/3) Id and it is
    Finding 1's tr(T)/3. The bare overlap is NOT the answer under noise --
    at T_NOISE the nine SIC derangements, every one -1/3 noiselessly,
    spread to nine DISTINCT prices (check_decker_outcome_order asserts
    both halves). And the zero offset is the DRAW's: off = 3 sum_k
    const_k [<R_g^T>] b_k = (3/V) [<R_g^T>] B t, so it vanishes for any
    belief list whatever and reports nothing about the mismatch -- drop
    irreducibility (R = [Id]) and it is (3/V) B t, the t itself of
    check_two_protocols only when the lists agree. Default: no mismatch.
    """
    a = s if s_actual is None else s_actual
    V = len(s)
    M = np.zeros((3, 3))
    off = np.zeros(3)
    for Rg in R:
        # strict: V normalizes the probabilities, so a short s_actual would
        # silently return a plausible depolarizing kappa off V - 1 terms
        for bk, ak in zip(s, a, strict=True):
            lin = (ak @ (T @ Rg)) / V                # coefficient of r in p(k)
            const = (1 + ak @ t) / V
            snap = 3.0 * (Rg.T @ bk)
            M += np.outer(snap, lin)
            off += snap * const
    return M / len(R), off / len(R)


def orbit_counts(s, R):
    """How often the R1 snapshot direction b R_g^T v0 hits each vertex.

    R1's coin: draw g and outcome b land the snapshot on b R_g^T v0.
    Uniform counts = the draw samples the POVM's vertices uniformly (the
    coin realizes the POVM); a zero = that vertex is never measured at all.
    """
    _, v = alignment(s)
    hits = np.zeros(len(s), dtype=int)
    for Rg in R:
        for b in (+1, -1):
            w = b * (Rg.T @ v)
            d = np.linalg.norm(s - w, axis=1)
            k = int(np.argmin(d))
            assert d[k] < 1e-9, "snapshot direction is not a vertex"
            hits[k] += 1
    return hits


def subgroup_lattice(R):
    """All subgroups of the rotation group R, as frozensets of indices.

    Pair closure finds them: close every generator pair {g, h} under
    multiplication. Completeness is then verified in situ, with no input
    from the classification: adjoining any single element to any subgroup
    found must land back in the family (induction on generating sets does
    the rest), so the returned lattice is provably the whole one.
    """
    n = len(R)
    idx = {rot_key(Rg): i for i, Rg in enumerate(R)}
    mul = np.array([[idx[rot_key(Ri @ Rj)] for Rj in R] for Ri in R])

    def close(gens):
        S = set(gens)
        while True:
            new = set(mul[np.ix_(sorted(S), sorted(S))].ravel().tolist()) - S
            if not new:
                return frozenset(S)
            S |= new

    subs = {close((i, j)) for i in range(n) for j in range(i, n)}
    assert all(close((*S, g)) in subs
               for S in subs for g in range(n) if g not in S), "lattice incomplete"
    return subs


_LATTICE = {}


def lattice(g):
    """subgroup_lattice(load_rotations(g)), memoized (the sweep reuses it)."""
    if g not in _LATTICE:
        _LATTICE[g] = subgroup_lattice(load_rotations(g))
    return _LATTICE[g]


def subgroup_kind(R, S):
    """Structural name of a rotation subgroup, from its element orders.

    Enough of the classification to label the sweep: cyclic when some
    element has the group's own order, Klein V at order 4 otherwise, the
    three polyhedral groups by (order, top element order), dihedral for the
    rest. Element orders are integers read off a float power, and a
    rotation's powers stay well away from the identity until they hit it.
    """
    n = len(S)
    if n == 1:
        return "1"
    orders = []
    for i in sorted(S):
        k, M = 1, R[i]
        while not np.allclose(M, np.eye(3), atol=1e-9):
            M, k = M @ R[i], k + 1
        orders.append(k)
    top = max(orders)
    if top == n:
        return f"C_{n}"
    if n == 4:
        return "V"
    if (n, top) in ((12, 3), (24, 4), (60, 5)):
        return {12: "T", 24: "O", 60: "I"}[n]
    return f"D_{n // 2}"


def by_order(R, subs):
    """{order: (count, sorted structural names)} of a set of subgroups."""
    out = {}
    for S in sorted(subs, key=lambda S: (len(S), sorted(S))):
        cnt, names = out.get(len(S), (0, set()))
        out[len(S)] = (cnt + 1, names | {subgroup_kind(R, S)})
    return {k: (c, sorted(ns)) for k, (c, ns) in sorted(out.items())}


def two_bars(solid):
    """The two bars for one solid, over the complete subgroup lattice.

    REALIZE (the coin reproduces the POVM: the seed's orbit is the whole
    vertex set, uniformly) and TWIRL (the estimator channel is exactly
    depolarizing) are tested independently on every subgroup of the solid's
    covariance group. The minimal draw is the smallest group clearing both,
    i.e. the smallest member of the intersection; that this equals the
    larger of the two bars is asserted, not assumed (see below). `binds`
    names which bar sets it, `draw_groups` the subgroups that attain it --
    five at the icosahedron, one everywhere else.

    The sweep is exhaustive over every finite subgroup of SO(3), not merely
    over the covariance group: if G's orbit of the seed is the whole vertex
    set then G permutes that set, hence sits inside the solid's rotation
    group -- so a draw outside the lattice cannot realize at all.
    """
    g = COVARIANCE[solid]
    R = load_rotations(g)
    subs = lattice(g)
    s = load_vertices(solid)
    irr, realize, twirl = set(), set(), set()
    reach = {}
    for S in subs:
        RS = R[sorted(S)]
        # irreducibility is one number: <chi, chi> = mean tr(R)^2 = 1
        if np.isclose(np.mean([np.trace(R[i]) ** 2 for i in S]), 1.0):
            irr.add(S)
        hits = orbit_counts(s, RS)
        reach[S] = int((hits > 0).sum()) // 2         # vertex axes reached
        if hits.min() == hits.max():
            realize.add(S)
        M, o = channel_R1(s, RS, T_NOISE, t_NOISE)
        # a gap, not a boundary: accepted deviations reach 4.4e-16 over both
        # lattices, the closest miss is 2.5e-2
        if (np.abs(M - T_NOISE[2, 2] * np.eye(3)).max() < 1e-9
                and np.allclose(o, 0, atol=1e-9)):
            twirl.add(S)
    bar_r, bar_t = min(map(len, realize)), min(map(len, twirl))
    # The draw must clear BOTH bars at once, so it is the smallest member of
    # the intersection; max(bar_r, bar_t) is only a lower bound for that,
    # since nothing forces a minimal twirling group to also realize. The
    # bound is attained because the two sets NEST (check_subgroup_sweep
    # asserts the nesting: whichever set is contained in the other supplies
    # the minimum). The icosahedron is where that has teeth -- realize and
    # twirl are the same five T-conjugates, not merely the same order; had
    # the two picked different T's, both bars would still read 12 while the
    # smallest group clearing both jumped to 60.
    both = realize & twirl
    draw = min(map(len, both))
    assert draw == max(bar_r, bar_t), (solid, draw, bar_r, bar_t)
    return {
        "group": g, "rotations": R, "subgroups": subs, "axes": len(s) // 2,
        "irr": irr, "realize": realize, "twirl": twirl, "reach": reach,
        "bar_realize": bar_r, "bar_twirl": bar_t, "draw": draw,
        "binds": ("both" if bar_r == bar_t else
                  "realize" if bar_r > bar_t else "twirl"),
        "min_realize": sorted({subgroup_kind(R, S) for S in realize
                               if len(S) == bar_r}),
        "min_twirl": sorted({subgroup_kind(R, S) for S in twirl
                             if len(S) == bar_t}),
        "n_min_realize": sum(1 for S in realize if len(S) == bar_r),
        "draw_groups": {S for S in both if len(S) == draw},
        "ceiling": max(reach[S] for S in subs if len(S) < len(R)),
    }


def atlas_vertices(solid):
    """symbolic_solids()[solid], permuted into povm_{solid}.npz order.

    That npz order IS the published numbering: povm_properties.py emits it as
    tab:povm-atlas, and paper/figures/platonic_solid_povms.tex draws the same
    indices.  symbolic_solids() does NOT match it -- same solids, same pose,
    same vertex sets, different numbering for the tetrahedron, cube and
    icosahedron.  So anything the thesis prints WITH a vertex index has to run
    here, on the numbering a reader can look up; see exact_vertices() below for
    why the permutation is safe, and det_witness() for what depends on it.
    """
    sym = symbolic_solids()[solid]
    num = np.array([[float(c) for c in v] for v in sym])
    out = []
    for row in load_vertices(solid):
        dist = np.linalg.norm(num - row, axis=1)
        near = np.sort(dist)[:2]
        assert near[0] < 1e-12 and near[1] > 0.1, (solid, row, near)
        out.append(sym[int(np.argmin(dist))])
    return out


# ---------------------------------------------------------------------------
# Atlas circuits: pricing the draw and the coin
# ---------------------------------------------------------------------------

def best_circuits(atlas):
    """Min-(magic, depth) dij circuit per SO(3) rotation, over the +-q pair.

    Each rotation appears twice in the binary group (as U and -U); the
    cheaper min-magic (Dijkstra) synthesis represents it. Returns
    {rot_key: (magic, depth, sequence)}.
    """
    best = {}
    for U, seq, depth, magic in zip(atlas["unitaries"], atlas["dij_sequences"],
                                    atlas["dij_depths"], atlas["dij_magic_costs"]):
        k = rot_key(rotation_from_unitary(U))
        cand = (int(magic), int(depth), str(seq))
        if k not in best or cand[:2] < best[k][:2]:
            best[k] = cand
    return best


def coset_representatives(s, R, circuits):
    """One cheapest atlas circuit per vertex axis of the solid.

    The g's whose snapshot hits a given axis {n, -n} (i.e. R_g^T v0 = +-n)
    form a coset of the stabilizer of v0; any representative realizes that
    axis, so the min-(magic, depth) one prices it. Asserts the cosets
    partition the group evenly (= vertex-transitivity). Returns one
    (magic, depth, sequence) triple per axis.
    """
    _, v = alignment(s)
    axes = []                       # one representative vertex per axis
    for n in s:
        if not any(np.allclose(n, -m, atol=1e-9) for m in axes):
            axes.append(n)
    reps, sizes = [], []
    for n in axes:
        members = [g for g, Rg in enumerate(R)
                   if np.allclose(Rg.T @ v, n, atol=1e-9)
                   or np.allclose(Rg.T @ v, -n, atol=1e-9)]
        assert members, "group not transitive on the vertex axes"
        sizes.append(len(members))
        reps.append(min(circuits[rot_key(R[g])] for g in members))
    assert len(set(sizes)) == 1 and sizes[0] * len(axes) == len(R), \
        "cosets do not partition the group evenly"
    return reps


def frame_potential(U, t):
    """Frame potential F_t = mean over pairs of |tr(U_a+ U_b)|^(2t).

    Equals the Haar value (the Catalan number C_t) iff the set is a unitary
    t-design; for a group the value is an integer, so failing a level
    overshoots the Catalan number by at least 1.
    """
    tr = np.einsum("aij,bij->ab", np.conj(U), U)
    return float((np.abs(tr) ** (2 * t)).mean())


# ---------------------------------------------------------------------------
# Symbolic layer: exact vertex sets, atlas gates, field arithmetic
# (TAU_SYM / SIG_SYM are declared with the other constants at the top -- the
# per-solid fields of the exact two-bars companion need them earlier.)
# ---------------------------------------------------------------------------

PAULI_SYM = [Matrix([[0, 1], [1, 0]]),
             Matrix([[0, -sI], [sI, 0]]),
             Matrix([[1, 0], [0, -1]])]


def symbolic_solids():
    """The five Platonic vertex sets, exact, in the atlas orientation."""
    s_ico = 1 / sqrt(2 + TAU_SYM)
    c_dod = 1 / sqrt(3)
    S = {}
    S["tetrahedron"] = [Matrix(v) / sqrt(3)
                        for v in [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]]
    S["octahedron"] = [Matrix(v) for v in
                       [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
    S["cube"] = [Matrix(v) / sqrt(3) for v in itertools.product([1, -1], repeat=3)]
    S["icosahedron"] = (
        [Matrix([a * s_ico * TAU_SYM, b * s_ico, 0]) for a in (1, -1) for b in (1, -1)]
        + [Matrix([0, a * s_ico * TAU_SYM, b * s_ico]) for a in (1, -1) for b in (1, -1)]
        + [Matrix([a * s_ico, 0, b * s_ico * TAU_SYM]) for a in (1, -1) for b in (1, -1)])
    S["dodecahedron"] = (
        [Matrix(v) / sqrt(3) for v in itertools.product([1, -1], repeat=3)]
        + [Matrix([0, a * c_dod * SIG_SYM, b * c_dod * TAU_SYM]) for a in (1, -1) for b in (1, -1)]
        + [Matrix([a * c_dod * SIG_SYM, b * c_dod * TAU_SYM, 0]) for a in (1, -1) for b in (1, -1)]
        + [Matrix([a * c_dod * TAU_SYM, 0, b * c_dod * SIG_SYM]) for a in (1, -1) for b in (1, -1)])
    return S


def atlas_gates():
    """The thesis gates whose rotation axes the solids inherit (SymPy)."""
    H = (1 / sqrt(2)) * Matrix([[1, 1], [1, -1]])
    S = Matrix([[1, 0], [0, sI]])
    return {
        "X": Matrix([[0, 1], [1, 0]]),
        "Z": Matrix([[1, 0], [0, -1]]),
        "F": H * S.conjugate().T,                    # face gate F = H S+
        "Phi": Rational(1, 2) * Matrix([[TAU_SYM + sI * SIG_SYM, 1],
                                        [-1, TAU_SYM - sI * SIG_SYM]]),
    }


def bloch_axis(U):
    """Rotation axis of a 2x2 unitary, exact and normalized.

    In SU(2) form U = cos(t/2) I - i sin(t/2) n.sigma, so
    tr(sigma_j U) = -2i sin(t/2) n_j: n is parallel to (i/2) tr(sigma_j U).
    """
    U = sp.simplify(U / sp.sqrt(U.det()))
    n = Matrix([sp.simplify(sI * sp.trace(P * U) / 2) for P in PAULI_SYM])
    n = sp.simplify(n / sp.sqrt(n.dot(n)))
    return sp.Matrix([sp.radsimp(sp.nsimplify(c)) for c in n])


def bloch_matrix(U):
    """SO(3) action of a 2x2 unitary, exact: R_ij = tr(sigma_i U sigma_j U+)/2.

    Quadratic in the entries of U, which is why the 1/sqrt2 of H and F never
    reaches SO(3) (it pairs off) and why global phase cancels outright -- the
    two facts that make Q(sqrt5), not K_R, the field of the ATLAS-GENERATED
    rotations, those of every thesis gate set with no T in it. Not of the
    others: Bloch(T) = Rz(45 deg) leaves Q(sqrt5) and stays in K_R, which is
    the whole of check_reorientation_obstruction's second half.
    """
    Ud = U.conjugate().T
    return Matrix(3, 3, lambda i, j:
                  sp.radsimp(sp.nsimplify(sp.simplify(
                      sp.trace(PAULI_SYM[i] * U * PAULI_SYM[j] * Ud) / 2))))


# Every entry of every polyhedral rotation matrix in the atlas pose, exactly:
# T and O are signed permutations, I adds the golden half-integers. Nine
# values, no two closer than 0.19.
#
# The list is a CENSUS, not a decision procedure: exact_rotations builds the
# matrices from main.py's canonical quaternions and checks their entries
# against these nine values. Lifting float rotations onto the list instead --
# nearest entry within 1e-9, one coordinate at a time -- would be rot_key's
# rounding grid again, a nine-value codebook in place of nine decimals, and
# would put a tolerance back in front of everything downstream.
ROT_ENTRIES = [sp.Integer(0), sp.Integer(1), sp.Integer(-1),
               Rational(1, 2), -Rational(1, 2),
               TAU_SYM / 2, -TAU_SYM / 2, SIG_SYM / 2, -SIG_SYM / 2]


_EXACT_ROT = {}


def _quat_to_rotation_sym(q):
    """Symbolic twin of export_numpy.quat_to_rotation -- same (w, -z, -y, -x)."""
    w, x, y, z = q.w, -q.z, -q.y, -q.x
    return Matrix([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y)],
    ])


def exact_rotations(g):
    """Rotations of T/O/I as exact matrices, in data/group_{g}.npz ROW ORDER.

    No matching step and no tolerance.  The npz is written by export_numpy's
    rotation_group_to_numpy(), which sorts the binary group's quaternions and
    keeps one per proj_hash class; re-running that path without the float cast
    returns the same elements in the same order BY CONSTRUCTION, so index i
    here is index i there.  proj_hash is main.py's canonical form, so the
    correspondence is decided, not searched for.

    Three checks, none of them load-bearing for the construction: SO(3)
    membership, and that the entries fall in the ROT_ENTRIES census.  Order
    against the npz is verified in check_exact_two_bars.

    Memoized: rebuilding I costs 0.5 s apiece.  The list is copied out so a
    caller cannot grow the cached one.
    """
    if g in _EXACT_ROT:
        return list(_EXACT_ROT[g])
    seen, reps = set(), []
    for q in sorted(geometric_group("2" + g)):
        ph = q.proj_hash()
        if ph not in seen:
            seen.add(ph)
            reps.append(q)
    out = []
    for q in reps:
        M = _quat_to_rotation_sym(q)
        assert sp.expand(M.T * M - sp.eye(3)) == sp.zeros(3, 3)
        assert sp.expand(M.det()) == 1
        assert all(any(sp.expand(M[i, j] - e) == 0 for e in ROT_ENTRIES)
                   for i in range(3) for j in range(3)), (g, M)
        out.append(M)
    _EXACT_ROT[g] = out
    return list(out)


def state_from_bloch(n):
    """Density matrix (I + n . sigma)/2 of a Bloch vector, exact."""
    rho = (sp.eye(2) + sum((nj * P for nj, P in zip(n, PAULI_SYM)),
                           sp.zeros(2, 2))) / 2
    return sp.simplify(rho)


def on_solid(n, verts):
    """Is the exact vector n one of the vertices?"""
    return any(sp.simplify(n - v) == Matrix([0, 0, 0]) for v in verts)


def in_field(x, theta):
    """Is the algebraic number x an element of Q(theta)?

    field_isomorphism embeds Q(x) into Q(theta) iff x lies in Q(theta).
    """
    x = sp.simplify(x)
    if x.is_Rational:
        return True
    return sp.field_isomorphism(sp.AlgebraicNumber(x),
                                sp.AlgebraicNumber(theta)) is not None


def dyadic_order(x, kmax=8):
    """Smallest k with 2^k x an algebraic integer, or None if there is none.

    Every thesis gate has entries in calR = Z[tau, i, sqrt2, 1/2]: algebraic
    integers over a power of two (H and F carry a 1/sqrt2, Phi a 1/2, while
    T's exp(i pi/4) is an algebraic integer outright). calR is a ring, closed
    under conjugation, so a sum or product of gate entries never leaves it
    -- which is what pins a deterministic protocol's branch amplitudes, and
    with them the traces of the effects those branches realize. On the
    rationals R cuts down to Z[1/2], where the test is decidable outright:
    p/q in lowest terms qualifies iff q is a power of two.
    """
    x = sp.nsimplify(sp.simplify(x))
    if x.is_Rational:
        q = sp.denom(x)
        k = sp.multiplicity(2, q) if q % 2 == 0 else 0
        return k if q // 2**k == 1 else None
    y = sp.Symbol("_y")
    for k in range(kmax + 1):
        if sp.minimal_polynomial(2**k * x, y, polys=True).LC() == 1:
            return k
    return None


def det_witness(verts):
    """(1-based indices, det) of the first non-coplanar vertex triple.

    tab:povm-exactness prints both, so both must come from one computation --
    and from atlas_vertices(), since the caption sends the reader to the atlas
    to look the indices up.  The first three vertices span for three solids and
    are coplanar for the octahedron and the icosahedron, whence (1,3,5) and
    (1,2,5) there.
    """
    for trip in itertools.combinations(range(len(verts)), 3):
        d = sp.simplify(Matrix.hstack(*[verts[i] for i in trip]).det())
        if d != 0:
            return tuple(i + 1 for i in trip), sp.radsimp(d)
    raise AssertionError("no spanning triple")            # verts span R^3


def det_invariant(verts):
    """det[v_a v_b v_c] of the first non-coplanar vertex triple.

    A rotation R has det R = 1, so det[Rv_a Rv_b Rv_c] = det[v_a v_b v_c]: the
    value is orientation-free, and every vertex being K_R-rational forces
    it into K_R -- one det outside K_R obstructs the whole solid.

    Which triple is a choice, and it moves the number: the sign flips with the
    triple's handedness, and the magnitude moves too on the icosahedron (two
    values) and the dodecahedron (five, realizing FOUR minimal polynomials).
    What it cannot move is the coset -- check_obstruction() sweeps every
    spanning triple of every solid and finds all their determinants in one
    K_R-multiple class -- which is the computational face of Lemma 5's "same
    square class, so any one triple decides", and the reason a table may print
    a single row per solid at all.
    """
    return det_witness(verts)[1]


def twirl_bar(solid):
    """Smallest order in the lattice whose R2 draw twirls this solid exactly.

    The twirl bar in its route-free form: R2 is defined for all five solids
    (R1 is not), so this is the one bar the tetrahedron also has.
    """
    g = COVARIANCE[solid]
    R, s = load_rotations(g), load_vertices(solid)
    kappa = np.trace(T_NOISE) / 3
    ok = set()
    for S in lattice(g):
        M, o = channel_R2(s, R[sorted(S)], T_NOISE, t_NOISE)
        # same classifier as two_bars: accepted deviations reach 8.9e-16 over
        # all five lattices, closest miss 1.1e-3
        if (np.abs(M - kappa * np.eye(3)).max() < 1e-9
                and np.allclose(o, 0, atol=1e-9)):
            ok.add(S)
    return min(map(len, ok)), ok


def _fmt_orders(d):
    """by_order output -> '3(x4 C_3), 12 (T)'."""
    return ", ".join(f"{o}(x{c} {'/'.join(ns)})" if c > 1 else f"{o} ({'/'.join(ns)})"
                     for o, (c, ns) in d.items())


def rank_one_twirl(Rs, v, w):
    """(3 E_g R_g^T v w^T R_g,  3 E_g R_g^T v): the R1 channel, reduced.

    R1's estimator channel depends on the noise only through the rank-one
    v w^T, with v = A^T zhat the seed vertex and w = A^T T^T zhat -- the
    alignment cancels between snapshot and readout, and the twirl never sees
    the noise map itself. Works over SymPy or numpy alike; the reduction is
    verified against channel_R1 subgroup by subgroup in check_coin_group.
    """
    n = len(Rs)
    if isinstance(v, Matrix):
        return (3 * sum((R.T * (v * w.T) * R for R in Rs), sp.zeros(3, 3)) / n,
                3 * sum((R.T * v for R in Rs), sp.zeros(3, 1)) / n)
    return (3 * np.mean([R.T @ np.outer(v, w) @ R for R in Rs], axis=0),
            3 * np.mean([R.T @ v for R in Rs], axis=0))


def _Rz(c, s): return Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def _Ry(c, s): return Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# One reorientation per solid, carrying DECKER's pose onto the atlas's: the
# (Fourier block size, R, field demand) of Table D.3. R is one representative
# of a coset of the solid's rotation group, so there are |G| of them -- 12, 24,
# 24, 60, 60. Shared by check_reorientation_obstruction (which prices R) and
# check_decker_outcome_order (which derives Decker's pose from his circuits and
# checks these R against it); defined once so the two cannot drift apart.
# The REPRESENTATIVE is load-bearing, not just the coset: tab:decker-labels
# prints each R's induced outcome permutation and the circuit figures draw
# U_R = R^-1, so every entry below must stay the literal factor product
# Table D.3 prints -- a coset-mate passes every set-level check here while
# owing a different permutation.
# (The block sits directly above REORIENT so lift_assign("REORIENT") carries
# it into the walkthrough by its own comment-walking rule, not as a lift
# neighbor's lexical baggage.)
REORIENT = {
    "tetrahedron": (2, _Rz(sqrt(2) / 2, sqrt(2) / 2), "sqrt2"),
    "octahedron": (3, _Rz(sqrt(2) / 2, sqrt(2) / 2)
                   * _Ry(sqrt(3) / 3, sqrt(6) / 3) * _Rz(-1, 0), "sqrt3"),
    "cube": (4, _Rz(sqrt(2) / 2, sqrt(2) / 2), "sqrt2"),
    "icosahedron": (3, _Ry(1 / (TAU_SYM * sqrt(3)), -TAU_SYM / sqrt(3)),
                    "sqrt3"),
    "dodecahedron": (5, _Ry(TAU_SYM / sqrt(TAU_SYM + 2),
                            1 / sqrt(TAU_SYM + 2)), "tau/sqrt(tau+2)"),
}
