"""
Numpy-native re-derivation of the three pillars, cross-checked against the
symbolic atlas.

Every "key computation" in this project is done *symbolically* (SymPy over
Q(sqrt(2), sqrt(5), i)) for maximal correctness -- `main.py` (binary polyhedral
groups + circuit synthesis) and `povm_properties.py` (Platonic-solid POVMs).
This script re-derives all three pillars from the mathematics directly, in
plain numpy, then diffs the result against the symbolic-derived
`code/data/*.npz` as a sanity check:

  Pillar 1  binary polyhedral groups 2T / 2O / 2I as finite subgroups of SU(2),
            built two independent ways -- 4D polytope geometry and the closure
            of the Conway & Smith generators -- that must agree; plus the
            second anchored copy of 2I (the Phi* group) and its exchange with
            the Phi copy under Clifford conjugation.
  Pillar 2  the four circuit synthesizers (min depth / min magic cost, each
            exact in SU(2) or up to global phase in PU(2) = U(2)/U(1) ≅ SO(3)).
  Pillar 3  the five Platonic-solid POVMs and their informational-completeness
            and spherical / projective t-design properties.

This is a *clean-room* re-implementation, not a line-port of the symbolic code:
it is structured the way numpy wants the problem posed, so it reaches the same
numbers by a genuinely different route (which is what makes it worth having as
an independent oracle). Three reframings carry the file:

  * A group is an `(N, 4)` float array. Quaternion multiplication is bilinear,
    so *all* pairwise products are a single `einsum` against the (4,4,4)
    structure-constant tensor, and "identity up to a rounding grid" is
    `np.unique` on rounded rows. Generator closure is: multiply the current set
    by the generators, dedup, repeat until the count stabilizes.

  * Synthesis is shortest paths on a *finite* Cayley graph. The group has <=120
    elements, so we label them 0..N-1 and each gate becomes a permutation of
    those labels (right-multiplication by a fixed element is a bijection). Min
    depth and min-magic-then-depth are then single-source shortest paths solved
    by vectorized Bellman-Ford relaxation -- no growing visited set, no search
    over float-quaternion space.

  * A t-design check is a moment-tensor comparison. The degree-t discrete
    moment tensor is `mean_v  n_v (x) ... (x) n_v` built with one `einsum`;
    it must equal the analytic sphere moment tensor through degree t.

Exact symbolic identity (`_to_basis` / `proj_hash` in main.py) is replaced by
float rounding: a quaternion is a length-4 float64 array; two are "the same"
iff their components agree to 9 decimals. A rounding key can fail two ways and
safety means ruling out both.

  * COLLISION -- two distinct elements landing on one key -- is ruled out by
    separation. Components are algebraic numbers from a small fixed alphabet,
    and distinct elements differ in some component by >= 0.309 (2I; 0.5 for 2T
    and 2O), against a grid of 1e-9.

  * SPLITTING -- one element landing on two keys, because float noise carries
    it across a rounding boundary -- is ruled out by the distance from an exact
    entry to that boundary, which is NOT the grid spacing. 2T's {0, +-1/2, +-1}
    are grid multiples and get the full half-spacing 5e-10; 2O's 1/sqrt2 clears
    a boundary by 3.13e-10; 2I's tau/2 and sigma/2 clear one by only 1.25e-10,
    tied because the two differ by exactly 1/2, itself a grid multiple. Worst
    float error over every call site (closure products, the SU(2) round trip,
    the 48 Clifford conjugations, replay of the synthesized words) is 4.2e-16,
    so 2I binds the margin at 3.0e5.

Splitting is the direction that binds, and it is the one a "float noise ~1e-13
against a 1e-9 grid" reading leaves unaddressed -- that argument names neither
quantity, and rules out only the failure that was never close: a tolerance is
warranted by the measured worst case against the measured separation, never by
an order-of-magnitude gesture. `qkey` stays float deliberately: this module is
main.py's independent oracle, so importing its canonical form would destroy the
cross-check.

It writes no .tex / .npz. When the symbolic-derived `code/data/*.npz` are
present it additionally diffs against them -- see `cross_check_npz` for which
quantities are compared and which are deliberately not.

    cd code && uv run numpy_atlas.py

Exits 0 with a verification report if everything agrees; raises (non-zero exit)
on the first disagreement.
"""

from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np

# =============================================================================
# Constants
# =============================================================================

SQRT2 = np.sqrt(2.0)
SQRT3 = np.sqrt(3.0)
SQRT5 = np.sqrt(5.0)
TAU = (1.0 + SQRT5) / 2.0       # golden ratio       (tau in Conway & Smith)
SIGMA = (SQRT5 - 1.0) / 2.0     # inverse golden ratio (sigma = 1/tau)

# np.allclose/isclose keep rtol=1e-5 alongside any atol, and it scales with the
# expected operand, not the residual -- so where the atol has to be the whole
# bound, the deviation is measured and compared explicitly.
TOL = 1e-9                      # matrix / scalar comparison tolerance
DESIGN_TOL = 1e-12              # t-design classifier: matches <= 2.2e-16, misses >= 1.1e-2
KEY_DECIMALS = 9                # quaternion identity grid
DEPTH_CAP = 10_000              # depth-digit budget when packing (magic, depth)

I2 = np.eye(2, dtype=np.complex128)
SX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
PAULI = np.stack([SX, SY, SZ])                 # (3, 2, 2), for n . sigma via einsum

SWAP4 = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
], dtype=np.complex128)

IDENT = np.array([1.0, 0.0, 0.0, 0.0])         # identity quaternion

DATA_DIR = Path(__file__).parent / "data"


# =============================================================================
# Core primitive: float quaternions, with rounding-based identity
# =============================================================================

def qmul(a, b):
    """Hamilton product a*b of two unit quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def qconj(q):
    """Conjugate = inverse for a unit quaternion."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


# Structure-constant tensor C[i,j,k] for the quaternion basis {1, i, j, k}:
# (a*b)_k = sum_{i,j} a_i b_j C[i,j,k]. Lets us batch *all* pairwise products of
# an (N,4) array as a single einsum -- the numpy-native form of group closure.
QC = np.array([[qmul(np.eye(4)[i], np.eye(4)[j]) for j in range(4)] for i in range(4)])


def qkey(q, nd=KEY_DECIMALS):
    """Hashable identity key: components rounded to `nd` decimals.

    The float stand-in for main.py's exact `_to_basis` tuple. (+ 0.0 maps the
    signed zero -0.0 to 0.0 so equal keys hash and compare identically.)
    """
    return tuple(round(float(v), nd) + 0.0 for v in q)


def qkey_proj(q, nd=KEY_DECIMALS):
    """Projective key identifying q with -q (SO(3) / global-phase equivalence).

    The float stand-in for main.py's `proj_hash`: lexicographic min of the key
    and its negation, so q and -q collapse to the same value.
    """
    k = qkey(q, nd)
    neg = tuple(-v + 0.0 for v in k)
    return min(k, neg)


def quat_to_unitary(q):
    """Unit quaternion -> 2x2 SU(2) matrix [[a, b], [-b*, a*]], a=w+ix, b=y+iz."""
    w, x, y, z = q
    a = w + 1j * x
    b = y + 1j * z
    return np.array([[a, b], [-np.conj(b), np.conj(a)]], dtype=np.complex128)


def unitary_to_quat(U):
    """2x2 SU(2) matrix -> unit quaternion (inverse of quat_to_unitary).

    Reads (w, x) off the (0,0) entry and (y, z) off the (0,1) entry; lets us map
    the npz `unitaries` rows back to quaternion keys.
    """
    a = U[0, 0]
    b = U[0, 1]
    return np.array([a.real, a.imag, b.real, b.imag])


def quat_to_rotation(q):
    """Unit quaternion -> 3x3 SO(3) Bloch rotation (q and -q give the same R).

    Physical convention: R is the rotation that rho -> U_q rho U_q^dagger
    induces on Bloch vectors (the units i, j, k act as pi rotations about
    z-hat, y-hat, x-hat), i.e. the textbook formula applied to (w,-z,-y,-x).
    """
    w, x, y, z = q[0], -q[3], -q[2], -q[1]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def _unique_rows(A):
    """Unique rows of an (N,4) array by the rounding grid, first-seen order.

    The numpy-native replacement for a dict-of-tuples dedup: round to the
    identity grid, take `np.unique` along axis 0, then index back into the
    *original* (unrounded) rows so representatives never drift.
    """
    keys = np.round(A, KEY_DECIMALS) + 0.0          # +0.0 kills signed-zero rows
    _, idx = np.unique(keys, axis=0, return_index=True)
    return A[np.sort(idx)]


def _dedup_proj(A):
    """One representative row per projective class {q, -q}, first-seen order."""
    seen, out = set(), []
    for q in A:
        k = qkey_proj(q)
        if k not in seen:
            seen.add(k)
            out.append(q)
    return np.array(out)


def _key_set(A):
    """Set of rounded-row tuples -- the float analogue of a set of group hashes."""
    return {tuple(r) for r in (np.round(A, KEY_DECIMALS) + 0.0)}


# =============================================================================
# Gate definitions (float SU(2) and U(2) forms)
# =============================================================================
# Convention G_SU2 = phase * G_U2 with det(G_SU2) = 1, matching main.py.

_U2 = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    "H": np.array([[1, 1], [1, -1]], dtype=np.complex128) / SQRT2,
    "S": np.array([[1, 0], [0, 1j]], dtype=np.complex128),
}
_U2["F"] = _U2["H"] @ _U2["S"].conj().T          # F = H S^dagger
_U2["Phi"] = 0.5 * np.array([
    [TAU + 1j * SIGMA, 1],
    [-1, TAU - 1j * SIGMA],
], dtype=np.complex128)

_SU2 = {
    "X": -1j * _U2["X"],
    "Z": -1j * _U2["Z"],
    "H": -1j * _U2["H"],
    "S": ((1 - 1j) / SQRT2) * _U2["S"],
}
_SU2["F"] = _SU2["H"] @ _SU2["S"].conj().T
_SU2["Phi"] = _U2["Phi"]                          # already det = 1

GATE_NAMES = ["X", "Z", "H", "S", "F", "Phi"]
MAGIC = {"X": 0, "Z": 0, "H": 0, "S": 0, "F": 0, "Phi": 1}   # only Phi is non-Clifford


def _su2_phase(name):
    """Recover phase with G_SU2 = phase * G_U2 from the max-magnitude entry."""
    u, s = _U2[name], _SU2[name]
    idx = np.unravel_index(np.argmax(np.abs(u)), u.shape)
    return s[idx] / u[idx]


PHASE = {g: _su2_phase(g) for g in GATE_NAMES}

# Verify the gate algebra exactly as main.py does: phase * U2 == SU2, det == 1.
for _g in GATE_NAMES:
    # residual <= 1 ulp (1.2e-16 over the six gates)
    _d_phase = np.abs(PHASE[_g] * _U2[_g] - _SU2[_g]).max()
    assert _d_phase < 1e-12, f"phase check {_g} (max |dev| = {_d_phase:.2e})"
    assert abs(np.linalg.det(_SU2[_g]) - 1.0) < TOL, f"det check {_g}"

QUAT = {g: unitary_to_quat(_SU2[g]) for g in GATE_NAMES}

# Registry: name -> (quaternion, inverse quaternion, magic cost, U2->SU2 phase).
GATES = {g: (QUAT[g], qconj(QUAT[g]), MAGIC[g], PHASE[g]) for g in GATE_NAMES}

GATESET = {
    "2T": ["X", "Z", "F"],
    "2O": ["X", "Z", "F", "H", "S"],
    "2I": ["X", "Z", "F", "Phi"],
}


def directed_gates(mode):
    """[(name, quat, magic), (name+"'", inv_quat, magic), ...] for the searches.

    Each generator contributes a forward edge and an adjoint edge; the adjoint
    carries the same magic cost (Phi and Phi' are equally non-Clifford).
    """
    out = []
    for n in GATESET[mode]:
        q, qi, magic, _ = GATES[n]
        out.append((n, q, magic))
        out.append((n + "'", qi, magic))
    return out


# =============================================================================
# Pillar 1 -- group generation (geometric and algebraic)
# =============================================================================

def _parity(p):
    """Parity of a permutation tuple via inversion count (0 = even)."""
    return sum(p[i] > p[j] for i in range(len(p)) for j in range(i + 1, len(p))) % 2


def geometric_group(mode):
    """Group elements from 4D polytope geometry, as a deduped (N,4) array."""
    blocks = [
        np.vstack([np.eye(4), -np.eye(4)]),                 # +-1, +-i, +-j, +-k
        np.array(list(product((0.5, -0.5), repeat=4))),     # (+-1 +-i +-j +-k)/2
    ]

    if mode == "2O":
        # dual 24-cell: +-1/sqrt(2) in each pair of coordinates
        val = 1.0 / SQRT2
        extra = []
        for i, j in combinations(range(4), 2):
            for si, sj in product((1.0, -1.0), repeat=2):
                v = [0.0] * 4
                v[i], v[j] = si * val, sj * val
                extra.append(v)
        blocks.append(np.array(extra))

    elif mode == "2I":
        # 96 extra vertices: even permutations of (1/2, tau/2, sigma/2, 0)
        vals = np.array([0.5, TAU / 2, SIGMA / 2, 0.0])
        extra = []
        for p in (q for q in permutations(range(4)) if _parity(q) == 0):
            pv = vals[list(p)]                  # pv[i] = vals[p[i]]
            zero_pos = p.index(3)               # where the 0 component sits
            for signs in product((1.0, -1.0), repeat=3):
                v, s = [0.0] * 4, iter(signs)
                for i in range(4):
                    if i != zero_pos:
                        v[i] = pv[i] * next(s)
                extra.append(v)
        blocks.append(np.array(extra))

    elif mode != "2T":
        raise ValueError(f"unknown mode: {mode}")

    return _unique_rows(np.vstack(blocks))


def conway_group(mode):
    """Closure of the Conway & Smith generators, via batched einsum products."""
    gen_w = np.array([-0.5, 0.5, 0.5, 0.5])                 # w = (-1 + i + j + k)/2
    gen_map = {
        "2T": np.array([0.0, 1.0, 0.0, 0.0]),                  # i_T = i
        "2O": np.array([0.0, 0.0, 1.0 / SQRT2, 1.0 / SQRT2]),  # i_O = (j+k)/sqrt2
        "2I": np.array([0.0, 0.5, SIGMA / 2, TAU / 2]),        # i_I = (i+sigma j+tau k)/2
    }
    gens = _unique_rows(np.vstack([gen_w, gen_map[mode]]))

    group = gens.copy()
    while True:
        # All products of the current set with the generators, both orders, at
        # once: (M,4) x (G,4) -> (M,G,4) via the structure-constant tensor.
        right = np.einsum("ai,bj,ijk->abk", group, gens, QC).reshape(-1, 4)
        left = np.einsum("ai,bj,ijk->abk", gens, group, QC).reshape(-1, 4)
        grown = _unique_rows(np.vstack([group, right, left]))
        if len(grown) == len(group):
            return grown
        group = grown


def verify_groups(mode):
    """Assert order, geometric == algebraic, closure, unit norm, SO(3) quotient."""
    order = {"2T": 24, "2O": 48, "2I": 120}[mode]
    geom = geometric_group(mode)
    conway = conway_group(mode)

    keys = _key_set(geom)
    assert len(keys) == order, f"{mode} geometric order {len(keys)} != {order}"
    assert len(_key_set(conway)) == order, f"{mode} Conway order != {order}"
    assert keys == _key_set(conway), f"{mode}: geometric and Conway groups differ"

    assert np.allclose(np.linalg.norm(geom, axis=1), 1.0, atol=TOL), f"{mode}: non-unit quaternion"

    # SU(2) <-> quaternion round trip: unitary_to_quat is the inverse of
    # quat_to_unitary, so mapping each element out to its 2x2 unitary and back
    # must recover it -- exercises both converters and pins the convention.
    for q in geom:
        assert qkey(unitary_to_quat(quat_to_unitary(q))) == qkey(q), f"{mode}: SU(2) round trip"

    # Closure: every pairwise product (one batched einsum) lands back in the group.
    prods = np.einsum("ai,bj,ijk->abk", geom, geom, QC).reshape(-1, 4)
    assert _key_set(prods) <= keys, f"{mode}: not closed under product"

    # SO(3) quotient: one rotation per antipodal pair {q, -q}.
    reps = _dedup_proj(geom)
    assert len(reps) == order // 2, f"{mode}: {len(reps)} rotations != {order // 2}"
    for q in reps:
        R = quat_to_rotation(q)
        assert np.allclose(R.T @ R, np.eye(3), atol=TOL), f"{mode}: rotation not orthogonal"
        assert abs(np.linalg.det(R) - 1.0) < TOL, f"{mode}: rotation det != 1"

    print(f"  {mode}: order {order} (geometric = Conway), closed, "
          f"SO(3) quotient {order // 2} rotations OK")
    return geom


def _closure(gens):
    """Subgroup generated by a (G,4) quaternion array, by right-multiplication.

    Same multiply / dedup / repeat idiom as `conway_group`, kept separate so
    `conway_group` stays liftable verbatim into the walkthrough notebook -- a
    contract `_build_numpy_walkthrough.py` asserts against this file's source.
    Positive words suffice: in a finite group, g^-1 = g^(order-1).
    """
    group = _unique_rows(np.vstack([IDENT[None, :], gens]))
    while True:
        prods = np.einsum("ai,bj,ijk->abk", group, gens, QC).reshape(-1, 4)
        grown = _unique_rows(np.vstack([group, prods]))
        if len(grown) == len(group):
            return grown
        group = grown


def verify_star_copy(groups):
    """The second anchored copy of 2I: <X, Z, F, Phi*> (thesis Section 3.4).

    Anchoring a copy of 2I -- requiring it to contain the symmetrized Pauli
    group -- pins it only up to a two-fold choice: Phi vs Phi*, its entry-wise
    image under the field automorphism sqrt(5) -> -sqrt(5) (tau -> -sigma,
    sigma -> -tau). Verifies the thesis' claims about the pair:

      * <X, Z, F, Phi*> is a second order-120 group: the 600-cell built from
        the *odd* coordinate permutations -- equivalently, every vertex of the
        Phi copy's 600-cell with its second and third coordinates swapped;
      * the two copies intersect in exactly 2T, the shared 24-cell;
      * conjugation by each of the 48 Clifford elements fixes the Phi copy
        (g in 2T) or carries it onto the Phi* copy (g outside 2T);
      * the Kubischta & Teixeira variant of Phi* (a further complex
        conjugation, = Z Phi*^dag Z^dag) generates the same second copy.
    """
    phi_star = 0.5 * np.array([
        [-SIGMA - 1j * TAU, 1],
        [-1, -SIGMA + 1j * TAU],
    ], dtype=np.complex128)
    assert abs(np.linalg.det(phi_star) - 1.0) < TOL, "Phi*: det != 1"

    star = _closure(np.vstack([QUAT["X"], QUAT["Z"], QUAT["F"], unitary_to_quat(phi_star)]))
    star_keys = {qkey(q) for q in star}
    keys_2i = {qkey(q) for q in groups["2I"]}
    keys_2t = {qkey(q) for q in groups["2T"]}

    assert len(star_keys) == 120, f"star copy order {len(star_keys)} != 120"
    assert star_keys != keys_2i, "star copy coincides with the Phi copy"
    assert star_keys & keys_2i == keys_2t, "copies do not intersect in exactly 2T"

    # Geometric identity: swapping the second and third coordinates of every
    # 600-cell vertex turns its even coordinate permutations odd (and fixes the
    # 24-cell block), so the swapped vertex set must *be* the star copy.
    star_geom = geometric_group("2I")[:, [0, 2, 1, 3]]
    assert star_keys == {qkey(q) for q in star_geom}, "star copy != odd-permutation 600-cell"

    # Clifford conjugation, swept over all 48 elements of 2O.
    for g in groups["2O"]:
        gi = qconj(g)
        conj_keys = {qkey(qmul(qmul(g, q), gi)) for q in groups["2I"]}
        expected = keys_2i if qkey(g) in keys_2t else star_keys
        assert conj_keys == expected, "Clifford conjugation does not fix/swap the copies as claimed"

    # K&T's Phi* is the complex conjugate of ours, = Z Phi*^dag Z^dag (the
    # thesis footnote's identity): a different representative, the same group.
    kt = phi_star.conj()
    sz = _SU2["Z"]
    assert np.allclose(kt, sz @ phi_star.conj().T @ sz.conj().T, atol=TOL), \
        "K&T Phi* != Z Phi*^dag Z^dag"
    star_kt = _closure(np.vstack([QUAT["X"], QUAT["Z"], QUAT["F"], unitary_to_quat(kt)]))
    assert {qkey(q) for q in star_kt} == star_keys, "K&T Phi* generates a different group"

    print("  2I*: <X,Z,F,Phi*> order 120 (= odd-permutation 600-cell), meets 2I exactly in 2T;")
    print("       all 48 Clifford conjugations fix (2T) or swap (2O \\ 2T) the copies; "
          "K&T's Phi* variant generates the same copy")


# =============================================================================
# Pillar 2 -- circuit synthesis as shortest paths on a finite Cayley graph
# =============================================================================

class _Search:
    """Result of one shortest-path run: a shortest-path tree over labelled nodes.

    Costs pack (magic, depth) as magic*DEPTH_CAP + depth so a single scalar
    relaxation minimizes magic first, depth second. Parent pointers reconstruct
    the gate sequence; lengths are read straight off the reconstructed sequence.
    """

    def __init__(self, index, id0, names, key_fn, cost, pnode, pgate, use_magic):
        self.index, self.id0, self.names, self.key_fn = index, id0, names, key_fn
        self.cost, self.pnode, self.pgate, self.use_magic = cost, pnode, pgate, use_magic

    def _node(self, q):
        return self.index[self.key_fn(q)]

    def seq(self, q):
        """Reconstruct the gate sequence for q by walking parents back to identity."""
        v, out = self._node(q), []
        while v != self.id0:
            out.append(self.names[self.pgate[v]])
            v = int(self.pnode[v])
        return out[::-1]

    def depth(self, q):
        return len(self.seq(q))

    def magic(self, q):
        if self.use_magic:
            return int(round(self.cost[self._node(q)])) // DEPTH_CAP
        return seq_magic(self.seq(q))


def _build_graph(reps, key_fn, dgates):
    """Labelled Cayley graph: index, identity node, edge names, permutations.

    For each directed gate g, perm[u] is the node reached from u by right-
    multiplying by g; right-multiplication by a fixed unit quaternion is a
    bijection, so perm is a permutation and `inv` (its inverse) gives, for each
    target node, its unique predecessor under g.
    """
    index = {key_fn(q): i for i, q in enumerate(reps)}
    id0 = index[key_fn(IDENT)]
    M = len(reps)
    names, edges = [], []
    for name, gq, magic in dgates:
        perm = np.array([index[key_fn(qmul(reps[u], gq))] for u in range(M)])
        inv = np.empty(M, dtype=int)
        inv[perm] = np.arange(M)
        names.append(name)
        edges.append((inv, magic))
    return index, id0, names, edges, key_fn


def _shortest_paths(graph, use_magic):
    """Vectorized Bellman-Ford from the identity over the labelled Cayley graph.

    Each gate relaxes all nodes at once: cand[v] = cost[pred(v)] + w, where
    pred(v) = inv[v] is v's unique predecessor under that gate. Edge weights are
    1 (min depth) or magic*DEPTH_CAP + 1 (min magic, then depth). Costs are
    integer-valued and strictly decrease along parents, so this converges and
    the parent tree is cycle-free.
    """
    index, id0, names, edges, key_fn = graph
    M = len(index)
    cost = np.full(M, np.inf)
    cost[id0] = 0.0
    pnode = np.full(M, -1, dtype=int)
    pgate = np.full(M, -1, dtype=int)

    changed = True
    while changed:
        changed = False
        for gi, (inv, magic) in enumerate(edges):
            w = (magic * DEPTH_CAP + 1) if use_magic else 1
            cand = cost[inv] + w
            better = cand < cost
            if better.any():
                cost[better] = cand[better]
                pnode[better] = inv[better]
                pgate[better] = gi
                changed = True

    return _Search(index, id0, names, key_fn, cost, pnode, pgate, use_magic)


def synthesize_all(mode, elements):
    """All four synthesizers for one group, as _Search objects:
    (bfs, dij, u2, dij_u2). bfs/dij work in SU(2) (exact, keyed by qkey); the
    U(2) variants work on the projective quotient (keyed by qkey_proj)."""
    dgates = directed_gates(mode)
    su2 = _build_graph(elements, qkey, dgates)
    u2 = _build_graph(_dedup_proj(elements), qkey_proj, dgates)
    return (
        _shortest_paths(su2, use_magic=False),   # BFS: min depth, exact SU(2)
        _shortest_paths(su2, use_magic=True),    # Dijkstra: min magic then depth
        _shortest_paths(u2, use_magic=False),    # U(2) BFS: min depth mod phase
        _shortest_paths(u2, use_magic=True),     # U(2) Dijkstra: min magic mod phase
    )


def replay(seq):
    """Multiply out a gate sequence to its resulting quaternion."""
    q = IDENT.copy()
    for name in seq:
        base, dag = name.rstrip("'"), name.endswith("'")
        gq, gi, _, _ = GATES[base]
        q = qmul(q, gi if dag else gq)
    return q


def seq_magic(seq):
    return sum(GATES[name.rstrip("'")][2] for name in seq)


def u2_phase_omega_power(seq, q_target):
    """Phase phi with U(2) sequence = phi * q_target, as integer k in phi = w^k,
    w = e^{i pi/4}. Mirrors _compute_u2_phase; asserts phi is an 8th root of
    unity (self-consistency)."""
    q_su2 = replay(seq)
    if qkey(q_su2) == qkey(q_target):
        sign = 1.0
    elif qkey(q_su2) == qkey(-q_target):
        sign = -1.0
    else:
        raise AssertionError("U(2) sequence does not match target up to sign")
    acc = 1.0 + 0j
    for name in seq:
        base, dag = name.rstrip("'"), name.endswith("'")
        _, _, _, ph = GATES[base]
        acc *= np.conj(ph) if dag else ph
    phi = sign / acc
    k = round(float(np.angle(phi)) / (np.pi / 4)) % 8
    assert abs(abs(phi) - 1.0) < TOL, "U(2) phase not unit modulus"
    assert abs(phi - np.exp(1j * np.pi / 4 * k)) < TOL, "U(2) phase not an 8th root of unity"
    return int(k)


def verify_synthesis(mode, elements, synth):
    """Replay every reconstructed word and confirm it hits the target."""
    bfs, dij, u2, dij_u2 = synth
    for q in elements:
        s = bfs.seq(q)
        assert qkey(replay(s)) == qkey(q), f"{mode}: BFS replay mismatch"

        s = dij.seq(q)
        assert qkey(replay(s)) == qkey(q), f"{mode}: Dijkstra replay mismatch"
        assert dij.magic(q) == seq_magic(s), f"{mode}: Dijkstra magic mismatch"

        s = u2.seq(q)
        assert qkey_proj(replay(s)) == qkey_proj(q), f"{mode}: U(2) BFS replay mismatch"
        u2_phase_omega_power(s, q)

        s = dij_u2.seq(q)
        assert qkey_proj(replay(s)) == qkey_proj(q), f"{mode}: U(2) Dijkstra replay mismatch"
        assert dij_u2.magic(q) == seq_magic(s), f"{mode}: U(2) Dijkstra magic mismatch"
        u2_phase_omega_power(s, q)

    max_depth = max(bfs.depth(q) for q in elements)
    max_magic = max(dij.magic(q) for q in elements)
    print(f"  {mode}: {len(elements)} elements verified across 4 synthesizers "
          f"(max BFS depth {max_depth}, max SU(2) magic cost {max_magic})")


# =============================================================================
# Pillar 3 -- Platonic-solid POVMs
# =============================================================================

def build_vertices():
    """Bloch-sphere unit vectors for the five solids, orderings matching
    povm_properties.py / Figure 2.2."""
    c = 1.0 / SQRT3
    s = 1.0 / np.sqrt(2.0 + TAU)
    tau, sig = TAU, SIGMA

    tetrahedron = np.array([
        [c, c, c], [-c, -c, c], [-c, c, -c], [c, -c, -c],
    ])
    octahedron = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ], dtype=float)
    cube = np.array([
        [-c, -c, -c], [c, -c, -c], [c, c, -c], [-c, c, -c],
        [-c, -c, c], [c, -c, c], [c, c, c], [-c, c, c],
    ])
    icosahedron = np.array([
        [s * tau, s, 0], [-s * tau, s, 0], [s * tau, -s, 0], [-s * tau, -s, 0],
        [0, s * tau, s], [0, -s * tau, s], [0, s * tau, -s], [0, -s * tau, -s],
        [s, 0, s * tau], [-s, 0, s * tau], [s, 0, -s * tau], [-s, 0, -s * tau],
    ])
    dodecahedron = np.array([
        [c, c, c], [c, c, -c], [c, -c, c], [c, -c, -c],
        [-c, c, c], [-c, c, -c], [-c, -c, c], [-c, -c, -c],
        [0, c * sig, c * tau], [0, c * sig, -c * tau],
        [0, -c * sig, c * tau], [0, -c * sig, -c * tau],
        [c * sig, c * tau, 0], [c * sig, -c * tau, 0],
        [-c * sig, c * tau, 0], [-c * sig, -c * tau, 0],
        [c * tau, 0, c * sig], [c * tau, 0, -c * sig],
        [-c * tau, 0, c * sig], [-c * tau, 0, -c * sig],
    ])
    return {
        "tetrahedron": tetrahedron,
        "octahedron": octahedron,
        "cube": cube,
        "icosahedron": icosahedron,
        "dodecahedron": dodecahedron,
    }


# Design strength is a property of the symmetry group, not the solid.
SOLID_GROUP = {
    "tetrahedron": "2T", "octahedron": "2O", "cube": "2O",
    "icosahedron": "2I", "dodecahedron": "2I",
}
SOLID_STRENGTH = {
    "tetrahedron": 2, "octahedron": 3, "cube": 3,
    "icosahedron": 5, "dodecahedron": 5,
}


def povm_elements(vertices):
    """E_k = (1/V)(I + n_k . sigma), vectorized over vertices."""
    n_dot_sigma = np.einsum("vi,iab->vab", vertices, PAULI)
    return (I2 + n_dot_sigma) / len(vertices)


def is_complete(elements):
    # rtol scales with I2, so it would decide the diagonal 1e4 looser than the
    # off-diagonal -- measure the deviation (matches <= 2.2e-16, bound TOL).
    dev = np.abs(elements.sum(axis=0) - I2).max()
    return bool(dev <= TOL)


def ic_rank(elements):
    """Rank of the V x 4 Pauli-vectorized effect matrix (4 = informationally complete)."""
    a0 = (elements[:, 0, 0] + elements[:, 1, 1]) / 2
    ax = (elements[:, 0, 1] + elements[:, 1, 0]) / 2
    ay = 1j * (elements[:, 0, 1] - elements[:, 1, 0]) / 2
    az = (elements[:, 0, 0] - elements[:, 1, 1]) / 2
    return int(np.linalg.matrix_rank(np.stack([a0, ax, ay, az], axis=1), tol=TOL))


def second_moment_ok(vertices):
    """(1/V) sum_k rho_k tensor rho_k == (I_4 + SWAP)/6."""
    rho = (I2 + np.einsum("vi,iab->vab", vertices, PAULI)) / 2
    kron = np.einsum("vij,vkl->vikjl", rho, rho).reshape(len(vertices), 4, 4)
    target = (np.eye(4) + SWAP4) / 6
    # the target's entries are 1/3 and 1/6, so rtol would decide this
    # (matches <= 8.3e-17, nearest genuine miss 1.3e-2)
    dev = np.abs(kron.mean(axis=0) - target).max()
    return bool(dev <= DESIGN_TOL)


def _double_factorial(n):
    if n <= 0:
        return 1.0
    r = 1.0
    while n > 0:
        r *= n
        n -= 2
    return r


def sphere_moment(indices):
    """Average of x_{i_1}...x_{i_t} over the unit sphere."""
    a = [0, 0, 0]
    for i in indices:
        a[i] += 1
    if any(ai % 2 for ai in a):
        return 0.0
    num = (_double_factorial(a[0] - 1)
           * _double_factorial(a[1] - 1)
           * _double_factorial(a[2] - 1))
    return num / _double_factorial(sum(a) + 1)


_LETTERS = "abcdefgh"


def _discrete_moment_tensor(vertices, t):
    """Degree-t discrete moment tensor mean_v n_v (x) ... (x) n_v, one einsum."""
    subs = ",".join("v" + _LETTERS[i] for i in range(t)) + "->" + _LETTERS[:t]
    return np.einsum(subs, *([vertices] * t)) / len(vertices)


def _sphere_moment_tensor(t):
    """Degree-t sphere moment tensor (analytic constants, 3^t entries)."""
    T = np.zeros((3,) * t)
    for idx in np.ndindex(*((3,) * t)):
        T[idx] = sphere_moment(idx)
    return T


def spherical_design_strength(vertices, t_max=6):
    """Largest t for which the discrete moment tensor matches the sphere's."""
    last = 0
    for t in range(1, t_max + 1):
        # the sphere tensor's entries are 1/3, 1/5, 1/15, so rtol would decide this
        dev = np.abs(_discrete_moment_tensor(vertices, t)
                     - _sphere_moment_tensor(t)).max()
        if dev > DESIGN_TOL:
            return last
        last = t
    return last


def frame_potential(vertices, t):
    """Phi_t = (1/V^2) sum_{i,j} ((1 + n_i . n_j)/2)^t."""
    V = len(vertices)
    overlaps = (1.0 + vertices @ vertices.T) / 2.0
    return float(np.sum(overlaps ** t) / V ** 2)


def welch_bound(t):
    """Welch lower bound on Phi_t for unit vectors in C^2: 1/(t+1)."""
    return 1.0 / (t + 1)


def projective_design_strength(vertices, t_max=6):
    """Largest t for which Phi_t saturates the Welch bound through t."""
    last = 0
    for t in range(1, t_max + 1):
        # saturation <= 1.7e-16, smallest genuine shortfall 1.5e-4
        if abs(frame_potential(vertices, t) - welch_bound(t)) > DESIGN_TOL:
            return last
        last = t
    return last


def angle_multiset(vertices):
    """{ |<psi_i|psi_j>|^2 : i != j } as sorted (value, multiplicity), desc."""
    V = len(vertices)
    overlaps = (1.0 + vertices @ vertices.T) / 2.0
    iu = np.triu_indices(V, 1)
    vals, counts = np.unique(np.round(overlaps[iu], KEY_DECIMALS) + 0.0, return_counts=True)
    assert counts.sum() == V * (V - 1) // 2, "angle-multiset multiplicities do not sum to C(V,2)"
    order = np.argsort(-vals)
    return list(zip(vals[order].tolist(), counts[order].tolist()))


def verify_povms(vertices_by_solid):
    """Self-check completeness, IC, second moment and design strengths."""
    for name, verts in vertices_by_solid.items():
        V = len(verts)
        assert np.allclose(np.linalg.norm(verts, axis=1), 1.0, atol=TOL), f"{name}: non-unit vertex"
        assert np.allclose(verts.sum(axis=0), 0.0, atol=TOL), f"{name}: vertices do not sum to zero"

        E = povm_elements(verts)
        assert is_complete(E), f"{name}: POVM not complete"
        assert ic_rank(E) == 4, f"{name}: not informationally complete"
        assert second_moment_ok(verts), f"{name}: second-moment identity fails"

        t_s = spherical_design_strength(verts)
        t_p = projective_design_strength(verts)
        expected = SOLID_STRENGTH[name]
        assert t_s == t_p == expected, f"{name}: design strength {t_s}/{t_p} != {expected}"

        phis = []
        for t in (2, 3, 4, 5):
            val = frame_potential(verts, t)
            mark = "*" if abs(val - welch_bound(t)) < DESIGN_TOL else " "
            phis.append(f"P{t}={val:.4f}{mark}")
        overlaps = " ".join(f"{c}x{v:.4f}" for v, c in angle_multiset(verts))
        print(f"  {name:<12} V={V:<2} {SOLID_GROUP[name]}  t_s=t_p={expected}  "
              f"{' '.join(phis)}")
        print(f"               overlaps: {overlaps}")


# =============================================================================
# Cross-check against the symbolic-derived npz
# =============================================================================

def cross_check_npz(groups, synth, vertices_by_solid):
    """Diff numpy results against code/data/*.npz (sympy-free np.load).

    The diff's reach differs by pillar, and the section comments carry the
    detail: group unitaries, rotations and gate matrices are value-for-value
    (paired off by rounding key, elementwise at 1e-12); synthesis compares
    the per-row graph invariants (depths, magic costs), while sequences and
    phases are tie-break dependent and deliberately not compared; POVM
    vertices are diffed row-for-row -- a transcription-and-ordering check on
    top of verify_povms' invariants. A missing npz raises (np.load), so a
    partial export fails loudly instead of quietly shrinking the diff.
    Returns False without checking anything when there is no export at all
    (main() then qualifies its closing banner), True once every diff has run.
    """
    if not (DATA_DIR.exists() and any(DATA_DIR.glob("*.npz"))):
        print("  (no code/data/*.npz found -- skipping; run export_numpy.py to generate)")
        return False

    # --- Groups + synthesis invariants ---
    for mode in ("2T", "2O", "2I"):
        d = np.load(DATA_DIR / f"group_{mode}.npz", allow_pickle=True)
        U = d["unitaries"]

        # The npz is sort-keyed and numpy first-seen, so the lists agree as sets,
        # not row for row -- but set equality alone decides on the 5e-10 rounding
        # half-grid, so the rows are also paired off by key and diffed. Residual
        # 1.6e-16 over the three groups; 1e-12 keeps a 12th-decimal npz
        # hand-patch -- the drift this guard exists to catch -- inside the guard.
        by_key = {qkey(q): q for q in groups[mode]}
        npz_keys = {qkey(unitary_to_quat(u)) for u in U}
        assert npz_keys == set(by_key), f"{mode}: npz group set differs"
        for u in U:
            d_u = np.abs(u - quat_to_unitary(by_key[qkey(unitary_to_quat(u))])).max()
            assert d_u < 1e-12, f"{mode}: npz unitary differs (max |dev| = {d_u:.2e})"

        bfs, dij, u2, dij_u2 = synth[mode]
        for i in range(len(U)):
            q = unitary_to_quat(U[i])
            # graph invariants (depth / magic) must match exactly; sequences and
            # phases are tie-break dependent and deliberately not compared.
            assert bfs.depth(q) == int(d["bfs_depths"][i]), f"{mode} row {i}: BFS depth"
            assert dij.magic(q) == int(d["dij_magic_costs"][i]), f"{mode} row {i}: Dijkstra magic"
            assert dij.depth(q) == int(d["dij_depths"][i]), f"{mode} row {i}: Dijkstra depth"
            assert u2.depth(q) == int(d["u2_depths"][i]), f"{mode} row {i}: U(2) BFS depth"
            assert dij_u2.magic(q) == int(d["dij_u2_magic_costs"][i]), f"{mode} row {i}: U(2) Dijkstra magic"
            assert dij_u2.depth(q) == int(d["dij_u2_depths"][i]), f"{mode} row {i}: U(2) Dijkstra depth"
        print(f"  group_{mode}.npz: {len(U)} unitaries + depth/magic invariants match")

    # --- Rotation groups (SO(3)) ---
    for name, mode in (("T", "2T"), ("O", "2O"), ("I", "2I")):
        R = np.load(DATA_DIR / f"group_{name}.npz")["rotations"]
        assert len(R) == len(groups[mode]) // 2, f"group_{name}.npz: wrong rotation count"
        for r in R:
            assert np.allclose(r.T @ r, np.eye(3), atol=TOL), f"group_{name}.npz: not orthogonal"
            assert abs(np.linalg.det(r) - 1.0) < TOL, f"group_{name}.npz: det != 1"
        # Value-for-value diff against the numpy-derived rotations. The symbolic
        # export orders representatives by sort key and numpy by first-seen, so
        # the lists differ in order -- compare as sets of rounded 3x3 matrices.
        # (quat_to_rotation is sign-blind, so the antipodal-rep choice is moot.)
        # Set equality alone decides on the 5e-10 rounding half-grid, so the rows
        # are paired off by that same key and diffed too (residual 8.9e-16, 1e-12).
        R_numpy = np.array([quat_to_rotation(q) for q in _dedup_proj(groups[mode])])
        keys_numpy = np.round(R_numpy.reshape(len(R_numpy), 9), KEY_DECIMALS) + 0.0
        by_key = {tuple(k): r for k, r in zip(keys_numpy, R_numpy)}
        assert set(by_key) == _key_set(R.reshape(len(R), 9)), \
            f"group_{name}.npz: rotation set differs"
        for r in R:
            d_r = np.abs(r - by_key[tuple(np.round(r.reshape(9), KEY_DECIMALS) + 0.0)]).max()
            assert d_r < 1e-12, f"group_{name}.npz: rotation differs (max |dev| = {d_r:.2e})"
        print(f"  group_{name}.npz: {len(R)} SO(3) rotations match")

    # --- Gate matrices ---
    d = np.load(DATA_DIR / "gates.npz", allow_pickle=True)
    names = [str(n) for n in d["names"]]
    # Hoisted: indexing an NpzFile re-reads and re-decodes the whole member,
    # so d["su2"][i] inside the loop would unpack all six gates six times.
    su2, u2, magic_cost = d["su2"], d["u2"], d["magic_cost"]
    for i, name in enumerate(names):
        # residual 1 ulp (1.6e-16 over the six gates); 1e-12 keeps a
        # 12th-decimal npz hand-patch -- the drift this guard exists to
        # catch -- inside the guard's reach
        d_su2 = np.abs(su2[i] - _SU2[name]).max()
        d_u2 = np.abs(u2[i] - _U2[name]).max()
        assert d_su2 < 1e-12, f"gate {name}: SU(2) differs (max |dev| = {d_su2:.2e})"
        assert d_u2 < 1e-12, f"gate {name}: U(2) differs (max |dev| = {d_u2:.2e})"
        assert int(magic_cost[i]) == MAGIC[name], f"gate {name}: magic cost differs"
    print(f"  gates.npz: {len(names)} gate matrices (SU(2)+U(2)) and magic costs match")

    # --- POVMs ---
    for name in vertices_by_solid:
        d = np.load(DATA_DIR / f"povm_{name}.npz", allow_pickle=True)
        verts = d["vertices"]
        # Value-for-value diff of the numpy-native vertices against the symbolic
        # npz, row-for-row (shared Figure 2.2 ordering). Without this the checks
        # below are all rotation/relabel invariant, so they would pass for any
        # correct realization and silently miss orientation/order drift.
        # residuals 2.2e-16 on the vertices, exactly 0.0 on the elements
        d_vert = np.abs(vertices_by_solid[name] - verts).max()
        assert d_vert < 1e-12, \
            f"povm_{name}: numpy vertices differ from npz (max |dev| = {d_vert:.2e})"
        elems = d["elements"]
        d_elem = np.abs(elems - povm_elements(verts)).max()
        assert d_elem < 1e-12, \
            f"povm_{name}: elements differ (max |dev| = {d_elem:.2e})"
        assert is_complete(elems), f"povm_{name}: not complete"
        assert ic_rank(elems) == 4, f"povm_{name}: not IC"
        t_s = spherical_design_strength(verts)
        t_p = projective_design_strength(verts)
        assert t_s == t_p == SOLID_STRENGTH[name], f"povm_{name}: design strength differs"
        assert str(d["symmetry_group"]) == SOLID_GROUP[name], f"povm_{name}: symmetry group differs"
        print(f"  povm_{name}.npz: V={len(verts)} vertices + elements + IC + t_s=t_p={SOLID_STRENGTH[name]} match")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    print("Numpy-native re-derivation of the symbolic atlas")
    print("=" * 60)

    print("\nPillar 1 -- binary polyhedral groups (float identity):")
    groups = {mode: verify_groups(mode) for mode in ("2T", "2O", "2I")}
    verify_star_copy(groups)

    print("\nPillar 2 -- circuit synthesis (shortest paths on the Cayley graph):")
    synth = {}
    for mode in ("2T", "2O", "2I"):
        synth[mode] = synthesize_all(mode, groups[mode])
        verify_synthesis(mode, groups[mode], synth[mode])

    print("\nPillar 3 -- Platonic-solid POVMs (* = saturates Welch bound):")
    vertices_by_solid = build_vertices()
    verify_povms(vertices_by_solid)

    print("\nCross-check against symbolic-derived code/data/*.npz:")
    diffed = cross_check_npz(groups, synth, vertices_by_solid)

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED" if diffed else
          "ALL CHECKS PASSED (cross-check skipped -- no export to diff against)")


if __name__ == "__main__":
    main()
