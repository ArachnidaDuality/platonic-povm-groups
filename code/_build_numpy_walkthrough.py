"""Builds code/numpy_walkthrough.ipynb from clean inline source.

Run with `cd code && uv run python _build_numpy_walkthrough.py`.

This script is the source of truth for the notebook. To edit a cell, modify
the corresponding `md(r'''...''')` or `code(r'''...''')` call below, then
re-run this script to regenerate the .ipynb. Do not edit the notebook
directly -- regeneration will overwrite manual edits.

The notebook is the numpy-native companion to the two symbolic walkthroughs
(`bpg-walkthrough.ipynb`, `povm_walkthrough.ipynb`). Where those mirror the
SymPy code in `main.py` / `povm_properties.py`, this one mirrors
`numpy_atlas.py` and explains the three reframings that make the numpy route
work: the structure-constant einsum (groups), the Cayley-graph Bellman-Ford
(synthesis), and the moment-tensor comparison (designs).

Every function and class in the code cells is lifted verbatim from
`numpy_atlas.py`, so a reader can recognize them one-for-one. Here the cells
are hand-copied rather than `ast`-extracted as in the randomization and shadow
builders, so `_assert_lifts_match_module` re-derives each definition from
`inspect.getsource` at build time and fails the build on any drift, docstrings
and comments included: fix drift in the module, then re-run this script, and
never patch a lifted definition here. Definitions are all that is pinned --
the surrounding constant blocks are adapted, dropping the module's file-path
plumbing and its import-time gate-algebra assertion.

Regenerating is only half the job: the builder emits empty outputs, so the
re-execute below is mandatory to keep the notebook's stored outputs current.
Both commands run from code/:

    uv run --with jupyter --with nbconvert jupyter nbconvert \
        --to notebook --execute --inplace numpy_walkthrough.ipynb
"""

import ast
import inspect
from pathlib import Path

import nbformat

import numpy_atlas as na

nb = nbformat.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbformat.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbformat.v4.new_code_cell(text.strip("\n")))


def _assert_lifts_match_module():
    """Fail the build unless every lifted definition still matches the module.

    Walks the assembled code cells and compares each module-level def / class
    against `inspect.getsource` of its `numpy_atlas.py` counterpart, byte for
    byte. Returns the number of definitions checked.
    """
    checked = 0
    for cell in cells:
        if cell.cell_type != "code":
            continue
        for node in ast.parse(cell.source).body:
            if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            production = getattr(na, node.name, None)
            assert production is not None, \
                f"`{node.name}` is defined in a code cell but not in numpy_atlas.py"
            assert ast.get_source_segment(cell.source, node) \
                == inspect.getsource(production).rstrip("\n"), \
                f"lifted `{node.name}` has drifted from numpy_atlas.py -- re-copy it"
            checked += 1
    return checked


# =============================================================================
# Cell 1 (md) -- Title + intro
# =============================================================================
md(r'''
# Numpy-Native Atlas — Walkthrough

A companion walkthrough for `code/numpy_atlas.py`, the **numpy-only** re-derivation of the
three pillars of this thesis. The two other walkthroughs explain the *symbolic* route: how the
SymPy code in `main.py` and `povm_properties.py` does every key computation exactly over
$\mathbb{Q}(\sqrt 2, \sqrt 5, i)$. This notebook is for the reader who wants to understand the
*numpy* route instead.

The numpy code is not a line-port of the symbolic code. It re-poses each problem the way numpy
wants it posed, so it reaches the same numbers by a genuinely different road — which is exactly
what makes it worth having as an **independent oracle**. Three reframings carry the whole file,
one per pillar:

1. **Groups are `(N, 4)` float arrays.** Quaternion multiplication is bilinear, so *all*
   pairwise products are a single `einsum` against a $(4,4,4)$ structure-constant tensor.
   Generator closure becomes: multiply, deduplicate, repeat until the count stops growing.
2. **Synthesis is shortest paths on a finite Cayley graph.** A group has $\le 120$ elements;
   label them $0..N-1$ and each gate becomes a *permutation* of those labels. Minimum depth and
   minimum-magic-then-depth are then single-source shortest paths, solved by a vectorized
   Bellman–Ford relaxation — no search over float-quaternion space.
3. **A $t$-design check is a moment-tensor comparison.** The degree-$t$ discrete moment tensor
   is one `einsum`; it must equal the analytic sphere moment tensor through degree $t$.

Underpinning all three is a single trick that replaces SymPy's exact identity: **float
rounding**. Everything below is plain `float64`. The numbers printed should agree, value for
value, with the symbolic walkthroughs, the thesis tables, and the stdout of
`cd code && uv run numpy_atlas.py`.

**References:**
- Conway & Smith, *On Quaternions and Octonions* (2003)
- Kubischta & Teixeira, *A Family of Quantum Codes with Exotic Transversal Gates* (2023)
- Renes, Blume-Kohout, Scott, Caves, *Symmetric Informationally Complete Quantum Measurements* (2004)
''')


# =============================================================================
# Cell 2 (md) -- The foundational trick: float identity by rounding
# =============================================================================
md(r'''
## The foundational trick: identity by rounding

The symbolic code asks "are these two quaternions equal?" by reducing to a canonical tuple over
$\mathbb{Q}(\sqrt 2, \sqrt 5)$ — exact, but slow, and the whole reason the symbolic pipeline is
careful. The numpy code answers the same question with **rounding**: a quaternion is a length-4
`float64` array, and two are "the same" iff their components agree to 9 decimal places.

This is not sloppy — it is *provably* safe here. But the proof has **two halves**, because a
rounding key can fail in two opposite ways, and the two need different arguments.

**Collision** — two genuinely distinct elements landing on one key. Ruled out by *separation*.
Every component of every group element is an algebraic number drawn from a small, fixed alphabet
(built from $\tfrac12$, $\tfrac{1}{\sqrt2}$, $\tfrac\tau2$, $\tfrac\sigma2$, $\dots$), so two
distinct elements differ in some component by at least $0.309$ in $2I$ — $0.5$ in $2T$ and $2O$ —
which is enormous against a grid of $10^{-9}$.

**Splitting** — one element landing on *two* keys, reached by two different routes, because float
noise carried it across a rounding boundary. Ruled out by the distance from an exact component to
the nearest such boundary. That is a *different quantity* from the grid spacing, and it is the one
that actually binds. $2T$'s entries $\{0, \pm\tfrac12, \pm1\}$ are exact multiples of $10^{-9}$, so
they sit a full half-spacing $5 \times 10^{-10}$ from any boundary. $2O$ brings in
$1/\sqrt2 = 0.707106781\,|\,1865\ldots$, which clears one by $3.13 \times 10^{-10}$. And $2I$ brings
$\tau/2 = 0.809016994\,|\,3749\ldots$ and $\sigma/2 = 0.309016994\,|\,3749\ldots$ — tied, because the
two differ by exactly $\tfrac12$, itself a grid multiple — clearing a boundary by only
$1.25 \times 10^{-10}$. Meanwhile the float error accumulated over every call site in `numpy_atlas.py`
(closure products, the $\mathrm{SU}(2)$ round trip, the 48 Clifford conjugations, replaying a synthesized gate sequence)
tops out at $4.2 \times 10^{-16}$. So the tightest margin anywhere here is the icosahedral one, and
it is still a factor of $3 \times 10^5$.

Both halves are needed, and only the second is anywhere near tight. Worth saying plainly, because
the tempting one-line version — *"noise is $10^{-13}$, the grid is $10^{-9}$, so we are fine"* —
argues the first half only, and names neither of the two quantities the second half turns on.

We make the key *hashable* (a rounded tuple) so it can index a dict or live in a set — the float
analogue of the symbolic canonical form. A second variant identifies $q$ with $-q$, the
$\mathrm{SO}(3)$ / global-phase equivalence used by the $\mathrm{U}(2)$ synthesizers.
''')


# =============================================================================
# Cell 3 (code) -- Setup: imports, constants, qkey demo
# =============================================================================
code(r'''
# === Imports and constants (lifted from numpy_atlas.py) ===

from itertools import combinations, permutations, product

import numpy as np

SQRT2 = np.sqrt(2.0)
SQRT3 = np.sqrt(3.0)
SQRT5 = np.sqrt(5.0)
TAU = (1.0 + SQRT5) / 2.0       # golden ratio        (tau in Conway & Smith)
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


print(f"tau   = {TAU:.6f}")
print(f"sigma = {SIGMA:.6f}    (tau * sigma = {TAU * SIGMA:.6f}, expect 1)")
print()

# Two quaternions that differ only by float-scale noise hash identically. Shown
# on the TIGHT case: an icosian, whose tau/2 and sigma/2 sit closer to a
# rounding boundary than any other entry in the file.
q  = np.array([TAU / 2, 0.5, SIGMA / 2, 0.0])
q_noisy = q + np.array([1e-13, -2e-13, 0.0, 1e-12])
print("qkey(q)       =", qkey(q))
print("qkey(q_noisy) =", qkey(q_noisy))
print("same key:", qkey(q) == qkey(q_noisy))
print()

# How much room was there? Not the grid spacing -- the distance from each exact
# component to the nearest rounding boundary. That is the quantity that binds.
room = 0.5e-9 - np.abs(q * 1e9 - np.round(q * 1e9)) * 1e-9
for v, r in zip(q, room):
    print(f"  {v:.9f}  ->  {r:.4e} from a boundary")
print(f"tightest {room.min():.4e}, against a worst measured float error of 4.2e-16")
print()

# ...while qkey_proj additionally collapses q and -q to one representative.
print("qkey_proj(q)  == qkey_proj(-q):", qkey_proj(q) == qkey_proj(-q))
''')


# =============================================================================
# Cell 4 (md) -- Pillar 1, idea 1: a group is an (N,4) array
# =============================================================================
md(r'''
## Pillar 1 — Binary polyhedral groups

### A group is an `(N, 4)` array, and multiplication is a tensor

A unit quaternion $q = w + xi + yj + zk$ doubles as an element of $\mathrm{SU}(2)$, via
$$q \;\longleftrightarrow\; \begin{pmatrix} w + ix & y + iz \\ -y + iz & w - ix \end{pmatrix}.$$
So a binary polyhedral group — $2T$ (order 24), $2O$ (order 48), $2I$ (order 120) — is just a
set of unit quaternions, which we store as the rows of an `(N, 4)` array.

The Hamilton product $(a \cdot b)$ is **bilinear** in $a$ and $b$. That means it is completely
described by a constant rank-3 tensor $C_{ijk}$ — the *structure constants* of the quaternion
algebra in the basis $\{1, i, j, k\}$:
$$(a \cdot b)_k \;=\; \sum_{i,j} a_i \, b_j \, C_{ijk}.$$
$C$ is nothing but the $4 \times 4$ quaternion multiplication table, stacked. Once we have it,
every product is a contraction — and, crucially, *all* products of a whole group at once become
a single `einsum`.
''')


# =============================================================================
# Cell 5 (code) -- qmul / qconj / QC tensor / converters
# =============================================================================
code(r'''
# === Quaternion primitives and the structure-constant tensor ===

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


# C[i,j,k]: (a*b)_k = sum_{i,j} a_i b_j C[i,j,k]. Each entry is just the product
# of two basis quaternions, so we build it by multiplying out the 4x4 table.
QC = np.array([[qmul(np.eye(4)[i], np.eye(4)[j]) for j in range(4)] for i in range(4)])


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


# The tensor IS the multiplication table. Read off i*j = k and j*i = -k:
print("QC shape:", QC.shape)
print("i * j =", QC[1, 2], "  (the k basis vector)")
print("j * i =", QC[2, 1], "  (minus k -- quaternions don't commute)")
''')


# =============================================================================
# Cell 6 (md) -- the batched einsum
# =============================================================================
md(r'''
### All pairwise products in one `einsum`

With $C$ in hand, the product of two stacks of quaternions $A$ `(M,4)` and $B$ `(G,4)` is
$$P_{abk} \;=\; \sum_{i,j} A_{ai} \, B_{bj} \, C_{ijk},$$
i.e. `np.einsum("ai,bj,ijk->abk", A, B, QC)` — an `(M, G, 4)` array holding *every* product
$A_a \cdot B_b$ at once. This single line replaces the doubly-nested Python loop over pairs that
a naive port would write, and it is the numpy-native engine behind both group closure and the
closure check.

Deduplication is the other half. Two rows are "the same element" iff they round to the same key,
so we round, take `np.unique` along the row axis, then index back into the *original* (unrounded)
rows — representatives never drift onto the grid.
''')


# =============================================================================
# Cell 7 (code) -- _unique_rows + einsum vs loop demo
# =============================================================================
code(r'''
# === Deduplication on the rounding grid, and the einsum == loop check ===

def _unique_rows(A):
    """Unique rows of an (N,4) array by the rounding grid, first-seen order.

    The numpy-native replacement for a dict-of-tuples dedup: round to the
    identity grid, take `np.unique` along axis 0, then index back into the
    *original* (unrounded) rows so representatives never drift.
    """
    keys = np.round(A, KEY_DECIMALS) + 0.0          # +0.0 kills signed-zero rows
    _, idx = np.unique(keys, axis=0, return_index=True)
    return A[np.sort(idx)]


def _key_set(A):
    """Set of rounded-row tuples -- the float analogue of a set of group hashes."""
    return {tuple(r) for r in (np.round(A, KEY_DECIMALS) + 0.0)}


# Take a handful of unit quaternions and form all pairwise products two ways.
sample = _unique_rows(np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.5, 0.5, 0.5, 0.5],
]))

by_loop = np.array([qmul(a, b) for a in sample for b in sample])
by_einsum = np.einsum("ai,bj,ijk->abk", sample, sample, QC).reshape(-1, 4)

print("products via loop and via einsum agree:", np.allclose(by_loop, by_einsum))
print("one einsum computed", by_einsum.shape[0], "products at once")
''')


# =============================================================================
# Cell 8 (md) -- two constructions
# =============================================================================
md(r'''
### Two independent constructions

A correctness check is only as good as its independence, so we build each group **two ways** and
demand they agree:

- **Geometric.** The group elements are the vertices of a 4D regular polytope. We write those
  vertex coordinates down directly: the 24-cell for $2T$, its dual added for $2O$, and the
  600-cell's golden-ratio vertices for $2I$.
- **Algebraic (Conway & Smith).** Start from two generators and take the *closure*: repeatedly
  multiply the current set by the generators (using the batched `einsum`), deduplicate, and stop
  when the count stabilizes. This is a fixed-point iteration on the `(N,4)` array.

If the polytope geometry and the generator closure produce the same rounded-key set, both are
almost certainly right.
''')


# =============================================================================
# Cell 9 (code) -- geometric_group + conway_group + agreement
# =============================================================================
code(r'''
# === Geometric (polytope) and algebraic (Conway & Smith) constructions ===

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


GROUPS = {}
print(f"{'group':6s} {'order':>5s}   geometric == Conway?")
print("-" * 36)
for mode in ("2T", "2O", "2I"):
    geom = geometric_group(mode)
    conway = conway_group(mode)
    agree = _key_set(geom) == _key_set(conway)
    assert agree, f"{mode}: constructions disagree"
    GROUPS[mode] = geom
    print(f"{mode:6s} {len(geom):>5d}   {agree}")
''')


# =============================================================================
# Cell 10 (md) -- closure + SO(3) quotient
# =============================================================================
md(r'''
### Closure and the $\mathrm{SO}(3)$ quotient

Two structural facts confirm we really have a *group*, and both fall out of the same machinery:

- **Closure.** Every product of two elements must land back in the set. That is one batched
  `einsum` over the whole group ($N^2$ products) and a subset test on rounded keys.
- **The double cover.** $\mathrm{SU}(2) \to \mathrm{SO}(3)$ is two-to-one: $q$ and $-q$ map to the
  *same* rotation. Collapsing antipodal pairs $\{q, -q\}$ leaves exactly $N/2$ rotations — the
  ordinary polyhedral group $T$, $O$, or $I$. We map each representative through `quat_to_rotation`
  — the *physical* Bloch convention, i.e. the rotation $\rho \mapsto U_q \rho U_q^\dagger$ induces
  on Bloch vectors — and confirm it is a genuine rotation ($R^\top R = \mathbf{I}$, $\det R = 1$).
''')


# =============================================================================
# Cell 11 (code) -- closure + projective dedup + rotations
# =============================================================================
code(r'''
# === Closure under multiplication, and the SO(3) quotient ===

def _dedup_proj(A):
    """One representative row per projective class {q, -q}, first-seen order."""
    seen, out = set(), []
    for q in A:
        k = qkey_proj(q)
        if k not in seen:
            seen.add(k)
            out.append(q)
    return np.array(out)


print(f"{'group':6s} {'|G|':>4s}  closed?  {'rotations':>9s}  all SO(3)?")
print("-" * 45)
for mode in ("2T", "2O", "2I"):
    G = GROUPS[mode]

    # Closure: every pairwise product (one batched einsum) is already in G.
    prods = np.einsum("ai,bj,ijk->abk", G, G, QC).reshape(-1, 4)
    closed = _key_set(prods) <= _key_set(G)

    # SO(3) quotient: one rotation per antipodal pair.
    reps = _dedup_proj(G)
    ok = all(
        np.allclose(quat_to_rotation(q).T @ quat_to_rotation(q), np.eye(3), atol=TOL)
        and abs(np.linalg.det(quat_to_rotation(q)) - 1.0) < TOL
        for q in reps
    )
    print(f"{mode:6s} {len(G):>4d}  {str(closed):>6s}   {len(reps):>9d}  {ok}")
''')


# =============================================================================
# Cell 12 (md) -- Pillar 2 intro
# =============================================================================
md(r'''
## Pillar 2 — Circuit synthesis as shortest paths

Once a group is built it is a finite, labelled set of $\le 120$ nodes. The synthesis question —
*what is the shortest gate sequence producing each element?* — is then a **graph** question, not a
search over the continuum of float quaternions.

The key observation: right-multiplication by a fixed gate $g$ is a **bijection** of the group
(if $q_1 g = q_2 g$ then $q_1 = q_2$). So each directed gate is a *permutation* of the node
labels $0..N-1$. The Cayley graph has the identity as source, and every node has exactly one
predecessor per gate. Shortest paths from the identity give the optimal words, and because the
predecessor relation is a clean permutation we can reconstruct each word by walking parent
pointers back to the identity.

First we need the gates themselves, in both $\mathrm{SU}(2)$ and $\mathrm{U}(2)$ form, with the
phase relating them and the magic cost (only $\Phi$ is non-Clifford).
''')


# =============================================================================
# Cell 13 (code) -- gate definitions + directed_gates
# =============================================================================
code(r'''
# === Gate definitions (float SU(2) and U(2) forms) ===
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
QUAT = {g: unitary_to_quat(_SU2[g]) for g in GATE_NAMES}

# name -> (quaternion, inverse quaternion, magic cost, U2->SU2 phase)
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


print("Gate set per group (each gate also contributes its adjoint as an edge):")
for mode in ("2T", "2O", "2I"):
    print(f"  {mode}: {[n for n, _, _ in directed_gates(mode)]}")
''')


# =============================================================================
# Cell 14 (md) -- the labelled graph
# =============================================================================
md(r'''
### The Cayley graph: each gate is a permutation

We label the group $0..N-1$ via a dict from rounded key to index. For a directed gate $g$, the
array `perm[u] = index[ round(reps[u] * g) ]` records *where each node goes* under right-multiply
by $g$. Because that map is a bijection, `perm` is a genuine permutation, and its **inverse** answers
the question shortest paths actually needs: for each *target* node $v$, who is its unique
*predecessor* under $g$? That inverse permutation `inv` is what we relax over.
''')


# =============================================================================
# Cell 15 (code) -- _build_graph + permutation demo
# =============================================================================
code(r'''
# === Build the labelled Cayley graph ===

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


# Build the SU(2) Cayley graph of 2T and look at the X edge.
graph_2T = _build_graph(GROUPS["2T"], qkey, directed_gates("2T"))
index, id0, names, edges, key_fn = graph_2T
x_perm_inv, x_magic = edges[names.index("X")]

print(f"2T has {len(index)} nodes; identity is node {id0}")
print(f"the X edge's predecessor map is a permutation: "
      f"{sorted(x_perm_inv.tolist()) == list(range(len(index)))}")
print(f"X has magic cost {x_magic}")
''')


# =============================================================================
# Cell 16 (md) -- Bellman-Ford
# =============================================================================
md(r'''
### Vectorized Bellman–Ford, with magic and depth in one number

Now we relax. Each gate updates *all* nodes simultaneously:
$$\texttt{cand}[v] \;=\; \texttt{cost}[\,\texttt{pred}(v)\,] + w,$$
where $\texttt{pred}(v) = \texttt{inv}[v]$ is $v$'s unique predecessor under that gate. We sweep
all gates until no cost improves — vectorized Bellman–Ford, no per-node priority queue.

The trick for handling *two* objectives at once: pack the cost as
$$\texttt{cost} \;=\; \texttt{magic} \times \texttt{DEPTH\_CAP} + \texttt{depth}.$$
With `DEPTH_CAP` larger than any achievable depth, a single scalar comparison minimizes **magic
first, depth second** — the "cheapest fault-tolerant, then shortest" objective — with no special
casing. Setting every edge weight to $1$ instead gives plain minimum depth (BFS). Costs are
integer-valued and strictly decrease along parent pointers, so the relaxation converges and the
parent tree is acyclic.

The `_Search` object below stores that shortest-path tree and reconstructs a gate sequence by
walking parents from a target back to the identity.
''')


# =============================================================================
# Cell 17 (code) -- _Search, _shortest_paths, replay; demo on 2T
# =============================================================================
code(r'''
# === Shortest paths from the identity, and word reconstruction ===

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


# Minimum-depth synthesis on 2T, then reconstruct + replay a few words.
bfs_2T = _shortest_paths(graph_2T, use_magic=False)
print("element (w,x,y,z)            depth  word            replay ok")
print("-" * 62)
for q in GROUPS["2T"][:6]:
    word = bfs_2T.seq(q)
    ok = qkey(replay(word)) == qkey(q)
    shown_q = np.round(q, 3) + 0.0           # + 0.0 prints -0.0 as 0.0
    print(f"{shown_q!s:28s}  {len(word):>3d}   {str(word):16s}  {ok}")
''')


# =============================================================================
# Cell 18 (md) -- the four synthesizers
# =============================================================================
md(r'''
### The four synthesizers, and where depth and magic part ways

The same machinery yields four synthesizers by varying two switches:

- **$\mathrm{SU}(2)$ vs $\mathrm{U}(2)$:** exact equality (key `qkey`) or equality up to global
  phase (projective key `qkey_proj`, which collapses $q \sim -q$). The $\mathrm{U}(2)$ graph runs
  on the $N/2$-node projective quotient.
- **depth vs magic:** edge weight $1$ (minimum depth) or $\texttt{magic}\cdot\texttt{DEPTH\_CAP}+1$
  (minimum magic, then depth).

For $2T$ and $2O$ the gate sets are entirely Clifford, so magic is always zero and the two
objectives coincide. $2I$ is where they part: its gate set includes the non-Clifford $\Phi$, and
minimizing $\Phi$-count can force a *longer* circuit. The cell below finds exactly the group
elements where the minimum-depth word and the minimum-magic word disagree.
''')


# =============================================================================
# Cell 19 (code) -- synthesize_all + BFS-vs-Dijkstra divergence on 2I
# =============================================================================
code(r'''
# === All four synthesizers; the depth/magic tradeoff on 2I ===

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


SYNTH = {mode: synthesize_all(mode, GROUPS[mode]) for mode in ("2T", "2O", "2I")}

bfs, dij, _, _ = SYNTH["2I"]
print("2I elements where minimum-depth and minimum-magic words disagree:")
print(f"  {'min-depth (BFS)':24s}  magic depth   {'min-magic (Dijkstra)':28s}  magic depth")
print("  " + "-" * 86)
shown = 0
for q in GROUPS["2I"]:
    bw, dw = bfs.seq(q), dij.seq(q)
    if (len(bw), seq_magic(bw)) != (len(dw), dij.magic(q)):
        print(f"  {str(bw):24s}  {seq_magic(bw):>5d} {len(bw):>5d}   "
              f"{str(dw):28s}  {dij.magic(q):>5d} {len(dw):>5d}")
        shown += 1
        if shown == 6:
            break
print()
print("BFS trades magic for depth (two Phi's, depth 2); Dijkstra trades depth for")
print("magic (one Phi, depth 4). Both replay to the same SU(2) element.")
''')


# =============================================================================
# Cell 20 (md) -- Pillar 3 intro
# =============================================================================
md(r'''
## Pillar 3 — Platonic-solid POVMs

Each Platonic solid, inscribed on the Bloch sphere as unit vectors $\{\hat n_k\}_{k=1}^V$, defines
a POVM with rank-1 effects
$$E_k \;=\; \frac{1}{V}\bigl(\mathbf{I} + \hat n_k \cdot \vec\sigma\bigr).$$
Two basic properties are quick linear-algebra facts in numpy:

- **Completeness** $\sum_k E_k = \mathbf{I}$ — the linear-in-Bloch terms cancel because the
  vertices sum to zero.
- **Informational completeness** — stack each effect's Pauli coordinates $(a_0, a_x, a_y, a_z)$
  into a $V \times 4$ matrix; the POVM is IC iff that matrix has rank $4$ (the effects span the
  real space of Hermitian $2\times 2$ operators).

The vertex orderings match `code/povm_properties.py` and Figure 2.2 of the thesis.
''')


# =============================================================================
# Cell 21 (code) -- build_vertices + povm_elements + completeness/IC
# =============================================================================
code(r'''
# === Vertices, POVM elements, completeness, informational completeness ===

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


VERTICES = build_vertices()
print(f"{'solid':13s} {'V':>3s}  complete?  IC rank")
print("-" * 35)
for name, verts in VERTICES.items():
    E = povm_elements(verts)
    print(f"{name:13s} {len(verts):>3d}  {str(is_complete(E)):>8s}   {ic_rank(E)}/4")
''')


# =============================================================================
# Cell 22 (md) -- moment tensors
# =============================================================================
md(r'''
### A $t$-design check is a moment-tensor comparison

The interesting property is the **design strength**. A vertex set is a spherical $t$-design when
its discrete average reproduces the uniform sphere average for every polynomial of degree $\le t$.
By linearity that reduces to matching *moment tensors*: for each degree $t$,
$$\underbrace{\frac{1}{V}\sum_v \hat n_v^{\otimes t}}_{\text{discrete}}
  \;\stackrel{?}{=}\;
  \underbrace{\frac{1}{4\pi}\int_{S^2} \hat n^{\otimes t}\, \mathrm{d}\sigma}_{\text{analytic}}.$$

In numpy the left side is **one `einsum`**: with subscripts `va,vb,...->ab...`, contracting $t$
copies of the `(V,3)` vertex array over the shared $v$ axis builds the whole degree-$t$ tensor at
once. The right side has the closed form (zero for any odd exponent, a ratio of double factorials
otherwise). The design strength is the largest $t$ for which the two tensors agree.

A second, equivalent lens is the **frame potential** $\Phi_t = \tfrac{1}{V^2}\sum_{i,j}
\bigl(\tfrac{1 + \hat n_i\cdot\hat n_j}{2}\bigr)^t$, which a projective $t$-design saturates against
the Welch bound $1/(t+1)$. We report both; they agree for every solid.
''')


# =============================================================================
# Cell 23 (code) -- moment tensors + design strengths
# =============================================================================
code(r'''
# === Discrete vs analytic moment tensors; spherical and projective strengths ===

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


# Show the degree-2 moment tensor matching for the octahedron, then the table.
oct_disc = _discrete_moment_tensor(VERTICES["octahedron"], 2)
print("octahedron degree-2 moment tensor (discrete):")
print(np.round(oct_disc, 4))
print("sphere degree-2 moment tensor (analytic, (1/3) I):")
print(np.round(_sphere_moment_tensor(2), 4))
print()

print(f"{'solid':13s} {'V':>3s}  {'t_s':>3s} {'t_p':>3s}   Phi_2..Phi_5 (* = saturates Welch)")
print("-" * 64)
for name, verts in VERTICES.items():
    t_s = spherical_design_strength(verts)
    t_p = projective_design_strength(verts)
    phis = []
    for t in (2, 3, 4, 5):
        val = frame_potential(verts, t)
        mark = "*" if abs(val - welch_bound(t)) < DESIGN_TOL else " "
        phis.append(f"{val:.4f}{mark}")
    print(f"{name:13s} {len(verts):>3d}  {t_s:>3d} {t_p:>3d}   {' '.join(phis)}")
''')


# =============================================================================
# Cell 24 (md) -- cross-check against the symbolic atlas
# =============================================================================
md(r'''
## Cross-check against the symbolic atlas

Everything above was rebuilt by hand in this notebook. Its real payoff, though, is as an
*independent oracle*: the numbers must match what the **symbolic** pipeline computed exactly over
$\mathbb{Q}(\sqrt 2, \sqrt 5, i)$. The symbolic results are exported to `code/data/*.npz` by
`export_numpy.py`, and `numpy_atlas.py` diffs against them.

Rather than re-implement that plumbing, we import the production module, confirm our hand-built
primitives reproduce its numbers exactly (an anti-drift guard), and then hand our hand-built groups,
synthesizers, and vertices to its `cross_check_npz`. Depth and magic are exact graph invariants
and are compared value-for-value; concrete gate sequences and phases are tie-break dependent and are
deliberately *not* compared.

(This section needs the export: run `cd code && uv run export_numpy.py` first. `cross_check_npz`
reports whether it found one, and the cell below stops on a *no* rather than passing on a diff that
never ran.)
''')


# =============================================================================
# Cell 25 (code) -- run cross_check_npz on what we built
# =============================================================================
code(r'''
# === Diff our hand-built results against the symbolic-derived npz ===

import numpy_atlas as na

# Anti-drift guard: the code cells above are lifted from numpy_atlas.py, so the
# primitives must agree exactly with the production module's -- same arithmetic
# run twice, so the only honest residual is 0.0, not a tolerance.
d_qc = np.abs(QC - na.QC).max()
assert d_qc == 0.0, f"structure-constant tensor drifted (max |dev| = {d_qc:.2e})"

# An emptied GATE_NAMES or VERTICES is itself drift, and would otherwise reach
# the asserts below as a vacuous max over nothing.
assert GATE_NAMES, "no gates to diff -- GATE_NAMES is empty"
assert VERTICES, "no solids to diff -- VERTICES is empty"

d_su2 = max((np.abs(_SU2[g] - na._SU2[g]).max() for g in GATE_NAMES), default=0.0)
assert d_su2 == 0.0, f"SU(2) gates drifted (max |dev| = {d_su2:.2e})"
na_vertices = na.build_vertices()
d_vert = max((np.abs(VERTICES[s] - na_vertices[s]).max() for s in VERTICES), default=0.0)
assert d_vert == 0.0, f"vertices drifted (max |dev| = {d_vert:.2e})"

# Hand the production cross-checker exactly the objects we built in this notebook.
# Its verdict is a return value, not an exception: handed no export it prints a
# skip and returns False, so without this assert a missing export would read as
# a silent pass -- and this cell is the only claim the notebook makes to being
# an oracle.
assert na.cross_check_npz(GROUPS, SYNTH, VERTICES), \
    "nothing in code/data to diff against -- run `uv run export_numpy.py` first"
''')


# =============================================================================
# Cell 26 (md) -- closing notes
# =============================================================================
md(r'''
## Closing notes

**The three reframings, recapped.** Each pillar became a different piece of numpy:

| Pillar | Symbolic question | Numpy reframing |
|---|---|---|
| Groups | exact quaternion equality + closure | `(N,4)` array; products = one `einsum` against the structure-constant tensor $C$; closure = multiply / dedup / repeat |
| Synthesis | search for optimal gate sequences | shortest paths on a finite Cayley graph; each gate = a permutation; vectorized Bellman–Ford with `(magic, depth)` packed into one scalar |
| POVMs | symbolic $t$-design certification | moment-tensor comparison via one `einsum`; closed-form sphere moments; Welch-bound saturation |

All of it rests on the **rounding grid**: exact algebraic identity is replaced by agreement to 9
decimals. Safe on both counts — distinct elements differ in some component by $\ge 0.309$, so nothing
collides; and no component sits closer than $1.25 \times 10^{-10}$ to a rounding boundary against a
worst float error of $4.2 \times 10^{-16}$, so nothing splits.

**What this buys us.** Because the numpy route poses the problems so differently from the SymPy
route, agreement between them is a strong, genuinely independent correctness signal. The
cross-check above confirms that the floats and the exact symbolic computation land on the same
numbers — for the groups, the synthesizers (depth and magic), and the POVM designs.

To run the production script end-to-end (build, self-verify, and diff against the symbolic npz):

```
cd code && uv run numpy_atlas.py
```
''')


# =============================================================================
# Assemble and write
# =============================================================================
n_lifted = _assert_lifts_match_module()

nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "pygments_lexer": "ipython3",
}

out = Path(__file__).parent / "numpy_walkthrough.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Wrote {out} ({len(cells)} cells, {n_lifted} lifted definitions verified)")
