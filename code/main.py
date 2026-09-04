"""
Binary Polyhedral Groups Atlas Generator

Generates tables for the binary tetrahedral (2T), binary octahedral (2O),
and binary icosahedral (2I) groups, including:
  - Unit quaternion representation
  - 2×2 unitary matrix (SU(2))
  - Optimal gate sequence in symmetrized (SU(2)) gates
  - Corresponding gate sequence in standard U(2) gates, matched only up to
    global phase, i.e. in PU(2) = U(2)/U(1) ≅ SU(2)/{±1} ≅ SO(3)
  - Magic cost and depth

Layout — every part below carries its own docstring; this is the index.
  Groups: `geometric_group` (polytope vertices) and `conway_group` (the Conway
    & Smith generators, closed under multiplication), reconciled by
    `verify_group`.
  Synthesis: a 2×2 grid, all four run on every group by `generate_group_data` —
    minimum depth (`synthesize_bfs`, `synthesize_u2`) and minimum magic cost
    (`synthesize_dijkstra`, `synthesize_dijkstra_u2`), the first of each pair
    matching in SU(2), the second up to global phase via `proj_hash`.
    Generators per group, adjoints included (the `GATES_*` lists): 2T = X, Z, F;
    2O = X, Z, F, H, S; 2I = X, Z, F, Φ — not nested, and Φ alone costs magic.
  Checks: `verify_synthesis` (every row reached, every word replays to its
    target, and every stored phase carries the word's standard-gate reading
    there), `verify_optimality` (the relations the grid forces between its
    four columns), `verify_atlas_layout` (the data claims Appendix A's
    preamble makes about the printed table), `verify_atlas_rotations` (the
    R_n̂(θ) column against U_q's conjugation action on the Paulis),
    `verify_differing_rows` (what Section 4.1.2 prints about the 17 rows of
    2I where BFS and Dijkstra differ, `table_b1_tex`'s rendering of them,
    that each row's `#` selects the same element in Table A.3 and links to
    it, and that the atlas marks exactly those rows), `verify_differing_rows_tex` (the written differing_rows.tex read
    back), `verify_depth_concentration` (the six depth counts behind
    Appendix B's lede, asserted here because the lede prints none of them), `verify_sample_row` (Appendix A's hand-typed sample row, parsed
    out of the thesis, cell for cell), `verify_atlas_nesting` (Table A.3
    opens with Table A.1) and `verify_atlas_tex` (the written atlas.tex read
    back, every cell and the skeleton around them against the data, and the
    B marks against Table B.1's rows and the preamble's legend, before it is
    put in place).
  Output: `atlas.tex` (Appendix A) prints one row per group element, the two
    rows of an antipodal pair (q, -q) kept together: each row's own min-magic
    SU(2) word (Dijkstra) with depth, Φ count and phase -- for 2T and 2O
    also its min-depth word, the two coinciding on every row, and the block's
    title says so -- and, once per pair, the Bloch rotation R_n̂(θ) and ±q,
    ±U_q with the common scalar out front. 2I's min-depth (BFS) words print
    nowhere in it: `atlas.txt` is their home, `differing_rows.tex`
    (Appendix B's Table B.1) carries the 17 rows where they differ from the
    printed ones -- each marked in the atlas by a superscript B left of its
    `#`, a link to that table, and anchored for the link back
    (`_atlas_key_cell`) -- and the preamble's sample row reprints one pair
    (`verify_sample_row`). The first row of a pair is
    its BFS-shallower member (`_pair_antipodal_rows`; the member Appendix F.3.4's gate-noise study runs) and
    is the projective answer: it minimises depth and (Φ, depth) over the
    pair, so the PU(2) runs print nowhere -- `verify_atlas_layout` asserts
    that they return its words. Pairs are ordered magic-first (`_atlas_key`).
    `atlas.txt` keeps the full 2×2 grid. `differing_rows.tex` (Appendix B) is
    the float around `table_b1_tex`: the 17 rows of 2I where the two SU(2)
    synthesizers disagree, in atlas.tex's own format and off its own emitters
    (`_atlas_colspec`, `_atlas_head`, `_word_cells`) so that the two tables
    read alike -- both word blocks in full, depth, Φ count, word and phase,
    keyed by `#`, the element's row of Table A.3, and closed by the element's
    own signed quaternion. Phase is per word there and the two blocks
    disagree on it on all 17 rows, which is why Appendix A's φ cannot be
    borrowed for a BFS word. Rows are ordered by BFS Φ count, not by `#`, so
    the caption's split between the rows Dijkstra improves and the rows it
    only rearranges reads down the page, then by `#` inside a tier, so a key
    can be found by eye. All three files are written beside this file.

References:
  Conway & Smith, "On Quaternions and Octonions" (2003)

Quaternion identity is decided in ℚ(√2, √5) via a rational-coefficient tuple
in the fixed basis {1, √2, √5, √10} (see `_to_basis`). `simplify` mostly keeps
intermediate expression trees small, but it also decides at two sites — the
`_PHASE_F` zero test and `_phase_to_omega_power`'s integrality check — each
loud on failure (an error or a reported mismatch), never silently wrong.
"""

import functools
import heapq
import re
from itertools import combinations, product
from pathlib import Path

from sympy import (
    I as symI,
    Matrix,
    Poly,
    Rational,
    Symbol,
    default_sort_key,
    expand,
    latex,
    radsimp,
    simplify,
    sqrt,
    sympify,
)

# =============================================================================
# Constants
# =============================================================================

SQRT2 = sqrt(2)
SQRT5 = sqrt(5)
SQRT10 = sqrt(10)
TAU = (1 + SQRT5) / 2      # golden ratio (τ in Conway & Smith)
SIGMA = (SQRT5 - 1) / 2    # inverse golden ratio (σ = τ⁻¹)

# Display symbols for golden ratio substitution
_tau_sym = Symbol('tau')       # latex() renders as \tau
_sig_sym = Symbol('sigma')     # latex() renders as \sigma
_tau_pretty = Symbol('τ')      # str() renders as τ
_sig_pretty = Symbol('σ')      # str() renders as σ

# Indeterminates for the basis-reduction polynomial ring (see _to_basis).
_S2 = Symbol('_basis_s2', positive=True)
_S5 = Symbol('_basis_s5', positive=True)


def _to_basis(expr):
    """Express an element of ℚ(√2, √5) as a 4-tuple of Rationals.

    Returns ``(a, b, c, d)`` such that ``expr = a + b·√2 + c·√5 + d·√10``.
    Mathematically a true canonical form: any two SymPy expressions that
    denote the same algebraic number in ℚ(√2, √5) produce the same tuple,
    so tuple equality and tuple hashing are exact identity tests.

    The reduction works by mapping √2, √5, √10 onto polynomial indeterminates,
    expanding, and folding back the relations √2² = 2 and √5² = 5.
    """
    e = sympify(expr)
    e = radsimp(e)
    e = e.xreplace({SQRT2: _S2, SQRT5: _S5, SQRT10: _S2 * _S5})
    e = expand(e)
    poly = Poly(e, _S2, _S5, domain='QQ')

    coeffs = [Rational(0)] * 4   # [1, √2, √5, √10]
    for (i, j), coef in poly.as_dict().items():
        reduced = Rational(coef) * Rational(2) ** (i // 2) * Rational(5) ** (j // 2)
        idx = (i % 2) + 2 * (j % 2)
        coeffs[idx] += reduced
    return tuple(coeffs)


def _is_zero(expr):
    """Exact zero test for an element of ℚ(√2, √5, i).

    Real and imaginary parts each lie in ℚ(√2, √5), where `_to_basis` is a
    true canonical form — so this decides zero rather than searching for a
    simplification. Raises on an argument outside the field.
    """
    real, imag = expand(expr).as_real_imag()
    return all(c == 0 for c in _to_basis(real) + _to_basis(imag))


def _golden_sub(expr, tau, sig):
    """Express a value in terms of τ and σ if it contains √5.

    Decomposes expr = a + b√5 via `_to_basis`, then rewrites using
    √5 = 2τ − 1 = 2σ + 1. Prefers whichever form eliminates the rational
    constant. Leaves expressions outside ℚ(√5) untouched.
    """
    e = simplify(expr)
    a, b_s2, c_s5, d_s10 = _to_basis(e)
    if c_s5 == 0:
        return e
    if b_s2 != 0 or d_s10 != 0:
        return e  # expression has √2 or √10 terms; leave unchanged
    # a + c√5 = (a − c) + 2c·τ  [since √5 = 2τ − 1]
    # a + c√5 = (a + c) + 2c·σ  [since √5 = 2σ + 1]
    if a - c_s5 == 0:
        return 2 * c_s5 * tau
    if a + c_s5 == 0:
        return 2 * c_s5 * sig
    return (a - c_s5) + 2 * c_s5 * tau


def _golden_sub_complex(expr, tau, sig):
    """Apply golden ratio substitution to a complex expression."""
    real, imag = expr.as_real_imag()
    re_sub = _golden_sub(real, tau, sig)
    im_sub = _golden_sub(imag, tau, sig)
    if im_sub == 0:
        return re_sub
    if re_sub == 0:
        return symI * im_sub
    return re_sub + symI * im_sub


# --- Atlas typesetting: common scalar out front, entries in {0, ±1, ±τ, ±σ} ---

@functools.lru_cache(maxsize=None)
def _atlas_unit(val):
    """Classify a scaled component as one of 0, ±1, ±τ, ±σ.

    Returns (sign, name) with name in {"0", "1", "tau", "sigma"}, or None if
    the value is none of these (e.g. the scale factor tried was wrong).
    Memoized: the atlas layer asks this a few thousand times (every component
    at every candidate scale, for the scale search, the quaternion, the
    unitary and the axis) about 17 distinct values, and `_golden_sub` runs a
    simplify pass on each call -- about four seconds of the run otherwise.
    """
    e = _golden_sub(val, _tau_sym, _sig_sym)
    if e == 0:
        return (1, "0")
    for name, unit in (("1", 1), ("tau", _tau_sym), ("sigma", _sig_sym)):
        if e == unit:
            return (1, name)
        if e == -unit:
            return (-1, name)
    return None


_ATLAS_UNIT_TEX = {"0": "0", "1": "1", "tau": r"\tau", "sigma": r"\sigma"}
_ATLAS_UNIT_SYM = {"0": 0, "1": 1, "tau": _tau_sym, "sigma": _sig_sym}


def _atlas_unit_tex(sign, name):
    """One component at unit scale; negatives braced as the thesis braces them."""
    body = _ATLAS_UNIT_TEX[name]
    return body if sign > 0 or name == "0" else "{-" + body + "}"


def _atlas_complex_tex(val):
    """One unitary entry at unit scale, printed as re ± i·im, real part first.

    SymPy's own ordering puts the imaginary term first for golden entries
    (i σ + τ); the atlas wants τ + iσ, and 1 − i, and a bare i.
    """
    real, imag = val.as_real_imag()
    rs, rn = _atlas_unit(real)
    is_, in_ = _atlas_unit(imag)
    re_tex = "" if rn == "0" else (("" if rs > 0 else "-") + _ATLAS_UNIT_TEX[rn])
    if in_ == "0":
        return re_tex or "0"
    im_body = "i" if in_ == "1" else "i" + _ATLAS_UNIT_TEX[in_]
    if not re_tex:
        return ("" if is_ > 0 else "-") + im_body
    return re_tex + (" + " if is_ > 0 else " - ") + im_body


def _latex_scale_prefix(c):
    """The factor in front: nothing for 1, \\tfrac{1}{2}, or \\tfrac{1}{\\sqrt{2}}."""
    if c == 1:
        return ""
    if c == Rational(1, 2):
        return r"\tfrac{1}{2}"
    if simplify(c - 1 / sqrt(2)) == 0:
        return r"\tfrac{1}{\sqrt{2}}"
    raise ValueError(f"no typesetting for scale factor {c}")

# =============================================================================
# Symbolic Quaternion
# =============================================================================

class Quaternion:
    """Immutable symbolic *unit* quaternion q = w + xi + yj + zk.

    Identity (hashing/equality) is decided by the rational-basis tuple of the
    four components — see `_to_basis`. The SymPy expression form is kept for
    the math role (multiplication intermediates) and for display.
    """

    __slots__ = ("w", "x", "y", "z", "_hash", "_tuple", "_basis", "_unitary", "_scale")

    def __init__(self, w, x, y, z):
        self.w = simplify(w)
        self.x = simplify(x)
        self.y = simplify(y)
        self.z = simplify(z)
        self._hash = None
        self._tuple = None
        self._basis = None
        self._unitary = None
        self._scale = None

    def __mul__(self, other):
        return Quaternion(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )

    def conjugate(self):
        """For unit quaternions - elements of SU(2) - the conjugate is the inverse."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def to_unitary(self):
        """Return 2×2 SU(2) matrix: [[a, b], [-b*, a*]] where a = w+ix, b = y+iz."""
        if self._unitary is None:
            a = self.w + symI * self.x
            b = self.y + symI * self.z
            self._unitary = Matrix([
                [a, b],
                [-b.conjugate(), a.conjugate()],
            ])
        return self._unitary

    def to_tuple(self):
        """SymPy-expression view of (w, x, y, z) — display / sorting only."""
        if self._tuple is None:
            self._tuple = (self.w, self.x, self.y, self.z)
        return self._tuple

    def basis_tuple(self):
        """Canonical identity: a tuple of four rational-coefficient 4-tuples in
        the basis {1, √2, √5, √10} of ℚ(√2, √5).
        """
        if self._basis is None:
            self._basis = (
                _to_basis(self.w),
                _to_basis(self.x),
                _to_basis(self.y),
                _to_basis(self.z),
            )
        return self._basis

    def neg(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def proj_hash(self):
        """Hash that identifies q and -q (SO(3) / projective equivalence).

        Picks the lexicographic min of the basis tuple and its negation so
        the result is deterministic across runs (independent of Python's
        per-process hash randomization).
        """
        bt = self.basis_tuple()
        neg_bt = tuple(tuple(-x for x in c) for c in bt)
        return hash(min(bt, neg_bt))

    def __eq__(self, other):
        return self.basis_tuple() == other.basis_tuple()

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.basis_tuple())
        return self._hash

    def __lt__(self, other):
        return default_sort_key(self.to_tuple()) < default_sort_key(other.to_tuple())

    def __repr__(self):
        return self.pretty()

    def pretty(self):
        def fmt(val):
            val = _golden_sub(val, _tau_pretty, _sig_pretty)
            return str(val).replace("sqrt(2)", "√2").replace(" ", "")
        return f"[{fmt(self.w)}, {fmt(self.x)}, {fmt(self.y)}, {fmt(self.z)}]"

    def pretty_matrix(self):
        U = self.to_unitary()
        def fmt(val):
            val = _golden_sub_complex(val, _tau_pretty, _sig_pretty)
            return str(val).replace("sqrt(2)", "√2").replace("I", "i").replace(" ", "")
        return f"[[{fmt(U[0,0])}, {fmt(U[0,1])}], [{fmt(U[1,0])}, {fmt(U[1,1])}]]"

    def latex_scale(self):
        """The scalar the atlas pulls out in front of this element.

        Every component of every element of 2T, 2O, 2I lies in
        c · {0, ±1, ±τ, ±σ} for c = 1 (the Paulis and ±1), 1/2 (the rest of
        the 24-cell and all of 2I's golden part), or 1/√2 (2O's outer 24).
        Returns that c as a SymPy number, or None if no candidate works --
        the caller then prints the raw components, so a change of group or
        convention degrades the typesetting, never the numbers. The same c
        serves the quaternion and the unitary, whose entries are w ± ix and
        ±y + iz at the same scale.
        """
        if self._scale is None:
            self._scale = False
            for c in (Rational(1), Rational(1, 2), 1 / sqrt(2)):
                if all(_atlas_unit(v / c) is not None for v in (self.w, self.x, self.y, self.z)):
                    self._scale = c
                    break
        return None if self._scale is False else self._scale

    def latex_quat(self):
        """LaTeX-formatted quaternion, common scalar out front."""
        c = self.latex_scale()
        if c is None:
            parts = [latex(_golden_sub(simplify(v), _tau_sym, _sig_sym)) for v in (self.w, self.x, self.y, self.z)]
            return "(" + ",\\, ".join(parts) + ")"
        parts = [_atlas_unit_tex(*_atlas_unit(v / c)) for v in (self.w, self.x, self.y, self.z)]
        return _latex_scale_prefix(c) + "(" + ",\\, ".join(parts) + ")"

    def latex_matrix(self):
        """LaTeX-formatted 2×2 unitary, the quaternion's scalar out front."""
        U = self.to_unitary()
        entries = [U[0, 0], U[0, 1], U[1, 0], U[1, 1]]
        c = self.latex_scale()
        if c is None:
            a, b, cc, d = [latex(_golden_sub_complex(simplify(v), _tau_sym, _sig_sym)) for v in entries]
            return rf"\begin{{psmallmatrix}} {a} & {b} \\ {cc} & {d} \end{{psmallmatrix}}"
        a, b, cc, d = [_atlas_complex_tex(v / c) for v in entries]
        return _latex_scale_prefix(c) + rf"\begin{{psmallmatrix}} {a} & {b} \\ {cc} & {d} \end{{psmallmatrix}}"


# =============================================================================
# Gate Definitions
# =============================================================================
# Each gate is defined in both its symmetrized SU(2) form and its standard U(2) form.
# The symmetrized form multiplies by a phase to place the gate in SU(2).
#
# Convention:  G_SU2 = phase * G_U2
# so that det(G_SU2) = 1.

def _mat_to_quat(U):
    """Convert a 2×2 SU(2) matrix to a unit quaternion."""
    a = U[0, 0]
    b = U[0, 1]
    re_a, im_a = a.as_real_imag()
    re_b, im_b = b.as_real_imag()
    return Quaternion(simplify(re_a), simplify(im_a), simplify(re_b), simplify(im_b))


# Standard U(2) gates
_X = Matrix([[0, 1], [1, 0]])
_Z = Matrix([[1, 0], [0, -1]])
_H = (1 / SQRT2) * Matrix([[1, 1], [1, -1]])
_S = Matrix([[1, 0], [0, symI]])
_F = _H * _S.H                   # F = H · S†  (the "face" gate)

_Phi = Rational(1, 2) * Matrix([
    [TAU + symI * SIGMA, 1],
    [-1, TAU - symI * SIGMA],
])

# Symmetrized SU(2) gates: multiply by phase so det = 1
_GX = -symI * _X
_GZ = -symI * _Z
_GH = -symI * _H
_GS = ((1 - symI) / SQRT2) * _S
_GF = _GH * _GS.H                # F gate in SU(2)
_GPhi = _Phi                      # Φ already has det = 1

# Phase factors: G_SU2 = phase * G_U2
_PHASE_X = -symI
_PHASE_Z = -symI
_PHASE_H = -symI
_PHASE_S = (1 - symI) / SQRT2    # = exp(-iπ/4)
_PHASE_F = simplify(_GF[0, 0] / _F[0, 0]) if simplify(_F[0, 0]) != 0 else simplify(_GF[1, 0] / _F[1, 0])
_PHASE_PHI = Rational(1)           # already in SU(2)

# Build quaternions from SU(2) gates
qX = _mat_to_quat(_GX)
qZ = _mat_to_quat(_GZ)
qH = _mat_to_quat(_GH)
qS = _mat_to_quat(_GS)
qF = _mat_to_quat(_GF)
qPhi = _mat_to_quat(_GPhi)

# Gate registry: (quaternion, inverse_quaternion, magic_cost, su2_to_u2_phase)
# "magic cost" counts non-Clifford gates (only Φ is non-Clifford here).
GATES = {
    "X":  (qX,  qX.conjugate(),  0, _PHASE_X),
    "Z":  (qZ,  qZ.conjugate(),  0, _PHASE_Z),
    "H":  (qH,  qH.conjugate(),  0, _PHASE_H),
    "S":  (qS,  qS.conjugate(),  0, _PHASE_S),
    "F":  (qF,  qF.conjugate(),  0, _PHASE_F),
    "Φ":  (qPhi, qPhi.conjugate(), 1, _PHASE_PHI),
}

# Verify phase definitions: phase * G_U2 must equal G_SU2
_U2_GATES = {"X": _X, "Z": _Z, "H": _H, "S": _S, "F": _F, "Φ": _Phi}
_SU2_GATES = {"X": _GX, "Z": _GZ, "H": _GH, "S": _GS, "F": _GF, "Φ": _GPhi}
for _gn in GATES:
    _phase = GATES[_gn][3]
    _diff = _phase * _U2_GATES[_gn] - _SU2_GATES[_gn]
    assert all(_is_zero(_e) for _e in _diff), f"Phase verification failed for {_gn}"


def _gate_list_bfs(names):
    """Build gate list for BFS synthesizer — generators and adjoints."""
    gates = []
    for name in names:
        q, q_inv, _cost, _phase = GATES[name]
        gates.append((name, q))
        gates.append((name + "†", q_inv))
    return gates


def _gate_list_dijkstra(names):
    """Build gate list for Dijkstra synthesizer (with magic cost)."""
    gates = []
    for name in names:
        q, q_inv, cost, _phase = GATES[name]
        gates.append((name, q, cost))
        gates.append((name + "†", q_inv, cost))
    return gates


# Gate sets for each group
GATES_BFS_2T = _gate_list_bfs(["X", "Z", "F"])
GATES_BFS_2O = _gate_list_bfs(["X", "Z", "F", "H", "S"])
GATES_BFS_2I = _gate_list_bfs(["X", "Z", "F", "Φ"])

GATES_DIJ_2T = _gate_list_dijkstra(["X", "Z", "F"])
GATES_DIJ_2O = _gate_list_dijkstra(["X", "Z", "F", "H", "S"])
GATES_DIJ_2I = _gate_list_dijkstra(["X", "Z", "F", "Φ"])

# PU(2) gate lists: same gate sets, but the synthesizers match up to global
# phase (q ≡ -q) via proj_hash, i.e. they search the projective quotient.
# Sequences are reported in the standard U(2) gates; the *matching* is PU(2).
GATES_U2_2T = _gate_list_bfs(["X", "Z", "F"])
GATES_U2_2O = _gate_list_bfs(["X", "Z", "F", "H", "S"])
GATES_U2_2I = _gate_list_bfs(["X", "Z", "F", "Φ"])

GATES_DIJ_U2_2T = _gate_list_dijkstra(["X", "Z", "F"])
GATES_DIJ_U2_2O = _gate_list_dijkstra(["X", "Z", "F", "H", "S"])
GATES_DIJ_U2_2I = _gate_list_dijkstra(["X", "Z", "F", "Φ"])


# =============================================================================
# Group Generation: Geometric (Polytope Vertices)
# =============================================================================

def geometric_group(mode):
    """Generate group elements as quaternions from 4D polytope geometry."""
    qs = []

    # 2T: the 24-cell's complete vertex set — the 8 + 16 built below.
    # 8 quaternions ±1, ±i, ±j, ±k
    for i in range(4):
        for s in [1, -1]:
            v = [0] * 4
            v[i] = s
            qs.append(Quaternion(*v))
    # 16 quaternions (±1 ± i ± j ± k)/2
    for s in product([Rational(1, 2), Rational(-1, 2)], repeat=4):
        qs.append(Quaternion(*s))

    if mode == "2T":
        return qs

    # 2O: 2T + dual 24-cell vertices
    if mode == "2O":
        val = 1 / SQRT2
        for idx in combinations(range(4), 2):
            for s in product([1, -1], repeat=2):
                v = [0] * 4
                v[idx[0]] = s[0] * val
                v[idx[1]] = s[1] * val
                qs.append(Quaternion(*v))
        return qs

    # 2I: 2T + 96 additional vertices from even permutations of (½, τ/2, σ/2, 0)
    if mode == "2I":
        vals = [Rational(1, 2), TAU / 2, SIGMA / 2, 0]
        even_perms = [
            (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
            (1, 0, 3, 2), (1, 2, 0, 3), (1, 3, 2, 0),
            (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
            (3, 0, 2, 1), (3, 1, 0, 2), (3, 2, 1, 0),
        ]
        seen = {hash(q) for q in qs}
        for p in even_perms:
            p_vals = [vals[p[i]] for i in range(4)]
            zero_pos = p.index(3)
            for signs in product([1, -1], repeat=3):
                v = [0] * 4
                s_ptr = 0
                for i in range(4):
                    if i == zero_pos:
                        continue
                    v[i] = p_vals[i] * signs[s_ptr]
                    s_ptr += 1
                q = Quaternion(*v)
                if hash(q) not in seen:
                    qs.append(q)
                    seen.add(hash(q))
        return qs

    raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# Group Generation: Algebraic (Conway & Smith Generators + BFS Closure)
# =============================================================================

def conway_group(mode):
    """Generate group via closure of Conway & Smith generators (Theorem 12)."""
    # Conway & Smith generator: w = (-1 + i + j + k) / 2
    gen_w = Quaternion(Rational(-1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2))

    gen_map = {
        "2T": Quaternion(0, 1, 0, 0),                                  # i_T = i
        "2O": Quaternion(0, 0, 1 / SQRT2, 1 / SQRT2),                  # i_O = (j+k)/√2
        "2I": Quaternion(0, Rational(1, 2), SIGMA / 2, TAU / 2),       # i_I = (i+σj+τk)/2
    }

    generators = [gen_w, gen_map[mode]]

    group = set(generators)
    queue = list(generators)
    hashes = {hash(g) for g in generators}

    while queue:
        curr = queue.pop(0)
        for g in generators:
            for prod in [curr * g, g * curr]:
                h = hash(prod)
                if h not in hashes:
                    hashes.add(h)
                    group.add(prod)
                    queue.append(prod)

    return list(group)


# =============================================================================
# Verification
# =============================================================================

def verify_group(mode):
    """Verify that geometric and algebraic constructions produce the same group."""
    g_geom = geometric_group(mode)
    g_conway = conway_group(mode)

    expected_orders = {"2T": 24, "2O": 48, "2I": 120}
    expected = expected_orders[mode]

    set_geom = {hash(q) for q in g_geom}
    set_conway = {hash(q) for q in g_conway}

    assert len(set_geom) == expected, f"{mode} geometric: expected {expected}, got {len(set_geom)}"
    assert len(set_conway) == expected, f"{mode} Conway: expected {expected}, got {len(set_conway)}"
    assert set_geom == set_conway, f"{mode}: geometric and Conway groups differ"

    print(f"  ✓ {mode}: order {expected}, geometric = algebraic")


# =============================================================================
# Synthesis: BFS (minimum depth)
# =============================================================================

def synthesize_bfs(target_quats, gates, max_depth=12):
    """
    Find shortest gate sequences for each target quaternion via BFS.

    Returns dict mapping hash(q) -> (sequence, depth).
    """
    from collections import deque

    targets = {hash(q) for q in target_quats}
    results = {}

    qI = Quaternion(1, 0, 0, 0)
    visited = {hash(qI): []}
    queue = deque([(qI, [])])

    while queue and len(results) < len(targets):
        curr, path = queue.popleft()
        h = hash(curr)

        if h in targets and h not in results:
            results[h] = (path, len(path))

        if len(path) >= max_depth:
            continue

        for name, g_q in gates:
            new_q = curr * g_q
            nh = hash(new_q)
            if nh not in visited:
                new_path = path + [name]
                visited[nh] = new_path
                queue.append((new_q, new_path))

    return results


# =============================================================================
# Synthesis: Dijkstra (minimum magic cost, then depth)
# =============================================================================

def synthesize_dijkstra(target_quats, gates, max_depth=12):
    """
    Find optimal gate sequences for each target quaternion.

    Optimizes first by magic cost, then by depth.
    Returns dict mapping hash(q) -> (sequence, magic_cost, depth).
    """
    targets = {hash(q) for q in target_quats}
    results = {}

    qI = Quaternion(1, 0, 0, 0)
    counter = 0
    pq = [(0, 0, counter, qI, [])]
    best_cost = {hash(qI): (0, 0)}

    while pq and len(results) < len(targets):
        mag, dep, _cnt, curr, path = heapq.heappop(pq)
        h = hash(curr)

        bm, bd = best_cost.get(h, (999, 999))
        if mag > bm or (mag == bm and dep > bd):
            continue

        if h in targets and h not in results:
            results[h] = (list(path), mag, dep)

        if dep >= max_depth:
            continue

        for name, g_q, g_cost in gates:
            new_q = curr * g_q
            nh = hash(new_q)
            nm, nd = mag + g_cost, dep + 1

            prev_m, prev_d = best_cost.get(nh, (999, 999))
            if nm < prev_m or (nm == prev_m and nd < prev_d):
                best_cost[nh] = (nm, nd)
                counter += 1
                heapq.heappush(pq, (nm, nd, counter, new_q, path + [name]))

    return results


# =============================================================================
# Synthesis: PU(2) BFS (matching up to global phase, i.e. q ≡ -q)
# =============================================================================

def synthesize_u2(target_quats, gates, max_depth=12):
    """
    Find shortest gate sequences matching each target up to global phase.

    Uses proj_hash (identifying q and -q), so the search runs on the projective
    quotient PU(2) ≅ SO(3) — 12/24/60 states, not 24/48/120 — and e.g. X² = I.
    Identifying ±q is the *whole* phase quotient here: a scalar ωI has
    determinant ω², so ±I are the only global phases SU(2) admits.
    Returns dict mapping proj_hash(q) -> (sequence, depth).
    """
    from collections import deque

    targets = {q.proj_hash() for q in target_quats}
    results = {}

    qI = Quaternion(1, 0, 0, 0)
    visited = {qI.proj_hash(): []}
    queue = deque([(qI, [])])

    while queue and len(results) < len(targets):
        curr, path = queue.popleft()
        ph = curr.proj_hash()

        if ph in targets and ph not in results:
            results[ph] = (path, len(path))

        if len(path) >= max_depth:
            continue

        for name, g_q in gates:
            new_q = curr * g_q
            nph = new_q.proj_hash()
            if nph not in visited:
                new_path = path + [name]
                visited[nph] = new_path
                queue.append((new_q, new_path))

    return results


# =============================================================================
# Synthesis: PU(2) Dijkstra (min magic cost, matching up to global phase)
# =============================================================================

def synthesize_dijkstra_u2(target_quats, gates, max_depth=12):
    """
    Find optimal gate sequences matching each target up to global phase.

    Combines Dijkstra's magic-cost-first priority with proj_hash matching
    (identifying q and -q).  This can find cheaper sequences than exact
    Dijkstra because a path to -q is equally valid.
    Returns dict mapping proj_hash(q) -> (sequence, magic_cost, depth).
    """
    targets = {q.proj_hash() for q in target_quats}
    results = {}

    qI = Quaternion(1, 0, 0, 0)
    counter = 0
    pq = [(0, 0, counter, qI, [])]
    best_cost = {qI.proj_hash(): (0, 0)}

    while pq and len(results) < len(targets):
        mag, dep, _cnt, curr, path = heapq.heappop(pq)
        ph = curr.proj_hash()

        bm, bd = best_cost.get(ph, (999, 999))
        if mag > bm or (mag == bm and dep > bd):
            continue

        if ph in targets and ph not in results:
            results[ph] = (list(path), mag, dep)

        if dep >= max_depth:
            continue

        for name, g_q, g_cost in gates:
            new_q = curr * g_q
            nph = new_q.proj_hash()
            nm, nd = mag + g_cost, dep + 1

            prev_m, prev_d = best_cost.get(nph, (999, 999))
            if nm < prev_m or (nm == prev_m and nd < prev_d):
                best_cost[nph] = (nm, nd)
                counter += 1
                heapq.heappush(pq, (nm, nd, counter, new_q, path + [name]))

    return results


# =============================================================================
# Verification of Synthesis Results
# =============================================================================

def _replay_sequence(seq):
    """Multiply out a gate sequence to get the resulting quaternion."""
    result = Quaternion(1, 0, 0, 0)  # identity
    for name in seq:
        base = name.rstrip("†")
        is_dag = name.endswith("†")
        q, q_inv, _cost, _phase = GATES[base]
        result = result * (q_inv if is_dag else q)
    return result


def _replay_sequence_u2(seq):
    """Multiply out a gate sequence in the standard U(2) gates.

    Order and adjoint convention mirror `_replay_sequence`; the route into
    U(2) is the one that does not pass through `GATES`' phase factors, so a
    phase checked against this product is checked independently of the
    mechanism that produced it.
    """
    result = Matrix.eye(2)
    for name in seq:
        base = name.rstrip("†")
        is_dag = name.endswith("†")
        G = _U2_GATES[base]
        result = result * (G.H if is_dag else G)
    return result


def _compute_u2_phase(seq, q_target):
    """Compute exact phase phi such that the standard-gate sequence = phi * q_target.

    The sequence is in U(2) gates and this identity is exact in U(2) — the phase
    is what turns a PU(2) match back into one.  Since G_SU2 = phase * G_U2, the
    U(2) product is (1/∏phases) * q_SU2.
    With q_SU2 = ±q_target (from proj_hash match), phi = sign / ∏phases.
    """
    if not seq:
        # Identity sequence: U(2) = SU(2) = I, so phi = 1 if target is I, -1 if target is -I
        q_su2 = Quaternion(1, 0, 0, 0)
    else:
        q_su2 = _replay_sequence(seq)

    if q_su2 == q_target:
        sign = Rational(1)
    elif q_su2 == q_target.neg():
        sign = Rational(-1)
    else:
        raise ValueError(f"PU(2) match failed: standard-gate sequence gives {q_su2}, expected ±{q_target}")

    accumulated = Rational(1)
    for name in seq:
        base = name.rstrip("†")
        is_dag = name.endswith("†")
        _q, _q_inv, _cost, phase = GATES[base]
        gate_phase = phase.conjugate() if is_dag else phase
        accumulated *= gate_phase

    return simplify(sign / accumulated)


def verify_synthesis(rows, mode):
    """
    Verify that every synthesized sequence actually produces the target quaternion.

    - Every element must be REACHED by all four synthesizers.  An element the
      search missed carries depth -1 and the sequence ["?"], and every per-row
      check below is guarded on depth >= 0, so without this first pass they
      skip exactly the rows that went wrong: the function would print "all N
      sequences verified" having verified nothing for them, while the "?" went
      on to be typeset in atlas.tex.
    - BFS and Dijkstra sequences must exactly equal the target (SU(2) match),
      and the phase atlas.tex prints beside each must carry the word's
      standard-gate product onto the target exactly.
    - PU(2) sequences must match up to global phase (proj_hash match), and the
      stored phase must carry the word's U(2) product onto the target exactly.
      That product is multiplied out in the standard gates by
      `_replay_sequence_u2`, so it shares no mechanism with the phase it
      checks — recomputing the phase here instead would compare a pure
      function against itself and could not fail.
    """
    errors = []
    for i, row in enumerate(rows, 1):
        q = row["quat"]

        # Reachability, before anything conditioned on it
        for col, label in (("bfs", "BFS"), ("dij", "Dijkstra"),
                           ("u2", "PU(2) BFS"), ("dij_u2", "PU(2) Dijkstra")):
            if row[f"{col}_depth"] < 0:
                errors.append(f"  Row {i} {label}: {q} not reached within the "
                              f"search bound (sequence {row[f'{col}_seq']})")

        # BFS / Dijkstra checks (exact SU(2) match) + the phase printed beside each
        for col, label in (("bfs", "BFS"), ("dij", "Dijkstra")):
            if row[f"{col}_depth"] < 0:
                continue
            got = _replay_sequence(row[f"{col}_seq"])
            if got != q:
                errors.append(f"  Row {i} {label}: expected {q}, got {got}")
            elif row[f"{col}_phase"] is not None:
                resid = _replay_sequence_u2(row[f"{col}_seq"]) - row[f"{col}_phase"] * q.to_unitary()
                if not all(_is_zero(e) for e in resid):
                    errors.append(f"  Row {i} {label} phase: stored {row[f'{col}_phase']} does not "
                                  f"carry the standard-gate reading of {row[f'{col}_seq']} to {q}")

        # PU(2) BFS check (match up to global phase, i.e. q ≡ -q) + phase consistency
        if row["u2_depth"] >= 0:
            got = _replay_sequence(row["u2_seq"])
            if got.proj_hash() != q.proj_hash():
                errors.append(f"  Row {i} PU(2) BFS: expected ±{q}, got {got}")
            elif row["u2_phase"] is not None:
                resid = _replay_sequence_u2(row["u2_seq"]) - row["u2_phase"] * q.to_unitary()
                if not all(_is_zero(e) for e in resid):
                    errors.append(f"  Row {i} PU(2) BFS phase: stored {row['u2_phase']} does not "
                                  f"carry {row['u2_seq']} to {q}")

        # PU(2) Dijkstra check (match up to global phase) + phase consistency
        if row["dij_u2_depth"] >= 0:
            got = _replay_sequence(row["dij_u2_seq"])
            if got.proj_hash() != q.proj_hash():
                errors.append(f"  Row {i} PU(2) Dijkstra: expected ±{q}, got {got}")
            elif row["dij_u2_phase"] is not None:
                resid = _replay_sequence_u2(row["dij_u2_seq"]) - row["dij_u2_phase"] * q.to_unitary()
                if not all(_is_zero(e) for e in resid):
                    errors.append(f"  Row {i} PU(2) Dijkstra phase: stored {row['dij_u2_phase']} does not "
                                  f"carry {row['dij_u2_seq']} to {q}")

    if errors:
        print(f"  ✗ {mode}: {len(errors)} synthesis errors:")
        for e in errors:
            print(e)
        raise AssertionError(f"{mode}: synthesis verification failed")
    else:
        print(f"  ✓ {mode}: all {len(rows)} sequences verified")


def verify_optimality(rows, mode):
    """
    Check the relations the 2x2 synthesis grid forces between its four columns.

    "Optimal" is what the atlas column headers claim and what the thesis calls
    these sequences; verify_synthesis does not reach it, since it replays a
    word and confirms it lands on the target, which a merely CORRECT word
    also does.  All four synthesizers run over the same generator set for a
    given group (2T: X, Z, F; 2O: X, Z, F, H, S; 2I: X, Z, F, Phi -- the
    three lists are not nested, 2I carries no H and no S), so each one's
    claim constrains the other three:

      - BFS minimizes depth, so bfs_depth <= dij_depth.
      - Dijkstra minimizes magic cost, so dij_magic <= bfs_magic.
      - Dijkstra breaks ties by depth, so wherever the two agree on magic they
        must agree on depth as well.
      - Both PU(2) searches run on the quotient of the same graph, seeing two
        representatives where the SU(2) searches see one, so they can only do
        better: u2_depth <= bfs_depth, dij_u2_magic <= dij_magic.

    Two of those five bite at 2I only.  Phi is the sole gate with a nonzero
    magic cost -- X, Z, F, H and S are all 0 -- and Phi is in 2I's set alone,
    so on 2T and 2O every word costs 0 and the two magic bounds read 0 <= 0.
    They cannot fail there.  The tie-break relation still bites, and bites
    harder: with all magic equal it demands dij_depth == bfs_depth on EVERY
    row, not just where the columns happen to agree.  But magic-optimality
    itself is evidenced at 2I and nowhere else, so the success line below
    reports per group rather than claiming it three times.

    None of this duplicates numpy_atlas.py, which re-derives each column's
    VALUES with an independent Bellman-Ford and compares them one by one.  A
    fault the two implementations share -- a mistyped magic cost, a generator
    missing from one of the four lists -- moves both sides of that comparison
    together and passes it.  These relations hold between columns of a single
    run, so they do not share the mechanism they are checking.
    """
    # Precondition, before any relation that -1 would satisfy vacuously.  With
    # bfs = dij = u2 = dij_u2 = -1 every comparison below is false and the
    # tie-break branch sees equal magic at equal depth, so an all-unreached row
    # passes all five checks and prints the success line.  generate_group_data
    # now rules that out upstream; asserting it here too keeps this function
    # safe to import or to call in a different order.
    unreached = [(i, col) for i, row in enumerate(rows, 1)
                 for col in ("bfs", "dij", "u2", "dij_u2")
                 if row[f"{col}_depth"] < 0]
    assert not unreached, (
        f"{mode}: {len(unreached)} unreached (row, synthesizer) slots reached "
        f"verify_optimality, e.g. {unreached[:3]} -- every relation below would "
        f"pass on them without checking anything"
    )

    errors = []
    for i, row in enumerate(rows, 1):
        if row["bfs_depth"] > row["dij_depth"]:
            errors.append(f"  Row {i}: BFS depth {row['bfs_depth']} exceeds "
                          f"Dijkstra's {row['dij_depth']}, so BFS is not depth-optimal")
        if row["dij_magic"] > row["bfs_magic"]:
            errors.append(f"  Row {i}: Dijkstra magic {row['dij_magic']} exceeds "
                          f"BFS's {row['bfs_magic']}, so Dijkstra is not magic-optimal")
        if row["dij_magic"] == row["bfs_magic"] and row["dij_depth"] != row["bfs_depth"]:
            errors.append(f"  Row {i}: equal magic {row['dij_magic']} but depths "
                          f"{row['bfs_depth']} (BFS) vs {row['dij_depth']} (Dijkstra); "
                          f"Dijkstra's depth tie-break should have matched BFS")
        if row["u2_depth"] > row["bfs_depth"]:
            errors.append(f"  Row {i}: PU(2) BFS depth {row['u2_depth']} exceeds "
                          f"SU(2)'s {row['bfs_depth']}, but it searches a quotient")
        if row["dij_u2_magic"] > row["dij_magic"]:
            errors.append(f"  Row {i}: PU(2) Dijkstra magic {row['dij_u2_magic']} exceeds "
                          f"SU(2)'s {row['dij_magic']}, but it searches a quotient")

    if errors:
        print(f"  ✗ {mode}: {len(errors)} optimality violations:")
        for e in errors:
            print(e)
        raise AssertionError(f"{mode}: synthesis optimality failed")

    # The BEST saving over all rows, which is what "up to" reports.  The worst
    # is 0 -- most rows save nothing (mean 0.12-0.17 across the three groups) --
    # so this is an upper bound on the gain and must not be read as a floor.
    max_gain = max(r["bfs_depth"] - r["u2_depth"] for r in rows)
    # Only claim magic-optimality where the magic column varies; see the
    # docstring.  A constant column would make the claim on 0 <= 0.
    magic_varies = len({r["bfs_magic"] for r in rows}
                       | {r["dij_magic"] for r in rows}) > 1
    line = (f"  ✓ {mode}: BFS depth-optimal, "
            + ("Dijkstra magic-optimal, " if magic_varies else "")
            + f"PU(2) saves up to {max_gain} gate(s)")
    if not magic_varies:
        line += ("; magic 0 on every word, so the two magic bounds are vacuous "
                 "and only Dijkstra's depth tie-break is under test")
    print(line)


def verify_atlas_layout(pairs, mode):
    """
    Check the claims Appendix A's preamble makes about the printed table.

    atlas.tex prints, for every group element, its own min-magic SU(2) word
    (Dijkstra) with depth, Φ count and phase -- for 2T and 2O also its
    min-depth word, the two coinciding -- and keeps the two rows of an
    antipodal pair (q, -q) together, BFS-shallower member first. Nothing from
    the two PU(2) synthesizers is printed, and 2I's min-depth words print
    outside it: in Appendix B's Table B.1 where they differ, and in the
    preamble's sample row. The preamble justifies that with four
    claims about the data, which the reader cannot check because the numbers
    they are about are the ones not printed:

      (1) The block's title: for 2T and 2O BFS and Dijkstra return the same
          word on every row, so the block is headed as both. For 2I they
          differ on exactly 17 rows, the count the preamble and Section
          4.1.2 print (`verify_differing_rows` pins what that section says of
          them, `table_b1_tex` renders them into Appendix B's Table B.1,
          atlas.txt keeps the grid) -- the
          count is pinned here, so a tie-break change that moved it fails
          loudly instead of desynchronising the atlas from the sentence.
      (2) The first row of a pair is the projective answer: it minimises the
          pair's BFS depth AND its min-magic (Φ, depth). The BFS half is
          forced by _pair_antipodal_rows's key; the Φ half cannot fail
          (-1 = X X costs no magic, so Φ is constant along every pair); the
          min-magic DEPTH half is a property of this data, true on all 96
          pairs, and is asserted, never assumed. This is what makes the first
          row's words the PU(2) optima: the quotient search's distance to a
          class is the minimum over its fibre.
      (3) The PU(2) synthesizers add no word of their own. Each returns, as a
          U(2) operator, the FIRST row's word at the first row's depth and
          Φ, on every pair. Operators are compared, never letters: on two 2I
          pairs, tied on (Φ, depth), PU(2) Dijkstra spells the second row's
          word (Z for Z†, the same standard-gate operator), where a letter
          check would fail for no reason -- and the operator form is what the
          preamble's recovery rule, read the first row's word in the
          standard gates, actually needs.
      (4) Along a pair, φ reads φ and -φ exactly when the two words are one
          standard-gate operator. Algebra forces the iff (M = φ_a U_q =
          φ_b U_{-q} = -φ_b U_q), so a failure here is a phase bug; it is
          asserted because the preamble's reuse rule φU = (-φ)(-U) rests on
          it, and the counts it reports are pinned in `_ATLAS_PAIR_COUNTS`.
    """
    rows = [r for pair in pairs for r in pair]
    for r in rows:
        # Same vacuity as verify_optimality: -1 <= -1 satisfies everything below.
        for col in ("bfs", "dij", "u2", "dij_u2"):
            assert r[f"{col}_depth"] >= 0, (
                f"{mode}: {col} depth is -1 on {r['quat']}, so the layout "
                f"claims below would hold vacuously and go unchecked"
            )

    # (1) the block's title: min depth and min magic coincide for 2T and 2O; 2I differs on 17 rows
    differing = [r for r in rows if r["bfs_seq"] != r["dij_seq"]]
    magic_varies = len({r["dij_magic"] for r in rows}) > 1
    if not magic_varies:
        assert not differing, (
            f"{mode}: BFS and Dijkstra disagree on {len(differing)} row(s) "
            f"(first: {differing[0]['quat']}), but atlas.tex heads this group's "
            f"block as min depth and min magic on the claim that they agree"
        )
    else:
        assert len(differing) == 17, (
            f"{mode}: BFS and Dijkstra disagree on {len(differing)} rows, not "
            f"the 17 that the preamble of Appendix A and Section 4.1.2 count"
        )

    def same_operator(seq_a, seq_b):
        resid = _replay_sequence_u2(seq_a) - _replay_sequence_u2(seq_b)
        return all(_is_zero(e) for e in resid)

    strict = {"bfs": 0, "dij": 0}       # pairs whose first row is strictly shallower
    same_op = {"bfs": 0, "dij": 0}      # pairs whose two words are one standard-gate operator
    respelt = []                        # pairs where a PU(2) word is spelt as the second row's
    for pi, (first, second) in enumerate(pairs, 1):
        # (2) the first row minimises BFS depth and min-magic (Φ, depth) over the pair
        assert first["bfs_depth"] <= second["bfs_depth"], (
            f"{mode} pair {pi}: second row is BFS-shallower "
            f"({second['bfs_depth']} < {first['bfs_depth']}); _pair_antipodal_rows's key is broken"
        )
        assert first["dij_magic"] == second["dij_magic"], (
            f"{mode} pair {pi}: Φ count differs along the pair "
            f"({first['dij_magic']} vs {second['dij_magic']}), but -1 = X X is free"
        )
        assert first["dij_depth"] <= second["dij_depth"], (
            f"{mode} pair {pi}: the second row's min-magic word is shallower "
            f"({second['dij_depth']} < {first['dij_depth']}), so the first row is not "
            f"the projective answer the preamble promises"
        )
        for col in ("bfs", "dij"):
            strict[col] += first[f"{col}_depth"] < second[f"{col}_depth"]

        # (3) the PU(2) synthesizers return the first row's words, as operators
        for u2_col, su2_col, label in (("u2", "bfs", "PU(2) BFS"),
                                       ("dij_u2", "dij", "PU(2) Dijkstra")):
            # Structural today: generate_group_data looks the class up once,
            # so both members hold the same list and this compares it with
            # itself. Kept as the statement of what (3) rests on, so a
            # per-element refactor of the PU(2) lookup cannot hand the two
            # members different words without failing here.
            assert first[f"{u2_col}_seq"] == second[f"{u2_col}_seq"], (
                f"{mode} pair {pi}: {label} gave the two members different words, "
                f"but it keys on proj_hash and sees one class"
            )
            word = first[f"{u2_col}_seq"]
            assert same_operator(word, first[f"{su2_col}_seq"]), (
                f"{mode} pair {pi}: {label} word {word} is not the first row's "
                f"{su2_col} word {first[f'{su2_col}_seq']} as a U(2) operator; dropping "
                f"the PU(2) columns loses it"
            )
            for stat in ("depth", "magic"):
                got, want = first[f"{u2_col}_{stat}"], first[f"{su2_col}_{stat}"]
                assert got == want, (
                    f"{mode} pair {pi}: {label} {stat} is {got} but the first row's is {want}"
                )
            if word != first[f"{su2_col}_seq"]:
                assert word == second[f"{su2_col}_seq"], (
                    f"{mode} pair {pi}: {label} word {word} is the first row's operator but "
                    f"neither row's spelling"
                )
                respelt.append((pi, label))

        # (4) φ and -φ along a pair iff one standard-gate operator
        for col in ("bfs", "dij"):
            same = same_operator(first[f"{col}_seq"], second[f"{col}_seq"])
            negated = _is_zero(first[f"{col}_phase"] + second[f"{col}_phase"])
            assert same == negated, (
                f"{mode} pair {pi} [{col}]: words {first[f'{col}_seq']} / {second[f'{col}_seq']} "
                f"are {'one' if same else 'different'} standard-gate operator(s) but the phases "
                f"{first[f'{col}_phase']} / {second[f'{col}_phase']} "
                f"{'do' if negated else 'do not'} negate -- the algebra says they must agree"
            )
            same_op[col] += same

    # The pair counts, pinned in `_ATLAS_PAIR_COUNTS`: a tie-break change that moved
    # one fails here instead of passing silently.
    want = _ATLAS_PAIR_COUNTS[mode]
    got = {"pairs": len(pairs), "same_op_bfs": same_op["bfs"], "same_op_dij": same_op["dij"],
           "strict_bfs": strict["bfs"], "strict_dij": strict["dij"]}
    assert got == want, f"{mode}: pair counts {got} differ from the pinned {want}"

    print(f"  ✓ {mode}: preamble claims hold on all {len(pairs)} pairs -- "
          + (f"BFS and Dijkstra differ on {len(differing)} rows (atlas.txt)"
             if magic_varies else "BFS and Dijkstra agree on every row")
          + f"; first row strictly shallower on {strict['bfs']} (BFS) / {strict['dij']} (Dijkstra); "
          f"one operator along {same_op['bfs']} / {same_op['dij']} pairs"
          + (f"; PU(2) words spelt as the second row's on {respelt}" if respelt else ""))


# What verify_atlas_layout counts, pinned. `same_op_*`: pairs whose two words
# (min depth / min magic) are one standard-gate operator, hence φ and -φ.
# `strict_*`: pairs whose first row is strictly shallower than its second,
# visible in the table as first-row depth < second-row depth. The thesis prints
# neither family; both are pinned as data facts, so a tie-break change still
# fails loudly here.
_ATLAS_PAIR_COUNTS = {
    "2T": {"pairs": 12, "same_op_bfs": 10, "same_op_dij": 10, "strict_bfs": 3, "strict_dij": 3},
    "2O": {"pairs": 24, "same_op_bfs": 22, "same_op_dij": 22, "strict_bfs": 5, "strict_dij": 5},
    "2I": {"pairs": 60, "same_op_bfs": 45, "same_op_dij": 52, "strict_bfs": 11, "strict_dij": 9},
}
# The totals over the three groups: of the 96 pairs, 77 are one operator under
# minimum depth and 84 under minimum magic (atlas.txt's header derives their
# complements, 19 and 12, from the data).
assert sum(v["pairs"] for v in _ATLAS_PAIR_COUNTS.values()) == 96
assert sum(v["same_op_bfs"] for v in _ATLAS_PAIR_COUNTS.values()) == 77
assert sum(v["same_op_dij"] for v in _ATLAS_PAIR_COUNTS.values()) == 84


# Angle of the pair's rotation, 2·arccos w in degrees, keyed by w at unit
# scale: (scale factor, unit) -> degrees. w = 0 is a half-turn at any scale.
_ANGLE_DEG = {
    (Rational(1), "1"): 0,
    (Rational(1, 2), "tau"): 72,
    (1 / sqrt(2), "1"): 90,
    (Rational(1, 2), "1"): 120,
    (Rational(1, 2), "sigma"): 144,
}


def pair_rotation(first, second):
    """The Bloch rotation R_n̂(θ) an antipodal pair is.

    q and -q rotate the Bloch sphere identically, and the rotation has one
    description with θ in [0°, 180°]: take the member with w ≥ 0, then
    θ = 2·arccos w and the axis is -(z, y, x) -- Section 2.4's convention,
    the naive (x, y, z) reversed and negated. Half-turns have w = 0 on both
    members and no preferred axis sign; the member whose axis has a positive
    first nonzero entry is taken. The identity has no axis.

    Returns (rep, axis, degrees): rep is the chosen member's row; axis is the
    direction at unit scale -- the common positive scalar dropped, so entries
    read 0, ±1, ±τ, ±σ -- as three (sign, name) pairs, or None for the
    identity; degrees is in {0, 72, 90, 120, 144, 180}. verify_atlas_rotations
    checks every return value against the conjugation action of U_q.
    """
    def classify(row):
        q = row["quat"]
        c = q.latex_scale()
        assert c is not None, f"{q}: no scale factor, cannot read off the rotation"
        w = _atlas_unit(q.w / c)
        axis = [_atlas_unit(v / c) for v in (-q.z, -q.y, -q.x)]
        return c, w, axis

    c, (ws, wn), axis = classify(first)
    if wn == "0":
        lead = next(sign for sign, name in axis if name != "0")
        rep = first if lead > 0 else second
        c, (ws, wn), axis = classify(rep)
        return rep, axis, 180
    rep = first if ws > 0 else second
    c, (ws, wn), axis = classify(rep)
    assert ws > 0
    if wn == "1" and c == 1:
        return rep, None, 0
    key = next(k for k in _ANGLE_DEG if simplify(k[0] - c) == 0 and k[1] == wn)
    return rep, axis, _ANGLE_DEG[key]


def _bloch_rotation_matrix(U):
    """The SO(3) matrix of U acting on the Paulis: U σ_j U† = Σ_i R_ij σ_i."""
    paulis = [Matrix([[0, 1], [1, 0]]), Matrix([[0, -symI], [symI, 0]]), Matrix([[1, 0], [0, -1]])]
    Ud = U.H
    return Matrix(3, 3, lambda i, j: expand((paulis[i] * U * paulis[j] * Ud).trace() / 2))


def _rodrigues(w, v):
    """I + 2w K(v) + 2 K(v)², the rotation of the unit quaternion (w, v)."""
    from sympy import eye
    K = Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return eye(3) + 2 * w * K + 2 * K * K


# How many pairs the WRONG axis conventions would pass in verify_atlas_rotations
# -- the check's teeth, pinned. Flipping the sign of v passes exactly the
# identity and the half-turns (the K(v) term is odd in v and vanishes with w),
# so that count is derived, not pinned. The naive (x, y, z) axis passes the
# identity and the half-turns about (a, b, a)- and (1, 0, -1)-type directions,
# where (x, y, z) = ±(-z, -y, -x) -- the per-group counts below.
_NAIVE_AXIS_PASSES = {"2T": 2, "2O": 4, "2I": 2}


def verify_atlas_rotations(pairs, mode):
    """
    Check the R_n̂(θ) column against what U_q does to the Paulis.

    The column is read off the quaternion by the thesis's convention --
    θ = 2·arccos w, axis -(z, y, x) -- and a convention is exactly the kind
    of thing that is quietly wrong: the textbook axis is (x, y, z), and for
    2T and 2O the two conventions give the same group, so only 2I would
    notice, and only in the icosahedral family it lands on (Appendix E.2).
    So the column is not trusted to the formula. For every pair, R is
    computed from the conjugation action U_q σ_j U_q† and compared, exactly,
    with the Rodrigues matrix of the printed axis and angle:

        R = I + 2w K(v) + 2 K(v)²,   v = -(z, y, x),   K(v) u = v × u,

    which is I + sin θ K(n̂) + (1 − cos θ) K(n̂)² at n̂ = v/|v|, θ = 2·arccos w,
    written without square roots so the field test decides it. The identity
    holds with the printed sign of v and fails with the opposite one (the
    K(v) term is odd in v), so it pins the axis's direction, not just its
    line; the angle is pinned separately by cos(θ/2) = w, and the other
    member is checked to give the same R, as a pair property must. Finally
    the two wrong conventions are run on the same pairs and must fail where
    `_NAIVE_AXIS_PASSES` says they fail -- a check that cannot reject a wrong
    convention is not a check.
    """
    from sympy import cos, pi
    flipped_pass = naive_pass = half_turns = 0
    for pi_, (first, second) in enumerate(pairs, 1):
        rep, axis, deg = pair_rotation(first, second)
        assert rep is first or rep is second
        q = rep["quat"]
        assert _is_zero(cos(pi * deg / 360) - q.w), (
            f"{mode} pair {pi_}: printed angle {deg}° but cos({deg}°/2) ≠ w = {q.w}"
        )
        v = Matrix([-q.z, -q.y, -q.x])
        if axis is None:
            assert deg == 0 and all(_is_zero(e) for e in v), f"{mode} pair {pi_}: axis omitted off the identity"
        else:
            c = q.latex_scale()
            for (sign, name), comp in zip(axis, v):
                printed = sign * _ATLAS_UNIT_SYM[name]
                assert _golden_sub(simplify(comp / c), _tau_sym, _sig_sym) == printed, (
                    f"{mode} pair {pi_}: printed axis entry {printed} is not {comp}/{c}"
                )
            if deg == 180:
                lead = next(sign for sign, name in axis if name != "0")
                assert lead > 0, f"{mode} pair {pi_}: half-turn axis not signed to a positive first nonzero entry"
                half_turns += 1
        R = _bloch_rotation_matrix(q.to_unitary())
        assert all(_is_zero(e) for e in (R - _rodrigues(q.w, v))), (
            f"{mode} pair {pi_}: U_q's action on the Paulis is not the printed rotation "
            f"({deg}° about {list(v)})"
        )
        other = second if rep is first else first
        R2 = _bloch_rotation_matrix(other["quat"].to_unitary())
        assert all(_is_zero(e) for e in (R - R2)), f"{mode} pair {pi_}: q and -q rotate differently"
        # the teeth: the wrong conventions, on the same pair
        flipped_pass += all(_is_zero(e) for e in (R - _rodrigues(q.w, -v)))
        naive_pass += all(_is_zero(e) for e in (R - _rodrigues(q.w, Matrix([q.x, q.y, q.z]))))
    assert flipped_pass == 1 + half_turns, (
        f"{mode}: the flipped axis sign passes {flipped_pass} pairs, not the identity "
        f"plus the {half_turns} half-turns"
    )
    assert naive_pass == _NAIVE_AXIS_PASSES[mode], (
        f"{mode}: the naive (x, y, z) axis passes {naive_pass} pairs, not the pinned "
        f"{_NAIVE_AXIS_PASSES[mode]}"
    )
    print(f"  ✓ {mode}: R_n̂(θ) matches U_q's conjugation action on all {len(pairs)} pairs "
          f"(flipped sign would pass {flipped_pass}, naive axis {naive_pass})")


def verify_atlas_nesting(all_data):
    """
    Check the preamble's sentence that Table A.3 opens with Table A.1.

    Pairs are ordered by magic cost first (`_atlas_key`), so 2I's twelve
    Clifford pairs -- its 2T subgroup -- come before its 48 golden ones. The
    thesis says more: that those twelve are Table A.1 over again. Same
    quaternions in the same order is forced by the shared key only if the
    printed words agree too, and 2I's searches run over a larger gate set
    than 2T's, so a tie broken towards a word 2T cannot spell would reorder
    the block silently. Hence the check: row for row, both members of every
    pair, same quaternion, same words, same depth, Φ and phase.
    """
    by_mode = {name.split()[-1]: pairs for name, _rows, pairs in all_data}
    t_pairs, i_pairs = by_mode["2T"], by_mode["2I"]
    clifford = [p for p in i_pairs if p[0]["dij_magic"] == 0]
    assert len(clifford) == len(t_pairs) == 12, (
        f"2I has {len(clifford)} Clifford pairs against 2T's {len(t_pairs)}"
    )
    assert clifford == i_pairs[:12], "2I's Clifford pairs are not its first twelve"
    for k, (tp, ip) in enumerate(zip(t_pairs, i_pairs), 1):
        for t_row, i_row in zip(tp, ip):
            for field in ("quat", "bfs_seq", "bfs_depth", "bfs_phase",
                          "dij_seq", "dij_depth", "dij_magic", "dij_phase"):
                assert t_row[field] == i_row[field], (
                    f"pair {k}: Table A.3 reads {field} = {i_row[field]} where "
                    f"Table A.1 reads {t_row[field]}; the preamble says the first "
                    f"twelve pairs of 2I are Table A.1 over again"
                )
    print("  ✓ 2I: its twelve Clifford pairs open Table A.3 and are Table A.1 row for row")


THESIS_TEX = Path(__file__).resolve().parent.parent / "paper" / "bsc-thesis.tex"


def _tex_word(cell, where):
    r"""The gate names of a typeset word.

    `\mathbf{X}` bold letters, `\Phi`, and `^\dagger` braced or not -- the
    thesis's spelling (`^\dagger`) and atlas.tex's (`^{\dagger}`) alike; `\Id`
    is the empty word. Anything else in the cell is a parse error, not a
    near miss.
    """
    body = cell.strip().strip("$").strip()
    if body == r"\Id":
        return []
    names = []
    for bold, phi, dagger, other in re.findall(r"\\mathbf\{([XZFHS])\}|(\\Phi)|(\^\{?\\dagger\}?)|(\S)", body):
        if bold:
            names.append(bold)
        elif phi:
            names.append("Φ")
        elif dagger:
            assert names, f"{where}: a dagger with no gate before it in {cell!r}"
            names[-1] += "†"
        else:
            raise AssertionError(f"{where}: unexpected token {other!r} in {cell!r}")
    return names


def _strip_tex_comments(tex):
    r"""LaTeX comments out. A `%` opens one unless it is escaped: `\%` is a
    percent sign, `\\%` is a row terminator followed by a comment -- what
    decides is the parity of the backslashes in front of it."""
    return re.sub(r"((?<!\\)(?:\\\\)*)%.*", r"\1", tex)


def table_b1_tex(rows, order=()):
    r"""The rows where BFS and Dijkstra differ, in the atlas's own format.

    The body of Appendix B's Table B.1: one line per row, keyed by the
    element's row of Table A.3 -- its pair number and a/b, from the same
    `_pair_antipodal_rows` pass the atlas numbers its pairs by -- then the
    two word blocks through `_word_cells`, then the element's own signed
    quaternion (the atlas prints +-q once per pair, which is no use here,
    where the two members of a pair are rows apart). A reader can therefore
    carry a row straight into Appendix A -- `#` is a link to the atlas row,
    which carries the B mark linking back (`_atlas_key_cell`) -- and, since
    the two blocks disagree on phi on every row of this table, must: the
    atlas prints Dijkstra's phase, which is the wrong one for the BFS word.

    `write_differing_rows` wraps this in the float. Rows follow `order`
    (quaternions), then the rest by BFS Phi count descending -- the biggest
    trades first and the four rows that only rearrange last, so the caption's
    13-and-4 split reads off the page in that order -- and, inside a tier, by
    the atlas key, so a reader holding one can find its row by eye instead of
    scanning all 17. The descending tiers are what the caption's split rests
    on; the order inside a tier is a lookup convenience.
    `verify_differing_rows` exercises both halves.
    """
    keyed = []
    for pi, pair in enumerate(_pair_antipodal_rows(rows), 1):
        for letter, row in zip("ab", pair):
            if row["bfs_seq"] != row["dij_seq"]:
                assert row["quat"].latex_scale() == Rational(1, 2), (
                    f"{row['quat']}: the differing rows are typed at scale 1/2"
                )
                keyed.append((f"{pi}{letter}", pi, letter, row))
    keyed.sort(key=lambda k: (-k[3]["bfs_magic"], k[1], k[2]))
    entries = [(key, row) for key, _pi, _letter, row in keyed]
    by_quat = {row["quat"]: (key, row) for key, row in entries}
    ordered = ([by_quat[q] for q in order if q in by_quat]
               + [(key, row) for key, row in entries if row["quat"] not in order])
    lines = []
    for key, row in ordered:
        cells = _word_cells(row, "bfs") + _word_cells(row, "dij") + [f"${row['quat'].latex_quat()}$"]
        lines.append(f"  \\hyperlink{{{_atlas_anchor(_B1_MODE, key)}}}{{{key}}} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def verify_differing_rows(rows, mode="2I"):
    """
    Pin what the thesis says about the rows where BFS and Dijkstra differ.

    Section 4.1.2 prints the facts, the preamble of Appendix A the count,
    and Appendix B's Table B.1 the rows themselves (`table_b1_tex` renders
    them, `write_differing_rows` writes the float, atlas.txt keeps the full
    grid they come out of). There are 17; every one has a
    single-Φ Dijkstra word; on 13 Dijkstra lowers the Φ count and on the
    other 4 both words already spend one Φ and differ at equal depth; the
    trade costs at most three extra Clifford gates and two extra levels of
    depth, the depth-2 ΦΦ becoming the depth-4 F X Φ† Z. verify_atlas_layout
    pins the 17 independently; this pins the rest, so a synthesizer change
    that moved any of it fails here naming the sentence it moved. The
    rendering is pinned too: one line per row in the atlas's format, each
    keyed by its own Table A.3 row.
    """
    differing = [r for r in rows if r["bfs_seq"] != r["dij_seq"]]
    assert len(differing) == 17, f"{mode}: {len(differing)} rows differ, not the 17 Section 4.1.2 prints"
    assert all(r["dij_magic"] == 1 for r in differing), (
        f"{mode}: a differing row's Dijkstra word is not single-Φ, but Section 4.1.2 says every one is"
    )
    lowered = [r for r in differing if r["bfs_magic"] > r["dij_magic"]]
    rearranged = [r for r in differing if r["bfs_magic"] == r["dij_magic"]]
    assert (len(lowered), len(rearranged)) == (13, 4), (
        f"{mode}: Dijkstra lowers the Φ count on {len(lowered)} rows and rearranges {len(rearranged)}, "
        f"not the 13 and 4 the thesis prints"
    )
    assert all(r["bfs_depth"] == r["dij_depth"] for r in rearranged), (
        f"{mode}: a rearranged row changes depth; the two words were to differ at equal depth"
    )
    extra_clifford = max((r["dij_depth"] - r["dij_magic"]) - (r["bfs_depth"] - r["bfs_magic"]) for r in differing)
    extra_depth = max(r["dij_depth"] - r["bfs_depth"] for r in differing)
    assert (extra_clifford, extra_depth) == (3, 2), (
        f"{mode}: the trade costs up to {extra_clifford} extra Clifford gates and {extra_depth} extra "
        f"levels of depth; Section 4.1.2 says three and two"
    )
    phi_phi = [r for r in differing if r["bfs_seq"] == ["Φ", "Φ"]]
    assert [r["dij_seq"] for r in phi_phi] == [["F", "X", "Φ†", "Z"]], (
        f"{mode}: Section 4.1.2's exhibit, the depth-2 ΦΦ becoming the depth-4 F X Φ† Z, reads "
        f"{[(r['bfs_seq'], r['dij_seq']) for r in phi_phi]} in the data"
    )
    rendered = table_b1_tex(rows)
    assert rendered.count("\n") == len(differing), "table_b1_tex did not render one line per differing row"
    keys = [_read_b1_key(line.strip().partition(" & ")[0], f"{mode}: Table B.1") for line in rendered.splitlines()]
    assert len(keys) == len(differing) and len(set(keys)) == len(differing), (
        f"{mode}: {len(set(keys))} of {len(differing)} rows carry a distinct Table A.3 key; the "
        f"'#' column is the reader's index into Appendix A and every row needs its own"
    )
    # The caption promises '#' is the element's row of Table A.3, so the key had
    # better select the same element there: pull the atlas's own lines out of its
    # own emitter and match Dijkstra's four cells against Table B.1's. This is
    # what a change to the pairing, the numbering or the cell format would break.
    # The atlas's B marks are the other half of the same promise: a row is
    # marked exactly when this table carries it (verify_atlas_tex reads the
    # same off the written file).
    atlas_dij, atlas_row = {}, {}
    for pi, (first, second) in enumerate(_pair_antipodal_rows(rows), 1):
        atlas_row[f"{pi}a"], atlas_row[f"{pi}b"] = first, second
        for line in atlas_pair_tex(pi, first, second, ["dij"], mark=mode).splitlines():
            cell, _, rest = line.strip().partition(" & ")
            key, marked = _read_atlas_key(cell, mode, f"{mode}: atlas row")
            assert marked == (key in keys), (
                f"{mode}: atlas row {key} is {'marked' if marked else 'unmarked'} but Table B.1 "
                f"{'does not carry' if marked else 'carries'} it"
            )
            atlas_dij[key] = rest.split(" & ")[:4]
    for line in rendered.splitlines():
        cell, _, rest = line.strip().partition(" & ")
        key = _read_b1_key(cell, f"{mode}: Table B.1")
        cells = rest.split(" & ")[4:8]
        assert cells == atlas_dij[key], (
            f"{mode}: Table B.1 row {key} prints {cells} for Dijkstra where Table A.3's row {key} "
            f"prints {atlas_dij[key]}; the caption sends the reader from one to the other"
        )
    # The printed order, both halves of it: tiers of BFS Phi count descending,
    # so the caption's 13-and-4 split reads down the page (the 13 it lowers all
    # spend 2 or 3, the 4 it rearranges spend 1, so monotone tiers put the 13
    # first), and, inside a tier, Table A.3's own key order, which is what
    # makes the table a lookup.
    ranked = [(-atlas_row[key]["bfs_magic"], int(key[:-1]), key[-1]) for key in keys]
    assert ranked == sorted(ranked), (
        f"{mode}: Table B.1's rows are out of order. They run by BFS Phi count, so the caption's "
        f"13-and-4 split reads top to bottom, then by the Table A.3 key, so a reader holding one "
        f"can find its row; the rendered order is {keys}"
    )
    print(f"  ✓ {mode}: the 17 differing rows read as Section 4.1.2 says "
          f"({len(lowered)} lowering the Φ count, {len(rearranged)} rearranging; "
          f"up to +{extra_clifford} Clifford gates, +{extra_depth} depth)")


_DEPTH_CONCENTRATION = (
    # mode, order, and the buckets behind Appendix B's lede: (depths, count)
    ("2T", 24, (((2,), 17),)),
    ("2O", 48, (((2,), 33), ((3,), 4))),
    ("2I", 120, (((2, 3), 99), ((4,), 12))),
)


def verify_depth_concentration(all_data):
    """Pin the six depth counts behind Appendix B's "concentrates" claim.

    Appendix B's lede says only that depth "concentrates just under the
    ceiling of Table 4.1"; what that means, of the sequences the atlas prints,
    is that 17 of 2T's 24 elements run at depth 2, 33 of 2O's 48 at depth 2
    and four at 3, and 99 of 2I's 120 at depth 2 or 3 and twelve at 4. The
    counts are asserted here, so the printed word keeps a checkable referent.
    The atlas prints Dijkstra's words, so these are dij_depth counts and not
    BFS ones -- the BFS distributions bpg-walkthrough.ipynb plots -- and the
    two disagree on 2I, where BFS puts 68 rows at depth 3 and 4 at depth 4.
    Nothing in print gives the distribution: Table 4.1 gives the maxima and
    nothing else.
    """
    by_mode = {name.split()[-1]: rows for name, rows, _pairs in all_data}
    for mode, order, buckets in _DEPTH_CONCENTRATION:
        rows = by_mode[mode]
        assert len(rows) == order, f"{mode}: {len(rows)} elements, not the {order} Appendix B's lede concentrates"
        for depths, count in buckets:
            got = sum(1 for r in rows if r["dij_depth"] in depths)
            where = " or ".join(str(d) for d in depths)
            assert got == count, (
                f"{mode}: {got} of its {order} printed sequences run at depth {where}; "
                f"Appendix B's \"concentrates just under the ceiling\" is {count}"
            )
    print("  ✓ Appendix B's lede: the printed sequences' depths concentrate as it claims "
          "(2T 17/24 at 2; 2O 33/48 at 2, 4 at 3; 2I 99/120 at 2 or 3, 12 at 4)")


# =============================================================================
# Table Generation
# =============================================================================

def generate_group_data(mode, bfs_gates, dij_gates, u2_gates, dij_u2_gates):
    """Generate full table data for one group using all four synthesizers.

    The four synthesis strategies form a 2×2 grid:
                        Exact (SU(2))    Up to phase (PU(2))
      Min depth         BFS              PU(2) BFS
      Min magic cost    Dijkstra         PU(2) Dijkstra
    """
    quats = geometric_group(mode)
    bfs_results = synthesize_bfs(quats, bfs_gates)
    dij_results = synthesize_dijkstra(quats, dij_gates)
    u2_results = synthesize_u2(quats, u2_gates)
    dij_u2_results = synthesize_dijkstra_u2(quats, dij_u2_gates)

    rows = []
    for q in quats:
        h = hash(q)
        ph = q.proj_hash()

        def _seq_magic(seq):
            return sum(GATES[n.rstrip("†")][2] for n in seq if n != "?")

        # BFS result (minimum circuit depth, generators + adjoints). bfs_phase
        # is the phase atlas.tex prints beside the word: the word read in the
        # STANDARD gates equals bfs_phase * U_q. Each row carries its own
        # word's phase, so the table never asks the reader to borrow a word
        # from the other member of the pair.
        if h in bfs_results:
            bfs_seq, bfs_dep = bfs_results[h]
            bfs_mag = _seq_magic(bfs_seq)
            bfs_phase = _compute_u2_phase(bfs_seq, q)
        else:
            bfs_seq, bfs_dep, bfs_mag = ["?"], -1, -1
            bfs_phase = None

        # Dijkstra result (minimum magic cost, then depth); dij_phase as above.
        if h in dij_results:
            dij_seq, dij_mag, dij_dep = dij_results[h]
            dij_phase = _compute_u2_phase(dij_seq, q)
        else:
            dij_seq, dij_mag, dij_dep = ["?"], -1, -1
            dij_phase = None

        # PU(2) BFS result (minimum depth, matching up to global phase)
        if ph in u2_results:
            u2_seq, u2_dep = u2_results[ph]
            u2_mag = _seq_magic(u2_seq)
            u2_phase = _compute_u2_phase(u2_seq, q)
        else:
            u2_seq, u2_dep, u2_mag = ["?"], -1, -1
            u2_phase = None

        # PU(2) Dijkstra result (minimum magic cost, matching up to global phase)
        if ph in dij_u2_results:
            dij_u2_seq, dij_u2_mag, dij_u2_dep = dij_u2_results[ph]
            dij_u2_phase = _compute_u2_phase(dij_u2_seq, q)
        else:
            dij_u2_seq, dij_u2_mag, dij_u2_dep = ["?"], -1, -1
            dij_u2_phase = None

        rows.append({
            "quat": q,
            "bfs_seq": bfs_seq,
            "bfs_depth": bfs_dep,
            "bfs_magic": bfs_mag,
            "bfs_phase": bfs_phase,
            "dij_seq": dij_seq,
            "dij_magic": dij_mag,
            "dij_depth": dij_dep,
            "dij_phase": dij_phase,
            "u2_seq": u2_seq,
            "u2_depth": u2_dep,
            "u2_magic": u2_mag,
            "u2_phase": u2_phase,
            "dij_u2_seq": dij_u2_seq,
            "dij_u2_magic": dij_u2_mag,
            "dij_u2_depth": dij_u2_dep,
            "dij_u2_phase": dij_u2_phase,
        })

    # Reachability, asserted at the producer so every consumer inherits it.
    # An element the search missed carries depth -1 and the sequence ["?"], and
    # both downstream paths treat that as data: main() typesets the "?" into
    # atlas.tex, and export_numpy.extract_synthesis -- which calls this function
    # directly and runs no verification of its own -- writes it into
    # group_{mode}.npz, with _phase_array mapping the None phase to a
    # plausible-looking omega^0.  The sort below would place such a row FIRST,
    # at the head of the table.  verify_synthesis asserts this too, per row and
    # with a better message, but only on main()'s path; this is the statement
    # that check is a corollary of.
    for i, row in enumerate(rows, 1):
        for col in ("bfs", "dij", "u2", "dij_u2"):
            assert row[f"{col}_depth"] >= 0, (
                f"{mode} row {i}: {row['quat']} not reached by {col} within the "
                f"search bound (max_depth=12). The rows are incomplete, and "
                f"neither atlas.tex nor group_{mode}.npz may be written from them"
            )

    rows.sort(key=lambda r: (r["bfs_depth"], str(r["bfs_seq"])))
    return rows


def _atlas_key(row):
    """The order atlas.tex reads in: magic cost, then depth, then the word.

    Magic first puts 2I's twelve Clifford pairs -- its 2T subgroup, Table A.1
    over again -- at the head of Table A.3 as a block, so the almost-Clifford
    finding is the table's shape (`verify_atlas_nesting`). For 2T and 2O
    every word has magic 0 and this is the depth order. `rows` (the npz
    order) keeps its own BFS key; only the typeset order changes.
    """
    return (row["dij_magic"], row["dij_depth"], str(row["dij_seq"]))


def _pair_antipodal_rows(rows):
    """Group rows into (q, -q) pairs; the pair's two rows print together.

    The FIRST row of a pair is its BFS-shallower member, ties broken by the
    word -- the same rule shadow_experiments.load_series applies when it
    takes one circuit per rotation for Appendix F's gate-noise study (it
    scans the npz rows, which are sorted by this very key, and keeps the
    first of minimum depth), so the atlas's first row is that study's word.
    The preamble tells the reader more: that the first row is the pair's
    projective answer, minimising (Φ, depth) under min magic as well. That
    half is not forced by this key; `verify_atlas_layout` asserts it.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[row["quat"].proj_hash()].append(row)

    pairs = []
    for pair_rows in groups.values():
        assert len(pair_rows) == 2, f"Expected pair, got {len(pair_rows)}"
        pair_rows.sort(key=lambda r: (r["bfs_depth"], str(r["bfs_seq"])))
        pairs.append((pair_rows[0], pair_rows[1]))

    pairs.sort(key=lambda p: _atlas_key(p[0]))
    return pairs


def format_seq(seq):
    """Format a gate sequence for display."""
    if not seq:
        return "I"
    return " ".join(seq)


def _phase_to_omega_power(phi):
    """Convert phase (always an 8th root of unity) to integer k where phi = ω^k.

    ω = e^{iπ/4} is the primitive 8th root of unity.

    That φ lands in μ₈ is forced by determinants (every standard gate's is a
    4th root of unity, and φ_G² · det G_U2 = 1), so this never fires — but it
    is checked rather than assumed, since int() would truncate a non-integer
    k silently and print a plausible wrong symbol.
    """
    from sympy import Abs, arg, pi
    k = simplify(arg(phi) / (pi / 4))
    if simplify(Abs(phi)) != 1 or not k.is_Integer:
        raise ValueError(f"phase {phi} is not an 8th root of unity (k = {k})")
    return int(k)


def format_phase(phi):
    """Format phase as ω^k for text display."""
    if phi is None:
        return "?"
    k = _phase_to_omega_power(phi)
    if k == 0:
        return "1"
    if k == 1:
        return "ω"
    if k == 4 or k == -4:
        return "-1"
    return f"ω^{k}"


def latex_phase(phi):
    """Format phase as ω^k for LaTeX display."""
    if phi is None:
        return "?"
    k = _phase_to_omega_power(phi)
    if k == 0:
        return "1"
    if k == 1:
        return r"\omega"
    if k == 4 or k == -4:
        return "-1"
    return rf"\omega^{{{k}}}"


def write_text_atlas(filename, all_data):
    """Write plain-text atlas file with double cover pairing."""
    # The two φ columns below are the PU(2) words' phases; atlas.tex prints the
    # SU(2) words' instead. The two readings part on exactly the pairs whose own
    # two SU(2) words are not one U(2) operator, so the header counts them off
    # the data rather than hard-coding the 77 and 84 these complement.
    def one_operator(key):
        return sum(1 for _, _, pairs in all_data for a, b in pairs
                   if a[key] is not None and b[key] is not None and _is_zero(a[key] + b[key]))

    n_pairs = sum(len(pairs) for _, _, pairs in all_data)
    differ_depth = n_pairs - one_operator("bfs_phase")
    differ_magic = n_pairs - one_operator("dij_phase")

    with open(filename, "w", encoding="utf-8") as f:
        f.write("BINARY POLYHEDRAL GROUPS ATLAS\n")
        f.write("=" * 80 + "\n")
        f.write("Verified via Conway & Smith (2003) generator theorem.\n")
        f.write("Four synthesis strategies (2×2 grid):\n")
        f.write("  BFS SU(2):      min depth, exact SU(2) match.\n")
        f.write("  Dijkstra SU(2): min magic cost then depth, exact SU(2) match.\n")
        f.write("  BFS PU(2):      min depth, matching up to global phase.\n")
        f.write("  Dijkstra PU(2): min magic cost then depth, matching up to global phase.\n")
        f.write("Phase φ: a word read in the standard gates equals φ · U_q, the row's unitary, with ω = e^(iπ/4) a primitive 8th root of unity.\n")
        f.write("\n")
        f.write("CONVENTIONS:\n")
        f.write("Gate sequences are in operator order (left to right): F X means F · X.\n")
        f.write("In circuit notation this corresponds to applying X first, then F.\n")
        f.write("F = HS† is included as a named generator for uniformity across the three groups.\n")
        f.write("For 2O, it is algebraically redundant with H and S.\n")
        f.write("\n")
        f.write("NOTE ON SU(2) REPRESENTATION:\n")
        f.write("Each standard gate G is symmetrized as G_SU(2) = phase · G_U(2) to ensure det = 1.\n")
        f.write("The dagger (†) denotes the algebraic inverse in this SU(2) representation.\n")
        f.write("For Hermitian gates (X, Z, H): G† physically means applying G with a global -1 phase.\n")
        f.write("\n")
        f.write("NOTE ON THE φ COLUMNS:\n")
        f.write("φ belongs to a word, not to an element: it collects the symmetrization scalars above and\n")
        f.write("the ± recording which member of the pair the word landed on.\n")
        f.write("Both φ columns here are the PU(2) words' -- the columns they sit in. One PU(2) word serves\n")
        f.write("both rows of a pair (the b-row's word cell is a ditto mark), so its φ reads φ on the a-row\n")
        f.write(f"and -φ on the b-row, on all {n_pairs} pairs.\n")
        f.write("atlas.tex prints the SU(2) min-magic words' phases instead: same definition, different word, so the\n")
        f.write(f"two files disagree on the b-rows of the {differ_magic} pairs (min magic) whose own two SU(2) words are not\n")
        f.write(f"one U(2) operator -- and would on {differ_depth} pairs under min depth, a block atlas.tex does not print:\n")
        f.write("this file is the home of the min-depth SU(2) words (Appendix A reprints one 2I pair).\n")
        f.write("Use the φ printed beside the word you actually run.\n")
        f.write("\n")
        f.write("Rows are grouped in antipodal pairs (q, -q) to show the SU(2) double cover, the BFS-shallower member as the a-row.\n")
        f.write("The PU(2) word is printed once per pair (the b-row shows a ditto mark): both elements are one PU(2) element, i.e. one SO(3) rotation.\n")
        f.write("Pairs are ordered by magic cost, then depth, then the word -- the order of atlas.tex.\n")
        f.write("\n")
        f.write("NOTE ON DIJKSTRA vs BFS:\n")
        f.write("For 2I, Dijkstra may prefer longer Clifford-heavy sequences over shorter ones\n")
        f.write("containing multiple Φ gates, reflecting the high cost of non-Clifford resources\n")
        f.write("in fault-tolerant quantum computing.\n\n")

        for group_name, rows, pairs in all_data:
            w = 290
            f.write(f"\n{'#' * w}\n")
            f.write(f"GROUP: {group_name} (Order {len(rows)}, {len(pairs)} antipodal pairs)\n")
            f.write(f"{'#' * w}\n")

            header = (
                f"{'Idx':<4} | "
                f"{'BFS SU(2)':<25} | {'Dep':<3} | "
                f"{'Dij SU(2)':<25} | {'Mag':<3} | {'Dep':<3} | "
                f"{'BFS PU(2)':<25} | {'Dep':<3} | {'φ':<6} | "
                f"{'Dij PU(2)':<25} | {'Mag':<3} | {'Dep':<3} | {'φ':<6} | "
                f"{'Quaternion':<50} | Unitary"
            )
            f.write(header + "\n")
            f.write("-" * w + "\n")

            for pi, (primary, antipodal) in enumerate(pairs, 1):
                # Primary row: full data including PU(2) columns
                f.write(
                    f"{pi:<3}a | "
                    f"{format_seq(primary['bfs_seq']):<25} | {primary['bfs_depth']:<3} | "
                    f"{format_seq(primary['dij_seq']):<25} | {primary['dij_magic']:<3} | {primary['dij_depth']:<3} | "
                    f"{format_seq(primary['u2_seq']):<25} | {primary['u2_depth']:<3} | {format_phase(primary['u2_phase']):<6} | "
                    f"{format_seq(primary['dij_u2_seq']):<25} | {primary['dij_u2_magic']:<3} | {primary['dij_u2_depth']:<3} | {format_phase(primary['dij_u2_phase']):<6} | "
                    f"{primary['quat'].pretty():<50} | {primary['quat'].pretty_matrix()}\n"
                )
                # Antipodal row: PU(2) columns show ditto marks
                f.write(
                    f"{pi:<3}b | "
                    f"{format_seq(antipodal['bfs_seq']):<25} | {antipodal['bfs_depth']:<3} | "
                    f"{format_seq(antipodal['dij_seq']):<25} | {antipodal['dij_magic']:<3} | {antipodal['dij_depth']:<3} | "
                    f"{'\"':<25} | {'\"':<3} | {format_phase(antipodal['u2_phase']):<6} | "
                    f"{'\"':<25} | {'\"':<3} | {'\"':<3} | {format_phase(antipodal['dij_u2_phase']):<6} | "
                    f"{antipodal['quat'].pretty():<50} | {antipodal['quat'].pretty_matrix()}\n"
                )
                f.write("\n")

    print(f"  Text atlas written to {filename}")


def tex_seq(seq):
    """A word for atlas.tex: symmetrized gates in bold, Φ plain, 𝟙 for the empty word.

    The bold letters are the thesis's own notation for the SU(2) generators
    (Section 3.2.1): X, Z, H, S, F carry a phase into SU(2) and print bold;
    Φ has det 1 already and prints plain. A reader who wants the standard
    gates reads the letters plain and picks up the row's φ.
    """
    if not seq:
        return r"$\Id$"
    parts = []
    for name in seq:
        base = name.rstrip("†")
        dag = r"^{\dagger}" if name.endswith("†") else ""
        letter = r"\Phi" if base == "Φ" else r"\mathbf{" + base + "}"
        parts.append(letter + dag)
    return "$" + "".join(parts) + "$"


def tex_axis(axis):
    """The pair's axis as a vertical 3-vector at unit scale, or --- for the identity."""
    if axis is None:
        return "---"
    rows = " \\\\ ".join(_atlas_unit_tex(sign, name) for sign, name in axis)
    return r"$\begin{psmallmatrix} " + rows + r" \end{psmallmatrix}$"


_BLOCK_TITLES = {"bfs": "Min depth (BFS)", "dij": r"Min $\Phi$ (Dijkstra)", "both": r"Min depth and min $\Phi$"}


def _printed_block(rows):
    """The one word block atlas.tex prints for a group, and its title.

    Every group prints its min-magic (Dijkstra) words only; the title says
    whether they are also the min-depth ones -- they are for 2T and 2O, on
    every row, and not for 2I, which differs on 17 rows (verify_atlas_layout
    (1)). 2I's min-depth words print in the preamble's sample row
    (verify_sample_row) and live in atlas.txt.
    """
    coincide = all(r["bfs_seq"] == r["dij_seq"] for r in rows)
    return ["dij"], {"dij": _BLOCK_TITLES["both" if coincide else "dij"]}


# What follows the word blocks, as (title, symbols, column spec) -- atlas.tex's
# per-pair columns, and Table B.1's single quaternion. The tail is data because
# the two tables share the head and the colspec below and differ only here:
# Appendix B keeps no pair columns and prints each row's OWN signed quaternion,
# so its symbol line reads $q$ where the atlas's reads $\pm q$.
_ATLAS_TAIL = (
    (r"$R_{\hat{n}}(\theta)$", ("$n$", r"$\theta$"), "c c"),
    ("Quaternion", (r"$\pm q$",), "l"),
    ("Unitary", (r"$\pm U_q$",), "l"),
)
_B1_TAIL = (("Quaternion", ("$q$",), "l"),)


def _atlas_colspec(cols, tail=_ATLAS_TAIL):
    r"""The column spec: `\#`, four columns per word block, then the tail's own."""
    return "r " + "c c l c " * len(cols) + " ".join(spec for _title, _syms, spec in tail)


def _atlas_head(cols, titles, tail=_ATLAS_TAIL):
    r"""The two header lines over word blocks `cols`, with the rules between them.

    Names above, symbols below. A tail entry of more than one column gets a
    `\multicolumn` and its own `\cmidrule`, as a word block does; a
    one-column entry spends both lines on itself ("Quaternion" over
    "$\pm q$"). Table B.1 is this head with `_B1_TAIL`, which is what makes
    the two tables one format and stops them drifting apart.
    """
    spans, rules, names, c0 = [], [], [], 2
    for col in cols:
        spans.append(rf"\multicolumn{{4}}{{c}}{{{titles[col]}}}")
        rules.append(rf"\cmidrule(lr){{{c0}-{c0 + 3}}}")
        names.append(r"Depth & $\Phi$ & Sequence & $\varphi$")
        c0 += 4
    for title, syms, _spec in tail:
        spans.append(title if len(syms) == 1 else rf"\multicolumn{{{len(syms)}}}{{c}}{{{title}}}")
        if len(syms) > 1:
            rules.append(rf"\cmidrule(lr){{{c0}-{c0 + len(syms) - 1}}}")
        names.append(" & ".join(syms))
        c0 += len(syms)
    return (r"\# & " + " & ".join(spans) + r" \\" + "\n"
            + "".join(rules) + "\n"
            + " & " + " & ".join(names) + r" \\")


# The B mark. A row of 2I whose min-depth word the atlas does not
# print -- a row Table B.1 carries -- gets a superscript B at the left of its
# `#`, a link to that table. \llap makes it zero-width: it hangs in the column's
# own \tabcolsep, so the `#` column is no wider and the 2I table, already at
# \textwidth, no wider either; and left of the key it stays apart from the
# row's own a or b. Behind it a zero-size anchor, raised so a viewer landing on
# it shows the row, is what Table B.1's `#` links back to. Per row, not per
# pair (54a is marked, 54b is not), and 2I only, the group Table B.1 is of.
_B1_LABEL = "tab:bfs-vs-dijkstra"
_B1_MODE = "2I"
_MARK_RE = re.compile(
    r"\\llap\{\\textsuperscript\{\\hyperref\[([^\]]*)\]\{B\}\}\}"
    r"\\raisebox\{2ex\}\[0pt\]\[0pt\]\{\\hypertarget\{([^}]*)\}\{\}\}")


def _atlas_anchor(mode, key):
    """The hypertarget name of a marked atlas row: what Table B.1's `#` links to."""
    return f"atlas:{mode}:{key}"


def _atlas_key_cell(key, mark):
    r"""A row's `#` cell: the key alone, or, when `mark` names the group, behind the B mark and the anchor."""
    if mark is None:
        return key
    return (rf"\llap{{\textsuperscript{{\hyperref[{_B1_LABEL}]{{B}}}}}}"
            rf"\raisebox{{2ex}}[0pt][0pt]{{\hypertarget{{{_atlas_anchor(mark, key)}}}{{}}}}{key}")


def _read_atlas_key(cell, mode, at):
    """A typeset `#` cell back to (key, marked), the mark's link and anchor checked on the way."""
    m = _MARK_RE.match(cell)
    key = cell[m.end():] if m else cell
    assert re.fullmatch(r"\d+[ab]", key), f"{at}: unexpected row label {cell!r}"
    if m:
        assert m.groups() == (_B1_LABEL, _atlas_anchor(mode, key)), (
            f"{at}: the B mark links to {m.group(1)!r} and anchors {m.group(2)!r}; "
            f"wanted {_B1_LABEL!r} and {_atlas_anchor(mode, key)!r}"
        )
    return key, bool(m)


def _read_b1_key(cell, at):
    """Table B.1's `#` cell back to its key, the link to the atlas row checked."""
    m = re.fullmatch(r"\\hyperlink\{([^}]*)\}\{(\d+[ab])\}", cell)
    assert m, f"{at}: unexpected `#` cell {cell!r}"
    assert m.group(1) == _atlas_anchor(_B1_MODE, m.group(2)), f"{at}: `#` {m.group(2)} links to {m.group(1)!r}"
    return m.group(2)


def _word_cells(row, col):
    """A row's four cells of one word block: depth, Φ count, word, phase."""
    return [str(row[f"{col}_depth"]), str(row[f"{col}_magic"]), tex_seq(row[f"{col}_seq"]),
            f"${latex_phase(row[f'{col}_phase'])}$"]


def _pair_cells(first, second):
    """The four cells a pair shares, spanning its two rows: axis, angle, ±q, ±U_q with the first row's sign."""
    qp = first["quat"]
    _rep, axis, deg = pair_rotation(first, second)
    angle = "$0$" if deg == 0 else f"${deg}^\\circ$"
    return [rf"\multirow{{2}}{{*}}{{{cell}}}" for cell in (
        tex_axis(axis), angle, f"$\\pm {qp.latex_quat()}$", f"$\\pm {qp.latex_matrix()}$")]


def atlas_pair_tex(pi, first, second, cols, mark=None):
    """The two typeset lines of pair `pi` with the word blocks `cols`: atlas.tex's rows,
    and the preamble's sample row when `cols` names both syntheses. With `mark`, the
    group's name, a row whose two words differ gets the B mark on its `#`
    (`_atlas_key_cell`) -- atlas.tex's case, never the sample row's, which prints
    both words itself."""
    assert mark is None or "bfs" not in cols, "a row printing both words needs no mark"
    a_row = [c for col in cols for c in _word_cells(first, col)] + _pair_cells(first, second)
    b_row = [c for col in cols for c in _word_cells(second, col)]

    def key(letter, row):
        return _atlas_key_cell(f"{pi}{letter}", mark if row["bfs_seq"] != row["dij_seq"] else None)

    return (f"  {key('a', first)} & " + " & ".join(a_row) + " \\\\*\n"
            + f"  {key('b', second)} & " + " & ".join(b_row) + " & & & & \\\\\n")


def write_latex_tables(filename, all_data):
    """Write the LaTeX atlas: one row per group element, pairs kept together.

    A row is a group element -- its own min-magic SU(2) word (Dijkstra) with
    depth, Φ count and phase φ; for 2T and 2O that is also its min-depth
    word, the two coinciding on every row, and the block's title says so
    (`_printed_block`) -- and the two rows of an antipodal pair (q, -q)
    print together, BFS-shallower member first, with the pair's shared data
    once, spanning both: the Bloch rotation R_n̂(θ) and ±q, ±U_q with the
    common scalar out front, the sign the first row's. The first row is the
    projective answer (verify_atlas_layout (2)), which is why nothing from
    the PU(2) runs is printed; 2I's min-depth words are in atlas.txt, in
    Appendix B's Table B.1 where the two syntheses differ -- those 17 rows
    carry the B mark (`_atlas_key_cell`) -- and, for one pair, in the
    preamble's sample row. Headers carry the conventions;
    the caption is one line and states no finding. main writes this to a
    staging path and lets verify_atlas_tex read it back before the file
    takes atlas.tex's place.
    """
    group_macro = {"2T": r"\TwoT", "2O": r"\TwoO", "2I": r"\TwoI"}
    with open(filename, "w", encoding="utf-8") as f:
        f.write("% Auto-generated binary polyhedral group tables\n")
        f.write("% Include this file in your document with \\input{...}\n")
        f.write("% Requires: \\usepackage{longtable, booktabs, multirow, hyperref}, mathtools\n")
        f.write("% (psmallmatrix), and the thesis macros \\Id, \\PU, \\TwoT, \\TwoO, \\TwoI.\n\n")
        f.write("% LAYOUT: one row per group element. The two rows of an antipodal pair (q, -q)\n")
        f.write("% print together (\\\\* forbids a page break between them), the BFS-shallower\n")
        f.write("% member first -- the member Appendix F's gate-noise study runs, and the pair's\n")
        f.write("% projective answer: it minimises depth and (Phi, depth) over the pair, which is\n")
        f.write("% why the PU(2) synthesizers get no columns (they return its words up to phase;\n")
        f.write("% verify_atlas_layout asserts all of this before this file is written).\n")
        f.write("% Per row: the element's own min-magic SU(2) word (Dijkstra) with Depth, Phi\n")
        f.write("% count, Sequence and phase; for 2T and 2O it is also the min-depth word, on\n")
        f.write("% every row, and the block's title says so. 2I's min-depth (BFS) words are in\n")
        f.write("% atlas.txt, in differing_rows.tex where they differ -- those rows carry a\n")
        f.write("% superscript B left of their #, a link to Table B.1, and an anchor its #\n")
        f.write("% links back to -- and one pair in the preamble's sample row. Per pair,\n")
        f.write("% spanning both rows:\n")
        f.write("% the Bloch rotation R_n(theta) and +-q, +-U_q, the sign the first row's.\n")
        f.write("% Pairs are ordered by magic cost, then depth, then the word, so Table A.3\n")
        f.write("% opens with Table A.1 (verify_atlas_nesting). atlas.txt keeps the full grid.\n\n")
        f.write("% CONVENTIONS:\n")
        f.write("% Sequences are in operator order: $FX$ means $F \\cdot X$, i.e. $X$ is applied first.\n")
        f.write("% Bold gates are the symmetrized SU(2) generators, $\\mathbf{G} = \\varphi_\\mathsf{G} G$\n")
        f.write("% with $\\det = 1$; $\\Phi$ is in SU(2) as it stands and prints plain. The dagger is\n")
        f.write("% the inverse in SU(2), so $\\mathbf{X}^\\dagger = -\\mathbf{X}$ and likewise $\\mathbf{Z}$, $\\mathbf{H}$.\n")
        f.write("% $F = HS^\\dagger$ is a named generator in all three groups (depth 1).\n")
        f.write("% $\\varphi$ is per row: the row's word, read in the standard gates, equals\n")
        f.write("% $\\varphi$ times the row's unitary; $\\varphi = \\omega^k$, $\\omega = e^{i\\pi/4}$.\n")
        f.write("% $R_{\\hat n}(\\theta)$: the pair's Bloch rotation, $\\theta = 2\\arccos w$ about\n")
        f.write("% $n = -(z, y, x)$ for the member with $w \\ge 0$ (Section 2.4's convention); $n$ is\n")
        f.write("% printed unnormalized, $\\hat n = n/|n|$, half-turn axes signed to a positive first nonzero entry.\n")
        f.write("% verify_atlas_rotations checks every pair against $U_q$'s conjugation action.\n")
        f.write("% Quaternion and unitary carry their common scalar out front; $\\tau$ is the golden\n")
        f.write("% ratio and $\\sigma$ its inverse.\n\n")

        for group_name, rows, pairs in all_data:
            mode = group_name.split()[-1]
            safe_label = group_name.replace(" ", "-").replace("(", "").replace(")", "").lower()
            cols, titles = _printed_block(rows)
            colspec = _atlas_colspec(cols)
            head = _atlas_head(cols, titles)
            title = group_name.rsplit(" ", 1)[0] + " $" + group_macro[mode] + "$"
            # One line, and no finding: the conventions a reader needs at the table.
            caption = (title + r". Gates in operator order; bold means symmetrized "
                       r"(Section~\ref{sec:workinginsu2}); for $\PU(2)$, read the $a$-row.")

            f.write(f"% --- {group_name} ---\n")
            f.write(rf"\begin{{longtable}}{{{colspec}}}" + "\n")
            f.write(r"\caption{" + caption + r"}\label{tab:" + safe_label + r"}\\" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(head + "\n")
            f.write(r"\midrule" + "\n")
            f.write(r"\endfirsthead" + "\n")
            f.write(r"\midrule" + "\n")
            f.write(head + "\n")
            f.write(r"\midrule" + "\n")
            f.write(r"\endhead" + "\n")

            for pi, (first, second) in enumerate(pairs, 1):
                f.write(atlas_pair_tex(pi, first, second, cols, mark=mode if mode == _B1_MODE else None))
                f.write(r"  \addlinespace[3pt]" + "\n")   # 12 pairs on a first page, 13 after

            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{longtable}" + "\n\n")

    print(f"  LaTeX tables written to {filename}")


_B1_COLS = ("bfs", "dij")   # both syntheses, the sample row's blocks and titles
# LaTeX's default, which is what the atlas runs at; pinned rather than inherited
# because Appendix B has to stay one page. Ten columns at \normalsize come to a
# natural 430.40pt against a \textwidth of 452.97pt, so there are 22.57pt spare
# -- a pt of \tabcolsep costs 2 x 10 = 20pt of that, which is why it is not 7.
_B1_TABCOLSEP = "6pt"

# Appendix A's shape, one table down: the same colspec builder, the same two
# header lines, the same block titles, and a one-column tail (`_B1_TAIL`).
# No \large -- the atlas sets none, and at ten columns it does not fit; matching
# the atlas is what lets a reader look rows up.
#
# \arraystretch is the one place this table cannot follow the atlas. Both print
# the quaternion as \tfrac{1}{2}(...), 16.0pt of stacked fraction, but the atlas
# prints it once per PAIR, so its rows come 30.1pt apart and the fractions clear
# each other by 14pt. Here every element carries its own, and at the document's
# 1.05 the rows are 13.56pt apart: one row's denominator and the next row's
# numerator overlap by 2.5pt of bounding box and touch as ink. 1.2 buys a 16.26pt
# row and ~2.7pt of air, and spends 51pt of the ~120pt the page has spare below
# the caption. Measured, not guessed, and the ceiling is the fit: at 1.3 the air
# reads loose for a data table, and the page still has to end above 772pt.
_B1_HEAD = "\n".join([
    r"\begin{table}[H]",
    r"\centering",
    r"\renewcommand{\arraystretch}{1.2}",
    r"\setlength{\tabcolsep}{" + _B1_TABCOLSEP + "}",
    r"\begin{tabular}{" + _atlas_colspec(_B1_COLS, _B1_TAIL) + "}",
    r"\toprule",
    _atlas_head(_B1_COLS, _BLOCK_TITLES, _B1_TAIL),
    r"\midrule",
]) + "\n"

_B1_FOOT = "\n".join([
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{%s}",
    r"\label{" + _B1_LABEL + "}",
    r"\end{table}",
]) + "\n"


def _b1_caption(rows):
    r"""Table B.1's caption, its numbers read off the data it captions.

    States the finding -- unlike the atlas captions, which state none: the
    17 rows are the exhibit for one sentence of Section 4.1.2, and the split
    between the rows Dijkstra improves and the rows it merely rearranges is
    the one thing here that section does not print. It also has to hand the
    reader the two columns that make this a lookup table: `#`, the element's
    row of Table A.3, and phi, which is the WORD's phase and not the
    element's -- the two blocks disagree on it on all 17 rows, asserted here
    because the caption says so. The count is named `\Phi`-count, never
    "magic cost", the name the code keeps and the thesis does not print;
    `verify_differing_rows_tex` refuses a file carrying "Mag".
    """
    differing = [r for r in rows if r["bfs_seq"] != r["dij_seq"]]
    lowered = [r for r in differing if r["bfs_magic"] > r["dij_magic"]]
    rearranged = [r for r in differing if r["bfs_magic"] == r["dij_magic"]]
    spent = sorted({r["bfs_magic"] for r in lowered}, reverse=True)
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    assert len(spent) == 2, f"the caption says '{words[spent[0]]} or ...' and reads {spent} BFS Phi counts"
    disagree = [r for r in differing if not _is_zero(r["bfs_phase"] - r["dij_phase"])]
    assert len(disagree) == len(differing), (
        f"the caption says the two blocks disagree on phi on all {len(differing)} rows, and "
        f"{len(differing) - len(disagree)} share one; that is the reason this table prints phi per block"
    )
    return (
        rf"The {len(differing)} elements of $\TwoI$ where the two synthesizers return different gate "
        rf"sequences, in the format of Appendix~\ref{{app:atlas}}: "
        rf"$\#$ is the element's row of Table~\ref{{tab:binary-icosahedral-group-2i}}, its pair number "
        rf"and $a$ or $b$, and $q$ its own signed quaternion, which that table prints $\pm$, once per "
        rf"pair. BFS minimizes depth; Dijkstra minimizes $\Phi$-count, then depth. The blocks disagree "
        rf"on $\varphi$ on all {len(disagree)} rows, so read the $\varphi$ beside the sequence you run. "
        rf"Rows run by BFS $\Phi$-count, then by $\#$, so the trade is read top to bottom: on the first "
        rf"{len(lowered)} Dijkstra lowers it, from {words[spent[0]]} or {words[spent[1]]} $\Phi$ gates "
        rf"down to one, sometimes at the cost of a longer sequence; on the last "
        rf"{words[len(rearranged)]} it is already one under BFS, "
        rf"so the two sequences differ only in arrangement, at equal depth. In every case the Dijkstra "
        rf"sequence contains exactly one $\Phi$."
    )


def write_differing_rows(filename, rows):
    r"""Write Appendix B: the float around `table_b1_tex`'s rows.

    One page, one table, caption included. `[H]` because Section B.1 is that
    page: left to float, the table would carry itself past the text it belongs
    to. `verify_differing_rows` checks the data and the rendering before this
    is written, and `verify_differing_rows_tex` reads the file back.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write("% Auto-generated: Appendix B, the rows of 2I where BFS and Dijkstra differ.\n")
        f.write("% Include this file in your document with \\input{...}.\n")
        f.write("% Requires: \\usepackage{booktabs, float, hyperref} and the thesis macro \\TwoI.\n\n")
        f.write("% LAYOUT: atlas.tex's, minus the per-pair columns, so a row carries over.\n")
        f.write("% One row per element, keyed by its atlas row -- # is the pair number and a/b\n")
        f.write("% of Table A.3, and a link to that row, which carries a superscript B linking\n")
        f.write("% back here -- then two word blocks, min depth (BFS) and min Phi (Dijkstra),\n")
        f.write("% of Depth, Phi count, Sequence and phase, then the element's own signed\n")
        f.write("% quaternion (Table A.3 prints +-q once per pair, which is no use here: the two\n")
        f.write("% members of a pair are rows apart). Rows run by BFS Phi count, so the caption's\n")
        f.write("% 13-and-4 split reads down the page, then by # inside a tier, so a key can be\n")
        f.write("% found by eye.\n\n")
        f.write("% CONVENTIONS: atlas.tex's, and the same emitters -- sequences in operator order,\n")
        f.write("% bold for the symmetrized SU(2) gates, the dagger the SU(2) inverse, $\\varphi$\n")
        f.write("% the WORD's phase with $\\varphi = \\omega^k$, $\\omega = e^{i\\pi/4}$. Phase is per\n")
        f.write("% block here and the two blocks disagree on it on every row, so the atlas's\n")
        f.write("% $\\varphi$ (Dijkstra's) is the wrong one for the BFS word. atlas.txt keeps the\n")
        f.write("% full 2x2 synthesis grid these two blocks are two quarters of.\n\n")
        f.write(_B1_HEAD)
        f.write(table_b1_tex(rows))
        f.write(_B1_FOOT % _b1_caption(rows))
    print(f"  Appendix B table written to {filename}")


def verify_differing_rows_tex(filename, rows):
    r"""Read Appendix B's table back: the skeleton, the body, the caption.

    The same gate as verify_atlas_tex, one size down. The body must be
    `table_b1_tex`'s output character for character -- it is written from it,
    so a mismatch means the write, not the data -- and the header, the label
    and the caption's counts must survive around it.
    """
    tex = filename.read_text(encoding="utf-8")
    body = table_b1_tex(rows)
    for piece in (_B1_HEAD, body, rf"\label{{{_B1_LABEL}}}", _b1_caption(rows)):
        assert piece in tex, f"{filename.name}: read back without {piece.splitlines()[0][:60]!r}"
    start = tex.index(_B1_HEAD) + len(_B1_HEAD)
    assert tex[start:start + len(body)] == body, f"{filename.name}: the rows are not flush against \\midrule"
    assert "Mag" not in tex, f"{filename.name}: 'Mag' is the retired name; the thesis prints $\\Phi$-count"
    print(f"  ✓ Appendix B: {body.count(chr(10))} rows read back, caption and label in place")


# Reading atlas.tex back: the printed symbols as numbers, by tables of this
# layer's own -- `\tau` IS (1 + √5)/2 here, whatever _ATLAS_UNIT_TEX says, and
# ω IS (1 + i)/√2, whatever _phase_to_omega_power computes.
_TEX_UNIT_VALUE = {"0": Rational(0), "1": Rational(1), r"\tau": TAU, r"\sigma": SIGMA}
_UNIT_VALUE_BY_NAME = {"0": Rational(0), "1": Rational(1), "tau": TAU, "sigma": SIGMA}
_TEX_SCALE = {"": Rational(1), r"\tfrac{1}{2}": Rational(1, 2), r"\tfrac{1}{\sqrt{2}}": 1 / sqrt(2)}
_TEX_SCALE_RE = r"(|\\tfrac\{1\}\{2\}|\\tfrac\{1\}\{\\sqrt\{2\}\})"
_OMEGA = (1 + symI) / SQRT2


def _tex_unit_value(tok, where):
    """`0`, `1`, `\\tau`, `\\sigma`, a negative braced as `{-\\tau}`, as a number."""
    m = re.fullmatch(r"(\{-)?(0|1|\\tau|\\sigma)(\})?", tok.strip())
    assert m and bool(m.group(1)) == bool(m.group(3)), f"{where}: unexpected component {tok!r}"
    return -_TEX_UNIT_VALUE[m.group(2)] if m.group(1) else _TEX_UNIT_VALUE[m.group(2)]


def _tex_complex_value(tok, where):
    """A unitary entry as _atlas_complex_tex prints it, as a number.

    Three shapes and nothing else: a real part alone (`0`, `-\\tau`), an
    imaginary part alone (`i`, `-i\\sigma`), or both joined by `+` or `-`
    (`\\tau + i\\sigma`, `1 - i`). The join is required -- `\\tau i\\sigma`
    is a dropped sign, which is one of the defects this read-back exists
    to catch, not an entry.
    """
    s = tok.strip()
    m = re.fullmatch(r"(-?)(0|1|\\tau|\\sigma)", s)
    if m:
        return (-1 if m.group(1) else 1) * _TEX_UNIT_VALUE[m.group(2)]
    m = re.fullmatch(r"(-?)i(\\tau|\\sigma)?", s)
    if m:
        return (-1 if m.group(1) else 1) * symI * (_TEX_UNIT_VALUE[m.group(2)] if m.group(2) else 1)
    m = re.fullmatch(r"(-?)(1|\\tau|\\sigma)\s*([+-])\s*i(\\tau|\\sigma)?", s)
    assert m, f"{where}: unexpected matrix entry {tok!r}"
    re_sign, re_unit, im_sign, im_unit = m.groups()
    return ((-1 if re_sign else 1) * _TEX_UNIT_VALUE[re_unit]
            + (-1 if im_sign == "-" else 1) * symI * (_TEX_UNIT_VALUE[im_unit] if im_unit else 1))


def _split_cells(line):
    """A table row's cells: split on `&` at brace depth 0 -- the multirow cells hold a psmallmatrix with `&` of its own."""
    cells, cur, depth = [], [], 0
    for ch in line:
        depth += (ch == "{") - (ch == "}")
        if ch == "&" and depth == 0:
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


def _unwrap(cell, pattern, where):
    m = re.fullmatch(pattern, cell, re.S)
    assert m, f"{where}: unexpected cell {cell!r}"
    return m.groups()


def _verify_atlas_skeleton(body, ncols, thesis, where):
    r"""One section's longtable around its rows: the parts the data rows do not cover.

    Checks relations, not strings -- a skeleton compared with the emitter's
    own strings could only pass. The colspec must be as wide as the rows;
    the head printed after \endfirsthead must be the one printed after
    \endhead; each head line must fill the columns exactly (a \multicolumn
    counting its span), and the \cmidrule under each spanning title must
    cover the span's columns and no others; the caption's \ref targets and
    the table's own \label must be the thesis's. Returns the body between
    \endhead and \bottomrule.
    """
    m = re.fullmatch(
        r"\\begin\{longtable\}\{([^}]*)\}\n"
        r"\\caption\{([^\n]*)\}\\label\{([^}]*)\}\\\\\n"
        r"\\toprule\n(.*?)\n\\midrule\n\\endfirsthead\n"
        r"\\midrule\n(.*?)\n\\midrule\n\\endhead\n"
        r"(.*)\\bottomrule\n\\end\{longtable\}\n\n",
        body, re.S)
    assert m, f"{where}: the longtable is not in the shape the read-back reads (colspec, caption+label, two heads, rows, \\bottomrule)"
    colspec, caption, label, head_first, head, rows = m.groups()
    assert re.fullmatch(r"[rcl ]+", colspec) and len(colspec.split()) == ncols, (
        f"{where}: colspec {colspec!r} has {len(colspec.split())} columns, the rows have {ncols}"
    )
    assert head_first == head, f"{where}: the first-page head differs from the head of the pages after"
    assert rf"\ref{{{label}}}" in thesis, f"{where}: the thesis never refers to \\label{{{label}}}"
    for ref in re.findall(r"\\ref\{([^}]*)\}", caption):
        assert rf"\label{{{ref}}}" in thesis, f"{where}: the caption refers to {ref}, which the thesis does not label"
    names, rules, symbols = head.split("\n")
    spans, pos = [], 1
    for cell in _split_cells(names.removesuffix(r" \\")):
        k = re.fullmatch(r"\\multicolumn\{(\d+)\}\{c\}\{.*\}", cell)
        width = int(k.group(1)) if k else 1
        if k:
            spans.append((pos, pos + width - 1))
        pos += width
    assert pos - 1 == ncols, f"{where}: the head's name line fills {pos - 1} columns of {ncols}"
    assert len(_split_cells(symbols.removesuffix(r" \\"))) == ncols, f"{where}: the head's symbol line does not fill {ncols} columns"
    assert [tuple(map(int, r)) for r in re.findall(r"\\cmidrule\(lr\)\{(\d+)-(\d+)\}", rules)] == spans, (
        f"{where}: the \\cmidrules {rules!r} do not underline the spanning titles {spans}"
    )
    return rows


def _check_word_cells(cells, row, col, at):
    """Four typeset cells of one word block against the row: depth, Φ, word, phase."""
    depth, magic, seq, phase = cells
    assert (int(depth), int(magic)) == (row[f"{col}_depth"], row[f"{col}_magic"]), (
        f"{at}: prints depth {depth}, Φ {magic}; the row has {row[f'{col}_depth']}, {row[f'{col}_magic']}"
    )
    assert _tex_word(seq, at) == row[f"{col}_seq"], f"{at}: prints {seq}, the row's word is {row[f'{col}_seq']}"
    m = re.fullmatch(r"\$(?:(1)|(-1)|(\\omega)|\\omega\^\{(-?\d)\})\$", phase)
    assert m, f"{at}: unexpected phase cell {phase!r}"
    k = 0 if m.group(1) else 4 if m.group(2) else 1 if m.group(3) else int(m.group(4))
    assert _is_zero(row[f"{col}_phase"] - _OMEGA ** (k % 8)), (
        f"{at}: prints φ = ω^{k}, the row's phase is {row[f'{col}_phase']}"
    )


def _check_pair_cells(cells, first, second, at):
    """An a-row's four \\multirow cells against the pair: axis, angle, ±q, ±U_q."""
    _rep, axis, deg = pair_rotation(first, second)
    ax_tex, deg_tex, q_tex, u_tex = (_unwrap(c, r"\\multirow\{2\}\{\*\}\{(.*)\}", at)[0] for c in cells)
    if axis is None:
        assert ax_tex == "---", f"{at}: identity pair prints axis {ax_tex!r}"
    else:
        (inner,) = _unwrap(ax_tex, r"\$\\begin\{psmallmatrix\} (.*) \\end\{psmallmatrix\}\$", at)
        got = [_tex_unit_value(t, at) for t in inner.split(r"\\")]
        want = [sign * _UNIT_VALUE_BY_NAME[name] for sign, name in axis]
        assert len(got) == 3 and all(_is_zero(g - w) for g, w in zip(got, want)), (
            f"{at}: prints axis {inner!r}, the pair's is {want}"
        )
    assert deg_tex == ("$0$" if deg == 0 else f"${deg}^\\circ$"), f"{at}: prints angle {deg_tex!r}, the pair's is {deg}°"
    scale, comps = _unwrap(q_tex, r"\$\\pm " + _TEX_SCALE_RE + r"\((.*)\)\$", at)
    vals = [_TEX_SCALE[scale] * _tex_unit_value(t, at) for t in comps.replace("\\,", "").split(",")]
    assert len(vals) == 4 and Quaternion(*vals) == first["quat"], (
        f"{at}: prints quaternion {q_tex!r}, the a-row's is {first['quat']}"
    )
    scale, inner = _unwrap(u_tex, r"\$\\pm " + _TEX_SCALE_RE + r"\\begin\{psmallmatrix\} (.*) \\end\{psmallmatrix\}\$", at)
    entries = [[_TEX_SCALE[scale] * _tex_complex_value(e, at) for e in r_.split("&")] for r_ in inner.split(r"\\")]
    U = first["quat"].to_unitary()
    assert [len(r_) for r_ in entries] == [2, 2] and all(
        _is_zero(entries[i][j] - U[i, j]) for i in range(2) for j in range(2)
    ), f"{at}: prints unitary {u_tex!r}, the a-row's is {first['quat'].pretty_matrix()}"


def verify_atlas_tex(filename, all_data):
    r"""
    Read the written atlas.tex back and check every printed cell against the data.

    The verifiers above check the rows; nothing else checks the typesetting
    -- a wrong TeX string for τ, a sign dropped from a matrix entry, the
    axis rows in the wrong order, a phase printed as the wrong power of ω --
    and the typesetting is all the reader sees. So the file is written to a
    staging path, parsed back here, and only then moved into place (main):
    the three sections in the data's order and no other; per section, the
    skeleton around the rows (`_verify_atlas_skeleton`) and a body holding
    nothing but the rows read and one spacer per pair; per row, its label
    and the `\\*` on each a-row, depth, Φ, word and φ; per pair, the axis,
    angle, ±q and ±U_q. The printed symbols are read as numbers by tables
    of this function's own, so the check does not share the emitter's
    (`\tau` is (1 + √5)/2 here, whatever _ATLAS_UNIT_TEX says, and ω^k is
    compared as the number (1 + i)^k/√2^k, not through the converter that
    printed k), and the quaternion and unitary are compared as exact field
    elements. A cell the parser cannot read is a failure, not a cell to skip
    -- which includes the raw components latex_quat falls back to when no
    scale factor fits.
    """
    tex = Path(filename).read_text(encoding="utf-8")
    thesis = THESIS_TEX.read_text(encoding="utf-8")
    parts = re.split(r"^% --- (.+?) ---\n", tex, flags=re.M)
    names, bodies = parts[1::2], parts[2::2]
    assert names == [name for name, _rows, _pairs in all_data], (
        f"atlas.tex: sections {names}; the data has {[name for name, _, _ in all_data]}, in that order"
    )
    n_rows, marked = 0, {}
    for (group_name, rows, pairs), body in zip(all_data, bodies):
        mode = group_name.split()[-1]
        where = f"atlas.tex {mode}"
        cols, _titles = _printed_block(rows)
        ncells = 4 * len(cols) + 4                  # after the row label: the word block(s), then axis, angle, ±q, ±U_q
        table = _verify_atlas_skeleton(body, ncells + 1, thesis, where)
        lines = re.findall(r"^  ([^ &]+) & (.*?) \\\\(\*?)$", table, flags=re.M)
        spacers = table.count("  \\addlinespace[3pt]\n")
        nonblank = sum(1 for line in table.splitlines() if line.strip())
        assert len(lines) == len(rows) and spacers == len(pairs) and nonblank == len(lines) + spacers, (
            f"{where}: {nonblank} lines in the body -- {len(lines)} rows read for {len(rows)} elements, "
            f"{spacers} spacers for {len(pairs)} pairs; a line of any other kind is an error"
        )
        for pi, (first, second) in enumerate(pairs, 1):
            for member, row, star in ((f"{pi}a", first, "*"), (f"{pi}b", second, "")):
                label, cells_tex, got_star = lines.pop(0)
                at = f"{where} row {member}"
                key, is_marked = _read_atlas_key(label, mode, at)
                assert (key, got_star) == (member, star), f"{at}: found row {key}{got_star}"
                assert is_marked == (mode == _B1_MODE and row["bfs_seq"] != row["dij_seq"]), (
                    f"{at}: {'carries' if is_marked else 'lacks'} the B mark, but its min-depth word "
                    f"{'is' if is_marked else 'is not'} the printed one"
                )
                if is_marked:
                    marked.setdefault(mode, []).append(key)
                cells = _split_cells(cells_tex)
                assert len(cells) == ncells, f"{at}: {len(cells)} cells, not {ncells}"
                for col in cols:
                    _check_word_cells(cells[:4], row, col, f"{at} [{col}]")
                    cells = cells[4:]
                if member.endswith("b"):
                    assert cells == [""] * 4, f"{at}: the shared cells are not empty on the b-row"
                else:
                    _check_pair_cells(cells, first, second, at)
                n_rows += 1
    # The marks are the rows of Table B.1, no more and no fewer, and the
    # preamble says what they are.
    b1_rows = {name.split()[-1]: r for name, r, _p in all_data}[_B1_MODE]
    b1_keys = [_read_b1_key(line.strip().partition(" & ")[0], "Table B.1") for line in table_b1_tex(b1_rows).splitlines()]
    assert sorted(marked.get(_B1_MODE, [])) == sorted(b1_keys), (
        f"atlas.tex: the B marks {marked} are not Table B.1's rows {sorted(b1_keys)}"
    )
    bounds = [thesis.find(s) for s in (r"\chapter{Binary Polyhedral Groups Atlas}", r"\input{../code/atlas.tex}")]
    assert -1 not in bounds and bounds[0] < bounds[1], "Appendix A's chapter line or its \\input of atlas.tex is not where the read-back looks"
    preamble = _strip_tex_comments(thesis[bounds[0]:bounds[1]]).splitlines()   # one paragraph per line (house style)
    assert any(r"\textsuperscript{B}" in par and rf"\ref{{{_B1_LABEL}}}" in par for par in preamble), (
        f"Appendix A's preamble no longer shows the B mark in the paragraph naming Table~\\ref{{{_B1_LABEL}}}; the mark needs its legend"
    )
    print(f"  ✓ atlas.tex reads back as the data it was typeset from ({n_rows} rows, every cell, the skeleton around them, "
          f"and the {len(b1_keys)} B marks, one per row of Table B.1)")


_SAMPLE_CAPTION = re.compile(r"Pair \$(\d+)\$ of Table~\\ref\{tab:binary-icosahedral-group-2i\}")
_SAMPLE_COLS = ["bfs", "dij"]


def verify_sample_row(all_data):
    r"""
    Check Appendix A's hand-typed sample row against the atlas, cell for cell.

    The preamble reprints one pair of Table A.3 with both syntheses -- the
    min-depth (BFS) block, which the tables themselves do not print, beside
    the min-magic (Dijkstra) one they do -- and reads every cell in its
    caption. The float is hand-typed, carries no label and no generator
    writes it, so this is its guard: the pair is found by the caption's
    "Pair $n$ of Table~\ref{...2i}", its tabular is the last one before the
    caption, and its two rows are read with atlas.tex's own decoders and
    compared with pair n of the data, both word blocks and the shared cells.
    The caption's claims are pinned too, as data: depth 3 in both blocks on
    both rows, Φ 2 under BFS against 1 under Dijkstra, θ = 120° off a real
    part of 1/2, the two rows one standard-gate operator in each block (so
    their phases negate), and the a-row's BFS word Z Φ Φ at φ = ω^2. On any
    failure the two lines the float should carry are printed, ready to paste.
    """
    thesis = _strip_tex_comments(THESIS_TEX.read_text(encoding="utf-8"))
    m = _SAMPLE_CAPTION.search(thesis)
    assert m, "Appendix A: no sample-row caption 'Pair $n$ of Table~\\ref{tab:binary-icosahedral-group-2i}' in the thesis"
    pi = int(m.group(1))
    pairs = {name.split()[-1]: p for name, _rows, p in all_data}["2I"]
    assert 1 <= pi <= len(pairs), f"Appendix A's sample row names pair {pi}; 2I has {len(pairs)} pairs"
    first, second = pairs[pi - 1]
    where = f"Appendix A sample row (2I pair {pi})"
    try:
        start = thesis.rindex(r"\begin{tabular}", 0, m.start())
        tabular = thesis[start:thesis.index(r"\end{tabular}", start)]
        colspec = re.match(r"\\begin\{tabular\}\{([^}]*)\}", tabular).group(1)
        assert len(colspec.split()) == 4 * len(_SAMPLE_COLS) + 5, f"{where}: colspec {colspec!r} is not the two-block shape"
        assert " & ".join(rf"\multicolumn{{4}}{{c}}{{{_BLOCK_TITLES[c]}}}" for c in _SAMPLE_COLS) in tabular, (
            f"{where}: the head does not span both syntheses under atlas.tex's titles"
        )
        lines = re.findall(r"^\s*(\d+[ab]) & (.*?) \\\\\*?\s*$", tabular, flags=re.M)
        assert [label for label, _ in lines] == [f"{pi}a", f"{pi}b"], (
            f"{where}: rows {[label for label, _ in lines]} read, wanted {pi}a and {pi}b"
        )
        for (label, cells_tex), row in zip(lines, (first, second)):
            at = f"{where} row {label}"
            cells = _split_cells(cells_tex)
            assert len(cells) == 4 * len(_SAMPLE_COLS) + 4, f"{at}: {len(cells)} cells, not {4 * len(_SAMPLE_COLS) + 4}"
            for col in _SAMPLE_COLS:
                _check_word_cells(cells[:4], row, col, f"{at} [{col}]")
                cells = cells[4:]
            if label.endswith("b"):
                assert cells == [""] * 4, f"{at}: the shared cells are not empty on the b-row"
            else:
                _check_pair_cells(cells, first, second, at)
        _rep, _axis, deg = pair_rotation(first, second)
        caption = {   # what the caption says, as data
            "depth 3 in both blocks on both rows": all(r[f"{c}_depth"] == 3 for r in (first, second) for c in _SAMPLE_COLS),
            "Φ 2 under BFS, 1 under Dijkstra": all((r["bfs_magic"], r["dij_magic"]) == (2, 1) for r in (first, second)),
            "θ = 120° off a real part of 1/2": deg == 120 and _is_zero(first["quat"].w - Rational(1, 2)),
            "one standard-gate operator per block, so the phases negate": all(
                _is_zero(first[f"{c}_phase"] + second[f"{c}_phase"]) for c in _SAMPLE_COLS),
            "the a-row's BFS word Z Φ Φ at φ = ω^2": first["bfs_seq"] == ["Z", "Φ", "Φ"] and _is_zero(first["bfs_phase"] - _OMEGA ** 2),
        }
        broken = [claim for claim, holds in caption.items() if not holds]
        assert not broken, f"{where}: the caption's claims no longer hold: {broken}"
    except AssertionError:
        print(f"  The sample row should read (both blocks, atlas.tex's format):\n{atlas_pair_tex(pi, first, second, _SAMPLE_COLS)}")
        raise
    print(f"  ✓ Appendix A's sample row is 2I pair {pi} cell for cell, both syntheses, and its caption's claims hold")


# =============================================================================
# Main
# =============================================================================

def main():
    print("Binary Polyhedral Groups Atlas Generator")
    print("=" * 50)

    # Verify all groups
    print("\nVerification:")
    for mode in ["2T", "2O", "2I"]:
        verify_group(mode)

    # Generate data
    runs = [
        ("Binary Tetrahedral Group 2T", "2T", GATES_BFS_2T, GATES_DIJ_2T, GATES_U2_2T, GATES_DIJ_U2_2T),
        ("Binary Octahedral Group 2O", "2O", GATES_BFS_2O, GATES_DIJ_2O, GATES_U2_2O, GATES_DIJ_U2_2O),
        ("Binary Icosahedral Group 2I", "2I", GATES_BFS_2I, GATES_DIJ_2I, GATES_U2_2I, GATES_DIJ_U2_2I),
    ]

    all_data = []
    for group_name, mode, bfs_gates, dij_gates, u2_gates, dij_u2_gates in runs:
        print(f"\nSynthesizing {group_name}...")
        rows = generate_group_data(mode, bfs_gates, dij_gates, u2_gates, dij_u2_gates)
        pairs = _pair_antipodal_rows(rows)
        all_data.append((group_name, rows, pairs))
        depths = [r["bfs_depth"] for r in rows]
        magics = [r["dij_magic"] for r in rows]
        print(f"  {len(rows)} elements ({len(pairs)} pairs) | "
              f"BFS depth {min(depths)}-{max(depths)} | "
              f"Dijkstra magic {min(magics)}-{max(magics)}")

    # Verify all synthesis results
    print("\nSynthesis verification:")
    for group_name, rows, pairs in all_data:
        mode = group_name.split()[-1]  # "2T", "2O", "2I"
        verify_synthesis(rows, mode)
        verify_optimality(rows, mode)
        verify_atlas_layout(pairs, mode)
        verify_atlas_rotations(pairs, mode)
        if mode == "2I":
            verify_differing_rows(rows, mode)
    verify_atlas_nesting(all_data)
    verify_depth_concentration(all_data)
    verify_sample_row(all_data)

    # Write outputs. Both .tex files are staged, read back, and only then put
    # in place, so a read-back failure gates the write like every other check
    # and leaves the tracked file as it was.
    print("\nWriting output...")
    here = Path(__file__).parent
    write_text_atlas(here / "atlas.txt", all_data)
    for name, write, verify, args in (
        ("atlas.tex", write_latex_tables, verify_atlas_tex, all_data),
        ("differing_rows.tex", write_differing_rows, verify_differing_rows_tex,
         {n.split()[-1]: r for n, r, _p in all_data}["2I"]),
    ):
        staged = here / (name + ".staged")
        try:
            write(staged, args)
            verify(staged, args)
        except BaseException:
            staged.unlink(missing_ok=True)
            raise
        staged.replace(here / name)
        print(f"  {name} in place: {here / name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
