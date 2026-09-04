"""
Platonic Solid POVM Vertices and Verification.

Builds the five Platonic-solid POVMs on the Bloch sphere from closed-form
vertex coordinates -- each solid's over its own real number field, tabulated
in _SOLID_FIELD_GENS -- and verifies their defining properties symbolically:
  - completeness: sum_k E_k = I
  - unit norm and zero sum of Bloch vectors
  - informational completeness (rank of V x 4 vec'd-effect matrix; the
    second-moment identity (1/V) sum_k |psi_k><psi_k|^2 = (1/6)(I + SWAP))
  - spherical t-design strength via discrete vs. continuous moment tensors
  - complex projective t-design strength via Welch-bound saturation of the
    frame potential
  - closed-form angle multiset {|<psi_i|psi_j>|^2 : i != j}
  - dual alignment of the tabulated pairs: every octahedron vertex is a cube
    face center and vice versa, likewise icosahedron <-> dodecahedron, and
    the tetrahedron's face centers are its antipodal copy
  - the covariance copy: conjugation by F permutes all five effect sets,
    conjugation by Phi permutes exactly the icosahedral pair's, and
    conjugation by Phi* permutes none -- the tabulated icosahedron and
    dodecahedron belong to the family served by the atlas copy <X, Z, F, Phi>

Every verdict is decided in CANONICAL FORM, not by `simplify(...) == 0`: each
solid's coordinates are coerced into its own algebraic number field, where
equal numbers have equal representatives and out-of-field input raises rather
than being quietly approximated. See the note above _SOLID_FIELD_GENS for why
the fields are per-solid and why they are real, and the one above _RADIAL for
why cross-solid claims clear their denominators instead of moving to the
compositum. Only two decisions stay numerical: which vertices form a face, and
which effect a conjugated effect is closest to -- searches that merely propose
candidates, every one of which is then confirmed exactly.

Vertex orderings match paper/figures/platonic_solid_povms.tex so the figure
indices coincide with the atlas row numbers.

Outputs (in code/data/):
  povm_atlas.tex            one vertex table, one block per solid (Table E.1)
  povm_properties.tex       the per-solid properties table (Table E.3), the
                            thesis's only printing of Phi_2..Phi_5
  povm_angle_multisets.tex  the pairwise-overlap multiset table; not \\input
                            by the thesis, kept as a repository resource that
                            every sweep regenerates, so `uv run everything.py`
                            stays byte-reproducible

Run with `uv run python code/povm_properties.py`.
"""

from itertools import groupby, product
from pathlib import Path

from sympy import (
    I as symI,
    Matrix,
    QQ,
    Rational,
    expand,
    eye,
    latex,
    simplify,
    sqrt,
    zeros,
)
from sympy.polys.matrices import DomainMatrix

from main import (
    SIGMA, TAU,
    _F, _Phi,
    _golden_sub, _sig_sym, _tau_sym,
)


# =============================================================================
# Constants
# =============================================================================

SIGMA_X = Matrix([[0, 1], [1, 0]])
SIGMA_Y = Matrix([[0, -symI], [symI, 0]])
SIGMA_Z = Matrix([[1, 0], [0, -1]])

SWAP4 = Matrix([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])


def _kron(A, B):
    """Kronecker product of two SymPy matrices."""
    m, n = A.shape
    p, q = B.shape
    return Matrix(m * p, n * q,
                  lambda r, c: A[r // p, c // q] * B[r % p, c % q])


# =============================================================================
# Vertex constructors
#
# Built by a single parameterized factory _build_vertices(tau_val, sig_val), so
# we can instantiate the same orderings twice: once with the numeric values of
# the golden-ratio constants (used for the symbolic verifications, which rely
# on tau = (1 + sqrt(5))/2 to reduce things like 1 + tau^2 to 2 + tau), and
# once with bare SymPy symbols tau, sigma (used purely for display, so the
# atlas tables show 1/sqrt(2 + tau), tau/sqrt(2 + tau), sigma/sqrt(3), ...
# instead of nested radicals).
#
# All orderings match paper/figures/platonic_solid_povms.tex so the figure
# vertex indices and atlas row numbers coincide -- checked coordinate by
# coordinate by export_numpy.verify_figure_orders(), not just asserted here.
# =============================================================================

def _build_vertices(tau, sig):
    c = 1 / sqrt(3)
    one = Rational(1)

    tetrahedron = (
        Matrix([ c,  c,  c]),
        Matrix([-c, -c,  c]),
        Matrix([-c,  c, -c]),
        Matrix([ c, -c, -c]),
    )

    octahedron = (
        Matrix([ one, 0, 0]),
        Matrix([-one, 0, 0]),
        Matrix([0,  one, 0]),
        Matrix([0, -one, 0]),
        Matrix([0, 0,  one]),
        Matrix([0, 0, -one]),
    )

    cube = (
        Matrix([-c, -c, -c]),
        Matrix([ c, -c, -c]),
        Matrix([ c,  c, -c]),
        Matrix([-c,  c, -c]),
        Matrix([-c, -c,  c]),
        Matrix([ c, -c,  c]),
        Matrix([ c,  c,  c]),
        Matrix([-c,  c,  c]),
    )

    s = 1 / sqrt(2 + tau)
    icosahedron = (
        Matrix([ s * tau,  s * one,      0]),
        Matrix([-s * tau,  s * one,      0]),
        Matrix([ s * tau, -s * one,      0]),
        Matrix([-s * tau, -s * one,      0]),
        Matrix([    0,  s * tau,  s * one]),
        Matrix([    0, -s * tau,  s * one]),
        Matrix([    0,  s * tau, -s * one]),
        Matrix([    0, -s * tau, -s * one]),
        Matrix([s * one,      0,  s * tau]),
        Matrix([-s * one,     0,  s * tau]),
        Matrix([s * one,      0, -s * tau]),
        Matrix([-s * one,     0, -s * tau]),
    )

    dodecahedron = (
        # 8 cube vertices
        Matrix([ c,  c,  c]),
        Matrix([ c,  c, -c]),
        Matrix([ c, -c,  c]),
        Matrix([ c, -c, -c]),
        Matrix([-c,  c,  c]),
        Matrix([-c,  c, -c]),
        Matrix([-c, -c,  c]),
        Matrix([-c, -c, -c]),
        # 12 golden-rectangle vertices
        Matrix([       0,  c * sig,    c * tau]),
        Matrix([       0,  c * sig,   -c * tau]),
        Matrix([       0, -c * sig,    c * tau]),
        Matrix([       0, -c * sig,   -c * tau]),
        Matrix([ c * sig,    c * tau,        0]),
        Matrix([ c * sig,   -c * tau,        0]),
        Matrix([-c * sig,    c * tau,        0]),
        Matrix([-c * sig,   -c * tau,        0]),
        Matrix([   c * tau,        0,  c * sig]),
        Matrix([   c * tau,        0, -c * sig]),
        Matrix([  -c * tau,        0,  c * sig]),
        Matrix([  -c * tau,        0, -c * sig]),
    )

    return {
        "tetrahedron":  tetrahedron,
        "octahedron":   octahedron,
        "cube":         cube,
        "icosahedron":  icosahedron,
        "dodecahedron": dodecahedron,
    }


# Numeric vertices: golden-ratio constants are (1 +/- sqrt(5))/2, so the
# coordinates are algebraic numbers and coerce into the fields below.
_NUMERIC = _build_vertices(TAU, SIGMA)

# Display vertices: tau, sigma are bare symbols, so the atlas tables preserve
# closed-form expressions like 1/sqrt(2 + tau) instead of nested radicals.
_DISPLAY = _build_vertices(_tau_sym, _sig_sym)


def numeric_vertices(name):
    return _NUMERIC[name]


def display_vertices(name):
    return _DISPLAY[name]


SOLIDS = (
    ("tetrahedron",  "Tetrahedron",  r"\TwoT",
     r"vertices $(\pm 1, \pm 1, \pm 1)/\sqrt{3}$ with even sign parity"),
    ("octahedron",   "Octahedron",   r"\TwoO",
     r"vertices $\pm\hat{e}_x, \pm\hat{e}_y, \pm\hat{e}_z$"),
    ("cube",         "Cube",         r"\TwoO",
     r"vertices $(\pm 1, \pm 1, \pm 1)/\sqrt{3}$"),
    ("icosahedron",  "Icosahedron",  r"\TwoI",
     r"cyclic permutations of $(0, \pm\tau, \pm 1)/\sqrt{2 + \tau}$"),
    ("dodecahedron", "Dodecahedron", r"\TwoI",
     r"cube $\cup$ cyclic permutations of $(0, \pm\sigma, \pm\tau)$, all $/\sqrt{3}$"),
)

# The design ladder, pinned. Both strengths fail quietly downward rather than
# raising (see spherical_design_strength), and the result goes straight into
# the properties table -- so main() asserts against this table, the one place
# in the repo the ladder is fixed to its values. The randomized suite
# (randomized_fragments.py) imports it and pins its own independent numerical
# recomputation to it, row by row.
EXPECTED_DESIGN = {
    "tetrahedron":  2,
    "octahedron":   3,
    "cube":         3,
    "icosahedron":  5,
    "dodecahedron": 5,
}


# Layout descriptors for the compact atlas: scalar to factor out of each
# component (so cells stay barebones: signed integers, tau, sigma, zero), and
# the scalar's LaTeX form for the per-solid section-header row. A scalar_tex
# of None signals unity (the octahedron) and the "components scaled by..."
# annotation is suppressed.
ATLAS_LAYOUTS = {
    "tetrahedron":  {"scalar": Rational(1) / sqrt(3),       "scalar_tex": r"\tfrac{1}{\sqrt{3}}",   "widest_mag": "1"},
    "octahedron":   {"scalar": Rational(1),                 "scalar_tex": None,                      "widest_mag": "1"},
    "cube":         {"scalar": Rational(1) / sqrt(3),       "scalar_tex": r"\tfrac{1}{\sqrt{3}}",   "widest_mag": "1"},
    "icosahedron":  {"scalar": 1 / sqrt(2 + _tau_sym),      "scalar_tex": r"\tfrac{1}{\sqrt{2+\tau}}", "widest_mag": r"\tau"},
    "dodecahedron": {"scalar": Rational(1) / sqrt(3),       "scalar_tex": r"\tfrac{1}{\sqrt{3}}",   "widest_mag": r"\tau"},
}

# Vertex panels packed per typeset row in the atlas tabular. Each panel
# contributes 2 columns: index $k$ (right-aligned) and the tuple
# $(n_x, n_y, n_z)$ (left-aligned).
PANELS_PER_ROW = 4

# Per-solid override; the octahedron gets a 2 x 3 sub-block (the 4 + 2
# pad-row layout breaks its three antipodal pairs awkwardly, and 1 x 6
# overflows the text width).
PANELS_PER_ROW_OVERRIDE = {"octahedron": 3}

# =============================================================================
# Canonical form: one algebraic number field per solid
#
# Every symbolic verdict below is decided by coercing into the solid's own
# number field and comparing against its zero, not by `simplify(...) == 0`.
# Two properties, both load-bearing. CANONICAL: any two expressions denoting
# the same number get the same representative, so equality needs no tolerance
# and no dependence on how the expression was written -- and the elements are
# hashable, which is what lets angle_multiset bin by key instead of by an
# O(V^2) scan of pairwise `simplify` calls. FAIL-LOUD: out-of-field input
# raises CoercionFailed rather than being quietly approximated, so a re-posed
# solid or a mistyped vertex cannot slip through. The declaration is checked,
# not assumed, which is what lets a field this narrow be taken without a bet.
#
# PER-SOLID, never one global field. Each solid's coordinates lie in a field
# of degree <= 4; one global field big enough for all five and the gates too
# is degree 16 and costs ~150x more for the same coercions. The table is
# pose-dependent -- the octahedron is Q-rational only in the atlas pose -- and
# the icosahedron's and dodecahedron's degree-4 fields are non-isomorphic,
# neither containing the other's generator. Dual solids, different fields; the
# field tracks (solid, pose), never the abstract solid. Nothing in this file
# rotates a solid, and the atlas pose is pinned far harder elsewhere (Table
# C.1, the section ladder, Decker's reorientation table), so this inherits that
# pin rather than adding one.
#
# The fields are REAL, and that is not an approximation. Writing the effects
# in the Pauli basis, E_k = (1/V)(I + n_k . sigma) has components
# (a_0, a_x, a_y, a_z) = (1, n_kx, n_ky, n_kz)/V -- the i in
# a_y = i(E_01 - E_10)/2 cancels -- so completeness, the vec'd-effect rank and
# the second moment are all statements about vertex coordinates. Where a
# genuinely complex expression does arise (conjugation by F or Phi), _is_zero
# splits it into real and imaginary parts, each of which lands back in the
# real field. Adjoining i would double every degree, taking the icosahedron
# and dodecahedron to degree 8 and their coercions from ~1.5 s to ~25 s.
#
# No simplify/radsimp/nsimplify in front of from_sympy: every coordinate
# coerces raw, and a normalising heuristic there would forfeit the fail-loud
# half -- an out-of-field value would get a second chance to be massaged into
# looking like an in-field one.
# =============================================================================

_SOLID_FIELD_GENS = {
    "tetrahedron":  (sqrt(3),),
    "octahedron":   (),                      # rational in the atlas pose
    "cube":         (sqrt(3),),
    "icosahedron":  (sqrt(2 + TAU),),
    "dodecahedron": (sqrt(3), sqrt(5)),
}

_FIELDS = {
    name: (QQ.algebraic_field(*gens) if gens else QQ)
    for name, gens in _SOLID_FIELD_GENS.items()
}


def solid_field(name):
    """The canonical domain for a solid's coordinates (see the note above)."""
    return _FIELDS[name]


def _is_zero(K, value):
    """Canonical zero test over K: exact, tolerance-free, and fail-loud.

    Real and imaginary parts are coerced separately so K itself stays real.
    Raises CoercionFailed if `value` is not in K -- which is a verdict, not a
    crash: it says the expression left the field it was declared to live in.
    """
    re_part, im_part = expand(value).as_real_imag()
    return K.from_sympy(re_part) == K.zero and K.from_sympy(im_part) == K.zero


def _is_positive(value):
    """Exact sign test for a real algebraic number -- no tolerance, no field.

    A field decides equality, not order, so a strict inequality needs the
    other tool: SymPy fixes the sign by interval refinement and returns None
    when it cannot, so a True here is a proof rather than a measurement. It
    also subsumes non-vanishing, which is why these sites do not also go
    through K.
    """
    sign = expand(value).is_positive
    assert sign is not None, f"could not decide the sign of {value}"
    return sign


# Radial denominators: each solid's vertices are (integer-ish tuple over
# Q(sqrt5)) / _RADIAL[name]. Multiplying a cross-solid claim through by the
# denominators it involves clears them and lands it in _PAIR_FIELD at degree 2,
# instead of the degree-8 field it would otherwise need -- the trick behind the
# `scale` argument of verify_dual_alignment and conjugation_permutation. The
# multiplier is a fixed positive number, so it can change neither a zero nor a
# sign, the only two things either of them decides.
_RADIAL = {
    "tetrahedron":  sqrt(3),
    "octahedron":   Rational(1),
    "cube":         sqrt(3),
    "icosahedron":  sqrt(2 + TAU),
    "dodecahedron": sqrt(3),
}

# The field every cross-solid claim lands in once cleared. Degree 2, so the
# per-solid split's reason -- cost -- does not bite here; fail-loud still does.
_PAIR_FIELD = QQ.algebraic_field(sqrt(5))


# =============================================================================
# POVM construction
# =============================================================================

def povm_elements(vertices):
    """E_k = (1/V) (I + n_k . sigma) as symbolic 2x2 matrices."""
    V = len(vertices)
    inv = Rational(1, V)
    return tuple(
        inv * (eye(2) + n[0] * SIGMA_X + n[1] * SIGMA_Y + n[2] * SIGMA_Z)
        for n in vertices
    )


# =============================================================================
# Symbolic verifications
# =============================================================================

def verify_completeness(elements, K):
    """sum_k E_k = I."""
    total = elements[0].copy()
    for E in elements[1:]:
        total = total + E
    for i in range(2):
        for j in range(2):
            assert _is_zero(K, total[i, j] - (1 if i == j else 0)), (
                f"completeness violated at [{i},{j}]: {total[i,j]}"
            )


def verify_unit_norm(vertices, K):
    for k, n in enumerate(vertices, 1):
        assert _is_zero(K, n[0] ** 2 + n[1] ** 2 + n[2] ** 2 - 1), (
            f"vertex {k} is not a unit vector: {n.T}"
        )


def verify_zero_sum(vertices, K):
    s = vertices[0].copy()
    for n in vertices[1:]:
        s = s + n
    for i in range(3):
        assert _is_zero(K, s[i]), f"vertex sum component {i} = {s[i]}"


def verify_ic_rank(elements, K):
    """Vec each effect in the Pauli basis {I, sigma_x, sigma_y, sigma_z} and
    check that the V x 4 matrix has rank 4 (full column rank).

    Rank is computed over K rather than by Gaussian elimination with a zero
    test bolted on. The default iszerofunc only checks .is_zero, which returns
    None for some Q(sqrt(5)) expressions and then OVER-REPORTS rank -- the
    heuristic has bitten here already, which is the argument for the whole
    file: over a field domain there is no zero test to get wrong.
    """
    rows = []
    for E in elements:
        # E = a0 I + ax sigma_x + ay sigma_y + az sigma_z. Every component is
        # real -- the i in ay cancels against sigma_y's -- so the row lands in
        # K, and coercion says so rather than being told so.
        a0 = (E[0, 0] + E[1, 1]) / 2
        ax = (E[0, 1] + E[1, 0]) / 2
        ay = symI * (E[0, 1] - E[1, 0]) / 2
        az = (E[0, 0] - E[1, 1]) / 2
        rows.append([K.from_sympy(expand(c)) for c in (a0, ax, ay, az)])
    r = DomainMatrix(rows, (len(rows), 4), K).rank()
    assert r == 4, f"vec'd-effect matrix has rank {r}, expected 4"


def verify_second_moment(vertices, K):
    """(1/V) sum_k |psi_k><psi_k|^{(x)2} = (1/6) (I_4 + SWAP).

    Uses |psi><psi| = (1/2)(I + n.sigma) so the tensor square stays symbolic
    without ever picking a phase representative for |psi>.
    """
    V = len(vertices)
    total = zeros(4, 4)
    for n in vertices:
        rho = (eye(2) + n[0] * SIGMA_X + n[1] * SIGMA_Y + n[2] * SIGMA_Z) / 2
        total = total + _kron(rho, rho)
    target = (eye(4) + SWAP4) / 6
    diff = total / V - target
    for i in range(4):
        for j in range(4):
            assert _is_zero(K, diff[i, j]), (
                f"second-moment violated at [{i},{j}]: {diff[i,j]}"
            )


def _double_factorial(n):
    """n!! with (-1)!! = 0!! = 1."""
    if n <= 0:
        return Rational(1)
    result = Rational(1)
    while n > 0:
        result *= n
        n -= 2
    return result


def sphere_moment(indices):
    """Sphere average (1/4 pi) integral_{S^2} x_{i_1} ... x_{i_t} d sigma.

    Vanishes if any partial degree is odd; otherwise equals
        (a_x - 1)!! (a_y - 1)!! (a_z - 1)!! / (a_x + a_y + a_z + 1)!!
    in the convention (-1)!! = 1.
    """
    a = [0, 0, 0]
    for i in indices:
        a[i] += 1
    if any(ai % 2 for ai in a):
        return Rational(0)
    K = a[0] + a[1] + a[2]
    num = (_double_factorial(a[0] - 1)
           * _double_factorial(a[1] - 1)
           * _double_factorial(a[2] - 1))
    den = _double_factorial(K + 1)
    return num / den


def spherical_design_strength(vertices, K, t_max=6):
    """Largest t for which (1/V) sum_k n_k^{(x)t} = sphere moment of degree t.

    Note the direction this decision fails in. Every other check here asserts
    a zero, so a zero test too weak to see one fires an assertion. This one
    exits on a NON-zero, so a zero test too weak to see one would return a
    smaller t -- no assertion, just a quieter answer, written straight into
    the properties table. Over K the non-equality is decidable, so the value
    returned is the strength and not a lower bound on it; main() then pins it
    against EXPECTED_DESIGN so a regression here has to be loud.
    """
    V = len(vertices)
    last_good = 0
    for t in range(1, t_max + 1):
        good = True
        for indices in product(range(3), repeat=t):
            discrete = Rational(0)
            for n in vertices:
                term = Rational(1)
                for i in indices:
                    term *= n[i]
                discrete = discrete + term
            if not _is_zero(K, discrete / V - sphere_moment(indices)):
                good = False
                break
        if not good:
            return last_good
        last_good = t
    return last_good


def frame_potential(vertices, t):
    """Phi_t = (1/V^2) sum_{i,j} |<psi_i|psi_j>|^{2t}
    computed in closed form from Bloch dot products.

    Returned UNSIMPLIFIED. The Welch comparison coerces this straight into the
    solid's field, and putting a simplification in front of that coercion
    would place a heuristic inside the decision -- the callers that display
    the value simplify it themselves, which is the right place for it. It
    costs nothing: coercing the raw sum is as fast as simplifying it first.
    """
    V = len(vertices)
    total = Rational(0)
    for i in range(V):
        ni = vertices[i]
        for j in range(V):
            nj = vertices[j]
            dot = ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2]
            overlap_sq = (1 + dot) / 2
            total = total + overlap_sq ** t
    return total / V ** 2


def welch_bound(t, d=2):
    """Welch lower bound on Phi_t for unit vectors in C^d: 1/C(d+t-1, t).

    For d = 2 this is 1/(t+1).
    """
    assert d == 2, "only d=2 supported"
    return Rational(1, t + 1)


def projective_design_strength(vertices, K, t_max=6):
    """Largest t for which Phi_t saturates the Welch bound 1/(t+1).

    Exits on a non-zero, so it fails in the same quiet direction as
    spherical_design_strength -- see the note there.
    """
    last_good = 0
    for t in range(1, t_max + 1):
        if not saturates_welch(frame_potential(vertices, t), t, K):
            return last_good
        last_good = t
    return last_good


def saturates_welch(phi_t, t, K):
    """Whether a frame potential meets the Welch bound exactly.

    One definition for all three callers -- the strength ladder, the bolding
    in the properties table, and main()'s stdout -- so the table cannot bold a
    cell the ladder disagrees with.
    """
    return _is_zero(K, phi_t - welch_bound(t))


def angle_multiset(vertices, K):
    """Multiset of |<psi_i|psi_j>|^2 over unordered pairs i != j.

    Returns a list of (value, multiplicity) sorted by descending value.

    Binning is by canonical KEY. Field elements are hashable, so equal
    overlaps collide in a dict by construction -- which is the property a zero
    test cannot supply: `simplify(a - b) == 0` decides equality but yields no
    key, so it can only ever back a linear scan over the bins, and one missed
    canonicalization silently splits a bin in two. The C(V, 2) guard against
    exactly that is structural here (every pair increments exactly one bin),
    so it stands as an invariant rather than as a defence.
    """
    V = len(vertices)
    bins = {}  # canonical key -> [display value, multiplicity]
    for i in range(V):
        for j in range(i + 1, V):
            ni, nj = vertices[i], vertices[j]
            dot = ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2]
            val = (1 + dot) / 2
            key = K.from_sympy(expand(val))
            if key in bins:
                bins[key][1] += 1
            else:
                bins[key] = [simplify(val), 1]
    total = sum(c for _, c in bins.values())
    expected = V * (V - 1) // 2
    assert total == expected, (
        f"angle multiset multiplicities sum to {total}, expected {expected} "
        f"= C({V}, 2)"
    )
    ordered = sorted(bins.values(), key=lambda e: -float(e[0].evalf()))
    # The bins are distinct as field elements, so the only thing the float
    # sort has to get right is their ORDER. Assert the margin rather than
    # trust it: the closest two distinct overlaps sit far above float noise.
    gaps = [float((ordered[k][0] - ordered[k + 1][0]).evalf(20))
            for k in range(len(ordered) - 1)]
    assert all(g > 1e-9 for g in gaps), (
        f"angle values too close to order by float: min gap {min(gaps)}"
    )
    return [(v, c) for v, c in ordered]


# =============================================================================
# Cross-solid verifications: dual alignment and the covariance copy
#
# Two facts the vertex tables silently encode (thesis Appendix E). First, the
# tabulated dual pairs are dual-ALIGNED: each vertex of one solid is the
# outward center of a face of its partner (octahedron <-> cube, icosahedron
# <-> dodecahedron, and the tetrahedron's face centers are its antipodal
# copy). Second, the tabulated icosahedron and dodecahedron belong to the same
# icosahedral family -- the one whose covariance group is <X, Z, F, Phi>:
# conjugation by Phi permutes their effect sets, conjugation by Phi* (the
# entry-wise image of Phi under sqrt(5) -> -sqrt(5), thesis Eq. (golden-gate))
# permutes neither, and F, common to all three binary polyhedral groups,
# permutes all five. Candidate matches are located numerically -- a float
# distance above 1e-6 is conclusive against ~1e-15 arithmetic error -- and
# every verdict is then decided exactly: a match entry-by-entry against its
# candidate, a miss by a witness effect whose image differs from every
# tabulated effect over the pair field. Most of the verdicts here are
# negative, so the misses assert their float margin too (see
# conjugation_permutation): nothing lands on the threshold unremarked.
# =============================================================================

_PHI_STAR = _Phi.subs(sqrt(5), -sqrt(5))


def _to_complex_tuple(E):
    """Flatten a symbolic 2x2 matrix into a tuple of Python complex numbers."""
    return tuple(complex(E[i, j].evalf(20)) for i in range(2) for j in range(2))


def conjugation_permutation(U, elements, scale):
    """Return the permutation k -> j with U E_k U^dagger = E_j, or None if
    conjugation by U does not map the effect set to itself.

    Candidates are found by float distance, and the float search only
    proposes: every candidate it offers is then confirmed exactly. A candidate
    within 1e-6 that fails that confirmation is a canonicalization failure
    rather than a near-miss (distinct effects sit at float distance >> 1e-6),
    so it raises instead of returning None. The None itself is a verdict too,
    decided exactly from the other side: the witness miss is confirmed to
    differ from every tabulated effect over the pair field, so the float
    only picks which effect testifies -- see also the margin assert.

    `scale` is V times the solid's radial denominator, which clears the
    denominators out of the confirmation (see _RADIAL): it decides in
    _PAIR_FIELD rather than in the solid's own field with i adjoined.
    """
    K = _PAIR_FIELD
    Ud = U.H
    originals = [_to_complex_tuple(E) for E in elements]
    perm = []
    for k, E in enumerate(elements):
        C = U * E * Ud
        cf = _to_complex_tuple(C)
        dists = [max(abs(x - y) for x, y in zip(cf, o)) for o in originals]
        j = min(range(len(dists)), key=dists.__getitem__)
        if dists[j] > 1e-6:
            # Assert the margin rather than trust the threshold: the miss is
            # what decides "does not permute", which is how most of the
            # tabulated verdicts are decided. The separation is two-sided and
            # enormous -- every true match sits at distance <= 6.5e-27, and
            # every miss at >= 8.9e-3 (>= 1.8e-2 for the one that actually
            # decides a None, being the first miss found). The bound below
            # sits between the two, so a genuine near-miss raises instead of
            # being filed as a clean miss.
            assert dists[j] > 1e-3, (
                f"effect {k} is neither a match nor a clean miss: best "
                f"distance {dists[j]} to effect {j} lands in the "
                f"undecidable band"
            )
            # The float miss only proposes the verdict; one witness decides
            # it: effect k's image differs from every tabulated effect over
            # the pair field, so no permutation can exist. (The identity
            # parts of any two effects cancel in the difference, so the
            # scaled entries stay in the pair field even across a miss.)
            for jj, Ej in enumerate(elements):
                assert not all(
                    _is_zero(K, (C[r, c] - Ej[r, c]) * scale)
                    for r in range(2) for c in range(2)
                ), (
                    f"float miss contradicted exactly: effect {k}'s image "
                    f"IS effect {jj}"
                )
            return None
        for r in range(2):
            for c in range(2):
                assert _is_zero(K, (C[r, c] - elements[j][r, c]) * scale), (
                    f"numeric match {k} -> {j} failed exact confirmation "
                    f"at [{r},{c}]: {C[r,c]} vs {elements[j][r,c]}"
                )
        perm.append(j)
    assert sorted(perm) == list(range(len(elements))), f"not a bijection: {perm}"
    return perm


def verify_dual_alignment(vertices_a, vertices_b, face_size, scale=1):
    """Every vertex a of solid A is the outward center of a face of solid B:
    the face_size vertices of B nearest to a (by Bloch overlap) have exactly
    equal overlaps, strictly larger than every other vertex's, and their
    centroid is a positive multiple of a.

    Face MEMBERSHIP is still located numerically -- which vertices form the
    face is a combinatorial choice, and the float search only proposes it.
    Everything the search proposes is then decided exactly: the two overlap
    inequalities by sign, the overlap equalities and the parallelism by
    coercion into the pair field.

    `scale` is the product of the two solids' radial denominators, which
    clears them out of the claim (see _RADIAL). It defaults to 1, for callers
    whose vertices are already unnormalized (dial_settings.py keeps them that
    way on purpose); a wrong scale cannot pass silently, because the value
    then fails to coerce.
    """
    K = _PAIR_FIELD
    for k, a in enumerate(vertices_a, 1):
        dots = [a.dot(b) for b in vertices_b]
        vals = [float(d.evalf(20)) for d in dots]
        order = sorted(range(len(vals)), key=vals.__getitem__, reverse=True)
        face, rest = order[:face_size], order[face_size:]
        assert _is_positive(dots[face[-1]] - dots[rest[0]]), (
            f"vertex {k}: no clean overlap gap after {face_size} vertices "
            f"({vals[face[-1]]} vs {vals[rest[0]]})"
        )
        for j in face[1:]:
            assert _is_zero(K, (dots[face[0]] - dots[j]) * scale), (
                f"vertex {k}: face overlaps unequal at vertex {j} of the dual"
            )
        centroid = zeros(3, 1)
        for j in face:
            centroid = centroid + vertices_b[j]
        cross = a.cross(centroid)
        for i in range(3):
            assert _is_zero(K, cross[i] * scale), (
                f"vertex {k}: face centroid not parallel, cross[{i}] = {cross[i]}"
            )
        assert _is_positive(a.dot(centroid)), (
            f"vertex {k}: face centroid anti-parallel"
        )


# =============================================================================
# Formatting helpers
# =============================================================================

def _fmt(value):
    """LaTeX-format a real expression, applying golden-ratio substitution
    so phrases like 2*tau - 1 collapse to sqrt(5) and rationals stay clean.
    Fractions render as \\tfrac so cells stay vertically compact in tables."""
    expr = _golden_sub(simplify(value), _tau_sym, _sig_sym)
    return latex(expr).replace(r"\frac", r"\tfrac")


# =============================================================================
# TeX writers
# =============================================================================

_HEADER = """% Auto-generated by code/povm_properties.py.
% Include this file with \\input{...}. Requires booktabs, multirow, float (for
% the [H] placement), mathtools, and the thesis macros \\braket, \\TwoT, \\TwoO, \\TwoI.
% Symbolic, exact over each solid's own number field; regenerate with `uv run python code/povm_properties.py`.

"""


def _split_sign_mag(value, scalar):
    """Divide a vertex component by the solid's radial scalar and return
    (sign, magnitude) LaTeX strings for the tuple's nested sign/magnitude
    column pair. Zeros use \\phantom{+} so the magnitude column still aligns.
    """
    reduced = simplify(value / scalar)
    # The caption promises the magnitude columns hold only a bare integer,
    # tau, sigma or 0, and Appendix D.1's alignment sweep reads its coordinate
    # shapes off that promise -- so the alphabet is decided here rather than
    # formatted. Squares, because the sign is split off in the lines below.
    assert any(simplify(reduced**2 - a**2) == 0
               for a in (0, 1, _tau_sym, _sig_sym)), reduced
    if reduced == 0:
        return (r"\phantom{+}", "0")
    if reduced.could_extract_minus_sign():
        return ("-", latex(-reduced))
    return ("+", latex(reduced))


# Nested-array column spec for a single tuple cell: three (sign, magnitude)
# pairs separated by commas, with \arraycolsep stripped so signs sit flush
# against their magnitudes.
_TUPLE_ARRAY_SPEC = r"@{}r@{}l@{,\,}r@{}l@{,\,}r@{}l@{}"


def _fmt_tuple(vertex, scalar, widest_mag):
    """Format a vertex as $(\\begin{array}...\\end{array})$ with sign/
    magnitude alignment across rows of the same block. Fixed parens
    (no \\left/\\right): tuple content is always single-line, so we don't
    need auto-sizing, and fixed parens stay text-height under any
    surrounding \\arraystretch.

    Each magnitude is wrapped as \\mathrlap{<mag>}\\hphantom{<widest_mag>}
    so every mag cell has the same width as the solid's widest magnitude
    (e.g., \\tau for the icosahedron / dodecahedron). Without this, the
    inner array is 1 row tall and so its mag columns are sized to that
    row's content alone, drifting the comma rightward in tuples whose
    first component is \\tau versus 0 or 1."""
    sx, mx = _split_sign_mag(vertex[0], scalar)
    sy, my = _split_sign_mag(vertex[1], scalar)
    sz, mz = _split_sign_mag(vertex[2], scalar)
    mx = rf"\mathrlap{{{mx}}}\hphantom{{{widest_mag}}}"
    my = rf"\mathrlap{{{my}}}\hphantom{{{widest_mag}}}"
    mz = rf"\mathrlap{{{mz}}}\hphantom{{{widest_mag}}}"
    body = f"{sx} & {mx} & {sy} & {my} & {sz} & {mz}"
    return (rf"$(\begin{{array}}{{{_TUPLE_ARRAY_SPEC}}}"
            f"{body}"
            r"\end{array})$")


def _fmt_index(k):
    """Small upright index with a trailing period."""
    return rf"{{\scriptsize ${k}.$}}"


def write_vertex_atlas(filename, solids_data):
    """Compact reference atlas: one booktabs tabular with section headers
    separating the solids and PANELS_PER_ROW vertex panels per row by
    default. The octahedron overrides this (see PANELS_PER_ROW_OVERRIDE)
    and its six vertices appear as a 2 x 3 block via a nested tabular.

    Each panel contributes two columns: a right-aligned scriptsize index $k$
    and a left-aligned tuple cell built as a nested math array whose
    sign/magnitude column pairs align vertically across rows. The common
    radial scalar is factored into each solid's section-header row, so
    tuple magnitudes hold only the bare integer, $\\tau$, $\\sigma$, or $0$.
    """
    N = PANELS_PER_ROW
    n_cols = 2 * N
    col_spec = r" @{\qquad} ".join([r"r@{\;\,}l"] * N)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(_HEADER)
        f.write(r"\begin{table}[H]\centering" + "\n")
        f.write(rf"\begin{{tabular}}{{{col_spec}}}" + "\n")
        f.write(r"\toprule" + "\n")

        items = list(solids_data.items())
        for idx, (name, info) in enumerate(items):
            display = info["display"]
            group = info["group"]
            V = info["V"]
            verts = display_vertices(name)
            layout = ATLAS_LAYOUTS[name]
            scalar = layout["scalar"]
            scalar_tex = layout["scalar_tex"]
            widest_mag = layout["widest_mag"]

            if scalar_tex is None:
                section = display
            else:
                section = (rf"{display} --- "
                           rf"components scaled by ${scalar_tex}$")

            f.write(f"% --- {display} ---\n")
            # Skip the divider above the first section; \toprule serves it.
            if idx > 0:
                f.write(r"\midrule" + "\n")
            f.write(rf"\multicolumn{{{n_cols}}}{{c}}{{{section}}} \\" + "\n")
            f.write(r"\addlinespace[2pt]" + "\n")

            per_row = PANELS_PER_ROW_OVERRIDE.get(name, N)
            if per_row == N:
                # Standard layout: pack N panels per row, padding the last
                # row with empty cells if V is not a multiple of N.
                for row_start in range(0, V, N):
                    cells = []
                    for i in range(N):
                        k = row_start + i + 1
                        if k <= V:
                            cells.append(_fmt_index(k))
                            cells.append(_fmt_tuple(verts[k - 1], scalar, widest_mag))
                        else:
                            cells.extend(["", ""])
                    f.write("  " + " & ".join(cells) + r" \\" + "\n")
            else:
                # Override layout: per_row panels per inner row inside a
                # nested tabular that spans the outer table via multicolumn.
                inner_spec = r" @{\qquad} ".join([r"r@{\;\,}l"] * per_row)
                inner_rows = []
                for row_start in range(0, V, per_row):
                    cells = []
                    for i in range(per_row):
                        k = row_start + i + 1
                        if k <= V:
                            cells.append(_fmt_index(k))
                            cells.append(_fmt_tuple(verts[k - 1], scalar, widest_mag))
                        else:
                            cells.extend(["", ""])
                    inner_rows.append(" & ".join(cells) + r" \\")
                f.write(rf"  \multicolumn{{{n_cols}}}{{c}}{{%" + "\n")
                f.write(rf"    \begin{{tabular}}{{{inner_spec}}}" + "\n")
                for row in inner_rows:
                    f.write(f"      {row}\n")
                f.write(r"    \end{tabular}%" + "\n")
                f.write(r"  } \\" + "\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\caption{Vertex coordinates for the five Platonic-solid POVMs. "
                r"Within each solid's block, the common radial scalar is "
                r"factored into the section-header row so that tuple "
                r"magnitudes hold only the bare integer, $\tau$, $\sigma$, "
                r"or $0$. Orderings match Figure~\ref{fig:platonic-solid-povms}.}"
                r"\label{tab:povm-atlas}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"  vertex coordinates -> {filename}")


_MULTISETS_HEADER = """% Auto-generated by code/povm_properties.py.
% The pairwise-overlap multiset table. NOT \\input by the thesis (Phi_2..Phi_5
% print once, in Table E.3); kept as a repository resource and regenerated by
% every sweep. Requires booktabs, multirow, float (for the [H] placement), and the
% thesis macros \\braket, \\TwoT, \\TwoO, \\TwoI.
% Symbolic, exact over each solid's own number field; regenerate with `uv run python code/povm_properties.py`.

"""


def write_properties(filename, solids_data, multisets_filename):
    """Two fragments, one per table. `filename` gets the per-solid properties
    table (Table E.3, `tab:povm-properties`), which the thesis \\input's;
    `multisets_filename` gets the pairwise-overlap multiset table
    (`tab:povm-angle-multisets`), which the thesis does not \\input (see the
    module docstring) but which every sweep regenerates as a repository
    resource.
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(_HEADER)

        # Summary table: rows blocked by symmetry group, with Group and t
        # spanned via \multirow within each block. Headlines the fact that
        # design strength is a property of the group, not the individual solid.
        f.write("% --- Summary ---\n")
        f.write(r"\begin{table}[H]\centering" + "\n")
        f.write(r"\setlength{\tabcolsep}{8pt}" + "\n")
        f.write(r"\renewcommand{\arraystretch}{1.25}" + "\n")
        f.write(r"\begin{tabular}{l c c c c c c c}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Solid & $V$ & Group & $t$ "
                r"& $\Phi_2$ & $\Phi_3$ & $\Phi_4$ & $\Phi_5$ \\" + "\n")
        f.write(r"\midrule" + "\n")

        items = list(solids_data.items())
        group_blocks = [
            (g, list(it))
            for g, it in groupby(items, key=lambda kv: kv[1]["group"])
        ]

        for block_idx, (group_key, block) in enumerate(group_blocks):
            n = len(block)
            t_val = block[0][1]["t_p"]
            for _, info in block:
                assert info["t_s"] == info["t_p"] == t_val, (
                    f"design-strength block invariant broken in {group_key}: "
                    f"expected t={t_val}, got t_s={info['t_s']}, t_p={info['t_p']}"
                )
            # Absolute lower bound on |X| for a complex projective t-design
            # in CP^{d-1} at d = 2: (1 + floor(t/2))(1 + ceil(t/2)). Solids
            # whose V hits this bound are tight projective t-designs.
            tight_V = (1 + t_val // 2) * (1 + (t_val + 1) // 2)
            for j, (name, info) in enumerate(block):
                display = info["display"]
                V = info["V"]
                V_cell = (rf"$\mathbf{{{V}}}$" if V == tight_V
                          else f"${V}$")
                if j == 0:
                    if n > 1:
                        group_cell = rf"\multirow{{{n}}}{{*}}{{${group_key}$}}"
                        t_cell = rf"\multirow{{{n}}}{{*}}{{{t_val}}}"
                    else:
                        group_cell = f"${group_key}$"
                        t_cell = f"{t_val}"
                else:
                    group_cell = ""
                    t_cell = ""
                phi_entries = []
                for t in (2, 3, 4, 5):
                    val = info["phi"][t]
                    saturates = saturates_welch(val, t, solid_field(name))
                    cell = f"${_fmt(val)}$"
                    if saturates:
                        cell = r"$\mathbf{" + _fmt(val) + "}$"
                    phi_entries.append(cell)
                f.write(f"  {display} & {V_cell} & {group_cell} & {t_cell} & "
                        + " & ".join(phi_entries) + r" \\" + "\n")

        # In-table footer note: surfaces the two universal facts (IC and
        # t_s = t_p) that otherwise hide in the caption.
        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{8}{c}{All five POVMs are IC; "
                r"$t_s = t_p$ throughout.} \\" + "\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        # Caption length is load-bearing: two more lines here would push this
        # [H] float onto a page of its own, where the caption as it stands lets
        # Appendix E close on the page before with ~2pt of slack. What those
        # lines would say -- t constant within each group block (Table 2.2's
        # caption states it), V and the non-saturating frame potentials varying
        # between solids -- the \multirow blocks show at sight, so it stays out.
        f.write(r"\caption{Symbolic properties of the Platonic-solid POVMs, "
                r"grouped by binary polyhedral symmetry group. Bolded $\Phi_t$ entries "
                r"saturate the Welch bound $\Phi_t \ge 1/(t+1)$ for $d=2$; "
                r"bolded $V$ entries mark tight projective $t$-designs, "
                r"hitting the absolute lower bound "
                r"$V \ge (1 + \lfloor t/2 \rfloor)(1 + \lceil t/2 \rceil)$ "
                r"at $d=2$.}" + "\n")
        f.write(r"\label{tab:povm-properties}" + "\n")
        f.write(r"\end{table}" + "\n\n")

    print(f"  properties -> {filename}")

    with open(multisets_filename, "w", encoding="utf-8") as f:
        f.write(_MULTISETS_HEADER)

        # Pairwise-overlap multiset table: a single 8-column tabular split
        # by binary polyhedral group, with a vertical rule between halves.
        # Each side contributes (Solid, V, overlap value, multiplicity);
        # solid name and V are \multirow-merged over the solid's rows so
        # each solid forms a clean visual block. 2T (Tetrahedron) and 2O
        # (Octahedron, Cube) on the left; 2I (Icosahedron, Dodecahedron)
        # on the right. Section breaks use partial \cmidrules per side.
        f.write("% --- Pairwise-overlap multisets ---\n")
        f.write(r"\begin{table}[H]\centering" + "\n")
        f.write(r"\renewcommand{\arraystretch}{1.15}" + "\n")
        f.write(r"\begin{tabular}{l c c c | l c c c}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"  Solid & $V$ & $|\braket{\psi_i}{\psi_j}|^2$ & Mult. "
                r"& Solid & $V$ & $|\braket{\psi_i}{\psi_j}|^2$ & Mult. \\"
                + "\n")
        f.write(r"\midrule" + "\n")

        def build_side(names):
            r"""One data row per overlap value. The solid name and V are
            emitted as \multirow on the first row of each solid block, and
            blank on subsequent rows; `rule_above` flags the second-and-
            later blocks for a \cmidrule on this side."""
            rows = []
            for solid_idx, name in enumerate(names):
                info = solids_data[name]
                n = len(info["angles"])
                for i, (val, count) in enumerate(info["angles"]):
                    first = (i == 0)
                    rows.append({
                        "solid": (rf"\multirow{{{n}}}{{*}}{{{info['display']}}}"
                                  if first else ""),
                        "V":     (rf"\multirow{{{n}}}{{*}}{{${info['V']}$}}"
                                  if first else ""),
                        "val":   f"${_fmt(val)}$",
                        "count": f"${count}$",
                        "rule_above": first and solid_idx > 0,
                    })
            return rows

        left = build_side(["tetrahedron", "octahedron", "cube"])
        right = build_side(["icosahedron", "dodecahedron"])

        blank = {"solid": "", "V": "", "val": "", "count": "",
                 "rule_above": False}
        while len(left) < len(right):
            left.append(blank)
        while len(right) < len(left):
            right.append(blank)

        for l, r in zip(left, right):
            if l["rule_above"]:
                f.write(r"  \cmidrule(lr){1-4}" + "\n")
            if r["rule_above"]:
                f.write(r"  \cmidrule(lr){5-8}" + "\n")
            f.write(f"  {l['solid']} & {l['V']} & {l['val']} & {l['count']}"
                    f" & {r['solid']} & {r['V']} & {r['val']} & {r['count']}"
                    r" \\" + "\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\caption{Closed-form pairwise-overlap multisets "
                r"$\{|\braket{\psi_i}{\psi_j}|^2 : i \ne j\}$ for the five "
                r"Platonic-solid POVMs, over the $\binom{V}{2}$ unordered pairs. "
                r"Solids are grouped by binary polyhedral symmetry: $\TwoT$ "
                r"and $\TwoO$ on the left, $\TwoI$ on the right. Each block "
                r"lists one solid's distinct overlap values and their "
                r"multiplicities; multiplicities sum to $\binom{V}{2}$. For "
                r"the four centrally symmetric solids (all but the "
                r"tetrahedron), the multiplicity at value $0$ equals $V/2$: "
                r"antipodal pairs are orthogonal.}" + "\n")
        f.write(r"\label{tab:povm-angle-multisets}" + "\n")
        f.write(r"\end{table}" + "\n\n")

    print(f"  angle multisets -> {multisets_filename}  (repository only; not \\input by the thesis)")


# =============================================================================
# Main
# =============================================================================

def _format_overlap(val):
    """Pretty-print an overlap-squared value (for stdout)."""
    g = _golden_sub(simplify(val), _tau_sym, _sig_sym)
    return str(g).replace("sqrt", "√")


def main():
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    print("Platonic Solid POVM Vertices and Verification")
    print("=" * 50)

    solids_data = {}
    for name, display, group, normalization in SOLIDS:
        print(f"\n{display} (group {group.replace(chr(92), '')}):")
        vertices = numeric_vertices(name)
        V = len(vertices)
        K = solid_field(name)

        verify_unit_norm(vertices, K)
        print(f"  unit norm OK ({V} vertices)")

        verify_zero_sum(vertices, K)
        print(f"  zero sum OK")

        elements = povm_elements(vertices)
        verify_completeness(elements, K)
        print(f"  completeness sum_k E_k = I OK")

        verify_ic_rank(elements, K)
        print(f"  vec'd-effect rank = 4/4 OK (IC)")

        verify_second_moment(vertices, K)
        print(f"  (1/V) sum_k rho_k^(x)2 = (1/6)(I + SWAP) OK")

        t_s = spherical_design_strength(vertices, K)
        print(f"  spherical design strength: {t_s}")

        t_p = projective_design_strength(vertices, K)
        print(f"  complex projective design strength: {t_p}")

        # Pin the ladder: both strengths fail quietly downward, so a regression
        # would otherwise land unremarked in the properties table.
        assert t_s == t_p == EXPECTED_DESIGN[name], (
            f"{name}: design strength t_s={t_s}, t_p={t_p}, "
            f"expected {EXPECTED_DESIGN[name]}"
        )

        phi = {}
        for t in (1, 2, 3, 4, 5):
            phi[t] = frame_potential(vertices, t)
            mark = ("  (saturates Welch)"
                    if saturates_welch(phi[t], t, K) else "")
            print(f"  Phi_{t} = {_format_overlap(phi[t])}{mark}")

        angles = angle_multiset(vertices, K)
        print(f"  angle multiset ({len(angles)} distinct values):")
        for val, count in angles:
            print(f"    {count:3d}  x  {_format_overlap(val)}")

        solids_data[name] = {
            "display": display,
            "group": group,
            "V": V,
            "normalization": normalization,
            "vertices": vertices,
            "elements": elements,
            "t_s": t_s,
            "t_p": t_p,
            "phi": phi,
            "angles": angles,
        }

    print("\nCross-solid verifications:")

    verts = {name: solids_data[name]["vertices"] for name in solids_data}
    dual_pairs = (
        ("octahedron",   "cube",         4),
        ("cube",         "octahedron",   3),
        ("icosahedron",  "dodecahedron", 5),
        ("dodecahedron", "icosahedron",  3),
    )
    for name_a, name_b, face_size in dual_pairs:
        verify_dual_alignment(verts[name_a], verts[name_b], face_size,
                              _RADIAL[name_a] * _RADIAL[name_b])
        print(f"  dual alignment OK: {name_a} vertices are "
              f"{name_b} face centers")
    antipodal_tetra = tuple(-v for v in verts["tetrahedron"])
    verify_dual_alignment(antipodal_tetra, verts["tetrahedron"], 3,
                          _RADIAL["tetrahedron"] ** 2)
    print("  dual alignment OK: the tetrahedron's face centers are its "
          "antipodal copy")

    # Expected verdicts pin the tabulated family to the atlas copy <X,Z,F,Phi>:
    # F (in 2T, hence in all three groups) must permute every effect set; Phi
    # must permute exactly the icosahedral pair's (its 72-degree rotation is
    # no symmetry of the other three solids); Phi* must permute none.
    expected = {
        "F":    {"tetrahedron": True,  "octahedron": True,  "cube": True,
                 "icosahedron": True,  "dodecahedron": True},
        "Phi":  {"tetrahedron": False, "octahedron": False, "cube": False,
                 "icosahedron": True,  "dodecahedron": True},
        "Phi*": {"tetrahedron": False, "octahedron": False, "cube": False,
                 "icosahedron": False, "dodecahedron": False},
    }
    for gate_name, U in (("F", _F), ("Phi", _Phi), ("Phi*", _PHI_STAR)):
        permuted = []
        for name in solids_data:
            info = solids_data[name]
            perm = conjugation_permutation(U, info["elements"],
                                           info["V"] * _RADIAL[name])
            got = perm is not None
            want = expected[gate_name][name]
            assert got == want, (
                f"conjugation by {gate_name} on {name}: "
                f"permutes={got}, expected {want}"
            )
            if got:
                permuted.append(name)
        outcome = ", ".join(permuted) if permuted else "none"
        print(f"  conjugation by {gate_name + ':':5s} permutes {outcome}")
    print("  covariance copy OK: the tabulated icosahedron and dodecahedron "
          "belong to the <X,Z,F,Phi> family")

    print()
    write_vertex_atlas(out_dir / "povm_atlas.tex", solids_data)
    write_properties(out_dir / "povm_properties.tex", solids_data,
                     out_dir / "povm_angle_multisets.tex")
    print("\nDone.")


if __name__ == "__main__":
    main()
