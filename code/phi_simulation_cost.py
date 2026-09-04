"""
Verification of the classical-simulation-cost claims made for the golden gate
in Appendix B.3.

The thesis states that sum-over-Cliffords simulators (Bravyi et al., Quantum
3, 181 (2019)) simulate a Clifford + Phi circuit in time mu(Phi)^(2m) poly(n),
where m counts Phi gates and the *gate extent* mu(Phi) is the smallest l1 norm
over decompositions of Phi into single-qubit Cliffords -- and that this extent
is itself a golden-ratio quantity. This script verifies the three claims
behind that paragraph and its footnote:

  Claim 1  The two-term decomposition
               Phi = (sigma (1+i)/2) S^3  +  (1/sqrt(2)) Z H
           is exact, so mu(Phi) <= sigma/sqrt(2) + 1/sqrt(2) = tau/sqrt(2).

  Claim 2  No Clifford decomposition does better: basis-pursuit (IRLS) over
           all 24 projective Cliffords converges to the same l1 norm, and a
           dual certificate y with ||A^H y||_inf <= 1 attains
           Re <y, vec(Phi)> = tau/sqrt(2), closing the duality gap. Hence
           mu(Phi) = tau/sqrt(2), and the per-Phi simulation cost is
           mu^2 = tau^2/2, i.e. 2^(2 log2 tau - 1) ~= 2^0.39 per gate. The
           same machinery recovers the known
           mu(T) = cos(pi/8) + (sqrt(2)-1) sin(pi/8) as a control.

  Claim 3  Phi lies outside the third level of the Clifford hierarchy:
           Phi P Phi^dag is non-Clifford for P = X, Y, Z (whereas T passes
           this test), so T-style injection with Clifford correction does not
           apply to Phi.

Claims 1 and 3 are exact statements -- an algebraic identity and a discrete
membership verdict -- and each is decided here in canonical form, by main.py's
`_to_basis`: every gate in this file lies in Q(sqrt2, sqrt5, i), which is the
atlas's own field. So is Claim 2's conclusion, settled there with exactly
2 of 24 dual constraints tight and the objective tau/sqrt(2) on the nose, so
weak duality makes mu(Phi) = tau/sqrt(2) a theorem rather than a number
certified to 1e-9. The T control's optimum is the one quantity that leaves
that field -- see EXACT_EXTENT.

The float layer searches and the exact layer decides: the banner below says
why both are kept, DUAL_WITNESS how the handover works. Writes nothing.

    cd code && uv run phi_simulation_cost.py

Exits 0 with a verification report if every claim checks out; raises
(non-zero exit) on the first failure.
"""

import numpy as np
import sympy as sp
from sympy import I as symI
from sympy import Matrix, Rational, eye, sqrt

from main import _is_zero, _to_basis   # the atlas's canonical form over
                                       # Q(sqrt2, sqrt5), real and imaginary
                                       # parts reduced separately

TAU = (1 + np.sqrt(5)) / 2          # golden ratio (Conway & Smith tau)
SIGMA = TAU - 1                     # inverse golden ratio (sigma = 1/tau)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Golden gate, Equation (eq:golden-gate) of the thesis.
PHI = 0.5 * np.array(
    [[TAU + 1j / TAU, 1], [-1, TAU - 1j / TAU]], dtype=complex
)


def proj_key(U):
    """Canonical key of a unitary modulo global phase (float-rounding grid)."""
    U = U / np.linalg.det(U) ** 0.5                 # det 1 up to sign
    flat = U.flatten()
    k = np.argmax(np.abs(flat) > 1e-9)
    U = U * (np.abs(flat[k]) / flat[k])             # first big entry > 0 real
    return tuple(np.round(U.flatten(), 9).tolist())


def clifford_group():
    """The 24 projective single-qubit Cliffords as closure of <S, H>."""
    group = {proj_key(I2): I2}
    frontier = [I2]
    while frontier:
        new = []
        for U in frontier:
            for G in (S, H):
                V = G @ U
                key = proj_key(V)
                if key not in group:
                    group[key] = V
                    new.append(V)
        frontier = new
    return group


def min_l1_over_cliffords(A, target, iters=5000, eps=1e-13):
    """Basis pursuit min ||c||_1 s.t. A c = vec(target), by IRLS (FOCUSS).

    The problem is convex, so the fixed point is the global optimum; the
    caller independently certifies it via the dual anyway.
    """
    b = target.flatten()
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(iters):
        w = np.abs(c) + eps
        AW = A * w
        y = np.linalg.solve(AW @ A.conj().T + 1e-15 * np.eye(4), b)
        c = w * (A.conj().T @ y)
    # Every guard in this module measures its deviation and bounds it: allclose
    # and isclose keep rtol=1e-5 alongside any atol, and it scales with the
    # expected operand, not the residual. Neither is called anywhere below.
    d_feas = np.abs(A @ c - b).max()      # residual ~8.7e-16
    assert d_feas < 1e-12, f"IRLS lost feasibility (max |dev| = {d_feas:.2e})"
    return c


def certified_extent(A, target, c):
    """Lower-bound mu via the dual: max Re<y,b> s.t. ||A^H y||_inf <= 1.

    A dual-feasible y is built from the support of the primal solution c
    (A_supp^H y = sign(c_supp)), rescaled into the feasible region. Weak
    duality then gives Re<y,b> <= mu for every decomposition, so a tiny
    primal-dual gap certifies optimality of ||c||_1.
    """
    b = target.flatten()
    support = np.abs(c) > 1e-8
    signs = c[support] / np.abs(c[support])
    y, *_ = np.linalg.lstsq(A[:, support].conj().T, signs, rcond=None)
    y = y / max(1.0, np.abs(A.conj().T @ y).max())   # force dual feasibility
    return np.real(np.vdot(y, b))


# =============================================================================
# The exact layer -- Claims 1 and 3, and Claim 2's conclusion, in the field
# =============================================================================
# Every gate above has entries in Q(sqrt2, sqrt5, i): H contributes 1/sqrt2, S
# and T contribute i, Phi contributes tau. Real and imaginary parts therefore
# land in Q(sqrt2, sqrt5), which is exactly the field `_to_basis` reduces to a
# 4-tuple -- so the atlas's own canonical form decides every identity and
# every membership question in this file, with no tolerance anywhere.
#
# The float pipeline above is kept and asserted to agree, never replaced. Two
# independently derived answers are worth more than one exact answer, and the
# numerical claims that remain -- the IRLS search, the duality gap -- have to
# stay runnable for the exact witness to have anything to confirm.

TAU_SYM = (1 + sqrt(5)) / 2           # golden ratio, exactly
SIG_SYM = TAU_SYM - 1                 # sigma = 1/tau = tau - 1

I2_SYM = eye(2)
X_SYM = Matrix([[0, 1], [1, 0]])
Y_SYM = Matrix([[0, -symI], [symI, 0]])
Z_SYM = Matrix([[1, 0], [0, -1]])
S_SYM = Matrix([[1, 0], [0, symI]])
H_SYM = Matrix([[1, 1], [1, -1]]) / sqrt(2)
T_SYM = Matrix([[1, 0], [0, (1 + symI) / sqrt(2)]])
PHI_SYM = Rational(1, 2) * Matrix([[TAU_SYM + symI / TAU_SYM, 1],
                                   [-1, TAU_SYM - symI / TAU_SYM]])


def _num(M):
    """A symbolic matrix as float64 numpy, for the agreement checks only."""
    return np.array(M.evalf(30), dtype=complex)


def _is_positive(value):
    """Exact sign test for a real algebraic number -- order, not equality.

    A field decides equality but not order, so the certificate's slack
    constraints need the other tool: SymPy fixes the sign by interval
    refinement and returns None when it cannot, so a True here is a proof.
    Same mechanism as `povm_properties.py`'s `_is_positive`.
    """
    sign = sp.expand(value).is_positive
    assert sign is not None, f"could not decide the sign of {value}"
    return sign


def exact_proj_key(U):
    """Canonical key of an exact 2x2 matrix modulo global phase.

    Dividing by the first nonzero entry fixes a representative of the
    projective class with no square root and no branch choice; the `_to_basis`
    tuples of the real and imaginary parts of the result are then a true
    canonical form. Two expressions denoting the same projective matrix give
    the same key, and the key is hashable -- so it can key
    `exact_clifford_group`'s `dict` exactly where `clifford_group` keys its own
    with the rounding grid.

    Which entry counts as "first nonzero" is itself decided in the field, so
    the 1e-9 threshold in `proj_key` has no counterpart here. Raises outside
    Q(sqrt2, sqrt5, i) rather than keying an approximation.
    """
    entries = [sp.expand(e) for e in U]
    pivot = next(e for e in entries if not _is_zero(e))
    key = []
    for e in entries:
        re, im = sp.expand(e / pivot).as_real_imag()
        key.extend(_to_basis(re))
        key.extend(_to_basis(im))
    return tuple(key)


def exact_clifford_group():
    """The 24 projective Cliffords as the closure of <S, H>, on canonical keys.

    Structurally the same breadth-first closure as `clifford_group`, with the
    rounding grid replaced by the canonical form -- so "have we seen this
    element already" is decided rather than measured.
    """
    group = {exact_proj_key(I2_SYM): I2_SYM}
    frontier = [I2_SYM]
    while frontier:
        new = []
        for U in frontier:
            for G in (S_SYM, H_SYM):
                V = sp.expand(G * U)
                key = exact_proj_key(V)
                if key not in group:
                    group[key] = V
                    new.append(V)
        frontier = new
    return group


def exact_is_clifford(U, cliffords):
    return exact_proj_key(U) in cliffords


# Exact dual witnesses for the two extents. Each was found by the float basis
# pursuit below and rounded to closed form; the search only proposes, and both
# dual conditions are re-decided in the field -- the same float-search /
# exact-confirm discipline `povm_properties.py` uses for its face finder.
_SIN8 = sqrt(2 - sqrt(2)) / 2         # sin(pi/8)
_COS8 = sqrt(2 + sqrt(2)) / 2         # cos(pi/8)

DUAL_WITNESS = {
    "Phi": [sqrt(2) / 3 + symI * sqrt(2) / 6, sqrt(2) / 6,
            -sqrt(2) / 6, sqrt(2) / 3 - symI * sqrt(2) / 6],
    "T": [sqrt(2) * _SIN8, sp.Integer(0), sp.Integer(0), _SIN8 * (1 + symI)],
}

# The optimum each witness attains. Phi's lies in the atlas's own field;
# the T control's does not -- cos(pi/8) generates Q(sqrt(2 - sqrt2)), a
# different degree-4 field, which contains sqrt2 but neither sqrt5 nor sqrt3.
EXACT_EXTENT = {"Phi": TAU_SYM / sqrt(2), "T": _COS8 + (sqrt(2) - 1) * _SIN8}
_EXTENT_ALIAS = {"Phi": sqrt(2) * TAU_SYM / 2, "T": sqrt(4 - 2 * sqrt(2))}
_EXTENT_LABEL = {"Phi": "tau/sqrt(2)",
                 "T": "cos(pi/8) + (sqrt2 - 1) sin(pi/8) = sqrt(4 - 2 sqrt2)"}

_T_FIELD = sp.QQ.algebraic_field(sqrt(2 - sqrt(2)))

# The optimal decompositions, as (coefficient, Clifford, |coefficient|) --
# support 2 in each case, found the same way as the witnesses above. The
# magnitudes are declared rather than computed because sqrt(re^2 + im^2)
# presents them as nested radicals that are in the field but not in
# `_to_basis`'s {1, sqrt2, sqrt5, sqrt10} shape; so each is pinned by its
# SQUARE, which is in shape, plus its sign.
EXACT_DECOMP = {
    "Phi": ((SIG_SYM * (1 + symI) / 2, S_SYM ** 3, SIG_SYM / sqrt(2)),
            (1 / sqrt(2), Z_SYM * H_SYM, 1 / sqrt(2))),
    "T": ((Rational(1, 2) + symI * (sqrt(2) - 1) / 2, I2_SYM,
           sqrt(4 - 2 * sqrt(2)) / 2),
          (Rational(1, 2) - symI * (sqrt(2) - 1) / 2, S_SYM,
           sqrt(4 - 2 * sqrt(2)) / 2)),
}


def _extent_is_zero(name, value):
    """Zero test for an extent residual, in the field that extent needs."""
    if name == "T":
        return _T_FIELD.from_sympy(value) == _T_FIELD.zero
    return _is_zero(value)


def check_exact_decomposition(name, target, exact_cliffords):
    """The upper bound: an explicit Clifford decomposition of l1 norm mu.

    Four things, and the third is the one an eyeball skips: the terms sum to
    the target entry for entry; every declared magnitude squares to the
    coefficient's; every term's matrix really is one of the 24 projective
    Cliffords the minimisation ranges over -- a decomposition over anything
    else would bound nothing; and the l1 norm is the claimed extent. With
    `check_exact_extent`'s lower bound this closes mu exactly.
    """
    l1 = sp.Integer(0)
    total = sp.zeros(2, 2)
    for coeff, C, mag in EXACT_DECOMP[name]:
        re, im = sp.expand(coeff).as_real_imag()
        assert _is_zero(sp.expand(re ** 2 + im ** 2 - mag ** 2)), (
            f"{name}: declared magnitude is not |coefficient|"
        )
        assert _is_positive(mag), f"{name}: magnitude must be the positive root"
        assert exact_is_clifford(C, exact_cliffords), (
            f"{name}: decomposition uses a matrix that is not a Clifford"
        )
        total += coeff * C
        l1 += mag
    assert all(_is_zero(e) for e in sp.expand(total - target)), (
        f"{name}: decomposition does not sum to the target"
    )
    assert _extent_is_zero(name, sp.expand(l1 - EXACT_EXTENT[name])), (
        f"{name}: the decomposition's l1 norm is not the claimed extent"
    )
    return len(EXACT_DECOMP[name])


def check_symbolic_gates_match():
    """The symbolic gates are the float ones, entry for entry.

    A boolean exact check does not test its own transcription, so every gate
    the exact layer uses is pinned against the float gate it is meant to be
    before any verdict rests on it.
    """
    pairs = ((I2_SYM, I2), (X_SYM, X), (Y_SYM, Y), (Z_SYM, Z), (S_SYM, S),
             (H_SYM, H), (T_SYM, T), (PHI_SYM, PHI))
    worst = max(np.abs(_num(A) - B).max() for A, B in pairs)
    assert worst < 1e-15, f"symbolic gate does not match its float twin: {worst:.2e}"
    return worst


def check_exact_cliffords(cliffords):
    """The exact Clifford group reproduces the float one, element for element.

    Matched by value rather than by size: each exact Clifford must be one of
    the float Cliffords up to phase, |tr(A^dag B)| = 2, and the correspondence
    must be a bijection. Two distinct projective Cliffords never overlap above
    sqrt2, so hits and misses are 2 - sqrt2 = 0.59 apart, and both bars are set
    inside that gap rather than on it: the matched overlap lands 4e-16 under 2
    against a 1e-12 bar, the best non-match at sqrt2, 0.086 under 1.5.
    """
    exact = exact_clifford_group()
    assert len(exact) == 24, f"exact Clifford closure gave {len(exact)}"
    matched, worst_hit, best_miss = {}, 2.0, 0.0
    floats = list(cliffords.values())
    for key, U in exact.items():
        num = _num(U)
        overlaps = [abs(np.trace(num.conj().T @ C)) for C in floats]
        j = int(np.argmax(overlaps))
        assert j not in matched, "two exact Cliffords matched one float Clifford"
        matched[j] = key
        worst_hit = min(worst_hit, overlaps[j])
        best_miss = max(best_miss, max(o for k, o in enumerate(overlaps) if k != j))
    assert len(matched) == 24
    assert worst_hit > 2 - 1e-12 and best_miss < 1.5, (worst_hit, best_miss)
    return exact, worst_hit, best_miss


def check_exact_extent(name, target, exact_cliffords):
    """Verify the dual certificate for mu(target) in the field.

    Weak duality: if target = sum_j c_j C_j is any Clifford decomposition and
    y satisfies |<C_j, y>| <= 1 for every Clifford C_j, then

        Re <y, target> = sum_j Re(c_j <y, C_j>) <= sum_j |c_j| = ||c||_1,

    so a feasible y lower-bounds mu(target) by Re <y, target>. Feasibility is
    phase-invariant in C_j, which is what makes the 24 projective classes the
    right index set rather than a choice of representatives.

    The two conditions are questions of different kinds and take different
    tools: feasibility is an ORDER question, settled by `_is_positive`, and
    the objective an EQUALITY one, settled by the field. Returns the number of
    tight constraints and the largest slack value, so the report asserts the
    margin rather than a threshold.
    """
    y = DUAL_WITNESS[name]
    tight, slack = 0, []
    for C in exact_cliffords.values():
        entries = list(C)
        ip = sp.expand(sum(sp.conjugate(entries[i]) * y[i] for i in range(4)))
        re, im = ip.as_real_imag()
        mag2 = sp.expand(re ** 2 + im ** 2)
        if _is_zero(sp.expand(mag2 - 1)):
            tight += 1
        else:
            assert _is_positive(1 - mag2), f"{name}: dual witness infeasible"
            slack.append(mag2)
    assert tight > 0, f"{name}: no constraint is tight, so y is not optimal"

    entries = list(target)
    obj = sp.expand(sum(sp.conjugate(y[i]) * entries[i] for i in range(4)))
    re, im = obj.as_real_imag()
    assert _extent_is_zero(name, sp.expand(re - EXACT_EXTENT[name])), (
        f"{name}: dual objective is not the claimed extent"
    )
    assert _extent_is_zero(name, sp.expand(im)), f"{name}: objective not real"
    return tight, max(slack, key=lambda v: float(v.evalf(30)))


def is_clifford(U, cliffords):
    return proj_key(U) in cliffords


def main():
    cliffords = clifford_group()
    assert len(cliffords) == 24, f"expected 24 Cliffords, got {len(cliffords)}"
    print(f"[ok] projective Clifford group <S, H>: {len(cliffords)} elements")

    # The exact companion, and the transcription guards that make it mean
    # something: the symbolic gates are pinned against their float twins, and
    # the two closures are matched element for element rather than by size.
    worst_gate = check_symbolic_gates_match()
    exact, hit, miss = check_exact_cliffords(cliffords)
    print(f"[ok] exact closure over Q(sqrt2, sqrt5, i): {len(exact)} elements on"
          " canonical keys, in bijection with the float group")
    print(f"     (gates agree to {worst_gate:.0e}; matched overlap {hit:.12f} = 2,"
          f" runner-up {miss:.4f} = sqrt2)")

    # residuals 1.3e-17 (unitarity) and 1.2e-16 (det)
    d_uni = np.abs(PHI @ PHI.conj().T - I2).max()
    d_det = abs(np.linalg.det(PHI) - 1)
    assert d_uni < 1e-12, f"Phi is not unitary (float; max |dev| = {d_uni:.2e})"
    assert d_det < 1e-12, f"det Phi != 1 (float; |dev| = {d_det:.2e})"
    assert not is_clifford(PHI, cliffords), "Phi must be non-Clifford"
    assert all(_is_zero(e) for e in
               sp.expand(PHI_SYM * PHI_SYM.conjugate().T - I2_SYM)), (
        "Phi is not unitary"
    )
    assert _is_zero(sp.expand(PHI_SYM.det() - 1)), "det Phi != 1"
    assert not exact_is_clifford(PHI_SYM, exact), "Phi must be non-Clifford"
    print("[ok] Phi is in SU(2) and is not a Clifford gate (both exactly)")

    # --- Claim 3: Clifford-hierarchy level 3 membership ------------------
    # T conjugates every Pauli into the Clifford group; Phi conjugates *no*
    # X, Y, Z into it (as stated in the thesis), which is
    # stronger than mere non-membership in level 3. Membership is a discrete
    # verdict, so it is decided on canonical keys and merely corroborated on
    # the rounding grid.
    for name, U, V, expected in (("T", T, T_SYM, True), ("Phi", PHI, PHI_SYM, False)):
        for pname, P, Q in (("X", X, X_SYM), ("Y", Y, Y_SYM), ("Z", Z, Z_SYM)):
            conj_clifford = is_clifford(U @ P @ U.conj().T, cliffords)
            conj_exact = exact_is_clifford(
                sp.expand(V * Q * V.conjugate().T), exact
            )
            assert conj_clifford == conj_exact == expected, (
                f"{name}: level-3 test surprised us on Pauli {pname}"
            )
    print("[ok] T lies in the third level of the Clifford hierarchy;"
          " Phi P Phi^dag is non-Clifford for P = X, Y, Z")

    # --- Claim 1: the explicit two-term decomposition is exact ------------
    S3 = S @ S @ S
    decomp = (SIGMA * (1 + 1j) / 2) * S3 + (1 / np.sqrt(2)) * (Z @ H)
    # residuals 1.2e-16 (decomposition) and 0.0 (mu)
    d_dec = np.abs(decomp - PHI).max()
    assert d_dec < 1e-14, f"decomposition != Phi (max |dev| = {d_dec:.2e})"
    mu_claimed = SIGMA / np.sqrt(2) + 1 / np.sqrt(2)
    d_mu = abs(mu_claimed - TAU / np.sqrt(2))                 # residual 0.0
    assert d_mu < 1e-14, (
        f"sigma/sqrt2 + 1/sqrt2 != tau/sqrt2 (|dev| = {d_mu:.2e})"
    )
    decomp_sym = ((SIG_SYM * (1 + symI) / 2) * S_SYM ** 3
                  + (1 / sqrt(2)) * (Z_SYM * H_SYM))
    assert all(_is_zero(e) for e in sp.expand(decomp_sym - PHI_SYM)), (
        "decomposition != Phi"
    )
    assert _is_zero(sp.expand(SIG_SYM / sqrt(2) + 1 / sqrt(2) - TAU_SYM / sqrt(2)))
    print("[ok] Phi = (sigma(1+i)/2) S^3 + (1/sqrt 2) ZH exactly,"
          " so mu(Phi) <= tau/sqrt(2) -- an identity in Q(sqrt2, sqrt5, i)")

    # --- Claim 2: nothing beats it (primal-dual certificate) --------------
    A = np.column_stack([C.flatten() for C in cliffords.values()])
    report = {}
    for name, U, mu_closed in (
        ("T", T, np.cos(np.pi / 8) + (np.sqrt(2) - 1) * np.sin(np.pi / 8)),
        ("Phi", PHI, TAU / np.sqrt(2)),
    ):
        c = min_l1_over_cliffords(A, U)
        mu = np.abs(c).sum()
        lower = certified_extent(A, U, c)
        assert mu - lower < 1e-9, f"{name}: duality gap {mu - lower:.2e}"
        assert abs(mu - mu_closed) < 1e-9, (   # LP residual ~9.7e-13
            f"{name}: mu = {mu} != closed form {mu_closed}"
        )
        report[name] = mu
        print(f"[ok] mu({name}) = {mu:.12f} matches closed form"
              f" (duality gap {mu - lower:.1e})")

    # The same lower bound, exactly: dual feasibility over all 24 projective
    # classes and the objective on the nose, which with Claim 1's upper bound
    # closes mu = tau/sqrt(2) as a theorem, not as a gap below 1e-9.
    for name, target in (("T", T_SYM), ("Phi", PHI_SYM)):
        assert _extent_is_zero(
            name, sp.expand(EXACT_EXTENT[name] - _EXTENT_ALIAS[name])
        ), f"{name}: the two spellings of the extent are not the same number"
        terms = check_exact_decomposition(name, target, exact)
        y = np.array([complex(v.evalf(30)) for v in DUAL_WITNESS[name]])
        drift = np.abs(np.abs(A.conj().T @ y).max() - 1.0)
        assert drift < 1e-12, f"{name}: rounded witness left the feasible set"
        tight, slack = check_exact_extent(name, target, exact)
        print(f"[ok] mu({name}) = {_EXTENT_LABEL[name]}, exactly and both ways:"
              f" a {terms}-term Clifford")
        print(f"     decomposition of that l1 norm above, and below a dual witness"
              f" with {tight} of 24")
        print(f"     constraints tight (largest slack |<C,y>|^2 = {slack})."
              " No tolerance either side")

    # gamma = 2 log2 mu, so mu(Phi)^2 = tau^2/2 and mu(T)^2 = 4 - 2 sqrt2.
    # Corollaries, not pins: check_exact_extent decides both in the field.
    # Residuals ~2.4e-12, set by the LP's precision on mu (~9e-13), not by
    # float roundoff, so 1e-9 is the honest bound.
    gamma_phi = 2 * np.log2(report["Phi"])
    gamma_t = 2 * np.log2(report["T"])
    d_phi = abs(gamma_phi - (2 * np.log2(TAU) - 1))
    d_t = abs(gamma_t - np.log2(4 - 2 * np.sqrt(2)))
    assert d_phi < 1e-9, f"gamma_Phi off 2 log2 tau - 1 by {d_phi:.2e}"
    assert d_t < 1e-9, f"gamma_T off log2(4 - 2 sqrt2) by {d_t:.2e}"
    print(f"[ok] per-gate simulation cost: Phi 2^{gamma_phi:.4f}"
          f" (= 2^(2 log2 tau - 1)) vs T 2^{gamma_t:.4f}")

    print("\nAll simulation-cost claims verified.")


if __name__ == "__main__":
    main()
