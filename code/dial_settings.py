"""All eight settings of the three binary dials of Appendix E.2.

Appendix E.2 names three binary dials that decide the icosahedral orientation,
and asserts that setting any two fixes the third. This script enumerates the
full 2^3 cube and shows the assertion is a *parity law*: a setting is consistent
exactly when an even number of dials is flipped -- four of the eight work.

  Dial 1  SEED        the icosahedron's seed (tau, 1, 0)  [EVEN: cyclic = A_3]
                      or its transpose (1, tau, 0)        [ODD]
  Dial 2  600-CELL    the completion of the 24-cell by even coordinate
                      permutations of (1/2, tau/2, sigma/2, 0) [EVEN] or by
                      odd ones [ODD]
  Dial 3  CONVENTION  the quaternion-to-rotation map: ours, q turning about
                      -(z, y, x) [EVEN], or the textbook's, about (x, y, z) [ODD]

"Consistent" means covariant: the projected group permutes the vertex set, so
one group copy and one set of atlas circuits serve the POVM. That is a statement
about the projected GROUP, and it does not say the vertex set stays put: a dial
can move a solid's coordinates and leave its covariance intact. Claim 4's
tetrahedron is the one place in the atlas where the two come apart.

Verified here, all exactly over Q(sqrt2, sqrt5, i) -- no floats decide anything
(the file's one float is the evalf(20) inside `dirs()`, which ranks magnitudes
to pick a scaling representative and decides no verdict):

  Claim 1  The geometric 600-cells ARE the algebraic copies: even permutations
           give <X, Z, F, Phi> element for element, odd ones <X, Z, F, Phi*>.
           (This is dial 2 stated twice, geometrically and algebraically.)

  Claim 2  The two conventions differ by exactly the relabel M: (x,y,z) ->
           -(z,y,x), the half-turn about xhat - zhat, element for element
           across both copies (216 distinct: they share their 2T). M is the
           Bloch rotation of the Clifford Z H Z.

  Claim 3  THE PARITY LAW. For the icosahedron and the dodecahedron alike, a
           setting is covariant iff an even number of its three dials is ODD.
           So: even seed / even 600-cell / our convention works (the atlas);
           odd / odd / ours works too; odd 600-cell with the even seed works,
           but only under the textbook convention. Setting any two dials
           determines the third, and there are four consistent worlds, not one.

  Claim 4  The tetrahedron, octahedron and cube are covariant in all eight
           settings. Covariant, not untouched. The seed transposition fixes all
           three vertex sets (and carries each of the other two solids to its
           other family), but the relabel M fixes only the octahedron's and the
           cube's: it carries the tetrahedron onto the OTHER tetrahedron
           inscribed in the same cube -- our four vertices negated, equivalently
           the cube's remaining four. Covariance survives that, because T is
           normal in O, so the projected group comes back to itself and permutes
           the new set just as well. What does not survive is the vertex LIST.
           (This split is Appendix E.2's "immune" paragraph.)

  Claim 5  Dual alignment (each icosahedron vertex the center of a dodecahedral
           face, and vice versa) is decided by the SEED dial ALONE. It is a
           statement about two vertex sets, so no group or convention can
           restore it: a mixed pair is unaligned in all eight settings.

  Claim 6  Dial 3 is not really binary, and it does not matter. Anchoring
           restricts the candidate relabelings to an O-copy, 24 of them. Every
           one of the 24 sends the icosahedral family to a family: the 12 in T
           preserve it, the 12 in O \\ T flip it. So the convention dial has 24
           settings but two outcomes, indexed by the class in O/T = Z_2 -- the
           sign character once more, and literally: each relabeling is a signed
           permutation of the three axes, and its class is the sign of that
           permutation (Appendix E.2's footnote). All three dials are the same
           Z_2. That same class decides the tetrahedron: the 12 in T fix its vertex set and the
           12 outside swap it for the other tetrahedron -- the same 12 and 12,
           relabeling for relabeling, that preserve and flip the icosahedral
           family.

  Claim 7  The Galois flip sqrt5 -> -sqrt5 carries each family to the other
           (as directions), for the icosahedron and dodecahedron alike: the
           substitution that swaps the two anchored copies swaps the two vertex
           families as well, which is the parity law's "flipping one dial flips
           the other" in one substitution.

    cd code && uv run dial_settings.py

Exits 0 with the dial table and a verification report; raises on first failure.
Writes nothing.
"""

from itertools import permutations, product

from sympy import Matrix, Rational, eye, expand, sqrt, trace

from main import _SU2_GATES, _to_basis, Quaternion, SIGMA, TAU, symI
from phi_star_copies import canon, close, key, star
from povm_properties import verify_dual_alignment

# =============================================================================
# Exact identity for real vectors over Q(sqrt2, sqrt5)
# =============================================================================
# Rotation matrix entries and vertex coordinates are real members of
# Q(sqrt2, sqrt5), where main.py's `_to_basis` is a true canonical form. Vertex
# sets are therefore compared as frozensets of coordinate triples -- an identity
# test, not a tolerance. Vertices are kept UNNORMALIZED (the atlas divides by
# sqrt(3) or sqrt(2+tau), which leave the field); rotations preserve length, so
# a permutation check is indifferent to the common scale.


def vkey(v):
    return tuple(_to_basis(c) for c in v)


def vset(vs):
    return frozenset(vkey(v) for v in vs)


# =============================================================================
# Dial 1: the two vertex families
# =============================================================================

def cyclic_orbit(seed):
    """Orbit under cyclic coordinate permutations and all sign flips."""
    a, b, c = seed
    out = set()
    for p in ((a, b, c), (c, a, b), (b, c, a)):
        for s in product([1, -1], repeat=3):
            out.add(tuple(si * pi for si, pi in zip(s, p)))
    return [Matrix(list(v)) for v in out]


CUBE = [Matrix([x, y, z]) for x, y, z in product([1, -1], repeat=3)]
TETRA = [Matrix(v) for v in ((1, 1, 1), (-1, -1, 1), (-1, 1, -1), (1, -1, -1))]
# The cube's other inscribed tetrahedron. Defined as the antipodes of ours; that
# this is also the cube's remaining four vertices is asserted, not assumed.
TETRA_OTHER = [-v for v in TETRA]
# The seed dial's move on coordinates: transposing the last two, which is what
# carries the seed (0, tau, 1) to (0, 1, tau).
P_SEED = Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

FAMILY = {
    ("icosahedron", "EVEN"): cyclic_orbit((0, TAU, 1)),
    ("icosahedron", "ODD"): cyclic_orbit((0, 1, TAU)),
    ("dodecahedron", "EVEN"): CUBE + cyclic_orbit((0, SIGMA, TAU)),
    ("dodecahedron", "ODD"): CUBE + cyclic_orbit((0, TAU, SIGMA)),
}
for par in ("EVEN", "ODD"):     # the three immune solids ignore the seed
    FAMILY[("octahedron", par)] = cyclic_orbit((1, 0, 0))
    FAMILY[("cube", par)] = CUBE
    FAMILY[("tetrahedron", par)] = TETRA

SOLIDS = ("icosahedron", "dodecahedron", "tetrahedron", "octahedron", "cube")
NVERTS = {"icosahedron": 12, "dodecahedron": 20, "octahedron": 6, "cube": 8, "tetrahedron": 4}


# =============================================================================
# Dial 3: the two conventions
# =============================================================================

PAULI = (Matrix([[0, 1], [1, 0]]),
         Matrix([[0, -symI], [symI, 0]]),
         Matrix([[1, 0], [0, -1]]))

# The relabel separating the conventions: (x,y,z) -> -(z,y,x), the half-turn
# about xhat - zhat.
M = Matrix([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])


def rot_ours(U):
    """R_ab = 1/2 Tr[sigma_a U sigma_b U^dag]: conjugation on the labeled Paulis."""
    return Matrix(3, 3, lambda a, b: expand(
        Rational(1, 2) * trace(PAULI[a] * U * PAULI[b] * U.H)))


def rot_textbook(U):
    """The textbook map: q = w + xi + yj + zk turns about (x, y, z), with the
    quaternion units identified naturally as i, j, k == xhat, yhat, zhat."""
    w, x = U[0, 0].as_real_imag()
    y, z = U[0, 1].as_real_imag()
    v = Matrix([x, y, z])
    cross = Matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    R = (w**2 - (x**2 + y**2 + z**2)) * eye(3) + 2 * v * v.T + 2 * w * cross
    return R.applyfunc(expand)


CONV = {"EVEN": rot_ours, "ODD": rot_textbook}


# =============================================================================
# Dial 2: the two anchored copies of 2I, geometric and algebraic
# =============================================================================

def geometric_600cell(parity):
    """24-cell + 96 vertices from the even/odd permutations of the base
    coordinate (1/2, tau/2, sigma/2, 0). Returns canonical keys."""
    qs = []
    for i in range(4):                                    # +-1, +-i, +-j, +-k
        for s in (1, -1):
            v = [0] * 4
            v[i] = s
            qs.append(Quaternion(*v))
    for s in product([Rational(1, 2), Rational(-1, 2)], repeat=4):
        qs.append(Quaternion(*s))
    vals = [Rational(1, 2), TAU / 2, SIGMA / 2, Rational(0)]
    for p in permutations(range(4)):
        inversions = sum(1 for i in range(4) for j in range(i + 1, 4) if p[i] > p[j])
        if (inversions % 2 == 0) != (parity == "EVEN"):
            continue
        p_vals = [vals[p[i]] for i in range(4)]
        zero_pos = p.index(3)
        for signs in product([1, -1], repeat=3):
            v, ptr = [0] * 4, 0
            for i in range(4):
                if i == zero_pos:
                    continue
                v[i] = p_vals[i] * signs[ptr]
                ptr += 1
            qs.append(Quaternion(*v))
    return {key(canon(q.to_unitary())) for q in qs}


# =============================================================================
# Verification
# =============================================================================

def permutes(group, conv, verts):
    """Does the projected group permute this vertex set? (i.e. is it covariant)"""
    target = vset(verts)
    for U in group.values():
        R = CONV[conv](U)
        if frozenset(vkey(R * v) for v in verts) != target:
            return False
    return True


def main():
    X, Z, F, S, H = (_SU2_GATES[g] for g in ("X", "Z", "F", "S", "H"))
    Phi = _SU2_GATES["Φ"]

    for (name, par), vs in FAMILY.items():
        assert len(vset(vs)) == NVERTS[name], f"{name}/{par} has the wrong vertex count"
    assert vset(FAMILY[("icosahedron", "EVEN")]) & vset(FAMILY[("icosahedron", "ODD")]) == frozenset(), \
        "the two icosahedral families should be disjoint"
    assert vset(FAMILY[("dodecahedron", "EVEN")]) & vset(FAMILY[("dodecahedron", "ODD")]) == vset(CUBE), \
        "the two dodecahedral families should share exactly the cube"
    print("[ok] families built: the icosahedra disjoint, the dodecahedra sharing"
          " exactly the cube")

    # --- Claim 1: geometry and algebra name the same two copies --------------
    COPY = {"EVEN": close([X, Z, F, Phi]), "ODD": close([X, Z, F, star(Phi)])}
    for parity in ("EVEN", "ODD"):
        assert len(COPY[parity]) == 120, f"the {parity} copy has order {len(COPY[parity])}"
        assert geometric_600cell(parity) == set(COPY[parity]), \
            f"the {parity}-permutation 600-cell is not its algebraic copy"
    print("[ok] the geometric 600-cells are the algebraic copies, element for"
          " element: even perms = <X,Z,F,Phi>, odd perms = <X,Z,F,Phi*>")

    # --- Claim 2: the conventions differ by exactly M ------------------------
    for parity in ("EVEN", "ODD"):
        for U in COPY[parity].values():
            assert rot_textbook(U) == (M * rot_ours(U) * M).applyfunc(expand), \
                "the two conventions differ by more than the relabel M"
    assert rot_ours(Z * H * Z) == M, "M should be the Bloch rotation of Z H Z"
    print("[ok] R_textbook = M R_ours M on both copies, element for element"
          " (216 distinct); M = phi(ZHZ), the xhat-zhat half-turn")

    # --- Claims 3 and 4: the dial cube ---------------------------------------
    COV = {"icosahedron": COPY, "dodecahedron": COPY,
           "tetrahedron": {p: close([X, Z, F]) for p in ("EVEN", "ODD")},
           "octahedron": {p: close([X, Z, F, S]) for p in ("EVEN", "ODD")},
           "cube": {p: close([X, Z, F, S]) for p in ("EVEN", "ODD")}}

    print()
    print("  seed   600-cell  convention  | icosahedron  dodecahedron  T / O / cube")
    print("  " + "-" * 74)
    results = {}
    for seed, cell, conv in product(["EVEN", "ODD"], repeat=3):
        row = {}
        for name in SOLIDS:
            row[name] = permutes(COV[name][cell], conv, FAMILY[(name, seed)])
            results[(name, seed, cell, conv)] = row[name]
        immune = all(row[n] for n in ("tetrahedron", "octahedron", "cube"))
        mark = lambda b: " yes " if b else "  no "
        print(f"  {seed:6s} {cell:9s} {conv:11s} |{mark(row['icosahedron']):^12s}"
              f"{mark(row['dodecahedron']):^14s}{mark(immune):^13s}")
    print()

    for name in ("icosahedron", "dodecahedron"):
        for seed, cell, conv in product(["EVEN", "ODD"], repeat=3):
            n_odd = sum(1 for d in (seed, cell, conv) if d == "ODD")
            assert results[(name, seed, cell, conv)] == (n_odd % 2 == 0), \
                f"{name} breaks the parity law at {seed}/{cell}/{conv}"
    print("[ok] PARITY LAW: covariant iff an even number of dials is ODD --"
          " four consistent settings of eight, so any two dials fix the third")

    for name in ("tetrahedron", "octahedron", "cube"):
        assert all(results[(name, s, c, v)] for s, c, v in product(["EVEN", "ODD"], repeat=3)), \
            f"the {name} should be covariant at all eight dial settings"

    # Covariant is not untouched. Two of the three dials move coordinates, and
    # they do not move the same ones: the seed transposition fixes all three of
    # these vertex sets, the relabel M fixes only two. (Appendix E.2's "immune"
    # paragraph turns on exactly this, so it is checked rather than assumed --
    # FAMILY hands the three solids the same vertex set at both seed parities.)
    for name in ("tetrahedron", "octahedron", "cube"):
        vs = FAMILY[(name, "EVEN")]
        assert frozenset(vkey(P_SEED * v) for v in vs) == vset(vs), \
            f"the seed transposition should fix the {name}'s vertex set"
    for name in ("icosahedron", "dodecahedron"):        # and move the other two
        assert frozenset(vkey(P_SEED * v) for v in FAMILY[(name, "EVEN")]) \
            == vset(FAMILY[(name, "ODD")]), \
            f"the seed transposition should carry the {name}'s EVEN family to its ODD one"

    assert vset(TETRA_OTHER) == vset(CUBE) - vset(TETRA), \
        "the other tetrahedron should be the cube's remaining four vertices"
    for name in ("octahedron", "cube"):
        vs = FAMILY[(name, "EVEN")]
        assert frozenset(vkey(M * v) for v in vs) == vset(vs), \
            f"M should fix the {name}'s vertex set"
    assert frozenset(vkey(M * v) for v in TETRA) == vset(TETRA_OTHER), \
        "M should carry our tetrahedron onto the other one"
    assert permutes(COV["tetrahedron"]["EVEN"], "EVEN", TETRA_OTHER), \
        "2T should permute the other tetrahedron too -- T is normal in O"

    print("[ok] the tetrahedron, octahedron and cube are covariant in all eight"
          " settings -- covariant, not untouched: the seed transposition fixes"
          " all three vertex sets, M fixes only the octahedron's and the cube's")
    print("[ok] M carries our tetrahedron onto the cube's other four vertices,"
          " and 2T permutes that tetrahedron just as well: the group returns,"
          " the vertex list does not")

    # --- Claim 5: dual alignment answers to the seed dial alone --------------
    for si, sd in product(["EVEN", "ODD"], repeat=2):
        try:
            verify_dual_alignment(FAMILY[("icosahedron", si)], FAMILY[("dodecahedron", sd)], 5)
            verify_dual_alignment(FAMILY[("dodecahedron", sd)], FAMILY[("icosahedron", si)], 3)
            aligned = True
        except AssertionError:
            aligned = False
        assert aligned == (si == sd), \
            f"dual alignment should hold exactly within a family (icosa {si}, dodeca {sd})"
    print("[ok] dual alignment holds exactly within a seed family, and no setting"
          " of dials 2 or 3 can restore it for a mixed pair")

    # --- Claim 6: the convention dial has 24 settings but two outcomes -------
    O3 = {tuple(vkey(rot_ours(U)[:, j]) for j in range(3)): rot_ours(U)
          for U in close([X, Z, F, S]).values()}
    T3 = {tuple(vkey(rot_ours(U)[:, j]) for j in range(3))
          for U in close([X, Z, F]).values()}
    assert len(O3) == 24 and len(T3) == 12
    ico_e = FAMILY[("icosahedron", "EVEN")]
    tgt = {"EVEN": vset(ico_e), "ODD": vset(FAMILY[("icosahedron", "ODD")])}
    tet = {"FIX": vset(TETRA), "SWAP": vset(TETRA_OTHER)}
    fixing = flipping = 0
    for k, R in O3.items():
        img = frozenset(vkey(R * v) for v in ico_e)
        assert img in (tgt["EVEN"], tgt["ODD"]), \
            "a relabeling sent the icosahedron to neither family"
        img_tet = frozenset(vkey(R * v) for v in TETRA)
        assert img_tet in (tet["FIX"], tet["SWAP"]), \
            "a relabeling sent the tetrahedron to neither tetrahedron"
        assert (img_tet == tet["FIX"]) == (img == tgt["EVEN"]), \
            "the tetrahedron and the icosahedron disagree on a relabeling"
        P = R.applyfunc(abs)        # forget the signs: the axis permutation alone
        assert P * P.T == eye(3), \
            "a relabeling should be a signed permutation of the axes"
        assert (P.det() == 1) == (k in T3), \
            "the class in O/T should be the sign of the axis permutation"
        if img == tgt["EVEN"]:
            fixing += 1
            assert k in T3, "a family-preserving relabeling outside T"
        else:
            flipping += 1
            assert k not in T3, "a family-flipping relabeling inside T"
    assert fixing == flipping == 12
    kM = tuple(vkey(M[:, j]) for j in range(3))
    assert kM in O3 and kM not in T3, "the textbook relabel should sit in O \\ T"
    # The T-vs-T-dagger quarter turn about z-hat: same coset, so same swap.
    RZ90 = Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert frozenset(vkey(RZ90 * v) for v in TETRA) == vset(TETRA_OTHER), \
        "the quarter turn should carry our tetrahedron onto the other one"
    kQ = tuple(vkey(RZ90[:, j]) for j in range(3))
    assert kQ in O3 and kQ not in T3, "the quarter turn should sit in O \\ T"
    print(f"[ok] the convention dial has {len(O3)} settings but two outcomes:"
          f" {fixing} preserve the family (exactly T), {flipping} flip it"
          f" (exactly O \\ T) -- the class in O/T = Z_2, the sign of each"
          " relabeling's own axis permutation: the same sign again")
    print(f"[ok] the same {fixing} and {flipping} decide the tetrahedron,"
          " relabeling for relabeling: T fixes its vertex set, O \\ T swaps it"
          " for the other tetrahedron")

    # --- Claim 7: one substitution flips both parities -----------------------
    def dirs(vs):
        """Scale-free identity of a centrally symmetric vertex set."""
        out = set()
        for v in vs:
            m = max([abs(c) for c in v], key=lambda e: float(e.evalf(20)))
            u = (v / m).applyfunc(expand)
            out.add(min(vkey(u), vkey(-u)))
        return frozenset(out)

    for name in ("icosahedron", "dodecahedron"):
        img = dirs([v.applyfunc(lambda e: expand(e.subs(sqrt(5), -sqrt(5))))
                    for v in FAMILY[(name, "EVEN")]])
        assert img == dirs(FAMILY[(name, "ODD")]), \
            f"sqrt5 -> -sqrt5 should carry the {name}'s EVEN family to its ODD one"
        assert img != dirs(FAMILY[(name, "EVEN")])
    print("[ok] sqrt5 -> -sqrt5 carries each family to the other, icosahedron and"
          " dodecahedron alike: one substitution, both parities")

    print("\nAll dial-setting claims verified.")


if __name__ == "__main__":
    main()
