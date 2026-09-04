"""
Verification of the anchor and the Phi-star convention.

Section "Three Routes, One Anchored Group Copy" pins a copy of a binary
polyhedral group -- SU(2) holds infinitely many conjugates of each -- by
requiring it to contain the symmetrized 2T = <X, Z, F>, the eight Paulis
included, and then asks how constraining that is: one anchored copy of 2T, one
of 2O, two of 2I. Unlike 2T and 2O, the binary icosahedral group has *two*
anchored realizations, related by Clifford conjugation. The thesis commits to
the <X, Z, F, Phi> copy (the atlas of Appendix A and everything downstream),
and names the other one via a second golden gate Phi*, defined in Section
"Canonical Generators" (Equation (eq:golden-gate) and its footnote). Verified
here: the claims made about that second copy, and the three counts.

  Claim 1  Our Phi* is the entry-wise image of Phi under sqrt5 -> -sqrt5,
           and equals the closed form printed in Eq. (eq:golden-gate),
               Phi* = 1/2 [[-sigma - i tau, 1], [-1, -sigma + i tau]].

  Claim 2  Kubischta & Teixeira obtain *their* Phi* from Phi by the same
           substitution and then a complex conjugate. The resulting matrix
           differs from ours; theirs is Z Phi*^dag Z^dag (which holds for the
           symmetrized and the standard Z alike, the phase cancelling in the
           conjugation).

  Claim 3  Both nevertheless generate the same target, element for element:
               <X, Z, F, Phi*_ours> = <X, Z, F, Phi*_KT>,
           a single subgroup of order 120. Claim 2 is the reason: Z and
           Phi*^dag both lie in <X, Z, F, Phi*>, so K&T's generator lies in
           our copy and their group is contained in ours -- and equal to it by
           order. The complex conjugate is absorbed by a Clifford the group
           already owns. Hence the choice of convention is immaterial, as the
           footnote claims.

  Claim 4  The Phi*-copy really is the *other* anchored copy: it is distinct
           from <X, Z, F, Phi>, meets it in exactly 2T = <X, Z, F> (24
           elements, leaving the 96 exotic gates of each disjoint), and
           S <X, Z, F, Phi> S^dag = <X, Z, F, Phi*>.

  Claim 5  The Paulis alone would not have pinned 2O, which is why the anchor
           is stated with F. Exactly four conjugate copies contain
           P = <X, Z>: the canonical one and the octahedron turned 45 degrees
           about each Pauli axis. Every impostor holds all eight Paulis (its
           turn axis stays four-fold, the other two become edge axes), meets
           the canonical copy in 16 elements and the canonical 24-cell in
           exactly the eight Paulis, contains no F, and is not the Clifford
           group. 2T is never in danger, under either anchor: P is its only
           Pauli-group subgroup, and 2T is normal in 2O.

  Claim 6  Under the anchor as printed, the three counts hold: one copy of 2T
           (containment forces equality, on order), one of 2O (of Claim 5's
           four, only the canonical one contains F), two of 2I (Claim 4).

Claim 5 is exhaustive by a finite argument, not by a search of SU(2). If
u 2O u^dag contains P, then u^dag P u is a subgroup of 2O isomorphic to P, and
there are exactly four of those (enumerated below). Fix for each such K a
conjugator v with v^dag P v = K -- the four below do exactly that -- and then
u v^dag normalizes P. Since N_SU(2)(P) = 2O (the normalizer is finite, P acting
irreducibly leaves it centralizer {+-1}, and the classification of the finite
subgroups of SU(2) puts nothing finite properly above 2O), u lies in 2O v, so
every anchored copy is a 2O-conjugate of one of the four exhibited. Their orbit
is closed below by breadth-first search over 2O's generators.

K&T state their Phi* verbally rather than as a matrix ("we obtain Phi* from Phi
by making the replacement sqrt5 -> -sqrt5 and then taking the complex
conjugate"), so what is encoded below is their recipe, not a transcribed
matrix. Their Phi agrees with ours.

Everything is exact: the gates come from main.py's SU(2) registry, and element
identity is decided by `key`, the canonical form set up in the next section. No
floats, so set equality is an identity test and not a rounding artifact.
Writes nothing.

Because the gates are imported rather than restated, this doubles as a
regression check on the Phi conventions in main.py: a drift in the golden gate,
in the symmetrized phases, or in the gate registry surfaces here as a failure.

    cd code && uv run phi_star_copies.py

Exits 0 with a verification report if every claim checks out; raises
(non-zero exit) on the first failure.
"""

from collections import deque

from sympy import Matrix, Rational, conjugate, expand, eye

from main import (
    _SU2_GATES,
    _to_basis,
    SIGMA,
    SQRT2,
    SQRT5,
    SQRT10,
    symI,
    TAU,
)

# =============================================================================
# Exact identity in Q(sqrt2, sqrt5, i)
# =============================================================================
# Every gate entry here lies in Q(sqrt2, sqrt5, i). Splitting an entry into its
# real and imaginary parts puts each in Q(sqrt2, sqrt5), where `_to_basis` gives
# a true canonical form: the rational coordinates in the basis {1, sqrt2, sqrt5,
# sqrt10}. An entry is therefore pinned by a pair of 4-tuples of Rationals, and
# a matrix by a 4-tuple of those pairs -- hashable, and exact.


def _split(e):
    """Complex entry -> (real coords, imag coords), each a 4-tuple of Rationals."""
    re, im = expand(e).as_real_imag()
    return _to_basis(re), _to_basis(im)


def _rebuild(coords):
    """Inverse of `_to_basis`: rational coordinates -> element of Q(sqrt2, sqrt5)."""
    a, b, c, d = coords
    return a + b * SQRT2 + c * SQRT5 + d * SQRT10


def key(M):
    """Canonical, hashable identity of an exact 2x2 matrix."""
    return tuple(_split(e) for e in M)


def canon(M):
    """Rewrite M's entries in canonical basis form.

    Mathematically the identity map; it keeps expression trees from swelling
    over the course of a closure, where products otherwise nest.
    """
    def c(e):
        re, im = _split(e)
        return _rebuild(re) + symI * _rebuild(im)
    return M.applyfunc(c)


def star(M):
    """Entry-wise image under sqrt5 -> -sqrt5, the substitution of Eq. (eq:golden-gate).

    This is the Galois automorphism of Q(sqrt2, sqrt5) fixing sqrt2, applied to
    the real and imaginary parts separately (so it fixes i). It negates exactly
    the sqrt5 and sqrt10 coordinates, and fixes X, Z and F.
    """
    def s(e):
        re, im = _split(e)
        flip = lambda c: c[0] + c[1] * SQRT2 - c[2] * SQRT5 - c[3] * SQRT10
        return flip(re) + symI * flip(im)
    return M.applyfunc(s)


def close(gens, limit=1000):
    """Exact multiplicative closure of a generating set. Returns {key: matrix}.

    `limit` caps the order and is a tripwire, not a tuning knob: every group
    here has order <= 120, but a mis-edited generator need not generate a finite
    subgroup of SU(2) at all, and the cap turns that from a hang into a failure.
    """
    gens = [canon(g) for g in gens]
    elems = {key(eye(2)): eye(2)}
    frontier = deque([eye(2)])
    while frontier:
        A = frontier.popleft()
        for g in gens:
            B = canon(A * g)
            k = key(B)
            if k not in elems:
                if len(elems) >= limit:
                    raise AssertionError(
                        f"closure exceeded {limit} elements -- the generators do"
                        f" not generate a finite group of the expected order"
                    )
                elems[k] = B
                frontier.append(B)
    return elems


# =============================================================================
# The anchor
# =============================================================================

def _conj(c, G):
    """c G c^dag, as {key: matrix}, for G a group given the same way."""
    out = {}
    for M in G.values():
        N = canon(c * M * c.H)
        out[key(N)] = N
    return out


def _pauli_subgroups(G):
    """The subgroups of G isomorphic to the symmetrized Pauli group.

    Every one of them is {+-1, +-a, +-b, +-ab} for a pair of anticommuting
    elements of order four, so the pairs enumerate the subgroups outright and
    no closure has to run.
    """
    minus_one = key(-eye(2))
    order4 = [M for M in G.values() if key(canon(M * M)) == minus_one]
    subs = set()
    for i, a in enumerate(order4):
        for b in order4[i + 1:]:
            if key(canon(a * b)) == key(canon(-(b * a))):
                ab = canon(a * b)
                subs.add(frozenset(key(M) for M in
                                   (eye(2), -eye(2), a, -a, b, -b, ab, -ab)))
    return subs


# =============================================================================
# Verification
# =============================================================================

def main():
    X, Z, F, H, S = (_SU2_GATES[g] for g in ("X", "Z", "F", "H", "S"))
    Phi = _SU2_GATES["Φ"]

    # --- Claim 1: our Phi* ---------------------------------------------------
    # The reference values are restated from the thesis rather than reused from
    # main.py: pinning the imported gate against the printed equation is the
    # regression check, so the duplication is the point.
    phi_tex = Rational(1, 2) * Matrix([
        [TAU + symI * SIGMA, 1],
        [-1, TAU - symI * SIGMA],
    ])
    assert key(Phi) == key(phi_tex), (
        "main.py's Phi disagrees with the closed form printed in"
        " Eq. (eq:golden-gate) -- the golden gate has drifted"
    )

    phi_star = star(Phi)

    for name, G in (("X", X), ("Z", Z), ("F", F)):
        assert key(star(G)) == key(G), f"sqrt5 -> -sqrt5 should fix {name}"

    # As printed in Eq. (eq:golden-gate), in terms of the inverse golden ratio:
    # sigma^-1 = tau, and the substitution sends tau -> -sigma, sigma -> -tau.
    phi_star_tex = Rational(1, 2) * Matrix([
        [-SIGMA - symI * TAU, 1],
        [-1, -SIGMA + symI * TAU],
    ])
    assert key(phi_star) == key(phi_star_tex), (
        "star(Phi) disagrees with the closed form printed in Eq. (eq:golden-gate)"
    )
    print("[ok] main.py's Phi matches Eq. (eq:golden-gate); Phi* = star(Phi)"
          " matches its printed closed form; star fixes X, Z, F")

    # --- Claim 2: K&T's Phi* -------------------------------------------------
    phi_star_kt = phi_star.applyfunc(conjugate)      # their recipe, verbatim
    assert key(phi_star_kt) != key(phi_star), (
        "K&T's Phi* should differ from ours as a matrix"
    )
    print("[ok] K&T's Phi* = conj(star(Phi)) differs from ours as a matrix")

    for zname, Zm in (("symmetrized", Z), ("standard", symI * Z)):
        rhs = canon(Zm * phi_star.H * Zm.H)
        assert key(rhs) == key(phi_star_kt), (
            f"K&T's Phi* != Z Phi*^dag Z^dag with the {zname} Z"
        )
    print("[ok] K&T's Phi* = Z Phi*^dag Z^dag (symmetrized and standard Z alike)")

    # --- Claim 3: same target ------------------------------------------------
    G_ours = set(close([X, Z, F, phi_star]))
    G_kt = set(close([X, Z, F, phi_star_kt]))

    assert len(G_ours) == 120, f"<X,Z,F,Phi*_ours> has order {len(G_ours)}, not 120"
    assert len(G_kt) == 120, f"<X,Z,F,Phi*_KT> has order {len(G_kt)}, not 120"
    assert G_ours == G_kt, (
        f"the two conventions generate different groups"
        f" (symmetric difference {len(G_ours ^ G_kt)})"
    )
    print(f"[ok] <X,Z,F,Phi*_ours> = <X,Z,F,Phi*_KT>: the same {len(G_ours)} elements,"
          f" symmetric difference {len(G_ours ^ G_kt)}")

    # --- Claim 4: it is the *other* anchored copy ----------------------------
    phi_copy = close([X, Z, F, Phi])                 # our committed atlas copy
    G_phi = set(phi_copy)
    tet_copy = close([X, Z, F])                      # the anchored 2T
    G_2T = set(tet_copy)

    assert len(G_phi) == 120, f"<X,Z,F,Phi> has order {len(G_phi)}, not 120"
    assert len(G_2T) == 24, f"<X,Z,F> has order {len(G_2T)}, not 24"
    assert G_ours != G_phi, "the Phi*-copy should not be the Phi-copy"
    assert key(phi_star) not in G_phi, "Phi* should lie outside the Phi-copy"
    assert G_phi & G_ours == G_2T, (
        f"the two copies should meet in exactly 2T, not in"
        f" {len(G_phi & G_ours)} elements"
    )
    print(f"[ok] the Phi*-copy is the other anchored copy: distinct from the"
          f" Phi-copy, meeting it in exactly 2T ({len(G_2T)} elements, leaving"
          f" {len(G_phi - G_2T)} exotic gates in each)")

    conj_by_S = {key(canon(S * M * S.H)) for M in phi_copy.values()}
    assert conj_by_S == G_ours, "S <X,Z,F,Phi> S^dag should be <X,Z,F,Phi*>"
    print("[ok] S <X,Z,F,Phi> S^dag = <X,Z,F,Phi*>: Clifford conjugation swaps"
          " the copies")

    # --- Claim 5: the Paulis alone would not pin 2O --------------------------
    G_2O = close([X, Z, F, H, S])                    # the anchored 2O = C_1
    P = close([X, Z])                                # the symmetrized Paulis
    assert len(G_2O) == 48, f"<X,Z,F,H,S> has order {len(G_2O)}, not 48"
    assert len(P) == 8, f"<X,Z> has order {len(P)}, not 8"

    # The 45-degree turn about z, and its images about x and y. Conjugation is
    # blind to global phase, so the standard T serves and keeps every entry in
    # Q(sqrt2, sqrt5, i); the symmetrized T would need cos(pi/8), which is not
    # in the field and would take `key` outside its canonical form.
    turn_z = Matrix([[1, 0], [0, (1 + symI) / SQRT2]])
    turn_x = canon(H * turn_z * H.H)
    turn_y = canon(S * turn_x * S.H)
    conjugators = {"canonical": eye(2), "x-turn": turn_x,
                   "y-turn": turn_y, "z-turn": turn_z}

    # The bridge to exhaustiveness (see the lemma in the module docstring): the
    # four conjugators realize *every* way a Pauli group sits inside 2O, so no
    # anchored copy escapes the orbit closed below.
    pauli_subs = _pauli_subgroups(G_2O)
    assert len(pauli_subs) == 4, (
        f"2O should hold four Pauli-group subgroups, not {len(pauli_subs)}"
    )
    assert {frozenset(_conj(v.H, P)) for v in conjugators.values()} == pauli_subs, (
        "the four conjugators should realize all four Pauli-group subgroups of 2O"
    )

    copies = {name: _conj(v, G_2O) for name, v in conjugators.items()}
    P_keys, canonical = set(P), set(copies["canonical"])
    assert canonical == set(G_2O), "conjugating by the identity should change nothing"
    for name, copy in copies.items():
        keys = set(copy)
        assert P_keys <= keys, f"the {name} copy should contain the Paulis"
        if name == "canonical":
            continue
        assert keys != canonical, f"the {name} copy should not be the canonical one"
        assert len(keys & canonical) == 16, (
            f"the {name} copy should meet the canonical one in 16 elements,"
            f" not {len(keys & canonical)}"
        )
        assert keys & G_2T == P_keys, (
            f"the {name} copy should meet the canonical 24-cell in exactly the"
            f" eight Paulis, not in {len(keys & G_2T)} elements"
        )
        assert key(F) not in keys, f"the {name} copy should not contain F"
    print(f"[ok] four copies of 2O contain <X, Z>: the canonical one and the"
          f" 45-degree turns about x, y, z, each meeting the canonical copy in"
          f" 16 elements and its 24-cell in exactly the {len(P_keys)} Paulis")

    orbit = {frozenset(copy): copy for copy in copies.values()}
    frontier = deque(orbit.values())
    while frontier:                                  # conjugation by 2O permutes
        G_ = frontier.popleft()                      # the anchored copies
        for c in (X, Z, F, H, S):
            nxt = _conj(c, G_)
            if frozenset(nxt) not in orbit:
                orbit[frozenset(nxt)] = nxt
                frontier.append(nxt)
    assert len(orbit) == 4, f"the anchored copies of 2O number {len(orbit)}, not four"
    assert all(P_keys <= keys for keys in orbit), "every copy in the orbit is anchored"
    print("[ok] and those four are all of them: the 2O-orbit of the four closes"
          " on itself, so the lemma leaves no other copy containing <X, Z>")

    assert _pauli_subgroups(tet_copy) == {frozenset(P)}, (
        "2T should hold exactly one Pauli-group subgroup"
    )
    for c in (X, Z, F, H, S):
        assert set(_conj(c, tet_copy)) == G_2T, "2T should be normal in 2O"
    print("[ok] 2T is pinned under either anchor: <X, Z> is its only Pauli-group"
          " subgroup, and 2T is normal in 2O = N(<X, Z>)")

    # --- Claim 6: the counts, under the anchor as printed --------------------
    # A copy of 2T containing <X, Z, F> is <X, Z, F>, on order alone. A copy of
    # 2O containing it contains the Paulis too, so it is one of the four above,
    # of which only the canonical one holds F. The two copies of 2I are Claim
    # 4's, each generated by <X, Z, F> and a golden gate.
    assert len(G_2T) == 24, f"<X,Z,F> has order {len(G_2T)}, not 24"
    with_F = [name for name, copy in copies.items() if key(F) in set(copy)]
    assert with_F == ["canonical"], f"exactly one copy should contain F, got {with_F}"
    assert G_2T <= G_phi and G_2T <= G_ours, "both copies of 2I should contain 2T"
    print("[ok] under the anchor as printed (contain <X, Z, F>): one copy of 2T,"
          " one of 2O -- the single-qubit Clifford group -- and two of 2I")

    print("\nAll anchor and Phi-star claims verified.")


if __name__ == "__main__":
    main()
