"""Appendix D.1, checked: two escapes exhibited exactly, and three claims
nothing else in the repo checks at all.

Sections 1-7 are the exhibit described below; sections 0, 8 and 9 are the
file's other half. See the section list.

Nothing here contradicts the *finding* of Appendix D.1 --- only the octahedron
is exactly implementable, and the four inexact solids stay barred by every
argument below. What it exhibits is the two coin-free routes to the
octahedron's non-dyadic 1/3 that Theorem 4's hypotheses have to exclude, so
that both can be re-verified in one run rather than re-derived from the prose:
escape B is exactly why the round bound cannot be dropped, and escape A exactly
why the discard ban is not a technicality. ``main()``'s closing block names the
hypothesis that stops each.

Writes nothing. Import it or run it; either way the tree is untouched.

    cd code && uv run weight_obstruction_escapes.py

The two escapes, and the clauses that stop them
-----------------------------------------------
Theorem 4 (``thm:weight``) reads:

    Call a protocol *deterministic* if it tosses no classical coin, discards
    no branch, and halts within a bounded number of rounds, every run
    returning an outcome; feedforward is still allowed. A deterministic
    protocol realizes only effects whose *weight* tr E_k lies in calR. [...]
    so a transitive covariant qubit POVM on V outcomes admits a deterministic
    implementation only if V is a power of two.

There are two coin-free routes to the octahedron's non-dyadic 1/3, and one
clause of that definition stops each.

*Escape A --- the discard.* The exhibition's circuit with the fourth branch
thrown away and the surviving six renormalized: no coin, but a division by the
acceptance probability, and (1/4) / (3/4) = 1/3. It is what *discards no
branch* is there for.

*Escape B --- the retry.* The same circuit with the fourth branch **looping**
rather than aborting: no coin, no discarded branch, an outcome on every run, and
six effects summing to I exactly --- so it also satisfies the justification the
coin paragraph gives for the discard ban (*"a POVM answers every shot, so its
effects sum to I with no acceptance probability to divide by"*), and V = 6 is
still not a power of two. What leaks is the branch **count**, not Lemma 2, and
only *halts within a bounded number of rounds* keeps it out.

Appendix D.2 bars three of the five reorientation prefixes by a field argument
that has to cover T as well as the Cliffords and Phi, since Appendix D.1's own
opener lists the gate sets as *"2T, Clifford, Clifford+T, 2I, Clifford+Phi, and
their unions"*. Sections 6 and 7 below split the five: T's Bloch rotation IS
the tetrahedron's and the cube's reorientation, so those two are exact over
Clifford+T, and what bars the other three is Lemma 6 rather than an enumeration
of Q(sqrt5) rotations.

What the sections check
-----------------------
0.  the field itself: K_R IS Frac(calR) n R, so the tests below test the
    right thing
1.  the baseline -- Theorem 1's coin construction, and where the 1/3 enters
2.  escape A -- 1/3 as a ratio of dyadic branch weights
3.  escape B -- 1/3 as the limit of dyadic partial sums
4.  discard, retry and coin buy that 1/3 by one and the same division
5.  what survives unbounded branching: the direction obstruction is count-blind
6.  T+ IS the tetrahedral and cubic reorientation, so two of the five are exact
7.  the other three demand sqrt3 or sqrt(tau+2), and D.1's Lemma 6 bars both
8.  Theorem 1's POSITIVE half: F is a 2T element at one phase, and every gate
    set contains 2T, so the octahedron's exhibition runs everywhere
9.  Corollary 7's premise: the coordinate shapes of all 44 inexact vertices

Sections 0, 8 and 9 are each the repo's only check of what they check: nothing
else identifies K_R with Frac(calR) n R, nothing else asserts F's membership in
2T, and Table E.1's generator formats the coordinate shapes rather than
deciding them.

Instruments
-----------
Predicates (``dyadic_order``, ``in_field``, ``bloch_matrix``) are imported from
``randomized_core`` rather than reimplemented, so every verdict here
is measured by the same instruments as Table D.1 and Table D.2. The identity
tests come from ``main`` for the same reason: every matrix below has entries in
Q(sqrt2, sqrt5, i) --- H and F contribute 1/sqrt2, S and T contribute i, the
solids contribute sqrt5 --- and ``_is_zero`` splits an entry into its real and
imaginary parts to reduce each with ``_to_basis``, whose canonical 4-tuple over
the real field Q(sqrt2, sqrt5) is what decides these zeros, rather than a search
for a simplification that exhibits them.

``in_field`` answers the other question, MEMBERSHIP regardless of presentation,
and neither instrument replaces the other: only ``_to_basis`` yields a hashable
key, which is what Section 5's direction count rests on.

Two objects fall outside ``_is_zero``'s reach and say so where they are used:
the reorientation matrices of Section 7, which carry sqrt3 and sqrt6, and the
solid coordinates of Section 9, which carry sqrt3 and sqrt(2+tau). Neither
carries a verdict --- both are equalities between two spellings of one number
--- and both go through ``sp.simplify`` instead, while every verdict in those
sections still goes through ``in_field``.

The DATA is imported for the same reason wherever a certified copy exists:
``REORIENT`` for the five reorientations (certified against Decker's own
printed columns in ``randomized_decker.py`` and against Table D.3 parsed out
of the thesis in ``_build_povm_cards.py``), ``symbolic_solids`` for the
vertices, ``atlas_gates`` for Phi, and ``main``'s quaternions and
``geometric_group`` for 2T, 2O and 2I. Section 7 keeps a second, angle-form
spelling of the reorientations and asserts it agrees with ``REORIENT``, so the
readable form stays in view without becoming a second source of truth.
"""

import sympy as sp
from sympy import Matrix, Rational, eye, sqrt, zeros

from main import (Quaternion, _is_zero,  # canonical form over Q(sqrt2, sqrt5)
                  _to_basis, geometric_group, qF, qH, qS, qX, qZ)
from randomized_core import (PAULI_SYM, REORIENT, SIG_SYM, TAU_SYM,
                             atlas_gates, bloch_matrix, dyadic_order, in_field,
                             symbolic_solids)


def _is_zero_matrix(M):
    """Exact zero test for a matrix over Q(sqrt2, sqrt5, i), entry by entry."""
    return all(_is_zero(e) for e in sp.expand(M))

sI = sp.I
KR = sqrt(2) + sqrt(5)                  # primitive element of K_R = Q(sqrt2, sqrt5)
K = KR + sI                             # ... and of K = Frac calR = K_R(i)

# calR = Z[tau, i, sqrt2, 1/2], as D.1 defines it: the four generators are
# what Section 0 pushes through to K, and what pins the field the whole file
# then tests against.
RING_GENERATORS = ((TAU_SYM, "tau"), (sI, "i"), (sqrt(2), "sqrt2"),
                   (Rational(1, 2), "1/2"))

H = (1 / sqrt(2)) * Matrix([[1, 1], [1, -1]])
S = Matrix([[1, 0], [0, sI]])
T = Matrix([[1, 0], [0, (1 + sI) / sqrt(2)]])
F = H * S.conjugate().T                 # facet gate F = H S+, Clifford
SX, SY, SZ = PAULI_SYM

# The three coin words of the octahedron's exhibition, and the Z readout.
WORDS = {0: eye(2), 1: F, 2: F.conjugate().T}
READOUT = {+1: Matrix([[1, 0]]), -1: Matrix([[0, 1]])}   # <0| and <1|

OCTA = [Rational(1, 6) * (eye(2) + s * P)
        for P in (SX, SY, SZ) for s in (+1, -1)]

# Section 9's alphabet: the reduced magnitude a vertex coordinate may show once
# its solid's radial scalar is divided out, which is the alphabet Table E.1's
# caption promises. (scalar, its name, the letters allowed, the letters that
# actually occur) per solid -- the last entry pins the sweep, since an alphabet
# stated loosely enough is no constraint.
_ONE_SIG_TAU = ((sp.Integer(1), "1"), (SIG_SYM, "sigma"), (TAU_SYM, "tau"))
_ONE_TAU = ((sp.Integer(1), "1"), (TAU_SYM, "tau"))

SHAPES = {
    "tetrahedron":  (1 / sqrt(3), "1/sqrt3", _ONE_SIG_TAU, {"1"}),
    "cube":         (1 / sqrt(3), "1/sqrt3", _ONE_SIG_TAU, {"1"}),
    "dodecahedron": (1 / sqrt(3), "1/sqrt3", _ONE_SIG_TAU, {"1", "sigma", "tau"}),
    "icosahedron":  (1 / sqrt(2 + TAU_SYM), "1/sqrt(2+tau)", _ONE_TAU, {"1", "tau"}),
}


def _matches_octahedron(effects):
    """Effect-for-effect multiset equality against the atlas octahedral POVM."""
    pool = list(OCTA)
    for E in effects:
        hit = next((j for j, N in enumerate(pool)
                    if _is_zero_matrix(E - N)), None)
        if hit is None:
            return False
        pool.pop(hit)
    return not pool


def _branch_rows(n_failures, j, s):
    """The row vector <a_b| of one branch of the two-ancilla protocol.

    Both ancillas start in |0>, take an F (F|0> = H|0>, so F splits an ancilla
    just as well, and F is a 2T atlas word where H is not), and are measured:
    each round pins a specific pair of outcomes with amplitude (1/sqrt2)^2 =
    1/2. After
    ``n_failures`` rounds that missed, the surviving round applies W_j to the
    data qubit and reads Z, so the composite C^2 -> C map is

        (1/2)^{n+1} <s| W_j,

    every entry a sum of products of gate entries: a member of calR.
    """
    return Rational(1, 2)**(n_failures + 1) * READOUT[s] * WORDS[j]


# ---------------------------------------------------------------------------
# Section 0: the instrument itself -- K_R IS Frac(calR) n R
# ---------------------------------------------------------------------------

def check_field_of_the_ring():
    """calR = Z[tau, i, sqrt2, 1/2] has Frac calR n R = Q(sqrt2, sqrt5) = K_R.

    Every verdict below, every row of Table D.2 and every vertex of Corollary
    7's sweep is a membership test against Q(sqrt2, sqrt5). The step from THE
    GATE SETS OF THIS THESIS to that field is what makes those the right tests,
    and it is the one link the rest of this file would otherwise assume -- KR
    above would be a constant typed in rather than a field derived. Four claims:

        Frac calR <= K    each of the four ring generators lies in K
        K <= Frac calR    sqrt5 = 2 tau - 1 lies in calR, and sqrt2 and i
                          are generators of it outright
        [K:Q] = 8 = 2 [K_R:Q] with i outside K_R, so {1, i} is a K_R-basis of K
        K n R = K_R       z = a + b i with a, b in K_R is real iff b = 0

    The third carries the fourth, and it is exhibited rather than argued: the
    8x8 coordinate matrix of {1, sqrt2, sqrt5, sqrt10} x {1, i} under
    ``_to_basis`` is the identity, so those eight are a Q-basis of K splitting
    into a real half and an i-half. ``_to_basis`` is the map ``_is_zero``
    decides zeros with, so what is checked here is the decomposition the
    instrument already performs, not a second opinion about it.
    """
    x = sp.Symbol("x")
    assert sp.degree(sp.minimal_polynomial(KR, x)) == 4, "[K_R:Q]"
    assert sp.degree(sp.minimal_polynomial(K, x)) == 8, "[K:Q]"

    for gen, name in RING_GENERATORS:
        assert in_field(gen, K), name                    # Frac calR <= K
    assert _is_zero(sqrt(5) - (2 * TAU_SYM - 1))         # sqrt5 is IN the ring
    assert in_field(sqrt(2), K) and in_field(sqrt(5), K)  # so K_R <= K, and
    assert not in_field(sI, KR)                          # i is what K adds

    # {1, i} spans K over K_R. The eight products of K_R's power basis with
    # 1 and with i have coordinate matrix I_8, so they are Q-independent and
    # exhaust the degree: a z in K is real exactly when its four i-coordinates
    # vanish, i.e. exactly when z lies in the Q-span of {1, sqrt2, sqrt5,
    # sqrt10}, which is K_R. That is the fourth claim, and the whole of it.
    quad = [sp.Integer(1), sqrt(2), sqrt(5), sqrt(10)]
    coords = []
    for b in quad + [sI * q for q in quad]:
        re, im = sp.expand(b).as_real_imag()
        coords.append(list(_to_basis(re)) + list(_to_basis(im)))
    assert Matrix(coords) == eye(8), Matrix(coords)

    print("  calR = Z[tau, i, sqrt2, 1/2], and Frac calR = K = Q(sqrt2, sqrt5, i):")
    print("  the four generators lie in K, and sqrt5 = 2 tau - 1 lies in calR")
    print("  [K:Q] = 8, [K_R:Q] = 4, i not in K_R -- so [K:K_R] = 2, and the")
    print("  coordinate matrix of {1, sqrt2, sqrt5, sqrt10} x {1, i} is I_8")
    print("  z in K is real iff its i-half vanishes:  K n R = Q(sqrt2, sqrt5)")
    print("[ok] the field every membership test below runs against is the field")
    print("     D.1 names -- derived from the ring, not typed in beside it")


# ---------------------------------------------------------------------------
# Section 1: the baseline -- Theorem 1's construction, with the coin
# ---------------------------------------------------------------------------

def check_coin_baseline():
    """Draw W uniformly from {I, F, F+}, apply, measure Z. Six effects."""
    effects = [Rational(1, 3) * WORDS[j].conjugate().T
               * Rational(1, 2) * (eye(2) + s * SZ) * WORDS[j]
               for j in WORDS for s in (+1, -1)]
    assert _is_zero_matrix(sum(effects, zeros(2, 2)) - eye(2))
    assert _matches_octahedron(effects)
    assert all(_is_zero(sp.trace(E) - Rational(1, 3)) for E in effects)
    assert dyadic_order(Rational(1, 3)) is None
    print("  three-word coin: six effects, sum = I, each tr = 1/3")
    print("  the 1/3 enters as a COIN PROBABILITY -- 1/3 is not in Z[1/2]")
    print("[ok] Theorem 1's construction reproduced; the coin is doing the work")


# ---------------------------------------------------------------------------
# Section 2: escape A -- the discard, barred by 'discards no branch'
# ---------------------------------------------------------------------------

def check_escape_discard():
    """Coin-free, but throws the fourth branch away and renormalizes."""
    accepted = [r.conjugate().T * r
                for j in WORDS for s in (+1, -1)
                for r in [_branch_rows(0, j, s)]]
    for E in accepted:                                   # dyadic branch weights
        assert _is_zero(sp.trace(E) - Rational(1, 4))
        assert dyadic_order(sp.trace(E)) == 2
    total = sum(accepted, zeros(2, 2))
    assert _is_zero_matrix(total - Rational(3, 4) * eye(2))
    p_accept = sp.trace(total) / 2
    assert _is_zero(p_accept - Rational(3, 4))           # rho-independent
    renormalized = [E / p_accept for E in accepted]
    assert _matches_octahedron(renormalized)
    assert _is_zero(Rational(1, 4) / Rational(3, 4) - Rational(1, 3))
    print("  six accepted effects, each of dyadic weight 1/4, summing to (3/4)I")
    print("  acceptance is rho-independent, so renormalizing is legitimate:")
    print("  (1/4) / (3/4) = 1/3 exactly -- the octahedral POVM, no coin tossed")
    print("[ok] escape A: the discard manufactures 1/3 as a ratio of weights")
    print("     (this is the escape Theorem 4's definition bans)")


# ---------------------------------------------------------------------------
# Section 3: escape B -- the retry, which only the round bound excludes
# ---------------------------------------------------------------------------

def check_escape_retry(depth=8):
    """Same circuit, fourth branch LOOPS. No coin, no discard, sums to I."""
    p_retry = Rational(1, 4)

    # Every branch, to any depth, is a legitimate Lemma 2 branch: amplitudes
    # in calR, weight dyadic. The lemma is not what breaks. (calR-membership
    # is 'algebraic integer over a power of two', and the power grows with the
    # depth, so dyadic_order needs its ceiling raised past the default 8.)
    for n in range(depth):
        for j in WORDS:
            for s in (+1, -1):
                row = _branch_rows(n, j, s)
                assert all(dyadic_order(e, kmax=depth + 4) is not None
                           for e in row)
                w = (row * row.conjugate().T)[0, 0]
                assert _is_zero(w - Rational(1, 4)**(n + 1))
                assert dyadic_order(w) == 2 * (n + 1)     # dyadic, every one
    print(f"  every branch to depth {depth}: amplitudes in calR, weight (1/4)^(n+1),")
    print("  dyadic without exception -- Lemma 2 and Step 1 hold branch by branch")

    # Partial sums stay dyadic; only the LIMIT leaves Z[1/2]. That is the gap.
    partials = []
    for n_max in (0, 1, 2, 5, depth - 1):
        w = sum(Rational(1, 4)**(n + 1) for n in range(n_max + 1))
        assert dyadic_order(w) is not None
        partials.append((n_max, w))
    limit = sp.summation(Rational(1, 4)**(sp.Symbol("n", integer=True) + 1),
                         (sp.Symbol("n", integer=True), 0, sp.oo))
    assert _is_zero(limit - Rational(1, 3))
    assert dyadic_order(limit) is None
    print("  partial sums of the branch weights, and the limit:")
    for n_max, w in partials:
        print(f"    rounds <= {n_max + 1:2d}   {str(w):>16s}   dyadic: yes")
    print(f"    limit        {str(limit):>16s}   dyadic: NO")

    # The realized POVM, exactly.
    geom = 1 / (1 - p_retry)
    effects = [geom * (r.conjugate().T * r)
               for j in WORDS for s in (+1, -1)
               for r in [_branch_rows(0, j, s)]]
    total = sum(effects, zeros(2, 2))
    assert _is_zero_matrix(total - eye(2))               # sums to I on the nose
    assert _matches_octahedron(effects)
    assert all(_is_zero(sp.trace(E) - Rational(1, 3)) for E in effects)

    # Expected rounds: finite, so this is not a pathological protocol.
    rounds = 1 / (1 - p_retry)
    assert _is_zero(rounds - Rational(4, 3))
    print()
    print("  the six effects sum to I EXACTLY, each with tr = 1/3")
    print(f"  P(a run returns an outcome) = 1;  expected rounds = {rounds}")
    print("  gates used: F = H S+ throughout, since F|0> = H|0> splits an ancilla")
    print("  too -- F lies in 2T up to phase, and every thesis gate set contains 2T,")
    print("  so this runs over all of them")
    print("[ok] escape B: coin-free, discard-free, outcome on every run, V = 6")
    print("     -- it meets every clause of Theorem 4's DETERMINISTIC but one, and")
    print("     the coin paragraph's own justification for the discard ban.")
    print("     The round bound in that definition is what excludes it")


# ---------------------------------------------------------------------------
# Section 4: the two escapes are one arithmetic
# ---------------------------------------------------------------------------

def check_same_door():
    """Discard and retry buy 1/3 the same way: a division by the acceptance."""
    by_hand = Rational(1, 4) / (1 - Rational(1, 4))
    n = sp.Symbol("n", integer=True)
    by_series = sp.summation(Rational(1, 4)**(n + 1), (n, 0, sp.oo))
    assert _is_zero(by_hand - Rational(1, 3))
    assert _is_zero(by_series - Rational(1, 3))
    print("  discard: (1/4) / (3/4)                = 1/3   (division, by hand)")
    print("  retry:   sum_{n>=0} (1/4)^(n+1)       = 1/3   (division, by series)")
    print("  coin:    a classical bias of 1/3                  (division, bought)")
    print("[ok] one door, three times: a positive factor the RING will not supply,")
    print("     supplied classically. 'Field = ring + coin' is the whole of it")


# ---------------------------------------------------------------------------
# Section 5: what survives -- the DIRECTION obstruction is count-blind
# ---------------------------------------------------------------------------

def check_directions_survive(depth=8):
    """Unbounded branching moves no Bloch vertex out of K_R^3.

    Step 2's kernel argument runs term by term over a sum of nonnegative
    numbers, so it does not care how many terms there are: rank one forces
    every |a_b> parallel, the effect's direction is one of them, and it lies
    in calR^2. Concretely, every branch of the retry protocol points at a Pauli
    axis no matter how deep, and no partial sum introduces a new direction.
    """
    seen = set()
    for n in range(depth):
        for j in WORDS:
            for s in (+1, -1):
                row = _branch_rows(n, j, s)
                E = row.conjugate().T * row
                nvec = Matrix([sp.trace(P * E) / sp.trace(E) for P in PAULI_SYM])
                assert all(in_field(c, KR) for c in nvec)
                assert _is_zero(nvec.dot(nvec) - 1)            # rank one, unit
                # The COUNT is the claim, so the key has to be canonical: two
                # spellings of one direction must collide and two directions
                # must not. `_to_basis` gives exactly that and is hashable,
                # which `simplify`-based equality is not.
                seen.add(tuple(_to_basis(sp.expand(c)) for c in nvec))
    assert len(seen) == 6, seen
    print(f"  branches to depth {depth}: {len(seen)} distinct directions, all in K_R^3")
    print("  and they are the six Pauli directions -- deeper never means elsewhere")

    # The tetrahedron is the control: no branch count reaches it, because its
    # vertices are not K_R-rational to begin with (Corollary 3, count-blind).
    tet = symbolic_solids()["tetrahedron"]
    assert all(not all(in_field(c, KR) for c in v) for v in tet)
    print(f"  control: every one of the tetrahedron's {len(tet)} vertices has a")
    print("  coordinate outside K_R, so no protocol reaches it at any depth")
    print("[ok] Corollary 3, Corollary 7, Lemmas 5-6 and Theorem 1's FIRST sentence")
    print("     are untouched by unbounded repetition. Only Theorem 4 needs the")
    print("     branch count bounded -- and Theorem 1's SECOND sentence with it")


# ---------------------------------------------------------------------------
# Section 6: T+ realizes the tetrahedral and cubic reorientation exactly
# ---------------------------------------------------------------------------

def _Rz(theta):
    return Matrix([[sp.cos(theta), -sp.sin(theta), 0],
                   [sp.sin(theta), sp.cos(theta), 0],
                   [0, 0, 1]])


def _Ry(theta):
    return Matrix([[sp.cos(theta), 0, sp.sin(theta)],
                   [0, 1, 0],
                   [-sp.sin(theta), 0, sp.cos(theta)]])


def check_T_realizes_two_reorientations():
    """Table D.3's R_z(45 deg) is drawn T+ in the figures, and correctly so."""
    R = sp.simplify(_Rz(sp.pi / 4))                     # the reorientation itself
    # A prefix acts on the effects by conjugation, so the GATE realizing R has
    # Bloch rotation R^{-1}. Table D.3's caption says so; T+ is what is drawn.
    got = bloch_matrix(T.conjugate().T)
    assert _is_zero_matrix(got - R.T), got               # R^{-1} = R^T for SO(3)
    assert _is_zero_matrix(bloch_matrix(T) - R)
    print("  bloch_matrix(T+) = R_z(45 deg)^{-1}, exactly -- so the prefix drawn in")
    print("  circuit_dec_tet.tex and circuit_dec_cube.tex IS the reorientation")
    print("  T is a gate of Clifford+T, which Appendix D.1's opener lists among")
    print("  'every gate set of this thesis'")
    print("[ok] the tetrahedron's and the cube's reorientation is EXACT over")
    print("     Clifford+T -- which is why the inexactness claim is scoped to")
    print("     gate sets with no T in them: 'only two of the five reorientations")
    print("     are exact, and only once T is adjoined' in D.2, 'no gate set here")
    print("     without a T in it' at the ch.4 caption")


# ---------------------------------------------------------------------------
# Section 7: the other three stay barred, and Lemma 6 is what bars them
# ---------------------------------------------------------------------------

def check_other_three_barred():
    """A gate sequence's Bloch entries lie in K_R, and three reorientations do not."""
    tau = TAU_SYM
    # The angle spelling, which is how D.2 reads, beside the cos/sin copy in
    # randomized_core.REORIENT, which is the certified one -- Decker's printed
    # columns and Table D.3 both bear on it, and a corrected factor order there
    # would otherwise leave the matrices below stale and this section passing.
    # REORIENT is the source; the spelling is asserted against it, not trusted.
    angle_form = {
        "tetrahedron":  _Rz(sp.pi / 4),
        "cube":         _Rz(sp.pi / 4),
        "octahedron":   _Rz(sp.pi / 4) * _Ry(sp.acos(1 / sqrt(3))) * _Rz(sp.pi),
        "icosahedron":  _Ry(-sp.acos(1 / (tau * sqrt(3)))),
        "dodecahedron": _Ry(sp.acos(tau / sqrt(tau + 2))),
    }
    reorientations = {s: REORIENT[s][1] for s in angle_form}
    for solid, A in angle_form.items():                  # sqrt3, sqrt6 and
        D = A - reorientations[solid]                    # sqrt(tau+2) put these
        assert all(sp.simplify(e) == 0 for e in D), solid  # outside _is_zero's
                                                         # field -- and this is
                                                         # a drift guard between
                                                         # two spellings, not one
                                                         # of the verdicts below
    print("  the five REORIENT matrices, as Table D.3 prints them, agree entry")
    print("  for entry with the angle spelling D.2 reads in -- one source, two")
    print("  presentations, and the source is the copy Decker's circuits certify")
    # Every gate here has entries in calR, and R_ij = tr(sigma_i U sigma_j U+)/2
    # is quadratic in them, so every realizable Bloch matrix is over calR n R,
    # inside K_R. Spot-check on the generators, T and Phi included -- Phi is
    # the one atlas gate that is not a Clifford and the only one carrying tau,
    # so leaving it out would show a narrower check than the sentence claims.
    for name, U in (("X", SX), ("Z", SZ), ("H", H), ("S", S), ("T", T), ("F", F),
                    ("Phi", atlas_gates()["Phi"])):
        R = bloch_matrix(U)
        assert all(in_field(e, KR) for e in R), name
    print("  X, Z, H, S, T, F, Phi: every Bloch entry is in K_R (quadratic in calR)")
    print("  -- so an entry outside K_R bars a rotation over EVERY thesis gate set")
    print()
    print(f"  {'solid':14s} {'reorientation entries in K_R?':>30s}   barred by")
    print("  " + "-" * 68)
    barred = set()
    for name, R in reorientations.items():
        R = sp.simplify(R)
        ok = all(in_field(e, KR) for e in R)     # in_field decides membership
                                                 # regardless of presentation, so
                                                 # the verdict never rests on the
                                                 # simplify above succeeding
        if not ok:
            barred.add(name)
        why = {"octahedron": "Lemma 6: sqrt3 not in K_R",
               "icosahedron": "Lemma 6: sqrt3 not in K_R",
               "dodecahedron": "Lemma 6: sqrt(tau+2) not a square"}.get(
                   name, "--- (exact, as T+)")
        print(f"  {name:14s} {str(ok):>30s}   {why}")
    assert barred == {"octahedron", "icosahedron", "dodecahedron"}, barred

    # Lemma 6's own two numbers, re-derived here so the link is not just prose.
    # The descent: a square in K_R lying in Q(sqrt5) forces alpha or alpha/2 to
    # be a square THERE, whose norm N(a + b sqrt5) = a^2 - 5b^2 must then be a
    # rational square. Four tests, and the printed values are the lemma's own.
    assert not in_field(sqrt(3), KR)
    assert not in_field(sqrt(tau + 2), KR)
    assert _is_zero(tau + 2 - (5 + sqrt(5)) / 2)
    expected = [Rational(16, 125), Rational(4, 125), sp.Integer(5), Rational(5, 4)]
    got = []
    for alpha in (Rational(2, 25) * (5 - sqrt(5)), (5 + sqrt(5)) / 2):
        for cand in (alpha, alpha / 2):
            e = sp.expand(cand)
            a, b = (sp.Poly(e, sqrt(5)).all_coeffs()[::-1] if e.has(sqrt(5))
                    else (cand, 0))
            norm = sp.simplify(a**2 - 5 * b**2)
            assert not sp.sqrt(norm).is_rational, (cand, norm)
            got.append(norm)
    assert got == expected, got
    print()
    print("  Lemma 6's descent, the four norms: "
          + ", ".join(str(v) for v in got))
    print("  every one has squarefree part 5, so none is a rational square")
    print("[ok] three of the five reorientations demand sqrt3 or sqrt(tau+2), and")
    print("     Lemma 6 bars both from K_R -- so D.2's conclusion survives for the")
    print("     octahedron, icosahedron and dodecahedron over Clifford+T too,")
    print("     proved by D.1's own lemma rather than by the Q(sqrt5) enumeration")


# ---------------------------------------------------------------------------
# Section 8: Theorem 1's positive half -- F is in 2T, at exactly one phase
# ---------------------------------------------------------------------------

def _closure(gens):
    """The subgroup of SU(2) generated by `gens`, with a word for each element.

    Keyed by ``basis_tuple``, the canonical identity of Section 5's directions
    -- so a containment can be EXHIBITED, by replaying the word, rather than
    read off a float comparison.
    """
    ident = Quaternion(1, 0, 0, 0)
    reached = {ident.basis_tuple(): ()}
    frontier = [ident]
    while frontier:
        nxt = []
        for q in frontier:
            word = reached[q.basis_tuple()]
            for name, g in gens:
                p = q * g
                if p.basis_tuple() not in reached:
                    reached[p.basis_tuple()] = word + (name,)
                    nxt.append(p)
        frontier = nxt
    return reached


def check_F_in_2T():
    """Theorem 1's exhibition uses only F, and F is a 2T element up to phase.

    The octahedron is realizable over EVERY gate set of this thesis, and the
    construction of Section 1 is why: its three coin words are I, F and F+,
    so the whole exhibition needs one gate. That gate is available everywhere
    because e^{-i pi/4} F lies in 2T and every gate set contains 2T -- two
    facts the appendix leans on and nothing else in the repo asserts.

    THE PHASE IS NOT FREE, which is why all three candidates are tested. Since
    det F = i, of the two eighth-root phases exactly one lands F in SU(2) at
    all: e^{+i pi/4} F has determinant -1 and bare F = H S+ has determinant i,
    and a determinant that is not 1 is not a unit quaternion. A sign slip here
    would leave every other assertion in this file passing.
    """
    G2T = {q.basis_tuple(): q for q in geometric_group("2T")}
    U2T = [q.to_unitary() for q in G2T.values()]
    minus = sp.expand(sp.exp(-sI * sp.pi / 4) * F)       # the symmetrized F
    plus = sp.expand(sp.exp(sI * sp.pi / 4) * F)         # the other eighth root

    dets = {sp.Integer(1): "1", sp.Integer(-1): "-1", sI: "i"}
    print(f"  {'candidate':16s} {'det':>4s}   an element of 2T?")
    print("  " + "-" * 48)
    verdicts = []
    for label, U in (("e^{-i pi/4} F", minus), ("e^{+i pi/4} F", plus),
                     ("F = H S+", F)):
        det = dets.get(sp.simplify(U.det()))
        assert det is not None, (label, U.det())         # the reason for the sign
        hit = any(_is_zero_matrix(U - W) for W in U2T)
        verdicts.append((det, hit))
        print(f"  {label:16s} {det:>4s}   {'YES' if hit else 'no'}")
    assert verdicts == [("1", True), ("-1", False), ("i", False)], verdicts

    # Every gate set D.1's opener lists contains 2T, and two closures decide
    # it: <X, Z, F> IS 2T -- the atlas's own 2T gate set -- and <H, S> is 2O,
    # which contains it. So a gate set contains 2T as soon as it contains the
    # Cliffords or the atlas's 2T gates, and each element of 2T is a WORD in
    # H and S, replayed here to confirm it. What is left is Clifford+T,
    # Clifford+Phi and the unions, supersets of a checked set by definition:
    # the one step in this section that reads the opener's list rather than
    # computing over it.
    tgates = _closure([("X", qX), ("Z", qZ), ("F", qF),
                       ("F+", qF.conjugate())])
    cliff = _closure([("H", qH), ("H+", qH.conjugate()),
                      ("S", qS), ("S+", qS.conjugate())])
    assert set(tgates) == set(G2T), "<X, Z, F> is not 2T"
    assert set(cliff) == {q.basis_tuple() for q in geometric_group("2O")}
    assert set(G2T) <= set(cliff), "2T is not inside <H, S>"
    assert set(G2T) <= {q.basis_tuple() for q in geometric_group("2I")}

    letters = {"H": qH, "H+": qH.conjugate(), "S": qS, "S+": qS.conjugate()}
    longest = 0
    for key in G2T:
        word = cliff[key]
        q = Quaternion(1, 0, 0, 0)
        for name in word:
            q = q * letters[name]
        assert q.basis_tuple() == key, word              # the word, replayed
        longest = max(longest, len(word))

    print()
    print(f"  <X, Z, F> closes at {len(tgates)} elements and IS 2T, element for element")
    print(f"  <H, S> closes at {len(cliff)} = |2O| and contains all {len(G2T)}, every one a")
    print(f"  word in H and S of length at most {longest}, replayed here to confirm it")
    print("  2T sits inside 2I as well, so the opener's list is covered:")
    print("    2T, 2I                            group itself / subgroup (checked)")
    print("    Clifford                          <H, S> = 2O (checked)")
    print("    Clifford+T, Clifford+Phi, unions  supersets of a checked set")
    print("[ok] Theorem 1's exhibition needs only F; F is 2T's element at the")
    print("     e^{-i pi/4} phase and at no other; and 2T is in every gate set")


# ---------------------------------------------------------------------------
# Section 9: Corollary 7's premise -- the coordinate shapes, all 44 vertices
# ---------------------------------------------------------------------------

def check_coordinate_shapes():
    """What the four inexact solids' vertex coordinates actually look like.

    Corollary 7's sweep asserts the VERDICT -- no vertex of the four lies in
    K_R^3 -- and asserts it vertex by vertex. What D.1 prints beside it is the
    FORM: every nonzero coordinate of a tetrahedron, cube or dodecahedron
    vertex is p/sqrt3 with |p| in {1, sigma, tau}, and of an icosahedron vertex
    p/sqrt(2+tau) with |p| in {1, tau}. That reading is taken off Table E.1,
    whose generator FORMATS the reduced magnitudes rather than deciding them,
    so it is decided here: all 44 vertices, in the alphabet the table's caption
    promises, with the realized letters pinned per solid.

    The form re-proves the verdict, and more sharply than the sweep needs. Each
    p lies in K_R and neither radial scalar does, so a nonzero coordinate is a
    nonzero K_R-multiple of something outside K_R: EVERY one of the 108 is
    outside, where Corollary 7 asks only for one per vertex.
    """
    solids = symbolic_solids()
    print(f"  {'solid':14s} {'V':>2s} {'coords':>7s}  {'scalar':^14s}  reduced magnitudes")
    print("  " + "-" * 64)
    n_vertices = n_coords = 0
    for solid, (scalar, scalar_name, alphabet, expected) in SHAPES.items():
        assert not in_field(scalar, KR), scalar          # the scalar is the bar
        seen, here = set(), 0
        for v in solids[solid]:
            n_vertices += 1
            for c in v:
                if sp.simplify(c) == 0:
                    continue
                here += 1
                p = sp.simplify(c / scalar)
                hit = [name for value, name in alphabet
                       if _is_zero(sp.expand(p**2 - sp.expand(value**2)))]
                assert len(hit) == 1, (solid, c, p, hit)
                seen.add(hit[0])
        assert seen == expected, (solid, seen)
        n_coords += here
        print(f"  {solid:14s} {len(solids[solid]):>2d} {here:>7d}  "
              f"{scalar_name:^14s}  {', '.join(sorted(seen))}")
    for value, name in _ONE_SIG_TAU:
        assert in_field(value, KR), name                 # ... but p never is
    assert (n_vertices, n_coords) == (44, 108), (n_vertices, n_coords)

    print()
    print("  every p lies in K_R and neither scalar does, so all 108 nonzero")
    print("  coordinates are outside K_R -- Corollary 7 needs one per vertex")
    print("[ok] the shapes D.1's Corollary 7 reads off Table E.1 are decided, not formatted,")
    print("     and they carry the sweep's verdict a second time")


# ---------------------------------------------------------------------------

def main():
    banner = "=" * 74
    print(banner)
    print("Appendix D.1: two escapes, and the hypotheses that close them")
    print(banner)
    for title, fn in (
        ("0. the field -- K_R IS Frac(calR) n R", check_field_of_the_ring),
        ("1. baseline -- the coin (Theorem 1's construction)", check_coin_baseline),
        ("2. escape A -- the discard (banned by the definition)", check_escape_discard),
        ("3. escape B -- the retry (banned by the round bound)", check_escape_retry),
        ("4. the two escapes are one arithmetic", check_same_door),
        ("5. what survives -- directions are count-blind", check_directions_survive),
        ("6. T+ realizes two of the five reorientations", check_T_realizes_two_reorientations),
        ("7. the other three stay barred, by Lemma 6", check_other_three_barred),
        ("8. Theorem 1's positive half -- F is in 2T", check_F_in_2T),
        ("9. Corollary 7's premise -- the coordinate shapes", check_coordinate_shapes),
    ):
        print(f"\n{title}\n" + "-" * len(title))
        fn()
    print("\n" + banner)
    print("Three printed statements, and what this run says each has to keep out:")
    print("  Theorem 4      'discards no branch' is what excludes escape A, and")
    print("                 'halts within a bounded number of rounds' escape B")
    print("  Theorem 1 s.2  the octahedron's exactness costs 'a coin, a discarded")
    print("                 branch, or unbounded repetition' -- three doors, and")
    print("                 sections 1 to 3 walk through all three")
    print("  D.2 / ch.4     the inexactness claim is scoped to gate sets with no T")
    print("                 in them, T+ being exact for two of the five: 'only once")
    print("                 T is adjoined' in D.2, spelled out at the ch.4 caption")
    print("Everything else in Appendix D.1 stands, and stands against both escapes.")
    print("Sections 0, 8 and 9 are this repo's only check of three things D.1 says:")
    print("  section 0      the membership tests run against Frac(calR) n R, and")
    print("                 that field is Q(sqrt2, sqrt5) -- derived, not typed in")
    print("  section 8      F is a 2T element at the phase e^{-i pi/4} and at no")
    print("                 other, so the octahedron's exhibition runs everywhere")
    print("  section 9      the coordinate shapes Corollary 7's sweep reads off")
    print("                 Table E.1 are decided here, not formatted there")
    print(banner)


if __name__ == "__main__":
    main()
