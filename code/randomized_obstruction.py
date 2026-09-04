"""Section 3 of the randomized-implementation suite: the exactness
obstruction (finding 3) -- direction and weight, the alignment's
inexactness, the gate axes, the octahedron's exactness, and the
reorientation obstruction.

randomized_implementations.py is the suite's entry point -- its docstring
carries the R1/R2 protocol definitions, the kappa/eta pin, the findings
index, the module map and the run instructions; `cd code && uv run
randomized_implementations.py` runs the whole suite. The finding proved
here:

  Finding 3  Exactness obstruction, in two halves that between them convict
             all five. DIRECTION (check_obstruction): a rank-1 POVM realized
             by ANY protocol over a gate set with matrix entries in a
             conjugation-closed field K (any ancillas, adaptivity, classical
             randomness) must have all its Bloch vertices in K_R^3,
             K_R = K n R. Testing the rotation-invariant det[v_a v_b v_c]
             against K_R = Q(sqrt2, sqrt5) -- the real field of every gate
             set in this thesis -- only the octahedron passes; the other
             four demand sqrt(3) or sqrt(5+2 sqrt5), the field extension
             surfacing as Decker's nested radicals in R2 and as the
             alignment A in R1. Which triple is a choice, and it moves the
             number but never the coset (det_invariant); the lemma is total
             vertex by vertex, so the alignment is inexact for every vertex
             choice and not just v0 (check_no_exact_alignment); and each
             inexact solid sits on the rotation axes of a magic gate it
             inherits from (check_gate_axes).

             WEIGHT (check_weight_obstruction) prices the DILATION route
             instead, where the direction lemma's escape hatch is closed: an
             effect's trace is a finite sum of gate-entry products, so it
             lands in calR n Q = Z[1/2], while a transitive covariant POVM
             on V outcomes needs tr E_k = 2/V -- so V must be a power of
             two. No pose escapes either test, so no DETERMINISTIC dilation
             over any thesis gate set is exact for any of the five.
             Deterministic is
             three bans, not one -- no coin, no discarded branch, a bounded
             number of rounds -- each closing one classical way to buy the
             division the ring will not supply. The two escapes the ban list
             exists to close are exhibited in weight_obstruction_escapes.py.
"""

import itertools

import numpy as np
import sympy as sp
from sympy import I as sI
from sympy import Matrix, Rational, sqrt

from randomized_core import (SOLIDS, TAU_SYM, SIG_SYM, COVARIANCE, T_NOISE,
                             t_NOISE, load_vertices, load_rotations,
                             channel_R2, atlas_vertices, symbolic_solids,
                             atlas_gates, bloch_axis, bloch_matrix,
                             state_from_bloch, on_solid, in_field,
                             dyadic_order, det_witness, det_invariant, _Rz,
                             REORIENT)
from randomized_field import exact_reposed_twirl_R2

# ---------------------------------------------------------------------------
# Section 3: finding 3 -- the exactness obstruction
# ---------------------------------------------------------------------------

def check_obstruction():
    # ATLAS order, not symbolic_solids() order: the table this feeds prints the
    # triple's indices and its caption sends the reader to tab:povm-atlas to
    # look them up, so the two numberings' disagreement -- three solids, and
    # atlas_vertices' docstring has the account -- has to be resolved HERE, in
    # the reader's favour.  It reaches the printed determinant as a sign, and
    # only on the tetrahedron and the icosahedron.
    solids_sym = {s: atlas_vertices(s) for s in SOLIDS}
    x = sp.Symbol("x")
    KR = sqrt(2) + sqrt(5)           # primitive element of Q(sqrt2, sqrt5)
    expected = {
        "tetrahedron": ((1, 2, 3), 27 * x**2 - 16, sqrt(3), "sqrt(3)"),
        "octahedron": ((1, 3, 5), x - 1, None, "-- (none)"),
        "cube": ((1, 2, 3), 27 * x**2 - 16, sqrt(3), "sqrt(3)"),
        "icosahedron": ((1, 2, 5), 125 * x**4 - 100 * x**2 + 16,
                        sqrt(5 + 2 * sqrt(5)), "sqrt(5+2 sqrt5)"),
        "dodecahedron": ((1, 2, 3), 27 * x**2 - 16, sqrt(3), "sqrt(3)"),
    }
    # The icosahedron carries two five-fold surds that look unrelated: the
    # determinant's sqrt(10 - 2 sqrt5) and the demanded sqrt(5 + 2 sqrt5).  Both
    # are the vertex normalizer sqrt(2 + tau) scaled by a unit -- up by tau, down
    # by tau -- so all three name one extension of Q(sqrt5), and 2 + tau is
    # Lemma 6's second named number.  That is how the lemma convicts the Demands
    # column, whose surd it never mentions.  The determinant's own square is the
    # lemma's first named number.  tab:povm-exactness prints both identities in a
    # midrule sandwich, so both are asserted here before anything is written.
    assert sp.simplify(sqrt(5 + 2 * sqrt(5)) - TAU_SYM * sqrt(2 + TAU_SYM)) == 0
    assert sp.simplify(sqrt(10 - 2 * sqrt(5))
                       - (2 / TAU_SYM) * sqrt(2 + TAU_SYM)) == 0
    assert sp.simplify(det_invariant(solids_sym["icosahedron"])**2
                       - sp.Rational(2, 25) * (5 - sqrt(5))) == 0
    print(f"  {'solid':14s} {'(a,b,c)':>9s} {'det[v_a v_b v_c]':>16s}  {'min poly':26s} {'in Q(sqrt2,sqrt5)?':>18s}  demands")
    print("  " + "-" * 96)
    for solid in SOLIDS:
        trip, d = det_witness(solids_sym[solid])
        mp = sp.minimal_polynomial(d, x)
        exact = in_field(d, KR)
        trip_exp, poly_exp, ext, demand = expected[solid]
        assert trip == trip_exp, (solid, trip)
        assert sp.expand(mp - poly_exp) == 0, (solid, mp)
        assert exact == (solid == "octahedron"), solid
        if ext is not None:          # the demanded extension admits it
            assert in_field(d, sqrt(2) + sqrt(5) + ext), solid
        print(f"  {solid:14s} {str(trip):>9s} {float(d):>+16.6f}  {str(mp):26s} "
              f"{str(exact):>18s}  {demand}")
    # Which triple is a choice, and the choice moves the number -- so the table
    # printing ONE row per solid needs the choice not to move the VERDICT.
    # Lemma 5 says so via square classes; the sweep says so directly, and says
    # more: every spanning triple's determinant is a K_R-multiple of the one
    # printed, one coset per solid.  (The Lean formalization needs the same
    # shape: a lemma stated at the bare surd sqrt3 applies to no solid at all,
    # and only its coset form reaches them.)  Cheap at 5s: the
    # dodecahedron's 960 spanning triples realize ten determinants, five
    # magnitudes and FOUR minimal polynomials, and all ten are K_R x sqrt3.
    sweep = {                        # spanning triples, dets, |dets|, min polys
        "tetrahedron": (4, 2, 1, 1),
        "octahedron": (8, 2, 1, 1),
        "cube": (32, 2, 1, 1),
        "icosahedron": (160, 4, 2, 1),
        "dodecahedron": (960, 10, 5, 4),
    }
    print()
    print(f"  {'solid':14s} {'spanning':>9s} {'dets':>5s} {'|dets|':>7s} {'min polys':>10s}"
          f"  one K_R-coset?")
    print("  " + "-" * 65)
    for solid in SOLIDS:
        verts, ref = solids_sym[solid], det_invariant(solids_sym[solid])
        dets, n = set(), 0
        for trip in itertools.combinations(range(len(verts)), 3):
            d = sp.simplify(Matrix.hstack(*[verts[i] for i in trip]).det())
            if d != 0:
                dets.add(sp.radsimp(d))
                n += 1
        coset = all(in_field(sp.radsimp(d / ref), KR) for d in dets)
        mags = {sp.radsimp(sp.Abs(d)) for d in dets}
        polys = {sp.minimal_polynomial(m, x) for m in mags}
        assert coset, solid
        assert (n, len(dets), len(mags), len(polys)) == sweep[solid], (
            solid, n, len(dets), len(mags), len(polys))
        print(f"  {solid:14s} {n:>9d} {len(dets):>5d} {len(mags):>7d} {len(polys):>10d}"
              f"  {str(coset):>14s}")
    print("[ok] one coset per solid: the triple moves the determinant -- sign always,")
    print("     magnitude on I and D, and on D the MINIMAL POLYNOMIAL too (four of them)")
    print("     -- but never the coset, so any one triple decides.  That is Lemma 5's")
    print("     square class, and it is what lets tab:povm-exactness print one row each")
    print()
    print("[ok] only the octahedral POVM is exact over K_R = Q(sqrt2, sqrt5) -- the real")
    print("     field of every thesis gate set; the rest demand the listed extensions")
    print("     (Decker's nested radicals in R2; the alignment A in R1 -- same magic,")
    print("     two hiding places).  The icosahedron's two surds are one extension under")
    print("     three names -- sqrt(5+2 sqrt5) = tau sqrt(2+tau) and sqrt(10-2 sqrt5) =")
    print("     (2/tau) sqrt(2+tau), the vertex normalizer scaled up and down by a unit --")
    print("     so Lemma 6 bars them by its second named number, the determinant by its first")


def check_no_exact_alignment():
    # finding 3, sharpened to the alignment: if a rotation W with entries
    # in K_R took ANY vertex v to +-zhat, then v = +-W^T zhat -- a row of
    # W -- would lie in K_R^3. So a vertex with a coordinate outside K_R
    # admits no K_R-rational alignment, hence (by finding 3) no exact
    # circuit over any thesis gate set; checking every vertex closes the
    # loophole of aligning a cleverer vertex than our v0
    KR = sqrt(2) + sqrt(5)
    solids_sym = symbolic_solids()
    cache = {}

    def member(c):
        c = sp.simplify(c)
        if c not in cache:
            cache[c] = in_field(c, KR)
        return cache[c]

    for solid in SOLIDS:
        if solid == "octahedron":
            continue
        verts = solids_sym[solid]
        assert all(not all(member(c) for c in v) for v in verts), solid
        print(f"  {solid:14s} every one of its {len(verts)} vertices has a coordinate outside K_R")
    print("[ok] no vertex of an inexact solid lies in K_R^3, so no K_R-rational rotation")
    print("     -- hence no exact circuit over any thesis gate set -- aligns any vertex")
    print("     to zhat: R1's alignment is inexact for every vertex choice, not just v0")


def check_gate_axes():
    solids_sym = symbolic_solids()
    landing = {"X": {"octahedron"}, "Z": {"octahedron"},
               "F": {"tetrahedron", "cube", "dodecahedron"},
               "Phi": {"icosahedron"}}
    for name, U in atlas_gates().items():
        n = bloch_axis(U)
        hits = {s for s in SOLIDS
                if on_solid(n, solids_sym[s]) or on_solid(-n, solids_sym[s])}
        assert hits == landing[name], (name, hits)
        print(f"  {name:4s} axis = {tuple(sp.nsimplify(c) for c in n)}   lies on: {sorted(hits)}")
    print("[ok] X/Z -> octahedron, F -> tetrahedron+cube+dodecahedron, Phi -> icosahedron:")
    print("     each inexact solid sits on the eigen-axes of a magic gate and inherits its magic;")
    print("     Phi's rotation axis IS an icosahedron vertex, in the atlas orientation")


def check_octahedron_exact():
    verts = symbolic_solids()["octahedron"]
    E = [Rational(1, 3) * state_from_bloch(v) for v in verts]
    assert sp.simplify(sum(E, sp.zeros(2, 2)) - sp.eye(2)) == sp.zeros(2, 2)
    axes = {(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)}
    assert {tuple(int(c) for c in v) for v in verts} == axes
    print("[ok] the octahedron IS the three Pauli bases (sum E_k = I, vertices = +-axes):")
    print("     its PROJECTIVE route is literally randomized-Pauli measurement (A = Id),")
    print("     and what vanishes here is the obstruction, not the distinction --")
    print("     twirled-native keeps its dilation, ancillas and all, and still reads")
    print("     tr T/3 where the projective route reads T_zz")


def check_weight_obstruction():
    # The companion to check_obstruction, and the reason the octahedron's
    # exactness has to be exhibited with a COIN. That lemma constrains an
    # effect's direction (its Bloch vertex); this one constrains its weight.
    #
    # Fix a protocol whose only randomness is quantum -- unitaries from the
    # gate set, ancillas in |0>, computational-basis readout, feedforward
    # allowed, but no classical coin, no discarded branch, and a bounded
    # number of rounds. Each branch acts on the data qubit as the row
    # vector <y|U(. (x) |0...0>), whose entries are sums of products of
    # gate entries, hence lie in calR (see dyadic_order). An outcome class
    # realizes sum_i |a_i><a_i| over FINITELY many branches, rank 1 only if
    # all the |a_i> are parallel, and its trace sum_i ||a_i||^2 is then a
    # finite sum, lying in calR n R = the reals of calR. A transitive covariant
    # POVM on V outcomes has tr E_k = 2/V, a rational, and calR n Q = Z[1/2]
    # -- so V must be a power of two.
    #
    # Unlike the vertex test this one needs no rotation invariance: weights
    # are orientation-blind to begin with, so a cleverer pose cannot save a
    # solid, Decker's or ours. What CAN save one is a classical operation on
    # the weight, and there are exactly three -- the three bans above. A
    # coin buys the octahedron's 1/3 as a bias, which is what the coin of
    # check_coset_coin spends; a discard buys it as 1/4 over 3/4; an
    # unbounded retry buys it as the series sum_{n>=0}(1/4)^(n+1). All
    # three deliver a PROBABILITY, never an amplitude, and
    # weight_obstruction_escapes.py exhibits the latter two.
    H = (1 / sqrt(2)) * Matrix([[1, 1], [1, -1]])
    S = Matrix([[1, 0], [0, sI]])
    gates = dict(atlas_gates())                      # X, Z, F, Phi
    gates.update({
        "H": H, "S": S,
        "T": Matrix([[1, 0], [0, (1 + sI) / sqrt(2)]]),
        "Phi*": Rational(1, 2) * Matrix([[-SIG_SYM - sI * TAU_SYM, 1],
                                         [-1, -SIG_SYM + sI * TAU_SYM]]),
        "CNOT": Matrix(4, 4, lambda i, j: int((i, j) in
                                              ((0, 0), (1, 1), (2, 3), (3, 2)))),
    })
    for name, U in sorted(gates.items()):
        assert sp.simplify(U * U.conjugate().T - sp.eye(U.rows)) == sp.zeros(U.rows)
        orders = [dyadic_order(e) for e in U]
        assert all(k is not None for k in orders), name
        print(f"  {name:5s} entries in calR at 2^-{max(orders)}")
    # closure is a ring axiom, but spot-check it on every two-letter word
    for A in gates.values():
        for B in gates.values():
            if A.rows == B.rows:
                assert all(dyadic_order(e) is not None for e in A * B)
    print("  every two-letter word stays in calR (closure under x, as a ring must)")
    print()
    KR = sqrt(2) + sqrt(5)
    # either numbering will do here, unlike check_obstruction(): weights are
    # order-free, and the direction column is a membership BOOLEAN, which the
    # coset sweep there shows the triple cannot move (one coset per solid, so
    # either all its determinants are in K_R or none are).  Nothing is printed
    # with a vertex index, so nothing has to be the reader's.
    solids_sym = symbolic_solids()
    weight_ok, direction_ok = set(), set()
    print(f"  {'solid':14s} {'V':>3s}  {'tr E_k':>7s}  {'weight in Z[1/2]?':>18s}"
          f"  {'vertices in K_R^3?':>19s}")
    print("  " + "-" * 70)
    for solid in SOLIDS:
        verts = solids_sym[solid]
        V = len(verts)
        w = Rational(2, V)
        effects = [w * state_from_bloch(v) for v in verts]
        assert sp.simplify(sum(effects, sp.zeros(2, 2)) - sp.eye(2)) == sp.zeros(2, 2)
        assert all(sp.simplify(sp.trace(E) - w) == 0 for E in effects)
        if dyadic_order(w) is not None:
            weight_ok.add(solid)
        if in_field(det_invariant(verts), KR):
            direction_ok.add(solid)
        print(f"  {solid:14s} {V:3d}  {str(w):>7s}  {str(solid in weight_ok):>18s}"
              f"  {str(solid in direction_ok):>19s}")
    assert weight_ok == {"tetrahedron", "cube"}, weight_ok
    assert direction_ok == {"octahedron"}, direction_ok
    assert not (weight_ok & direction_ok)            # no solid clears both
    assert dyadic_order(Rational(1, 3)) is None
    print("[ok] two independent obstructions, and between them they convict all five:")
    print("     the tetrahedron and cube fail on direction (sqrt3), the icosahedron and")
    print("     dodecahedron on both, the octahedron on weight alone -- 2/V is in Z[1/2]")
    print("     only for V a power of two. So NO deterministic dilation over any thesis")
    print("     gate set realizes any Platonic solid POVM exactly, in any orientation --")
    print("     deterministic meaning no coin, no discarded branch, and a bounded number")
    print("     of rounds. The octahedron's 1/3 is bought classically, through one of the")
    print("     three: a coin's bias, a 1/4-over-3/4 discard, or an unbounded retry's")
    print("     series (weight_obstruction_escapes.py exhibits the last two)")


def check_reorientation_obstruction():
    # Third obstruction in the same key, and the one that prices Decker's
    # POSE. His circuits park the solid's cyclic symmetry axis on zhat --
    # the size of his Fourier block IS that axis's order -- while the atlas
    # parks the Pauli axes there. The two differ by a fixed
    # REORIENTATION, and the question is whether that correction comes free.
    #
    # It does not, and the reason is a third field, smaller than K_R. A
    # gate's action on the Bloch sphere is quadratic in its entries, so the
    # 1/sqrt2 of H and F pairs off and never reaches SO(3): every Clifford
    # acts as a signed permutation matrix, and Phi as a turn with entries in
    # {+-1/2, +-sig/2, +-tau/2}. Q(sqrt5) is a field, hence closed under
    # products, so EVERY rotation an ATLAS-GENERATED gate set can realize
    # has SO(3) entries in Q(sqrt5) -- and global phase is invisible down
    # here, so this settles the question up to phase, the physical one. The
    # quantifier is atlas-generated and NOT 'any thesis gate set': the
    # enumeration above covers the Cliffords and Phi, and T is not in it.
    #
    # The five reorientations demand sqrt2 (tetrahedron, cube), sqrt3
    # (octahedron, icosahedron) or tau/sqrt(tau+2) (dodecahedron), none of
    # them in Q(sqrt5). So NO reorientation is an ATLAS word, and none is
    # exact over an ATLAS-GENERATED gate set -- one whose every SINGLE-QUBIT
    # gate is an atlas word up to phase: 2T, Clifford, 2I, Clifford+Phi and
    # their unions, i.e. every thesis gate set with no T in it. (The scope
    # matters: CNOT is in every thesis gate set and is no atlas word.)
    #
    # Adjoining T moves the line, exactly twice. Bloch(T) = Rz(45 deg), so
    # the tetrahedron's and the cube's correction IS T-dagger, exact over
    # Clifford+T -- the gate the circuit figures draw. The assert closing
    # this block is that fact, stated as T's Bloch matrix leaving Q(sqrt5).
    # The other three stay barred over EVERY thesis gate set, unions
    # included, by K_R and Lemma 6: a Bloch matrix is quadratic in entries
    # of calR, so its entries lie in K_R, and neither sqrt3 (octahedron,
    # icosahedron) nor sqrt(tau+2) (dodecahedron) does.
    #
    # GUARD, and the trap this check exists to spring: K_R-exactness is
    # NECESSARY, not sufficient. T clears check_obstruction's field test
    # (entries in Q(sqrt2, i)) and check_weight_obstruction's ring test
    # (exp(i pi/4) is an algebraic integer outright), yet is no ATLAS word
    # -- its rotation has order 8 where the polyhedral groups stop at 5.
    #
    # What this obstruction costs: nothing beyond exposition. A reorientation
    # is fixed and g-independent, so it neither disturbs the twirl nor costs
    # an exactness -- no deterministic dilation was exact for any solid to
    # begin with, in any pose, Decker's or ours. Not by the weight test
    # alone: it clears the tetrahedron and cube (the weight_ok assert above
    # says exactly that), and it is check_obstruction's direction test,
    # pose-blind too, that convicts those two. Between them, all five.
    KR = sqrt(2) + sqrt(5)
    Q5 = sqrt(5)
    gates = dict(atlas_gates())                      # X, Z, F, Phi
    gates.update({"H": (1 / sqrt(2)) * Matrix([[1, 1], [1, -1]]),
                  "S": Matrix([[1, 0], [0, sI]])})
    T_GATE = Matrix([[1, 0], [0, (1 + sI) / sqrt(2)]])
    phi_entries = {Rational(1, 2), -Rational(1, 2),
                   TAU_SYM / 2, -TAU_SYM / 2, SIG_SYM / 2, -SIG_SYM / 2}
    bloch = {name: bloch_matrix(U) for name, U in sorted(gates.items())}
    for name, R in bloch.items():
        assert sp.simplify(R.T * R - sp.eye(3)) == sp.zeros(3)
        assert sp.simplify(R.det() - 1) == 0
        assert all(in_field(e, Q5) for e in R), name
        if name == "Phi":
            assert all(sp.simplify(e) in {sp.simplify(c) for c in phi_entries}
                       or e == 0 for e in R)
            print(f"  {name:4s} entries in "
                  "{+-1/2, +-sig/2, +-tau/2} -- irrational, but Q(sqrt5)")
        else:
            assert all(e.is_Integer for e in R), name
            print(f"  {name:4s} signed permutation matrix (Clifford: integral)")
    # U -> R is a homomorphism, so closure is the field's, not a coincidence
    assert sp.simplify(bloch_matrix(gates["X"] * gates["Phi"])
                       - bloch["X"] * bloch["Phi"]) == sp.zeros(3)
    for A in bloch.values():                         # closure, as a field must
        for B in bloch.values():
            assert all(in_field(e, Q5) for e in A * B)
    print("  every two-letter word stays in Q(sqrt5) (closure under x)")
    print()
    # The trap, made explicit on the gate the tetrahedron's correction is.
    assert all(in_field(sp.re(e), KR) and in_field(sp.im(e), KR) for e in T_GATE)
    assert all(dyadic_order(e) is not None for e in T_GATE)
    assert not all(in_field(e, Q5) for e in bloch_matrix(T_GATE))
    assert not in_field(sqrt(2) / 2, Q5)
    print("  T: entries in K_R (passes section 3's field test) and in calR (passes")
    print("     the weight test), yet its Bloch matrix needs 1/sqrt2 -- NOT in")
    print("     Q(sqrt5). So T is no atlas word -- field-exact is not atlas.")
    print()
    # Decker's poses, as his circuits produce them (REORIENT, above). Verified
    # here to be genuine Decker poses: zhat carries an axis of the order of his
    # Fourier block, and the pose is not the atlas's. check_decker_outcome_order
    # closes the loop the other way, deriving the poses from the circuits.
    reorient = REORIENT

    def vset(vs):
        return {tuple(sp.radsimp(sp.nsimplify(sp.simplify(c))) for c in v)
                for v in vs}

    def z_order(V):
        for m in (5, 4, 3, 2):
            R = _Rz(sp.cos(2 * sp.pi / m), sp.sin(2 * sp.pi / m))
            if vset([R * v for v in V]) == vset(V):
                return m
        return 1

    # What the pose does NOT cost the estimator it DOES cost the tails, the
    # fourth moment being read against the PAULI axes (Appendix F). The
    # mechanism is which vertices R2 effectively measures: R_g^T n_k, which
    # covariance makes the solid's own vertex set in the atlas pose and a
    # union of rotated copies in Decker's.
    def tail_weight(s, rots, r):
        """81 E[(effective vertex coordinate)^4] under R2, on state r."""
        tot = np.zeros(3)
        for Rg in rots:
            eff = s @ Rg                             # rows R_g^T n_k
            p = (1 + eff @ r) / len(s)               # p(k | g)
            tot += 81 * (p[:, None] * eff ** 4).sum(axis=0)
        return tot / len(rots)

    ATLAS_TAIL = {"tetrahedron": 9.0, "cube": 9.0, "octahedron": 27.0,
                  "icosahedron": 81 / 5, "dodecahedron": 81 / 5}
    DECKER_TAIL = {"tetrahedron": 15.0, "cube": 15.0, "octahedron": 12.0,
                   "icosahedron": 81 / 5, "dodecahedron": 81 / 5}
    PROBES = (np.zeros(3), np.array([0.31, -0.47, 0.62]),
              np.array([-0.5, 0.0, 0.25]))           # state-independence too
    tails, gap = {}, {}

    solids_sym = symbolic_solids()
    print(f"  {'solid':14s} {'F_m':>4s} {'Decker z':>9s} {'atlas z':>8s}"
          f"  {'demands':>16s}  {'in Q(sqrt5)?':>12s}  {'R2 kappa @ his pose':>19s}"
          f"  {'exact vs float':>14s}")
    print("  " + "-" * 108)
    for solid in SOLIDS:
        m, R, demand = reorient[solid]
        V = [Matrix(3, 1, list(v)) for v in solids_sym[solid]]
        D = [R.T * v for v in V]                     # Decker's pose
        assert sp.simplify(R.T * R - sp.eye(3)) == sp.zeros(3)
        assert sp.simplify(R.det() - 1) == 0
        assert vset([R * d for d in D]) == vset(V)   # R really reorients
        assert z_order(D) == m                       # his Fourier block's axis
        assert vset(D) != vset(V)                    # and it is not our pose
        assert not all(in_field(e, Q5) for e in R)
        # the whole coset falls with its representative: every h in the
        # covariance group is an atlas word, so h and h^T are Q(sqrt5)-rational
        # by the lemma above, and hR is Q(sqrt5)-rational iff R is
        for B in bloch.values():
            assert not all(in_field(e, Q5) for e in B * R)
        # ...and yet the pose costs the twirled-native estimator nothing.
        # R2's snapshot follows the vertex list the estimator BELIEVES --
        # here Decker's pose fed as belief and device alike -- and its
        # twirl needs the drawn group to be irreducible, not to be the
        # measured POVM's covariance group -- so feeding it Decker's pose
        # while still drawing g from the atlas-anchored group returns the
        # same channel; the coset scan below is what happens when the two
        # lists PART. The reorientation buys a NAME, not a measurement.
        s_atlas = load_vertices(solid)
        s_decker = s_atlas @ np.array(R, dtype=float)
        rots = load_rotations(COVARIANCE[solid])
        M, off = channel_R2(s_decker, rots, T_NOISE, t_NOISE)
        d_M = np.abs(M - (np.trace(T_NOISE) / 3) * np.eye(3)).max()  # meas. 2.6e-15
        assert d_M < 1e-10, f"{solid}: reposed R2 not depol (max |dev| = {d_M:.2e})"
        assert np.allclose(off, 0, atol=1e-9), solid
        # ...and the POSE is quantified, not probed. The two asserts above are
        # the C = R case of an identity in an arbitrary 3x3 C, so what they
        # test is a corollary of a theorem this file states. The float pair is
        # the numeric agreement check every exact companion is paired with:
        # the generic verdict is a boolean, so exact_channel_R2 over its
        # symbolic coefficient ring is required to reproduce channel_R2 value
        # for value at this pose and T_NOISE.
        generic, evaluate = exact_reposed_twirl_R2(solid)
        assert generic, solid
        Me, offe = evaluate(np.array(R, dtype=float), T_NOISE, t_NOISE)
        gap[solid] = max(np.abs(Me - M).max(), np.abs(offe - off).max())
        assert gap[solid] < 1e-12, (solid, gap[solid])
        # ...but it does cost the tails, on every probe and every axis alike
        for r in PROBES:
            ta, td = (tail_weight(s_atlas, rots, r),
                      tail_weight(s_decker, rots, r))
            # the tails run to 27; worst deviation over the five solids and
            # three probes is 2.1e-14
            d_ta = np.abs(ta - ATLAS_TAIL[solid]).max()
            d_td = np.abs(td - DECKER_TAIL[solid]).max()
            assert d_ta < 1e-10, f"{solid}: atlas tail moved (max |dev| = {d_ta:.2e})"
            assert d_td < 1e-10, f"{solid}: Decker tail moved (max |dev| = {d_td:.2e})"
        tails[solid] = (ATLAS_TAIL[solid], DECKER_TAIL[solid])
        print(f"  {solid:14s} {m:4d} {z_order(D):9d} {z_order(V):8d}"
              f"  {demand:>16s}  {'no':>12s}  {M[0, 0]:>19.9f}"
              f"  {gap[solid]:>14.1e}")
    assert z_order([Matrix(3, 1, list(v))
                    for v in solids_sym["octahedron"]]) == 4
    # Negative control on the generic verdict, so it cannot be passing
    # vacuously: the theorem's one hypothesis is that the DRAWN group is
    # irreducible, so a reducible draw has to fail it at every pose exactly as
    # it fails at one. check_exact_scalars uses this same C_3 coin, which
    # realizes the octahedral POVM and twirls nothing.
    assert not exact_reposed_twirl_R2(
        "octahedron", [sp.eye(3), bloch["F"], bloch["F"] * bloch["F"]])[0]
    print("[ok] every rotation realizable over an ATLAS-GENERATED gate set has")
    print("     SO(3) entries in Q(sqrt5); all five reorientations leave it, so")
    print("     none is an atlas word. Adjoining T makes exactly two exact --")
    print("     the tetrahedron's and the cube's, as T+, the gate the circuit")
    print("     figures draw -- while the other three stay barred over every")
    print("     thesis gate set by K_R and Lemma 6. Each inexact one is still")
    print("     approximable to any accuracy, Clifford+Phi being universal.")
    print("     Costs nothing new: a reorientation is fixed and")
    print("     g-independent, and no dilation was exact in any pose anyway")
    print(f"[ok] and it costs the ESTIMATOR nothing: R2 returns tr(T)/3 = "
          f"{np.trace(T_NOISE) / 3:.6f} in")
    print("     Decker's pose as in ours, all five solids, offset still zero.")
    print("     The pose is a question of naming, not of function -- what needs")
    print("     it is the claim that prepending U_g rotates OUR vertices by g")
    print("[ok] and that is a THEOREM, not two poses that happened to agree:")
    print("     M(sC) = tr(C^T C T)/3 Id_3 and off(sC) = 0 identically in an")
    print("     arbitrary 3x3 C and in (T, t), so no rotation is missed and no")
    print("     parametrisation of SO(3) has to be got right. The measured")
    print("     vertices reach the channel only through sum_k n_k n_k^T and")
    print("     sum_k n_k, and a pose moves neither. Proved by running")
    print("     exact_channel_R2 verbatim over Q(sqrt5)[C, T, t]")
    print()
    print(f"  {'solid':14s} {'tail 81 E[o^4], atlas':>22s} {'Decker':>10s}")
    print("  " + "-" * 50)
    for solid in SOLIDS:
        a, d = tails[solid]
        print(f"  {solid:14s} {a:22.4f} {d:10.4f}")
    print("[ok] and yet the pose DOES move the tails: the fourth moment is read")
    print("     against the Pauli axes, so the 2- and 3-designs shift (9 -> 15,")
    print("     27 -> 12) while the 5-designs, owning the sphere's moments")
    print("     through order four in every orientation, hold at 81/5. State-")
    print("     independent on every probe, and equal on all three axes.")
    # ..."costs the estimator nothing" also carries a proviso the scan below
    # makes exact: nothing PROVIDED the belief moves with the gate. A valid
    # reorientation is any member of the left coset (rotation group).R -- if
    # S and R both carry Decker's pose onto the atlas then S R^-1 permutes
    # the vertex axes -- and each member induces its own labelling of the
    # unreoriented run: believe m d_k while the device measures d_k -- the
    # relabelling applied WITHOUT the gate. The two-list channel prices
    # member m at exactly kappa = tr(m T)/3, tr(m)/3 noiseless, zero offset
    # (belief conjugates on the left, device on the right; sum_k d_k d_k^T =
    # (V/3) Id collapses B to (V/3) m, and the draw reduces the mismatch to
    # that trace).
    # Two exact laws follow, universal in solid and pose -- Schur again,
    # <(tr hX)^2> = |X|_F^2/3 over an irreducible draw: <kappa> = 0 (the
    # group average of h vanishes) and <kappa^2> = |R0 T|_F^2/27 =
    # |T|_F^2/27, noiselessly (X = R0, |R0|_F^2 = 3) exactly 1/9. So a
    # single member is never "the" per-solid figure;
    # the printed figures, Table D.4's, price rotation AND labelling against
    # Decker's own outcome order (check_decker_outcome_order) and land
    # inside the ranges below.
    #
    # BOTH noise settings are run, and only the noisy one tests the WIRING.
    # At T = Id the channel is symmetric in (belief, device) -- tr(B) is --
    # so swapping the two lists, or writing m for m.T (tr m = tr m^T), is
    # invisible: 0 of the 180 members notice either. And those two mutations
    # are ONE probe, not two: B = (V/3) m here, so both land the channel on
    # tr(m^T T)/3. Under T_NOISE that fires on the 170 members with m
    # asymmetric (the 10 symmetric ones stay blind), between 2.2e-03 and
    # 1.9e-01 against asserts at 1e-10. The belief-in-the-probabilities
    # mutation (probabilities built from b, collapsing the channel to
    # tr(T)/3) fires on all 180 in both settings, binding at 3.19e-02
    # noisy, 9.96e-02 noiseless (the dodecahedron's 1 - max kappa). Accept
    # side, worst over all 180: 1.8e-15 / 2.0e-17 (M, off) noiseless,
    # 1.2e-15 / 3.2e-17 noisy -- five orders of accept slack, seven of
    # reject. The offset asserts are weaker than they look: the noiseless
    # leg has t = 0, where off vanishes for ANY draw (R = [Id] included) by
    # the frame's zero sum alone; only the noisy leg's off =
    # (3/V) <R_g^T> B t pins the draw's irreducibility, and neither leg
    # sees the mismatch.
    print()
    print(f"  {'solid':14s} {'coset':>6s} {'induced kappa range':>20s}"
          f" {'min |kappa|':>12s} {'worst premium':>14s} {'mean premium':>13s}")
    print("  " + "-" * 87)
    noisy_min = 1.0
    for solid in SOLIDS:
        R0 = np.array(reorient[solid][1], dtype=float)
        rots = load_rotations(COVARIANCE[solid])
        s_decker = load_vertices(solid) @ R0
        kaps = np.empty(len(rots))
        kapn = np.empty(len(rots))
        # memb, not m -- m is this function's Fourier order, unpacked above
        for i, h in enumerate(rots):
            memb = h @ R0
            kaps[i] = np.trace(memb) / 3
            kapn[i] = np.trace(memb @ T_NOISE) / 3
            for Tn, tn, kap in ((np.eye(3), np.zeros(3), kaps[i]),
                                (T_NOISE, t_NOISE, kapn[i])):
                M, off = channel_R2(s_decker @ memb.T, rots, Tn, tn,
                                    s_actual=s_decker)
                d_M = np.abs(M - kap * np.eye(3)).max()
                assert d_M < 1e-10, \
                    f"{solid}: member not depol at tr(mT)/3 (max |dev| = {d_M:.2e})"
                assert np.allclose(off, 0, atol=1e-9), solid
        d_mean = abs(kaps.mean())
        d_msq = abs((kaps ** 2).mean() - 1 / 9)
        assert d_mean < 1e-12, f"{solid}: coset <kappa> != 0 ({d_mean:.2e})"
        assert d_msq < 1e-12, f"{solid}: coset <kappa^2> != 1/9 ({d_msq:.2e})"
        d_mean_n = abs(kapn.mean())
        d_msq_n = abs((kapn ** 2).mean() - (T_NOISE ** 2).sum() / 27)
        assert d_mean_n < 1e-12, f"{solid}: noisy <kappa> != 0 ({d_mean_n:.2e})"
        assert d_msq_n < 1e-12, \
            f"{solid}: noisy <kappa^2> != |T|_F^2/27 ({d_msq_n:.2e})"
        # a trace-zero member (a 120 deg rotation) would price at kappa = 0,
        # noiselessly a kill. None occurs, and every solid's nearest miss
        # sits PAST 120 deg, not short of it (the signed minimum is negative
        # on all five, asserted): the closest is the icosahedron's
        # kappa = -0.0117, i.e. 121.2 deg. Stated, so it cannot arrive as
        # an 'inf' in an otherwise assert-guarded table.
        assert np.abs(kaps).min() > 1e-9, f"{solid}: trace-zero coset member"
        assert kaps[np.abs(kaps).argmin()] < 0, f"{solid}: nearest miss short of 120 deg"
        noisy_min = min(noisy_min, np.abs(kapn).min())
        prem = 1 / kaps ** 2
        print(f"  {solid:14s} {len(rots):>6d}   [{kaps.min():+7.4f}, {kaps.max():+7.4f}]"
              f" {np.abs(kaps).min():>12.4f} {prem.max():>13.1f}x {prem.mean():>12.1f}x")
    print("[ok] and the proviso is priced: every coset member's induced labelling")
    print("     is exactly depolarizing at kappa = tr(m T)/3, zero offset -- the")
    print("     two-list channel, element by element, noiseless AND at the probe")
    print("     noise, the setting that tells belief from device. The columns")
    print("     above are the NOISELESS family tr(m)/3, centred on ZERO with")
    print("     RMS 1/3 (<kappa> = 0, <kappa^2> = 1/9; at the probe the laws")
    print("     are 0 and |T|_F^2/27, all four asserted), so applying the")
    print("     relabelling WITHOUT the gate turns a free correction into a")
    print("     shot premium the RMS reads as 9x. An understatement twice")
    print("     over: 1/kappa^2 is convex, the mean premium column running")
    print("     18x to 297x, and the noisy family comes far closer to the")
    print(f"     kill -- min |kappa| = {noisy_min:.1e}, a {1 / noisy_min ** 2:.1e}x premium. The kappa")
    print("     RANGES bound Table D.4's five (asserted there); premia they")
    print("     do not bound: D.4's dodecahedron undercuts every member's")
    print("     |kappa|, outpaying the worst column 6.6x. No single member")
    print("     is ever 'the' per-solid figure.")
