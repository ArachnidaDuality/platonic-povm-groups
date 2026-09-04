"""Section 2 of the randomized-implementation suite: the two jobs of
randomness (finding 4) -- the coset coin, the minimal and universal twirls,
the subgroup sweep and its exact companion, the C_3 coin-group witness and
the non-group draws beside it, and the atlas resources -- plus Section 5's
design-ladder remark.

randomized_implementations.py is the suite's entry point -- its docstring
carries the R1/R2 protocol definitions, the kappa/eta pin, the findings
index, the module map and the run instructions; `cd code && uv run
randomized_implementations.py` runs the whole suite. The finding proved
here:

  Finding 4  Randomness does two jobs -- REALIZE the POVM (a V/2-way coin
             over coset representatives; needs transitivity on the vertex
             axes) and TWIRL the noise (needs only irreducibility, i.e. a
             unitary 2-design). R1's single draw must do both, so its bill
             is the larger of two independent bars, and the two behave
             quite differently. The TWIRL bar is FLAT at T (order 12) for
             every solid, and that is a theorem rather than a sweep result:
             of the finite subgroups of SO(3) only T, O and I are
             irreducible, and T is the smallest -- nothing below T can twirl
             anything, in either protocol (check_universal_twirl). The
             REALIZE bar CLIMBS 3 -> 4 -> 12 -> 60 with the axis count.
             The bars therefore CROSS, exactly at the icosahedron: the
             twirl binds for the octahedron and the cube, both bind at the
             icosahedron, and realization binds for the dodecahedron.
             check_subgroup_sweep settles all four exhaustively over the
             complete subgroup lattices of O and I -- and thereby over EVERY
             finite subgroup of SO(3), since a draw that realizes permutes
             the vertex set and hence lies inside the solid's rotation
             group. Three things fall out there: the realizing and twirling
             SETS nest, and the nesting inverts at the icosahedron, where
             they coincide; the protocol's twirl test reproduces Schur
             exactly (twirl == irreducible, on all four); and since 2T is
             all-Clifford, every Phi in R1's ledger is charged to
             REALIZATION, the dodecahedron alone paying any
             (check_atlas_resources).

             The WITNESS worth printing is check_coin_group's: of the four
             coins only the octahedron's, {I, F, F+}, is closed under
             multiplication -- the cyclic C_3 about (1,1,1). It realizes the
             octahedral POVM exactly and twirls nothing, and the failure has
             a closed form, exact and for arbitrary noise. Schur's
             "irreducibly" made visible on the reader's own object.

             Its non-group companion is check_wilkens_layers, the
             in-repo assertion behind the paragraph that follows the
             witness (Appendix F.3.3.3). Wilkens et al. (arXiv:2603.28307,
             Fig. 1 and Eq. (6)) run randomized-projective on the octahedron
             with the six-element transversal {I, X} x {I, H, HS+} in place
             of the 2O draw, the X flip LAST before the readout, and
             calibrate their Eq. (5)'s f~ = (sigma_Z|Lambda|sigma_Z)/3 --
             our eta = T_zz/3. Layer by layer, exact and for arbitrary
             measurement-side (T, t): the flip kills the offset and T_zx,
             the three axes spread T_zz over the diagonal, and the residue
             is T_zy alone -- zero for their bit-flip readout, where the
             channel collapses to their Eq. (12), not for a rotation about
             xhat; the Klein completion {I, X, Y, Z} x {I, H, HS+} (12
             elements, still no group, H not in T) is exactly scalar. The
             order is load-bearing: X FIRST leaves T_zx in the residue
             beside T_zy, and an offset. So a non-group draw suffices once
             the noise class is narrowed -- the two bars above are minima
             over subgroups, as the thesis prints them, and this ensemble
             is why it prints "among groups" in so many words.

             check_flip_completion promotes that ensemble to a per-solid
             construction on the suite's OWN coin: the full Klein layer
             {I, X, Y, Z}, drawn with the coin and applied
             AFTER the fixed alignment. For every decomposable solid the
             2V-element ensemble realizes the POVM and is exactly
             depolarizing at T_zz for EVERY measurement-side (T, t) --
             the full group draw's channel, needing nothing of the coin
             beyond one representative per axis -- and it is a group
             exactly once: the octahedron's completion IS T (the coin is
             a transversal of the Klein subgroup in T = V x| C_3), while
             16/24/40 elements close under nothing. The flip order is
             load-bearing for exactly two solids, by a multiset fact:
             drawn BEFORE the alignment, V x coin lands inside T for
             both Phi-free coins -- the icosahedron's covers T uniformly
             twice (the minimal draw in disguise, so either order is
             exact), the cube's covers T at multiplicities {1, 2} (a
             non-uniform average: realization survives, the twirl does
             not) -- and the dodecahedron loses both jobs, its coin
             never re-spreading the collapsed Klein orbit of v0. The
             payoff is the dodecahedron, 0.4 Phi per shot against the
             full 2I draw's 0.8, and 0.4 is the FLOOR for every
             realizing draw of atlas words around the fixed alignment,
             group or not: a Phi-free word lies in the 2T copy, whose
             orbit misses the four inscribed-cube-diagonal axes, and a
             Phi-free POST-word cannot re-aim the seed -- no two
             dodecahedral vertex axes are orthogonal, and the 144-pair
             sweep decides it outright; the anchored-O pin extends the
             same guard to a Clifford-augmented gate set. The two bars
             stay minima over SUBGROUPS as printed; beyond groups the
             [0.4, 0.8] window closes at the floor.
"""

import itertools

import numpy as np
import sympy as sp
from sympy import Matrix, Rational

from randomized_core import (SOLIDS, COVARIANCE, T_NOISE, t_NOISE,
                             load_vertices, load_rotations, load_atlas,
                             rotation_from_unitary, rot_key, alignment,
                             channel_R1, channel_R2, orbit_counts, lattice,
                             subgroup_kind, by_order, two_bars, best_circuits,
                             coset_representatives, frame_potential,
                             symbolic_solids, atlas_gates, bloch_matrix,
                             exact_rotations, state_from_bloch, twirl_bar,
                             _fmt_orders, rank_one_twirl)
from randomized_field import (FIELD_GENS, FIELD_NAME, solid_field, to_field,
                              _mm, _mv, _tr, _eye, _as_float, exact_lattice,
                              exact_vertices, exact_alignment,
                              exact_orbit_directions, exact_orbit_counts,
                              exact_twirls, exact_twirls_R2, exact_two_bars,
                              exact_probe_span_ok)

def check_exact_two_bars():
    # The two-bars result is the one headline number in this module that the
    # float pipeline decides by tolerance three times over.  Here it is decided
    # by canonical form instead, and the two are required to agree.
    for g in ("T", "O", "I"):
        mul, subs = exact_lattice(g)
        R = load_rotations(g)
        idx = {rot_key(Rg): i for i, Rg in enumerate(R)}
        fmul = np.array([[idx[rot_key(Ri @ Rj)] for Rj in R] for Ri in R])
        assert (mul == fmul).all(), g            # entry for entry, not just the lattice
        assert subs == lattice(g), g
        num = np.array([[[float(e) for e in M.row(i)] for i in range(3)]
                        for M in exact_rotations(g)])
        assert np.abs(num - R).max() < 1e-12, g  # and the ROW ORDER agrees
    print("  exact rotations reproduce group_{T,O,I}.npz row for row; the exact")
    print("  multiplication tables agree with the rounding grid's entry for entry,")
    print("  so the lattices (10 / 30 / 59) coincide -- not merely in count")

    # The tetrahedron has ONE bar, not two: its SIC is not antipodal, so R1 --
    # and with it `realize` -- is undefined, and exact_two_bars is R1-shaped
    # throughout. So it is done here on the twirl side alone, over R2's generic
    # (T, t). Without it the ledger's fifth "smallest group that twirls" cell
    # is the only one still decided by the float sweep.
    K_tet = solid_field("tetrahedron")
    s_tet = exact_vertices("tetrahedron", K_tet)
    R_tet = [to_field(M, K_tet) for M in exact_rotations("T")]
    tw_tet, irr_tet = set(), set()
    for S in exact_lattice("T")[1]:
        RS = [R_tet[i] for i in sorted(S)]
        if exact_twirls_R2(s_tet, RS, K_tet):
            tw_tet.add(S)
        if sum((sum((RS[m][i][i] for i in range(3)), K_tet.zero) ** 2
                for m in range(len(RS))), K_tet.zero) == K_tet.convert(len(RS)):
            irr_tet.add(S)
    assert tw_tet == irr_tet                      # Schur again, exact on both sides
    assert tw_tet == twirl_bar("tetrahedron")[1]  # ... and subgroup for subgroup
    assert {len(S) for S in tw_tet} == {12}       # T alone, the whole group

    print(f"\n  {'solid':14s} {'field':16s} {'deg':>3s}  {'realize':>7s} {'twirl':>5s}"
          f"   agrees with the float sweep")
    print("  " + "-" * 82)
    print(f"  {'tetrahedron':14s} {FIELD_NAME['tetrahedron']:16s}"
          f" {K_tet.ext.minpoly.degree():>3d}  {'n/a':>7s}"
          f" {min(map(len, tw_tet)):>5d}   twirl + Schur identical (R1 undefined)")
    for solid in ("octahedron", "cube", "icosahedron", "dodecahedron"):
        e, b = exact_two_bars(solid), two_bars(solid)
        assert e["realize"] == b["realize"], solid
        assert e["twirl"] == b["twirl"], solid
        assert e["twirl"] == e["irr"], solid     # Schur, both sides exact now
        assert (e["bar_realize"], e["bar_twirl"]) == (b["bar_realize"], b["bar_twirl"])
        K = solid_field(solid)
        deg = K.ext.minpoly.degree() if FIELD_GENS[solid] else 1
        print(f"  {solid:14s} {FIELD_NAME[solid]:16s} {deg:>3d}  {e['bar_realize']:>7d}"
              f" {e['bar_twirl']:>5d}   realize + twirl + Schur all identical")

    # The narrow per-solid fields are safe BECAUSE the declaration is checked
    # rather than assumed. Exercise that -- the dodecahedron's vertices must
    # not coerce into the cube's field, so a re-posed solid or a mistyped
    # vertex raises instead of passing quietly.
    try:
        exact_vertices("dodecahedron", solid_field("cube"))
        raise AssertionError("out-of-field vertex coerced -- not fail-loud")
    except sp.polys.polyerrors.CoercionFailed:
        pass
    print("[ok] the two bars are exact: realize 3/4/12/60, twirl flat at 12 -- across")
    print("     all FIVE solids, the tetrahedron's single bar included -- crossing")
    print("     at the icosahedron -- no tolerance anywhere in the lattice, the orbit")
    print("     test or the twirl test. The twirl verdict is quantified over EVERY")
    print("     measurement-side (T, t), so it no longer rests on the T_NOISE probe;")
    print("     T_NOISE is confirmed a faithful witness rather than assumed to be one")


# ---------------------------------------------------------------------------
# Section 2: finding 4 -- the two jobs of randomness
# ---------------------------------------------------------------------------

def check_coset_coin():
    solids_sym = symbolic_solids()
    for solid in SOLIDS:
        verts = solids_sym[solid]
        V = len(verts)
        paired = all(any(sp.simplify(v + w) == Matrix([0, 0, 0]) for w in verts)
                     for v in verts)
        if not paired:
            assert solid == "tetrahedron"
            print(f"  {solid:14s} indecomposable -- no coin realization; Naimark forced")
            continue
        effects = [Rational(2, V) * state_from_bloch(v) for v in verts]
        assert sp.simplify(sum(effects, sp.zeros(2, 2)) - sp.eye(2)) == sp.zeros(2, 2)
        # the coin: pick one of V/2 axes w.p. 2/V, measure the {v, -v} basis
        axes = []
        for v in verts:
            if not any(sp.simplify(v + u) == Matrix([0, 0, 0])
                       or sp.simplify(v - u) == Matrix([0, 0, 0]) for u in axes):
                axes.append(v)
        coin = [Rational(2, V) * state_from_bloch(sgn * v)
                for v in axes for sgn in (1, -1)]
        remaining = list(effects)    # effect-for-effect multiset equality
        for C in coin:
            hit = next(j for j, E in enumerate(remaining)
                       if sp.simplify(C - E) == sp.zeros(2, 2))
            remaining.pop(hit)
        assert not remaining
        print(f"  {solid:14s} {V // 2}-way coin + projective readout"
              f" = the same {V} effects, exactly")
    print("[ok] the coin realizes the POVM itself (the effects, not merely the statistics)")


def check_minimal_twirl():
    R_T = load_rotations("T")
    kappa_R1 = T_NOISE[2, 2]
    print(f"R1 with the 12-rotation T draw (|2T| = 24 elements, all Clifford):")
    for solid in ("octahedron", "cube", "icosahedron", "dodecahedron"):
        s = load_vertices(solid)
        hits = orbit_counts(s, R_T)
        # the twirl works regardless: Schur needs only irreducibility, which
        # T has -- exactly depolarizing at the same kappa = T_zz, zero offset
        M, o = channel_R1(s, R_T, T_NOISE, t_NOISE)
        d_M = np.abs(M - kappa_R1 * np.eye(3)).max()      # measured 2.2e-16
        assert d_M < 1e-10, f"{solid}: T draw not depol at T_zz (max |dev| = {d_M:.2e})"
        assert np.allclose(o, 0, atol=1e-9)
        if solid == "dodecahedron":
            # ... but the REALIZATION fails: the T-orbit of v0 covers only
            # 6 of the 10 vertex axes -- 8 vertices are never measured, so
            # this is a (perfectly unbiased) 6-axis measurement, NOT the
            # dodecahedral POVM. R1's one draw must do both jobs, so the
            # dodecahedron forces the full 2I: transitivity on 10 axes needs
            # order divisible by 10 (T's 12 fails), leaving D5 as the only
            # proper candidate -- and its invariant C5 axis line makes it
            # reducible, so it cannot twirl.
            assert sorted(set(hits.tolist())) == [0, 2] and int((hits > 0).sum()) == 12
            print(f"  {solid:14s} twirl OK (kappa = T_zz) -- but orbit covers 6/10 axes:")
            print(f"  {'':14s} NOT the dodecahedral POVM; realization forces the full 2I draw")
            continue
        assert hits.min() == hits.max() == 24 // len(s), hits
        print(f"  {solid:14s} orbit uniform x{hits[0]}; depolarizing at kappa = {M[0, 0]:.6f}")
    print("[ok] g ~ Unif(T) realizes AND twirls octahedron/cube/icosahedron at kappa = T_zz;")
    print("     the dodecahedron is the sole solid whose projective route needs 2I")


def check_universal_twirl():
    R_T = load_rotations("T")
    kappa = np.trace(T_NOISE) / 3
    for solid in SOLIDS:
        s = load_vertices(solid)
        M, o = channel_R2(s, R_T, T_NOISE, t_NOISE)
        d_M = np.abs(M - kappa * np.eye(3)).max()         # measured 7.8e-16
        assert d_M < 1e-10, \
            f"{solid}: T draw not depol at tr(T)/3 (max |dev| = {d_M:.2e})"
        assert np.allclose(o, 0, atol=1e-9)
    print(f"[ok] R2 with g ~ Unif(T): exactly depolarizing at kappa = tr(T)/3 = {kappa:.6f}")
    print("     for ALL FIVE solids, tetrahedron and dodecahedron included:")
    print("     2T is the UNIVERSAL minimal twirl -- irreducibility is all R2 needs")

    # ... and it is minimal in the strong sense: sweep each solid's own
    # lattice and nothing below order 12 twirls anything. The bar is FLAT at
    # T across all five, which is why the ledger's twirl row is constant --
    # a theorem in disguise, since the finite subgroups of SO(3) are cyclic,
    # dihedral, T, O, I and only the last three are irreducible.
    print()
    for solid in SOLIDS:
        R = load_rotations(COVARIANCE[solid])
        bar, ok = twirl_bar(solid)
        irr = {S for S in lattice(COVARIANCE[solid])
               if np.isclose(np.mean([np.trace(R[i]) ** 2 for i in S]), 1.0)}
        assert ok == irr and bar == 12, (solid, bar)
        print(f"  {solid:14s} R2 twirls for {_fmt_orders(by_order(R, ok))}"
              f" -- bar at {bar}")
    print("[ok] the twirl bar is FLAT at T (order 12) for every solid, in both")
    print("     protocols: nothing smaller is irreducible, so nothing smaller twirls")


def check_subgroup_sweep():
    # Finding 4, exhaustively and on BOTH sides. The order-counting argument
    # ("transitivity on 10 axes needs order divisible by 10 -> only D5 ->
    # reducible") gets a brute-force replacement: every subgroup of O and of
    # I, tested independently for the two jobs -- realize (uniform vertex
    # hits) and twirl (estimator channel exactly depolarizing under the
    # generic probe) -- on all four decomposable solids.
    #
    # Deliberately NOT reported: how BADLY a subgroup that fails to twirl
    # fails, i.e. the size of the residual anisotropy ||M - T_zz I||. That
    # number is not a property of the group. Rerun under a second probe and
    # the failures reorder completely -- on the dodecahedron the order-5 and
    # order-10 subgroups go from the worst reducible draws to the best. Only
    # the zero/nonzero dichotomy is real, and `twirl == irr` is exactly it.
    #
    # subgroup_kind reads element orders off a float matrix power, and its
    # NAMES reach two printed tables (the two bar cells of the ledger and of
    # the sweep). Pin the whole census per lattice: a misclassification is then
    # loud here instead of silently relabelling a cell -- V as C_4 costs the
    # cube's realize bar its second name and changes nothing else in the file.
    census = {"O": {"1": 1, "C_2": 9, "C_3": 4, "C_4": 3, "D_3": 4, "D_4": 3,
                    "T": 1, "V": 4, "O": 1},
              "I": {"1": 1, "C_2": 15, "C_3": 10, "C_5": 6, "D_3": 10, "D_5": 6,
                    "T": 5, "V": 5, "I": 1}}
    keys_T = frozenset(rot_key(Rg) for Rg in load_rotations("T"))
    for g, n_subs, iso in (("O", 30, "S_4"), ("I", 59, "A_5")):
        R = load_rotations(g)
        subs = lattice(g)
        assert len(subs) == n_subs, (g, len(subs))
        kinds = sorted(subgroup_kind(R, S) for S in subs)
        assert {k: len(list(v)) for k, v in itertools.groupby(kinds)} == census[g], g
        print(f"  the lattice of {g}: {len(subs)} subgroups -- "
              + _fmt_orders(by_order(R, subs)))
        # irreducibility is one number per subgroup: <chi,chi> = mean tr^2 = 1
        irr = {S for S in subs
               if np.isclose(np.mean([np.trace(R[i]) ** 2 for i in S]), 1.0)}
        assert {len(S) for S in irr} == {12, len(R)}
        for S in irr:                    # every order-12 one is a copy of T
            assert len(S) == len(R) or any(
                frozenset(rot_key(R[h] @ R[i] @ R[h].T) for i in S) == keys_T
                for h in range(len(R)))
        print(f"  {'':16s} irreducible: {_fmt_orders(by_order(R, irr))}"
              f" -- every order-12 one verified conjugate to T")
    print("  (completeness self-verified in situ; 30 and 59 are the known subgroup")
    print("   counts of S_4 and A_5, so each census doubles as a free cross-check)")

    print(f"\n  {'solid':14s} {'realizing subgroups':38s} {'twirling subgroups':18s}"
          f" {'bars':9s} {'binds':8s} minimal draw")
    print("  " + "-" * 104)
    # counts AND names: the name at the realize bar is a printed cell, so it is
    # pinned with the order counts rather than discarded from by_order's output.
    expect = {"octahedron": ({3: (4, ["C_3"]), 6: (4, ["D_3"]), 12: (1, ["T"]),
                              24: (1, ["O"])}, "twirl", 12),
              "cube": ({4: (4, ["C_4", "V"]), 8: (3, ["D_4"]), 12: (1, ["T"]),
                        24: (1, ["O"])}, "twirl", 12),
              "icosahedron": ({12: (5, ["T"]), 60: (1, ["I"])}, "both", 12),
              "dodecahedron": ({60: (1, ["I"])}, "realize", 60)}
    bars = {}
    for solid in ("octahedron", "cube", "icosahedron", "dodecahedron"):
        b = bars[solid] = two_bars(solid)
        R, subs = b["rotations"], b["subgroups"]
        irr = {S for S in subs
               if np.isclose(np.mean([np.trace(R[i]) ** 2 for i in S]), 1.0)}
        assert b["twirl"] == irr, solid          # the protocol test IS Schur
        counts, binds, draw = expect[solid]
        assert by_order(R, b["realize"]) == counts, solid
        assert b["min_twirl"] == ["T"], solid    # the other printed name cell
        assert b["binds"] == binds and b["draw"] == draw, solid
        print(f"  {solid:14s} {_fmt_orders(by_order(R, b['realize'])):38s}"
              f" {_fmt_orders(by_order(R, b['twirl'])):18s}"
              f" {b['bar_realize']:>2d} vs{b['bar_twirl']:>3d}"
              f" {b['binds']:>8s}"
              f" {'/'.join(b['min_realize'] if binds == 'realize' else b['min_twirl'])}"
              f" ({draw})")

    # The bars do not merely cross in size: the two SETS nest, and the
    # nesting inverts exactly at the icosahedron, where they coincide.
    for solid, rel in (("octahedron", "twirl<realize"), ("cube", "twirl<realize"),
                       ("icosahedron", "equal"), ("dodecahedron", "realize<twirl")):
        b = bars[solid]
        if rel == "equal":
            assert b["realize"] == b["twirl"], solid
        elif rel == "twirl<realize":
            assert b["twirl"] < b["realize"], solid
        else:
            assert b["realize"] < b["twirl"], solid
    print("\n  the two sets NEST, and the nesting inverts at the icosahedron:")
    print("    octahedron, cube   twirl   < realize   -- every twirling draw realizes")
    print("    icosahedron        twirl   = realize   -- the same six subgroups")
    print("    dodecahedron       realize < twirl     -- every realizing draw twirls")

    # The dodecahedron's ceiling, and the five-inscribed-cubes argument made
    # quantitative.
    b = bars["dodecahedron"]
    assert b["ceiling"] == 6, b["ceiling"]
    R, s = b["rotations"], load_vertices("dodecahedron")
    _, v0 = alignment(s)
    # The inscribed-cube identification is a COUNT of distinct orbit points and
    # a pairwise |cos| -- both identity questions, both decided on the rounding
    # grid below.  Exact companions run alongside and must agree.
    Kd = solid_field("dodecahedron")
    Rd = [to_field(M, Kd) for M in exact_rotations("I")]
    _, v0e = exact_alignment(exact_vertices("dodecahedron", Kd), Kd)
    assert np.abs(np.array([float(Kd.to_sympy(c)) for c in v0e]) - v0).max() < 1e-12
    split = {}
    for S in (S for S in b["subgroups"] if len(S) == 12):
        r = b["reach"][S]
        split[r] = split.get(r, 0) + 1
        if r == 4:                        # the orbit IS an inscribed cube
            orbit = np.unique(np.round([Rg.T @ v0 for Rg in R[sorted(S)]], 9), axis=0)
            gram = np.abs(orbit @ orbit.T)
            # the deviation here is not float noise but the 9-decimal round the
            # orbit dedup above needs: it caps |gram - 1/3| at 3e-9 by
            # construction, and 2.3e-10 is measured
            d_gram = np.abs(gram[np.triu_indices(4, 1)] - 1 / 3).max()
            assert d_gram < 1e-8, \
                f"orbit is no inscribed cube (max |dev| = {d_gram:.2e})"
            oe = exact_orbit_directions([Rd[i] for i in sorted(S)], v0e, Kd)
            assert len(oe) == len(orbit) == 4, (len(oe), len(orbit))
            assert all(sum((p[k] * q[k] for k in range(3)), Kd.zero)
                       in (Kd.one / 3, -Kd.one / 3)
                       for j, p in enumerate(oe) for q in oe[j + 1:])
    assert split == {4: 2, 6: 3}, split
    print(f"\n  dodecahedron: no proper subgroup reaches more than {b['ceiling']}/10 vertex axes")
    print("    the five T's split 2 + 3: two send v0 around the 4 body diagonals of an")
    print("    inscribed cube (pairwise |cos| = 1/3, verified), three around the other 6")
    print("    -- v0 lies on exactly 2 of I's 5 inscribed cubes. Realization ALONE")
    print("    convicts the dodecahedron; the twirl is not even needed.")

    # The icosahedron's near-misses: which of the twelve C_5's and D_5's fall
    # one axis short, which axis each misses, and why the other two do not.
    b = bars["icosahedron"]
    R, s = b["rotations"], load_vertices("icosahedron")
    near = [S for S in b["subgroups"] if b["reach"][S] == 5]
    assert {len(S) for S in near} == {5, 10} and len(near) == 10
    for S in near:
        RS = R[sorted(S)]
        hits = orbit_counts(s, RS)
        # the 72-degree element: trace 1 + 2 cos 72 = tau, the largest any
        # non-identity rotation of a subgroup of I attains
        C5 = max((M for M in RS if not np.allclose(M, np.eye(3), atol=1e-9)),
                 key=np.trace)
        w, V = np.linalg.eig(C5)
        ax = np.real(V[:, np.argmin(np.abs(w - 1))])
        assert all(min(np.linalg.norm(s[k] - ax), np.linalg.norm(s[k] + ax)) < 1e-9
                   for k in np.flatnonzero(hits == 0))
    print("\n  icosahedron: ten of the twelve C_5's and D_5's reach 5 of the 6 axes --")
    print("    the axis each misses is the one its five-fold rotation fixes. The other")
    print("    two are seed-aligned and never leave v0's own axis, reaching 1. Neither")
    print("    is the ceiling: the five order-12 T-conjugates reach all 6 and realize")

    print("\n[ok] exhaustive over all 30 + 59 subgroups, hence over EVERY finite subgroup")
    print("     of SO(3) (a draw that realizes permutes the vertex set, so it lies")
    print("     inside the solid's rotation group): the twirl bar is T for all four,")
    print("     the realize bar climbs 3 -> 4 -> 12 -> 60, and they cross at the")
    print("     icosahedron -- the order-counting argument, brute-forced and generalized")
    return bars


def coin_rotations(solid):
    """The coin itself: the min-(magic, depth) representative per vertex axis.

    coset_representatives prices the axes; this returns the rotations behind
    those prices, so the coin can be asked whether it is a GROUP.
    """
    g = COVARIANCE[solid]
    R, s = load_rotations(g), load_vertices(solid)
    circuits = best_circuits(load_atlas(g))
    _, v = alignment(s)
    axes = []
    for n in s:
        if not any(np.allclose(n, -m, atol=1e-9) for m in axes):
            axes.append(n)
    out = []
    for n in axes:
        members = [i for i, Rg in enumerate(R)
                   if np.allclose(Rg.T @ v, n, atol=1e-9)
                   or np.allclose(Rg.T @ v, -n, atol=1e-9)]
        # same lexicographic min as coset_representatives, so this returns the
        # rotations behind exactly the words Table D.1 prints
        out.append(R[min(members, key=lambda i: circuits[rot_key(R[i])])])
    return np.array(out)


def exact_coin(solid):
    """coin_rotations(solid) over the solid's field, and its INDEX list.

    The witness printed in Appendix F.3.3.2 -- only the octahedron's coin is a
    group -- is a closure verdict, and closure is an identity question: it asks
    which element a product IS.  The float version answers it on rot_key's
    9-decimal grid.  Here the axes, the cosets and (in exact_coin_is_group) the
    closure itself are decided by field equality instead.

    What deliberately stays float is the min over circuit COSTS: that selects
    WHICH representative of a coset to take -- a choice, not an identity -- and
    it keys on rot_key exactly as the float coin does, so both return the same
    atlas index by construction.  The returned indices let the caller check the
    two coins agree element for element rather than merely both being coins.
    """
    g = COVARIANCE[solid]
    K = solid_field(solid)
    R = [to_field(M, K) for M in exact_rotations(g)]
    s = exact_vertices(solid, K)
    circuits, Rf = best_circuits(load_atlas(g)), load_rotations(g)
    _, v = exact_alignment(s, K)
    axes = []
    for n in s:
        if not any([-c for c in n] == m for m in axes):
            axes.append(n)
    idx = []
    for n in axes:
        neg = [-c for c in n]
        members = [i for i, Rg in enumerate(R) if _mv(_tr(Rg), v, K) in (n, neg)]
        assert members, (solid, "group not transitive on the vertex axes")
        idx.append(min(members, key=lambda i: circuits[rot_key(Rf[i])]))
    return [R[i] for i in idx], idx, K


def exact_coin_is_group(C, K):
    """Is the coin closed under multiplication?  Canonical keys, no grid.

    Field elements are canonical AND hashable, which is exactly what a set
    membership test needs and what simplify(a-b)==0 cannot supply.
    """
    keys = {tuple(x for row in M for x in row) for M in C}
    return all(tuple(x for row in _mm(A, B, K) for x in row) in keys
               for A in C for B in C)


def check_coin_group():
    # The witness: of the four coins only the OCTAHEDRON's is closed under
    # multiplication, so it realizes the POVM exactly while twirling nothing
    # -- Schur's "irreducibly" made necessary on the reader's own object.
    #
    # The verdict is decided TWICE: once on rot_key's rounding grid, once by
    # canonical field equality (exact_coin / exact_coin_is_group), and the two
    # are required to agree.  Closure asks which element a product IS, so it is
    # an identity computation, so it is decided in canonical form as well;
    # the float path stays because it is what the rest of this check keys on.
    groups = {}
    for solid in ("octahedron", "cube", "icosahedron", "dodecahedron"):
        C = coin_rotations(solid)
        keys = {rot_key(M) for M in C}
        closed = all(rot_key(A @ B) in keys for A in C for B in C)
        Ce, idx, K = exact_coin(solid)
        assert np.abs(np.array([[[float(K.to_sympy(e)) for e in row] for row in M]
                                for M in Ce]) - C).max() < 1e-12, solid
        assert [rot_key(load_rotations(COVARIANCE[solid])[i]) for i in idx] \
            == [rot_key(M) for M in C], solid    # the SAME coset representatives
        closed_exact = exact_coin_is_group(Ce, K)
        assert closed_exact == closed, solid
        hits = orbit_counts(load_vertices(solid), C)
        assert list(hits) == exact_orbit_counts(
            exact_vertices(solid, K), Ce, K), solid
        assert hits.min() == hits.max(), solid       # every coin realizes
        groups[solid] = closed
        print(f"  {solid:14s} {len(C):>2d}-word coin, realizes;"
              f" closed under multiplication: {'YES -- a group' if closed else 'no'}"
              f"  [exact: {'group' if closed_exact else 'not closed'}]")
    assert groups == {"octahedron": True, "cube": False,
                      "icosahedron": False, "dodecahedron": False}

    # dim End_G(R^3) = trace of the averaging projector X -> E_g R^T X R,
    # which is (1/|G|) sum tr(R)^2: three scalars for C_3, one for T/O/I.
    R_F = bloch_matrix(atlas_gates()["F"])
    C3 = [sp.eye(3), R_F, R_F * R_F]
    assert sp.expand(R_F**3 - sp.eye(3)) == sp.zeros(3, 3)
    # ... and the octahedron's coin IS that C_3, as a set of matrices rather
    # than as a set of words -- the identification the sentence above asserts.
    Ke = solid_field("octahedron")
    assert {tuple(x for row in M for x in row) for M in exact_coin("octahedron")[0]} \
        == {tuple(x for row in to_field(M, Ke) for x in row) for M in C3}
    assert R_F * Matrix([1, 1, 1]) == Matrix([1, 1, 1])   # the (1,1,1) axis
    print("  the octahedron's coin IS {I, R_F, R_F^2} as matrices, not just as words,")
    print("  and R_F fixes (1,1,1) -- the 120-degree turn about the body diagonal")
    dims = {}
    for name, Rs in (("C_3", C3), ("T", exact_rotations("T")),
                     ("O", exact_rotations("O")), ("I", exact_rotations("I"))):
        dims[name] = sp.simplify(sum(sp.trace(M) ** 2 for M in Rs) / len(Rs))
    assert dims == {"C_3": 3, "T": 1, "O": 1, "I": 1}, dims
    print("\n  dim End_G(R^3) = (1/|G|) sum tr(R)^2:  "
          + ",  ".join(f"{k} -> {v}" for k, v in dims.items()))
    print("  (the commutant of a reducible draw is 3-dimensional -- three scalars to")
    print("   calibrate, not one -- while T, O and I each buy Schur's single scalar)")

    # The channel identity, SYMBOLICALLY and for ARBITRARY noise: six free
    # symbols, no probe matrix anywhere. Over an irreducible group Schur
    # returns (v.w) Id_3 for any seed; over the coin the readout row of T
    # survives entire and merely cycles.
    v, w = Matrix(sp.symbols("v_1:4", real=True)), Matrix(sp.symbols("w_1:4", real=True))
    for name, Rs in (("T", exact_rotations("T")), ("O", exact_rotations("O")),
                     ("I", exact_rotations("I"))):
        M, off = rank_one_twirl(Rs, v, w)
        assert sp.simplify(M - v.dot(w) * sp.eye(3)) == sp.zeros(3, 3), name
        assert sp.simplify(off) == sp.zeros(3, 1), name
    print("\n  [exact, arbitrary v and w] over T, O, I:  M = (v.w) Id_3,  offset = 0")
    print("     and v.w = (A^T zhat).(A^T T^T zhat) = zhat^T T^T zhat = T_zz, the")
    print("     alignment cancelling because A is a rotation -- exact or not")

    a, b, c, tz = sp.symbols("T_zx T_zy T_zz t_z", real=True)
    M3, off3 = rank_one_twirl(C3, Matrix([0, 0, 1]), Matrix([a, b, c]))
    M3, off3 = sp.simplify(M3), sp.simplify(tz * off3)
    assert M3 == Matrix([[c, a, b], [b, c, a], [a, b, c]]), M3
    assert off3 == tz * Matrix([1, 1, 1]), off3
    print("\n  [exact, arbitrary noise] over the coin C_3 = <R_F>, seed v = zhat:")
    print(f"     M = circ(T_zz, T_zx, T_zy) = {M3.tolist()}")
    print(f"     offset = t_z (1,1,1) = {off3.T.tolist()}")
    print("     -- C_3 preserves the entire readout row of T and merely cycles it,")
    print("     where T destroys everything in that row but its diagonal entry.")
    print("     Conjugation-averaging preserves the trace, so both channels have")
    print("     trace 3 T_zz: the coin's diagonal is already right and only the")
    print("     off-diagonal circulant survives. The offset is t_z times the axis")
    print("     C_3 fixes, and dies for T because the solid is centered.")

    # The bridge: that reduction IS channel_R1, on every subgroup of both
    # lattices and all four solids -- so the symbolic claim above is a claim
    # about the protocol, not about a formula resembling it.
    worst = 0.0
    for solid in ("octahedron", "cube", "icosahedron", "dodecahedron"):
        s = load_vertices(solid)
        R = load_rotations(COVARIANCE[solid])
        A, v0 = alignment(s)
        zhat = np.array([0.0, 0.0, 1.0])
        # both this and the T_zz check below, which consumes the very same
        # A^T zhat, measure 1.1e-16
        d_v0 = np.abs(A.T @ zhat - v0).max()
        assert d_v0 < 1e-12, f"{solid}: A^T zhat != v0 (max |dev| = {d_v0:.2e})"
        vv, ww = A.T @ zhat, A.T @ T_NOISE.T @ zhat
        d_zz = abs(vv @ ww - T_NOISE[2, 2])
        assert d_zz < 1e-12, f"{solid}: v.w != T_zz (|dev| = {d_zz:.2e})"
        for S in lattice(COVARIANCE[solid]):
            RS = R[sorted(S)]
            M, o = channel_R1(s, RS, T_NOISE, t_NOISE)
            M2, o2 = rank_one_twirl(RS, vv, ww)
            worst = max(worst, np.abs(M - M2).max(),
                        np.abs(o - t_NOISE[2] * o2).max())
    assert worst < 1e-12, worst
    print("\n  reduction verified against channel_R1 on all 178 (subgroup, solid)")
    print(f"  pairs of both lattices: max |difference| = {worst:.1e}")

    # ... and the numeric instance the module's own probe produces, kept as
    # corroboration of the symbolic identity rather than as the claim.
    C = coin_rotations("octahedron")
    M, o = channel_R1(load_vertices("octahedron"), C, T_NOISE, t_NOISE)
    row = T_NOISE[2]                      # (T_zx, T_zy, T_zz), cycled
    d_M = np.abs(M - np.array([np.roll(row, k + 1) for k in range(3)])).max()
    d_o = np.abs(o - t_NOISE[2] * np.ones(3)).max()
    assert d_M < 1e-12 and d_o < 1e-12, (d_M, d_o)
    print("\n  on the module's generic probe the coin reads")
    print(f"     M = {np.round(M, 4).tolist()}, offset = {np.round(o, 4).tolist()}")
    print("[ok] the octahedron's coin is a group and does exactly one of the two jobs;")
    print("     it is the counterexample that makes Schur's hypothesis necessary")


def check_wilkens_layers():
    # The paragraph after the C_3 witness (Appendix F.3.3.3). Wilkens et al.
    # (arXiv:2603.28307) run randomized-projective on the octahedron with a
    # six-element NON-group standing in for the 2O draw: a uniform gate from
    # {I, H, HS+} selects the axis -- the coin's job -- and a random Pauli-X
    # flip follows it, LAST before the readout (their Fig. 1; the snapshot of
    # their Eq. (6) is U+ X |b><b| X U). Their per-qubit calibration
    # coefficient, Eq. (5)'s f~ = (sigma_Z|Lambda|sigma_Z)/3, is T_zz/3: our
    # eta, randomized-projective's scalar over 3. What the thesis prints is
    # the layer accounting of that ensemble under an ARBITRARY measurement-
    # side affine map r -> T r + t, and here it is asserted exactly. Four
    # symbols quantify over all twelve parameters because channel_R1 reads
    # row z of T A R_g and t_z and nothing else, for any list of rotations
    # (rank_one_twirl); the reduction is checked against channel_R1 itself
    # on the probe below. The thesis displays (1/|G|) sum R^T z z^T T R, so
    # the canonical 3 is divided back out.
    s = load_vertices("octahedron")
    A, v0 = alignment(s)                        # rank_one_twirl's v = A^T zhat: here A = I exactly
    assert np.array_equal(A, np.eye(3)) and np.array_equal(v0, [0.0, 0.0, 1.0]), (A, v0)
    G = atlas_gates()
    H = (1 / sp.sqrt(2)) * Matrix([[1, 1], [1, -1]])
    RX, RZ, RF, RH = (bloch_matrix(G["X"]), bloch_matrix(G["Z"]),
                      bloch_matrix(G["F"]), bloch_matrix(H))
    RY, I3 = RX * RZ, sp.eye(3)
    ex, ey, ez = Matrix([1, 0, 0]), Matrix([0, 1, 0]), Matrix([0, 0, 1])
    assert RH.T * ez == ex and RF.T * ez == ey    # H reads out xhat, HS+ reads out yhat
    a, b, c, tz = sp.symbols("T_zx T_zy T_zz t_z", real=True)

    def layer_avg(Rs):
        M, off = rank_one_twirl(Rs, ez, Matrix([a, b, c]))
        return sp.expand(M / 3), sp.expand(tz * off / 3)

    def after(last, first):                     # D drawn from `last` acts AFTER U
        return [D * U for D in last for U in first]

    # Closure and distinctness ask which element a product IS -- identity
    # computations, decided in canonical form: every entry is a SymPy Integer
    # (asserted, fail-loud), so a tuple of entries is a canonical, hashable
    # key; the float grid then decides a second time and
    # the two verdicts are required to agree, as in check_coin_group.
    def key(M):
        assert all(x.is_Integer for x in M), M
        return tuple(M)

    def floats(Rs):
        return np.array([np.array(M.tolist(), dtype=float) for M in Rs])

    def distinct(Rs):
        n = len({key(M) for M in Rs})
        assert n == len({rot_key(R) for R in floats(Rs)})
        return n

    def closed(Rs):
        exact = all(key(P * Q) in {key(M) for M in Rs} for P in Rs for Q in Rs)
        F, grid = floats(Rs), {rot_key(R) for R in floats(Rs)}
        assert exact == all(rot_key(P @ Q) in grid for P in F for Q in F)
        return exact

    flip, coin, klein = [I3, RX], [I3, RH, RF], [I3, RX, RY, RZ]
    residue = b / 3 * (ez * ey.T - ex * ey.T + ey * ex.T)

    # {I, X} alone: row z = (0, T_zy, T_zz), nothing else, and no offset
    M, off = layer_avg(flip)
    assert M == Matrix([[0, 0, 0], [0, 0, 0], [0, b, c]]) and off == sp.zeros(3, 1), M
    # {I, H, HS+} alone: a uniform diagonal T_zz/3, every off-diagonal, and the
    # offset of a three-axis coin -- it realizes the octahedron and is no group
    Mc, offc = layer_avg(coin)
    assert Mc == Matrix([[c, -b, a], [b, c, a], [a, b, c]]) / 3, Mc
    assert offc == tz / 3 * Matrix([1, 1, 1]) and not closed(coin)
    # jointly, X LAST as in Eq. (6): (T_zz/3) I + (T_zy/3)(z y^T - x y^T + y x^T),
    # zero offset; six elements, not closed. Scalar iff T_zy = 0: the diagonal
    # is uniform and the non-scalar part depends on T_zy alone, vanishing at
    # T_zy = 0 and not identically
    six = after(flip, coin)
    M6, off6 = layer_avg(six)
    assert M6 == sp.expand(c / 3 * I3 + residue) and off6 == sp.zeros(3, 1), M6
    nonscalar = M6 - M6[0, 0] * I3
    assert M6[0, 0] == M6[1, 1] == M6[2, 2] and nonscalar.free_symbols == {b}
    assert nonscalar.subs(b, 0) == sp.zeros(3, 3) and nonscalar != sp.zeros(3, 3)
    assert distinct(six) == 6 and not closed(six)
    # the order is load-bearing (control): X FIRST leaves both T_zx and T_zy in
    # the residue, (T_zx/3) y z^T + (T_zy/3) z y^T, and an offset t_z/3 (1, 0, 0)
    # -- the flip cannot kill what the coin rotates in behind it
    xfirst = after(coin, flip)
    M6f, off6f = layer_avg(xfirst)
    assert M6f == sp.expand(c / 3 * I3 + a / 3 * ey * ez.T + b / 3 * ez * ey.T), M6f
    assert off6f == tz / 3 * Matrix([1, 0, 0]), off6f
    assert distinct(xfirst) == 6 and not closed(xfirst)
    # the Klein completion {I, X, Y, Z} x {I, H, HS+}: exactly scalar, zero
    # offset, twelve elements -- and still no group, since H is not in T and T
    # is the only subgroup of O of order 12
    twelve = after(klein, coin)
    M12, off12 = layer_avg(twelve)
    assert M12 == c / 3 * I3 and off12 == sp.zeros(3, 1), M12
    assert distinct(twelve) == 12 and not closed(twelve)
    keys_T = {rot_key(R) for R in load_rotations("T")}
    assert rot_key(np.array(RH.tolist(), dtype=float)) not in keys_T
    assert rot_key(np.array(RF.tolist(), dtype=float)) in keys_T
    # their coin against Table D.1's C_3 = {I, F, F+}: shares I and F = HS+,
    # swaps F+ for H, so the C_3 circulant is not its witness -- the same six
    # off-diagonal slots, and only row xhat differs: H puts (-T_zy, T_zx)
    # where the circulant has (T_zx, T_zy)
    C3 = [I3, RF, RF * RF]
    assert RF * RF == RF.T and closed(C3)
    assert {key(M) for M in coin} & {key(M) for M in C3} == {key(I3), key(RF)}
    Mc3 = layer_avg(C3)[0]
    assert Mc3 == Matrix([[c, a, b], [b, c, a], [a, b, c]]) / 3      # check_coin_group's form
    assert all(Mc[i, j] != 0 and Mc3[i, j] != 0 for i in range(3) for j in range(3) if i != j)
    assert (Mc - Mc3)[1:, :] == sp.zeros(2, 3) and (Mc - Mc3)[0, :] == Matrix([[0, -a - b, a - b]]) / 3
    # their noise model, Eqs. (10)-(12): a classical bit-flip readout, P(read
    # 1 | 0) = p01 and P(read 0 | 1) = p10, measures Z_read = E_0 - E_1 with
    # E_0 = (1 - p01)|0><0| + p10 |1><1|. Its readout row is DERIVED, not
    # entered: Z_read = t_z I + T_zx X + T_zy Y + T_zz Z gives T_zx = T_zy = 0,
    # T_zz = 1 - p01 - p10, t_z = p10 - p01, and the six-element channel
    # collapses to ((1 - 2 pflip)/3) I with 2 pflip = p01 + p10 -- their
    # calibration coefficient, the offset (the asymmetry) gone. The thesis's
    # counterexample is a rotation about xhat by theta, whose readout row is
    # (0, sin theta, cos theta): T_zy = sin theta, and the residue stays.
    p01, p10, th = sp.symbols("p_01 p_10 theta", real=True)
    X2, Y2, Z2 = (Matrix([[0, 1], [1, 0]]), Matrix([[0, -sp.I], [sp.I, 0]]),
                  Matrix([[1, 0], [0, -1]]))
    E0 = (1 - p01) * (sp.eye(2) + Z2) / 2 + p10 * (sp.eye(2) - Z2) / 2
    Zread = E0 - (sp.eye(2) - E0)
    bitflip = {sym: sp.expand((Zread * P).trace() / 2)
               for sym, P in ((tz, sp.eye(2)), (a, X2), (b, Y2), (c, Z2))}
    assert bitflip == {a: 0, b: 0, c: 1 - p01 - p10, tz: p10 - p01}, bitflip
    assert sp.expand(M6.subs(bitflip) - (1 - p01 - p10) / 3 * I3) == sp.zeros(3, 3)
    row_x = bloch_matrix(sp.cos(th / 2) * sp.eye(2) - sp.I * sp.sin(th / 2) * X2)[2, :]
    assert row_x == Matrix([[0, sp.sin(th), sp.cos(th)]]), row_x
    rot_x = {a: row_x[0], b: row_x[1], c: row_x[2], tz: 0}
    assert sp.simplify(M6.subs(rot_x) - sp.cos(th) / 3 * I3 - residue.subs(rot_x)) \
        == sp.zeros(3, 3) and residue.subs(rot_x) != sp.zeros(3, 3)
    # the residue kills zhat, so the |0> calibration reads T_zz/3 for every
    # affine map and cannot see it: a bias, not a premium (the print's last
    # sentence on this ensemble; Appendix F.2.2's blindness in a milder form)
    assert residue * ez == sp.zeros(3, 1) and M6 * ez == c / 3 * ez
    # the bridge to the protocol: channel_R1 on the octahedron under the
    # module's probe agrees with the symbolic forms (canonical 3 restored),
    # offsets included -- the coin alone and X first each carry one, t_z
    # sized -- and all four ensembles realize the POVM, every vertex hit
    # equally often
    probe = {a: T_NOISE[2, 0], b: T_NOISE[2, 1], c: T_NOISE[2, 2], tz: t_NOISE[2]}
    worst, largest_offset = 0.0, 0.0
    for Rs, Msym, osym, hits_each in ((coin, Mc, offc, 1), (six, M6, off6, 2),
                                      (xfirst, M6f, off6f, 2), (twelve, M12, off12, 4)):
        Rn = floats(Rs)
        M, o = channel_R1(s, Rn, T_NOISE, t_NOISE)
        Mn = 3 * np.array(Msym.subs(probe).tolist(), dtype=float)
        on = 3 * np.array(osym.subs(probe).tolist(), dtype=float).ravel()
        worst = max(worst, np.abs(M - Mn).max(), np.abs(o - on).max())
        largest_offset = max(largest_offset, np.abs(o).max())
        hits = orbit_counts(s, Rn)
        assert hits.min() == hits.max() == hits_each, hits
    assert worst < 1e-12 and abs(largest_offset - abs(t_NOISE[2])) < 1e-12, (worst, largest_offset)

    print("  Wilkens et al.'s ensemble on the octahedron, X flip last, (1/|G|) sum R^T z z^T T R:")
    print(f"  {'layer':22s} {'channel':52s} offset")
    print("  " + "-" * 86)
    print(f"  {'{I, X}':22s} {'row z = (0, T_zy, T_zz), nothing else':52s} 0")
    print(f"  {'{I, H, HS+}':22s} {'T_zz/3 on the diagonal, every off-diagonal':52s} t_z/3 (1,1,1)")
    print(f"  {'both, X last (6)':22s} {'(T_zz/3) I + (T_zy/3)(z y^T - x y^T + y x^T)':52s} 0")
    print(f"  {'both, X first (6)':22s} {'(T_zz/3) I + (T_zx/3) y z^T + (T_zy/3) z y^T':52s} t_z/3 (1,0,0)   (control)")
    print(f"  {'Klein x coin (12)':22s} {'(T_zz/3) I':52s} 0")
    print("  none of the four ensembles is a group (H is not in T); bit-flip readout")
    print("  collapses the six to ((1 - 2 pflip)/3) I, their Eq. (12); a rotation about")
    print(f"  xhat does not; channel_R1 agrees on the probe to {worst:.1e}, offsets included,")
    print("  and all four realize")
    print("[ok] a non-group draw suffices once the noise class is narrowed: the six-")
    print("     element channel is exact iff T_zy = 0. The two bars are minima over")
    print("     subgroups, as printed; the thesis says 'among groups' in so many")
    print("     words, and this is the ensemble that makes that necessary")


def check_flip_completion():
    # check_wilkens_layers' ensemble, promoted to a per-solid construction on
    # the suite's OWN coin: draw (word, flip) uniformly from the min-Phi coin
    # times the full Klein layer {I, X, Y, Z}, the flip applied AFTER the
    # fixed alignment -- run the word, align, flip, read Z. As effective
    # draws that is A^T P A U (channel_R1 applies A on top of the draw, so
    # A . A^T P A U = P A U, the circuit just named), and a conjugated flip
    # can never re-aim the seed: A^T P A v0 = A^T P zhat = +-A^T zhat = +-v0,
    # a sign the snapshot folds into the outcome b. Two consequences, both
    # asserted below: realization is inherited from the coin axis for axis --
    # every vertex hit exactly 4 times, so vertex n's aggregated effect is
    # (4/|E|)(I + n.sigma)/2 = (2/V)(I + n.sigma)/2, the solid's own POVM
    # element, exactly -- and the ensemble's Phi bill is the coin's, since
    # the flips are Phi-free words and the alignment is the protocol's fixed
    # overhead, drawn by nobody.
    #
    # The channel: exactly depolarizing at T_zz, zero offset, for EVERY
    # measurement-side (T, t) -- the full group draw's channel (finding 1) --
    # from 2V elements. The mechanism owes Schur nothing: averaging the four
    # flips with their snapshot signs kills every readout-row component but
    # T_zz zhat^T, and the coin then spreads zhat zhat^T into Id/3 because
    # its axes average n n^T to Id/3 (the vertex set is a tight frame,
    # sum over V of n n^T = (V/3) Id). Nor could Schur apply: the ensemble
    # is a group exactly once. At the octahedron {I, X, Y, Z} x {I, F, F+}
    # IS T -- the coin is a transversal of the Klein subgroup in T = V x| C_3,
    # so the completion reassembles the minimal draw itself -- while the
    # other three close under nothing: closure is refuted element for
    # element below, and each of the three even leaves its covariance
    # group, witnessed by one element (asserted): A^T Z A, the half-turn
    # about v0's axis, a 3-, 5- and 3-fold axis of the cube's,
    # icosahedron's and dodecahedron's rotation group. Only the
    # octahedron's v0 sits on a 4-fold axis, which is why its completion
    # may close at all. (Not every conjugated flip leaves: A^T Y A is
    # icosahedral, A^T X A dodecahedral -- the Z witness is the account,
    # not "never a symmetry".)
    G = atlas_gates()
    RXs, RZs = bloch_matrix(G["X"]), bloch_matrix(G["Z"])
    flips_sym = [sp.eye(3), RXs, RXs * RZs, RZs]         # I, X, Y = XZ, Z
    signs = (1, -1, -1, 1)                               # P zhat = s_P zhat
    ez = Matrix([0, 0, 1])
    for P, s_P in zip(flips_sym, signs, strict=True):
        assert P.T == P and P.T * ez == s_P * ez, P      # symmetric half-turns
    flips_f = [np.array(P.tolist(), dtype=float) for P in flips_sym]
    # the Klein layer lies IN T: what lets V x| C_3 reassemble T at the
    # octahedron, and what will price the flips at zero Phi in every atlas
    # (their circuits are asserted Phi-free per solid, in the bill below)
    keys_T = {rot_key(Rg) for Rg in load_rotations("T")}
    assert all(rot_key(P) in keys_T for P in flips_f)

    def key(M):
        return tuple(M[i][j] for i in range(3) for j in range(3))

    expect = {"octahedron": (12, True), "cube": (16, False),
              "icosahedron": (24, False), "dodecahedron": (40, False)}
    bills = {"octahedron": 0, "cube": 0, "icosahedron": 0,
             "dodecahedron": Rational(2, 5)}
    worst = 0.0

    print(f"  {'solid':14s} {'|E|':>3s} {'group':>5s} {'hits':>4s}"
          f"  {'channel, EVERY (T, t)':21s} {'Phi/shot':>8s}"
          f"  flips BEFORE the alignment (control)")
    print("  " + "-" * 104)
    for solid in ("octahedron", "cube", "icosahedron", "dodecahedron"):
        s = load_vertices(solid)
        V = len(s)
        Cf = coin_rotations(solid)
        Ce, _, K = exact_coin(solid)
        se = exact_vertices(solid, K)
        Af, _ = alignment(s)
        Ae, v0e = exact_alignment(se, K)
        d_A = np.abs(_as_float(Ae, K) - Af).max()
        assert d_A < 1e-12, (solid, d_A)                 # measured 1.1e-16
        # A^T A = Id exactly -- the premise that reads A . (A^T P A U) as
        # P A U, the circuit named above
        assert _mm(_tr(Ae), Ae, K) == _eye(K), solid
        Pe = [to_field(P, K) for P in flips_sym]

        # the conjugated flip fixes the seed AXIS; the sign is its whole action
        ApA = [_mm(_mm(_tr(Ae), P, K), Ae, K) for P in Pe]
        for Q, s_P in zip(ApA, signs, strict=True):
            assert _mv(Q, v0e, K) == [s_P * c for c in v0e], solid

        Ee = [_mm(Q, U, K) for Q in ApA for U in Ce]
        Ef = [Af.T @ P @ Af @ U for P in flips_f for U in Cf]
        d_E = max(np.abs(_as_float(M, K) - R).max()
                  for M, R in zip(Ee, Ef, strict=True))
        assert d_E < 1e-12, (solid, d_E)                 # measured 2.2e-16

        # distinct and closed are identity questions: decided on canonical
        # field keys AND on rot_key's grid, the two verdicts required to agree
        grid = {rot_key(R) for R in Ef}
        assert len({key(M) for M in Ee}) == len(grid) == 2 * V \
            == expect[solid][0], solid
        closed = exact_coin_is_group(Ee, K)
        assert closed == all(rot_key(P @ Q) in grid for P in Ef for Q in Ef)
        assert closed == expect[solid][1], solid
        if solid == "octahedron":
            # closed -- and IS T, on the field keys and on the grid alike
            assert grid == keys_T
            assert {key(M) for M in Ee} \
                == {key(to_field(M, K)) for M in exact_rotations("T")}

        # realize, decided twice; x4 is what makes the effects the POVM's
        hits = orbit_counts(s, np.array(Ef))
        assert list(hits) == exact_orbit_counts(se, Ee, K), solid
        assert hits.min() == hits.max() == 4, (solid, hits)

        # the channel for EVERY (T, t): exact_twirls' four probes decide all
        # twelve noise parameters (linearity, and only row z of T and t_z
        # can enter channel_R1 at all -- for an arbitrary rotation list, not
        # just a group). exact_probe_span_ok is the standing transcription
        # guard that rows x and y really cannot contribute: it guards
        # exact_channel_R1, not the ensemble. The bridge to channel_R1 on the
        # probe, offsets included, is the value-for-value pairing every exact
        # companion here carries; the worst deviation is tracked and printed
        # at the end.
        assert exact_twirls(se, Ee, K), solid
        assert exact_probe_span_ok(se, Ee, K), solid
        M, off = channel_R1(s, np.array(Ef), T_NOISE, t_NOISE)
        d = max(np.abs(M - T_NOISE[2, 2] * np.eye(3)).max(), np.abs(off).max())
        assert d < 1e-12, (solid, d)
        worst = max(worst, d)

        # the bill: each coin word appears once per flip, and the flips are
        # Phi-free words -- asserted, not narrated: their cheapest atlas
        # circuits carry no Phi -- so the ensemble mean is the coin mean, 0
        # everywhere but the dodecahedron's 2/5. Nothing about the coin
        # beyond one-representative-per-axis entered the channel above; the
        # min-Phi choice is what makes 2/5 the attained floor.
        g = COVARIANCE[solid]
        circuits = best_circuits(load_atlas(g))
        assert all(circuits[rot_key(P)][0] == 0 for P in flips_f), solid
        reps = coset_representatives(s, load_rotations(g), circuits)
        bill = Rational(4 * sum(r[0] for r in reps), 2 * V)
        assert bill == bills[solid], (solid, bill)
        # the header comment's witness that the three big ensembles leave
        # the covariance group: A^T Z A, the half-turn about v0's own axis
        if solid != "octahedron":
            assert rot_key(Af.T @ flips_f[3] @ Af) \
                not in {rot_key(Rg) for Rg in load_rotations(g)}, solid

        # control: the SAME flips drawn BEFORE the alignment (draws P U);
        # the octahedron's A = Id makes that the completion itself, so there
        # is nothing separate to run there. The order is load-bearing for
        # exactly two solids, and the mechanism is a MULTISET: both Phi-free
        # coins put V x coin inside T, so the before-order is a T-supported
        # draw -- the icosahedron's 24 pairs cover T uniformly TWICE, the
        # minimal draw in disguise (Schur applies after all), while the
        # cube's 16 pairs cover T at multiplicities {1, 2}, a non-uniform
        # average that loses the twirl even though the vertex marginal still
        # evens out. The dodecahedron loses both jobs: the Klein orbit of
        # v0 collapses to two axes (X v0 = -v0, asserted -- but shared by
        # the icosahedron, whose uniform cover survives it, so the collapse
        # alone decides nothing) and the ten-word coin does not re-spread
        # them, hits 2..6. Twirl failures are asserted as margins (measured
        # 4.7e-2 / 5.3e-2 on the probe against 1e-12 passes) and NOT
        # reported as sizes -- check_subgroup_sweep's standing policy: only
        # the zero/nonzero dichotomy is a property of the draw.
        if solid == "octahedron":
            assert np.array_equal(Af, np.eye(3)) and Ae == _eye(K)
            note = "same ensemble (A = Id)"
        else:
            Eb_e = [_mm(P, U, K) for P in Pe for U in Ce]
            Eb_f = [P @ U for P in flips_f for U in Cf]
            Mb, ob = channel_R1(s, np.array(Eb_f), T_NOISE, t_NOISE)
            db = max(np.abs(Mb - T_NOISE[2, 2] * np.eye(3)).max(),
                     np.abs(ob).max())
            hb = orbit_counts(s, np.array(Eb_f))
            assert list(hb) == exact_orbit_counts(se, Eb_e, K), solid
            mult = {}
            for R in Eb_f:
                kb = rot_key(R)
                mult[kb] = mult.get(kb, 0) + 1
            if solid == "cube":
                assert set(mult) == keys_T \
                    and sorted(mult.values()) == [1] * 8 + [2] * 4, solid
                assert not exact_twirls(se, Eb_e, K) and db > 1e-2, solid
                assert hb.min() == hb.max() == 4, solid
                note = "in T at mult {1,2}: twirl LOST, realizes"
            elif solid == "icosahedron":
                assert set(mult) == keys_T and set(mult.values()) == {2}, solid
                assert exact_twirls(se, Eb_e, K) and db < 1e-12, solid
                assert hb.min() == hb.max() == 4, solid
                note = "the T draw twice over: still exact"
            else:
                assert _mv(Pe[1], v0e, K) == [-c for c in v0e], solid
                assert not exact_twirls(se, Eb_e, K) and db > 1e-2, solid
                assert (int(hb.min()), int(hb.max())) == (2, 6), solid
                note = "BOTH jobs lost (hits 2..6)"
        print(f"  {solid:14s} {2 * V:>3d} {('T !' if closed else 'no'):>5s}"
              f" {'x4':>4s}  {'T_zz Id_3, offset 0':21s}"
              f" {float(bill):>8.1f}  {note}")

    # the FLOOR. 0.4 is not merely attained -- it bounds every realizing
    # draw of atlas words around the fixed alignment, group or not, and the
    # quantifier is the protocol SHAPE: any drawn words BEFORE and AFTER
    # the alignment (Phi-free words multiply into the 2T copy on each side,
    # which is closed, so any number of free layers is one free word).
    # Realization forces a uniform axis marginal. A Phi-free PRE-word lies
    # in the 2T copy, whose orbit reaches only six of the ten axes. A
    # Phi-free POST-word cannot re-aim the seed: in T it sends zhat to a
    # coordinate axis, A^T of a horizontal one is perpendicular to v0, and
    # no two dodecahedral vertex axes are orthogonal (the census below:
    # pairwise |cos| is 1/3 or sqrt5/3, never 0) -- so realizing forces it
    # to fix the zhat axis, which inside T means the Klein layer already
    # covered. Hence a shot landing on one of the four unreached axes
    # spent a Phi somewhere: >= 0.4 per shot, attained by the completion.
    # The four unreached axes are the diagonals of ONE inscribed cube
    # (pairwise |cos| = 1/3, decided in the field), and the picture closes
    # as a dichotomy, asserted axis for axis: Phi-free iff a zero
    # coordinate survives iff the T draw reaches it.
    s = load_vertices("dodecahedron")
    K = solid_field("dodecahedron")
    se = exact_vertices("dodecahedron", K)
    Ae, v0e = exact_alignment(se, K)
    Af, v0 = alignment(s)
    R_I = load_rotations("I")
    reps = coset_representatives(s, R_I, best_circuits(load_atlas("I")))
    axes_f, vidx, axes_e = [], [], []
    for i, n in enumerate(s):
        if not any(np.allclose(n, -m, atol=1e-9) for m in axes_f):
            axes_f.append(n)
            vidx.append(i)
            axes_e.append(list(se[i]))
    assert len(axes_f) == len(reps) == 10
    # the per-axis minima as a multiset (coset_representatives minimises
    # magic-first over every member of each coset), and which four they are
    assert sorted(r[0] for r in reps) == [0] * 6 + [1] * 4, reps
    costly = [k for k, r in enumerate(reps) if r[0] > 0]
    hits_T = orbit_counts(s, load_rotations("T"))
    assert {k for k in range(10) if hits_T[vidx[k]] == 0} == set(costly)
    # the 0.8 the completion halves, pinned locally: 96 of the 120 atlas
    # words carry the single Phi (mean 0.8/shot for the full 2I draw)
    magic_I = load_atlas("I")["dij_magic_costs"]
    assert len(magic_I) == 120 and int(magic_I.sum()) == 96 \
        and int(magic_I.max()) == 1

    third = K.one / 3
    s53 = K.from_sympy(sp.sqrt(5)) / 3
    # the angle census over all ten axes: {1/3, sqrt5/3}, both realized,
    # and in particular NO orthogonal pair anywhere
    both = set()
    for a in range(10):
        for b in range(a + 1, 10):
            dot = sum((axes_e[a][t] * axes_e[b][t] for t in range(3)), K.zero)
            assert dot in (third, -third, s53, -s53), (a, b)
            both.add(dot in (third, -third))
    assert both == {True, False}
    for a in range(4):
        for b in range(a + 1, 4):
            dot = sum((axes_e[costly[a]][t] * axes_e[costly[b]][t]
                       for t in range(3)), K.zero)
            assert dot in (third, -third)                # one cube's diagonals
    assert K.zero in v0e
    assert all((K.zero in axes_e[k]) == (k not in costly) for k in range(10))

    # the post-word sweep: all 144 Phi-free (pre, post) pairs, the snapshot
    # U^T A^T W^T zhat decided over K and on the float grid, the verdicts
    # required to agree -- it is a vertex exactly when the post-word fixes
    # the zhat axis (the 48 Klein cases), and then always on a free axis
    ez_f = np.array([0.0, 0.0, 1.0])
    ez_e = [K.zero, K.zero, K.one]
    R_T_f = load_rotations("T")
    R_T_e = [to_field(M, K) for M in exact_rotations("T")]
    on_vertex = 0
    for Wf, We in zip(R_T_f, R_T_e, strict=True):
        top_e = _mv(_tr(We), ez_e, K)
        fixes_z = top_e in (ez_e, [-c for c in ez_e])
        atop_e = _mv(_tr(Ae), top_e, K)
        atop_f = Af.T @ (Wf.T @ ez_f)
        for Uf, Ue in zip(R_T_f, R_T_e, strict=True):
            w_e = _mv(_tr(Ue), atop_e, K)
            hit = [k for k in range(10)
                   if w_e == axes_e[k] or w_e == [-c for c in axes_e[k]]]
            hit_f = bool(np.linalg.norm(s - Uf.T @ atop_f, axis=1).min() < 1e-9)
            assert bool(hit) == hit_f == fixes_z, (bool(hit), hit_f, fixes_z)
            if hit:
                on_vertex += 1
                assert hit[0] not in costly
    assert on_vertex == 48

    # the geometry pin: the anchored poses share EXACTLY T (decided on the
    # field keys and on the grid), O is the signed permutations (exact census),
    # and no rotation of the anchored O takes v0 anywhere near the four
    # diagonals -- structurally: a signed permutation cannot shed v0's zero
    # coordinate, and the diagonals have none. This is the guard for a
    # Clifford-AUGMENTED gate set: were H or S free gates, Phi-free words
    # would generate 2O rather than 2T, and the four axes stay unreached.
    keys_O = {rot_key(Rg) for Rg in load_rotations("O")}
    assert keys_O & {rot_key(Rg) for Rg in R_I} == keys_T
    K5 = sp.QQ.algebraic_field(sp.sqrt(5))

    def key5(M):
        F = to_field(M, K5)
        return tuple(F[i][j] for i in range(3) for j in range(3))

    assert ({key5(M) for M in exact_rotations("O")}
            & {key5(M) for M in exact_rotations("I")}
            == {key5(M) for M in exact_rotations("T")})
    for Mo in exact_rotations("O"):
        assert all(Mo[i, j] in (-1, 0, 1) for i in range(3) for j in range(3))
    sep = min(np.abs(w - sg * np.asarray(axes_f[k])).max()
              for Rg in load_rotations("O") for w in (Rg @ v0, Rg.T @ v0)
              for k in costly for sg in (1, -1))
    assert sep > 0.1, sep                    # a margin, not a threshold

    print("\n  dodecahedron floor: Phi-free iff a zero coordinate iff the T draw")
    print("  reaches it, axis for axis (six of ten); the other four are one inscribed")
    print("  cube's diagonals (|cos| = 1/3 pairwise, exact). No two vertex axes are")
    print("  orthogonal, so of the 144 Phi-free (pre, post)-word pairs exactly the")
    print("  48 Klein cases realize, all on free axes: a free post-layer cannot")
    print("  re-aim the seed, and the floor quantifies over words on BOTH sides of")
    print("  the alignment. O cap I = T in the anchored poses (both verdicts), and")
    print(f"  the anchored O misses the four diagonals by {sep:.2f} (max-norm) -- the")
    print("  pin that keeps the floor under a Clifford-augmented gate set")
    print("[ok] the flip-completed coin does both jobs exactly for every affine")
    print("     (T, t) on all four decomposable solids -- channel_R1 agrees on the")
    print(f"     probe to {worst:.1e}, offsets included -- and is a group exactly")
    print("     once: the octahedron's completion IS T, the minimal draw. The")
    print("     dodecahedron pays the coin's 0.4 Phi per shot against the full 2I")
    print("     draw's 0.8, and 0.4 is the floor for every realizing draw of atlas")
    print("     words around the fixed alignment, group or not. The two bars stay")
    print("     minima over SUBGROUPS, as printed; beyond groups the [0.4, 0.8]")
    print("     window closes at the floor")


def check_atlas_resources():
    # the 2T draw is free: 24 Clifford words of depth <= 2
    atlas_T = load_atlas("T")
    assert int(atlas_T["bfs_depths"].max()) == 2
    assert int(atlas_T["dij_magic_costs"].max()) == 0
    print("[ok] 2T: all 24 elements at BFS depth <= 2, magic 0 (gate set <X, Z, F>)")

    # ... and stays free inside the bigger gate sets
    keys_T = {rot_key(Rg) for Rg in load_rotations("T")}
    for g in ("O", "I"):
        atlas = load_atlas(g)
        idx = [i for i, U in enumerate(atlas["unitaries"])
               if rot_key(rotation_from_unitary(U)) in keys_T]
        assert len(idx) == 24
        assert int(atlas["dij_magic_costs"][idx].max()) == 0
        assert int(atlas["dij_depths"][idx].max()) <= 2
        print(f"[ok] the 2T subgroup inside 2{g}: 24 elements, dij depth <= 2, magic 0"
              f" -- the twirl draw is Phi-free in the 2{g} gate set")

    # the coin, priced per axis by its cheapest coset representative
    claims = {"octahedron": (1, 0), "cube": (1, 0),
              "icosahedron": (2, 0), "dodecahedron": (2, 1)}
    print(f"\n  {'solid':14s} {'axes':>4s} {'max depth':>9s} {'max Phi':>7s}  coin representatives (dij)")
    print("  " + "-" * 86)
    for solid, (d_max, m_max) in claims.items():
        s = load_vertices(solid)
        g = COVARIANCE[solid]
        reps = coset_representatives(s, load_rotations(g),
                                     best_circuits(load_atlas(g)))
        magics = [r[0] for r in reps]
        depths = [r[1] for r in reps]
        seqs = [r[2] if r[2] else "I" for r in reps]
        assert max(depths) <= d_max and max(magics) <= m_max, (solid, reps)
        if solid == "dodecahedron":
            # the per-axis minima pinned exactly: four axes cost a Phi even in
            # their cheapest coset representative, six are free -- mean 0.4
            # per shot, the floor check_flip_completion attains
            assert sorted(magics) == [0] * 6 + [1] * 4, magics
        print(f"  {solid:14s} {len(reps):>4d} {max(depths):>9d} {max(magics):>7d}"
              f"  {', '.join(seqs)}")
    print("[ok] coin: octahedron/cube depth <= 1, icosahedron <= 2 (all 0 Phi);")
    print("     dodecahedron <= 2 with <= 1 Phi -- the coin is cheap everywhere. What")
    print("     costs is the DRAW, which must clear both bars (next check)")

    # the dodecahedron's bill: the full 2I draw -- and it is charged to
    # REALIZATION, not to the twirl
    atlas_I = load_atlas("I")
    magic = atlas_I["dij_magic_costs"]
    assert int(atlas_I["dij_depths"].max()) <= 4 and int(magic.max()) <= 1
    assert int(magic.sum()) == 96 and np.isclose(magic.mean(), 0.8)
    print("[ok] full 2I draw (the dodecahedron's REALIZATION bill): dij depth <= 4 AND")
    print(f"     <= 1 Phi simultaneously; mean {magic.mean():.1f} Phi ="
          f" {int(magic.sum())}/{len(magic)} elements")

    # the magic economics, per solid: price the CHEAPEST draw that clears
    # both bars. Where several subgroups tie at the bar (the icosahedron's
    # five T-conjugates, only one of which is the atlas's own 2T) the
    # protocol may pick any, so the bill is the minimum over them.
    print()
    for solid in ("octahedron", "cube", "icosahedron", "dodecahedron"):
        b = two_bars(solid)
        R = b["rotations"]
        circuits = best_circuits(load_atlas(b["group"]))
        both = b["draw_groups"]
        bill = min((max(circuits[rot_key(R[i])][0] for i in S),
                    max(circuits[rot_key(R[i])][1] for i in S)) for S in both)
        assert bill == ((1, 4) if solid == "dodecahedron" else (0, 2)), (solid, bill)
        print(f"  {solid:14s} minimal draw = order {b['draw']}"
              f" ({'/'.join(b['min_twirl' if b['binds'] != 'realize' else 'min_realize'])}),"
              f" {len(both)} candidate(s); cheapest costs"
              f" {bill[0]} Phi at depth <= {bill[1]}")
    print("[ok] every Phi in the randomized-projective ledger is charged to REALIZATION:")
    print("     the twirl bar is T for all four solids and 2T is all-Clifford, so a draw")
    print("     carries magic only where realization pushed it past T -- the dodecahedron")
    print("     alone. The golden gate is strictly necessary in exactly one case.")


# ---------------------------------------------------------------------------
# Section 5 (remark): the unitary-design ladder
# ---------------------------------------------------------------------------

def check_unitary_designs():
    catalan = {1: 1, 2: 2, 3: 5, 4: 14, 5: 42, 6: 132}
    strength = {"T": 2, "O": 3, "I": 5}
    for g, t_max in strength.items():
        U = load_atlas(g)["unitaries"]
        vals = {t: frame_potential(U, t) for t in range(1, t_max + 2)}
        for t in range(1, t_max + 1):
            assert abs(vals[t] - catalan[t]) < 1e-9, (g, t, vals[t])
        assert vals[t_max + 1] > catalan[t_max + 1] + 0.5, (g, vals)
        shown = "  ".join(f"F_{t} = {vals[t]:7.3f}" for t in sorted(vals))
        print(f"  2{g}: {shown}   -> exact {t_max}-design, not {t_max + 1}")
    print("  (Haar values are the Catalan numbers 1, 2, 5, 14, 42, 132.)")
    print("[ok] 2T/2O/2I are exact unitary 2-/3-/5-designs (2I overshoots: it meets t = 5")
    print("     exactly and first fails at t = 6, the degree of the icosahedral invariant);")
    print("     the twirl needs only a 2-design, and 2T is the minimal group one in d = 2")
