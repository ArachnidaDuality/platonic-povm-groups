"""
Export numeric data to .npz files in data/.

Produces:
  group_{2T,2O,2I}.npz  — SU(2) unitaries + gate sequences (4 synthesis strategies)
  group_{T,O,I}.npz     — SO(3) rotation matrices
  gates.npz             — gate matrices (SU(2) and U(2)), magic costs, gate set assignments
  povm_{solid}.npz      — Bloch sphere vertices, POVM elements, shadow reconstruction params,
                          num_vertices, symmetry_group
  everything.npz        — the above under prefixed keys (unitaries -> group_2T, vertices ->
                          povm_cube_vertices), with two mismatches: each solid's num_vertices
                          and symmetry_group are dropped, and the gate arrays keep the gate_
                          prefix that gates.npz strips

The numeric layer everything downstream reads: numpy_atlas.py diffs its independent
re-derivation against these files, randomized_core.py and shadow_experiments.py take
their canonical vertices and circuits from them.  So this is a contract, not a dump,
and most of the module is the checking of it -- every file goes out through _save
against a declared manifest, and the post-save checks run on the written bytes.
Two things bite: vertex row order is load-bearing, and the figure guard reaches
outside code/, so an edit to paper/figures/ can fail this script.
"""

import re
import numpy as np
import sympy as sp
from pathlib import Path

from povm_properties import numeric_vertices

from main import (
    geometric_group, generate_group_data, _phase_to_omega_power,
    _SU2_GATES, _U2_GATES, GATES,
    GATES_BFS_2T, GATES_BFS_2O, GATES_BFS_2I,
    GATES_DIJ_2T, GATES_DIJ_2O, GATES_DIJ_2I,
    GATES_U2_2T, GATES_U2_2O, GATES_U2_2I,
    GATES_DIJ_U2_2T, GATES_DIJ_U2_2O, GATES_DIJ_U2_2I,
)

FIGURES = Path(__file__).resolve().parent.parent / "paper" / "figures"


# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=np.complex128)
SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Symbolic -> numeric conversion
# ---------------------------------------------------------------------------
def quat_to_numpy(q):
    """Convert a symbolic Quaternion to a numpy complex128 2x2 unitary."""
    U = q.to_unitary()
    return np.array(U.evalf().tolist(), dtype=np.complex128)


def sympy_mat_to_numpy(M):
    """Convert a SymPy 2x2 Matrix to numpy complex128."""
    return np.array(M.evalf().tolist(), dtype=np.complex128)


def quat_to_rotation(q):
    """Convert a unit quaternion to its 3x3 SO(3) Bloch rotation matrix.

    q = w + xi + yj + zk  ->  R such that  v' = R v  is the physical Bloch
    action rho -> U_q rho U_q^dagger of the thesis unitary U_q (the units
    i, j, k act as pi rotations about z-hat, y-hat, x-hat), i.e. the textbook
    formula applied to (w, -z, -y, -x).
    Both q and -q map to the same R (double cover).
    """
    w = complex(q.w.evalf()).real
    x = -complex(q.z.evalf()).real
    y = -complex(q.y.evalf()).real
    z = -complex(q.x.evalf()).real
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def rotation_group_to_numpy(mode):
    """Return the polyhedral rotation group (SO(3)) as (N/2, 3, 3) float64.

    Picks one representative per antipodal pair {q, -q} and converts to
    a 3x3 rotation matrix.  T has 12, O has 24, I has 60 elements.
    """
    quats = sorted(geometric_group(mode))
    seen = set()
    representatives = []
    for q in quats:
        ph = q.proj_hash()
        if ph not in seen:
            seen.add(ph)
            representatives.append(q)
    return np.array([quat_to_rotation(q) for q in representatives])


# ---------------------------------------------------------------------------
# Platonic solid vertices on the Bloch sphere
#
# Single source of truth: the symbolic vertex orderings in povm_properties.py.
# Those same orderings generate the Appendix E atlas table and match the blade
# indices in Figure 2.2 (paper/figures/_povm_sphere_preamble.tex, the shared
# preamble the POVM sphere figures input). We evaluate them numerically here so
# the .npz exports, the atlas table, and the figure all coincide by
# construction rather than via a parallel re-implementation that can silently
# drift in orientation or ordering.
# ---------------------------------------------------------------------------
def _vertices(name):
    """Numeric (V, 3) float64 Bloch vectors for `name`, in atlas/figure order."""
    sym = numeric_vertices(name)
    return np.array([[float(sp.N(c)) for c in v] for v in sym], dtype=np.float64)


# ---------------------------------------------------------------------------
# POVM construction
# ---------------------------------------------------------------------------
def povm_elements(vertices):
    """Construct POVM elements E_k = (1/V)(I + n_k . sigma) from Bloch vectors."""
    V = len(vertices)
    elements = np.empty((V, 2, 2), dtype=np.complex128)
    for k, n in enumerate(vertices):
        elements[k] = (1 / V) * (I2 + n[0] * SIGMA_X + n[1] * SIGMA_Y + n[2] * SIGMA_Z)
    return elements


# ---------------------------------------------------------------------------
# Verification
#
# Every guard below measures its deviation and bounds it explicitly: np.allclose
# keeps rtol=1e-5 alongside any atol, and rtol scales with the expected operand,
# not with the residual.  Measured worst cases over everything exported:
# 2.3e-16 (unitarity), 8.9e-16 (orthogonality and det), 1.1e-16 (vertex norms),
# 2.2e-16 (completeness), 1.8e-15 (2-design, the dodecahedron).  The one
# allclose left standing is the zero-sum check, whose reference is 0.0 -- there
# rtol * 0 == 0 and the atol genuinely is the bound.
# ---------------------------------------------------------------------------
def verify_unitarity(matrices, label):
    worst = 0.0
    for i, U in enumerate(matrices):
        d_unit = np.abs(U.conj().T @ U - I2).max()
        assert d_unit < 1e-12, \
            f"{label}[{i}] not unitary (max |dev| = {d_unit:.2e})"
        worst = max(worst, d_unit)
    print(f"  {label}: {len(matrices)} unitary matrices OK "
          f"(max |dev| = {worst:.1e})")


def verify_rotations(matrices, label):
    I3 = np.eye(3)
    worst_orth = worst_det = 0.0
    for i, R in enumerate(matrices):
        d_orth = np.abs(R.T @ R - I3).max()
        assert d_orth < 1e-12, \
            f"{label}[{i}] not orthogonal (max |dev| = {d_orth:.2e})"
        d_det = abs(np.linalg.det(R) - 1.0)
        assert d_det < 1e-12, \
            f"{label}[{i}] det != 1 (|dev| = {d_det:.2e})"
        worst_orth = max(worst_orth, d_orth)
        worst_det = max(worst_det, d_det)
    print(f"  {label}: {len(matrices)} rotation matrices OK "
          f"(max |dev| = {worst_orth:.1e} orth, {worst_det:.1e} det)")


def verify_phases(d, gate_u2, label):
    """Replay every exported (word, phase) pair: the word in the standard gates equals omega^k U_q.

    Runs on the reloaded group file: all four sequence columns, each word
    multiplied out in numpy with the U(2) gate matrices gates.npz carries,
    against the file's own unitaries and its own phase column.  This is the
    warrant for exporting the phases at all, as code on the bytes.  It takes
    the standard gates from main.py (_U2_GATES, the same six matrices the
    thesis defines) and nothing else: not the per-gate phase factors
    _compute_u2_phase multiplies, not _phase_to_omega_power's conversion to
    k -- so a wrong sign or a wrong root of unity in either fails here, not
    in a reader's script.
    """
    omega = np.exp(1j * np.pi / 4)
    by_name = dict(zip(GATE_NAMES, gate_u2))
    U = d["unitaries"]
    worst, n = 0.0, 0
    for col in ("bfs", "dij", "u2", "dij_u2"):
        for i, (word, k) in enumerate(zip(d[f"{col}_sequences"], d[f"{col}_phases"])):
            prod = I2.copy()
            for tok in ([] if word == "I" else str(word).split()):
                G = by_name[tok.rstrip("†")]
                prod = prod @ (G.conj().T if tok.endswith("†") else G)
            dev = np.abs(prod - omega ** int(k) * U[i]).max()
            assert dev < 1e-12, \
                f"{label} row {i} {col}: the word {str(word)!r} read in the standard gates " \
                f"is not omega^{k} U_q (max |dev| = {dev:.2e})"
            worst = max(worst, dev)
            n += 1
    print(f"  {n} (word, phase) pairs replay to omega^k U_q in the standard gates "
          f"(max |dev| = {worst:.1e})")


def verify_povm_completeness(elements, label):
    total = elements.sum(axis=0)
    d_complete = np.abs(total - I2).max()
    assert d_complete < 1e-12, \
        f"{label} POVM not complete (max |dev| = {d_complete:.2e}): sum =\n{total}"
    print(f"  {label}: POVM completeness OK (max |dev| = {d_complete:.1e})")


def verify_vertices(vertices, label):
    d_norm = np.abs(np.linalg.norm(vertices, axis=1) - 1.0).max()
    assert d_norm < 1e-12, \
        f"{label} vertices not unit norm (max |dev| = {d_norm:.2e})"
    vertex_sum = vertices.sum(axis=0)
    assert np.allclose(vertex_sum, 0.0, atol=1e-12), \
        f"{label} vertices don't sum to zero: {vertex_sum}"
    print(f"  {label}: unit norm + zero sum OK (max |dev| = {d_norm:.1e})")


def verify_spherical_2design(vertices, label):
    V = len(vertices)
    cov = vertices.T @ vertices
    expected = (V / 3) * np.eye(3)
    d_design = np.abs(cov - expected).max()
    assert d_design < 1e-12, \
        f"{label} not a spherical 2-design (max |dev| = {d_design:.2e}):\n" \
        f"{cov}\nexpected:\n{expected}"
    print(f"  {label}: spherical 2-design OK (max |dev| = {d_design:.1e})")


def verify_canonical_dual(elements, a, b, label):
    """The exported shadow_a, shadow_b really do invert the measurement.

    They are the canonical dual's coefficients, D_k = a E_k + b I, and the
    single property that makes them a dual is that reconstruction is exact:
    sum_k Tr[E_k rho] D_k = rho for every rho.  No other module in this repo
    reads shadow_a/b back, so this is the only check they ever get.

    Linearity makes the test finite: it suffices on the Pauli basis, which is
    what the four-fold loop below covers.

    The deviation is measured and asserted rather than handed to np.allclose:
    measured on the icosahedron, a scaled by (1 + 1e-6) still passes allclose,
    at a max deviation of 3.0e-06.  The correct coefficients land at 4.4e-16
    (0.0 for the octahedron), so the bound below keeps ~2000x headroom while a
    mistyped coefficient anywhere near the right one now fails.  Rule: assert
    the margin, not the threshold.
    """
    basis = [I2, SIGMA_X, SIGMA_Y, SIGMA_Z]
    duals = a * elements + b * I2
    worst = 0.0
    for name, rho in zip(("I", "X", "Y", "Z"), basis):
        recon = sum(np.trace(E @ rho) * D for E, D in zip(elements, duals))
        dev = np.abs(recon - rho).max()
        assert dev < 1e-12, \
            f"{label}: canonical dual a={a}, b={b} fails to reconstruct " \
            f"sigma_{name} (max |dev| = {dev:.2e}):\n{recon}"
        worst = max(worst, dev)
    print(f"  {label}: canonical dual (a={a}, b={b}) reconstructs exactly "
          f"OK (max |dev| = {worst:.1e})")


def verify_alignment(vertices, elements, label):
    """Row k of `elements` is built from row k of `vertices`, on the written file.

    Every verify_* above runs on unwritten locals, where the two agree by
    construction because one was just built from the other.  Measured worst
    case over the five solids: exactly 0.0.
    """
    d_align = np.abs(elements - povm_elements(vertices)).max()
    assert d_align < 1e-12, \
        f"{label}: stored elements are not row-for-row aligned with the stored " \
        f"vertices (max |dev| = {d_align:.2e})"
    print(f"  {label}: elements aligned with vertices row-for-row "
          f"(max |dev| = {d_align:.1e})")


# ---------------------------------------------------------------------------
# The figures' hand-typed vertex tables
#
# Row order here is the published numbering -- Table E.1's rows, Figure 2.2's
# blade indices, Table D.4's permutation column -- and randomized_core's
# alignment() picks its winner by an argmax that TIES on the cube, icosahedron
# and dodecahedron, where row order alone breaks it.
# ---------------------------------------------------------------------------
FIGURE_SOLIDS = {"tet": "tetrahedron", "oct": "octahedron", "cube": "cube",
                 "ico": "icosahedron", "dod": "dodecahedron"}

# The geometry constants the \PVert rows may reach, and nothing else.  The
# figures' other 35 pgfmath macros are drawing machinery that will not evaluate
# here (\NdotV is written against #1/#2; \viewX's sin takes degrees, sympy's
# radians), so expansion is lazy and the reached set is pinned instead.
FIGURE_MACROS = {"PHI", "IPHI", "TET", "ICOS", "PHIICOS", "PHITET", "IPHITET"}


def _expand(expr, defs, reached):
    r"""Resolve \MACRO references in a pgfmath expression, transitively."""
    for _ in range(50):                    # the real chains are three deep
        m = re.search(r"\\([A-Za-z]+)", expr)
        if m is None:
            return expr
        name = m.group(1)
        assert name in defs, f"\\{name} is used but never \\pgfmathsetmacro'd"
        reached.add(name)
        expr = expr[:m.start()] + f"({defs[name]})" + expr[m.end():]
    raise AssertionError(f"macro expansion does not terminate: {expr}")


def _parse_figure(path):
    r"""Parse a figure's \PVert table into {solid: [v_1, ..., v_V]}, exactly.

    The counting asserts are load-bearing: a caller that zipped this against
    the code's vertex list would pass on a deleted last row.
    """
    text = path.read_text()
    defs = dict(re.findall(r"\\pgfmathsetmacro\{\\([A-Za-z]+)\}\{([^{}]*)\}", text))
    rows = re.findall(
        r"^\\PVert\{(\w+)\}\{(\d+)\}\{([^{}]*)\}\{([^{}]*)\}\{([^{}]*)\}", text, re.M)
    assert len(rows) == 50, f"{path.name}: {len(rows)} \\PVert rows, expected 50"

    reached, by_solid = set(), {}
    for tag, idx, *coords in rows:
        assert tag in FIGURE_SOLIDS, f"{path.name}: unknown solid tag '{tag}'"
        by_solid.setdefault(FIGURE_SOLIDS[tag], []).append(
            (int(idx), [sp.sympify(_expand(c, defs, reached)) for c in coords]))
    assert reached == FIGURE_MACROS, \
        f"{path.name}: rows reach {sorted(reached)}, expected {sorted(FIGURE_MACROS)}"

    assert set(by_solid) == set(PLATONIC_SOLIDS), \
        f"{path.name}: has rows for {sorted(by_solid)}"
    for name, rows_of in by_solid.items():
        V, got = PLATONIC_SOLIDS[name]["V"], sorted(i for i, _ in rows_of)
        assert len(rows_of) == V, \
            f"{path.name}: {name} has {len(rows_of)} rows, expected {V}"
        assert got == list(range(1, V + 1)), \
            f"{path.name}: {name} indices are {got}, not 1..{V}"
    return {n: [v for _, v in sorted(r)] for n, r in by_solid.items()}


def verify_figure_orders(handles):
    """All 300 figure coordinates, against the symbolic vertices and the npz.

    The symbolic leg is exact; the float leg runs on the reloaded npz and is
    the only cover on _vertices' float(sp.N(c)) cast.  Diffing the two figures
    against each other is what makes _povm_sphere_preamble.tex's claim to copy
    its vertex tables verbatim from platonic_solids.tex checkable.
    """
    a = _parse_figure(FIGURES / "_povm_sphere_preamble.tex")
    b = _parse_figure(FIGURES / "platonic_solids.tex")
    worst = 0.0
    for name in PLATONIC_SOLIDS:
        V = PLATONIC_SOLIDS[name]["V"]
        sym, disk = numeric_vertices(name), handles[f"povm_{name}"]["vertices"]
        assert len(sym) == V == len(disk), \
            f"{name}: {len(sym)} symbolic and {len(disk)} stored vertices, expected {V}"
        for k in range(V):
            for i in range(3):
                assert sp.simplify(a[name][k][i] - b[name][k][i]) == 0, \
                    f"{name}[{k + 1}] component {i}: platonic_solid_povms.tex has " \
                    f"{a[name][k][i]}, platonic_solids.tex has {b[name][k][i]}"
                assert sp.simplify(a[name][k][i] - sym[k][i]) == 0, \
                    f"{name}[{k + 1}] component {i}: the figures have " \
                    f"{a[name][k][i]}, povm_properties.py has {sym[k][i]}"
                dev = abs(float(a[name][k][i]) - float(disk[k][i]))
                assert dev < 1e-12, \
                    f"{name}[{k + 1}] component {i}: the figures have " \
                    f"{a[name][k][i]}, povm_{name}.npz has {disk[k][i]} " \
                    f"(|dev| = {dev:.2e})"
                worst = max(worst, dev)
    print(f"  2 x 50 \\PVert rows: 300 coordinates match the symbolic vertices "
          f"exactly, and the npz (max |dev| = {worst:.1e})")


# ---------------------------------------------------------------------------
# Synthesis data extraction
# ---------------------------------------------------------------------------
SYNTHESIS_RUNS = {
    "2T": (GATES_BFS_2T, GATES_DIJ_2T, GATES_U2_2T, GATES_DIJ_U2_2T),
    "2O": (GATES_BFS_2O, GATES_DIJ_2O, GATES_U2_2O, GATES_DIJ_U2_2O),
    "2I": (GATES_BFS_2I, GATES_DIJ_2I, GATES_U2_2I, GATES_DIJ_U2_2I),
}


def _seq_to_str(seq):
    """Join a gate sequence list into a single string, or 'I' for identity."""
    return " ".join(seq) if seq else "I"


def _phase_array(rows, key):
    """Extract phase omega-power from rows, defaulting to 0 for None."""
    return np.array([
        _phase_to_omega_power(r[key]) if r[key] is not None else 0
        for r in rows
    ], dtype=np.int64)


def extract_synthesis(mode):
    """Run all four synthesizers and return arrays aligned with element order.

    Returns dict with keys for all four strategies:
      SU(2) BFS:      bfs_sequences, bfs_depths, bfs_phases
      SU(2) Dijkstra: dij_sequences, dij_depths, dij_magic_costs, dij_phases
      PU(2) BFS:      u2_sequences, u2_depths, u2_phases
      PU(2) Dijkstra: dij_u2_sequences, dij_u2_depths, dij_u2_magic_costs, dij_u2_phases
    The *_phases are omega-powers: the word read in the standard gates equals
    omega^k times the row's unitary (main.py's _compute_u2_phase). The SU(2)
    words' phases are the ones atlas.tex prints, and every exported pair is
    replayed on the written file (verify_phases).
    plus _rows, the underlying generate_group_data rows, which the caller pops.
    """
    bfs_gates, dij_gates, u2_gates, dij_u2_gates = SYNTHESIS_RUNS[mode]
    rows = generate_group_data(mode, bfs_gates, dij_gates, u2_gates, dij_u2_gates)

    return {
        "bfs_sequences":       np.array([_seq_to_str(r["bfs_seq"]) for r in rows]),
        "bfs_depths":          np.array([r["bfs_depth"] for r in rows], dtype=np.int64),
        "bfs_phases":          _phase_array(rows, "bfs_phase"),
        "dij_sequences":       np.array([_seq_to_str(r["dij_seq"]) for r in rows]),
        "dij_depths":          np.array([r["dij_depth"] for r in rows], dtype=np.int64),
        "dij_magic_costs":     np.array([r["dij_magic"] for r in rows], dtype=np.int64),
        "dij_phases":          _phase_array(rows, "dij_phase"),
        "u2_sequences":        np.array([_seq_to_str(r["u2_seq"]) for r in rows]),
        "u2_depths":           np.array([r["u2_depth"] for r in rows], dtype=np.int64),
        "u2_phases":           _phase_array(rows, "u2_phase"),
        "dij_u2_sequences":    np.array([_seq_to_str(r["dij_u2_seq"]) for r in rows]),
        "dij_u2_depths":       np.array([r["dij_u2_depth"] for r in rows], dtype=np.int64),
        "dij_u2_magic_costs":  np.array([r["dij_u2_magic"] for r in rows], dtype=np.int64),
        "dij_u2_phases":       _phase_array(rows, "dij_u2_phase"),
        "_rows": rows,
    }


# ---------------------------------------------------------------------------
# Gate set export
# ---------------------------------------------------------------------------
# Canonical gate order and group assignments
GATE_NAMES = ["X", "Z", "H", "S", "F", "Φ"]
GATE_SETS = {
    "2T": ["X", "Z", "F"],
    "2O": ["X", "Z", "F", "H", "S"],
    "2I": ["X", "Z", "F", "Φ"],
}


def build_gate_arrays():
    """Build (su2, u2, magic_cost) arrays in GATE_NAMES order.

    The matching name array is built at the call site, not here.
    """
    su2 = np.array([sympy_mat_to_numpy(_SU2_GATES[g]) for g in GATE_NAMES])
    u2 = np.array([sympy_mat_to_numpy(_U2_GATES[g]) for g in GATE_NAMES])
    magic = np.array([GATES[g][2] for g in GATE_NAMES], dtype=np.int64)
    return su2, u2, magic


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
PLATONIC_SOLIDS = {
    "tetrahedron":  {"V": 4,  "group": "2T"},
    "octahedron":   {"V": 6,  "group": "2O"},
    "cube":         {"V": 8,  "group": "2O"},
    "icosahedron":  {"V": 12, "group": "2I"},
    "dodecahedron": {"V": 20, "group": "2I"},
}

# Group orders, pinned at the export site because nothing else pins them here.
# The run prints both counts and would otherwise check neither.  The rotation
# dedup keys on proj_hash -- an int, in a bare set, with no __eq__ fallback to
# resolve a collision the way a set of Quaternions has -- so a collision would
# silently DROP an element; and generate_group_data never counts its rows.
# main.py's verify_group asserts the binary orders, but on a separately built
# set and in a function this module does not call.
BINARY_ORDERS = {"2T": 24, "2O": 48, "2I": 120}
ROTATION_ORDERS = {"T": 12, "O": 24, "I": 60}

# What each .npz is contracted to hold.  Declared, not derived from the kwargs
# the call site passes -- a kwargs-derived manifest restates the call and can
# only ever pass.
SYNTHESIS_KEYS = (
    "bfs_sequences", "bfs_depths", "bfs_phases",
    "dij_sequences", "dij_depths", "dij_magic_costs", "dij_phases",
    "u2_sequences", "u2_depths", "u2_phases",
    "dij_u2_sequences", "dij_u2_depths", "dij_u2_magic_costs", "dij_u2_phases",
)
NPZ_MANIFESTS = {
    **{f"group_{m}": ("unitaries",) + SYNTHESIS_KEYS for m in BINARY_ORDERS},
    **{f"group_{g}": ("rotations",) for g in ROTATION_ORDERS},
    "gates": ("names", "su2", "u2", "magic_cost",
              "gateset_2T", "gateset_2O", "gateset_2I"),
    **{f"povm_{n}": ("vertices", "elements", "shadow_a", "shadow_b",
                     "num_vertices", "symmetry_group") for n in PLATONIC_SOLIDS},
}


def _save(path, manifest, **arrays):
    """Write `arrays` to `path`, reload it, and hand the reloaded file back.

    allow_pickle=False is the gate: savez_compressed takes an object array
    without a word -- a stray dict, Path or unevaluated sympy expression goes
    straight in -- and the reload turns that into a failure here instead of two
    scripts downstream.  Handing the file back is what lets verify_alignment
    and verify_figure_orders run on the bytes rather than on unwritten locals.
    """
    np.savez_compressed(path, **arrays)
    loaded = np.load(path, allow_pickle=False)
    on_disk = {k: loaded[k] for k in loaded.files}   # an object array raises here
    extra, missing = set(on_disk) - set(manifest), set(manifest) - set(on_disk)
    assert not extra and not missing, \
        f"{path.name}: written keys do not match the declared manifest " \
        f"(undeclared: {sorted(extra)}; missing: {sorted(missing)})"
    for k, v in arrays.items():
        want, got = np.asarray(v), on_disk[k]        # scalars store as 0-d arrays
        assert (want.shape, want.dtype) == (got.shape, got.dtype), \
            f"{path.name}[{k}]: wrote {want.shape} {want.dtype}, " \
            f"read back {got.shape} {got.dtype}"
        assert np.array_equal(want, got), \
            f"{path.name}[{k}]: values changed across the round trip"
    return loaded


def everything_key(stem, key):
    """Where per-file key `key` of `stem`.npz lands in everything.npz -- the
    module docstring's prefix map, written out so it can be checked.  None means
    the docstring declares the key dropped."""
    if stem.startswith("group_"):
        return stem if key in ("unitaries", "rotations") else f"{stem}_{key}"
    if stem == "gates":                    # mismatch 2: the kept gate_ prefix
        return key if key.startswith("gateset_") else f"gate_{key}"
    if key in ("num_vertices", "symmetry_group"):
        return None                        # mismatch 1: dropped from the combined file
    return f"{stem}_{key}"


def everything_manifest(handles):
    """everything.npz's declared keys, as {combined key: (stem, per-file key)}.

    Handing this to _save as its manifest is what holds the prefix map in both
    directions, and the collision assert is the only thing that catches a
    `combined` key silently overwritten by a later loop.
    """
    manifest = {}
    for stem, d in handles.items():
        for k in d.files:
            ek = everything_key(stem, k)
            if ek is None:
                continue
            assert ek not in manifest, \
                f"everything.npz: {ek} is claimed by {manifest[ek]} " \
                f"and by ({stem}, {k})"
            manifest[ek] = (stem, k)
    return manifest


def verify_everything_map(manifest, handles, combined_npz):
    """Every everything.npz key carries its per-file source's values."""
    for ek, (stem, k) in manifest.items():
        assert np.array_equal(combined_npz[ek], handles[stem][k]), \
            f"everything.npz[{ek}] differs from {stem}.npz[{k}]"
    print(f"  everything.npz: {len(manifest)} keys, prefix map exhaustive both "
          f"ways, values match their per-file sources")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)
    combined = {}  # collect everything for the single-file export
    handles = {}   # the reloaded file of each per-file export, keyed by stem

    # --- Binary polyhedral groups (SU(2)) with synthesis ---
    print("Binary polyhedral groups (SU(2)):")
    su2, u2, magic = build_gate_arrays()   # exported below; the U(2) matrices replay the words here
    for mode in ["2T", "2O", "2I"]:
        print(f"  Synthesizing {mode}...")
        synth = extract_synthesis(mode)
        rows = synth.pop("_rows")
        matrices = np.array([quat_to_numpy(r["quat"]) for r in rows])
        assert len(matrices) == BINARY_ORDERS[mode], \
            f"{mode}: got {len(matrices)} elements, expected {BINARY_ORDERS[mode]}"
        verify_unitarity(matrices, mode)
        handles[f"group_{mode}"] = _save(
            out_dir / f"group_{mode}.npz", NPZ_MANIFESTS[f"group_{mode}"],
            unitaries=matrices,
            **synth,
        )
        verify_phases(handles[f"group_{mode}"], u2, mode)
        combined[f"group_{mode}"] = matrices
        for k, v in synth.items():
            combined[f"group_{mode}_{k}"] = v
        print(f"  -> group_{mode}.npz ({len(matrices)} elements, with gate sequences)")

    # --- Polyhedral rotation groups (SO(3)) ---
    ROTATION_GROUPS = {"T": "2T", "O": "2O", "I": "2I"}
    print("\nPolyhedral rotation groups (SO(3)):")
    for name, binary in ROTATION_GROUPS.items():
        rotations = rotation_group_to_numpy(binary)
        assert len(rotations) == ROTATION_ORDERS[name], \
            f"{name}: got {len(rotations)} rotations, expected {ROTATION_ORDERS[name]}"
        verify_rotations(rotations, name)
        handles[f"group_{name}"] = _save(
            out_dir / f"group_{name}.npz", NPZ_MANIFESTS[f"group_{name}"],
            rotations=rotations)
        combined[f"group_{name}"] = rotations
        print(f"  -> group_{name}.npz ({len(rotations)} elements)")

    # --- Gate sets ---
    print("\nGate sets:")
    for i, g in enumerate(GATE_NAMES):
        verify_unitarity(su2[i:i+1], f"gate {g} (SU2)")
    stored_names = np.array([g.replace("Φ", "Phi") for g in GATE_NAMES])
    gate_data = dict(
        gate_names=stored_names,
        gate_su2=su2,
        gate_u2=u2,
        gate_magic_cost=magic,
        gateset_2T=np.array(GATE_SETS["2T"]),
        gateset_2O=np.array(GATE_SETS["2O"]),
        gateset_2I=np.array([g.replace("Φ", "Phi") for g in GATE_SETS["2I"]]),
    )
    # Only 2I's set needs the Phi replacement, which is safe only because no
    # other set contains Phi.  gate_names is spelled the same way; nothing
    # reads gateset_* back, so this file is the only place that can say so.
    known = set(stored_names.tolist())
    for mode in GATE_SETS:
        strays = [g for g in gate_data[f"gateset_{mode}"].tolist() if g not in known]
        assert not strays, \
            f"gateset_{mode}: {strays} not among gate_names {sorted(known)}"

    handles["gates"] = _save(out_dir / "gates.npz", NPZ_MANIFESTS["gates"], **{
        k.removeprefix("gate_"): v for k, v in gate_data.items()
    })
    combined.update(gate_data)
    print(f"  -> gates.npz ({len(GATE_NAMES)} gates)")
    for mode, gates in GATE_SETS.items():
        print(f"     {mode}: {gates}")

    # --- Platonic solid POVMs ---
    print("\nPlatonic solid POVMs:")
    for name, info in PLATONIC_SOLIDS.items():
        vertices = _vertices(name)
        assert len(vertices) == info["V"], \
            f"{name}: got {len(vertices)} vertices, expected {info['V']}"
        elements = povm_elements(vertices)
        a = 3 * info["V"] / 2
        b = -1.0

        verify_vertices(vertices, name)
        verify_spherical_2design(vertices, name)
        verify_povm_completeness(elements, name)
        verify_canonical_dual(elements, a, b, name)

        d = _save(
            out_dir / f"povm_{name}.npz", NPZ_MANIFESTS[f"povm_{name}"],
            vertices=vertices,
            elements=elements,
            shadow_a=a,
            shadow_b=b,
            num_vertices=info["V"],
            symmetry_group=info["group"],
        )
        handles[f"povm_{name}"] = d
        verify_alignment(d["vertices"], d["elements"], name)
        combined[f"povm_{name}_vertices"] = vertices
        combined[f"povm_{name}_elements"] = elements
        combined[f"povm_{name}_shadow_a"] = a
        combined[f"povm_{name}_shadow_b"] = b
        print(f"  -> povm_{name}.npz (V={info['V']}, group={info['group']}, a={a}, b={b})")

    # --- The figures' hand-typed vertex tables ---
    print("\nFigure vertex tables (paper/figures/):")
    verify_figure_orders(handles)

    # --- Combined single file ---
    ev_manifest = everything_manifest(handles)
    ev = _save(out_dir / "everything.npz", ev_manifest, **combined)
    print("\n  -> everything.npz (the combined file)")
    verify_everything_map(ev_manifest, handles, ev)

    print("\nDone.")


if __name__ == "__main__":
    main()
