"""Builds code/randomization_walkthrough.ipynb from the randomized_* suite.

Run with `cd code && uv run python _build_randomization_walkthrough.py`.

This script is the source of truth for the notebook. To edit a markdown cell,
modify the corresponding `md(r'''...''')` call below, then re-run this script
to regenerate the .ipynb. Do not edit the notebook directly -- regeneration
will overwrite manual edits.

The notebook is the walkthrough for `randomized_implementations.py` and the
seven sibling modules it binds: the standalone, full-length treatment of
R1 (randomized-projective) vs R2 (twirled-native) -- one reframing, "a
protocol is which maps you average", applied five times, with zero RNG
anywhere (that is the stated pedagogy).

Code cells are not hand-copied: `lift()`, `lift_assign()` and `lift_body()`
extract function sources, module-level assignments (with their attached
comment blocks) and dedented function bodies via `ast` at build time, so the
cells are verbatim by construction and regeneration tracks module edits
automatically. The lift helpers search ALL EIGHT `randomized_*` files -- a
name is found wherever it lives, a missing name dies naming the files
searched, an ambiguous one dies naming every file that defines it.

Before writing the .ipynb the builder statically proves the notebook is
closed, three ways: every name a code cell needs must resolve to a builtin,
an import, or a definition made by that or an earlier cell (lifted or
glue); a name bound in MORE than one cell never crosses a cell boundary --
whatever a cell reuses, it rebinds, so a deleted glue line dies at build
time instead of executing against a stale value leaked from an earlier cell
-- and at cell top level nothing is used before its own binding (function
bodies run at call time and are exempt). A violation fails the build
loudly, naming the cell and the names; _closure_selfcheck() keeps all three
failure classes firing.

The notebook executes with cwd `code/` and writes nothing -- importing the
suite (or calling `main()`) writes nothing; the six .tex fragments are
emitted only under the entry point's `__main__` guard, which the notebook
never takes. The claim is enforced at runtime too: the first code cell
arms an audit hook refusing any write-mode open under the repository
(__pycache__ excepted), so a writer reached through ANY idiom or call
dies mid-execute -- the builder's _write_guard is only a build-time
lint for the obvious patterns. The receipts cell likewise re-hashes the
eight modules plus this builder against build-time sha256 pins, so a
committed notebook that has drifted behind a later edit fails its
re-execute naming the remedy, instead of publishing stale sources. The one
adaptation: `randomized_core` resolves `DATA` next to itself via
`__file__`, which a notebook lacks, so a glue cell redefines the same
trailing-slash string relative to the cwd (asserted against the module's
own DATA at build time).

Regenerating is only half the job: the builder emits empty outputs, so the
re-execute below is mandatory to keep the committed outputs current. Both
commands run from code/, and the notebook's own closing notes give the same
pair:

    uv run --with jupyter --with nbconvert jupyter nbconvert \
        --to notebook --execute --inplace randomization_walkthrough.ipynb
"""

import ast
import builtins
import hashlib
import symtable
import textwrap
from pathlib import Path

import nbformat

HERE = Path(__file__).parent

MODULES = (
    "randomized_implementations",
    "randomized_core",
    "randomized_field",
    "randomized_scalars",
    "randomized_twojobs",
    "randomized_obstruction",
    "randomized_decker",
    "randomized_fragments",
)

SRC = {m: (HERE / f"{m}.py").read_text(encoding="utf-8") for m in MODULES}
TREE = {m: ast.parse(SRC[m]) for m in MODULES}
LINES = {m: SRC[m].splitlines() for m in MODULES}

# Build-time currency pins (consumed by the receipts cell): sha256 of the
# files the notebook is generated from, this builder included. The committed
# notebook re-hashes them on every execute, so a notebook that has drifted
# behind a later edit dies at re-execute naming the remedy -- textual
# identity is the anti-drift guarantee; the receipts cell's behavioral
# spot-checks are a demonstration on top of it.
PINNED = {m: hashlib.sha256((HERE / f"{m}.py").read_bytes()).hexdigest()
          for m in MODULES}
PINNED["_build_randomization_walkthrough"] = hashlib.sha256(
    (HERE / "_build_randomization_walkthrough.py").read_bytes()).hexdigest()


def _assign_targets(node):
    """Top-level names bound by an ast.Assign or AnnAssign (plain and tuple
    targets; an annotated assignment binds one plain name)."""
    if isinstance(node, ast.AnnAssign):
        return [node.target.id] if isinstance(node.target, ast.Name) else []
    out = []
    for t in node.targets:
        if isinstance(t, ast.Name):
            out.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            out += [e.id for e in t.elts if isinstance(e, ast.Name)]
    return out


def _find(name, kind):
    """(module, node) of the unique top-level def/assign `name`; loud errors."""
    hits = []
    for m in MODULES:
        for node in TREE[m].body:
            if kind == "func" and isinstance(node, ast.FunctionDef) \
                    and node.name == name:
                hits.append((m, node))
            elif kind == "assign" \
                    and isinstance(node, (ast.Assign, ast.AnnAssign)) \
                    and getattr(node, "value", None) is not None \
                    and name in _assign_targets(node):
                hits.append((m, node))
    if not hits:
        raise KeyError(f"no top-level {kind} {name!r} in any of: "
                       + ", ".join(f"{m}.py" for m in MODULES))
    if len(hits) > 1:
        raise KeyError(f"{kind} {name!r} is ambiguous -- defined in: "
                       + ", ".join(f"{m}.py" for m, _ in hits))
    return hits[0]


def _with_leading_comments(m, start):
    """Walk `start` (0-based) up over the contiguous comment block above it."""
    while start - 1 >= 0 and LINES[m][start - 1].lstrip().startswith("#"):
        start -= 1
    return start


def lift(*names, comments=False):
    """Verbatim source of module-level function defs, in the given order."""
    chunks = []
    for name in names:
        m, node = _find(name, "func")
        start = (node.decorator_list[0].lineno if node.decorator_list
                 else node.lineno) - 1
        if comments:
            start = _with_leading_comments(m, start)
        chunks.append("\n".join(LINES[m][start:node.end_lineno]))
    return "\n\n\n".join(chunks)


def lift_assign(*names):
    """Verbatim module-level assignments (with their leading comment blocks),
    in the given order."""
    chunks = []
    for name in names:
        m, node = _find(name, "assign")
        start = _with_leading_comments(m, node.lineno - 1)
        chunks.append("\n".join(LINES[m][start:node.end_lineno]))
    return "\n\n".join(chunks)


def lift_body(name):
    """A function's body, docstring dropped, dedented to top level.

    Used to run a check driver's interior as narrative notebook code. Only
    valid for drivers with no `return` (the closure check parses the result,
    so a violation dies at build time, not at execute time).
    """
    m, node = _find(name, "func")
    assert node.body[0].lineno > node.lineno, f"{name}: one-line def?"
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
            and isinstance(first.value.value, str):
        start = first.end_lineno            # drop the docstring, keep the rest
    else:
        # the first body statement's own line, not node.lineno + 1 -- a
        # wrapped signature would otherwise leak into the body -- plus any
        # comment block directly above it (the climb stops at the signature,
        # which is not a comment line)
        start = _with_leading_comments(m, first.lineno - 1)
    body = "\n".join(LINES[m][start:node.end_lineno])
    return textwrap.dedent(body).strip("\n")


nb = nbformat.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbformat.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbformat.v4.new_code_cell(text.strip("\n")))


# =============================================================================
# Title + intro
# =============================================================================
md(r'''
# Randomized Implementations of the Platonic-Solid POVMs — Walkthrough

A companion walkthrough for `code/randomized_implementations.py` and the seven sibling modules it
binds (`randomized_{core,field,scalars,twojobs,obstruction,decker,fragments}.py`) — the
verification suite behind Section 5.2.3, Appendix D, and Appendix F.3 of the thesis.

The thesis distinguishes two protocols that both carried the name *randomized implementation*:

- **R1 — randomized-projective.** Draw a rotation $g$ uniformly from a finite group $G$, apply
  $U_g$, apply one fixed *alignment* $A$ (vertex axis $\to \hat z$), read out in the $Z$ basis.
  No ancillas; needs antipodal vertex pairs, so the tetrahedral SIC is excluded;
  estimator-channel factor $T_{zz}$. (The literature's "randomized measurements" primitive.)
- **R2 — twirled-native.** Draw $g$, apply $U_g$, then run the *fixed* native
  (Decker/Naimark) circuit and relabel outcomes by $g$ in classical post-processing. Ancillas;
  works for all five solids, SIC included; estimator-channel factor $\operatorname{tr}T/3$.
  (The literature's measurement/readout twirling.)

**One reframing carries the whole suite: a protocol is *which maps you average*.** It is applied
five times below.

1. **Two estimator channels (§1).** R1's draw sits in the outcome probability *and* in the snapshot, so the
   average twirls the composition readout$\,\circ\,$noise — the rank-one $\hat z\hat z^\top T$
   conjugated into the solid's frame. R2's readout never depends on the draw, so the average
   twirls the noise *alone*. One lemma, two conjugated objects, two scalars — the
   estimator-channel factor identifies the protocol.
2. **Two jobs (§2).** The drawn set must *be* the POVM (its orbit sweeps the vertex axes
   uniformly: realization) and must *average to a scalar* (irreducibility: the twirl). Two
   independent properties of the same set of maps, and they come apart — the bars cross at the
   icosahedron.
3. **Exactness (§3).** Every map has to be compiled from the gate set, so it carries the gate
   field with it. Direction (vertices outside the gates' real field) and weight (effect traces
   outside the dyadic ring) between them convict all five solids; what survives for the
   octahedron is a classical coin, not a circuit.
4. **A wrong list (§3 cont.).** Believe one vertex list while the device measures another, and
   the same average launders the entire mismatch into a single overlap $\kappa$ — zero offset,
   never a bias, a $1/\kappa^2$ shot premium. Decker's outcome order, priced per solid.
5. **Gate noise (§1c).** Put the noise *inside* the drawn word and there is no longer one fixed
   map to average — the noise is correlated with $g$ and Schur is silent. What decides the
   residual's order in $\gamma$ is the word set's prefix multiset: the reframing earns its keep
   by failing informatively.

**Zero RNG anywhere — and that is the pedagogy.** Nothing here samples. Every "randomized"
quantity is a finite group average computed as an explicit sum: float64 linear algebra asserted
at $10^{-9}$ (the algebraic numbers involved are well separated), SymPy exactly where field
membership, or a claim quantified over *arbitrary* noise, gate noise or state, *is* the claim.
Where a verdict is an identity (which
element a product is, whether two vertices are antipodes) it is decided a second time by
canonical form over a small number field, and the two answers are required to agree.

**References:**
- Decker, Janzing & Beth, *Quantum circuits for single-qubit measurements corresponding to platonic solids* (2004); Decker, *Implementation of group-covariant POVMs* (2005) — the native circuits, rebuilt in §3 cont.
- Elben, Flammia, Huang, Kueng, Preskill, Vermersch & Zoller, *The randomized measurement toolbox* (2022) — randomized-projective's literature home
- Chen, Yu, Zeng & Flammia, *Robust shadow estimation* (2021) — calibration under a measurement twirl, twirled-native's literature home
- Nguyen, Bönsel, Steinberg & Gühne, *Optimising shadow tomography with generalised measurements* (2022) — Platonic-solid POVM shadows (the closest adjacent work)
- Gross, Audenaert & Eisert, *Evenly distributed unitaries* (2007) — $2T$ as the minimal group 2-design in $d=2$
- Bannai, Navarro, Rizo & Tiep, *Unitary t-groups* (2018); Roy & Scott, *Unitary designs and codes* (2009) — the design ladder
- Hirao, Nozaki & Tasaka (2025) — the spherical-design side of the same group orbits
- Conway & Smith, *On Quaternions and Octonions* (2003) — the $\tau$/$\sigma$ convention
''')


# =============================================================================
# The map + the contract
# =============================================================================
md(r'''
## The map, and the contract

`randomized_implementations.py` is the entry point: its `main()` *is* the run order, and the
banners it prints are the suite's Sections 0–5. The bindings live in seven flat siblings, a DAG
with tools below checks:

| notebook | suite section | finding | modules doing the work |
|---|---|---|---|
| §0 canonical data | 0 | the npz pins | `core`, check in `scalars` |
| §1 two protocols, two estimator channels | 1 | 1 + 2 | `core` (channels), `field` (exact kit), `scalars` |
| §1b calibration mismatch | 1 | 1 | `scalars` (+ a lazy, write-free `shadow_experiments` import) |
| §1c gate noise | 1 | 6 | `scalars` |
| §1d the alignment | 1 | — | `scalars` |
| §2 the two jobs | 2 | 4 | `twojobs` (+ sweep machinery in `core`) |
| §3 the exactness obstruction | 3 | 3 | `obstruction` (+ symbolic layer in `core`, theorem in `field`) |
| §3 cont. Decker's circuits | 3 cont. | 5 | `decker` |
| §4 the ledger | 4 | — | `fragments` |
| §5 the design ladder | 5 | remark | `twojobs` |
| §6 the receipts | — | — | the entry point's `main()` |

**The contract of this notebook.**

- Every module definition shown below is lifted *verbatim, mechanically* (`ast`, at build time)
  from whichever of the eight files owns it — nothing is hand-copied, so the cells track module
  edits by construction. Hand-written code is only glue: imports, one `DATA` adaptation, small
  demos and comparisons.
- One adaptation: the module resolves its data directory next to itself via `__file__`, which a
  notebook lacks. The setup cell redefines the same trailing-slash string relative to the cwd
  (`code/`). Everything else is untouched.
- The notebook **writes nothing** — and enforces it. The first code cell arms a runtime
  tripwire (an audit hook) refusing any write-mode open under the repository, so the claim
  holds for every idiom and every call into the imported suite, not just the patterns a build
  lint can see. Importing any module of the suite — or calling `main()` — writes nothing; the
  six thesis fragments are emitted only when the entry point runs as a script.
- The cells below select for depth on the through-line rather than mirroring every check; the
  final cell closes the gap four ways: it re-hashes the modules (and the builder) against
  build-time sha256 pins, so a committed notebook that has drifted behind a later edit cannot
  execute quietly; it spot-checks lifted primitives against the imported production modules;
  it re-derives the ledger's `spec_sheet()`; and it runs the suite's `main()` end to end, so
  every check not staged here still runs and still passes, in this very notebook. Only the
  fragment writers themselves stay unexecuted.
''')


# =============================================================================
# Setup
# =============================================================================
code(r'''
# === Setup (notebook glue): imports + the one adaptation ===

import itertools
import math
from pathlib import Path

import numpy as np
import sympy as sp
from sympy import I as sI
from sympy import Matrix, Rational, sqrt

# The one adaptation: randomized_core resolves DATA next to itself via
# __file__, which a notebook lacks. Same trailing-slash string semantics,
# with the notebook executing from code/.
DATA = f"{Path('data')}/"

# The exact-rotation layer rebuilds T/O/I from main.py's canonical
# quaternions; importing main is write-free (its writers sit under __main__).
from main import geometric_group
''')

code(r'''
# === The write tripwire (glue): "writes nothing", enforced at runtime ===

# Every Python-level file write funnels through the `open` audit event, so
# one hook turns the header's claim into a runtime guarantee covering every
# idiom (open, np.savez, Path.write_text, savefig, ...) and every call into
# the imported suite -- not just patterns a static lint can see. Scope:
# refuse write-mode opens under the repository; __pycache__ is exempt (a
# fresh clone's first import compiles bytecode); reads stay free.
import os
import sys

_REPO = Path.cwd().resolve().parent        # the notebook executes in code/


def _no_tree_writes(event, args):
    if event != "open":
        return
    path, mode, flags = args
    if isinstance(mode, str):
        writey = bool(set(mode) & set("wxa+"))
    else:                                  # os.open passes mode=None
        writey = bool(flags & (os.O_WRONLY | os.O_RDWR))
    if not writey or path is None:
        return
    try:
        p = Path(os.fsdecode(path))
    except TypeError:                      # e.g. an integer fd: not a path
        return
    p = p if p.is_absolute() else Path.cwd() / p
    if "__pycache__" not in p.parts and p.resolve().is_relative_to(_REPO):
        raise RuntimeError(f"write tripwire: refusing write-mode open of {p}")


sys.addaudithook(_no_tree_writes)

# ...and the proof it is armed, in this committed output: a write-mode open
# under code/ must die, refused before the file is created.
try:
    open("data/_tripwire_probe", "w").close()
except RuntimeError:
    print("write tripwire armed: a write-mode open under code/ was refused")
else:
    raise AssertionError("tripwire failed to arm")
assert not Path("data/_tripwire_probe").exists()
''')


# =============================================================================
# Section 0 -- canonical data
# =============================================================================
md(r'''
## 0. The canonical data, and the generic probe

Inputs are the thesis's own symbolic exports in `code/data/`: `povm_*.npz` (Bloch vertices and
effects, in the published numbering of the POVM vertex table), `group_{T,O,I}.npz` (the rotation
groups) and `group_2{T,O,I}.npz` (the binary groups with their synthesized circuits — the rows of
Appendix A, i.e. the *atlas words* the coin and the draw will be priced in).

One fixed measurement-side noise $r \mapsto Tr + t$ serves as the probe for every float check.
It is chosen *generic* — the two candidate scalars distinct, offset nonzero, $T$ anisotropic —
so no depolarizing verdict below can pass by accident; the next cell asserts exactly that
(the same three asserts `main()` opens with). Section 1a will then remove the probe from the
load-bearing path altogether by quantifying over every $(T, t)$.
''')

code(
    "# === Canonical constants and loaders (lifted from randomized_core.py) ===\n\n"
    + lift_assign("SOLIDS", "TAU_SYM", "SIG_SYM", "COVARIANCE",
                  "T_NOISE", "t_NOISE", "PAULI")
    + "\n\n\n"
    + lift("load_vertices", "load_elements", "load_rotations", "load_atlas",
           "rotation_from_unitary", "rot_key")
)

code(r'''
# === The cast, and the probe's genericity (glue) ===

print(f"{'solid':14s} {'V':>3s} {'grp':>4s} {'|G|':>4s} {'|2G|':>5s}")
for solid in SOLIDS:
    s, g = load_vertices(solid), COVARIANCE[solid]
    print(f"{solid:14s} {len(s):3d} {g:>4s} {len(load_rotations(g)):4d}"
          f" {len(load_atlas(g)['unitaries']):5d}")

print(f"\nprobe noise r -> T r + t:  T_zz = {T_NOISE[2, 2]:.6f}"
      f"   tr(T)/3 = {np.trace(T_NOISE) / 3:.6f}")

# main()'s own preamble asserts, restated: the probe is generic
assert abs(T_NOISE[2, 2] - np.trace(T_NOISE) / 3) > 0.05   # scalars distinct
assert np.linalg.norm(t_NOISE) > 0.05                      # offset nonzero
assert np.linalg.norm(T_NOISE - np.trace(T_NOISE) / 3 * np.eye(3)) > 0.05
print("generic: the candidate scalars differ, the offset is nonzero, T is")
print("anisotropic -- no depolarizing verdict below can pass by accident")
''')

md(r'''
### 0a. The symbolic twins

Alongside the float layer the suite carries exact twins: the five vertex sets as SymPy vectors,
the four gates whose rotation axes the solids will turn out to inherit ($X$, $Z$, the face gate
$F = HS^\dagger$, the golden gate $\Phi$), and exact Bloch axes/actions. Two subtleties both
live here:

- **Two numberings.** `symbolic_solids()` and the published npz order agree as *sets* (same
  solids, same pose) but number the tetrahedron, cube and icosahedron differently.
  `atlas_vertices()` permutes the symbolic vertices into the published order — the one a reader
  can look up — and anything the thesis prints *with a vertex index* runs on it.
- **Identity questions decided twice.** Wherever a verdict is an identity, a canonical-form
  companion re-decides it and the two must agree; the float pipeline stays in place (each exact
  companion pairs with a value-for-value numeric agreement check, because a boolean exact test
  does not test its own transcription).

`check_canonical_data` then pins the whole data layer: npz vertices $=$ exact solids (with
margin), effects $= \frac1V(\mathrm{Id} + \hat n_k\cdot\vec\sigma)$, gates $=$ `gates.npz`
projectively, the $\pm U$ pairs of each binary group projecting onto exactly the rotation group,
and $T < O$, $T < I$.
''')

code(
    "# === The symbolic layer (lifted from randomized_core.py) ===\n\n"
    + lift_assign("PAULI_SYM")
    + "\n\n\n"
    + lift("symbolic_solids", "atlas_gates", "state_from_bloch", "on_solid",
           "bloch_axis", "bloch_matrix", "atlas_vertices")
)

code(
    "# === Section 0: pin the claims to the canonical data "
    "(check_canonical_data) ===\n\n"
    + lift_body("check_canonical_data")
)


# =============================================================================
# Section 1 -- two protocols, two estimator channels
# =============================================================================
md(r'''
## 1. Two protocols, two estimator channels — the reframing, first application

A protocol's *estimator channel* is the affine map $r \mapsto Mr + \mathrm{off}$ from the
state's Bloch vector to the mean reconstructed snapshot ($3\times$ the mean sampled vertex —
the canonical dual's factor). Both protocols below compute it as an **exact group average** —
a finite sum over draws and outcomes, no sampling — under the shared probe noise.

- In R1 the drawn $g$ sits in the outcome probability *and* in the snapshot $3b\,R_g^\top v_0$,
  so the average twirls the composition readout$\,\circ\,$noise;
- in R2 the readout ranges over all $V$ effects whatever the draw, so summing outcomes first
  contracts the vertex covariance $\sum_k \hat n_k\hat n_k^\top = \frac V3\,\mathrm{Id}$ and the
  average twirls the noise alone.

Same lemma, different object handed to it — hence different scalars, printed below. R1 needs
antipodal vertex pairs (its coin measures $\{v, -v\}$ bases), so the tetrahedron is out; R2
keeps it — *the SIC is not the price of the twirl; the ancilla is* (finding 2). And sharpness:
with **no** draw the channel is the noise map itself, offset and all — what the twirl removes is
really there.

**Notation, pinned before any number appears.** Throughout the suite $\kappa$ is the *estimator
channel's* multiplier, ideal $1$ — equivalently the overlap of the believed measurement with the
performed one, so any fixed misspecification costs a $1/\kappa^2$ shot premium. The thesis's
Appendix F.3.1 and `shadow_experiments.py` write $\eta$ for the *calibration scalar*, the
shrinkage of the twirled measurement channel **before** the canonical dual's factor $3$. The
dictionary is $\kappa = 3\eta$: ideal $\kappa = 1$ is noiseless $\eta = 1/3$. Premia are ratios
and read the same in either convention; a bare $1/\kappa^2$ carried into $\eta$'s units is
$1/(9\eta^2)$.
''')

code(
    "# === The two protocols as exact estimator channels "
    "(lifted from randomized_core.py) ===\n\n"
    + lift("is_decomposable", "alignment", "channel_R1", "channel_R2",
           "orbit_counts")
)

code(
    "# === Section 1: findings 1 + 2 at the probe (check_two_protocols) ===\n\n"
    + lift_body("check_two_protocols")
)

md(r'''
### The mechanism: who decides the readout axis

The two protocols differ in exactly one place. In R1 the same draw that rotates the state also
decides — through the fixed $\hat z$ readout — which lab direction of the noise gets probed:
draw and readout are perfectly correlated, the correlation sits *inside* the average, and what
survives is the noise seen along the readout axis, $T_{zz}$. In R2 the readout is the same fixed
POVM every shot; there is nothing for the draw to correlate with, the noise is seen whole, and
only its isotropic mean survives, $\operatorname{tr}T/3$.

The module's own reduction says this in one line: R1's channel depends on the noise only through
the rank-one $v_0 w^\top$ with $w = A^\top T^\top \hat z$ — and since $A$ is a rotation,
$v_0 \cdot w = \hat z^\top T^\top \hat z = T_{zz}$, the alignment cancelling between snapshot
and readout. The demo verifies the reduction against `channel_R1` and runs R2's contraction
explicitly.

The moral — the thesis's own — is **calibrate the protocol you run**: fit a
randomized-projective experiment with $\operatorname{tr}T/3$, or a twirled-native one with
$T_{zz}$, and the miscalibration is exactly the anisotropy of the noise — the gap between the
two printed factors above.
''')

code(
    "# === R1 reduced to a rank-one twirl (lifted), plus R2's contraction "
    "(glue) ===\n\n"
    + lift("rank_one_twirl")
    + "\n\n\n"
    + r'''
zhat = np.array([0.0, 0.0, 1.0])
s = load_vertices("icosahedron")
R = load_rotations("I")
A, v0 = alignment(s)

# R1: the average twirls the composition readout . noise -- the rank-one
# v0 w^T, w = A^T T^T zhat. The reduction must BE channel_R1 -- asserted,
# not just printed: executing green must mean the identity held.
M1, off1 = rank_one_twirl(R, v0, A.T @ T_NOISE.T @ zhat)
Mf, of_ = channel_R1(s, R, T_NOISE, t_NOISE)
assert np.abs(M1 - Mf).max() < 1e-12
assert np.abs(t_NOISE[2] * off1 - of_).max() < 1e-12
print(f"reduction vs channel_R1:  max|dM| = {np.abs(M1 - Mf).max():.1e}, "
      f"max|doff| = {np.abs(t_NOISE[2] * off1 - of_).max():.1e}")
print(f"kappa = v0 . w = zhat^T T^T zhat = T_zz = "
      f"{v0 @ (A.T @ T_NOISE.T @ zhat):.6f}")

# ...and once with a draw that does NOT twirl (the trivial group), where the
# offsets are far from zero. On the irreducible draw above both offsets
# vanish, so this control is what keeps the offset comparison from passing
# vacuously; the module's own bridge asserts the identity on all 178
# (subgroup, solid) pairs of both lattices (check_coin_group).
R_triv = np.eye(3)[None]
M1t, off1t = rank_one_twirl(R_triv, v0, A.T @ T_NOISE.T @ zhat)
Mft, offt = channel_R1(s, R_triv, T_NOISE, t_NOISE)
assert np.abs(offt).max() > 0.1            # genuinely nonzero
assert np.abs(M1t - Mft).max() < 1e-12
assert np.abs(t_NOISE[2] * off1t - offt).max() < 1e-12
print(f"\ntrivial-draw control: max|off| = {np.abs(offt).max():.3f} != 0,"
      f" and the reduction is still exact")

# R2: summing the V outcomes first contracts sum_k n_k n_k^T = (V/3) Id,
# so the average twirls the noise ALONE and the trace is tr(T)/3.
print(f"\nvertex covariance: max|s^T s - (V/3) Id| = "
      f"{np.abs(s.T @ s - (len(s) / 3) * np.eye(3)).max():.1e}")
G = 3 * np.mean([Rg.T @ (T_NOISE / 3) @ Rg for Rg in R], axis=0)
assert np.abs(G - (np.trace(T_NOISE) / 3) * np.eye(3)).max() < 1e-12
print(f"twirl of the noise alone: max|G - (tr T/3) Id| = "
      f"{np.abs(G - (np.trace(T_NOISE) / 3) * np.eye(3)).max():.1e}")
'''.strip("\n")
)

md(r'''
### 1a. The exact upgrade: both factors are identities in the noise

`check_two_protocols` decided both factors with `allclose` at the single probe. Here each is
quantified over **every** measurement-side $(T, t)$: the channels are linear in the noise, so a
spanning set of probes decides the whole probe space — four probes for R1 (only row $z$ of $T$
and $t_z$ can enter at all), thirteen for R2 (all of $T$ enters; the offset is affine in $t$).
The verdicts run over each solid's own small number field — no tolerance — and, per the
canonical-form discipline, each boolean is paired with a value-for-value float agreement at the
probe, because an exact check does not test its own transcription.

Machinery, in two lifts: the exact rotation layer (T/O/I rebuilt from `main.py`'s canonical
quaternions, in npz row order, no grid and no matching step), then the number-field kit — the
per-solid fields, fail-loud coercion, exact alignment (radical- and transcendental-free:
$A = I + K_w + K_w^2/(1+v_z)$, rational in $v_0$), and literal transcriptions of both channels
over an arbitrary coefficient domain $K$.

The two negative controls at the end are the point: at $T = E_{00}$ (so
$\operatorname{tr}T = 1$ but $T_{zz} = 0$) R2 returns $\mathrm{Id}_3/3$ exactly where R1 returns
$0$ — finding 1 as an identity, not a gap between two decimals — and a reducible draw fails both
tests outright.
''')

code(
    "# === The exact rotation layer (lifted from randomized_core.py) ===\n\n"
    + lift_assign("ROT_ENTRIES")
    + "\n\n\n"
    + lift("_quat_to_rotation_sym")
    + "\n\n\n"
    + lift_assign("_EXACT_ROT")
    + "\n\n\n"
    + lift("exact_rotations")
)

code(
    "# === The number-field kit (lifted from randomized_field.py) ===\n\n"
    + lift_assign("FIELD_GENS", "FIELD_NAME")
    + "\n\n\n"
    + lift("solid_field", "to_field", "_mm", "_mv", "_tr", "_eye",
           "exact_vertices", "exact_alignment", "exact_is_decomposable",
           "exact_orbit_directions", "exact_orbit_counts", "exact_channel_R1",
           "_probe", "exact_twirls", "exact_channel_R2", "exact_twirls_R2",
           "_exact_probe_noise", "_as_float")
)

code(
    "# === Findings 1 + 2, exactly and generically (check_exact_scalars) ===\n\n"
    + lift_body("check_exact_scalars")
)

md(r'''
### 1b. The one misspecification that is a bias, not a premium

Everything above (and everything in §3 cont.) prices a fixed misspecification of the
*measurement* as a $1/\kappa^2$ shot premium. One mistake escapes that pricing, and it turns on
the same two scalars: **a calibration constant carried over from the other protocol.**

The hinge is that a calibration is *empirical*. Learned on the run it reconstructs, it absorbs
whatever the apparatus actually did — an inexact gate, a wrong list, any fixed $(T, t)$ — which
is why those cost shots and never truth. A constant carried across protocols was never learned
on the run being reconstructed. The estimator divides by the believed constant once per touched
site, so a weight-$w$ Pauli term is multiplied by exactly $(\kappa_{\rm run}/\kappa_{\rm cal})^w$
— a bias no shot count removes.

The check proves the law on Appendix F's own estimator — `shadow_experiments.py` imported
lazily and write-free, nothing read from its npz; every number recomputed through its
`noisy_effects` $\to$ `born_tensor` $\to$ `exact_estimator_mean` pipeline on its critical TFIM
ground state — and pins the appendix's printed numbers, both swap directions, plus the control
that separates the two regimes: reconstructed with the constant of the run itself, every
observable is exact.
''')

code(
    "# === The weight-w mismatch law (check_calibration_mismatch) ===\n\n"
    + lift_body("check_calibration_mismatch")
)

md(r'''
### 1c. Finding 6 — gate noise separates the protocols a second time, as an *order* — the reframing, fifth application

Measurement-side noise is one channel shared by every draw: Schur applies, and §1a is exact.
Per-gate noise arrives *inside* the drawn circuit, in an amount and orientation correlated with
$g$ — there is no longer one fixed map to average, and Schur is silent. This is where Appendix
F's numerical study found the twirled-native $Z_0$ residual second order in the damping
$\gamma$ where the projective one is linear; the check below proves that claim as a theorem —
over a *generic* symbolic state, a *generic* dilation strength, and **every** choice of atlas
representative — so it covers the study rather than reproducing it (nothing is read from its
npz).

The mechanism: to first order every insertion reaches the estimator through its word *prefix*
alone — post-processing hits the insertion with $R_g^\top = (S_iP_i)^\top$ and the suffix
cancels, $R_g^\top S_i = P_i^\top$ — so the first-order displacement is a sum over the prefix
multiset of the drawn words. Over the $T$ draw that sum has no $z$ component for any of the
$2^{12}$ representative choices, and the first-order matrix term is *diagonal* — scalar on the
bare row alone — so the $|0\rangle$ calibration, one scalar off the $zz$ entry, absorbs what
the $Z_0$ column sees; the $Z_0$ residual therefore starts at $\gamma^2$. The verdict needs
both the protocol *and* the draw — R1 on the same words is linear, and R2 over the
$O$ draw is linear again, for all $2^{24}$ representative choices.

(The two estimator helpers are deliberately written in the shadow study's normalisation — the
calibration scalar $\eta$, noiseless $1/3$ — rather than $\kappa$; the calibrated residual is a
ratio and cannot see the convention.)
''')

code(
    "# === Gate-noise machinery (lifted from randomized_scalars.py) ===\n\n"
    + lift_assign("NOISE_GATES", "_NOISE_ROT")
    + "\n\n\n"
    + lift("_noise_gate_rotations", "_parse_word", "exact_word_rotation",
           "exact_draw", "gate_noise_channel", "_estimator_R1",
           "_estimator_R2", "calibrated_residual", "_taylor", "_prefix_sum",
           "_mz_reachable", "_exact_seed")
)

code(
    "# === The gate-noise theorem (check_gate_noise_residual) ===\n\n"
    + lift_body("check_gate_noise_residual")
)

md(r'''
### 1d. The alignment, met here and priced in §3

R1's fixed alignment $A$ is the identity exactly when $\hat z$ is already a vertex — the
octahedron, and only it. For every other solid $A$ is not even an element of the covariance
group, so it is genuinely an extra fixed rotation the projective route appends to every drawn
word — and §3 will show it is
*inexact* over every thesis gate set, for every vertex choice. That inexactness is where R1
hides the same field extension R2 pays as Decker's nested radicals: same magic, two hiding
places.
''')

code(
    "# === The alignment per solid (check_alignment) ===\n\n"
    + lift_body("check_alignment")
)


# =============================================================================
# Section 2 -- the two jobs of randomness
# =============================================================================
md(r'''
## 2. The two jobs of randomness — the reframing, second application

Finding 4. The draw's randomness does two jobs at once, and they are *independent properties of
which maps you average*:

- **Realize.** R1 has no ancilla, so the POVM itself must emerge from the draw: outcome
  $(g, b)$ lands the snapshot on $3b\,R_g^\top v_0$, and the drawn set realizes the POVM iff that
  coin hits every vertex uniformly — a $V/2$-way coin over coset representatives. Needs
  transitivity on the vertex axes.
- **Twirl.** The estimator channel collapses to one scalar iff the drawn set acts irreducibly —
  Schur's hypothesis, nothing about vertices at all.

The first check below shows the coin reproduces the POVM's *effects* — the operators, not
merely the statistics — exactly, solid by solid; the second shows the minimal irreducible draw
$T$ twirling every solid while its orbit *fails to realize* the dodecahedron (it covers 6 of the
10 vertex axes: a perfectly unbiased measurement of the wrong POVM). R1's one draw must do both
jobs, so its bill is the larger of two independent bars.
''')

code(
    "# === The coin realizes the POVM itself (check_coset_coin) ===\n\n"
    + lift_body("check_coset_coin")
)

code(
    "# === The minimal twirl, and where realization fails "
    "(check_minimal_twirl) ===\n\n"
    + lift_body("check_minimal_twirl")
)

md(r'''
### The sweep: exhaustive over every finite subgroup of $SO(3)$

The order-counting argument gets a brute-force replacement: *every* subgroup of $O$ and of $I$
— complete lattices, found by pair closure and verified complete in situ — tested independently
for the two jobs on all four decomposable solids. The sweep is exhaustive over every finite
subgroup of $SO(3)$, not merely over the covariance groups: a draw that realizes permutes the
vertex set, hence already lies inside the solid's rotation group.

Three things fall out, all printed below: the twirl bar is flat at $T$ while the realize bar
climbs $3 \to 4 \to 12 \to 60$, crossing at the icosahedron — so it is the *twirl* that forces
the octahedron's and the cube's draws up to order 12 (both realize already, at orders 3 and 4),
both bars bind at the icosahedron, and realization alone binds for the dodecahedron — never
"the $T$ draw happens to also realize three of them"; the protocol's twirl test
reproduces Schur *exactly* (twirling $=$ irreducible, subgroup for subgroup); and the two sets
**nest**, the nesting inverting exactly at the icosahedron, where they coincide. The
dodecahedron block makes the five-inscribed-cubes argument quantitative — no proper subgroup
reaches more than 6 of its 10 axes, so realization alone convicts it.

(The float sweep decides by tolerance three times over — lattice grid, orbit argmin, twirl
`allclose`. Its exact companion `check_exact_two_bars` lifts all three mechanisms to canonical
form and quantifies the twirl over every $(T, t)$; it is not staged here — the final cell's
`main()` runs it.)
''')

code(
    "# === Sweep machinery (lifted from core + twojobs) ===\n\n"
    + lift("subgroup_lattice")
    + "\n\n\n"
    + lift_assign("_LATTICE")
    + "\n\n\n"
    + lift("lattice", "subgroup_kind", "by_order", "two_bars", "twirl_bar",
           "_fmt_orders")
    + "\n\n\n"
    + lift("check_subgroup_sweep")
    + "\n\n\n# bind the returned bars: their repr is a page of arrays\n"
      "_ = check_subgroup_sweep()"
)

code(
    "# === R2's bar: 2T is the UNIVERSAL minimal twirl "
    "(check_universal_twirl) ===\n\n"
    + lift_body("check_universal_twirl")
)

md(r'''
### The witness: a coin that realizes and twirls nothing

Of the four coins only the octahedron's is closed under multiplication — the cyclic $C_3$ about
the body diagonal $(1,1,1)$, i.e. $\{\mathrm{Id}, F, F^\dagger\}$ as rotations, not just as
words. It realizes the octahedral POVM *exactly* and twirls *nothing*, and the failure has a
closed form for arbitrary noise: the coin preserves the entire readout row of $T$ and merely
cycles it (a circulant), where an irreducible group destroys everything in that row but its
diagonal entry. Schur's "irreducibly" made necessary on the reader's own object — and the
commutant dimensions $\frac1{|G|}\sum_g (\operatorname{tr}R_g)^2$ say why: three scalars to
calibrate for $C_3$, one for $T$, $O$, $I$.

Closure is an identity question, so the verdict is decided twice — on the rounding grid and by
canonical field equality — and the symbolic channel identity is bridged back to `channel_R1` on
every (subgroup, solid) pair of both lattices.
''')

code(
    "# === Coin machinery (lifted from core + twojobs) ===\n\n"
    + lift("best_circuits", "coset_representatives")
    + "\n\n\n"
    + lift("coin_rotations", "exact_coin", "exact_coin_is_group")
)

code(
    "# === The C_3 witness (check_coin_group) ===\n\n"
    + lift_body("check_coin_group")
)

md(r'''
### What the draw costs: atlas resources

The bars priced in atlas words. The $2T$ draw is free — all 24 elements at BFS depth $\le 2$,
magic $0$, and still free inside the bigger gate sets — and the coin is cheap everywhere. What
costs is the *draw*, which must clear both bars: since the twirl bar is $T$ for every solid and
$2T$ is all-Clifford, **every $\Phi$ in R1's ledger is charged to realization** — the
dodecahedron alone pays any, and the golden gate is strictly necessary in exactly one case.
''')

code(
    "# === Pricing the draw and the coin (check_atlas_resources) ===\n\n"
    + lift_body("check_atlas_resources")
)

md(r'''
*Novelty note.* The ingredients are classical — a 2-design twirl depolarizes (Schur), $2T$ is
the minimal group 2-design in $d = 2$ (Gross–Audenaert–Eisert 2007), and measurement twirling
is standard error-mitigation practice — but the realize/twirl decoupling for the Platonic
POVMs, the axis-correlated scalar $T_{zz}$ vs $\operatorname{tr}T/3$ distinction, and $2T$'s
universal-minimal-twirl status appear to be original to this thesis. The closest adjacent
work, Nguyen et al. (2022), uses Platonic transitivity for a different purpose (simplifying
the canonical dual), and models readout noise on the dilation ancillas rather than
group-averaging it.
''')


# =============================================================================
# Section 3 -- the exactness obstruction
# =============================================================================
md(r'''
## 3. The exactness obstruction — the reframing, third application

Finding 3. The maps a protocol averages must be *compiled*, and compilation carries the gate
set's arithmetic with it. Two independent obstructions, two different invariants:

- **Direction.** A rank-1 POVM realized by *any* protocol over gates with matrix entries in a
  conjugation-closed field $K$ — any ancillas, adaptivity, classical randomness — must have all
  its Bloch vertices in $K_\mathbb{R}^3$. The witness is the rotation-invariant
  $\det[v_a\,v_b\,v_c]$ of a spanning vertex triple, tested against
  $K_\mathbb{R} = \mathbb{Q}(\sqrt2,\sqrt5)$ — the real field of every gate set of the thesis:
  $2T$, Clifford ($2O$), Clifford${}+T$, $2I$, Clifford${}+\Phi$ and their unions all have
  entries in $K = \mathbb{Q}(\sqrt2,\sqrt5,i)$, and the entangling Clifford each is adjoined
  with is free, CNOT's entries being $0$ and $1$. Which
  triple is a choice, and the choice moves the number — sign always, magnitude and even the
  minimal polynomial on some solids — but never the $K_\mathbb{R}$-coset, so any one triple
  decides. (The triples are printed in the published vertex numbering, which is why
  `atlas_vertices` matters.)
- **Weight** (next cell). The direction lemma's escape hatch — a dilation — is closed by the
  ring: an effect's trace is a finite sum of gate-entry products, landing in
  $\mathcal{R}\cap\mathbb{Q} = \mathbb{Z}[1/2]$, while a transitive covariant POVM on $V$ outcomes
  needs $\operatorname{tr}E_k = 2/V$. So $V$ must be a power of two — and *deterministic* is
  three bans, not one: no coin, no discarded branch, a bounded number of rounds, each closing
  one classical way to buy the division the ring will not supply.

Between them the two convict all five solids; the octahedron fails weight alone, which is why
its exactness must be exhibited with a *coin* — randomness spent to buy $1/3$ as a probability,
never as an amplitude.
''')

code(
    "# === Field instruments (lifted from randomized_core.py) ===\n\n"
    + lift("in_field", "dyadic_order", "det_witness", "det_invariant")
    + "\n\n\n# === The direction obstruction (check_obstruction) ===\n\n"
    + lift_body("check_obstruction")
)

md(r'''
### The weight half

Every thesis gate — Cliffords, $T$, $\Phi$ and its conjugate partner, CNOT — has entries in
$\mathcal{R} = \mathbb{Z}[\tau, i, \sqrt2, \tfrac12]$: algebraic integers over a power of two.
$\mathcal{R}$ is a ring closed under conjugation, so no branch amplitude of any deterministic
protocol ever leaves it. The table then convicts by weight where direction is silent and vice
versa — no solid clears both.
''')

code(
    "# === The weight obstruction (check_weight_obstruction) ===\n\n"
    + lift_body("check_weight_obstruction")
)

md(r'''
### Three corollaries

- **No exact alignment, for any vertex.** The direction lemma is total vertex by vertex: a
  $K_\mathbb{R}$-rational rotation taking *any* vertex to $\pm\hat z$ would put that vertex in
  $K_\mathbb{R}^3$ — so R1's alignment is inexact for every vertex choice, not just the
  conventional $v_0$.
- **Each inexact solid sits on the axes of a magic gate.** $X/Z$ land on the octahedron, $F$ on
  tetrahedron, cube and dodecahedron, and $\Phi$'s rotation axis *is* an icosahedron vertex in
  the published pose — and the eigenstates of $F$ and $\Phi$ are precisely their gate sets'
  magic states, so the field extension a solid demands is the magic of the gate it sits on:
  *the measurement inherits the magic*, and in atlas orientation a Platonic POVM is exactly
  implementable iff its vertices are the Pauli axes.
- **The octahedron has nothing to hide — and the protocols still do not merge there.**
  $A = \mathrm{Id}$, so the projective route on the octahedron *is* random Pauli measurement,
  and the coin realization proves no radicals are *forced* on any route — Decker's octahedral
  dilation still writes $\sqrt{(3\pm\sqrt3)/18}$ into its $U_A$, but that is the
  construction's choice, not the geometry's demand. What vanishes at the octahedron is the
  obstruction, not the distinction: twirled-native keeps its dilation, ancillas and all, and
  still reads $\operatorname{tr}T/3$ where the projective route reads $T_{zz}$.
''')

code(
    "# === No exact alignment anywhere (check_no_exact_alignment) ===\n\n"
    + lift_body("check_no_exact_alignment")
    + "\n\n# === The gate axes (check_gate_axes) ===\n\n"
    + lift_body("check_gate_axes")
    + "\n\n# === The octahedron IS the Pauli bases (check_octahedron_exact) ===\n\n"
    + lift_body("check_octahedron_exact")
)

md(r'''
### The reorientation: Decker's pose against ours

Decker's circuits park each solid's cyclic symmetry axis on $\hat z$ (the size of his Fourier
block is that axis's order); the published pose parks the Pauli axes there. The two differ by a
fixed reorientation per solid — one representative of a coset of the rotation group — and the
question is whether that correction comes free. It does not, by a third field smaller than
$K_\mathbb{R}$: a gate's Bloch action is quadratic in its entries, so the $1/\sqrt2$ of $H$ and
$F$ pairs off and every rotation an *atlas-generated* gate set realizes has $SO(3)$ entries in
$\mathbb{Q}(\sqrt5)$ — *atlas-generated* meaning every **single-qubit** gate an atlas word up
to phase: $2T$, Clifford, $2I$, Clifford$+\Phi$ and their unions, i.e. every thesis gate set
with no $T$ in it. (The scope matters: CNOT is in every thesis gate set and is no atlas word.)
All five reorientations leave that field, so none is an atlas word.
Adjoining $T$ moves the line exactly twice — $\mathrm{Bloch}(T) = R_z(45^\circ)$, the tetrahedron's
and cube's correction, the $T^\dagger$ the circuit figures draw — while the other three stay
barred over every thesis gate set. And the gap between the fields is a trap worth naming:
$K_\mathbb{R}$-exactness is *necessary, never sufficient* — $T$ clears §3's field and ring
tests side by side and is still no atlas word, which is exactly what the glue cell springs.

And yet **the pose costs the R2 estimator nothing**, and that is a theorem rather than two
poses that happened to agree: running `exact_channel_R2` verbatim over the polynomial ring
$\mathbb{Q}(\sqrt5)[C, T, t]$,

$$M(sC) = \frac{\operatorname{tr}(C^\top C\,T)}{3}\,\mathrm{Id}_3, \qquad \mathrm{off}(sC) = 0$$

identically in an *arbitrary* $3\times3$ matrix $C$ and the noise — the measured vertices reach
the channel only through $\sum_k \hat n_k\hat n_k^\top$ and $\sum_k \hat n_k$, and a pose moves
neither. The proviso: the pose costs nothing only *provided the belief moves with the gate*. Each coset
member $m = hR_0$ induces its own labelling of the unreoriented run, priced by the two-list
channel at exactly $\kappa = \operatorname{tr}(mT)/3$ — a family centred on zero, with RMS
$1/3$ noiselessly (the $\operatorname{tr}(m)/3$ law; at the probe the noisy second moment is
$|T|_F^2/27$) — so no single member is ever "the" per-solid figure.

This cell distills each half once (the full `check_reorientation_obstruction` — all five
solids, both noise settings, the tail shift — runs inside the final cell's `main()`).
''')

code(
    "# === Reorientations + the pose theorem (lifted from core + field) ===\n\n"
    + lift("_Rz")
    + "\n\n\n"
    + lift("_Ry")
    + "\n\n\n"
    + lift_assign("REORIENT")
    + "\n\n\n"
    + lift_assign("_RADIAL", "_POSE_FIELD", "_POSE_RING")
    + "\n\n\n"
    + lift("_pose_ring", "exact_reposed_twirl_R2")
)

code(r'''
# === Distilled: the field lemma, the pose theorem, the coset laws (glue) ===

Q5 = sqrt(5)

# (a) The field lemma. Every Clifford acts as a signed permutation, Phi as a
# golden turn -- all inside Q(sqrt5), a field, hence closed under words. The
# Clifford quantifier needs Clifford GENERATORS, so H and S join the atlas
# four (as in the module's own check), and the pair closure below is the
# receipt that <Bloch(H), Bloch(S)> is the full order-24 Clifford rotation
# group -- not the order-12 group the atlas four alone would generate.
gates = dict(atlas_gates())                      # X, Z, F, Phi
gates.update({"H": (1 / sqrt(2)) * Matrix([[1, 1], [1, -1]]),
              "S": Matrix([[1, 0], [0, sI]])})
bloch = {}
for name, U in sorted(gates.items()):
    Rb = bloch[name] = bloch_matrix(U)
    assert all(in_field(e, Q5) for e in Rb), name
    kind = ("signed permutation" if all(e.is_Integer for e in Rb)
            else "golden half-integers")
    print(f"  {name:4s} Bloch matrix: {kind:22s} -- in Q(sqrt5)")
G24 = {sp.ImmutableMatrix(bloch["H"]), sp.ImmutableMatrix(bloch["S"])}
while True:
    grown = G24 | {sp.ImmutableMatrix(a * b) for a in G24 for b in G24}
    if len(grown) == len(G24):
        break
    G24 = grown
assert len(G24) == 24
assert all(all(e.is_Integer for e in Mb) for Mb in G24)
print("  closure: <Bloch(H), Bloch(S)> = 24 signed permutations -- every"
      " Clifford")
# ... and the trap: T's entries pass section 3's field and ring tests, yet
# Bloch(T) = Rz(45 deg) needs 1/sqrt2 -- field-exact is not atlas.
T_GATE = Matrix([[1, 0], [0, (1 + sI) / sqrt(2)]])
assert not all(in_field(e, Q5) for e in bloch_matrix(T_GATE))
print("  T    Bloch(T) = Rz(45 deg): NOT in Q(sqrt5) -- no atlas word")

# (b) All five reorientations leave Q(sqrt5): none is an atlas word.
print()
for solid in SOLIDS:
    m_fb, R0s, demand = REORIENT[solid]
    assert not all(in_field(e, Q5) for e in R0s), solid
    print(f"  {solid:14s} Fourier block {m_fb}, demands {demand}:"
          f" not in Q(sqrt5)")

# (c) The pose theorem, on the SIC (the smallest orbit), with its float
# transcription guard -- and the reducible negative control.
ok, evaluate = exact_reposed_twirl_R2("tetrahedron")
assert ok
R0 = np.array(REORIENT["tetrahedron"][1], dtype=float)
M, off = channel_R2(load_vertices("tetrahedron") @ R0,
                    load_rotations("T"), T_NOISE, t_NOISE)
Me, offe = evaluate(R0, T_NOISE, t_NOISE)
gap = max(np.abs(Me - M).max(), np.abs(offe - off).max())
# the caller contract: the generic boolean above must pair
# with a value-for-value float bridge -- asserted, as the module's own
# caller asserts it, so a drift here dies instead of printing large
assert gap < 1e-12, gap
print(f"\n  tetrahedron: M(sC) = tr(C^T C T)/3 Id, off = 0, identically in"
      f" (C, T, t);")
print(f"  evaluated at C = Decker's R and the probe vs channel_R2:"
      f" max|diff| = {gap:.1e}")
R_F = bloch_matrix(atlas_gates()["F"])
assert not exact_reposed_twirl_R2("octahedron",
                                  [sp.eye(3), R_F, R_F * R_F])[0]
print("  negative control: the reducible C_3 coin fails the theorem, at"
      " every pose")

# (d) The coset laws, noiseless family tr(m)/3: centred on zero, RMS 1/3.
print(f"\n  {'solid':14s} {'coset':>6s} {'induced kappa range':>21s}"
      f" {'<k>':>9s} {'<k^2>':>7s}")
for solid in SOLIDS:
    R0 = np.array(REORIENT[solid][1], dtype=float)
    rots = load_rotations(COVARIANCE[solid])
    kaps = np.array([np.trace(h @ R0) / 3 for h in rots])
    assert abs(kaps.mean()) < 1e-12
    assert abs((kaps ** 2).mean() - 1 / 9) < 1e-12
    print(f"  {solid:14s} {len(rots):>6d}  [{kaps.min():+7.4f},"
          f" {kaps.max():+7.4f}] {kaps.mean():>9.1e}"
          f" {(kaps ** 2).mean():>7.4f}")

# ... and one member fed through the two-list channel itself: believe m d_k,
# measure d_k, and the average returns exactly tr(m T)/3 Id, zero offset.
R0 = np.array(REORIENT["icosahedron"][1], dtype=float)
rots = load_rotations("I")
s_d = load_vertices("icosahedron") @ R0
memb = rots[5] @ R0
M, off = channel_R2(s_d @ memb.T, rots, T_NOISE, t_NOISE, s_actual=s_d)
kap = np.trace(memb @ T_NOISE) / 3
assert np.abs(M - kap * np.eye(3)).max() < 1e-10
assert np.abs(off).max() < 1e-9
print(f"\n  coset member #5 (icosahedron), two-list channel: kappa ="
      f" {kap:+.6f},")
print(f"  max|M - kappa Id| = {np.abs(M - kap * np.eye(3)).max():.1e},"
      f"  max|off| = {np.abs(off).max():.1e}")
''')


# =============================================================================
# Section 3 (cont.) -- Decker's circuits, outcome order, tail weight
# =============================================================================
md(r'''
## 3 (cont.) Decker's outcome order — the reframing, fourth application

Finding 5. A reorientation carries vertices to vertices but not indices to indices, so a second
correction stands between Decker's circuits and our vertex list: a relabelling of outcomes.
Pricing it needs a datum the rotation never sees — his vertex list **in his outcome order** —
so the five circuits are rebuilt from his own formulas. The order is not ours to choose: he
fixes it twice, as the columns of his printed $M$ and as the numbered vertices of his figures,
so the rebuild is *self-certifying* — it must reproduce his printed columns value for value, in
his order, anchored on his stated vertex 1, and a wrong Fourier convention dies loudly instead
of returning a plausible permutation.

Then the reframing again: skipping the relabelling means the estimator *believes* one list of
maps while the device performs another — and for an irreducible draw any fixed mismatch twirls
to exactly $\frac{\operatorname{tr}(BT)}{V}\,\mathrm{Id}$ with zero offset,
$B = \sum_k b_ka_k^\top$ pairing belief against device. No bias — no relabelling can tilt a
twirled estimator — but the estimator shrinks by the overlap $\kappa$, for a $1/\kappa^2$ shot
premium the table prices per solid, from benign to catastrophic. That the premium is *exactly*
$1/\kappa^2$ for every state is a second fact, checked below: each coordinate Pauli's
single-shot second moment stays $3/\kappa^2$, which needs the atlas pose's coordinate
half-turns, not just irreducibility. The anchors close the story with labels that have no geometry behind them:
the antipodal belief, every SIC derangement, and the exhaustive scramble law over all $V!$
bijections — where, noiselessly, $\kappa$ can vanish
outright and the estimator is killed rather than taxed.
''')

code(
    "# === Decker's circuits, rebuilt from his formulas "
    "(lifted from randomized_decker.py) ===\n\n"
    + lift_assign("_DA3", "_DP", "_DM", "_DAD", "_DGD", "_DGI",
                  "FOURIER_SIGN", "_ANCHOR_TOC", "_ANCHOR_ID",
                  "DECKER_ANCHOR")
    + "\n\n\n"
    + lift("decker_fourier", "_pad_block", "_cnot", "bloch_of_ket",
           "decker_columns", "decker_circuit", "decker_vertices")
)

code(
    "# === His outcome order, and what mislabelling costs "
    "(check_decker_outcome_order) ===\n\n"
    + lift_body("check_decker_outcome_order")
)

md(r'''
### The tail weight: what the correction buys

The mislabelling premium said what skipping the corrections *costs*; this functional says what
performing them *buys*. The single-Pauli estimate's fourth moment is
$27\,\langle w_x^4 + w_y^4 + w_z^4\rangle$ over the swept snapshot directions — asserted below
to be the exact protocol average at every pose, draw (the minimal $T$ draw included), Pauli and
state: state-free, axis-free, draw-blind, a property of the *posed vertex set* alone. Second
moments cannot see any of it — a union of rotated copies of a 2-design is still a 2-design, so
the variance is $3$ either way — which is why the pose surfaces in the fourth moment or
nowhere. Two SOS
identities pin its range, floor exactly the eight cube directions, ceiling exactly the six
Pauli axes — so the published pose is the tetrahedron's and cube's global optimum and the
octahedron's global pessimum, and the unreoriented (Decker-pose) numbers of Appendix D.2 follow
exactly, the 5-designs immovable at the sphere's own value — and in Decker's pose the cube's
famously light tails do not merely thicken against the octahedron's but *reverse*: Appendix D.2's
point that the cube's advantage is a property of its atlas orientation, demonstrated. The same
functional prices Appendix
F.3.3's $T$-draw orbit POVMs of the dodecahedron — the inscribed-cube eight against the golden
twelve, $\sigma^4 + \tau^4 = 7$ behind the split.
''')

code(
    "# === The tail-weight functional (check_tail_weight) ===\n\n"
    + lift_body("check_tail_weight")
)


# =============================================================================
# Section 4 -- the ledger
# =============================================================================
md(r'''
## 4. The implementation ledger

The suite's Section 4 collects the bill: native (Naimark) ancilla counts against the projective
route's atlas circuits, the field demand each solid's exactness obstruction names, and the draw
that clears both bars. Decker's native circuits are Naimark dilations — a register of dimension
$\ge V$, prepared by inter-orbit unitaries whose amplitudes carry exactly the radicals the
lemma forces; that bound makes their ancilla counts minimal *within* the dilation route, so on
their own terms they are not beatable. The projective route beats them by *leaving* those terms
— one qubit, a coin, Clifford-cheap coset circuits — wherever antipodality permits it to exist.

(Run as a script — never from this notebook — the entry point also emits this ledger and five
more fragments as LaTeX into `code/data/`; every printed number is recomputed from the same
loaders and primitives the checks used. The thesis's Table 5.2 is this same ledger transposed —
solids as columns, the master spec sheet `spec_sheet()` derives from these primitives — so the
two shapes are one table.)
''')

code(
    "# === Section 4: the implementation ledger (print_ledger) ===\n\n"
    + lift_body("print_ledger")
)

md(r'''
### The three corners

Disambiguated and priced, the closing trade-off is a triangle, not a dichotomy:

| corner | circuit | calibration | SIC? | price |
|---|---|---|---|---|
| **native** | Decker dilation, no random gates | *assumes* depolarizing; fits one scalar $\hat\eta$ from $\vert0\rangle$ shots | yes | structurally blind to any channel fixing $\vert0\rangle$ |
| **twirled-native** | native + a $2T$ draw | depolarizing is a *theorem*; factor $\operatorname{tr}T/3$ | yes | the dilation's ancillas, plus a depth-$\le2$ all-Clifford draw |
| **randomized-projective** | one qubit: $U_g$, $A$, measure $Z$ | depolarizing is a theorem; factor $T_{zz}$ | no | the SIC forfeit, the alignment $A$, and — for the dodecahedron alone — the golden gate per shot |

Two sentences carry the whole picture. **The SIC is not the price of the twirl — the ancilla
is:** twirled-native keeps the tetrahedron and twirls it, paying only the dilation it already
owed. **The magic cannot be dodged, only relocated:** the projective route saves the ancillas
and pays $A$; the native route pays the radicals; the octahedron is the one POVM whose
*projective* corner has no field bill at all — $A = \mathrm{Id}$, literally random Pauli
measurements (its native corner still cannot be exact: the weight obstruction stands in every
pose, and Decker's octahedral circuit carries a $\sqrt3$ from his orientation). The other two
bills survive even there: the native corner keeps its blind spot, and the twirled-native
corner its ancillas and its own factor — the corners never collapse; their common obstruction
just vanishes at that one point.
''')


# =============================================================================
# Section 5 -- the design ladder
# =============================================================================
md(r'''
## 5. Remark: the unitary-design ladder

The twirl needs only a unitary 2-design, and $2T$ is the minimal *group* one in $d = 2$ — but
the three binary groups climb higher: exact 2-/3-/5-designs, read off frame potentials against
the Catalan numbers (for a group the potential is an integer, so failing a level overshoots by
at least 1; $2I$ meets $t = 5$ exactly and first fails at $t = 6$, the degree of the icosahedral
invariant). The glue line re-derives the *spherical* design strengths of the vertex sets by
float moment averaging and pins them to `povm_properties.py`'s `EXPECTED_DESIGN` — the one
place the repo fixes the ladder's values. A pin, not a corroboration: the exact number-field
derivation behind those five integers runs inside that script's own `main()`, not here — two
mechanisms sharing nothing, meeting at one hardcoded dict.
''')

code(
    "# === Frame potentials + spherical strengths (lifted) ===\n\n"
    + lift("frame_potential", "design_strength")
    + "\n\n\n# === The ladder (check_unitary_designs) ===\n\n"
    + lift_body("check_unitary_designs")
    + "\n\n\n"
    + r'''
# glue: the SPHERICAL strengths of the vertex sets, pinned to the ladder's
# single source (import is write-free; povm_properties' own exact derivation
# asserts against the same dict when THAT script runs, not here)
from povm_properties import EXPECTED_DESIGN

got = {solid: design_strength(load_vertices(solid)) for solid in SOLIDS}
assert got == EXPECTED_DESIGN, got
print("spherical t-designs:  "
      + "  ".join(f"{s} {t}" for s, t in got.items()))
print("pinned to povm_properties.EXPECTED_DESIGN (the ladder's single source)")
'''.strip("\n")
)


# =============================================================================
# Section 6 -- the receipts
# =============================================================================
md(r'''
## 6. The receipts

The cells above *are* the modules — lifted mechanically at build time — but a committed
notebook can drift after later module edits. This cell closes the gap:

1. **currency**: the build pinned a sha256 of each of the eight module files *and of the
   builder*; the cell re-hashes them now. Any later edit — lifted or not, code or comment —
   fails here naming the rebuild + re-execute pair, so a stale committed notebook cannot
   execute quietly. Textual identity is the whole anti-drift guarantee; the next item is a
   demonstration on top of it;
2. **behavioral spot-checks**: the notebook's rebuilt primitives against the imported
   production modules, each side fed its own inputs — same arithmetic run twice, so the honest
   residual is $0.0$, not a tolerance;
3. **the full report**: the ledger's `spec_sheet()` re-derived and pinned (read-only — the
   writers live in `write_fragments`, behind the `__main__` guard), then the entry point's
   `main()` end to end. Every check staged above runs again inside it, and every check *not*
   staged here — the exact two-bars companion, the full reorientation check, and the rest —
   runs too. The only thing in the suite this notebook never executes is the fragment writers
   themselves: the LaTeX layer over the rows `spec_sheet` just pinned, which is exactly what
   *writes nothing* buys.

The six thesis fragments are emitted only when `randomized_implementations.py` runs as a
script.
''')

_pin_lines = "\n".join(f'    "{name}": "{PINNED[name]}",'
                       for name in sorted(PINNED))
code(
    r'''
# === Anti-drift: currency, spot-checks, spec_sheet, then main() ===

import hashlib
import io
import sys

import randomized_implementations as ri
import randomized_core as rc
import randomized_scalars as rsc
import randomized_twojobs as rtj
import randomized_field as rfl
import randomized_decker as rdk
import randomized_fragments as rfr

# (1) Currency. The files on disk must be the files this notebook was built
# from -- pinned at build time, byte for byte, builder included. Any later
# edit fails HERE, naming the remedy: textual identity is the anti-drift
# guarantee; the spot-checks below are a demonstration on top of it.
_PINNED = {
'''.strip("\n")
    + "\n" + _pin_lines + "\n"
    + r'''
}
for _name in sorted(_PINNED):
    _got = hashlib.sha256(Path(f"{_name}.py").read_bytes()).hexdigest()
    assert _got == _PINNED[_name], (
        f"{_name}.py changed since this notebook was built -- rebuild and "
        "re-execute:\n"
        "  uv run python _build_randomization_walkthrough.py\n"
        "  uv run --with jupyter --with nbconvert jupyter nbconvert "
        "--to notebook --execute --inplace randomization_walkthrough.ipynb")
print(f"currency: {len(_PINNED)} files match their build-time sha256 pins\n")

# (2) Behavioral spot-checks, each side fed its own inputs -- same
# arithmetic run twice, so the honest residual is 0.0, not a tolerance.
assert np.array_equal(T_NOISE, ri.T_NOISE)
assert np.array_equal(t_NOISE, ri.t_NOISE)
s, R = load_vertices("icosahedron"), load_rotations("I")
for f_nb, f_mod in ((channel_R1, ri.channel_R1), (channel_R2, ri.channel_R2)):
    M_nb, o_nb = f_nb(s, R, T_NOISE, t_NOISE)
    M_md, o_md = f_mod(rc.load_vertices("icosahedron"),
                       rc.load_rotations("I"), rc.T_NOISE, rc.t_NOISE)
    assert np.array_equal(M_nb, M_md) and np.array_equal(o_nb, o_md)

# core: alignment, rot_key, the two bars, the ladder, the symbolic layer
A_nb, v_nb = alignment(s)
A_md, v_md = rc.alignment(rc.load_vertices("icosahedron"))
assert np.array_equal(A_nb, A_md) and np.array_equal(v_nb, v_md)
assert rot_key(R[7]) == rc.rot_key(rc.load_rotations("I")[7])
b_nb, b_md = two_bars("cube"), rc.two_bars("cube")
assert all(b_nb[k] == b_md[k] for k in
           ("bar_realize", "bar_twirl", "binds", "draw", "min_realize"))
U_T = load_atlas("T")["unitaries"]
assert frame_potential(U_T, 2) == rc.frame_potential(
    rc.load_atlas("T")["unitaries"], 2)
s_c = load_vertices("cube")
assert design_strength(s_c) == rc.design_strength(rc.load_vertices("cube"))
assert sp.simplify(det_invariant(atlas_vertices("cube"))
                   - rc.det_invariant(rc.atlas_vertices("cube"))) == 0

# field: the exact kit over Q (octahedron) AND over the icosahedron's
# algebraic number field -- compared by domain equality, which `is` could
# only ever decide for the QQ singleton
K = solid_field("octahedron")
assert K == rfl.solid_field("octahedron")
s_e = exact_vertices("octahedron", K)
assert s_e == rfl.exact_vertices("octahedron", K)
R_e = [to_field(M_, K) for M_ in exact_rotations("O")]
R_em = [rfl.to_field(M_, K) for M_ in rfl.exact_rotations("O")]
assert R_e == R_em
assert exact_channel_R2(s_e, R_e, *_probe(K, entry=(0, 0)), K) == \
    rfl.exact_channel_R2(rfl.exact_vertices("octahedron", K), R_em,
                         *rfl._probe(K, entry=(0, 0)), K)
K5 = solid_field("icosahedron")
assert K5 == rfl.solid_field("icosahedron") and K5 != sp.QQ
assert exact_vertices("icosahedron", K5) == \
    rfl.exact_vertices("icosahedron", K5)

# scalars: the word layer;  twojobs: the coin;  decker: the circuits
assert _parse_word("X F†") == rsc._parse_word("X F†")
toks_nb, Rs_nb = exact_draw("T")
toks_md, Rs_md = rsc.exact_draw("T")
assert toks_nb == toks_md and Rs_nb == Rs_md
assert np.array_equal(coin_rotations("octahedron"),
                      rtj.coin_rotations("octahedron"))
d_nb, live_nb, W_nb = decker_vertices("dodecahedron")
d_md, live_md, W_md = rdk.decker_vertices("dodecahedron")
assert np.array_equal(d_nb, d_md) and live_nb == live_md
assert np.array_equal(W_nb, W_md)
print("spot-checks: the rebuilt primitives agree with the modules\n")

# (3) The ledger's own derivation, re-derived and pinned. Read-only: the
# writers live in write_fragments, behind the __main__ guard -- and the
# tripwire in the first cell would refuse them anyway.
_spec = rfr.spec_sheet()
assert set(_spec) == set(SOLIDS)
print(f"spec_sheet: {len(_spec)} ledger rows re-derived; ladder and bar"
      f" agreement pinned\n")

# ... then the entry point's own report, end to end.
class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def writable(self):
        return True

    def write(self, text):
        for st in self.streams:
            st.write(text)
        return len(text)

    def flush(self):
        for st in self.streams:
            st.flush()


_buf = io.StringIO()
_stdout = sys.stdout
sys.stdout = _Tee(_stdout, _buf)
try:
    ri.main()
finally:
    sys.stdout = _stdout
print(f"\n[ok] blocks in the report above: {_buf.getvalue().count('[ok]')}")
'''.strip("\n")
)


# =============================================================================
# Closing notes
# =============================================================================
md(r'''
## Closing notes

**What was shown**, finding by finding:

| finding | where | the one line |
|---|---|---|
| 1 | §1, §1a, §1b | the estimator-channel factor identifies the protocol — $T_{zz}$ vs $\operatorname{tr}T/3$, both *identities* in the noise; and the one mistake priced differently: a constant carried across protocols is a bias, everything else a premium |
| 2 | §1, §1a | the SIC is not the price of the twirl; the ancilla is — R2 twirls the tetrahedron exactly, R1 cannot even be defined for it |
| 3 | §3 | direction ($K_\mathbb{R}$) and weight ($\mathbb{Z}[1/2]$) between them convict all five; the octahedron survives only through the coin, and *deterministic* is three bans, not one |
| 4 | §2 | realize and twirl are independent properties of the drawn set; the bars cross at the icosahedron, the sweep is exhaustive over every finite subgroup of $SO(3)$, and the $C_3$ coin is the witness that Schur's hypothesis is necessary |
| 5 | §3 cont. | Decker's circuits rebuilt in his own outcome order; a skipped relabelling twirls to one overlap $\kappa$ — a $1/\kappa^2$ premium, never a bias — and the tail weight is an exact pose functional with the published pose extremal |
| 6 | §1c | gate noise separates the protocols as an *order* in $\gamma$: the twirled-native $Z_0$ residual is second order on the $2T$ draw — a fact needing both the protocol and the draw's prefix multiset — where every projective row stays linear |

**The through-line, once more.** A protocol is which maps you average: the object handed to the
group average decides the scalar (§1), the drawn set's orbit and its irreducibility decide the
two jobs (§2), the gate field decides which maps exist exactly (§3), a mismatch between
believed and performed maps twirls to one overlap (§3 cont.), and noise correlated with the
draw is precisely where "which maps you average" stops being well-posed — and the failure is
itself exactly priceable (§1c).

**Where this sits in the repo.** The suite backs Section 5.2.3 (the implementation ledger),
Appendix D (exactness, Decker's circuits, outcome order) and Appendix F.3 (the estimator
channels, the two-bars sweep, the $C_3$ witness); the definitional seam is Chapter 4's *Two
Randomized Implementations* subsection, whose dashed-box figure is the twirled-native picture.
The numerical shadow study that consumes the
two channels lives in `shadow_experiments.py`, with its own walkthrough
(`shadow_walkthrough.ipynb`) — the two notebooks meet at the two-protocol distinction and
otherwise divide the labor: the variance landscape, dual optimization and Monte Carlo live
there; the exact scalars, the two jobs, the obstructions and the pricing live here.

To run the production suite end to end (writes the six LaTeX fragments; deterministic):

```
cd code && uv run randomized_implementations.py
```

To regenerate this notebook after editing the builder (never edit the .ipynb directly):

```
cd code && uv run python _build_randomization_walkthrough.py
uv run --with jupyter --with nbconvert jupyter nbconvert --to notebook --execute --inplace randomization_walkthrough.ipynb
```
''')


# =============================================================================
# Static self-checks, then assemble and write
# =============================================================================

IGNORED_NAMES = {"__conditional_annotations__"}   # PEP 649 symtable artifact


def _cell_symbols(src, label):
    """(provided, required_top, required_nested) of one source, via symtable.

    provided: names assigned or imported at top level. required_top: names
    referenced at top level without a top-level binding -- these evaluate at
    cell-execution time, so they are order-sensitive. required_nested:
    globals referenced from nested scopes (function bodies) -- these
    evaluate at call time and are exempt from the order pass.
    """
    table = symtable.symtable(src, label, "exec")
    provided, req_top, req_nested = set(), set(), set()

    def walk(tab, top):
        for sym in tab.get_symbols():
            nm = sym.get_name()
            if top:
                if sym.is_assigned() or sym.is_imported():
                    provided.add(nm)
                elif sym.is_referenced():
                    req_top.add(nm)
            elif sym.is_global():
                req_nested.add(nm)
        for child in tab.get_children():
            walk(child, False)

    walk(table, True)
    return provided, req_top - IGNORED_NAMES, req_nested - IGNORED_NAMES


def _closure_check(cell_list):
    """Prove the notebook closed, three ways, before anything is written.

    1. Resolution: every name a code cell needs resolves to a builtin, a
       binding made by that cell, or a single-assignment carry -- a name
       bound in exactly one earlier cell (the lifts, the imports, the
       constants).
    2. The scratch discipline: a name bound in MORE than one cell never
       crosses a cell boundary -- whatever a cell reuses, it rebinds. This
       is what makes a deleted glue line die at build time instead of
       executing against a stale value leaked from an earlier cell.
    3. Order: at cell top level nothing is used before its own binding.
       Function bodies are exempt (they run at call time); every top-level
       statement's prefix is re-analyzed, so `print(x)` before `x = 3`
       fails even though the full cell binds `x`.

    Violations die loudly, naming the cell, the names and -- for scratch
    leaks -- every cell binding the name. _closure_selfcheck() keeps all
    three failure classes firing.
    """
    known_builtins = set(dir(builtins))
    infos, problems = [], []
    for idx, cell in enumerate(cell_list):
        if cell["cell_type"] != "code":
            continue
        src = cell["source"]
        try:
            ast.parse(src)
        except SyntaxError as e:
            problems.append(f"cell {idx}: syntax error at line {e.lineno}: {e.msg}")
            continue
        infos.append((idx, src) + _cell_symbols(src, f"<cell {idx}>"))
    binders = {}
    for idx, _, provided, _, _ in infos:
        for nm in provided:
            binders.setdefault(nm, []).append(idx)
    for idx, src, provided, req_top, req_nested in infos:
        head = src.splitlines()[0][:64]
        carries = {nm for nm, cs in binders.items()
                   if len(cs) == 1 and cs[0] < idx}
        missing = (req_top | req_nested) - known_builtins - provided - carries
        if missing:
            notes = [f"{nm} (bound in cells {binders[nm]})" if nm in binders
                     else f"{nm} (bound nowhere)" for nm in sorted(missing)]
            problems.append(f"cell {idx} ({head!r}): unresolved or "
                            f"scratch-leaked: {'; '.join(notes)}")
            continue
        lines = src.splitlines()
        for stmt in ast.parse(src).body:
            prefix = "\n".join(lines[:stmt.end_lineno])
            _, p_top, _ = _cell_symbols(prefix, f"<cell {idx} prefix>")
            early = p_top - known_builtins - carries
            if early:
                problems.append(f"cell {idx} ({head!r}): used before "
                                f"definition (by line {stmt.end_lineno}): "
                                f"{sorted(early)}")
                break
    if problems:
        raise SystemExit("lift-completeness check FAILED:\n  "
                         + "\n  ".join(problems))


def _closure_selfcheck():
    """The checker must keep catching its three target classes -- one
    deliberate mutation each, kept as permanent fixtures -- and keep
    passing the two legitimate patterns it exists to permit."""
    def cc(*sources):
        return [{"cell_type": "code", "source": s} for s in sources]

    for bad in (cc("undefined_name_xyz + 1"),               # unresolved
                cc("sl = 1", "sl = 2", "print(sl)"),        # scratch leak
                cc("print(zzz)\nzzz = 3")):                 # use before def
        try:
            _closure_check(bad)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"closure check failed to catch {bad}")
    _closure_check(cc("carry = 1", "print(carry)"))     # 1-cell carry: fine
    _closure_check(cc("x: int = 1\nprint(x)"))          # PEP 649: no ghost


def _write_guard(cell_list):
    """Build-time LINT for the two obvious writer patterns. The actual
    write-freedom guarantee is the notebook's own tripwire cell -- a runtime
    audit hook covering every idiom and every call -- this just catches the
    known patterns before the ~3-minute execute would. The tripwire cell
    deliberately stages one refused write-mode open, so it is exempt."""
    for idx, cell in enumerate(cell_list):
        if cell["cell_type"] != "code":
            continue
        if "The write tripwire (glue)" in cell["source"]:
            continue
        assert "write_fragments(" not in cell["source"], \
            f"cell {idx} lifts/calls the fragment writer"
        assert not ("open(" in cell["source"] and '"w"' in cell["source"]), \
            f"cell {idx} appears to open a file for writing"


# The setup cell duplicates randomized_core.DATA's trailing-slash-string
# contract (a notebook has no __file__): bind the duplicate to the original
# so a module-side idiom change dies at build, not two minutes into the
# mandatory re-execute (the lifted loaders concatenate DATA + name as
# strings).
import randomized_core as _rc
assert isinstance(_rc.DATA, str) and _rc.DATA.endswith("/"), _rc.DATA

_closure_selfcheck()
_closure_check(cells)
_write_guard(cells)

nb.cells = cells
for i, cell in enumerate(nb.cells):
    cell["id"] = f"cell-{i:02d}"            # deterministic ids across rebuilds
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "pygments_lexer": "ipython3",
}
nbformat.validate(nb)

out = HERE / "randomization_walkthrough.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"Wrote {out} ({len(cells)} cells: {n_code} code, "
      f"{len(cells) - n_code} markdown); closure/scratch/order checks "
      f"passed, write lint clean, {len(PINNED)} files pinned")
