"""Build ``code/data/recipe_tet_data.tex`` -- the data of Figure 1.1.

Figure 1.1 works the tetrahedral POVM end to end on one page, and every number
on it is already owned by some other generator: the vertices by
``povm_properties``, the effects by ``export_numpy``, the twelve rotations and
their words by ``main``, the five-solid foot strip by
``randomized_implementations``.  A figure that re-typed any of them would be a
fifth owner, and the one that drifts silently -- so nothing here is typed.  The
four blocks this writes are derived from the group and POVM data and then
checked back against the very tables the page cites, which is the only reason a
reader may trust the page against Appendix A, D and E without opening them.

What it emits, as four ``\\newcommand``s that ``recipe_tet_body.tex`` uses:

``\\RecipeMasterTable``    band A: k, the Bloch vector, the effect in factored
                          form, the outcome bits, and <0|E_k|0> -- the
                          probabilities on |0>, headed by the matrix element
                          rather than by p(k) so that the column names its own
                          state before step 4 has defined p(k).
``\\RecipeCircuitParams``  band B: U_A, T-dagger, and alpha/beta to four places.
``\\RecipeRotationTable``  band C: the twelve rotations of T, both atlas words,
                          the axis, and what outcomes 1234 read as once the
                          word is prepended -- g^{-1}, the relabelling, NOT
                          the rotation's action on the vertices (the column
                          answers "I prepended your word and read 1234; what
                          did I measure?", and read site by site the frames
                          show the same thing).
``\\RecipeSeriesStrip``    band D: the same six steps across all five solids.

The twelve checks it runs before it will write anything (``verify()``):

1.  the POVM axioms -- sum E = 1, trace 1/2, rank one, and the SIC overlaps 1/3;
2.  the printed factored matrices against ``povm_tetrahedron.npz``;
3.  p(k) = Tr[E_k |0><0|], which is also the top-left entry of E_k, against the
    four decimals actually printed;
4.  the inversion r = 3 sum_k p(k) n_k, on z-hat and on an off-axis state;
5.  the twelve permutations, computed by conjugation: exactly twelve, each
    realized by exactly two elements of 2T, and those two are U and -U;
6.  every printed axis against the rotation it labels -- and against the axis
    ``atlas.tex`` prints for the same pair;
7.  both atlas words of all twelve pairs against rows 1a-12b of ``atlas.tex``,
    character for character, the words themselves rendered from the npz
    synthesis sequences and multiplied back out to the npz unitary;
8.  the algebra the page states in words: FF = -F-dagger, F^3 = -1,
    XZ = -ZX, and perm(X) after perm(Z) = perm(XZ) -- and, for the printed
    readings, the reverse: reads(FX) = reads(X) after reads(F), which is why
    the page prints no composition rule beyond the order-blind pair 10;
9.  the outcome key, by rebuilding Decker's Naimark circuit numerically and
    matching each computational outcome to its effect -- Table D.4's
    tetrahedron row reproduced rather than asserted; then the same circuit
    with each of the twelve a-words prepended, which must read outcomes as
    the printed column says, pair by pair; and the same circuit with its
    T-dagger removed, whose effects must be ours turned by R_z -- which is
    all "T-dagger reorients the published circuit to our pose" claims;
10. the foot strip's cells against ``randomized_ledger.tex`` (Table 5.2),
    ``randomized_ledger_appendix.tex`` (Table D.1), Table D.3 in the thesis
    source, and the wire counts of the five ``circuit_dec_*.tex``;
11. the three token frames of ``recipe_tet_frames.tex`` against g-inverse --
    a numeral drawn at site m is g^{-1}(m), and getting that backwards is the
    single likeliest error on the page; read site by site, frame 1 must be
    pair 2's printed row and frame 2 pair 3's;
12. the facts ``recipe_tet_body.tex`` hand-types in prose -- the permutations,
    the pair numbers, the axes, the angles, the bit order and the two clauses
    of Theorem 1 -- each rebuilt from the group and then required verbatim, so
    the sentences may be rewritten but the facts in them cannot drift.

One more check has no number because it guards a copy rather than a datum:
``verify()`` ends by requiring the ``\\Qcircuit`` body of
``recipe_tet_frames``' sibling ``recipe_tet_circuit.tex`` to be the one in
Figure 4.1, whitespace aside.  The two are duplicated on purpose rather than
pulling Figure 4.1's inline block out into a shared file; this check is what
makes the duplication safe.

Edit this builder, never ``code/data/recipe_tet_data.tex`` (regeneration
overwrites manual edits; the file lives beside the other fragments the thesis
``\\input``s from ``../code/data/``).  Only the prose, the step heads and the caption of
Figure 1.1 are hand-written, and they live in ``recipe_tet_body.tex`` and
``bsc-thesis.tex`` respectively.

Run with ``uv run _build_recipe_figure.py``.
"""

import re
import sys
from pathlib import Path

import numpy as np

# --- where things live -----------------------------------------------------
# In the repo this file sits in code/ and the tree is code/ + paper/.  Placed
# at a tree root that holds code/ and paper/ instead, it finds both roots
# there, so neither is assumed.
_HERE = Path(__file__).resolve().parent
CODE = _HERE if (_HERE / "data").is_dir() else _HERE / "code"
PAPER = (_HERE if (_HERE / "paper").is_dir() else CODE.parent) / "paper"
DATA = CODE / "data"
FIGURES = PAPER / "figures"
OUT = DATA / "recipe_tet_data.tex"

SOLIDS = ["tetrahedron", "octahedron", "cube", "icosahedron", "dodecahedron"]
SHORT = {"tetrahedron": "tet", "octahedron": "oct", "cube": "cube",
         "icosahedron": "icos", "dodecahedron": "dodec"}

TOL = 1e-15          # what "equal" means for a 2x2 built out of radicals
LOOSE = 1e-9         # ... and for anything that has been through an eigensolve

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.array([[1, 0], [0, -1]], complex)
PAULI = [SX, SY, SZ]
I2 = np.eye(2, dtype=complex)


# --- reading the repo ------------------------------------------------------

def load():
    """The three npz files the page's own numbers come from."""
    povm = np.load(DATA / "povm_tetrahedron.npz", allow_pickle=True)
    group = np.load(DATA / "group_2T.npz", allow_pickle=True)
    gates = np.load(DATA / "gates.npz", allow_pickle=True)
    return povm, group, gates


def gate_table(gates):
    """Name -> SU(2) matrix, daggers included.  The dagger is the SU(2)
    inverse, so X-dagger is -X and not X; that is the atlas's convention and
    the reason a word cannot be read as a string of Pauli letters."""
    tab = {str(n): np.array(u) for n, u in zip(gates["names"], gates["su2"])}
    tab.update({f"{n}\u2020": np.array(u).conj().T for n, u in tab.items()})
    tab["I"] = I2
    return tab


def word_matrix(word, tab):
    """Operator order: the leftmost letter is applied last."""
    out = I2
    for letter in word.split():
        out = out @ tab[letter]
    return out


def word_latex(word):
    r"""``F\u2020 Z`` -> ``\mathbf{F}^{\dagger}\mathbf{Z}``, exactly the shape
    ``atlas.tex`` prints, so the two can be compared character for character."""
    if word == "I":
        return r"\Id"
    out = []
    for letter in word.split():
        base = letter.rstrip("\u2020")
        body = base if base == "Phi" else r"\mathbf{%s}" % base
        if base == "Phi":
            body = r"\Phi"
        out.append(body + (r"^{\dagger}" if letter.endswith("\u2020") else ""))
    return "".join(out)


def read_atlas_2t():
    """Rows 1a-12b of Table A.1, from ``code/atlas.tex``.

    Follows ``main.py``'s ``verify_sample_row`` precedent: parse the emitted
    LaTeX rather than trust that it says what the generator meant.  Returns
    {(pair, 'a'|'b'): {...}} with the word, depth, magic cost, and -- for the
    a-row, which carries the pair's \\multirow cells -- the rotation axis and
    angle.
    """
    text = (CODE / "atlas.tex").read_text()
    start = text.index("% --- Binary Tetrahedral Group 2T ---")
    end = text.index("% --- Binary Octahedral Group 2O ---")
    block = text[start:end]
    rows = {}
    row_re = re.compile(
        r"^\s*(\d+)([ab])\s*&\s*(\d+)\s*&\s*(\d+)\s*&\s*\$(.+?)\$\s*&(.*)$")
    axis_re = re.compile(
        r"\\begin\{psmallmatrix\}\s*(.+?)\s*\\end\{psmallmatrix\}", re.S)
    for line in block.splitlines():
        m = row_re.match(line)
        if not m:
            continue
        pair, ab, depth, magic, word, rest = m.groups()
        entry = {"word": word, "depth": int(depth), "magic": int(magic)}
        if ab == "a":
            # The a-row carries the pair's \multirow cells: the axis n, then
            # theta, then the quaternion and the unitary.  The identity's axis
            # cell is "---" and the unitary cell is itself a psmallmatrix, so
            # the "---" has to be tested for before the matrix is looked for.
            axis_cell = rest.split(r"\multirow")[1] if r"\multirow" in rest else ""
            am = axis_re.search(axis_cell)
            if am and "---" not in axis_cell:
                entry["axis"] = tuple(
                    int(x.replace("{", "").replace("}", "").strip())
                    for x in am.group(1).split(r"\\"))
            else:
                entry["axis"] = None
            th = re.search(r"\{\$(\d+)\^\\circ\$\}", rest)
            entry["theta"] = int(th.group(1)) if th else 0
        rows[(int(pair), ab)] = entry
    assert len(rows) == 24, f"atlas.tex 2T block parsed to {len(rows)} rows, not 24"
    return rows


def read_tabular(text, label):
    """The body lines of the tabular that carries ``label``."""
    i = text.index(label)
    head = text.rindex(r"\begin{tabular}", 0, i)
    tail = text.index(r"\end{tabular}", head)
    return [l.strip() for l in text[head:tail].splitlines() if l.strip().endswith(r"\\")]


def cells(line):
    return [c.strip() for c in line.rsplit(r"\\", 1)[0].split("&")]


def read_ledger():
    """Table 5.2's rows, keyed by their (stripped) row label."""
    text = (DATA / "randomized_ledger.tex").read_text()
    out = {}
    for line in read_tabular(text, r"\label{tab:implementation-ledger}"):
        c = cells(line)
        if len(c) != 6 or c[0].startswith(r"\multicolumn"):
            continue
        out[c[0].replace(r"\quad", "").strip()] = c[1:]
    return out


def read_coin_words():
    """Table D.1's coin words, one list per solid."""
    text = (DATA / "randomized_ledger_appendix.tex").read_text()
    out = {}
    for line in read_tabular(text, r"\label{tab:decker-vs-coin}"):
        c = cells(line)
        m = re.match(r"(\w+) \((\d+)\)", c[0])
        if not m:
            continue
        words = [] if c[1].startswith("---") else [w.strip() for w in c[1].split(",")]
        out[m.group(1).lower()] = {"V": int(m.group(2)), "words": words}
    return out


def read_pose():
    """Table D.3's reorientation and outcome columns, one row per solid."""
    text = (PAPER / "bsc-thesis.tex").read_text()
    out = {}
    for line in read_tabular(text, r"\label{tab:decker-pose}"):
        c = cells(line)
        if len(c) != 5 or not c[0][:1].isupper():
            continue
        out[c[0].lower()] = {"R": c[3], "outcomes": c[4]}
    return out


def read_wire_counts():
    """The ancilla count of each Decker circuit, counted off its own figure:
    one ``\\lstick`` per wire, the data wire among them."""
    out = {}
    for name in SOLIDS:
        src = (FIGURES / f"circuit_dec_{SHORT[name]}.tex").read_text()
        out[name] = src.count(r"\lstick")
    return out


def read_frames():
    """The three token lists of ``recipe_tet_frames.tex``."""
    src = (FIGURES / "recipe_tet_frames.tex").read_text()
    body = src[src.index(r"\begin{document}"):]
    return [tuple(int(x) for x in m)
            for m in re.findall(r"\\MiniTet\{[\d.]+\}\{(\d)\}\{(\d)\}\{(\d)\}\{(\d)\}", body)]


def read_qcircuit(text):
    """The ``\\Qcircuit`` body, brace-balanced, whitespace normalised.

    Comment lines go first: both files talk *about* \\Qcircuit above the
    block they draw with it."""
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))
    i = text.index(r"\Qcircuit")
    depth, j = 0, text.index("{", i)
    k = j
    while True:
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    return " ".join(text[i:k + 1].split())


# --- the geometry ----------------------------------------------------------

def bloch(u):
    """The SO(3) rotation a unitary induces on the Bloch sphere."""
    return np.array([[0.5 * np.trace(PAULI[a] @ u @ PAULI[b] @ u.conj().T).real
                      for b in range(3)] for a in range(3)])


def permutation(u, verts):
    """Which vertex each vertex is carried to, 1-based."""
    R = bloch(u)
    out = []
    for k in range(len(verts)):
        w = R @ verts[k]
        d = np.linalg.norm(verts - w, axis=1)
        assert d.min() < LOOSE, "conjugation left the vertex set"
        out.append(int(np.argmin(d)) + 1)
    return tuple(out)


def reading(u, E):
    """What outcome k reads as once u is prepended to the data wire, 1-based.

    Prepending u makes the effective effects u^dagger E_k u = E_{g^{-1}(k)},
    so outcome k fires vertex g^{-1}(k).  Computed from the effects and not by
    inverting ``permutation``, so that the two can be checked against each
    other (check 5) and against the rebuilt circuit (check 9)."""
    out = []
    for k in range(len(E)):
        Ek = u.conj().T @ E[k] @ u
        d = [np.abs(Ek - E[j]).max() for j in range(len(E))]
        assert min(d) < LOOSE, "conjugation left the effect set"
        out.append(int(np.argmin(d)) + 1)
    return tuple(out)


def inverse(perm):
    """The inverse of a 1-based permutation tuple."""
    inv = [0] * len(perm)
    for k, v in enumerate(perm):
        inv[v - 1] = k + 1
    return tuple(inv)


def rotation(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def axis_of(u):
    """(integer axis, angle in degrees) of a Bloch rotation, sign fixed by
    reproducing the rotation rather than by the eigensolver's arbitrary
    choice."""
    R = bloch(u)
    deg = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if deg < LOOSE:
        return None, 0.0
    ev, evec = np.linalg.eig(R)
    a = np.real(evec[:, int(np.argmin(np.abs(ev - 1)))])
    a = a / np.max(np.abs(a))
    n = tuple(int(round(x)) for x in a)
    assert np.allclose(np.array(n), a, atol=LOOSE), f"axis {a} is not integral"
    if not np.allclose(rotation(n, deg), R, atol=LOOSE):
        n = tuple(-x for x in n)
        assert np.allclose(rotation(n, deg), R, atol=LOOSE), "axis sign undetermined"
    return n, deg


def axis_name(n, verts):
    r"""An integer axis, named by the vertex it passes through -- so the
    rotation table's axis column and the sphere's chips read as one object.
    Falls back to the Pauli axes for the three half-turns."""
    if n is None:
        return "---"
    v = np.asarray(n, float)
    v = v / np.linalg.norm(v)
    for k in range(len(verts)):
        if np.allclose(v, verts[k], atol=LOOSE):
            return r"$\hat n_%d$" % (k + 1)
        if np.allclose(v, -verts[k], atol=LOOSE):
            return r"$-\hat n_%d$" % (k + 1)
    for lab, ax in (("x", (1, 0, 0)), ("y", (0, 1, 0)), ("z", (0, 0, 1))):
        if np.allclose(v, np.array(ax, float), atol=LOOSE):
            return r"$\hat %s$" % lab
        if np.allclose(v, -np.array(ax, float), atol=LOOSE):
            return r"$-\hat %s$" % lab
    raise AssertionError(f"axis {n} is neither a vertex axis nor a Pauli axis")


def factored(E):
    r"""E_k in the form the page prints, E_k = (sqrt3/12) M_k, with each entry
    of M_k recognised rather than typed.  Factoring is Table E.1's own move and
    it buys a whole type size: the factored 2x2 sets at 10pt in the width the
    nested 1/4[1 + 1/sqrt3, ...] form needs at 9pt."""
    s = np.sqrt(3)
    catalogue = [(s + 1, r"\sqrt3+1"), (s - 1, r"\sqrt3-1"),
                 (1 + 1j, "1+i"), (1 - 1j, "1-i"),
                 (-1 + 1j, "-1+i"), (-1 - 1j, "-1-i")]
    M = E * 12 / s
    out = []
    for a in range(2):
        row = []
        for b in range(2):
            hits = [t for v, t in catalogue if abs(M[a, b] - v) < 1e-12]
            assert len(hits) == 1, f"entry {M[a, b]} is not one of the six atoms"
            row.append(hits[0])
        out.append(row)
    return out


def naimark_effects(povm, Ug=None, reorient=True):
    r"""Decker's tetrahedral dilation, rebuilt from its own factorisation:
    W = (1 (x) H) diag(1,1,1,i) (U_A (x) 1) CNOT_{data->anc} (1 (x) T-dagger),
    in the |ancilla, data> ordering, the ancilla starting in |0>.  Returns
    {(a, d): the 2x2 effective effect on the data qubit}.  ``reorient=False``
    drops the T-dagger, i.e. runs the solid box as published.
    """
    al = np.sqrt((3 + np.sqrt(3)) / 12)
    be = np.sqrt((3 - np.sqrt(3)) / 12)
    UA = np.sqrt(2) * np.array([[al, be], [be, -al]], complex)
    Td = np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex)
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    CNOT = np.zeros((4, 4), complex)               # data controls, ancilla flips
    for a in range(2):
        for d in range(2):
            CNOT[2 * (a ^ d) + d, 2 * a + d] = 1
    CS = np.diag([1, 1, 1, 1j]).astype(complex)    # ancilla controls, S on data

    W = np.kron(I2, Td) if reorient else np.eye(4, dtype=complex)
    if Ug is not None:
        W = W @ np.kron(I2, Ug)
    W = np.kron(I2, H) @ CS @ np.kron(UA, I2) @ CNOT @ W

    out = {}
    for a in range(2):
        for d in range(2):
            proj = np.zeros((4, 4), complex)
            proj[2 * a + d, 2 * a + d] = 1
            out[(a, d)] = (W.conj().T @ proj @ W)[np.ix_([0, 1], [0, 1])]
    return out


def naimark_outcomes(povm, Ug=None):
    r"""{(a, d): vertex} -- which of our effects each computational outcome
    of the rebuilt circuit is, with ``Ug`` (if any) prepended to the data wire.

    Rebuilding it is what makes step 3's ``bits`` column a reproduction of
    Table D.4's tetrahedron row rather than a restatement of it, and band C's
    reading column a reproduction of what the circuit does with a word in its
    dashed box.
    """
    E = povm["elements"]
    out, worst = {}, 0.0
    for ad, Ek in naimark_effects(povm, Ug).items():
        d_k = [np.abs(Ek - E[k]).max() for k in range(4)]
        worst = max(worst, min(d_k))
        out[ad] = int(np.argmin(d_k)) + 1
    assert worst < 1e-12, f"the rebuilt circuit misses its effects by {worst:.1e}"
    assert sorted(out.values()) == [1, 2, 3, 4], "the outcome key is not a bijection"
    return out, worst


# --- the twelve rotations --------------------------------------------------

def rotations(povm, group, gates):
    """One record per rotation of T: the two atlas words, the axis, the
    permutation (g, which vertex goes where) and the reading (g^{-1}, what
    outcomes 1234 read as with the word prepended -- the printed column).
    Pairing is derived (the two elements with the same permutation), not
    looked up."""
    verts = povm["vertices"]
    E = povm["elements"]
    U = group["unitaries"]
    seqs = [str(s) for s in group["dij_sequences"]]
    tab = gate_table(gates)
    atlas = read_atlas_2t()

    perms = {}
    for i in range(len(U)):
        perms.setdefault(permutation(U[i], verts), []).append(i)
    assert len(perms) == 12, f"{len(perms)} distinct permutations, not 12"
    assert all(len(v) == 2 for v in perms.values()), "some rotation is not two-to-one"

    out = []
    for pair in range(1, 13):
        wa, wb = atlas[(pair, "a")]["word"], atlas[(pair, "b")]["word"]
        ia = next(i for i in range(len(U)) if word_latex(seqs[i]) == wa)
        ib = next(i for i in range(len(U)) if word_latex(seqs[i]) == wb)
        perm = permutation(U[ia], verts)
        reads = reading(U[ia], E)
        n, deg = axis_of(U[ia])
        out.append({"pair": pair, "ia": ia, "ib": ib, "a": wa, "b": wb,
                    "perm": perm, "reads": reads, "axis": n, "deg": deg,
                    "axis_tex": axis_name(n, verts),
                    "atlas_axis": atlas[(pair, "a")]["axis"],
                    "atlas_theta": atlas[(pair, "a")]["theta"]})
    return out, perms


# --- the foot strip --------------------------------------------------------

def series():
    """One row per solid, every cell read off a table the page cites."""
    ledger = read_ledger()
    coins = read_coin_words()
    pose = read_pose()
    wires = read_wire_counts()

    rows = []
    for name in SOLIDS:
        i = SOLIDS.index(name)
        npz = np.load(DATA / f"povm_{name}.npz", allow_pickle=True)
        V = int(npz["num_vertices"])
        group = str(npz["symmetry_group"])
        register = int(re.search(r"\((\d+)\)", ledger["Naimark register"][i]).group(1))
        dead = int(ledger["Dead outcomes"][i])
        R = pose[name]["R"]
        exact_over = "over Clifford" in R
        words = coins[name]["words"]
        phis = sum(1 for w in words if r"\Phi" in w)

        if not words:
            coin = "none: no antipodes"
        elif ledger["Field demand"][i].startswith("---"):
            coin = r"%d axes --- \textbf{exact}" % len(words)
        elif phis == 0 and len(words) > 4:
            coin = r"%d axes, no $\Phi$" % len(words)
        elif phis:
            coin = r"%d axes, %d with one $\Phi$" % (len(words), phis)
        else:
            coin = "%d axes" % len(words)

        rows.append({
            "name": name.capitalize(), "V": V,
            "group": {"2T": r"\TwoT", "2O": r"\TwoO", "2I": r"\TwoI"}[group],
            "anc": wires[name] - 1, "dead": dead, "register": register,
            "reorient": (r"$T^\dagger$, over Clifford${}+T$" if exact_over
                         else "no thesis gate set"),
            "relabel": "identity" if r"k \mapsto k" in pose[name]["outcomes"] else "permuted",
            "coin": coin, "coin_words": words, "R": R,
            "ledger_V": int(ledger[r"Vertices $V$"][i]),
            "ledger_axes": ledger["Coin words, one per axis"][i],
        })
    return rows


# --- the twelve checks -----------------------------------------------------

def verify(povm, group, gates, rots, perms, rows, frames, key, pk, ab):
    """Every assertion runs before a byte is written; a failure writes nothing."""
    E = povm["elements"]
    verts = povm["vertices"]
    U = group["unitaries"]
    tab = gate_table(gates)
    atlas = read_atlas_2t()
    report = []

    # 1. the POVM axioms, and the SIC condition
    assert np.abs(sum(E) - I2).max() < TOL, "the effects do not resolve the identity"
    assert max(abs(np.trace(E[k]) - 0.5) for k in range(4)) < TOL, "trace is not 1/2"
    assert all(np.linalg.matrix_rank(E[k], tol=1e-9) == 1 for k in range(4)), "not rank one"
    ov = [abs(4 * np.trace(E[j] @ E[k]).real) for j in range(4) for k in range(j + 1, 4)]
    assert len(ov) == 6 and max(abs(o - 1 / 3) for o in ov) < 1e-14, "not a SIC"
    report.append(f"  1. axioms: sum E = 1, Tr = 1/2, rank 1, six overlaps 1/3 "
                  f"(worst {max(abs(o - 1/3) for o in ov):.1e})")

    # 2. the printed factored matrices
    s = np.sqrt(3)
    atoms = {r"\sqrt3+1": s + 1, r"\sqrt3-1": s - 1, "1+i": 1 + 1j, "1-i": 1 - 1j,
             "-1+i": -1 + 1j, "-1-i": -1 - 1j}
    worst = 0.0
    for k in range(4):
        M = np.array([[atoms[t] for t in row] for row in factored(E[k])])
        worst = max(worst, np.abs((s / 12) * M - E[k]).max())
    assert worst < TOL, f"printed matrices differ from the npz by {worst:.1e}"
    report.append(f"  2. printed matrices vs npz: worst {worst:.1e}")

    # 3. p(k), printed to four places, and the top-left entry
    rho = np.diag([1, 0]).astype(complex)
    p = np.array([np.trace(E[k] @ rho).real for k in range(4)])
    assert abs(p.sum() - 1) < TOL, "the probabilities do not sum to 1"
    assert max(abs(p[k] - E[k][0, 0].real) for k in range(4)) < TOL, \
        "p(k) is not the top-left entry of E_k"
    assert [f"{x:.4f}" for x in p] == pk, f"printed decimals {pk} are not {p}"
    report.append(f"  3. p(k) = Tr[E rho] = (E)_00 = {', '.join(pk)}")

    # 4. the inversion
    for r in (np.array([0.0, 0.0, 1.0]), np.array([0.31, -0.52, 0.17])):
        rho_r = 0.5 * (I2 + sum(r[a] * PAULI[a] for a in range(3)))
        pr = np.array([np.trace(E[k] @ rho_r).real for k in range(4)])
        rec = 3 * sum(pr[k] * verts[k] for k in range(4))
        assert np.abs(rec - r).max() < 1e-14, f"inversion missed {r} by {np.abs(rec-r).max():.1e}"
    report.append("  4. r = 3 sum p(k) n_k recovers z-hat and an off-axis vector")

    # 5. the double cover
    assert len(perms) == 12, "not twelve rotations"
    for rec in rots:
        assert np.allclose(U[rec["ia"]], -U[rec["ib"]]), \
            f"pair {rec['pair']}: the two words are not U and -U"
        assert permutation(U[rec["ia"]], verts) == permutation(U[rec["ib"]], verts)
        assert rec["reads"] == inverse(rec["perm"]) == reading(U[rec["ib"]], E), \
            f"pair {rec['pair']}: reading {rec['reads']} is not the inverse of {rec['perm']}"
    assert sorted(r["reads"] for r in rots) == sorted(r["perm"] for r in rots), \
        "inverting the twelve readings did not permute the twelve rows"
    report.append("  5. twelve rotations, each two elements, each pair U and -U; "
                  "every printed reading is the inverse of its rotation")

    # 6. the axes
    for rec in rots:
        assert rec["axis_tex"].startswith("$") or rec["pair"] == 1, "unnamed axis"
        if rec["axis"] is None:
            assert rec["pair"] == 1, "only the identity may have no axis"
            continue
        assert abs(rec["deg"] - rec["atlas_theta"]) < 1e-6, \
            f"pair {rec['pair']}: {rec['deg']:.1f} deg against atlas {rec['atlas_theta']}"
        a = np.array(rec["axis"], float)
        b = np.array(rec["atlas_axis"], float)
        assert np.allclose(a / np.linalg.norm(a), b / np.linalg.norm(b), atol=LOOSE), \
            f"pair {rec['pair']}: axis {rec['axis']} against atlas {rec['atlas_axis']}"
        assert np.allclose(np.abs(a / np.linalg.norm(a)),
                           np.abs(np.array([1, 0, 0])), atol=LOOSE) or \
            any(np.allclose(a / np.linalg.norm(a), sgn * verts[k], atol=LOOSE)
                for k in range(4) for sgn in (1, -1)) or \
            any(np.allclose(a / np.linalg.norm(a), sgn * np.eye(3)[j], atol=LOOSE)
                for j in range(3) for sgn in (1, -1)), "axis is neither vertex nor Pauli"
    report.append("  6. every axis is a vertex or Pauli axis, and is the atlas's own")

    # 7. the words, character for character, and multiplied back out
    seqs = [str(x) for x in group["dij_sequences"]]
    for rec in rots:
        for side, idx in (("a", rec["ia"]), ("b", rec["ib"])):
            assert word_latex(seqs[idx]) == atlas[(rec["pair"], side)]["word"] == rec[side], \
                f"pair {rec['pair']}{side}: {rec[side]} is not atlas.tex's word"
            assert np.abs(word_matrix(seqs[idx], tab) - U[idx]).max() < 1e-12, \
                f"pair {rec['pair']}{side}: the word does not multiply out to its unitary"
        assert atlas[(rec["pair"], "a")]["magic"] == 0, "a 2T word with magic cost"
        assert atlas[(rec["pair"], "a")]["depth"] <= 2, "a 2T word deeper than 2"
    report.append("  7. all 24 words match atlas.tex and multiply out; depth <= 2, Phi = 0")

    # 8. the algebra printed in words
    def U_of(word):
        return U[next(i for i, s in enumerate(seqs) if s == word)]
    F, Fd, X, Z, XZ = (U_of("F"), U_of("F\u2020"), U_of("X"), U_of("Z"), U_of("X Z"))
    assert np.allclose(F @ F, -Fd), "FF is not -F-dagger"
    assert np.allclose(F @ F @ F, -I2), "F^3 is not -1"
    assert np.allclose(X @ Z, -Z @ X), "X and Z commute"
    assert np.allclose(X @ Z, XZ), "XZ is not pair 10's a-word"
    pX, pZ = permutation(X, verts), permutation(Z, verts)
    comp = tuple(pX[pZ[k] - 1] for k in range(4))
    assert comp == permutation(XZ, verts) == (3, 4, 1, 2), "perm(X) after perm(Z) is not perm(XZ)"
    assert reading(XZ, E) == reading(Z @ X, E) == (3, 4, 1, 2), "pair 10's reading is order-sensitive"
    FX = U_of("F X")
    rF, rX = reading(F, E), reading(X, E)
    assert reading(FX, E) == tuple(rX[rF[k] - 1] for k in range(4)), \
        "reads(FX) is not reads(X) after reads(F) -- the readings compose in reverse"
    assert reading(FX, E) != tuple(rF[rX[k] - 1] for k in range(4)), \
        "reads(FX) composes in operator order -- then the anti-order caveat is void"
    report.append("  8. FF = -F-dagger, F^3 = -1, XZ = -ZX, perm(X)perm(Z) = perm(XZ) = 3412; "
                  "reads(FX) = reads(X) after reads(F)")

    # 9. the outcome key
    key2, worst9 = naimark_outcomes(povm)
    assert key2 == key, "the outcome key moved between derivation and check"
    assert key == {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}, \
        f"the rebuilt circuit's key is {key}"
    assert ab == ["00", "01", "10", "11"], f"printed bits {ab}"
    keyF, _ = naimark_outcomes(povm, U_of("F"))
    assert [keyF[(a, d)] for a in (0, 1) for d in (0, 1)] == [1, 3, 4, 2], \
        "prepending F does not relabel outcomes by g-inverse"
    for rec in rots:                                  # the printed column, off the circuit
        key_g, _ = naimark_outcomes(povm, U[rec["ia"]])
        assert tuple(key_g[(a, d)] for a in (0, 1) for d in (0, 1)) == rec["reads"], \
            f"pair {rec['pair']}: the circuit reads {key_g} but the page prints {rec['reads']}"
    # T-dagger is the reorientation: the box as published measures our effects
    # turned by a rotation about z (Table D.3: R_z(45 deg), over Clifford+T),
    # and that turned copy is not ours.
    Td = np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex)
    box = naimark_effects(povm, reorient=False)
    for (a, d), Ek in box.items():
        assert np.abs(Ek - Td @ E[key[(a, d)] - 1] @ Td.conj().T).max() < 1e-12, \
            "without T-dagger the box does not measure our effects turned by T-dagger"
    assert min(np.abs(Ek - E[j]).max() for Ek in box.values() for j in range(4)) > 0.1, \
        "the box as published already measures our pose -- nothing to reorient"
    Rz = bloch(Td)
    assert abs(Rz[2, 2] - 1) < LOOSE and abs(np.trace(Rz) - (1 + 2 * np.cos(np.pi / 4))) < LOOSE, \
        "T-dagger's Bloch rotation is not a 45-degree turn about z"
    pose = read_pose()["tetrahedron"]["R"]
    assert "R_z(45" in pose and "over Clifford" in pose, f"Table D.3's tetrahedron row reads {pose!r}"
    report.append(f"  9. rebuilt Naimark circuit: 00,01,10,11 -> E1..E4 "
                  f"(residual {worst9:.1e}); with F -> 1,3,4,2; all twelve readings "
                  f"off the circuit; without T-dagger the box is ours turned 45 deg about z")

    # 10. the foot strip
    ledger = read_ledger()
    for i, row in enumerate(rows):
        assert row["V"] == row["ledger_V"], f"{row['name']}: V {row['V']} vs ledger"
        assert 2 ** (row["anc"] + 1) == row["register"], \
            f"{row['name']}: {row['anc']} ancillas cannot address {row['register']}"
        assert row["dead"] == row["register"] - row["V"], \
            f"{row['name']}: dead {row['dead']} but {row['register']} - {row['V']}"
        axes = row["ledger_axes"]
        if row["coin_words"]:
            assert int(axes) == len(row["coin_words"]), \
                f"{row['name']}: ledger says {axes} axes, Table D.1 lists {len(row['coin_words'])}"
        else:
            assert axes.startswith("---"), f"{row['name']}: coin words but no axes"
    assert [r["anc"] for r in rows] == [1, 2, 2, 3, 4], \
        f"ancilla counts {[r['anc'] for r in rows]} -- an easy count to get wrong"
    assert [r["dead"] for r in rows] == [0, 2, 0, 4, 12], "dead outcomes moved"
    assert rows[0]["relabel"] == "identity" and all(r["relabel"] == "permuted" for r in rows[1:]), \
        "the tetrahedron is no longer the one solid with the identity relabelling"
    # Table D.4's permutation column has a machine-readable twin in the
    # fragment, one % key comment line per solid, which is what is read
    # here; the card build asserts the two copies equal.
    labels = (DATA / "randomized_labels.tex").read_text()
    tet_line = next(l for l in labels.splitlines() if l.startswith("% key tetrahedron:"))
    top, bot = tet_line.split(":", 1)[1].split("->")
    top = [int(x) for x in top.split()]
    bot = [int(x) for x in bot.split()]
    assert bot == [t + 1 for t in top], "Table D.4's tetrahedron key is no longer the identity"
    assert "exact" in rows[1]["coin"] and not any("exact" in r["coin"] for r in rows if r is not rows[1]), \
        "the octahedron is no longer the one exact coin"
    report.append("  10. foot strip: V, register, ancillas, dead, coin axes, "
                  "reorientation and relabelling all read off Tables 5.2, D.1, D.3, D.4")

    # 11. the token frames
    assert len(frames) == 3, f"{len(frames)} frames, not 3"
    g_by_pair = {r["pair"]: r["perm"] for r in rots}
    turn = g_by_pair[2]                                   # F, 1234 -> 1423
    assert turn == (1, 4, 2, 3), f"pair 2's permutation is {turn}"
    want = []
    g = (1, 2, 3, 4)
    for _ in range(3):
        inv = [0] * 4
        for k in range(4):
            inv[g[k] - 1] = k + 1                         # numeral at site m is g^{-1}(m)
        want.append(tuple(inv))
        g = tuple(turn[g[k] - 1] for k in range(4))       # compose one more F
    assert frames == want, f"token frames {frames} against g-inverse {want}"
    assert frames[1][3] == 2, "numeral 2 does not land bottom-left after one F"
    by_pair_reads = {r["pair"]: r["reads"] for r in rots}
    assert frames[1] == by_pair_reads[2] and frames[2] == by_pair_reads[3], \
        "read site by site, the frames are not pair 2's and pair 3's printed rows"
    report.append(f"  11. token frames {frames} are g-inverse, frame by frame, "
                  f"and are pair 2's and pair 3's printed rows")

    # 12. the facts the prose hand-types
    #
    # recipe_tet_body.tex is the one hand-written file on the page, and what it
    # types are permutations, pair numbers, axes and angles -- exactly the class
    # of fact that goes wrong silently.  Every literal below is rebuilt from the
    # group first and then required to be present verbatim, so the sentences can
    # be rewritten freely while the facts inside them cannot drift.
    prose = " ".join(l for l in (FIGURES / "recipe_tet_body.tex").read_text().splitlines()
                     if not l.lstrip().startswith("%"))
    prose = " ".join(prose.split())
    quoted = []

    def says(s, why):
        assert s in prose, f"recipe_tet_body.tex no longer says {s!r} -- {why}"
        quoted.append(s)

    by_pair = {r["pair"]: r for r in rots}

    # the banner's operator-order declaration
    assert np.allclose(word_matrix("F X", tab), U_of("F") @ U_of("X")), \
        "the atlas word FX is not F after X"
    says(r"$\mathbf{FX}$ applies $\mathbf X$ first", "operator order")

    # band A's note counts the vertices
    assert len(verts) == 4, "the tetrahedron has stopped having four vertices"
    says("$V=4$", "V")

    # band B: the bit order, and the two clauses the exactness claim needs
    assert key == {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}, "bit order"
    says("top wire first", "the bits are (ancilla, data), Table D.4's own order")
    says(r"\textit{deterministic}", "the load-bearing word of Theorem 1's second sentence")
    says("only the octahedral POVM admits an exact implementation",
         "Theorem 1's positive half, which must survive every trim")

    # band C, first rider: F turns 120 degrees about vertex 1's axis, twice is
    # pair 3, and the pairs that turn 180 degrees are exactly 4, 5 and 10
    assert abs(by_pair[2]["deg"] - 120) < 1e-6 and by_pair[2]["axis_tex"] == r"$\hat n_1$", \
        f"pair 2 is {by_pair[2]['deg']:.3f} deg about {by_pair[2]['axis_tex']}"
    says(r"$120^\circ$ about vertex~1's axis", "pair 2's angle and axis")
    assert any(np.allclose(U_of("F") @ U_of("F"), s * U[by_pair[3]["ia"]]) for s in (1, -1)), \
        "F.F is not pair 3's rotation"
    says("twice is pair~3", "F.F is pair 3")
    half = sorted(r["pair"] for r in rots if abs(r["deg"] - 180) < 1e-6)
    assert half == [4, 5, 10], f"the half-turns are pairs {half}, not 4, 5, 10"
    says(r"Pairs 4, 5, 10 turn $180^\circ$", "the three half-turns")
    says(r"$\mathbf F^3=-\Id$", "F cubed")

    # head 2 and band B: T-dagger is the reorientation (checked off the
    # circuit in 9), and the circuit printed is a dilation
    says(r"$T^\dagger$ reorients", "T-dagger is the reorientation, Table D.3's tetrahedron row")
    says("this one included", "the dilation printed just above is the one Theorem 1 covers")
    says("the coin below", "band B's pointer at the strip's last column; its head must say coin")
    oct_words = next(r for r in rows if r["name"] == "Octahedron")["coin_words"]
    printed_coin = ", ".join(re.sub(r"\\mathbf\{(\w)\}", r"\\mathbf \1", w.strip("$"))
                             for w in oct_words)
    says(r"$\{%s\}$" % printed_coin, "band B's printed coin is Table D.1's octahedron row, word for word")
    says(r"the $\bra{0}E_k\ket{0}$ column above", "band B names the master table's last column by its head")

    # head 5 defines the rotation table's last column as the READING, g-inverse
    says("then read as its row", "head 5 is what makes the column mean g-inverse and not g")

    # band C, second rider: pair 2's two words and their common reading
    assert "".join(str(x) for x in by_pair[2]["perm"]) == "1423", "pair 2's image moved"
    assert "".join(str(x) for x in by_pair[2]["reads"]) == "1342", "pair 2's reading moved"
    says(r"$1234\mapsto1342$", "pair 2's reading, g-inverse")
    assert np.allclose(U[by_pair[2]["ib"]], -U[by_pair[2]["ia"]]), "pair 2 is not +-U"
    says(r"$\mathbf F^\dagger\mathbf F^\dagger=-\mathbf F$", "pair 2's b-word")

    # band C, third rider: pair 10 is the half-turn about y-hat, and XZ = -ZX
    assert by_pair[10]["axis_tex"] == r"$\hat y$" and abs(by_pair[10]["deg"] - 180) < 1e-6, \
        f"pair 10 is {by_pair[10]['deg']:.3f} deg about {by_pair[10]['axis_tex']}"
    says(r"pair~10's half-turn about $\hat y$", "pair 10's axis")
    says(r"$\mathbf{XZ}=-\mathbf{ZX}$", "X and Z anticommute")

    # band C, fourth rider: the relabelling direction, the likeliest error here
    says(r"outcomes $1,2,3,4$ read as vertices $1,3,4,2$",
         "prepending F relabels outcomes by g-inverse")
    report.append(f"  12. {len(quoted)} hand-typed facts in recipe_tet_body.tex "
                  f"(permutations, pair numbers, axes, angles, bit order) verified")

    # and the copied circuit
    here = read_qcircuit((FIGURES / "recipe_tet_circuit.tex").read_text())
    there = read_qcircuit((PAPER / "bsc-thesis.tex").read_text()[
        (PAPER / "bsc-thesis.tex").read_text().index(r"\label{fig:tet_povm_circuit}") - 4000:])
    assert here == there, "recipe_tet_circuit.tex has drifted from Figure 4.1"
    report.append("  --. recipe_tet_circuit.tex is Figure 4.1's block, whitespace aside")

    return report


# --- emitting --------------------------------------------------------------

def master_table(povm, key, pk, ab):
    E = povm["elements"]
    verts = povm["vertices"] * np.sqrt(3)
    L = [r"\newcommand{\RecipeMasterTable}{{\small",
         r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{0.90}%",
         r"\begin{tabular}{@{}c l c c r@{}}", r"\toprule",
         r"$k$ & $\hat n_k\ (\times\tfrac{1}{\sqrt3})$ & "
         r"$E_k\ (\times\tfrac{\sqrt3}{12})$ & bits & $\bra{0}E_k\ket{0}$ \\", r"\midrule"]
    for k in range(4):
        n = ",".join("%+d" % round(x) for x in verts[k])
        m = factored(E[k])
        L.append(r"%d & $(%s)$ & $\pmat{%s & %s \\ %s & %s}$ & $%s$ & $%s$ \\"
                 % (k + 1, n, m[0][0], m[0][1], m[1][0], m[1][1], ab[k], pk[k]))
    L += [r"\bottomrule", r"\end{tabular}}}"]
    return L


def circuit_params():
    al = np.sqrt((3 + np.sqrt(3)) / 12)
    be = np.sqrt((3 - np.sqrt(3)) / 12)
    # One line per fact.  The column is 136pt and the two definitions do not
    # share a line without \sloppy stretching the spaces to a hole; broken here
    # they set flush left at the same four lines and the same height.
    return [r"\newcommand{\RecipeCircuitParams}{%",
            r"$U_A=\sqrt2\bigl(\begin{smallmatrix}\alpha & \beta\\ \beta & "
            r"-\alpha\end{smallmatrix}\bigr)$\par",
            r"$T^\dagger=\mathrm{diag}(1,e^{-i\pi/4})$\par",
            r"\vskip1pt $\alpha=\sqrt{(3{+}\sqrt3)/12}=%.4f$\par" % al,
            r"$\beta=\sqrt{(3{-}\sqrt3)/12}=%.4f$}" % be]


def rotation_table(rots):
    L = [r"\newcommand{\RecipeRotationTable}{{\small",
         r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{0.88}%",
         r"\begin{tabular}{@{}r l l c c@{}}", r"\toprule",
         r"pair & $a$-word & $b$-word & axis & $1234\mapsto$ \\", r"\midrule"]
    # The last column is the READING, g-inverse: with the word prepended,
    # outcomes 1234 read as these vertices (head 5 says so in words; the
    # frames show it site by site).  Not the rotation's action on the vertices.
    for r in rots:
        L.append(r"%d & $%s$ & $%s$ & %s & %s \\"
                 % (r["pair"], r["a"], r["b"], r["axis_tex"],
                    "".join(str(x) for x in r["reads"])))
    L += [r"\bottomrule", r"\end{tabular}}}"]
    return L


def series_strip(rows):
    L = [r"\newcommand{\RecipeSeriesStrip}{{\footnotesize",
         r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{0.88}%",
         r"\begin{tabular}{@{}l c c c c l l l@{}}", r"\toprule",
         r"The same six steps run & $V$ & group & anc. & dead "
         r"& reorientation exact? & relabelling & "
         r"\hyperref[tab:decker-vs-coin]{coin over axes} \\",
         r"\midrule"]
    for i, r in enumerate(rows):
        name = r"\textbf{%s}" % r["name"] if i == 0 else r["name"]
        V = r"\textbf{%d}" % r["V"] if i == 0 else "%d" % r["V"]
        L.append(r"%s & %s & $%s$ & %d & %d & %s & %s & %s \\"
                 % (name, V, r["group"], r["anc"], r["dead"],
                    r["reorient"], r["relabel"], r["coin"]))
    L += [r"\bottomrule", r"\end{tabular}}}"]
    return L


HEADER = r"""% Auto-generated by code/_build_recipe_figure.py -- DO NOT EDIT.
% The four data blocks of Figure 1.1 (paper/figures/recipe_tet_body.tex
% \input's this as ../code/data/recipe_tet_data and calls them).  Every number is derived from code/data/*.npz and checked back
% against Tables A.1, 5.2, D.1, D.3 and D.4 before this file is written; see
% the builder's docstring for the twelve assertions.
% Requires: booktabs, mathtools, hyperref, and the thesis macros \Id, \pmat,
% \bra, \ket, \TwoT, \TwoO, \TwoI.
% Regenerate with `cd code && uv run _build_recipe_figure.py`.
"""


def build():
    povm, group, gates = load()
    rots, perms = rotations(povm, group, gates)
    rows = series()
    frames = read_frames()

    key, _ = naimark_outcomes(povm)
    inverse = {v: k for k, v in key.items()}
    ab = ["%d%d" % inverse[k + 1] for k in range(4)]
    p = [np.trace(povm["elements"][k] @ np.diag([1, 0]).astype(complex)).real
         for k in range(4)]
    pk = [f"{x:.4f}" for x in p]

    report = verify(povm, group, gates, rots, perms, rows, frames, key, pk, ab)

    L = [HEADER.rstrip(), ""]
    L += master_table(povm, key, pk, ab) + [""]
    L += circuit_params() + [""]
    L += rotation_table(rots) + [""]
    L += series_strip(rows)
    text = "\n".join(L) + "\n"

    # The prose names two emitted heads; both must be what it says they are.
    strip_head = next(l for l in series_strip(rows) if "reorientation exact?" in l)
    assert "coin" in strip_head, "band B says 'the coin below' but the strip's last head has no coin in it"
    master_head = next(l for l in master_table(povm, key, pk, ab) if "bits" in l)
    assert r"$\bra{0}E_k\ket{0}$" in master_head, "the master table's last head moved"
    # The emitted reading column, row by row, against the verified records --
    # so that printing the rotation g instead of the reading g^{-1} cannot
    # slip past the checks, which otherwise test records and not text.
    printed = {}
    for line in rotation_table(rots):
        m = re.match(r"^(\d+) & .* & (\d{4}) \\\\$", line)
        if m:
            printed[int(m.group(1))] = tuple(int(c) for c in m.group(2))
    assert printed == {r["pair"]: r["reads"] for r in rots}, "the emitted reading column is not the readings"
    assert printed[2] == (1, 3, 4, 2), "pair 2's printed row is not 1342, which the prose says it is"
    report.append("  --. the strip's last head says coin; the master table's last head is <0|E_k|0>; "
                  "the emitted reading column is the readings, pair 2 = 1342")
    return text, report


def main(argv=()):
    text, report = build()
    OUT.write_text(text)
    print(f"wrote {OUT}")
    print(f"  {len(text.splitlines())} lines, four blocks")
    for line in report:
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
