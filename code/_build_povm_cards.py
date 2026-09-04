r"""Build ``code/data/card_{tet,oct,cube,icos,dodec}_data.tex`` -- the data of the
five Appendix D cards, "the *X* POVM, end to end".

Each card works one Platonic solid POVM end to end on one page, and labels
rather than narrates: Figure 1.1's own header strip, four objects, one label
per object, and a pointer per band -- 508 words over the five cards.

Every number on all five is already owned by some other generator -- the
vertices by ``povm_properties``, the effects by ``export_numpy``, the coin words
and kappa by ``randomized_implementations``, Decker's circuits by
``randomized_decker``, the header strip by ``_build_recipe_figure`` (Figure 1.1)
-- so nothing here is typed.  What this module does is assemble them, check each
one back against the very table the card cites, and refuse to write if any check
fails.

What it emits per solid, as ``\newcommand``s that
``paper/figures/card_<solid>_body.tex`` uses:

``\CTitle``            "The <adjective> POVM, end to end".
``\CStrip``            THE HEADER STRIP: ``\RecipeSeriesStrip``'s own column
                       heads and this solid's own row, minus the name column,
                       plus one spanned footer carrying the exactness
                       conjunction in nine words.  A reader who has met
                       Figure 1.1 recognises the row instead of reading it
                       (check 30).
``\CHeadA..D``         the four band heads;
``\CPtrStrip``,        the pointer column -- ten live \ref's per card (nine on
``\CPtrA..D``          the tetrahedron), apparatus rather than prose, and what
                       pays for the cuts (check 34).
``\CardVertexTable``   V <= 8: k, the Bloch vector, the effect in factored form,
                       and <0|E_k|0> -- headed by the matrix element, never by
                       p(k), so the column names its own state.
``\CardVertexBlockA``  V >= 12: the same table without the effect column, split
``\CardVertexBlockB``  into two blocks of V/2 that set side by side.
``\CIdent``            one line: the identities the vertex set satisfies, then
                       the one thing about THIS solid a reader should recognise.
``\CIdentB``           the A_5 pair only: a worked E_1, nested under the two
                       vertex blocks at zero words, for the covariance rule to
                       act on (check 33).
``\CParams``           the circuit's parameters, ONE FORMULA PER LINE, at
                       \small: U_R in closed form where the drawing does not
                       already name it, the g^{-1} rider, the CNOT kets, the
                       Fourier block, and Mtilde-dagger under its one
                       gloss -- the noun, and the box that implements it.
``\CardSide``          a centred stack of short atoms in the slack beside the
                       circuit: U_A, alpha, beta (and S, H on the tetrahedron),
                       or B and C on the A_5 pair.  Zero words.
``\CardKey``           the register key as a rectangle: rows the upper
                       (inter-orbit) register, columns the lower (Fourier) one,
                       cells our vertex number, --- where the outcome is dead.
                       Its corner heads read `upper` and `lower`, so the
                       rectangle states its own bit convention.
``\CKappa``            the price of running the published circuit against our
                       list: kappa, the shot premium, and never a bias.
``\CRays``             the coin over axes, one atlas word per vertex axis, each
                       ray written |0>-vertex first.  Math only.
``\CCoin``             the protocol line -- draw, apply, align, measure Z -- or,
                       for the tetrahedron, the negative and its contrast with
                       the twirl.

THE CHECKS.  All of them run before a byte is written; a failure writes nothing.
``_povm_cards_controls.py`` shows each of them failing on one corrupted literal
(100 negative controls, all aborting, 0 stale).

 1. the POVM axioms -- sum E = 1, Tr E = 2/V, rank one -- the design strengths
    2/3/3/5/5, and the tetrahedron alone satisfying the SIC condition;
 2. the printed factored matrices against ``povm_<solid>.npz``, atom by atom,
    for V <= 8; for V >= 12 the one printed general form rebuilt for every k;
 3. <0|E_k|0> as (E_k)_00, as Tr[E_k |0><0|] and as (1 + z_k)/V, to the four
    decimals actually printed -- and the column summing to exactly 1;
 5. covariance: E_g(k) = U_g E_k U_g^dagger over every g and every k;
 6. every printed tuple parsed back to the npz under the printed radial scalar,
    and against ``povm_atlas.tex`` (Table E.1) cell for cell;
 7. every coin word a row of ``atlas.tex`` that multiplies out to its rotation,
    with the printed Phi count;
 8. bold by the claim: no bold symbol in a rotation claim, no plain symbol in an
    SU(2) claim, over every emitted string;
 9. the rebuilt Mtilde-dagger . iota, reoriented by Table D.3's R, reproducing
    ``npz['elements']`` for every live outcome, and reproducing none unreoriented;
10. the counts -- V, register 2^n, ancillae, dead -- agreeing across the npz,
    the card circuit's \lstick count, ``decker_circuit`` and the chapter head;
11. direction: the printed relabelling is g^{-1}, and "acts on" appears in no
    emitted fragment and in no body file; no italic run-in heads;
12. the card's literals, verbatim on the page (body + emitted): the title,
    the seven strip cells, the strip footer, the four band heads, the g^{-1}
    rider, the Naimark gloss, and the float's caption in bsc-thesis.tex;
13. the five circuits rebuilt and matched to the figure files;
14. the pose unchanged, by token diff against the frozen ``circuit_dec_*.tex``,
    admitting exactly the declared differences and no others, with both new
    gates OUTSIDE the solid \gategroup;
15. the key derived TWICE and required equal -- once from ``decker_vertices``
    and REORIENT, once parsed out of ``randomized_labels.tex``;
16. the dead set as a RULE -- {n : n mod 2^ell >= m} -- counts 0/2/0/4/12;
17. U_R = R^{-1} for Table D.3's R, symbolically, with D.3's sign convention;
    and the formula printed exactly once, on the card or in the drawing;
18. kappa and 1/kappa^2 character for character from ``randomized_labels.tex``;
19. the coin words in vertex-scan order, every ray antipodal, the LEFT-HAND
    index of every ray the |0> vertex, and the uniform mixture reproducing the
    printed effects; 19b the protocol line states the order it prices and the
    reversed order is built and required to miss the vertex set; 19c exactly
    one route per card to Appendix A's table, and the operator-order rider
    exactly where a composite word is printed; 19d the five draw notes,
    each backed and each pinned to its own card -- T derived by closing the
    icosahedron's coin words (order 12, and a subgroup of all three
    covariance groups), its SIGNED orbit of each card's seed giving the
    tetrahedron the cube's eight effect for effect, uniform counts on the
    octahedron, cube and icosahedron, and only 12 of the dodecahedron's 20
    vertices; no coin twirling; every verdict Table 5.2's own;
20. the alignment claim, and the tetrahedron printing its negative as head 4
    plus the contrast between the two randomized protocols;
21. the chip set and its legibility, recomputed from the panel's own projection
    at the emitted \PanelScale (2.47 on all five, Figure 1.1's own);
22. every \PVert coordinate coming from ``_povm_sphere_preamble.tex``;
23. nothing printed today vanishes: every math token of the five CURRENT
    captions, parsed from a frozen copy, still printed -- 23b respelt to the
    same value, 23c re-homed to the chapter head with both halves checked;
24. all five ``fig:dec_*_circuit`` labels retained exactly once;
26. the type floor: no fraction and no smallmatrix in a \footnotesize fragment
    -- so no numeral is set as a math script in the only two fragments that
    still set there (the strip, the pointer column).  It is NOT a claim about
    the whole page: at \small a first-level script prints 6.97pt, and the
    vertex table's \tfrac heads do.  26b the only sub-7pt glyph on the page is
    a Figure 1.1 strip literal whose own cell prints its base symbol upright;
27. the hand-typed LaTeX closed against the code, not against another string;
28. the chapter head's U_g sentence, self-policing; 28b D.2's opening sentence;
29. the printed surface parsed back OUT of the emitted file and compared to the
    object: every vertex row, the key with its corner heads and its empty cells
    counted against the dead set, every ray, the seven strip cells and the
    footer, head 3's wording, the identities line's trace and design strength,
    the alignment label, the worked E_1, the printed U_R as a rotation with its
    composition order, and kappa's pointer;
30. the strip's seven column heads and this solid's seven cells string-equal to
    ``\RecipeSeriesStrip``'s, after dropping its name column.  This is the
    recognition, made a check;
31. the shared literals byte-identical on all five, and the exactness footer
    printed exactly once per card;
32. the chapter head carries the unfactored A, numbered,
    Decker's sqrt(2/V) rescaling shown as its own factor and gamma, delta
    paired across the solids by name, before either A_5 card may be written
    -- A is printed nowhere else, and both cards point at it; 32b parses the
    display's prefactor and matrix body and the paragraph's four radicands
    back out and requires them to rebuild decker_circuit's rows;
33. the worked E_1 equals (1/V)(1 + n_1 . sigma) and the npz, its printed atoms
    parsed back out of the emitted file;
34. the pointer set: ten pointers per card, nine on the tetrahedron, plus the
    inline ones (Table D.3, and Equations (D.1), (D.2) on the A_5 pair), every one a
    label declared in ``bsc-thesis.tex`` or in a fragment it \inputs -- the
    ``.aux`` is a fallback only, so a fresh clone passes -- and Figure 1.1 on
    all five;
35. the hand-written BODY closes over the emitted data file: it \inputs its own
    solid's file, once, and calls every macro that file defines with content.
    Nothing else joins the two halves of a card: check 12 builds
    (emitted + body) and every literal it wants is already in the emitted half,
    so a body that stops calling ``\CStrip``, or \inputs another solid's data,
    would ship with every other check reporting pass;
36. every derived warning is a predicate on the value it qualifies: Table
    D.4's printed kappa must carry the sign of the kappa recomputed from the
    two vertex lists (18's own number), head 3's em-dash gloss rides on the
    dead count, and the two retired per-card kappa riders may not re-enter
    (the note above ``kappa_num`` has the reasons).  The noun is the
    estimator's, never eta's -- no card may print "calibration": that name
    belongs to eta, the calibration scalar (noiseless 1/3), while a card
    prices the estimator channel's kappa = 3 eta.  And the free pairing the
    kappa label prints ("Run U_R and read the key, or skip U_R and read
    Decker's own numbered list ...: kappa = 1, free") and the misread it
    prices (his circuit with our list, index for index) are both checked to
    be D.2's own claims.

Edit this builder, never ``code/data/card_*_data.tex``.  Only the layout lives
in ``paper/figures/card_<solid>_body.tex`` (``card_oct_body.tex`` is the
template; the other four declare its header binding); the caption lives in
``bsc-thesis.tex`` and is checked here.

Run with ``uv run _build_povm_cards.py``.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# --- where things live -----------------------------------------------------
# In the repo this file sits in code/ and the tree is code/ + paper/.  It
# also runs from a scratch copy whose code/ is a symlink to the real one, so
# both roots are found rather than assumed.
_HERE = Path(__file__).resolve().parent
CODE = _HERE if (_HERE / "data").is_dir() else _HERE / "code"
PAPER = (_HERE if (_HERE / "paper").is_dir() else CODE.parent) / "paper"
DATA = CODE / "data"
FIGURES = PAPER / "figures"

# Both roots on the path before the tree's own modules are imported: in the
# repo this file sits beside them, in a scratch copy code/ is a symlink.
for _p in (str(CODE), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _build_recipe_figure import (bloch, cells, gate_table, read_coin_words,   # noqa: E402
                                  read_ledger, read_pose, read_tabular,
                                  word_matrix)
from randomized_core import (COVARIANCE, REORIENT, alignment, best_circuits,   # noqa: E402
                             coset_representatives, design_strength,
                             is_decomposable, load_atlas, load_rotations,
                             load_vertices, rot_key)
from randomized_decker import (DECKER_ANCHOR, FOURIER_SIGN, _cnot,            # noqa: E402
                               _pad_block, decker_circuit, decker_columns,
                               decker_fourier, decker_vertices)
from randomized_fragments import coin_column, latex_word                       # noqa: E402
from randomized_twojobs import coin_rotations                                  # noqa: E402

SOLIDS = ["tetrahedron", "octahedron", "cube", "icosahedron", "dodecahedron"]
SHORT = {"tetrahedron": "tet", "octahedron": "oct", "cube": "cube",
         "icosahedron": "icos", "dodecahedron": "dodec"}
PANEL = {"tetrahedron": "tet", "octahedron": "oct", "cube": "cube",
         "icosahedron": "ico", "dodecahedron": "dod"}
# The panel each card includes.  Four are the cards' own stubs; the
# tetrahedron's is Figure 1.1's, reused literally rather than redrawn, so
# that the figure and the card agree by construction instead of by luck.
# Assertion 21 measures whichever file is named here, so the reuse is checked
# and not exempted.
SPHERE_SRC = {"tetrahedron": "recipe_tet_sphere.tex",
              "octahedron": "card_oct_sphere.tex",
              "cube": "card_cube_sphere.tex",
              "icosahedron": "card_icos_sphere.tex",
              "dodecahedron": "card_dodec_sphere.tex"}
GROUP_TEX = {"T": r"\Tgroup", "O": r"\Ogroup", "I": r"\Igroup"}
# Appendix A's table for each covariance group.  A card that prints atlas
# words must carry a route to the rows they are drawn from -- and that table's
# caption is also where "Gates in operator order" is stated, which is what
# decides a composite word like F-dagger-Phi.  Head 4 carries it on the three
# cards with a four-pointer head; on the A_5 pair, whose heads 3 and 4 share
# one line and two pointers, the coin block carries it instead, and the
# generator puts it wherever the body has not already.
ATLAS_TAB = {"T": "tab:binary-tetrahedral-group-2t",
             "O": "tab:binary-octahedral-group-2o",
             "I": "tab:binary-icosahedral-group-2i"}
# The covariance field prints BOTH groups, the thesis's own way ("covariant
# under $\TwoG$ (on the Bloch sphere, $\Ggroup$)"): the binary group is what
# assertion 5 actually verifies over and what the coin block draws from, and
# the rotation group is what the sphere shows.  Printing only one leaves a
# reader who knows 2O has order 48 looking at a group of half the size, and on
# the tetrahedron card it would put $\Tgroup$ in the verdict against $\TwoT$ in
# the coin block -- two groups for one draw on one page.
BIN_TEX = {"T": r"\TwoT", "O": r"\TwoO", "I": r"\TwoI"}
SOLID_ADJ = {"tetrahedron": "tetrahedral", "octahedron": "octahedral",
             "cube": "cubic", "icosahedron": "icosahedral",
             "dodecahedron": "dodecahedral"}

TOL = 1e-15          # what "equal" means for a 2x2 built out of radicals
LOOSE = 1e-9         # ... and for anything that has been through an eigensolve
MARGIN = 0.1         # the reject side of every discrete (permutation) decision

SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.array([[1, 0], [0, -1]], complex)
PAULI = [SX, SY, SZ]
I2 = np.eye(2, dtype=complex)

WORD = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        12: "twelve", 16: "sixteen", 20: "twenty", 32: "thirty-two"}

# The tuple atoms Table E.1 factors each solid's coordinates into, and the
# radial scalar it factors out.  Both are CHECKED against povm_atlas.tex
# (assertion 6); this table only says what to look for.
RADIAL = {"tetrahedron": (r"\tfrac{1}{\sqrt3}", 1 / np.sqrt(3)),
          "octahedron": (None, 1.0),
          "cube": (r"\tfrac{1}{\sqrt3}", 1 / np.sqrt(3)),
          "icosahedron": (r"\tfrac{1}{\sqrt{2+\phig}}",
                          1 / np.sqrt(2 + (1 + np.sqrt(5)) / 2)),
          "dodecahedron": (r"\tfrac{1}{\sqrt3}", 1 / np.sqrt(3))}
TAU = (1 + np.sqrt(5)) / 2
ATOMS = {"1": 1.0, "0": 0.0, r"\phig": TAU, r"\invphig": TAU - 1}
# the closed-form arccos arguments the five drawn U_R boxes use, and the only
# ones rotation_from_tex will accept -- an unrecognised angle is an error, not
# a silently skipped factor
ANGLE_ARGS = {r"1/\sqrt3": 1 / np.sqrt(3),
              r"1/(\phig\sqrt3)": 1 / (TAU * np.sqrt(3)),
              r"\phig/\sqrt{\phig+2}": TAU / np.sqrt(TAU + 2)}


# ---------------------------------------------------------------------------
# Reading the repo
# ---------------------------------------------------------------------------

def load(solid):
    return np.load(DATA / f"povm_{solid}.npz", allow_pickle=True)


def read_card_circuit(solid, prefix="card_dec_"):
    return (FIGURES / f"{prefix}{SHORT[solid]}.tex").read_text()


def uncommented(text):
    """A circuit file's drawing, its header comment stripped -- the comments
    talk ABOUT the wires and the gategroups, so nothing may be counted over
    them."""
    return "\n".join(l for l in text.splitlines()
                      if not l.lstrip().startswith("%"))


def read_labels_table():
    r"""Table D.4's five rows: {solid: (live outcomes, our vertices, kappa,
    kappa's rider, 1/kappa^2)}.

    The permutation comes out of the printed two-line \smallmatrix and is
    asserted equal to the fragment's machine-readable twin, one comment line
    per solid ahead of the tabular (``% key <solid>: 0 1 2 3 -> 1 2 3 4``, the
    copy _build_recipe_figure.py reads).  kappa and 1/kappa^2 are kept as
    the STRINGS the table prints and are never re-rounded, the cube's
    "$-1/3$ exactly" rider included.  A \smallmatrix carries ampersands of
    its own, so the row is split from the RIGHT -- the last two separators
    are the ones between the three real columns.
    """
    text = (DATA / "randomized_labels.tex").read_text()
    keys = {}
    for m in re.finditer(r"^% key (\w+): ([\d ]+) -> ([\d ]+)$", text, re.M):
        keys[m.group(1)] = ([int(x) for x in m.group(2).split()],
                            [int(x) for x in m.group(3).split()])
    out = {}
    for line in read_tabular(text, r"\label{tab:decker-labels}"):
        body = line.rsplit(r"\\", 1)[0].strip()
        if " & " not in body or "smallmatrix" not in body:
            continue
        body, prem = body.rsplit(" & ", 1)
        body, kap = body.rsplit(" & ", 1)
        name, mat = body.split(" & ", 1)
        m = re.search(r"smallmatrix\}(.+?)\\\\(.+?)\\end", mat)
        assert m, f"Table D.4's {name} row has no two-line permutation"
        top = [int(x) for x in m.group(1).replace("&", " ").split()]
        bot = [int(x) for x in m.group(2).replace("&", " ").split()]
        solid = name.strip().lower()
        assert keys.get(solid) == (top, bot), \
            f"Table D.4's {name} row disagrees with its % key line"
        k = re.match(r"\$(.+?)\$(.*)$", kap.strip())
        pr = re.match(r"\$(.+?)\$$", prem.strip())
        assert k and pr, f"Table D.4's {name} row: kappa {kap!r}, premium {prem!r}"
        out[solid] = (top, bot, k.group(1), k.group(2).strip(),
                      pr.group(1))
    assert len(out) == 5, f"Table D.4 parsed to {len(out)} rows, not 5"
    return out


def read_povm_atlas():
    """Table E.1, cell for cell: {solid: (scalar or None, [(sign, atom) x3])}.

    Follows ``main.py``'s ``verify_sample_row`` precedent -- parse the emitted
    LaTeX rather than trust that it says what its generator meant.
    """
    text = (DATA / "povm_atlas.tex").read_text()
    heads = [(m.start(), m.group(1), m.group(2)) for m in re.finditer(
        r"\\multicolumn\{8\}\{c\}\{(\w+)(?: --- components scaled by "
        r"\$(\\tfrac\{1\}\{\\sqrt\{[^}]*\}\})\$)?\}", text)]
    assert len(heads) == 5, f"Table E.1 has {len(heads)} solid headers, not 5"
    bounds = [(h[0], heads[i + 1][0] if i + 1 < len(heads) else len(text))
              for i, h in enumerate(heads)]
    cell_re = re.compile(
        r"\{\\scriptsize \$(\d+)\.\$\} & \$\(\\begin\{array\}"
        r"\{@\{\}r@\{\}l@\{,\\,\}r@\{\}l@\{,\\,\}r@\{\}l@\{\}\}(.+?)"
        r"\\end\{array\}\)\$")
    out = {}
    for (start, name, scal), (a, b) in zip(heads, bounds):
        rows = {}
        for m in cell_re.finditer(text[a:b]):
            body = [x.strip() for x in m.group(2).split("&")]
            assert len(body) == 6, body
            tup = []
            for s, v in zip(body[0::2], body[1::2]):
                atom = re.match(r"\\mathrlap\{(.+?)\}\\hphantom", v).group(1)
                tup.append(("+" if s == r"\phantom{+}" else s, atom))
            rows[int(m.group(1))] = tup
        out[name.lower()] = (scal, rows)
    return out


def read_frozen_captions():
    r"""The five CURRENT Appendix D circuit captions, for assertion 23.

    ``paper/figures/_card_captions_frozen.tex`` is the freeze; it is not
    ``\input`` by anything and exists only so that "nothing printed today
    vanishes" stays checkable after the cards have replaced the captions.
    While ``bsc-thesis.tex`` still carries them, both sources are read and
    required to agree character for character -- which is what makes the freeze
    provably a freeze rather than a second, drifting copy.
    """
    def five(text):
        out = {}
        for solid in SOLIDS:
            lab = r"\label{fig:dec_%s_circuit}" % SHORT[solid]
            i = text.find(lab)
            if i < 0:
                continue
            out[solid] = " ".join(text[text.rindex(r"\caption{", 0, i):i].split())
        return out

    thesis = (PAPER / "bsc-thesis.tex").read_text()
    live = five(thesis)
    fz = FIGURES / "_card_captions_frozen.tex"
    frozen = five(fz.read_text()) if fz.is_file() else {}
    if frozen:
        assert len(frozen) == 5, f"the freeze holds {len(frozen)} captions, not 5"
        # a solid whose float still includes circuit_dec_*.pdf has not been
        # carded yet, so its live caption must still be the frozen one; a solid
        # whose float is a card has no live caption left to compare
        for s in SOLIDS:
            if r"figures/circuit_dec_%s.pdf" % SHORT[s] not in thesis:
                continue
            assert live.get(s) == frozen[s], \
                f"the freeze and bsc-thesis.tex's {s} caption have diverged"
        return frozen
    assert len(live) == 5, (
        "assertion 23 has no source for the five captions: bsc-thesis.tex no "
        "longer carries them and paper/figures/_card_captions_frozen.tex does "
        "not exist")
    return live


def series_strip_row(solid):
    r"""(column spec, heads, this solid's row) of Figure 1.1's own foot strip.

    ``\RecipeSeriesStrip`` in ``code/data/recipe_tet_data.tex`` is what
    Figure 1.1 prints under its own recipe: one row per solid, eight columns,
    the first of them the solid's name.  A card prints the same heads and the
    same row minus that first column -- its title says the solid -- so the
    reader who has met Figure 1.1 recognises the strip instead of reading it.

    Figure 1.1 sets its OWN solid's row in bold (it is the solid that figure
    works), and a ``\textbf`` wrapping a whole cell is that highlight and is
    dropped here.
    A ``\textbf`` INSIDE a cell is content -- the octahedron's
    ``3 axes --- \textbf{exact}`` -- and is kept.  Assertion 30 re-derives all
    of this and compares cell by cell.
    """
    text = (DATA / "recipe_tet_data.tex").read_text()
    i = text.index(r"\newcommand{\RecipeSeriesStrip}")
    m = re.compile(r"\\begin\{tabular\}\{([^}]*(?:\}[^}]*)*?)\}\n").search(text, i)
    assert m, "the Figure 1.1 strip has no tabular"
    spec = m.group(1)
    body = text[m.end():text.index(r"\end{tabular}", m.end())]
    rows = [cells(l.strip()) for l in body.splitlines()
            if l.strip().endswith(r"\\")]
    assert len(rows) == 6, f"the Figure 1.1 strip has {len(rows)} rows, not 6"
    # drop the NAME column from the spec, keeping Figure 1.1's own @{} padding
    ms = re.fullmatch(r"(@\{\}\s*)[lcr](\s*.*)", spec)
    assert ms, f"unparsable Figure 1.1 strip column spec {spec!r}"

    def unbold(c):
        m_ = re.fullmatch(r"\\textbf\{(.*)\}", c)
        return m_.group(1) if m_ else c

    heads = [unbold(c) for c in rows[0][1:]]
    want = solid.capitalize()
    hit = [r for r in rows[1:] if unbold(r[0]) == want]
    assert len(hit) == 1, f"{solid}: {len(hit)} rows named {want} in Figure 1.1"
    return ms.group(1) + ms.group(2).lstrip(), heads, \
        [unbold(c) for c in hit[0][1:]]


def read_fourier_block(src):
    r"""(m, ell) off a circuit's own Fourier gate: the transform's size and how
    many wires the lower register spends on it.

    ``\multigate{j}{F_m ...}`` spans j+1 wires; the tetrahedron draws its
    two-point transform as the single-wire ``\gate{H}`` its caption calls
    ``F_2``.  Both readings are checked against REORIENT's m in assertion 16.
    """
    m = re.search(r"\\multigate\{(\d+)\}\{\\mathsf\{F\}_(\d+)\^\\dagger", src)
    if m:
        return int(m.group(2)), int(m.group(1)) + 1
    assert r"\gate{H}" in src, "no Fourier block and no H in the circuit"
    return 2, 1


# ---------------------------------------------------------------------------
# The geometry
# ---------------------------------------------------------------------------

def nearest(vertices, w, what):
    """argmin with a MARGIN, not a tolerance.

    What is being decided is discrete -- which vertex -- so one wrong entry
    does not perturb a printed number, it jumps it to a different one.  The
    assertion is therefore on the separation: inside 1e-9, runner-up beyond
    0.1 (``randomized_decker.check_decker_outcome_order``'s own discipline).
    """
    d = np.linalg.norm(vertices - w, axis=1)
    near = np.sort(d)[:2]
    assert near[0] < LOOSE and near[1] > MARGIN, f"{what}: separation {near}"
    return int(np.argmin(d))


def antipode(vertices, k):
    """The 0-based index of -n_k, or None."""
    for j in range(len(vertices)):
        if np.allclose(vertices[j], -vertices[k], atol=LOOSE):
            return j
    return None


def factored_tuple(solid, n):
    """A vertex tuple as Table E.1 factors it: the radial scalar out front, the
    components bare, each recognised rather than typed."""
    scal, unit = RADIAL[solid]
    out = []
    for x in n:
        r = x / unit
        hit = [(nm, v) for nm, v in ATOMS.items() if abs(abs(r) - v) < 1e-9]
        assert len(hit) == 1, f"{solid}: component {x} is {r} of the unit"
        nm, v = hit[0]
        out.append(("+" if v == 0.0 else ("-" if r < 0 else "+"), nm))
    return out


def tex_tuple(tup):
    r"""(sign, atom) triples -> ``(+1,\,\phantom{+}0,\,-\phig)``.

    An unsigned component keeps a ``\phantom{+}`` so that the tuples align down
    the column, which is Table E.1's own move.
    """
    return "(" + ",\\,".join(
        (s if a != "0" else r"\phantom{+}") + a for s, a in tup) + ")"


EFFECT_ATOMS = {
    # entry of V * E_k -> the token printed for it.  Recognised, never typed;
    # assertion 2 multiplies the printed tokens back out and diffs the npz.
    "tetrahedron": [(np.sqrt(3) + 1, r"\sqrt3+1"), (np.sqrt(3) - 1, r"\sqrt3-1"),
                    (1 + 1j, "1+i"), (1 - 1j, "1-i"),
                    (-1 + 1j, "-1+i"), (-1 - 1j, "-1-i")],
    "octahedron": [(0.0, "0"), (1.0, "1"), (2.0, "2"), (-1.0, "-1"),
                   (1j, "i"), (-1j, "-i")],
    "cube": [(np.sqrt(3) + 1, r"\sqrt3+1"), (np.sqrt(3) - 1, r"\sqrt3-1"),
             (1 + 1j, "1+i"), (1 - 1j, "1-i"),
             (-1 + 1j, "-1+i"), (-1 - 1j, "-1-i")],
}
# ... and the scalar those tokens are printed against.  The tetrahedron and the
# cube go through 1/(V sqrt3) so that their entries are integers in Z[sqrt3, i];
# the octahedron's are already integral at 1/V.
EFFECT_SCALE = {"tetrahedron": (np.sqrt(3) / 12, r"\tfrac{\sqrt3}{12}"),
                "octahedron": (1 / 6, r"\tfrac16"),
                "cube": (np.sqrt(3) / 24, r"\tfrac{\sqrt3}{24}")}


def factored_effect(solid, E):
    """E_k as the card prints it: one scalar out front, four recognised atoms."""
    scale, _ = EFFECT_SCALE[solid]
    M = E / scale
    out = []
    for a in range(2):
        row = []
        for b in range(2):
            hits = [t for v, t in EFFECT_ATOMS[solid] if abs(M[a, b] - v) < 1e-12]
            assert len(hits) == 1, f"{solid}: entry {M[a, b]} is not an atom"
            row.append(hits[0])
        out.append(row)
    return out


def unfactor_effect(solid, toks):
    """The printed tokens multiplied back out -- assertion 2's other half."""
    scale, _ = EFFECT_SCALE[solid]
    look = {t: v for v, t in EFFECT_ATOMS[solid]}
    return scale * np.array([[look[t] for t in row] for row in toks])


# ---------------------------------------------------------------------------
# Per-solid facts: everything derived once, then checked, then printed
# ---------------------------------------------------------------------------

def facts(solid, has_card_files=True):
    """Every datum a card prints, derived from the tree and cross-referenced."""
    npz = load(solid)
    v = np.asarray(npz["vertices"], float)
    E = np.asarray(npz["elements"], complex)
    V = len(v)
    f = {"solid": solid, "V": V, "v": v, "E": E,
         "group": GROUP_TEX[COVARIANCE[solid]],
         "bingroup": BIN_TEX[COVARIANCE[solid]],
         "design": design_strength(v),
         "p0": [(1 + v[k, 2]) / V for k in range(V)]}

    # --- the circuit, rebuilt (never re-derived by hand) --------------------
    W = decker_circuit(solid)
    d, live, _ = decker_vertices(solid)
    n = int(round(np.log2(len(W))))
    f.update(W=W, d=d, live=live, n=n, reg=2 ** n,
             dead=[i for i in range(2 ** n) if i not in live])

    src = read_card_circuit(solid) if has_card_files else \
        (FIGURES / f"circuit_dec_{SHORT[solid]}.tex").read_text()
    f["circuit_src"] = src
    f["frozen_src"] = (FIGURES / f"circuit_dec_{SHORT[solid]}.tex").read_text()
    f["wires"] = uncommented(src).count(r"\lstick")
    f["m"], f["ell"] = read_fourier_block(uncommented(src))
    f["upper"] = n - f["ell"]

    # --- the key, derived HERE from decker_vertices + Table D.3's R ---------
    # decker_circuit's rows are the CONJUGATED POVM vectors; decker_vertices is
    # what applies the np.conj.  Building the key off the rows directly reverses
    # every Fourier orbit and prints a plausible, wrong map -- assertion 15 is
    # the second derivation that catches it.
    R = np.array(REORIENT[solid][1].evalf(), dtype=float)
    f["R"] = R
    f["key"] = {i: nearest(v, R @ x, f"{solid} outcome {i}") + 1
                for i, x in zip(live, d)}

    # --- Table D.3, D.4, D.1, 5.2 -------------------------------------------
    f["pose"] = read_pose()[solid]
    f["labels"] = read_labels_table()[solid]
    f["ledger"] = read_ledger()
    f["coins"] = read_coin_words()[solid]
    # the single-shot tail weight, 27<sum_a w_a^4>, in each pose: D.2's own
    # 9 -> 15 / 27 -> 12 / 81:5, and what decides whether the card may say
    # "nothing is lost" about running unreoriented
    f["tail_atlas"] = float(27 * np.mean(np.sum(v ** 4, axis=1)))
    f["tail_decker"] = float(27 * np.mean(np.sum(np.asarray(d, float) ** 4,
                                                 axis=1)))
    f["exact_reorientation"] = "over Clifford" in f["pose"]["R"]
    f["identity_relabel"] = r"k \mapsto k" in f["pose"]["outcomes"]

    # --- the coin ------------------------------------------------------------
    f["decomposable"] = is_decomposable(v)
    if f["decomposable"]:
        A, v0 = alignment(v)
        f["A_is_identity"] = bool(np.allclose(A, np.eye(3), atol=1e-12))
        f["align_vertex"] = nearest(v, v0, f"{solid} alignment seed") + 1
        g = COVARIANCE[solid]
        Rs, circ = load_rotations(g), best_circuits(load_atlas(g))
        reps = coset_representatives(v, Rs, circ)
        rots = coin_rotations(solid)
        axis_reps = []
        seen = []
        for i, nn in enumerate(v):
            if not any(np.allclose(nn, -m, atol=LOOSE) for m in seen):
                seen.append(nn)
                axis_reps.append(i)
        rays = []
        for (mag, dep, seq), ai, Rg in zip(reps, axis_reps, rots):
            assert circ[rot_key(Rg)] == (mag, dep, seq), \
                f"{solid}: coin_rotations and coset_representatives disagree"
            anti = antipode(v, ai)
            assert anti is not None, f"{solid}: axis {ai + 1} has no antipode"
            # |0> reads the +1 branch of Z, whose snapshot direction is
            # R_g^T v0: the drawn word first, the fixed alignment after it.
            # The alignment cancels between snapshot and readout and must NOT
            # be applied a second time (two rays then fire one vertex).
            fired = nearest(v, Rg.T @ v0, f"{solid} coin |0> vertex")
            assert {fired, ai, anti} == {ai, anti}, \
                f"{solid}: |0> fires vertex {fired + 1}, off the axis it prices"
            other = anti if fired == ai else ai
            rays.append({"word": latex_word(seq), "seq": seq, "Rg": Rg,
                         "zero": fired + 1, "one": other + 1,
                         "phi": seq.count("\u03a6")})
        f["rays"] = rays
        f["coin_words_sorted"] = coin_column(solid)[0]
    else:
        f["A_is_identity"] = False
        f["align_vertex"] = None
        f["rays"] = None
        f["coin_words_sorted"] = None
    return f


# ---------------------------------------------------------------------------
# Emitting
# ---------------------------------------------------------------------------
#
# The card's surface labels rather than narrates: Figure 1.1's own strip, four
# objects, one label per object, and a pointer per band.  Every fact is
# printed as an object or as a label, and every one of them is checked.
#
# The literals below are hand-typed LaTeX, and every one of them is closed
# against the code before a byte is written: the strip against Figure 1.1's own
# \RecipeSeriesStrip (check 30), the heads and labels against the objects they
# name (checks 12, 29), the parameter atoms against decker_circuit (27), the
# worked E_1 against the npz (33), kappa against Table D.4 (18), the rays
# against the atlas (7, 19).

# --- shared literals: one emitter each, byte-identical on all five (31) ----
# Figure 1.1 states it negative-first ("no deterministic protocol ... any
# dilation, this one included ... only the octahedral POVM admits an exact
# implementation").  A compression that keeps the appositive and drops its
# host -- "exact, this dilation included" -- reads as including the dilation
# among the EXACT things, the opposite of Theorem 1.  The em-dash clause
# fixes the polarity in one word, and every card prints both halves, the
# positive and the negative, in one cell.
STRIP_FOOTER = (r"only the octahedral coin is exact --- this dilation is not "
                r"(Theorem~\ref{thm:main}, "
                r"Appendix~\ref{sec:decker:exactness})")
HEAD_A = (r"Vertices and effects, "
          r"$E_k = \tfrac1V(\Id + \hat n_k \cdot \sigmavec)$.")
# The head says what the box IS, not merely that it may be left out.  "an
# optional prepended rotation" is the thesis's own verb ("prepending U_g",
# Sections 4.2.3-4), covers the fixed use (rotated POVMs) and the drawn one
# (the twirl), and leaves what the rotation DOES to the full-width U_g rider
# below the circuit: the head's width cannot take a second clause (head 2
# measures 261.29pt with its pointers, against a 452.9679pt line).
HEAD_B = r"Circuit; the dashed $U_g$ is an optional prepended rotation."
# The convention lives on the object it governs: the key rectangle carries
# `upper` (stub) and `lower` (spanned) heads whose bit widths say which wires,
# so the head says only which end of the register is read first.
HEAD_C = r"Register $\to$ vertex, top wire first"
# The key's dead cells print "---" and nothing on the card said what that
# meant: "dead" is the strip's column head, undefined there and in Figure 1.1
# too, so the card cannot lean on the figure for it.  Two words, on the three
# cards that HAVE dead cells and only those (assertion 36 ties it to the
# count).
HEAD_C_DEAD = r"; --- never fires"
# "no ancilla" is what says the coin REPLACES the dilation rather than running
# inside it -- the whole of band 4, and inheritable from neither Figure 1.1
# (which never explains the coin) nor the protocol line (which is compatible
# with running inside the dilation).  It negates the strip's own `anc.` cell
# two bands above, so it costs the reader no new noun.
HEAD_D = r"Coin over axes, no ancilla, $\ket0$-vertex first"
HEAD_D_NONE = r"No coin over axes"
# The rider states the convention, not the topic: naming an order convention
# without saying which one states no fact, and reading F-dagger-Phi backwards
# names the wrong vertex on four of the dodecahedron's ten rays.  It is
# printed on the only two cards with composite words.  The convention is
# three words, and they are the octahedron's own for the same convention one
# card earlier ("rightmost factor first", on its three-factor U_R).  "factor"
# is what stops the two "first"s reading as one: the head then says which
# vertex of the RAY is written first and which factor of the WORD acts first,
# each with its own noun.  Measured with its three pointers: 437.53pt against
# the 452.9679pt line, 15.44pt of slack -- the tightest head on the five
# cards, and a fragile invariant (re-measure before rewording head 4 or
# adding a fourth pointer).  ", rightmost first" measures 408.89.
ORDER_RIDER = r", rightmost factor first"
# Direction, on every card: the relabelling is printed as g^{-1}, and no card
# says "acts on" (check 11).  It rides the U_R clause, never the U_R formula
# -- nothing may be glued to the right of that atom (see the body template's
# header).  The rider also names the drawn box's JOB ("twirls the noise"),
# because band 3's kappa label leans on "the drawn twirl" and head 2's width
# cannot carry the clause; the
# draw is from the solid's own binary polyhedral group, all of which act
# irreducibly, so the twirl claim needs no caveat here (F.3.2's hypothesis).
# "twirls" absolute is Section 4.2.4's own use ("the second job is to
# twirl"); "twirls the noise" was measured and WRAPS the A_5 U_R lines
# (dodecahedron 490.81pt against the 452.9679pt line, +11.98pt of card),
# where this wording sets 448.53pt -- one line, 4.4pt of slack.  Re-measure
# before adding a word.
UG_RIDER = r"a drawn $U_g$ twirls and relabels outcomes by $g^{-1}$."
# D.2: "Reorientation and relabelling are one choice: run $U_R$ and relabel,
# or do neither and use Decker's circuit and numbered vertex list.  Either
# way, the estimator is unbiased with its variance unmoved, $\kappa = 1$ and
# offset zero.  Only a mismatch costs."  A label that prices the mismatch and
# never prints the free branch leaves a card-only reader concluding that
# $U_R$ -- exact over no gate set of this thesis on three of five cards -- is
# compulsory.  Two words buy the branch, in D.2's own pairing (assertion 36
# requires D.2 to still say it).  The label is sentences rather than
# telegraph, because a reader meeting $\kappa$ here needs the gesture, and it
# names Decker with the cite the caption already carries.
# What it prices is HIS circuit read with OUR list, index for index, exactly
# as Table D.4's caption and D.2 say ("his circuit with our list gives the
# $\kappa$ in Table D.4").  It is NOT the key read with $U_R$ skipped: that
# relabelling-without-the-gate misread prices at Tr[R]/3 (octahedron
# -0.179335, dodecahedron +0.900434), which is a different number on every
# card but the tetrahedron, whose key is the identity permutation.  So the
# label (1) opens with the card's own protocol -- run U_R, read the key --
# as the first free branch, (2) names the priced misread in D.2's own words,
# and (3) leaves the definition of $\kappa$ to Table D.4's caption, one
# \ref away in this band's own pointer.  "either way $\kappa = 1$, free."
# is 29(h)'s literal and its free-before-priced order; "exact" is BANNED in
# this label -- on the card that word belongs to gate-set exactness, and the
# strip above answers it "no" for $U_R$ on three of five solids.
# The wording is a budget, not a style: the box is the band's height wherever
# it outgrows the key, and the dodecahedron's is the binding float.  Every
# word on the label carries a checked claim; re-measure before adding one.
# The cite is tied, "list~\cite{...}": untied, the tetrahedron alone breaks
# between "list" and its cite and drops the cite onto the next line as its
# first token.  The tie leaves the space its stretch, so it forbids only that
# break and the four cards that never take it are unmoved.
KAPPA_FMT = (r"Run $U_R$ and read the key, or skip $U_R$ and read "
             r"Decker's own numbered list~"
             r"\cite{decker2004quantumcircuitssinglequbit}: either way "
             r"$\kappa = 1$, free. His circuit with our list, index for "
             r"index, is the mismatch: $\kappa = %s$; under the drawn "
             r"twirl it costs a $1/\kappa^2 = %s\times$ premium in shots, "
             r"never a bias.")
PROTOCOL_FMT = r"Draw one, apply it, then the alignment (%s); measure $Z$."
# Measured against the 452.9679pt line WITH its label ("Figure D.5.: "), in
# the real class: "$U_R$ and the dashed $U_g$ are ours." runs 489.50pt on the
# three two-digit cites -- 36.53pt over -- and microtype plus a caption
# paragraph with no \emergencystretch sets it as ONE overfull line hanging
# 19.0bp into the right margin, not as two.  Without "the dashed" it is
# 434.08pt, 18.89pt of slack, and costs nothing: head 2 names the dashed box
# one band above ("the dashed $U_g$ is an optional prepended rotation").
CAPTION_FMT = (r"The solid box is \textcite[Fig.~%d]"
               r"{decker2004quantumcircuitssinglequbit}'s; "
               r"$U_R$ and $U_g$ are ours.")
DECKER_FIG = {"tetrahedron": 7, "octahedron": 11, "cube": 9,
              "icosahedron": 15, "dodecahedron": 13}
# The pointer column: apparatus, not prose.  Ten per card (nine on the
# tetrahedron, which has no atlas words to route), every one a live \ref.
# The title line's pointer must be true on all five cards, the tetrahedron's
# included, and short: the title row is \CTitle (\large\bfseries, 275.39pt on
# the dodecahedron) + \hfil + this at \footnotesize, against the 452.9679pt
# line.  It names WHERE, not what: Figure 1.1's banner carries the reciprocal
# pointer ("All five solids in Appendix D"), so the two title lines route to
# each other and each says which chapter it is routing to.  The noun is
# `reference', not `card': "card" is this project's word for these pages
# and the thesis prints it nowhere.
PTR_STRIP = (r"Introduction reference: "
             r"Figure~\ref{fig:tetrahedron-recipe}")
PTR_A = r"Table~\ref{tab:povm-atlas} $\cdot$ Equation~\eqref{eq:povmform}"
PTR_B = r"Section~\ref{subsec:rotated-povms} $\cdot$ Table~\ref{tab:decker-pose}"
PTR_C = r"Table~\ref{tab:decker-labels} $\cdot$ Section~\ref{sec:decker:using}"
PTR_D = (r"Table~\ref{tab:decker-vs-coin} $\cdot$ "
         r"Section~\ref{subsec:randomized-implementations}")

# The one thing about THIS solid a reader should recognise, after the four
# identities every card states.  Each is a fact, not decoration, and each is
# checked: the SIC by assertion 1, the Pauli eigenprojectors by the printed
# matrices beside it, the numbering by assertion 21's chip rule.
IDENT_TAIL = {
    "tetrahedron": r"; $|\braket{\psi_j}{\psi_k}|^2 = \tfrac13$ for "
                   r"$j \neq k$: the SIC condition.",
    "octahedron": r"; the Pauli eigenprojectors, $\times\tfrac13$.",
    "cube": r".",
    # "chip" appears nowhere in the printed thesis (grep: 0 hits in
    # bsc-thesis.tex, none in the compiled PDF) -- it is a TikZ macro name.
    # These two cards would have been the first and only place a reader met
    # it, undefined, and it is the only explanation of why 6 of 12 and 10 of
    # 20 vertices carry a numeral.  Said in the reader's words instead.
    "icosahedron": r"; numbered at each axis's near vertex.",
    "dodecahedron": r"; numbered at each axis's near vertex.",
}
# One sentence per card about the projective draw, so that the five pages
# carry F.3.3's ladder; on the icosahedron alone the two-jobs sentence is
# read by itself as a fact, and against the other four as the CROSSING RUNG:
#
#     tetrahedron   no projective route -- draw anyway and you assemble the CUBE
#     octahedron    realize needs 3 of T's 12; the TWIRL forces the full draw
#     cube          realize needs 4 of T's 12; the TWIRL forces the full draw
#     icosahedron   the bars meet: one draw does BOTH
#     dodecahedron  T reaches 6 of 10 axes; REALIZATION forces the full 2I
#                   -- among GROUPS, and the flip completion beats it
#
# Every verdict is Table 5.2's own `Which one binds' cell, and check 19d
# pins each note to that cell before a byte is written.  Five traps:
#
#   * the numerals are the PRINTED RAY LIST above the sentence, not a
#     subgroup order.  They agree on these two cards only (3 = |C_3|,
#     4 = |V_4|) and part company at the icosahedron, 6 words against T's
#     12 -- which is exactly why that card is the crossing one.  So the
#     sentence says "these three", pointing at the rays, and 19d derives
#     the numeral from len(f["rays"]) rather than trusting the word.
#   * the tetrahedron's is the one claim the thesis prints nowhere else:
#     bsc-thesis.tex states the exclusion five times and the consequence
#     never (the tetrahedron does not fail; it becomes the cube).  Its
#     geometric half is dial_settings.py's.  A Z readout reports an AXIS,
#     both poles, so the assembled POVM is the covariant one on the CLOSED
#     orbit +-(T.v0), and the tetrahedron is the only Platonic solid whose
#     vertex orbit is not already closed.
#   * the dodecahedron must NOT be told it would get the inscribed cube.
#     F.3.3's footnote splits its ten axes 4 + 6 under T and the alignment
#     vertex selects one -- and vertex 9, the atlas seed, selects the SIX
#     (randomized_decker's twelve-vertex family).
#   * every "forces" on these five pages is a minimum over SUBGROUPS, which
#     is the quantifier Table 5.2's own row labels carry ("Smallest GROUP
#     that realizes / twirls") and the one bsc-thesis.tex prints in Section
#     5.2.3 ("among groups, T is the least that twirls").  It is printed on
#     the three cards that state a minimum, because randomized_twojobs'
#     check_flip_completion makes the unscoped reading false on the
#     dodecahedron: the ten-word coin times the Klein layer {1, X, Y, Z}
#     drawn AFTER the fixed alignment -- forty elements, not a group --
#     realizes the POVM and is exactly depolarizing at T_zz for every
#     measurement-side (T, t), the full 2I draw's own channel, at 0.4 Phi
#     per shot against 2I's 0.8 (and 0.4 is the floor for any realizing draw
#     of atlas words around the alignment, group or not).  Four things the
#     dodecahedron's sentence prints are load-bearing: the SCOPE, the flip's
#     POSITION -- last, which the card names from the readout end (drawn
#     before the alignment instead, the solid loses BOTH jobs: its Klein
#     orbit of v0 collapses to two axes and the coin does not re-spread
#     them) -- the flip's RELABELLING (X or Y reads the printed pair
#     reversed; without the swap the four flips average the readout to
#     (t/2) Id and the card prescribes a measurement that reports nothing),
#     and the numeral, which is the ensemble, ten words times four flips.
#     19d pins the first three and derives the swap set from the gates; its
#     numeral test pins the fourth.  The octahedron and the cube are scoped
#     too and lose nothing by it: the octahedron's own completion IS T (its
#     coin is a transversal of the Klein subgroup in T = V x| C_3), and the
#     cube's is 16 elements, larger than T -- so on those two cards nothing
#     beats the 2T draw either way.
#     The icosahedron's sentence needs no scope: its completion is 24
#     against T's 12, and drawn in the other order it is the T draw twice
#     over.
#   * the dodecahedron's note sets TWO LINES in the real \PBOX (917.79pt
#     natural, \vtop depth 14.5pt), and that second line is what this card's
#     float reserve buys: it is the tallest float of the five, with 21.30pt
#     of reserve against \textheight = 700.50687pt, and a band-4 line costs
#     about 11pt.
#     Line one ends on the sentence boundary "forces $\TwoI$." and line two
#     opens on "With".  A THIRD line costs about as much again, and the
#     wrapping is not read off the natural width -- 888.29pt breaks to three
#     where 880.80 still sets two, because $\Id, X, Y, Z$ and $0.4\,\Phi$
#     are unbreakable -- so re-measure before adding a word: set the
#     candidate in a \vtop at 452.9679pt and read its depth, 2.5pt for one
#     line, ~14pt for two.  If height is ever wanted back,
#     card_dodec_body.tex's cut ladder is the source, and its first rung
#     (drop the worked $E_1$, -25.34pt) pays for this line twice over.
DRAW_NOTE = {
    "tetrahedron": r" Align a vertex and draw from $\TwoT$ anyway: each "
                   r"readout reports an axis, not a vertex, and the eight "
                   r"effects assembled are the cube's "
                   r"(Figure~\ref{fig:dec_cube_circuit}).",
    "octahedron": r" Realization needs only these %s; among groups it is the "
                  r"twirl that forces the full $\TwoT$ draw "
                  r"(Section~\ref{sec:shadows:bills}).",
    "cube": r" Realization needs only these %s; among groups it is the twirl "
            r"that forces the full $\TwoT$ draw "
            r"(Section~\ref{sec:shadows:bills}).",
    # The phrase `does randomness's two jobs at once' is this card's alone --
    # 19d pins it, and two negative controls are written against that string.
    "icosahedron": r" A full $\TwoT$ draw does randomness's two jobs at once: "
                   r"it realizes the POVM and twirls the noise "
                   r"(Section~\ref{sec:shadows:bills}).",
    # The one card the flip completion overturns.  "among groups" is what
    # makes the first sentence true, and the flip's POSITION is what makes
    # the second one true -- drawn before the alignment instead, the same
    # ensemble loses both jobs.  Both are pinned by 19d.  19b bans the phrase
    # "after the alignment" outright on these pages, so the position is named
    # from the readout end.  Five things the sentence prints are load-bearing:
    # the SCOPE, which the protocol line one line above would otherwise
    # contradict (ten words realize); the POSITION; the flip's RELABELLING --
    # panel D reads pairs $\ket0$-vertex first and a drawn X or Y flips the Z
    # outcome, so outcome 0 then reports the SECOND vertex of the pair, and
    # read in the printed order regardless the four flips average the readout
    # to (t/2) Id, a measurement that reports nothing; the numeral, which is
    # the ensemble, ten words times four flips, not the words alone; and
    # $0.8$ as $\TwoI$'s, a bill the card would otherwise print nowhere.
    # "With a drawn flip ..., these forty twirl too" is chosen over "Adding a
    # drawn flip ...: these forty twirl too", whose participle dangles (the
    # forty do no adding), at an identical wrap; every combination of "before
    # readout", "these forty" and a named 0.8 sets three lines (941-1029pt).
    "dodecahedron": r" Among groups, realization forces $\TwoI$. With a drawn "
                    r"flip $\Id, X, Y, Z$ last ($X$ or $Y$ swaps the pair), "
                    r"these %s twirl too, at $0.4\,\Phi$, not $\TwoI$'s $0.8$.",
}
# The ray count, spelled, for the two cards whose note points at the list.
NUMERAL = {3: "three", 4: "four", 6: "six", 10: "ten", 40: "forty"}
# |{Id, X, Y, Z}|: the flip layer the dodecahedron's note draws last, the
# factor between its printed ray list and the ensemble its numeral counts.
KLEIN = 4

# The worked effect, on the two cards whose vertex table has no room for an
# effect column.  It nests under the two vertex blocks, in slack the sphere
# leaves, at zero words -- and assertion 33 rebuilds it from the npz.
WORKED_E1 = {
    # The icosahedron's slashed entries run the matrix to the column's edge,
    # so its fractions are set vertical.  \dfrac keeps the numerals at text
    # size (8.97pt at \small; \tfrac would drop them to the 6.97pt script
    # size) and the panel, not the right column, still sets the row height.
    # The dodecahedron's entries are short and stay slashed.
    "icosahedron": r"$E_1 = \tfrac1{12}\pmat{1 & "
                   r"\dfrac{\phig-i}{\sqrt{2+\phig}} \\ "
                   r"\dfrac{\phig+i}{\sqrt{2+\phig}} & 1}$, "
                   r"$E_{g(k)} = U_g E_k U_g^\dagger$.",
    "dodecahedron": r"$E_1 = \tfrac1{20}\pmat{1+1/\sqrt3 & (1-i)/\sqrt3 \\ "
                    r"(1+i)/\sqrt3 & 1-1/\sqrt3}$, "
                    r"$E_{g(k)} = U_g E_k U_g^\dagger$.",
}
# The Naimark gloss, on the M-tilde-dagger line of all five cards.  That line
# states how the box's factors compose but never that it IS the box, and its
# noun is defined 27 pages back, in Section 4.2.2, with no pointer to it among
# the card's ten.  "the solid box" is the card's own pairing -- head 2 names the
# DASHED U_g, and the drawing carries exactly those two gategroups -- and it is
# the chapter head's own noun for Decker's block.  It also gives the strip
# footer's "this dilation" an antecedent: "Naimark" appears nowhere else on the
# five pages.  Measured on the tetrahedron, the longest of the five formulas:
# 438.23pt against the 452.9679pt line, 14.73pt of slack, so one line on all
# five and no card's reserve moves.  "the boxed circuit" misses by 3.02pt.
NAIMARK_GLOSS = r", the Naimark extension unitary the solid box implements"
# The parameter block, ONE FORMULA PER LINE: set as a justified paragraph,
# the inline 4x4 opens a four-line hole.  Line 1 is assembled from the drawn
# U_R; these are the rest.  Every atom here is closed against the code by
# assertion 27.
PARAM_LINES = {
    "tetrahedron": [
        r"$Q^\dagger$: the CNOT $\ket{00}\mapsto\ket{00}$, "
        r"$\ket{11}\mapsto\ket{01}$; the ancilla-controlled $S$ realizes "
        r"$\mathrm{diag}(1,1,1,i)$",
        r"$\tilde{M}^\dagger = (I_2 \otimes \mathsf{F}_2)\,"
        r"\mathrm{diag}(1,1,1,i)\,(U_A \otimes I_2)\,Q^\dagger$"
        + NAIMARK_GLOSS,
    ],
    "octahedron": [
        r"$Q_8^\dagger$: the CNOT $\ket{000}\mapsto\ket{000}$, "
        r"$\ket{101}\mapsto\ket{001}$; "
        r"$\mathsf{F}_3^\dagger \oplus I_1 \in \C^{4\times4}$, lower two wires",
        r"$\tilde{M}^\dagger_8 = "
        r"(U_A \otimes (\mathsf{F}_3^\dagger \oplus I_1))\,Q_8^\dagger$"
        + NAIMARK_GLOSS,
    ],
    "cube": [
        r"$Q^\dagger$: the CNOT $\ket{000}\mapsto\ket{000}$, "
        r"$\ket{101}\mapsto\ket{001}$; $\mathsf{F}_4^\dagger$, lower two wires",
        r"$\tilde{M}^\dagger = (U_A \otimes \mathsf{F}_4^\dagger)\,Q^\dagger$"
        + NAIMARK_GLOSS,
    ],
    "icosahedron": [
        r"$A^\dagger$: Equation~\eqref{eq:adagger}; "
        r"$A$: Equation~\eqref{eq:aunfactored}",
        r"$u_\pm = \sqrt{1/2 \pm \sqrt{(2+\phig)/20}}$, "
        r"$v_\pm = \mp\sqrt{1/2 \pm \invphig/(2\sqrt3)}$, each $\pm1$ "
        r"in the two rotation boxes is $\pm\sqrt{1/2}$",
        r"$Q_{16}^\dagger$: the CNOT $\ket{0000}\mapsto\ket{0000}$, "
        r"$\ket{0101}\mapsto\ket{0001}$; $\mathsf{F}_3^\dagger \oplus I_1$, "
        r"lower two wires",
        r"$\tilde{M}_{16}^\dagger = "
        r"(A^\dagger \otimes (\mathsf{F}_3^\dagger \oplus I_1))\,Q_{16}^\dagger$"
        + NAIMARK_GLOSS,
    ],
    "dodecahedron": [
        r"$A^\dagger$: Equation~\eqref{eq:adagger}; "
        r"$A$: Equation~\eqref{eq:aunfactored}",
        r"$u_\pm = \sqrt{1/2 \pm \phig/(2\sqrt3)}$, "
        r"$v_\pm = \mp\sqrt{1/2 \pm 1/(2\sqrt{2+\phig})}$, each $\pm1$ "
        r"in the two rotation boxes is $\pm\sqrt{1/2}$",
        r"$Q_{32}^\dagger$: the CNOT $\ket{00000}\mapsto\ket{00000}$, "
        r"$\ket{01001}\mapsto\ket{00001}$; $\mathsf{F}_5^\dagger \oplus I_3$, "
        r"lower three wires",
        r"$\tilde{M}_{32}^\dagger = "
        r"(A^\dagger \otimes (\mathsf{F}_5^\dagger \oplus I_3))\,Q_{32}^\dagger$"
        + NAIMARK_GLOSS,
    ],
}
# The short-atom stack beside the circuit: zero words, and it sits level with
# the boxes it names.  The A_5 pair gets B and C -- the two matrices that
# DIFFER between the pair, so they belong to the cards; the unfactored A, which
# they share, is the chapter head's (assertion 32).
SIDE_ATOMS = {
    "tetrahedron": [r"$U_A = \sqrt2\,\pmat{\alpha & \beta \\ \beta & -\alpha}$",
                    r"$\alpha = \sqrt{(3+\sqrt3)/12}$",
                    r"$\beta = \sqrt{(3-\sqrt3)/12}$",
                    r"$S = \mathrm{diag}(1,i)$, $H = \mathsf{F}_2$"],
    "octahedron": [r"$U_A = \sqrt3\,\pmat{\alpha & \beta \\ \beta & -\alpha}$",
                   r"$\alpha = \sqrt{(3+\sqrt3)/18}$",
                   r"$\beta = \sqrt{(3-\sqrt3)/18}$"],
    "cube": [r"$U_A = 2\,\pmat{\alpha & \beta \\ \beta & -\alpha}$",
             r"$\alpha = \sqrt{(3+\sqrt3)/24}$",
             r"$\beta = \sqrt{(3-\sqrt3)/24}$"],
    "icosahedron": [r"$B = \pmat{u_- & -u_+ \\ u_+ & u_-}$",
                    r"$C = \pmat{v_- & v_+ \\ v_+ & -v_-}$"],
    "dodecahedron": [r"$B = \pmat{u_- & -u_+ \\ u_+ & u_-}$",
                     r"$C = \pmat{v_- & v_+ \\ v_+ & -v_-}$"],
}
# Two per-card kappa riders ("Negative: the estimate's sign flips.", "Near
# $0$: ill-conditioned to invert.") are RETIRED.  Both leaned on an unprinted
# hypothesis: the sign flip and the inversion belong to the UNcalibrated
# read, while "never a bias" holds for the calibrated one, so beside that
# clause they read as its contradiction -- the same silent scope the label
# itself must avoid.  The sign rider also glossed the wrong object: D.4's
# caption warns that the CALIBRATION's sign flips, inviting discard as an
# artefact; the calibrated estimate never flips.  What the riders warned
# about survives on the band -- the minus sign is printed in kappa itself,
# the fatality in the premium (10621.86x), and the harmless-to-hostile
# ladder in Table D.4's caption, this band's own pointer.  Their predicate
# discipline (no card carries a warning its own number does not earn) lives
# on in assertion 36: the printed kappa must carry the recomputed sign, and
# a re-entering rider sentence is an abort.
def kappa_num(f):
    """kappa recomputed from the two vertex lists (assertion 18's own value)."""
    return float(np.mean([f["v"][j] @ f["d"][j] for j in range(f["V"])]))


def card_title(f):
    return r"The %s POVM, end to end" % SOLID_ADJ[f["solid"]]


def strip(f):
    r"""The header strip: Figure 1.1's foot strip, this solid's row, verbatim.

    The card's single structural idea.  Figure 1.1's ``\RecipeSeriesStrip``
    prints one row per solid under eight column heads; the card prints the
    same heads and the same row, minus the name column (the card's title says
    the solid), plus one spanned footer.  A reader who has met the figure
    recognises the row instead of reading it, and assertion 30 makes that
    recognition a build check instead of a hope -- cell by cell,
    string-equal.

    The footer carries the exactness conjunction, and it keeps the clause a
    compression drops: *this dilation included*, which is what binds
    Theorem 1 to the circuit printed 10cm below it.  Every card prints the
    positive with the negative, in one cell, with the negative beside it in
    ``reorientation exact?``.
    """
    spec, heads, row = series_strip_row(f["solid"])
    return [r"\newcommand{\CStrip}{{\footnotesize"
            r"\setlength{\tabcolsep}{4pt}"
            r"\renewcommand{\arraystretch}{0.88}%",
            r"\begin{tabular}{%s}" % spec,
            r"\toprule",
            " & ".join(heads) + r"\\",
            r"\midrule",
            " & ".join(row) + r"\\",
            r"\midrule",
            r"\multicolumn{%d}{@{}l@{}}{%s}\\" % (len(heads), STRIP_FOOTER),
            r"\bottomrule",
            r"\end{tabular}}}"]


def heads(f):
    """The four band heads and the five pointer slots."""
    solid = f["solid"]
    d4 = HEAD_D_NONE if not f["rays"] else HEAD_D + (
        # where a printed word is COMPOSITE the order decides the rotation
        # (FX applies X first), so the rider is emitted exactly where there is
        # a composite to disambiguate, and never as decoration (check 19c)
        ORDER_RIDER if any(" " in r["word"] for r in f["rays"]) else "")
    p4 = PTR_D + ("" if not f["rays"] else
                  r" $\cdot$ Table~\ref{%s}" % ATLAS_TAB[COVARIANCE[solid]])
    return [r"\newcommand{\CTitle}{%s}" % card_title(f),
            r"\newcommand{\CPtrStrip}{%s}" % PTR_STRIP,
            r"\newcommand{\CHeadA}{%s}" % HEAD_A,
            r"\newcommand{\CPtrA}{%s}" % PTR_A,
            r"\newcommand{\CHeadB}{%s}" % HEAD_B,
            r"\newcommand{\CPtrB}{%s}" % PTR_B,
            r"\newcommand{\CHeadC}{%s}" % (HEAD_C + (HEAD_C_DEAD if f["dead"]
                                                     else "")),
            r"\newcommand{\CPtrC}{%s}" % PTR_C,
            r"\newcommand{\CHeadD}{%s}" % d4,
            r"\newcommand{\CPtrD}{%s}" % p4]


def vertex_rows(f, with_effect):
    solid, V = f["solid"], f["V"]
    scal, _ = RADIAL[solid]
    cols = r"@{}r l c r@{}" if with_effect else r"@{}r l r@{}"
    head = [r"$k$",
            r"$\hat n_k$" + (r"\ $(\times%s)$" % scal if scal else "")]
    if with_effect:
        head.append(r"$E_k\ (\times%s)$" % EFFECT_SCALE[solid][1])
    head.append(r"$\bra{0}E_k\ket{0}$")
    return cols, head


def vertex_table(f, ks, with_effect):
    # An effect column makes every row a two-line pmatrix, and array/tabular
    # gives abutting rows no gap once the content outgrows \@arstrut: measured,
    # consecutive delimiters TOUCH and read as one continuous squiggle, and the
    # last row's meets \bottomrule.  booktabs' \addlinespace is what separates
    # them: neither \arraystretch (the strut it scales is shorter than the
    # matrices) nor \\[2.5pt] (absorbed by the pmatrix's own depth) moves them.
    cols, head = vertex_rows(f, with_effect)
    gap = r" \\" + (r" \addlinespace[3pt]" if with_effect else "")
    L = [r"\begin{tabular}{%s}" % cols, r"\toprule",
         " & ".join(head) + r" \\", r"\midrule"]
    for k in ks:
        row = [r"$%d$" % (k + 1),
               "$%s$" % tex_tuple(factored_tuple(f["solid"], f["v"][k]))]
        if with_effect:
            m = factored_effect(f["solid"], f["E"][k])
            row.append(r"$\pmat{%s & %s \\ %s & %s}$"
                       % (m[0][0], m[0][1], m[1][0], m[1][1]))
        row.append(r"$%.4f$" % f["p0"][k])
        L.append(" & ".join(row) + gap)
    L += [r"\bottomrule", r"\end{tabular}"]
    return L


def vertex_blocks(f):
    V = f["V"]
    if V <= 8:
        return [r"\newcommand{\CardVertexTable}{%"] + \
            vertex_table(f, range(V), True) + ["}"]
    half = V // 2
    out = [r"\newcommand{\CardVertexBlockA}{%"] + \
        vertex_table(f, range(half), False) + ["}", ""]
    out += [r"\newcommand{\CardVertexBlockB}{%"] + \
        vertex_table(f, range(half, V), False) + ["}"]
    return out


def frac_tex(d):
    r"""``\tfrac1d`` -- braced past nine, which is where TeX needs it."""
    return r"\tfrac1%s" % (d if d < 10 else "{%d}" % d)


def identities(f):
    """One line: the identities the vertex set satisfies, then the one thing
    about THIS solid a reader should recognise.

    The column head names its own state and the panel draws its own axes, so
    what is left for this line is the identities and the recognition.
    """
    V, solid = f["V"], f["solid"]
    assert V % 2 == 0, f"{solid}: V is odd; 2/V does not reduce"
    line = (r"$\sum_k E_k = \Id$, $\Tr E_k = %s$, rank one, a $%d$-design%s"
            % (frac_tex(V // 2), f["design"], IDENT_TAIL[solid]))
    out = [r"\newcommand{\CIdent}{%s}" % line]
    # The A_5 pair's vertex blocks carry no effect column, so the covariance
    # rule is what supplies the other V-1 effects -- and it needs one worked
    # effect to act on.  It nests under the blocks, in the slack the sphere
    # leaves, at zero words (assertion 33 rebuilds it from the npz).
    out.append(r"\newcommand{\CIdentB}{%s}" % WORKED_E1.get(solid, ""))
    return out


def key_rectangle(f):
    r"""The register key as a rectangle -- and it names its own axes.

    Rows are the upper (inter-orbit) register, columns the lower (Fourier)
    one, so the dead outcomes are visibly empty columns.  The corner stub
    reads ``upper`` and the spanned column head ``lower``: with the bit widths
    of the stubs and the heads saying which wires those are, the rectangle
    states the convention itself, and head 3 has only to say which end is
    read first.
    """
    up, ell, key = f["upper"], f["ell"], f["key"]
    L = [r"\newcommand{\CardKey}{%",
         r"\begin{tabular}{@{}c %s@{}}" % " ".join("c" * (2 ** ell)),
         r"\toprule",
         r"& \multicolumn{%d}{c}{\footnotesize lower} \\" % (2 ** ell),
         r"{\footnotesize upper} & "
         + " & ".join(r"$\mathtt{%s}$" % format(c, "0%db" % ell)
                      for c in range(2 ** ell)) + r" \\",
         r"\midrule"]
    for r_ in range(2 ** up):
        row = [key.get(r_ * 2 ** ell + c) for c in range(2 ** ell)]
        L.append(r"$\mathtt{%s}$ & " % format(r_, "0%db" % up)
                 + " & ".join(str(x) if x else "---" for x in row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", "}"]
    return L


def kappa_label(f):
    """The price of running Decker's circuit against our list.

    Both free branches first -- the card's own protocol (run U_R, read the
    key) beside D.2's do-neither branch -- then the one misread the printed
    kappa actually prices, in D.2's own words: his circuit with our list,
    index for index.  Not "the key with U_R skipped": that misread prices at
    Tr[R]/3, a different number on every card but the tetrahedron's
    (KAPPA_FMT's comment has it).  D.2 derives the tails; band 3's pointer
    carries the definition to Table D.4.
    """
    _, _, kap, _, prem = f["labels"]
    return [r"\newcommand{\CKappa}{%s}"
            % (KAPPA_FMT % (kap, prem))]


def rays(f):
    """The coin's ray list: one atlas word per vertex axis, |0>-vertex first.

    Math only, no connective -- head 4 says the reading order once, and the
    list shows one word per axis without a sentence saying so.
    """
    if not f["rays"]:
        return [r"\newcommand{\CRays}{}"]
    # The pair separator is a comma, not an en-dash: a dash reads as a range
    # ("F-dagger 1--4", and the dodecahedron's "1--8" invites eight
    # vertices).  A comma prints the two vertices as the pair head 4
    # announces, and it is NARROWER: the dodecahedral ray line, the most
    # fragile object on the five pages, measures 449.42pt at \small with the
    # dashes against the 452.9679pt line.
    return [r"\newcommand{\CRays}{%s}"
            % (r" $\cdot$ ".join(r"%s~$%d$,\,$%d$"
                                 % (r["word"], r["zero"], r["one"])
                                 for r in f["rays"]))]


def protocol(f):
    """The protocol line: what a reader does with the list above it.

    The ORDER is the drawn word first, the fixed alignment after it (Table
    D.1's own caption, and channel_R1's p(b) from A R_g r); check 19b builds
    the other order and requires it to miss the vertex set.  The alignment is
    named as a LABEL -- ``($\\Id$ here)``, ``($9 \\mapsto \\hat z$, inexact)``
    -- a label, not a sentence.

    Every card closes on its DRAW_NOTE, the rung of F.3.3's ladder that is
    this solid's: which of randomness's two jobs forces the draw to be as
    big as it is.  The three that state a minimum are scoped to GROUPS, the
    dodecahedron's names the draw that beats its own, and that draw carries
    the relabelling without which it measures nothing.  Check 19d derives
    all five.
    """
    solid = f["solid"]
    note = DRAW_NOTE[solid]
    if "%s" in note:
        # the numeral is derived, never a typed word.  On the octahedron and
        # the cube it is the length of the ray list printed directly above the
        # sentence, also the smallest realizing group's order, and 19d keeps
        # the two readings from drifting apart.  On the dodecahedron it is
        # the ENSEMBLE the sentence prices: that list times the four
        # flips, check_flip_completion's forty elements, not a group -- no
        # group below 2I realizes at all, so no subgroup order is nearby.
        n = len(f["rays"]) * (KLEIN if solid == "dodecahedron" else 1)
        note %= NUMERAL[n]
    if not f["decomposable"]:
        # The tetrahedron: the one card where the two randomized protocols
        # meet, and they must meet as a CONTRAST.  Without the second
        # sentence a reader concludes this solid admits no randomization at
        # all, which is false -- the dashed slot is drawn on its own circuit
        # one band above.
        # "No antipodal pair" would be the third statement of one fact
        # inside four centimetres -- the strip's "none: no antipodes" cell
        # and head 4's "No coin over axes" already carry it -- so what is
        # printed is the two facts neither of them states, plus the one the
        # thesis owns nowhere else: drawing anyway is not a failure, it is a
        # different POVM, and the reader can be pointed at its card.
        return [r"\newcommand{\CCoin}{The dilation is forced; a drawn $U_g$ "
                r"still twirls it.%s}" % note]
    where = (r"$\Id$ here" if f["A_is_identity"] else
             r"$%d \mapsto \hat z$, inexact" % f["align_vertex"])
    line = PROTOCOL_FMT % where
    if f["A_is_identity"]:
        # the thesis's one exact implementation, named where the reader is
        # standing in front of it: it is what the strip's "3 axes --- exact"
        # and band 1's "the Pauli eigenprojectors" add up to, and no reader
        # should have to add them
        line += r" Random Pauli measurement."
    return [r"\newcommand{\CCoin}{%s}" % (line + note)]


def card_side(f):
    """The short-atom stack beside the circuit: centred, zero words."""
    L = SIDE_ATOMS[f["solid"]]
    body = r"\NIS{4pt}".join(r"\CTR{\hsize}{%s}" % a for a in L)
    return [r"\newcommand{\CardSide}{%s}" % body]


def params(f):
    r"""The parameter block: ONE FORMULA PER LINE, at \small.

    Set as a justified paragraph, the inline 4x4 opens a four-line hole.
    One formula per line removes the hole, the 4x4 lives in the chapter head,
    and the block sets at \small rather than \footnotesize, which is what
    keeps the sub-7pt glyph count at 2 in the whole five-card set.

    Nothing may be glued to the right of the U_R atom: at \footnotesize the
    octahedron's product measured 181.64pt bare and 190.89pt with an em-dash
    attached.  Every rider follows the CLAUSE, never the formula.
    """
    ur = f["UR_tex"]
    # The reorientation clause.  Where the drawing already names the box in
    # full (\gate{U_R = T^\dagger}) the line does not print the formula a
    # second time.
    drawn_in_full = (r"\gate{U_R = %s}" % ur) in uncommented(f["circuit_src"])
    if drawn_in_full:
        lead = r"$U_R$"
    else:
        # a product of three rotations does not say which acts first, and read
        # the other way it is a different rotation
        order = (", rightmost factor first,"
                 if len(re.findall(r"R_\{?\\hat\s*([zy])\}?\(", ur)) > 1
                 else "")
        lead = r"$U_R = %s$%s" % (ur, order)
    L = [lead + r" inverts Table~\ref{tab:decker-pose}'s $R$; " + UG_RIDER]
    L += PARAM_LINES[f["solid"]]
    return [r"\newcommand{\CParams}{%s}" % r"\par\vskip1pt ".join(L)]


def caption(f):
    """The caption, one sentence and one line on all five.

    No bold lead (the card's own \\large\\bfseries title stands at the head of
    the same float, and the thesis has no \\listoffigures), no provenance tail
    (the pointer column carries E.1, D.4 and D.2 at the heads that use them).
    Emitted here so that the generator owns the one cold name on the page and
    can check it against bsc-thesis.tex; the float itself carries the text.
    """
    return CAPTION_FMT % DECKER_FIG[f["solid"]]


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------

def _brace(t, k):
    """(contents, index after) for the ``{...}`` (or single token) at t[k]."""
    if t[k] != "{":
        return t[k], k + 1
    depth, j = 1, k + 1
    while depth:
        assert j < len(t), f"unbalanced braces in {t!r}"
        depth += (t[j] == "{") - (t[j] == "}")
        j += 1
    return t[k + 1:j - 1], j


def _tex_number(t):
    r"""Evaluate a printed closed form: digits, ``\sqrt``, ``\tfrac``, + - * /.

    Rewrites the LaTeX into ordinary arithmetic one construct at a time,
    innermost first, and then refuses anything that is not arithmetic -- so an
    unrecognised macro is an error, never a silently dropped factor.
    """
    t = t.replace(" ", "")
    # \phig and \invphig are the thesis's golden ratio and its inverse
    # (Conway and Smith's tau and sigma, defined in bsc-thesis.tex's
    # preamble).  They are
    # rewritten to arithmetic HERE rather than admitted as opaque symbols, so
    # the "refuses anything that is not arithmetic" guarantee below still
    # holds: an unrecognised macro is still an error.
    t = t.replace(r"\invphig", "((sqrt(5)-1)/2)").replace(r"\phig",
                                                          "((1+sqrt(5))/2)")
    for mac, fmt in ((r"\dfrac", "(({0})/({1}))"),
                     (r"\tfrac", "(({0})/({1}))"), (r"\frac", "(({0})/({1}))"),
                     (r"\sqrt", "sqrt({0})")):
        while mac in t:
            k = t.index(mac)
            a, j = _brace(t, k + len(mac))
            if fmt.count("{1}"):
                b, j = _brace(t, j)
                t = t[:k] + fmt.format(_tex_number(a), _tex_number(b)) + t[j:]
            else:
                t = t[:k] + fmt.format(_tex_number(a)) + t[j:]
    t = re.sub(r"(\d|\))(?=\()", r"\1*", t)
    t = re.sub(r"(\d|\))(?=sqrt)", r"\1*", t)
    t = re.sub(r"\)(?=\d)", r")*", t)
    assert re.fullmatch(r"[-+*/().\d]*", t.replace("sqrt", "")), \
        f"unparsed closed form {t!r}"
    def _sqrt(x):
        # a printed closed form that has gone imaginary is a corrupted printed
        # closed form; say so, rather than leaking a nan into the next eval
        assert float(x) >= 0.0, f"sqrt of {x} in the printed closed form {t!r}"
        return float(x) ** 0.5
    val = float(eval(t, {"__builtins__": {}}, {"sqrt": _sqrt}))
    assert np.isfinite(val), f"closed form {t!r} is not a real number"
    return "%.17g" % val


def _tex_complex(t):
    r"""Evaluate a printed closed form that may be complex: ``i`` admitted.

    ``_tex_number``'s rewriting, plus the imaginary unit, for the one place a
    card prints a complex closed form -- the worked $E_1$ on the two $A_5$
    cards (assertion 33).  Same guarantee: an unrecognised macro is an error,
    never a silently dropped factor.
    """
    t = t.replace(" ", "").replace("\\,", "")
    t = t.replace(r"\invphig", "((sqrt(5)-1)/2)").replace(r"\phig",
                                                          "((1+sqrt(5))/2)")
    for mac, fmt in ((r"\dfrac", "(({0})/({1}))"),
                     (r"\tfrac", "(({0})/({1}))"), (r"\frac", "(({0})/({1}))"),
                     (r"\sqrt", "sqrt({0})")):
        while mac in t:
            k = t.index(mac)
            a, j = _brace(t, k + len(mac))
            if fmt.count("{1}"):
                b, j = _brace(t, j)
                t = t[:k] + fmt.format(_tex_complex(a), _tex_complex(b)) + t[j:]
            else:
                t = t[:k] + fmt.format(_tex_complex(a)) + t[j:]
    t = re.sub(r"(?<![a-z])i(?![a-z])", "1j", t)
    t = re.sub(r"(\d|\))(?=\()", r"\1*", t)
    t = re.sub(r"(\d|\))(?=sqrt)", r"\1*", t)
    t = re.sub(r"\)(?=\d)", r")*", t)
    assert re.fullmatch(r"[-+*/().\dj]*", t.replace("sqrt", "")), \
        f"unparsed closed form {t!r}"

    def _sqrt(x):
        assert complex(x).imag == 0 and complex(x).real >= 0.0, \
            f"sqrt of {x} in the printed closed form {t!r}"
        return complex(x) ** 0.5
    val = complex(eval(t, {"__builtins__": {}}, {"sqrt": _sqrt}))
    assert np.isfinite(val.real) and np.isfinite(val.imag), \
        f"closed form {t!r} is not a number"
    return val


def _pm_radical(tex):
    r"""``\tfrac{1}{10}\sqrt{50 \pm 5\sqrt{10(5+\sqrt5)}}`` -> (upper, lower).

    Substitutes the two signs of the one ``\pm``/``\mp`` pair and evaluates
    both, so that four hand-typed radicals on the two A_5 cards are checked as
    numbers rather than trusted as strings.
    """
    def one(sign):
        t = tex.replace(r"\pm", "+" if sign > 0 else "-") \
               .replace(r"\mp", "-" if sign > 0 else "+")
        return float(_tex_number(t))
    assert r"\pm" in tex or r"\mp" in tex, f"no sign pair in {tex!r}"
    return one(+1), one(-1)


def rotation_from_tex(s):
    r"""``R_{\hat z}(180^\circ)R_{\hat y}(-\arccos(1/\sqrt3))`` -> SO(3).

    Parses the rotation product a file WRITES, so that assertion 17 compares
    Table D.3's algebra with the picture the card draws rather than with a
    retyping of either.  Table D.3 writes ``R_z``, the circuits ``R_{\hat z}``;
    both are the same rotation and both are read here.  Angles only in the
    closed forms the two files use -- an unrecognised one is an error, never a
    silently skipped factor.
    """
    tab = ANGLE_ARGS

    def ang(txt):
        txt = txt.strip()
        sign = -1.0
        if txt.startswith("-"):
            txt, sign = txt[1:].strip(), -1.0
        else:
            sign = 1.0
        m = re.fullmatch(r"(\d+)\^\\circ", txt)
        if m:
            return sign * np.radians(float(m.group(1)))
        m = re.fullmatch(r"\\arccos\((.*)\)", txt)
        assert m and m.group(1) in tab, f"unparsed angle {txt!r}"
        return sign * np.arccos(tab[m.group(1)])

    def rz(a):
        return np.array([[np.cos(a), -np.sin(a), 0],
                         [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    def ry(a):
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                         [-np.sin(a), 0, np.cos(a)]])

    txt = s.replace(r"R_{\hat z}", "R_z").replace(r"R_{\hat y}", "R_y")
    out, seen, i, spans = np.eye(3), 0, 0, []
    while True:
        m = re.compile(r"R_([zy])\(").search(txt, i)
        if not m:
            break
        depth, j = 1, m.end()
        while depth:
            assert j < len(txt), f"unbalanced parentheses in {s!r}"
            depth += (txt[j] == "(") - (txt[j] == ")")
            j += 1
        out = out @ (rz if m.group(1) == "z" else ry)(ang(txt[m.end():j - 1]))
        spans.append((m.start(), j))
        seen, i = seen + 1, j
    assert seen, f"no rotation factors in {s!r}"
    # every character outside the factors read must be punctuation: a product of
    # rotations and nothing else, so no factor can be silently skipped
    rest, k = "", 0
    for a, b in spans:
        rest, k = rest + txt[k:a], b
    rest += txt[k:]
    assert not rest.strip("$ \t"), f"unread material {rest!r} in {s!r}"
    return out


def _circuit_grid(text):
    r"""(header, rows of cells, gategroups) of a ``\Qcircuit`` body.

    Every ``smallmatrix`` is masked out before the split, because the A_5
    circuits draw rotation boxes that carry ``&`` between their entries and
    ``\\`` between their rows -- split naively, the icosahedron's four wires
    come out as eleven.  Each ``\gategroup`` is lifted off its cell and
    returned separately, so a cell can be compared for what it DRAWS while its
    fencing is compared as six numbers.
    """
    t = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))
    t = t[t.index(r"\Qcircuit"):t.rindex("}")]
    head, body = t.split("{", 1)
    masks, groups = [], []

    def mask(m):
        masks.append(m.group(0))
        return "\x00%d\x00" % (len(masks) - 1)

    def grab(m):
        groups.append(list(m.groups()))
        return "\x01%d\x01" % (len(groups) - 1)

    body = re.sub(r"\\begin\{smallmatrix\}.*?\\end\{smallmatrix\}", mask, body,
                  flags=re.S)
    body = re.sub(r"\\gategroup\{(\d+)\}\{(\d+)\}\{(\d+)\}\{(\d+)\}"
                  r"\{([^{}]*)\}\{([^{}]*)\}", grab, body)
    rows, at = [], []
    for r_, line in enumerate(body.split(r"\\")):
        cells, here = [], []
        for c_, cell in enumerate(line.split("&")):
            for g in re.findall(r"\x01(\d+)\x01", cell):
                here.append((r_, c_, groups[int(g)]))
            cell = re.sub(r"\x01\d+\x01", "", cell)
            cell = re.sub(r"\x00(\d+)\x00",
                          lambda m: masks[int(m.group(1))], cell)
            cells.append(" ".join(cell.split()))
        rows.append(cells)
        at += here
    return " ".join(head.split()), rows, at


def token_diff_circuit(card, frozen, solid, report):
    r"""Assertion 14: the pose is unchanged, enforced by a cell-by-cell diff.

    The card circuit and the frozen ``circuit_dec_<solid>.tex`` are parsed
    into grids and compared cell for cell.  Exactly these differences are
    admissible, and they are checked rather than tolerated:

      (i)   one inserted column 2, carrying ``\gate{U_g}`` on the data row and
            ``\qw`` on every other wire, with its own dashed ``\gategroup``;
      (ii)  the reorientation cell's label becoming ``U_R`` (octahedron,
            icosahedron, dodecahedron) or ``U_R = T^\dagger`` (tetrahedron,
            cube -- the two exact ones, which must keep drawing the gate);
      (iii) the solid ``\gategroup``'s two column indices shifted by exactly
            +1, its rows and its style untouched, and still fencing the same
            cell of the published block;
      (iv)  on the A_5 pair only, the wire spacing ``@R``, and only downwards.
            Nothing else in the header may move -- ``@C`` included.

    Any other difference, anywhere, is a failure: Decker's published pose is
    reoriented by $U_R$ and never re-drawn, and this is that rule made
    mechanical.
    """
    ha, A, ga = _circuit_grid(frozen)
    hb, B, gb = _circuit_grid(card)
    # --- (iv) the header ----------------------------------------------------
    ca, ra = re.search(r"@C=([\d.]+em)", ha), re.search(r"@R=([\d.]+em)", ha)
    cb, rb = re.search(r"@C=([\d.]+em)", hb), re.search(r"@R=([\d.]+em)", hb)
    assert ca and ra and cb and rb, f"{solid}: unreadable \\Qcircuit header"
    assert ca.group(1) == cb.group(1), \
        f"{solid}: column spacing {ca.group(1)} -> {cb.group(1)}"
    assert ha.replace(ra.group(1), "R") == hb.replace(rb.group(1), "R"), \
        f"{solid}: the \\Qcircuit header changed beyond @R"
    spacing = ""
    if ra.group(1) != rb.group(1):
        assert float(rb.group(1)[:-2]) < float(ra.group(1)[:-2]), \
            f"{solid}: @R grew, {ra.group(1)} -> {rb.group(1)}"
        spacing = f", wire spacing {ra.group(1)} -> {rb.group(1)}"
    # --- (i) the inserted column -------------------------------------------
    assert len(A) == len(B), f"{solid}: {len(A)} wires -> {len(B)}"
    data = len(B) - 1                       # the rho wire is the last row
    for r_ in range(len(B)):
        assert len(B[r_]) == len(A[r_]) + 1, \
            f"{solid}: wire {r_+1} has {len(B[r_])} cells, not {len(A[r_])+1}"
        want = r"\gate{U_g}" if r_ == data else r"\qw"
        assert B[r_][1] == want, \
            f"{solid}: wire {r_+1}'s inserted cell is {B[r_][1]!r}, not {want!r}"
    # --- (ii) the reorientation, and (i)'s "and nothing else" --------------
    # cell 0 is the \lstick label and does not move; the new column is cell 1,
    # so every published cell from 1 on sits one to the right on the card
    for r_ in range(len(B)):
        assert A[r_][0] == B[r_][0], \
            f"{solid}: wire {r_+1}'s \\lstick label changed"
    reo = None
    for c_ in range(1, len(A[data])):
        if A[data][c_].startswith(r"\gate{") and B[data][c_ + 1] != A[data][c_]:
            assert reo is None, f"{solid}: two gates changed on the data wire"
            reo = c_
    assert reo is not None, f"{solid}: the reorientation gate did not change"
    old = re.fullmatch(r"\\gate\{(.+)\}", A[data][reo]).group(1)
    new = re.fullmatch(r"\\gate\{(.+)\}", B[data][reo + 1]).group(1)
    assert new == (r"U_R = T^\dagger" if r"T^\dagger" in old else "U_R"), \
        f"{solid}: the reorientation is drawn {new!r}; an exact one must " \
        f"still print its gate, an inexact one must print only $U_R$"
    for r_ in range(len(B)):
        for c_ in range(1, len(A[r_])):
            if r_ == data and c_ == reo:
                continue
            assert A[r_][c_] == B[r_][c_ + 1], \
                f"{solid}: wire {r_+1} cell {c_+1}: {A[r_][c_]!r} -> " \
                f"{B[r_][c_+1]!r}"
    # --- (iii) the gategroups ----------------------------------------------
    assert len(ga) == 1, f"{solid}: {len(ga)} gategroups in the frozen circuit"
    assert len(gb) == 2, f"{solid}: {len(gb)} gategroups on the card"
    (ra_, ca_, pa), = ga
    solidb = [g for g in gb if g[2][5] == "-"]
    dashb = [g for g in gb if g[2][5] == "--"]
    assert len(solidb) == 1 and len(dashb) == 1, f"{solid}: gategroup styles"
    rb_, cb_, pb = solidb[0]
    assert (rb_, cb_) == (ra_, ca_ + 1), \
        f"{solid}: the solid gategroup moved off its cell"
    ia, ib = [int(x) for x in pa[:4]], [int(x) for x in pb[:4]]
    assert ib == [ia[0], ia[1] + 1, ia[2], ia[3] + 1] and pa[4:] == pb[4:], \
        f"{solid}: solid gategroup {pa} -> {pb}, not a +1 column shift"
    assert ib[1] > 3, \
        f"{solid}: the solid gategroup starts at column {ib[1]}, so one of " \
        f"the two new gates is inside Decker's published block"
    dr, dc, dp = dashb[0]
    assert (dr, dc) == (data, 1), f"{solid}: the dashed gategroup is not on " \
        f"the inserted cell of the data wire"
    assert [int(x) for x in dp[:4]] == [data + 1, 2, data + 1, 2], \
        f"{solid}: the dashed gategroup fences {dp[:4]}, not the U_g cell"
    report.append(f"  14. {solid}: pose unchanged -- {len(B)} wires diffed "
                  f"cell by cell, solid gategroup {ia} -> {ib} (+1 columns, "
                  f"same cell), U_g dashed at column 2 outside it, "
                  f"reorientation {old!r} -> {new!r}{spacing}, nothing else "
                  f"differs")


def verify(F, bodies, report):
    """Every assertion runs before a byte is written; a failure writes nothing."""
    atlas_e1 = read_povm_atlas()
    labels = read_labels_table()
    chapter = (PAPER / "bsc-thesis.tex").read_text()

    # --- 1. the POVM axioms, the design strengths, the SIC ------------------
    designs = []
    for s in SOLIDS:
        f = F[s]
        E, V = f["E"], f["V"]
        assert np.abs(sum(E) - I2).max() < TOL, f"{s}: effects do not resolve 1"
        assert max(abs(np.trace(E[k]) - 2 / V) for k in range(V)) < TOL, \
            f"{s}: trace is not 2/V"
        assert all(np.linalg.matrix_rank(E[k], tol=1e-9) == 1 for k in range(V)), \
            f"{s}: some effect is not rank one"
        designs.append(f["design"])
    assert designs == [2, 3, 3, 5, 5], f"design strengths {designs}"
    ov = [abs(4 * np.trace(F["tetrahedron"]["E"][j] @ F["tetrahedron"]["E"][k]).real)
          for j in range(4) for k in range(j + 1, 4)]
    assert len(ov) == 6 and max(abs(o - 1 / 3) for o in ov) < 1e-14, "not a SIC"
    for s in SOLIDS[1:]:
        E, V = F[s]["E"], F[s]["V"]
        o = [abs(V * np.trace(E[j] @ E[k]).real) / 2
             for j in range(V) for k in range(j + 1, V)]
        assert max(abs(x - 1 / 3) for x in o) > 1e-6, f"{s} is a SIC too?"
    report.append("  1. axioms on all five: sum E = 1, Tr E = 2/V, rank one; "
                  "designs 2/3/3/5/5; the tetrahedron alone a SIC")

    # --- 2. the printed factored matrices, atom by atom ---------------------
    worst = 0.0
    for s in SOLIDS:
        f = F[s]
        if f["V"] <= 8:
            for k in range(f["V"]):
                back = unfactor_effect(s, factored_effect(s, f["E"][k]))
                worst = max(worst, np.abs(back - f["E"][k]).max())
        else:                       # the one printed general form, for every k
            for k in range(f["V"]):
                gen = (I2 + sum(f["v"][k][a] * PAULI[a] for a in range(3))) / f["V"]
                worst = max(worst, np.abs(gen - f["E"][k]).max())
    assert worst < 1e-15, f"printed effects differ from the npz by {worst:.1e}"
    report.append(f"  2. printed effects vs npz on all five: worst {worst:.1e} "
                  f"(V<=8 atom by atom; V>=12 the general form, every k)")

    # --- 3. <0|E_k|0>, three ways, and the column sum -----------------------
    # This block is the pattern the rest of the printed surface follows in
    # assertion 29: the cells are parsed back OUT of the emitted string and
    # compared to the object.  (2 and 6 re-derive; 29 is what reads the file.)
    rho0 = np.diag([1, 0]).astype(complex)
    for s in SOLIDS:
        f = F[s]
        p = np.array([np.trace(f["E"][k] @ rho0).real for k in range(f["V"])])
        assert max(abs(p[k] - f["E"][k][0, 0].real) for k in range(f["V"])) < TOL
        assert max(abs(p[k] - f["p0"][k]) for k in range(f["V"])) < TOL, \
            f"{s}: <0|E|0> is not (1 + z_k)/V"
        assert abs(p.sum() - 1) < TOL, f"{s}: the column sums to {p.sum()}"
        assert abs(sum(float("%.4f" % x) for x in p) - 1) < 5e-4, \
            f"{s}: the PRINTED column does not sum to 1"
        # ... and the column as EMITTED, cell by cell: comparing two
        # computed arrays and re-rounding would let a %.3f or a swapped row
        # ship.  The tuples (6) and the effect matrices (2) are parsed back
        # the same way.
        tbl = " ".join(l for l in F[s]["emitted"]
                       if r"$\bra{0}E_k\ket{0}$" in l or " & $" in l)
        cells = re.findall(r"&\s*\$(-?\d\.\d+)\$\s*\\\\", tbl)
        assert len(cells) == f["V"], \
            f"{s}: parsed {len(cells)} printed <0|E|0> cells, not {f['V']}"
        for k, c in enumerate(cells):
            assert len(c.split(".")[1]) == 4, f"{s}: cell {c} is not 4 places"
            assert abs(float(c) - round(f["p0"][k], 4)) < 1e-12, \
                f"{s}: printed <0|E_{k+1}|0> = {c}, computed {f['p0'][k]}"
    report.append("  3. <0|E_k|0> = (E_k)_00 = Tr[E_k |0><0|] = (1+z_k)/V on "
                  "all five, the exact column sums to 1, and every printed "
                  "cell parses back to four places")

    # --- 5. covariance ------------------------------------------------------
    for s in SOLIDS:
        f = F[s]
        atl = load_atlas(COVARIANCE[s])
        worst5 = 0.0
        for U in atl["unitaries"]:
            U = np.array(U)
            for k in range(f["V"]):
                Ek = U @ f["E"][k] @ U.conj().T
                worst5 = max(worst5, min(np.abs(Ek - f["E"][j]).max()
                                         for j in range(f["V"])))
        assert worst5 < 1e-12, f"{s}: covariance misses by {worst5:.1e}"
    report.append("  5. covariance E_g(k) = U_g E_k U_g^dagger over every g of "
                  "COVARIANCE[solid] and every k, all five")

    # --- 6. the tuples, and Table E.1 cell for cell -------------------------
    for s in SOLIDS:
        f = F[s]
        scal, unit = RADIAL[s]
        e1_scal, e1_rows = atlas_e1[s]
        want = (None if scal is None
                else scal.replace(r"\phig", r"\tau").replace("tfrac", "tfrac"))
        got = None if e1_scal is None else e1_scal
        if want is None:
            assert got is None, f"{s}: Table E.1 factors a scalar we do not"
        else:
            assert got is not None and got.replace(r"\sqrt{3}", r"\sqrt3") \
                .replace(r"\sqrt{2+\tau}", r"\sqrt{2+\phig}") == \
                scal.replace(r"\sqrt{3}", r"\sqrt3"), \
                f"{s}: radial scalar {got!r} against ours {scal!r}"
        for k in range(f["V"]):
            mine = factored_tuple(s, f["v"][k])
            theirs = [(sg, at.replace(r"\tau", r"\phig")
                       .replace(r"\sigma", r"\invphig")) for sg, at in e1_rows[k + 1]]
            assert mine == theirs, \
                f"{s} vertex {k+1}: {mine} against Table E.1's {theirs}"
            back = np.array([(1 if sg == "+" else -1) * ATOMS[at] * unit
                             for sg, at in mine])
            assert np.abs(back - f["v"][k]).max() < 1e-12, \
                f"{s} vertex {k+1}: the printed tuple does not parse back"
    report.append("  6. every printed tuple parses back to the npz and equals "
                  "Table E.1's cell for cell, all 50 vertices")

    # --- 7. the coin words against atlas.tex --------------------------------
    atlas_tex = (CODE / "atlas.tex").read_text()
    gates = np.load(DATA / "gates.npz", allow_pickle=True)
    tab = gate_table(gates)
    n_words = 0
    for s in SOLIDS:
        f = F[s]
        if not f["rays"]:
            continue
        for r in f["rays"]:
            # atlas.tex writes the dagger braced; latex_word does not
            w = r["word"].strip("$").replace(r"^\dagger", r"^{\dagger}")
            w = w.replace(" ", "")
            assert w == r"\Id" or w in atlas_tex.replace(" ", ""), \
                f"{s}: coin word {r['word']} is not a row of atlas.tex"
            # the atlas writes Phi with the Greek letter; gates.npz names it
            seq = (r["seq"] or "I").replace("\u03a6", "Phi")
            U = word_matrix(seq, tab)
            assert np.abs(bloch(U) - r["Rg"]).max() < 1e-9, \
                f"{s}: {r['word']} does not multiply out to its rotation"
            n_words += 1
        phis = sum(r["phi"] for r in f["rays"])
        d2 = f["coins"]["words"]
        assert sum(1 for w in d2 if r"\Phi" in w) == phis, \
            f"{s}: Phi count {phis} against Table D.1"
    report.append(f"  7. all {n_words} coin words are atlas.tex rows, multiply "
                  f"out to their rotations, and carry Table D.1's Phi count")

    # --- 8. bold by the claim -----------------------------------------------
    for s in SOLIDS:
        blob = " ".join(F[s]["emitted"])
        for claim in re.findall(r"[^.]*turns? the[^.]*\.", blob):
            assert r"\mathbf" not in claim, f"{s}: bold in a rotation claim"
        assert r"\mathbf{F}" not in blob or "HS" not in blob, \
            f"{s}: bold F appears beside HS-dagger"
        for m in re.finditer(r"(\\Id|\\mathbf\{[XZFHS]\})", blob):
            pass
        # every atlas word printed is bold (an SU(2) claim), Phi excepted
        for r in (F[s]["rays"] or []):
            w = r["word"]
            assert w == r"$\Id$" or r"\mathbf" in w or r"\Phi" in w, \
                f"{s}: atlas word {w} is not set as an SU(2) claim"
    report.append("  8. bold by the claim over every emitted string: atlas "
                  "words bold, no bold symbol inside a rotation claim")

    # --- 9. the rebuilt circuit -> our effects ------------------------------
    for s in SOLIDS:
        f = F[s]
        Rb = f["R"]
        # the rebuilt row's Bloch vector, reoriented, IS the npz vertex it is
        # printed against -- so the circuit drawn measures our effects
        for i, x in zip(f["live"], f["d"]):
            k = f["key"][i] - 1
            assert np.abs(Rb @ x - f["v"][k]).max() < 1e-9, \
                f"{s}: outcome {i} does not land on vertex {k+1}"
            Ek = (I2 + sum((Rb @ x)[a] * PAULI[a] for a in range(3))) / f["V"]
            assert np.abs(Ek - f["E"][k]).max() < 1e-12, \
                f"{s}: outcome {i}'s effect is not npz element {k+1}"
        # ... and WITHOUT the reorientation it measures the published pose,
        # which is not ours: the sentence "U_R reorients the published circuit"
        # checked off the drawing rather than asserted
        assert min(np.abs(x - f["v"][j]).max() for x in f["d"]
                   for j in range(f["V"])) > 1e-3, \
            f"{s}: the box as published already measures our pose"
    report.append("  9. the rebuilt Mtilde-dagger.iota, reoriented by Table "
                  "D.3's R, reproduces npz['elements'] for every live outcome "
                  "on all five; unreoriented it reproduces none")

    # --- 10. the counts -----------------------------------------------------
    ledger = read_ledger()
    for i, s in enumerate(SOLIDS):
        f = F[s]
        assert f["V"] == int(ledger[r"Vertices $V$"][i]), f"{s}: V vs Table 5.2"
        assert f["reg"] == int(re.search(r"\((\d+)\)",
                                         ledger["Naimark register"][i]).group(1))
        assert len(f["dead"]) == int(ledger["Dead outcomes"][i]) \
            == f["reg"] - f["V"], f"{s}: dead count"
        assert f["wires"] == f["n"], \
            f"{s}: {f['wires']} \\lstick against {f['n']} register bits"
        assert f["wires"] == uncommented(f["frozen_src"]).count(r"\lstick"), \
            f"{s}: the card's wire count is not the published circuit's"
    assert [len(F[s]["dead"]) for s in SOLIDS] == [0, 2, 0, 4, 12], "dead moved"
    assert "$2$, $4$, and $12$ of the basis outcomes" in chapter, \
        "the chapter head no longer says 2, 4 and 12"
    report.append("  10. V, register 2^n, wires and dead counts agree across "
                  "the npz, Table 5.2, decker_circuit and the card circuits' "
                  "own \\lstick counts; the head's \"2, 4 and 12\" stands")

    # --- 11. direction ------------------------------------------------------
    for s in SOLIDS:
        blob = " ".join(F[s]["emitted"])
        assert "acts on" not in blob, f"{s}: an emitted fragment says 'acts on'"
        assert r"relabels outcomes by $g^{-1}$" in blob, \
            f"{s}: the parameter block does not state the direction as g^-1"
        assert r"relabels outcomes by $g$" not in blob and \
            r"rotates the effects by $g$" not in blob, f"{s}: direction is g"
    for name, text in bodies.items():
        # over the DRAWING, not the header comment -- which quotes both rules
        drawn = uncommented(text)
        assert "acts on" not in drawn, f"{name} says 'acts on'"
        assert not re.search(r"^\\textit\{", drawn, re.M), \
            f"{name} has an italic run-in head"
    report.append("  11. the printed relabelling is g^{-1} on all five; "
                  "\"acts on\" in no emitted fragment and no body file; no "
                  "italic run-in heads")

    # --- 13. the circuits rebuilt, matched, anchored ------------------------
    for s in SOLIDS:
        f = F[s]
        cols = decker_columns(s)
        assert len(f["live"]) == len(cols) == f["V"], f"{s}: live count"
        assert f["live"] == sorted(f["live"]), f"{s}: padding reordered outcomes"
        assert sorted(f["key"]) == f["live"], f"{s}: key keys are not the live rows"
        for j in range(f["V"]):
            a = f["d"][j]
            b = np.array([2 * (np.conj(cols[j][0]) * cols[j][1]).real,
                          2 * (np.conj(cols[j][0]) * cols[j][1]).imag,
                          abs(cols[j][0]) ** 2 - abs(cols[j][1]) ** 2])
            b = b / np.linalg.norm(b)
            assert np.abs(a - b).max() < 1e-10, f"{s}: column {j} differs"
        assert np.abs(f["d"][0] - DECKER_ANCHOR[s]).max() < 1e-10, \
            f"{s}: outcome 0 is not Decker's stated vertex 1"
        assert np.abs(sum(np.outer(np.conj(r), r) for r in f["W"])
                      - I2).max() < 1e-12, f"{s}: the rows are not a POVM"
    report.append("  13. all five circuits rebuilt: live rows the printed "
                  "register values in order, decker_columns value for value in "
                  "Decker's order, DECKER_ANCHOR matched, every match under "
                  "the margin discipline (< 1e-9, runner-up > 0.1)")

    # --- 14. the pose, by token diff ----------------------------------------
    for s in SOLIDS:
        if F[s]["has_card_circuit"]:
            token_diff_circuit(F[s]["circuit_src"], F[s]["frozen_src"], s, report)
        else:
            report.append(f"  14. {s}: SKIPPED -- card_dec_{SHORT[s]}.tex does "
                          f"not exist yet")

    # ... and the two A_5 circuits print at ONE wire spacing.  The cut is
    # @R=1.5em -> 1.2em on the dodecahedron, the binding card; the icosahedron
    # does not need the 9.8pt, but the pair is read side by side and an
    # asymmetry in wire spacing would be the first thing a reader compared.
    # ... off the DRAWING, never off the file's own header: that header writes
    # the cut as "@R=1.5em -> 1.2em", so a search over the raw text reports the
    # spacing the card was cut FROM and a real asymmetry would pass unseen.
    rs = {s: re.search(r"@R=([\d.]+em)",
                       uncommented(F[s]["circuit_src"])).group(1)
          for s in SOLIDS if F[s]["has_card_circuit"]}
    a5 = [rs[s] for s in ("icosahedron", "dodecahedron") if s in rs]
    assert len(set(a5)) <= 1, f"the A_5 cards draw at {a5}, two spacings"
    if len(a5) == 2:
        report.append(f"  14. the A_5 pair prints at one wire spacing, {a5[0]}")

    # --- 15. the key, derived twice -----------------------------------------
    for s in SOLIDS:
        f = F[s]
        top, bot, _, _, _ = labels[s]
        assert top == sorted(f["key"]), \
            f"{s}: Table D.4's live outcomes {top} against ours {sorted(f['key'])}"
        assert bot == [f["key"][i] for i in top], \
            f"{s}: Table D.4 maps {bot}, we derived {[f['key'][i] for i in top]}"
        # the conjugate trap, made into a check: the same rebuild WITHOUT the
        # np.conj decker_vertices applies must NOT reproduce the table
        # bloch_of_ket(W[k]) instead of bloch_of_ket(conj(W[k])): the y
        # component flips, every Fourier orbit reverses, and the key that comes
        # out is a permutation of the right values -- plausible, and wrong.
        raw = [np.array([2 * (np.conj(r[0]) * r[1]).real,
                         +2 * (np.conj(r[0]) * r[1]).imag,
                         abs(r[0]) ** 2 - abs(r[1]) ** 2]) for r in f["W"]]
        wrong = []
        for i in f["live"]:
            x = raw[i] / np.linalg.norm(raw[i])
            d = np.linalg.norm(f["v"] - f["R"] @ x, axis=1)
            wrong.append(int(np.argmin(d)) + 1)
        if s != "tetrahedron":
            assert wrong != bot, \
                f"{s}: the conjugate-blind rebuild agrees, so 15 tests nothing"
    report.append("  15. the key derived twice and equal on all five -- once "
                  "from decker_vertices + REORIENT, once parsed out of "
                  "Table D.4's printed column (its % key twin asserted equal "
                  "in the parse); the conjugate-blind rebuild is shown "
                  "to disagree on four of five, which is what makes it a check")

    # --- 16. the dead rule ---------------------------------------------------
    for s in SOLIDS:
        f = F[s]
        assert f["m"] == REORIENT[s][0], \
            f"{s}: the circuit's Fourier size {f['m']} against REORIENT's " \
            f"{REORIENT[s][0]}"
        assert 2 ** f["ell"] >= f["m"], f"{s}: the transform does not fit"
        rule = {i for i in range(f["reg"]) if (i % 2 ** f["ell"]) >= f["m"]}
        assert rule == set(f["dead"]), \
            f"{s}: dead {sorted(f['dead'])} against the rule {sorted(rule)}"
    report.append("  16. dead == {n : n mod 2^ell >= m} on all five, ell parsed "
                  "off each circuit's own Fourier gate; counts 0/2/0/4/12")

    # --- 17. U_R = R^{-1} ----------------------------------------------------
    for s in SOLIDS:
        f = F[s]
        Rd3 = rotation_from_tex(f["pose"]["R"].split(",")[0])
        assert np.abs(Rd3 - f["R"]).max() < 1e-12, \
            f"{s}: Table D.3's R is not REORIENT's"
        if f["exact_reorientation"]:
            # drawn T-dagger; its Bloch rotation is R^{-1} by construction
            Td = np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex)
            assert np.abs(bloch(Td) - f["R"].T).max() < 1e-12, \
                f"{s}: T-dagger's Bloch rotation is not R^{{-1}}"
        else:
            got = rotation_from_tex(f["UR_tex"])
            assert np.abs(got - f["R"].T).max() < 1e-12, \
                f"{s}: the drawn U_R is not R^{{-1}} -- an angle's sign is " \
                f"the single most plausible silent error on a card"
        # The card prints the drawing's own string, character for
        # character -- once.  Two of the five draw the box in full
        # (\gate{U_R = T^\dagger}), and there the parameter line says only
        # "$U_R$ inverts ...", because saying it twice is a repetition.  So
        # the formula is required on the page, and required NOT to be printed
        # a second time.
        page17 = " ".join(f["emitted"]) + " " + uncommented(f["circuit_src"])
        assert page17.count(r"$U_R = %s$" % f["UR_tex"]) \
            + page17.count(r"\gate{U_R = %s}" % f["UR_tex"]) == 1, \
            f"{s}: the drawn U_R is printed 0 or 2 times, not once"
        # ... and the box really does carry the published solid onto the atlas
        for i, x in zip(f["live"], f["d"]):
            assert np.abs(f["R"] @ x - f["v"][f["key"][i] - 1]).max() < 1e-9
    report.append("  17. U_R = R^{-1} for Table D.3's R on all five, parsed "
                  "from the drawing and from the table, with D.3's sign "
                  "convention; and R carries the published solid onto the atlas")

    # --- 18. kappa, character for character ---------------------------------
    for s in SOLIDS:
        f = F[s]
        _, _, kap, rider0, prem = f["labels"]
        emitted = " ".join(f["emitted"])
        assert r"$\kappa = %s$" % kap in emitted, \
            f"{s}: kappa {kap!r} is not printed character for character"
        # the premium prints inside the 1/kappa^2 atom
        assert r"= %s\times$" % prem in emitted, f"{s}: premium {prem!r}"
        # kappa prices the circuit run AS PUBLISHED with outcomes read
        # against our list index for index: the overlap of the believed vertex
        # (ours, in order) with the measured one (his, in his order).
        k_num = float(np.mean([f["v"][j] @ f["d"][j] for j in range(f["V"])]))
        shown = (float(kap.split("/")[0]) / float(kap.split("/")[1])
                 if "/" in kap else float(kap))
        assert abs(k_num - shown) < 5e-7, \
            f"{s}: recomputed kappa {k_num} against the printed {kap}"
        assert abs(1 / k_num ** 2 - float(prem)) < 5e-3 * float(prem), \
            f"{s}: 1/kappa^2 {1/k_num**2} against the printed {prem}"
    report.append("  18. kappa and 1/kappa^2 taken character for character "
                  "from randomized_labels.tex on all five, each cross-checked "
                  "against a fresh recomputation (the cube's -1/3 exactly)")

    # --- 27. the hand-typed LaTeX in PARAM_LINES and EFFECT_SCALE -----------
    # These two tables are the only places the generator holds LaTeX it did
    # not derive.  Assertion 23 reaches PARAM_LINES alone, and only against
    # ANOTHER hand-typed string, the frozen caption; EFFECT_SCALE's LaTeX
    # half is closed by nothing but this check: a wrong scale in the
    # vertex-table head multiplies every printed effect and passes every
    # other assertion.
    for s in SOLIDS:
        f = F[s]
        blob = " ".join(f["emitted"])
        # (i) the effect scale: the printed \tfrac against its own numeric half
        if s in EFFECT_SCALE:
            num, tex = EFFECT_SCALE[s]
            m = re.fullmatch(r"\\tfrac(?:\{(.+?)\}\{(.+?)\}|(\d)(\d))", tex)
            assert m, f"{s}: unparsable effect scale {tex!r}"
            top, bot = (m.group(1), m.group(2)) if m.group(1) else \
                (m.group(3), m.group(4))
            val = {r"\sqrt3": np.sqrt(3), "1": 1.0}[top] / float(bot)
            assert abs(val - num) < 1e-15, \
                f"{s}: printed scale {tex} = {val}, used {num}"
            assert tex in blob, f"{s}: the effect scale is not on the card"
        # (ii) the printed CNOT kets against _cnot's own permutation
        kets = re.findall(r"\\ket\{([01]+)\}\\mapsto\\ket\{([01]+)\}", blob)
        assert len(kets) == 2, f"{s}: {len(kets)} printed CNOT kets, not 2"
        # the ancilla-register width is the upper block's, read off the drawn
        # Fourier gate -- never a per-solid constant
        nq, na = f["n"], f["upper"]
        P = _cnot(nq, ctrl=nq - 1, targ=na - 1)
        for a, b in kets:
            assert len(a) == len(b) == nq, f"{s}: ket {a} is not {nq} bits"
            i, j = int(a, 2), int(b, 2)
            assert P[j, i] == 1, \
                f"{s}: the printed CNOT sends |{a}> to |{b}>, _cnot does not"
        # (iii) alpha, beta and U_A's prefactor against decker_circuit's own
        #       first column.  W[:, 0] is column 0 of the head (the isometry
        #       picks e_0), and on all three small solids that column is a
        #       kron: reshaped (2^upper, 2^ell) it must be rank one and its
        #       left singular vector must be the printed (alpha, beta).
        rad = re.findall(r"\\sqrt\{\(3([+-])\\sqrt3\)/(\d+)\}", blob)
        if rad:
            assert len(rad) == 2 and rad[0][0] == "+" and rad[1][0] == "-" \
                and rad[0][1] == rad[1][1], f"{s}: printed alpha/beta {rad}"
            n_ = float(rad[0][1])
            al, be = np.sqrt((3 + np.sqrt(3)) / n_), np.sqrt((3 - np.sqrt(3)) / n_)
            pm = re.search(r"\$U_A = (\\sqrt2|\\sqrt3|2)\\,\\pmat", blob)
            assert pm, f"{s}: no U_A prefactor printed"
            pref = {r"\sqrt2": np.sqrt(2), r"\sqrt3": np.sqrt(3),
                    "2": 2.0}[pm.group(1)]
            UA = pref * np.array([[al, be], [be, -al]])
            assert np.abs(UA @ UA.T - np.eye(2)).max() < 1e-14, \
                f"{s}: the printed U_A is not unitary"
            col = f["W"][:, 0].reshape(2 ** f["upper"], -1)
            u, sv, _ = np.linalg.svd(col)
            assert sv[1] < 1e-12, f"{s}: the head's first column is not a kron"
            got = np.abs(u[:, 0])
            want = np.abs(UA[:, 0]) / np.linalg.norm(UA[:, 0])
            assert np.abs(got - want).max() < 1e-12, \
                f"{s}: printed (alpha, beta) {want} against the circuit {got}"
        # (iv) the A_5 pair print u_pm and v_pm in closed form; B and C built
        #      from them must be unitary.  (The full A^dag = (I2 + -sigma_z)
        #      (I2 x B) R (I2 x C) closure needs R, which only the drawn
        #      circuit fixes, and lands with those two cards.)
        # Read each constant out of its OWN inline-math atom rather than out
        # of a fixed span of the sentence around it: an earlier form keyed on
        # "...$ and" and, when the connective became "...$, and", matched
        # nothing -- and `if uv:` then skipped the whole check in silence.
        # Requiring the two atoms on exactly the two A_5 solids is what makes
        # a missing one a failure instead of a skip.
        atoms = dict(re.findall(r"\$(u_\\pm|v_\\pm) = ([^$]+)\$", blob))
        want_uv = COVARIANCE[s] == "I"
        assert (len(atoms) == 2) == want_uv, \
            f"{s}: printed u_pm/v_pm atoms {sorted(atoms)}, expected " \
            + ("both" if want_uv else "neither")
        if want_uv:
            up_, um_ = _pm_radical(atoms[r"u_\pm"])
            vp_, vm_ = _pm_radical(atoms[r"v_\pm"])
            for M, nm in ((np.array([[um_, -up_], [up_, um_]]), "B"),
                          (np.array([[vm_, vp_], [vp_, -vm_]]), "C")):
                assert np.abs(M @ M.T - np.eye(2)).max() < 1e-12, \
                    f"{s}: the printed {nm} is not unitary"
    report.append("  27. the generator's hand-typed LaTeX closed against the "
                  "code, not against another string: the effect scale parses "
                  "to its own numeric half; both printed CNOT kets are cells "
                  "of _cnot(n, ctrl=n-1, targ=na-1); the printed alpha, beta "
                  "and U_A prefactor reproduce decker_circuit's own first "
                  "column (rank one, left singular vector); the A_5 pair's "
                  "printed u_pm, v_pm build unitary B and C")

    # --- 19. the coin's words, axes and rays --------------------------------
    for s in SOLIDS:
        f = F[s]
        if not f["rays"]:
            continue
        mine = sorted(r["word"] for r in f["rays"])
        theirs = sorted(w.strip() for w in f["coin_words_sorted"].split(", "))
        assert mine == theirs, f"{s}: coin multiset {mine} against D.1 {theirs}"
        A, v0 = alignment(f["v"])
        for r in f["rays"]:
            a, b = r["zero"] - 1, r["one"] - 1
            assert np.allclose(f["v"][b], -f["v"][a], atol=LOOSE), \
                f"{s}: ray {r['zero']}--{r['one']} is not antipodal"
            assert nearest(f["v"], r["Rg"].T @ v0, f"{s} ray") == a, \
                f"{s}: {r['word']}'s |0> vertex is not the left-hand index"
        zeros = [r["zero"] for r in f["rays"]]
        assert len(set(zeros)) == len(zeros) == len(f["rays"]), \
            f"{s}: two rays fire the same |0> vertex"
        # applying the alignment a SECOND time is the error that produces
        # exactly that collapse, so it is shown to be detectable rather than
        # merely avoided: on every solid whose alignment is not the identity,
        # A R_g^T v0 leaves the vertex set outright
        if not f["A_is_identity"]:
            off = max(np.sort(np.linalg.norm(f["v"] - A @ r["Rg"].T @ v0,
                                             axis=1))[0] for r in f["rays"])
            assert off > MARGIN, \
                f"{s}: the alignment applied twice still lands on vertices " \
                f"(worst miss {off:.3f}) -- the trap is undetectable here"
    # ... and the headline claim itself: "the N reproduce the V effects
    # above exactly".  That sentence carries "this thesis's one exact
    # implementation", and no other assertion tests it.
    worst19, exact19, worst19b = 0.0, [], 0.0
    tab19 = gate_table(np.load(DATA / "gates.npz", allow_pickle=True))
    for s in SOLIDS:
        f = F[s]
        if not f["rays"]:
            continue
        A, v0 = alignment(f["v"])
        blob19 = " ".join(f["emitted"])
        tot = np.zeros((2, 2), complex)
        for r in f["rays"]:
            # The snapshot axis of the pair.  The order is the drawn word
            # FIRST and the alignment after it -- Table D.1's own caption
            # ("the one fixed rotation every projective shot applies after
            # the drawn word") and channel_R1's p(b) from A R_g r.  In the
            # Heisenberg picture that measures (A R_g)^T zhat = R_g^T v0,
            # since A v0 = zhat.  The card's own "apply it, then the
            # alignment"; assertion 19b is the tripwire on the other order.
            nvec = r["Rg"].T @ v0
            for sg, which in ((+1, "zero"), (-1, "one")):
                Ek = (I2 + sg * sum(nvec[a] * PAULI[a] for a in range(3))) / f["V"]
                worst19 = max(worst19, np.abs(Ek - f["E"][r[which] - 1]).max())
                tot += Ek
        assert np.abs(tot - I2).max() < 1e-12, f"{s}: the coin does not resolve 1"
        # 19b.  The card TELLS the reader an order, and the order is not free:
        # the drawn word first, the fixed alignment after it (Table D.1's own
        # caption, and channel_R1's p(b) from A R_g r).  "apply it after the
        # alignment" is the other order, and no check that re-derives
        # R_g^T v0 from the code can see it printed.  So the WRONG order is
        # built here and required to miss:
        # A^T R_g^T zhat leaves the vertex set on every solid whose alignment
        # is not the identity, and the sentence is required to name the order
        # that does not.
        assert "apply it, then the alignment" in blob19, \
            f"{s}: the protocol line does not state the order it prices"
        assert "after the alignment" not in blob19, \
            (f"{s}: the coin block puts the alignment BEFORE the drawn word; "
             f"that measures A^T R_g^T zhat, not R_g^T v0")
        if not f["A_is_identity"]:
            miss = max(np.sort(np.linalg.norm(
                f["v"] - A.T @ (r["Rg"].T @ np.array([0.0, 0.0, 1.0])),
                axis=1))[0] for r in f["rays"])
            assert miss > MARGIN, \
                (f"{s}: the reversed order lands on the vertex set "
                 f"(worst miss {miss:.3f}) -- the trap is undetectable here")
            worst19b = max(worst19b, miss)
        # ... and where the alignment is the identity the words ARE gates, so
        # the same statement holds of the SU(2) matrices themselves.  That is
        # the octahedron's "this thesis's one exact implementation", and it is
        # now a check rather than a claim.
        if f["A_is_identity"]:
            tot2 = np.zeros((2, 2), complex)
            for r in f["rays"]:
                U = word_matrix((r["seq"] or "I").replace("\u03a6", "Phi"),
                                tab19)
                for proj, which in ((np.diag([1, 0]), "zero"),
                                    (np.diag([0, 1]), "one")):
                    Ek = U.conj().T @ proj.astype(complex) @ U / len(f["rays"])
                    worst19 = max(worst19,
                                  np.abs(Ek - f["E"][r[which] - 1]).max())
                    tot2 += Ek
            assert np.abs(tot2 - I2).max() < 1e-12, f"{s}: gates do not resolve 1"
            exact19.append(s)
    assert exact19 == ["octahedron"], f"gate-exact coins: {exact19}"
    assert worst19 < 1e-12, \
        f"the coin mixture misses the printed effects by {worst19:.1e}"
    report.append("  19. coin words in vertex-scan order, multiset equal to "
                  "coin_column's string; every ray antipodal; every left-hand "
                  "index the |0> vertex argmin_k |n_k - R_g^T v0|, the "
                  "alignment applied exactly once; and the uniform mixture "
                  f"reproduces every npz effect (worst {worst19:.1e}) and "
                  "sums to 1 -- the card's word 'exactly', checked")
    report.append("  19b. every coin block states the order it prices (the "
                  "drawn word, then the fixed alignment); the reversed order "
                  f"leaves the vertex set by at least {worst19b:.3f}")

    # --- 19c. the route to the atlas rows, exactly once per card ------------
    for s in SOLIDS:
        f = F[s]
        if not f["rays"]:
            continue
        lab = ATLAS_TAB[COVARIANCE[s]]
        body = uncommented(bodies.get(f"card_{SHORT[s]}_body.tex", ""))
        hits = (" ".join(f["emitted"]) + " " + body).count(r"\ref{%s}" % lab)
        assert hits == 1, \
            (f"{s}: prints atlas words and points at {lab} {hits} times "
             f"(want exactly one: head 4's third pointer)")
        assert (r"$\cdot$ Table~\ref{%s}" % lab) in " ".join(heads(f)), \
            f"{s}: the atlas route is not head 4's pointer"
        # ... and where a printed word is COMPOSITE the card must say in which
        # order to read it, because the two orders are different rotations.
        comp = [r["word"] for r in f["rays"] if " " in r["word"]]
        # ... scoped to HEAD 4, because the same three words are the
        # octahedron's own for the same convention on a different object (its
        # U_R, the one product of three rotations on the five cards).  One
        # convention, one wording, two objects: that is the point, so the
        # check reads the head that governs the ray list rather than the page.
        h4 = [b for b in heads(f) if r"\newcommand{\CHeadD}" in b][0]
        assert (ORDER_RIDER in h4) == bool(comp), \
            (f"{s}: {len(comp)} composite atlas words printed and head 4's "
             f"operator-order rider disagrees")
    report.append("  19c. every card printing atlas words carries exactly one "
                  "route to its group's Appendix A table, and names the "
                  "operator order wherever it prints a composite word "
                  + str({SHORT[s]: sum(1 for r in (F[s]["rays"] or [])
                                       if " " in r["word"]) for s in SOLIDS}))

    # --- 19d. the five draw notes, each backed and each pinned --------------
    # Each card carries one rung of F.3.3's ladder -- the icosahedron's is
    # its crossing, "A full 2T draw does randomness's two jobs at once: it
    # realizes the POVM and twirls the noise", where the bars meet.  Every
    # note is derived here rather than trusted, and no two cards may carry
    # the same one.
    #
    # The ladder is F.3.3's, and its verdicts are Table 5.2's `Which one
    # binds' row, read back out of the emitted ledger below -- so a card can
    # never disagree with the table that prices it.  The backing is local and
    # cheap: one rotation group T, derived (not looked up) by closing the
    # icosahedron's six coin words, and the SIGNED orbit of each card's
    # alignment seed under it.  Signed because a Z readout reports an axis:
    # the snapshot direction is b R_g^T v0 for b = +-1, so the assembled POVM
    # always lives on the antipodally CLOSED orbit (randomized_core's
    # orbit_counts; the trap is to read `orbit' as the bare G.v0, which is
    # half the vertex set on the octahedron).  That one fact is the whole
    # ladder: closure is why the tetrahedron becomes the cube, and uniformity
    # of the counts is what `realize' means.
    NOTE_PINS = {
        # substring                              -> the cards that may print it
        "does randomness's two jobs at once": ["icosahedron"],
        "the eight effects assembled are the cube's": ["tetrahedron"],
        "it is the twirl that forces": ["octahedron", "cube"],
        # The scope every "forces" on these pages carries: three cards
        # state a minimum and all three must say over WHAT, because
        # check_flip_completion's forty-element ensemble beats the group
        # minimum on the dodecahedron and the unscoped sentence is false.
        # (Probed before the verdict pin, so a de-scoped dodecahedron fails
        # on its scope.)
        "among groups": ["octahedron", "cube", "dodecahedron"],
        r"realization forces $\TwoI$": ["dodecahedron"],
        # ... and the flip's position, half the content of the second
        # sentence: drawn before the alignment instead, the same ensemble
        # loses both jobs (check_flip_completion's control column).  19b bans
        # the phrase "after the alignment" on these pages, so the position is
        # named from the readout end, as "last", where it is equally exact.
        r"flip $\Id, X, Y, Z$ last": ["dodecahedron"],
        # ... and the other half, the relabelling; the literal itself is
        # derived from the gates below, this probe only says WHICH card.
        "swaps the pair": ["dodecahedron"],
    }
    # case-blind: the scope OPENS the dodecahedron's sentence ("Among
    # groups, ...") and closes the octahedron's and the cube's
    for probe19d, owners19d in NOTE_PINS.items():
        carriers = [s for s in SOLIDS
                    if probe19d.lower() in " ".join(F[s]["emitted"]).lower()]
        assert carriers == owners19d, \
            f"`{probe19d}' is on {carriers}, not {owners19d}"
    # every card says which of the two jobs binds, and says what the ledger says
    binds19d = dict(zip(SOLIDS, read_ledger()["Which one binds"]))
    VERDICT = {"tetrahedron": "---", "octahedron": "twirl", "cube": "twirl",
               "icosahedron": "both", "dodecahedron": "realize"}
    assert binds19d == VERDICT, \
        f"Table 5.2's verdicts moved: {binds19d}, the cards assume {VERDICT}"

    # The dodecahedron's forty, from the gates and the ray list.  A
    # flip P drawn last measures P^T zhat = s_P zhat, the rotation being
    # Tr(sigma_a P sigma_b P^+)/2: s = -1 for X and Y, +1 for Id and Z.  So
    # X or Y reads the printed pair reversed -- the parenthesis, built here
    # from the computed set rather than typed -- and over ten words, four
    # flips and two outcomes every vertex is reported exactly 2|E|/V = 4
    # times: realization is the coin's, axis for axis, the sign folded into
    # the outcome.  Without the swap the four flips average the readout to
    # (t/2) Id, a measurement that reports nothing.  The twirl itself,
    # exactly depolarizing at T_zz for every affine (T, t), is
    # randomized_twojobs.check_flip_completion's and is not re-derived here.
    fd19 = F["dodecahedron"]
    ez19 = np.array([0.0, 0.0, 1.0])
    swap19d, sign19d = [], {"Id": 1.0}
    for nm19, P19 in zip("XYZ", PAULI):
        R19 = np.array([[np.trace(PAULI[a19] @ P19 @ PAULI[b19]
                                  @ P19.conj().T).real / 2
                         for b19 in range(3)] for a19 in range(3)])
        s19 = float(ez19 @ R19 @ ez19)
        assert np.abs(R19 @ ez19 - s19 * ez19).max() < 1e-12, \
            f"the flip {nm19} does not fix the readout axis"
        sign19d[nm19] = s19
        if s19 < 0:
            swap19d.append(nm19)
    assert swap19d == ["X", "Y"], f"the flips that swap the pair are {swap19d}"
    assert len(sign19d) == KLEIN, "the flip layer is not the Klein four"
    lit19d = "($%s$ or $%s$ swaps the pair)" % tuple(swap19d)
    assert lit19d in " ".join(fd19["emitted"]), \
        f"the dodecahedron's note does not print `{lit19d}'"
    _, v0d19 = alignment(fd19["v"])
    hits19d7 = [0] * fd19["V"]
    for r19 in fd19["rays"]:
        for s19 in sign19d.values():
            for b19 in (1.0, -1.0):
                hits19d7[nearest(fd19["v"], b19 * s19 * (r19["Rg"].T @ v0d19),
                                 "flip-completion snapshot")] += 1
    assert hits19d7 == [4] * fd19["V"], \
        f"the forty report the vertices {hits19d7} times, not 4 each"
    assert NUMERAL[KLEIN * len(fd19["rays"])] in " ".join(fd19["emitted"]), \
        "the dodecahedron's numeral is not the ensemble's size"
    fi = F["icosahedron"]
    _, v0i = alignment(fi["v"])

    def _key19d(R):
        # + 0.0 canonicalizes -0.0, whose bytes differ from +0.0's
        return (np.round(R, 9) + 0.0).tobytes()

    group19d = {_key19d(np.eye(3)): np.eye(3)}
    frontier = [r["Rg"] for r in fi["rays"]]
    while frontier:
        Rn = frontier.pop()
        kb = _key19d(Rn)
        if kb in group19d:
            continue
        group19d[kb] = Rn
        frontier += [Rn @ r["Rg"] for r in fi["rays"]]
    Ts19d = list(group19d.values())
    assert len(Ts19d) == 12, \
        f"the icosahedral coin words close to order {len(Ts19d)}, not T's 12"
    hits19d = sorted(nearest(fi["v"], Rn.T @ v0i, "two-jobs orbit")
                     for Rn in Ts19d)
    assert hits19d == list(range(fi["V"])), \
        f"the T orbit of the seed covers vertices {hits19d}, not all twelve"
    worst19d = 0.0
    for a19 in range(3):
        for b19 in range(3):
            M19 = np.zeros((3, 3)); M19[a19, b19] = 1.0
            avg19 = sum(Rn @ M19 @ Rn.T for Rn in Ts19d) / len(Ts19d)
            worst19d = max(worst19d, np.abs(
                avg19 - np.trace(M19) / 3.0 * np.eye(3)).max())
    assert worst19d < 1e-12, \
        f"the T twirl is not scalar (residual {worst19d:.1e})"

    # ... and the same T is every other card's draw too.  It is one group in
    # one orientation, not five: Section 3.4's anchoring puts the atlas's 2T
    # inside both 2O and 2I, which is what lets an all-Clifford draw be
    # quoted on a card whose own group is 2I.
    for s in SOLIDS:
        Rs19d = {_key19d(R) for R in load_rotations(COVARIANCE[s])}
        assert {_key19d(R) for R in Ts19d} <= Rs19d, \
            f"{s}: the derived T is not a subgroup of its covariance group"

    # The signed orbit under T, per card: which vertices a full 2T draw
    # reports, and how often.
    reach19d = {}
    for s in SOLIDS:
        fs19 = F[s]
        _, v019 = alignment(fs19["v"])
        target19 = F["cube"] if s == "tetrahedron" else fs19
        counts19 = [0] * target19["V"]
        for Rn in Ts19d:
            for b19 in (1.0, -1.0):
                counts19[nearest(target19["v"], b19 * (Rn.T @ v019),
                                 f"{s} draw snapshot")] += 1
        reach19d[s] = counts19

    # The tetrahedron: the note names another solid, so the note is checked
    # against that solid's own data.  Its four vertices are non-antipodal, so
    # the closed orbit is twice the size -- and the accident worth printing is
    # that the closure is another member of the family, vertex for vertex and
    # effect for effect (dial_settings.py asserts the geometric half).
    ct19 = reach19d["tetrahedron"]
    assert ct19 == [3] * 8, \
        f"the T draw off a tetrahedron vertex hits {ct19}, not the cube's 8x3"
    for k19, n19 in enumerate(F["cube"]["v"]):
        Ek19 = (I2 + sum(n19[a19] * PAULI[a19] for a19 in range(3))) / 8
        assert np.abs(Ek19 - F["cube"]["E"][k19]).max() < 1e-12, \
            f"the assembled effect {k19 + 1} is not the cube POVM's"

    # The octahedron and the cube: realization is the printed ray list, and
    # it is strictly cheaper than the twirl.  Both halves are checked -- the
    # coin realizes (assertion 19 above, uniformly) and the coin does NOT
    # twirl, which is what entitles "it is the twirl that forces".  The
    # octahedron's is the thesis's own witness (F.3.3.2's C_3); the cube's four
    # words are not even a group, and fail wider.
    for s in ("octahedron", "cube"):
        fs19 = F[s]
        assert reach19d[s] == [24 // fs19["V"]] * fs19["V"], \
            f"{s}: the T draw is not uniform over the vertex set"
        assert NUMERAL[len(fs19["rays"])] in " ".join(fs19["emitted"]), \
            f"{s}: the note's numeral is not the printed ray count"
        cw19 = 0.0
        for a19 in range(3):
            for b19 in range(3):
                M19 = np.zeros((3, 3)); M19[a19, b19] = 1.0
                avg19 = sum(r["Rg"] @ M19 @ r["Rg"].T
                            for r in fs19["rays"]) / len(fs19["rays"])
                cw19 = max(cw19, np.abs(
                    avg19 - np.trace(M19) / 3.0 * np.eye(3)).max())
        assert cw19 > 0.1, \
            f"{s}: the coin twirls after all (residual {cw19:.1e}) -- the " \
            f"note's `it is the twirl that forces' has lost its warrant"

    # The dodecahedron: the one card where realization is what binds, so the
    # thing to show is that the all-Clifford draw FAILS to realize.  It does
    # not fail by a little: eight of the twenty vertices are never reported.
    # (Which eight depends on the seed -- vertex 9 lands on the twelve-vertex
    # family, not the inscribed cube -- which is why the note stops at the
    # verdict and names no shape.)
    cd19 = reach19d["dodecahedron"]
    seen19 = sum(1 for c19 in cd19 if c19)
    assert seen19 == 12 and set(cd19) == {0, 2}, \
        f"the T draw reaches {seen19} of the dodecahedron's 20 vertices"
    assert reach19d["icosahedron"] == [2] * 12, \
        "the T draw is not uniform over the icosahedron's vertices"

    report.append("  19d. five draw notes, each backed and each pinned to its "
                  "own card: T derived from the icosahedron's coin words "
                  "(order 12, inside all three covariance groups), its signed "
                  "orbit of the seed "
                  + str({SHORT[s]: (sum(1 for c in reach19d[s] if c),
                                    max(reach19d[s])) for s in SOLIDS})
                  + " as (vertices reported, times each) against "
                  + str({SHORT[s]: F[s]["V"] for s in SOLIDS})
                  + ", the tetrahedron's eight matching the cube POVM effect "
                  f"for effect, the twirl scalar to {worst19d:.1e} and no "
                  "coin scalar at all, every verdict Table 5.2's own")

    # --- 20. the alignment claim --------------------------------------------
    ids = [s for s in SOLIDS if F[s]["decomposable"] and F[s]["A_is_identity"]]
    assert ids == ["octahedron"], f"alignment is the identity for {ids}"
    for s in SOLIDS:
        f = F[s]
        if f["align_vertex"] is None:
            continue
        assert f["align_vertex"] - 1 == int(np.argmax(f["v"][:, 2])), \
            f"{s}: the printed alignment vertex is not the argmax-z one"
    assert not F["tetrahedron"]["decomposable"], "the tetrahedron decomposed"
    tet20 = " ".join(F["tetrahedron"]["emitted"])
    assert HEAD_D_NONE in tet20 and HEAD_D not in tet20, \
        "the tetrahedron must print its negative as head 4, not omit the band"
    # WHY there is no coin is Figure 1.1's own strip cell, `none: no
    # antipodes`, printed at the top of this card -- so a sentence below it
    # would say it a THIRD time, four centimetres from the second.  What is
    # checked is that the reason is on the page, not which of the three
    # objects carries it.
    assert "none: no antipodes" in tet20, \
        "the tetrahedron does not say why there is no coin"
    assert "No antipodal pair" not in tet20, \
        ("the tetrahedron restates its strip cell: `none: no antipodes' is "
         "already printed at the top of the card, and head 4 already says "
         "`No coin over axes'")
    assert "The dilation is forced;" in tet20, \
        "the tetrahedron does not say that the dilation is not optional"
    # ... and the two randomized protocols must not be readable as one, so
    # the card that has only the twirl has to SAY it has the twirl --
    # otherwise its reader concludes this solid admits no randomization.
    assert "a drawn $U_g$ still twirls it." in tet20, \
        "the tetrahedron drops the contrast between the two protocols"
    report.append("  20. alignment == identity for the octahedron and nothing "
                  "else; the printed alignment vertices are the argmax-z ones "
                  f"({[F[s]['align_vertex'] for s in SOLIDS[1:]]}); the "
                  "tetrahedron is indecomposable and prints the negative")

    # --- 21. chips and legibility -------------------------------------------
    for s in SOLIDS:
        f = F[s]
        sep, chips = chip_geometry(f)
        assert f["chips_all"] == (len(chips) == f["V"]), f"{s}: chip bookkeeping"
        if not f["has_sphere"]:
            report.append(f"  21. {s}: PARTIAL -- the rule says {len(chips)} "
                          f"chips, minimum projected separation {sep:.1f}pt at "
                          f"PanelScale {f['panel_scale']}; no card panel drawn "
                          f"yet, so the 19pt floor is not asserted")
            continue
        assert sep >= 19.0, \
            f"{s}: minimum projected chip separation {sep:.1f}pt against the " \
            f"19pt floor (two chip diameters) with {len(chips)} chips"
        src = (FIGURES / f["sphere_src"]).read_text()
        drawn = sorted(int(m) for m in re.findall(r"\\chipnode\{(\d+)\}", src))
        assert drawn == sorted(chips), \
            f"{s}: {f['sphere_src']} chips {drawn}, the rule says {sorted(chips)}"
        scale = float(re.search(r"\\renewcommand\{\\PanelScale\}\{([\d.]+)\}",
                                src).group(1))
        assert abs(scale - f["panel_scale"]) < 1e-9, f"{s}: PanelScale"
        # No card makes a prose axes claim: the panel draws its own axes
        # exactly as Figure 1.1's does and neither names them, so what is
        # checked here is the geometry, which is what a wrong pose or a scale
        # breaks.  The one claim printed is the chip rule, a label on the two
        # A_5 cards, required here to agree with the rule the panel was drawn
        # by -- and the label does not say "chip": the word appears nowhere
        # in the printed thesis (it is a TikZ macro name), so these two cards
        # would introduce it undefined.
        blob = " ".join(f["emitted"])
        CHIPLAB = "numbered at each axis's near vertex"
        assert (CHIPLAB in blob) == (not f["chips_all"]), \
            (f"{s}: the card says {CHIPLAB!r}: {CHIPLAB in blob}; the panel "
             f"numbers {len(chips)} of {f['V']} vertices")
        assert "chip" not in blob, \
            f"{s}: the card prints the word `chip', which the thesis never does"
        off = sorted(set(f["chip_offsets"].values()))
        report.append(f"  21. {s}: {len(chips)} chips at PanelScale {scale} in "
                      f"{f['sphere_src']}, offsets {off}pt, minimum projected "
                      f"separation {sep:.1f}pt against a 9.5pt chip and a "
                      f"19pt floor")
    # ... and the panel is UNIFORM across the five, which is the single thing
    # this design spends its freed height on: Figure 1.1's own 2.47 on all
    # five, so the same solid never prints at two sizes in one thesis.
    scales = sorted({F[s]["panel_scale"] for s in SOLIDS if F[s]["has_sphere"]})
    if len(scales) == 1:
        report.append(f"  21. PanelScale {scales[0]} on all five, "
                      f"Figure 1.1's own")
    else:
        report.append(f"  21. PARTIAL -- PanelScale {scales}, not uniform")

    # --- 22. every \PVert coordinate ----------------------------------------
    import export_numpy
    export_numpy.verify_figure_orders({f"povm_{s}": load(s) for s in SOLIDS})
    pre = (FIGURES / "_povm_sphere_preamble.tex").read_text()
    assert pre.count(r"\PVert{") == 50, "the preamble no longer holds 50 rows"
    for s in SOLIDS:
        if not F[s]["has_sphere"]:
            continue
        src = (FIGURES / F[s]["sphere_src"]).read_text()
        assert r"\PVert{" not in src, f"{s}: a coordinate was re-typed"
        assert r"\input{_povm_sphere_preamble}" in src
    report.append("  22. export_numpy.verify_figure_orders() clean (50 rows); "
                  "no card sphere re-types a coordinate")

    # --- 23. nothing printed today vanishes ---------------------------------
    # A caption's math token survives in exactly one of three ways, tried in
    # this order, and there is no fourth:
    #
    #  (a) SAME STRING, up to spellings that set identically -- \sqrt3 for
    #      \sqrt{3}, \pmat{A} for \bigl(\begin{smallmatrix}A\end{smallmatrix}
    #      \bigr), thin spaces, whitespace.  This is how most tokens survive.
    #  (b) SAME VALUE: the card prints the same left-hand side with a closed
    #      form that EVALUATES equal.  Four constants on the two A_5 cards are
    #      respelt -- (1/10) for \tfrac{1}{10}, (3+\sqrt5)/24 for
    #      \tfrac{3+\sqrt5}{24} -- because assertion 26's type floor bans a
    #      \tfrac from a \footnotesize block: it sets its numerals at 6.0pt,
    #      under the 7pt floor, and \small only lifts them to 6.97pt, on
    #      every card.  What must not vanish is the constant, not its spelling,
    #      and comparing NUMBERS here is stricter than comparing strings: a
    #      wrong constant, respelt consistently, passes (a) and fails (b).
    #  (c) RE-HOMED to the chapter head: the A_5 pair's shared four-stage
    #      factorization of A-dagger, which the head owns one page earlier so
    #      that the two cards carry the same sentence and neither depends on
    #      the other's page number or on float ordering.  Admissible only if
    #      the head really prints the token AND the card really points at the
    #      head -- both checked below, so this is a redirection with a verified
    #      target and not an exemption.  Delete the head's display, or the
    #      card's pointer, and the build stops.
    frozen = read_frozen_captions()
    thesis = (PAPER / "bsc-thesis.tex").read_text()
    # the chapter head is the prose from Appendix D's \chapter to the first
    # card float -- the region the cards point back into
    _h0 = thesis.index(r"\label{app:decker}")
    _h1 = thesis.index(r"\begin{figure}[p]", _h0)
    head = " ".join(thesis[_h0:_h1].split())
    # The pointer a card must carry for a token it leaves to the chapter head.
    # Every pointer on a card is a live \ref: the chapter head numbers the
    # display and the cards \eqref it, rather than pointing in prose.
    HEAD_POINTER = r"\eqref{eq:adagger}"

    def norm(x):
        x = re.sub(r"\\bigl\(\\begin\{smallmatrix\}(.*?)"
                   r"\\end\{smallmatrix\}\\bigr\)", r"\\pmat{\1}", x)
        # the chapter head sets its re-homed matrices as displays, so the same
        # matrix reaches this comparison as \begin{pmatrix}...\end{pmatrix}
        x = re.sub(r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}",
                   r"\\pmat{\1}", x)
        x = re.sub(r"\\tfrac\{(\w+)\}\{(\w+)\}", r"\1/\2", x)
        x = re.sub(r"\\tfrac(\w)(\w)", r"\1/\2", x)
        x = re.sub(r"(?<!\\)\\[,;!]", "", x)              # thin spaces
        # a dress is a spelling, and the freeze predates one: the head now
        # writes Decker's controlled-rotation block \mathsf{R}, dressed to
        # clear the reorientation R the cards point at
        x = re.sub(r"\\mathsf\{(\w+)\}", r"\1", x)
        x = x.replace(r"\bigl(", "(").replace(r"\bigr)", ")")
        return re.sub(r"\s+", "", re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", x))

    def value_of(tok):
        """The number a printed closed form denotes, or None if it is not one.

        A token with a \\pm/\\mp pair denotes the PAIR, so both signs are
        evaluated and both must agree -- u_+ and u_- are two constants, and a
        card that got one of them right is not a card that printed them.
        """
        rhs = tok.strip("$").split("=", 1)[1]
        try:
            if r"\pm" in rhs or r"\mp" in rhs:
                return _pm_radical(rhs)
            return (float(_tex_number(rhs)),)
        except (AssertionError, SyntaxError, ValueError, ZeroDivisionError,
                KeyError, IndexError):
            return None

    for s in SOLIDS:
        if not F[s]["has_body"]:
            report.append(f"  23. {s}: SKIPPED -- card_{SHORT[s]}_body.tex does "
                          f"not exist yet")
            continue
        blob = " ".join(F[s]["emitted"]) + " " \
            + uncommented(bodies[f"card_{SHORT[s]}_body.tex"])
        blob = " ".join(blob.split())
        flat = norm(blob)
        card_toks = re.findall(r"\$[^$]+\$", blob)
        toks = caption_tokens(frozen[s])
        same, revalued, rehomed, absorbed = [], [], [], []
        absorbed_scale = None
        for t in toks:
            if norm(t) in flat:
                same.append(t)
                continue
            # (b) same left-hand side, same number, different spelling
            if "=" in t:
                lhs = norm(t.strip("$").split("=", 1)[0])
                want = value_of(t)
                mates = [c for c in card_toks if "=" in c
                         and norm(c.strip("$").split("=", 1)[0]) == lhs]
                got = [value_of(c) for c in mates]
                hit = [g for g in got if g is not None and want is not None
                       and len(g) == len(want)
                       and all(abs(a - b) < LOOSE for a, b in zip(g, want))]
                if hit:
                    assert not any(g is not None and want is not None
                                   and len(g) == len(want)
                                   and any(abs(a - b) >= LOOSE
                                           for a, b in zip(g, want))
                                   for g in got), \
                        f"{s}: the card prints two different values for {lhs}"
                    revalued.append((t, mates[got.index(hit[0])]))
                    continue
            # (c) re-homed to the chapter head, both halves checked.  The
            # head sets it as a DISPLAY with its sentence's comma inside the
            # math, so the delimiters are dropped and the content compared.
            #
            # One token is re-homed with its scalar ABSORBED:
            # Decker prints the A_5 pair's A as sqrt3 (icosahedron) or sqrt5
            # (dodecahedron) times entries he rescaled by sqrt(2/V) -- the
            # caption's sqrt(1/6), sqrt(1/10) -- and the head prints the
            # unit-normalized entries with his two factors in front, written
            # sqrt{m} sqrt{2/V} for both solids at once (the head's own
            # principle: what the pair shares lives in the head, and this is
            # what buys the dodecahedron its reserve).  So the token is
            # matched by the same matrix body in the head, whose prefactor,
            # read at this solid's m and V, must equal the caption's times
            # the rescaling; the caption's rescaling token is then absorbed
            # with it, and the card mentions no rescaling any more.
            mc = re.fullmatch(r"\$([A-Za-z]+)=(.+?)\\pmat\{(.+)\}\$",
                              norm(t))
            mh = mc and re.search(r"%s&?=(.+?)\\pmat\{%s\}"
                                  % (mc.group(1), re.escape(mc.group(3))),
                                  norm(head))
            if mh:
                scale = float(np.sqrt(2 / F[s]["V"]))
                cap = float(_tex_number(mc.group(2)))
                got = float(_tex_number(
                    mh.group(1).replace("{m}", "{%d}" % (F[s]["V"] // 4))
                    .replace("/V}", "/%d}" % F[s]["V"])))
                assert abs(got - cap * scale) < LOOSE, \
                    (f"{s}: the chapter head prints {mc.group(1)} = "
                     f"{mh.group(1)}(...), but the caption's {mc.group(2)} on "
                     f"entries rescaled by sqrt(2/V) is {cap * scale:.9f}, "
                     f"not {got:.9f}")
                assert HEAD_POINTER in blob, \
                    (f"{s}: {t} is left to the chapter head, but the card "
                     f"never points at it ({HEAD_POINTER!r} absent)")
                rehomed.append(t)
                absorbed_scale = scale
                continue
            if absorbed_scale is not None and "=" not in t:
                try:
                    v = float(_tex_number(t.strip("$")))
                except (AssertionError, SyntaxError, ValueError,
                        ZeroDivisionError, KeyError, IndexError):
                    v = None
                if v is not None and abs(v - absorbed_scale) < LOOSE:
                    absorbed.append(t)
                    continue
            if norm(t.strip("$")) in norm(head):
                assert HEAD_POINTER in blob, \
                    (f"{s}: {t} is left to the chapter head, but the card "
                     f"never points at it ({HEAD_POINTER!r} absent)")
                rehomed.append(t)
                continue
            assert False, \
                (f"{s}: today's caption prints {t}; the card does not print "
                 f"it, prints no equal value for its left-hand side, and the "
                 f"chapter head does not carry it either")
        for t, c in revalued:
            report.append(f"      23b. {s}: {t} respelt as {c} "
                          f"-- equal to {LOOSE:g}")
        for t in rehomed:
            report.append(f"      23c. {s}: {t} left to the chapter head, "
                          f"which prints it, and the card points there")
        for t in absorbed:
            report.append(f"      23d. {s}: {t} absorbed into the chapter "
                          f"head's prefactor, sqrt(2/V) times the caption's")
        report.append(f"  23. {s}: all {len(toks)} math tokens of today's "
                      f"caption survive -- {len(same)} verbatim, "
                      f"{len(revalued)} respelt to the same value, "
                      f"{len(rehomed)} re-homed to the chapter head, "
                      f"{len(absorbed)} absorbed into its prefactor")

    # --- 26. the type floor, on the fragments set at \footnotesize ----------
    # Measured in the real class: a math script at \footnotesize prints at
    # 5.98pt, at \small 6.97pt.  Every fragment a reader retypes is set at
    # \small, so what sets at \footnotesize is exactly the header strip and
    # the pointer column -- and a fraction or a smallmatrix carrying a value
    # may not appear there.
    FLOOR_BAN = (r"\tfrac", r"\frac", r"\begin{smallmatrix}", r"\dfrac")
    for s in SOLIDS:
        fn = " ".join(F[s]["fn_blocks"])
        bad = [b for b in FLOOR_BAN if b in fn]
        assert not bad, (
            f"{s}: {bad} in a fragment that sets at \\footnotesize, where "
            f"math scripts print at 5.98pt -- under the 7pt floor")
        # 26b.  At \small the parameter scripts print at 6.97pt and no
        # numeral is left below 8.97pt at all, so the rule binds on the strip
        # alone and is the strong one:
        #
        #   (i)  the emitted surface sets NO numeral below 8.97pt, and
        #   (ii) every symbol that does print below 7pt is a Figure 1.1
        #        literal (check 30) whose own cell carries it upright at
        #        8.97pt -- `$T^\dagger$, over Clifford${}+T$`.
        digits = {x for x in re.findall(r"[_^]\{?(-?\d+)\}?", fn)}
        assert not digits, \
            (f"{s}: {sorted(digits)} set as a \\footnotesize math script "
             f"(5.98pt) -- under the 7pt floor and under the 8.97pt a "
             f"printed value has to reach")
        tiny = set(re.findall(r"[_^]\{?(\\?[A-Za-z]+)\}?", fn))
        assert tiny <= {r"\dagger"}, \
            f"{s}: {sorted(tiny)} set as a \\footnotesize math script"
        # ... read off the EMITTED strip, not the source: this is a claim
        # about the page, and 30 closes the page against Figure 1.1's strip
        sb26 = " ".join(" ".join(F[s]["fn_blocks"]).split())
        sb26 = sb26[sb26.index(r"\begin{tabular}"):sb26.index(r"\bottomrule")]
        cells26 = [c.strip() for c in sb26.split(r"\midrule")[1].rstrip(
            "\\").split("&")]
        for cell in cells26:
            for base in re.findall(r"\$?([A-Za-z])\^\\dagger\$?", cell):
                assert re.search(r"(?<![A-Za-z\\])%s(?![A-Za-z])" % base,
                                 re.sub(r"[A-Za-z]\^\\dagger", "", cell)), \
                    (f"{s}: the strip prints {base}^dagger, whose dagger sets "
                     f"at 5.98pt, and no upright {base} in the same cell")
        # The claim is scoped to the fragments this check inspects (the strip
        # and the pointer column, the only ones still set at \footnotesize).
        # It is NOT a claim about the whole page: at \small a first-level
        # script prints 6.97pt, and \tfrac numerators in the vertex table's
        # heads do print there.
        report.append(f"  26. {s}: no fraction and no smallmatrix in a "
                      f"footnotesize fragment -- so no numeral is set as a "
                      f"math script in the two fragments that still set at "
                      f"\\footnotesize (the strip and the pointer column); "
                      f"26b: the only sub-7pt glyph on the page is the dagger "
                      f"of a Figure 1.1 strip literal whose own cell prints "
                      f"the base symbol upright")

    # --- 28. the chapter head's U_g sentence, self-policing ------------------
    # The head says what the appendix's circuits do with the optional slot, and
    # a card that DRAWS it makes "is omitted here" false in the shipped PDF.
    # No wording is true of every file set, so the wording is checked against
    # the file set rather than fixed.
    drawn = [s for s in SOLIDS
             if (FIGURES / f"card_dec_{SHORT[s]}.tex").is_file()
             and r"\gate{U_g}" in uncommented(
                 (FIGURES / f"card_dec_{SHORT[s]}.tex").read_text())]
    want = ("is omitted here" if not drawn else
            "is drawn dashed on each" if len(drawn) == len(SOLIDS) else
            "is drawn dashed where a card carries it")
    lead = "The optional pre-rotation $U_g$ of Section~"
    i = chapter.index(lead)
    sent = chapter[i:chapter.index(".", i + len(lead) + 60) + 1]
    assert want in sent, \
        (f"the chapter head must say {want!r} while {len(drawn)} of "
         f"{len(SOLIDS)} card circuits draw the slot -- it says {sent!r}")
    report.append(f"  28. the chapter head says {want!r}, true of the "
                  f"{len(drawn)} card circuit(s) that draw the dashed slot")

    # --- 28b. D.2's opening sentence -----------------------------------------
    # D.2 opens by naming where the two corrections are recorded, and it must
    # name the figures rather than their captions: the captions ATTRIBUTE the
    # relabelling ("the register map Table D.4's") and the dodecahedron's does
    # not mention it at all, while the cards' bodies print it in full.
    # Checked here so it cannot rot.
    D2_OLD = "the outcome relabelling their captions record"
    D2_NEW = "the outcome relabelling they record"
    assert D2_OLD not in chapter, \
        ("bsc-thesis.tex still says %r; with the cards in place the captions "
         "no longer record it -- the bodies do" % D2_OLD)
    assert chapter.count(D2_NEW) == 1, \
        ("bsc-thesis.tex must say %r once in D.2's opening (found %d)"
         % (D2_NEW, chapter.count(D2_NEW)))
    for s in SOLIDS:
        assert r"\newcommand{\CardKey}" in " ".join(F[s]["emitted"]), \
            f"{s}: D.2 says the figures record the relabelling; this one " \
            f"prints no key"
    report.append("  28b. D.2's opening names the figures, not their captions, "
                  "as where the outcome relabelling is recorded, and all five "
                  "print the key it means")

    # --- 24. the labels -----------------------------------------------------
    for s in SOLIDS:
        lab = r"\label{fig:dec_%s_circuit}" % SHORT[s]
        assert chapter.count(lab) == 1, f"{lab} appears {chapter.count(lab)} times"
    report.append("  24. all five fig:dec_*_circuit labels present exactly once")


    # --- 29. the printed surface, parsed back out of the emitted file ----
    # Most assertions above check a value the generator RE-DERIVES.  That is
    # not the same as checking the file: patch an emitter's own output line and
    # the derived value stays right while the printed one is wrong, and the
    # build ships.  Everything below therefore reads the STRING the reader
    # reads and compares it to the object, cell by cell -- the pattern
    # assertion 3 already used for the <0|E_k|0> column.
    #
    # It reads the whole surface: the seven strip cells and the footer, head
    # 3's wording, the key WITH its `upper`/`lower` corner heads, the ray
    # list, the protocol line, the worked E_1's atoms, the U_R string as a
    # rotation, and kappa's pointer.
    for s in SOLIDS:
        f = F[s]
        blob = "\n".join(f["emitted"])
        one = " ".join(blob.split())
        V, ell, up, m = f["V"], f["ell"], f["upper"], f["m"]

        # (a) every vertex row: index, tuple, effect atoms, probability.
        # The row is split into cells at brace depth ZERO, because an effect
        # cell is a \pmat whose own & and \\ are inside braces: a flat split
        # cuts the matrix in half and compares halves.
        def cells_of(line):
            out, cur, d = [], "", 0
            for ch in line:
                d += (ch == "{") - (ch == "}")
                if ch == "&" and d == 0:
                    out.append(cur.strip()); cur = ""
                else:
                    cur += ch
            out.append(cur.strip())
            return out

        rows = []
        for line in blob.splitlines():
            mrow = re.match(r"^\$(\d+)\$ &(.*?)\s*\\\\(?:\s*\\addlinespace"
                            r"\[[\d.]+pt\])?\s*$", line)
            if mrow:
                rows.append((mrow.group(1), mrow.group(2)))
        assert len(rows) == V, f"{s}: {len(rows)} printed vertex rows, not {V}"
        for kk, rest in rows:
            k = int(kk) - 1
            assert 0 <= k < V, f"{s}: printed vertex index {kk}"
            cells_ = cells_of(rest)
            want = "$%s$" % tex_tuple(factored_tuple(s, f["v"][k]))
            assert cells_[0] == want, \
                f"{s}: row {kk} prints {cells_[0]} for a vertex that is {want}"
            if s in EFFECT_SCALE:
                assert len(cells_) == 3, f"{s}: row {kk} has {len(cells_)} cells"
                mm = factored_effect(s, f["E"][k])
                assert cells_[1] == (r"$\pmat{%s & %s \\ %s & %s}$"
                                     % (mm[0][0], mm[0][1], mm[1][0], mm[1][1])), \
                    f"{s}: row {kk} prints the effect {cells_[1]}"
            else:
                assert len(cells_) == 2, f"{s}: row {kk} has {len(cells_)} cells"
            assert cells_[-1] == r"$%.4f$" % f["p0"][k], \
                f"{s}: row {kk} prints {cells_[-1]} for <0|E|0>"
        assert sorted(int(k) for k, _ in rows) == list(range(1, V + 1)), \
            f"{s}: the printed vertex indices are not 1..{V}"

        # (b) the key: its corner heads, its column heads, its row heads and
        # every cell.  The corner heads are what carry the bit convention:
        # `upper` on the stub and `lower` spanning the columns say which
        # wires the two bit fields are, and their WIDTHS say how many.
        kb = blob[blob.index(r"\newcommand{\CardKey}"):]
        kb = kb[:kb.index(r"\end{tabular}")]
        assert r"\multicolumn{%d}{c}{\footnotesize lower}" % (2 ** ell) in kb, \
            f"{s}: the key does not head its columns `lower`, spanning {2**ell}"
        assert r"{\footnotesize upper} &" in kb, \
            f"{s}: the key does not head its stub `upper`"
        heads_ = re.findall(r"\$\\mathtt\{([01]+)\}\$", kb.split(r"\midrule")[0])
        assert heads_ == [format(c, "0%db" % ell) for c in range(2 ** ell)], \
            f"{s}: the key's column heads are {heads_}"
        assert all(len(h) == ell for h in heads_), \
            f"{s}: a column head is not {ell} bits wide"
        krows = re.findall(r"^\$\\mathtt\{([01]+)\}\$ & (.+?) \\\\",
                           kb, re.M)
        assert [r_ for r_, _ in krows] == [format(r_, "0%db" % up)
                                           for r_ in range(2 ** up)], \
            f"{s}: the key's row heads are {[r_ for r_, _ in krows]}"
        dead_cells = 0
        for rb, rest in krows:
            cells_ = [c.strip() for c in rest.split("&")]
            assert len(cells_) == 2 ** ell, f"{s}: key row {rb}: {len(cells_)}"
            for cb, cell in zip(heads_, cells_):
                nn = int(rb + cb, 2)
                want = str(f["key"][nn]) if nn in f["key"] else "---"
                dead_cells += want == "---"
                assert cell == want, \
                    (f"{s}: key cell (row {rb}, column {cb}) = register "
                     f"{nn} prints {cell!r}, not {want!r}")
        # the dead RULE, drawn rather than said: the empty cells are exactly
        # the dead outcomes, and the strip's `dead` cell counts them
        assert dead_cells == len(f["dead"]), \
            f"{s}: {dead_cells} empty key cells against {len(f['dead'])} dead"

        # (c) every ray, in order, |0> vertex first
        if f["rays"]:
            pr = re.findall(r"(\$[^$]+\$|\\Id)~\$(\d+)\$,\\,\$(\d+)\$", one)
            assert len(pr) == len(f["rays"]), \
                f"{s}: {len(pr)} printed rays, not {len(f['rays'])}"
            for (w, a, b), r_ in zip(pr, f["rays"]):
                assert (w, int(a), int(b)) == (r_["word"], r_["zero"],
                                               r_["one"]), \
                    (f"{s}: printed ray {w} on {a},{b} against the derived "
                     f"{r_['word']} on {r_['zero']},{r_['one']}")

        # (d) the strip, cell by cell, out of the emitted file -- and against
        # Figure 1.1's own row (30 checks the same thing at the source; this
        # checks what actually got written)
        sb = one[one.index(r"\newcommand{\CStrip}"):]
        sb = sb[:sb.index(r"\end{tabular}")]
        sr = [c.strip() for c in sb.split(r"\midrule")]
        assert len(sr) == 3, f"{s}: the strip has {len(sr)-1} rules, not 2"
        spec30, heads30, row30 = series_strip_row(s)
        got_h = [c.strip() for c in sr[0].split(r"\toprule")[1].rstrip(
            "\\").split("&")]
        got_r = [c.strip() for c in sr[1].rstrip("\\").split("&")]
        assert got_h == heads30, f"{s}: printed strip heads {got_h}"
        assert got_r == row30, f"{s}: printed strip row {got_r}"
        assert (r"\multicolumn{%d}{@{}l@{}}{%s}" % (len(heads30), STRIP_FOOTER)
                in sb), f"{s}: the strip footer is not the shared literal"

        # (e) head 3's wording, and the protocol line's alignment label
        want_c = HEAD_C + (HEAD_C_DEAD if f["dead"] else "")
        assert r"\newcommand{\CHeadC}{%s}" % want_c in one, \
            f"{s}: head 3 no longer states the register convention"
        if f["align_vertex"]:
            want_al = (r"($\Id$ here)" if f["A_is_identity"] else
                       r"($%d \mapsto \hat z$, inexact)" % f["align_vertex"])
            assert want_al in one, \
                f"{s}: the printed alignment label is not {want_al!r}"
            assert one.count(r"; measure $Z$.") == 1, \
                f"{s}: the protocol line does not end in the measurement"

        # (e2) the identities line: the trace and the design strength, both
        # values a reader retypes, parsed out of the printed string
        ml = re.search(r"\\newcommand\{\\CIdent\}\{(.*?)\}\n", blob + "\n",
                       re.S)
        assert ml, f"{s}: no identities line"
        assert (r"$\Tr E_k = %s$" % frac_tex(V // 2)) in ml.group(1), \
            f"{s}: the printed Tr E_k is not 2/V"
        dg = re.search(r"a \$(\d+)\$-design", ml.group(1))
        assert dg and int(dg.group(1)) == f["design"], \
            (f"{s}: the printed design strength is {dg and dg.group(1)}, "
             f"the vertex set is a {f['design']}-design")
        assert r"$\sum_k E_k = \Id$" in ml.group(1) and "rank one" in ml.group(1), \
            f"{s}: the identities line no longer states the axioms"

        # (f) the worked E_1's atoms, where one is printed
        if s in WORKED_E1:
            e1 = re.search(r"\\newcommand\{\\CIdentB\}\{(.*?)\}\n", blob + "\n",
                           re.S)
            assert e1 and WORKED_E1[s] in one, \
                f"{s}: the worked E_1 is not printed"
            assert r"$E_{g(k)} = U_g E_k U_g^\dagger$" in one, \
                f"{s}: the covariance rule is not beside the worked effect"
        else:
            assert r"\newcommand{\CIdentB}{}" in one, \
                f"{s}: a worked E_1 is printed where the effect column is"

        # (g) the U_R string, parsed back out as a rotation, and the
        # composition order wherever it is a product
        ur = re.search(r"\$U_R = (.+?)\$ (?:inverts|, rightmost)", one) or \
            re.search(r"\$U_R = (.+?)\$,", one)
        if ur:
            assert np.abs(rotation_from_tex(ur.group(1))
                          - f["R"].T).max() < 1e-12, \
                f"{s}: the PRINTED U_R is not R^{{-1}}"
            nfac = len(re.findall(r"R_\{?\\hat\s*([zy])\}?\(", ur.group(1)))
        else:
            # the two cards whose drawing names the box in full print no
            # formula here; the drawing carries it, and 17 closes it
            assert r"\gate{U_R = %s}" % f["UR_tex"] \
                in uncommented(f["circuit_src"]), \
                f"{s}: no U_R printed on the card and none drawn"
            nfac = 1
        assert (", rightmost factor first," in one) == (nfac > 1), \
            f"{s}: the composition order of a product U_R is unstated"
        # no arccos is glossed in degrees -- the reader retypes the closed
        # form, which is what is printed -- so no degree may appear
        assert not re.findall(r"\$\d+\.\d+\^\\circ\$", one), \
            f"{s}: a degree gloss is printed; the closed form is what prints"

        # (h) kappa's two branches and its pointer, in kappa's own band
        kap = f["labels"][2]
        kn = [b for b in f["emitted"] if r"\newcommand{\CKappa}" in b]
        assert len(kn) == 1 and r"\kappa = " in kn[0], \
            f"{s}: kappa is not printed once, in its own label"
        # BOTH branches, and in this order: the free pairing first, the priced
        # mismatch second.  Printing only the second reads as "$U_R$ is
        # compulsory" (36 checks that D.2 still licenses the pairing).
        assert kn[0].count(r"\kappa = ") == 2, \
            f"{s}: the kappa label prints {kn[0].count(chr(92)+'kappa = ')} " \
            f"kappas, not the free one and the priced one"
        assert kn[0].index(r"$\kappa = 1$, free.") \
            < kn[0].index(r"$\kappa = %s$" % kap), \
            f"{s}: the priced mismatch is printed before the free pairing"
        assert r"\ref{tab:decker-labels}" in " ".join(heads(f)), \
            f"{s}: kappa's band does not point at its definition"
    report.append("  29. the printed surface parsed back out of the emitted "
                  "file and compared to the object, not to a re-derivation: "
                  "every vertex row (index, tuple, effect atoms, probability), "
                  "the key's `upper`/`lower` corner heads, its column and row "
                  "heads and every cell (the empty ones counted against the "
                  "dead set), every ray in order |0>-vertex first, the seven "
                  "strip cells and the footer, head 3's wording, the "
                  "alignment label, the worked E_1, the printed U_R as a "
                  "rotation with its composition order, and kappa's pointer")

    # --- 30. the strip IS Figure 1.1's, cell by cell -----------------------
    # The recognition, made a check.  The card prints Figure 1.1's own row
    # under its own heads, so a reader who has met the figure recognises it
    # instead of reading it -- and that only works if it really is the same
    # string.  Compared after dropping the name column (the card's title says
    # the solid) and after unwrapping a \textbf that wraps a WHOLE cell
    # (Figure 1.1 bolds its own worked solid's row; a \textbf inside a cell is
    # content).
    seen30 = {}
    for s in SOLIDS:
        spec, heads30, row30 = series_strip_row(s)
        assert len(heads30) == len(row30) == 7, \
            f"{s}: {len(heads30)} strip columns after dropping the name, not 7"
        assert len(spec.split()) == 7, f"{s}: strip column spec {spec!r}"
        blob30 = " ".join(F[s]["emitted"])
        for h in heads30:
            assert h in blob30, f"{s}: the card drops Figure 1.1's head {h!r}"
        for c in row30:
            assert c in blob30, \
                f"{s}: the card's strip cell {c!r} is not Figure 1.1's"
        seen30[SHORT[s]] = heads30
        # the coin cell is Figure 1.1's link, not a second vocabulary for
        # the twirl
        assert r"\hyperref[tab:decker-vs-coin]{coin over axes}" \
            in blob30, f"{s}: the coin head is not Figure 1.1's hyperlink"
    assert all(h == seen30["tet"] for h in seen30.values()), \
        "the five cards print different strip heads"
    report.append("  30. the strip's seven column heads and each solid's seven "
                  "cells are string-equal to \\RecipeSeriesStrip's in "
                  "code/data/recipe_tet_data.tex, name column dropped: "
                  + " | ".join(seen30["tet"][:4]) + " | ...")

    # --- 31. the shared literals, byte-identical on all five ----------------
    # Nine strings are the same on every card by design (the footer, the three
    # shared heads, the U_g rider, the kappa label's frame, the protocol
    # line's frame, the pointer sets).  A card that quietly says one of them
    # differently is the failure mode a five-page series has and a one-page
    # figure does not.
    shared31 = {"strip footer": STRIP_FOOTER, "head 1": HEAD_A,
                "head 2": HEAD_B, "head 3": HEAD_C,
                "the U_g rider": UG_RIDER,
                "the strip pointer": PTR_STRIP, "pointer 1": PTR_A,
                "pointer 2": PTR_B, "pointer 3": PTR_C}
    for name, lit in shared31.items():
        miss = [SHORT[s] for s in SOLIDS
                if lit not in " ".join(F[s]["emitted"])]
        assert not miss, f"{name} is not on {miss}"
    for s in SOLIDS:
        one31 = " ".join(" ".join(F[s]["emitted"]).split())
        assert one31.count(STRIP_FOOTER) == 1, \
            f"{s}: the exactness footer is printed {one31.count(STRIP_FOOTER)} times"
    report.append(f"  31. all {len(shared31)} shared literals byte-identical "
                  f"on all five, and the exactness footer printed exactly once "
                  f"per card")

    # --- 32. the unfactored A is in the chapter head ------------------------
    # A is printed nowhere else, and the two A_5 cards are laid out on the
    # assumption that it is not on them: measured, the same five parameter
    # formulas are 119.00pt in the 300pt column A leaves beside it and
    # 75.19pt at full width, and A alone is 48.00 x 134.07pt.  So the
    # cards refuse to be written until the head really carries it.
    a5 = [s for s in SOLIDS if COVARIANCE[s] == "I"]
    head32 = " ".join(chapter.split())
    has_A = r"\begin{pmatrix}\alpha & \beta & \gamma & \delta" in head32
    for s in a5:
        assert has_A, \
            (f"{s}: the card leaves the unfactored $A$ to the chapter head "
             f"and the head does not print it")
        assert r"\newcommand{\CardSide}" in " ".join(F[s]["emitted"])
        assert r"\pmat{\alpha & \beta & \gamma & \delta" \
            not in " ".join(F[s]["emitted"]), f"{s}: A is back on the card"
    # ... the unfactored display must be NUMBERED, show Decker's rescaling
    # sqrt(2/V) as a factor of its own (his scaling has to stay
    # reconstructible from the head, so the two factors are not collapsed to
    # the 1/sqrt2 they multiply to), pair gamma and delta across
    # the two solids by name, and each card must point at it by \eqref: the
    # reader carries nothing between card and head, and the cards print
    # nothing of A but that pointer.
    assert r"\sqrt{\tfrac{2}{V}}\begin{pmatrix}\alpha" in head32, \
        ("the unfactored display no longer shows Decker's sqrt(2/V) rescaling "
         "as its own factor")
    for pair in (r"\gamma_\mathrm{icos}^2 = \delta_\mathrm{dodec}^2",
                 r"\delta_\mathrm{icos}^2 = \gamma_\mathrm{dodec}^2"):
        assert pair in head32, \
            f"the chapter head no longer states the pairing {pair}"
    for s in a5:
        assert r"\label{eq:aunfactored}" in head32, \
            (f"{s}: the card points at Equation~\\eqref{{eq:aunfactored}} and "
             f"the chapter head does not label its unfactored A")
        assert r"\eqref{eq:aunfactored}" in " ".join(F[s]["emitted"]), \
            f"{s}: the A^dagger line does not point at the unfactored display"
    # ... and the four-stage display it re-homes must be NUMBERED, because
    # the cards point at it by \eqref -- the alternative is prose aimed four
    # pages back, on the two cards whose mixing unitary is not printed in
    # full.
    for s in a5:
        assert r"\label{eq:adagger}" in head32, \
            (f"{s}: the card points at Equation~\\eqref{{eq:adagger}} and the "
             f"chapter head does not label its four-stage display")
        assert r"\eqref{eq:adagger}" in " ".join(F[s]["emitted"]), \
            f"{s}: the A^dagger line does not point at the labelled display"
    report.append("  32. the chapter head carries the unfactored A, numbered "
                  "eq:aunfactored, Decker's sqrt(2/V) rescaling shown as its "
                  "own factor and gamma, delta paired across the solids by "
                  "name, and labels its four-stage display eq:adagger; both "
                  "A_5 cards point at both, and neither prints A")

    # --- 32b. the head's A, rebuilt from the printed paragraph -------------
    # The string checks above pin the display's SHAPE; this pins its VALUE.
    # Decker's sqrt3 (icosahedron) and sqrt5 (dodecahedron) sit in front of
    # entries he rescaled by sqrt(2/V), and the two combine to 1/sqrt2 for
    # both solids -- exactly the kind of bookkeeping that drifts unread.  So
    # the prefactor and the matrix body are parsed out of the display, the
    # alpha/beta pair and the two gamma/delta pairing values out of the
    # paragraph under it, the roots taken positive as it says, and
    # kron(A^dag, F_b^dag) Q^dag iota must reproduce decker_circuit's rows: a
    # wrong prefactor, a wrong radicand or an exchanged pairing fails here
    # and nowhere else.  The forms are load-bearing: reword them only with
    # this parse in view.
    para = re.search(r"\\label\{eq:aunfactored\}(.*?)\\begin\{figure\}",
                     chapter, re.S)
    assert para, "the chapter head's eq:aunfactored paragraph is not parseable"
    para = " ".join(para.group(1).split())
    mA = re.search(r"A\s*(?:\\;)?\s*&?=\s*(?:\\;)?\s*(.+?)\\begin\{pmatrix\}"
                   r"(.+?)\\end\{pmatrix\}", para)
    assert mA, "the unfactored display does not read A = (prefactor)(pmatrix)"
    rows32 = [[c.strip() for c in r.split("&")]
              for r in mA.group(2).split(r"\\")]
    assert len(rows32) == 4 and all(len(r) == 4 for r in rows32), rows32
    # The four values are read wherever the paragraph puts them -- a sentence,
    # $$ displays, or rows of the numbered display (gathered or aligned) --
    # as `alpha^2 = X, ... beta^2 = Y,` and `gamma_icos^2 = delta_dodec^2 =
    # X, ... delta_icos^2 = gamma_dodec^2 = Y,`: the two of a pair on one
    # line, each closed form ending at its comma.  `alpha^2 + beta^2 = ...`
    # on the A line does not match, since its `=` does not follow alpha^2.
    SEP = r",\s*(?:\\quad|\\qquad|&)?\s*"
    ab = re.search(r"\\alpha\^2\s*&?=\s*([^,]+?)" + SEP
                   + r"\\beta\^2\s*&?=\s*([^,]+?),", para)
    assert ab, ("the paragraph under eq:aunfactored no longer gives "
                "`alpha^2 = X, beta^2 = Y,' on one line")
    gd = re.search(r"\\gamma_\\mathrm\{icos\}\^2 = \\delta_\\mathrm\{dodec\}\^2"
                   r"\s*&?=\s*([^,]+?)" + SEP
                   + r"\\delta_\\mathrm\{icos\}\^2 = \\gamma_\\mathrm\{dodec\}\^2"
                   r"\s*&?=\s*([^,]+?),", para)
    assert gd, ("the paragraph under eq:aunfactored no longer gives the two "
                "gamma/delta pairing values on one line")
    sq_ab = (float(_tex_number(ab.group(1))), float(_tex_number(ab.group(2))))
    sq_gi, sq_di = float(_tex_number(gd.group(1))), float(_tex_number(gd.group(2)))
    worst32 = 0.0
    for s in a5:
        icos = s == "icosahedron"
        nq, na, m = (4, 2, 3) if icos else (5, 2, 5)
        V = F[s]["V"]
        assert V == 4 * m, (s, V, m)
        pref_tex = (mA.group(1).replace(r"\,", "")
                    .replace("{m}", "{%d}" % m).replace("{V}", "{%d}" % V))
        pref32 = float(_tex_number(pref_tex))
        root = {"alpha": sq_ab[0], "beta": sq_ab[1],
                "gamma": sq_gi if icos else sq_di,
                "delta": sq_di if icos else sq_gi}
        root = {k: v ** 0.5 for k, v in root.items()}      # positive roots

        def entry(c, root=root):
            m_ = re.fullmatch(r"(-?)\\(alpha|beta|gamma|delta)", c)
            assert m_, f"{s}: unparsed entry {c!r} in the unfactored A"
            return (-1 if m_.group(1) else 1) * root[m_.group(2)]
        A32 = pref32 * np.array([[entry(c) for c in r] for r in rows32])
        assert abs(A32.T @ A32 - np.eye(4)).max() < LOOSE, \
            (f"{s}: the A parsed from eq:aunfactored is not orthogonal "
             f"(prefactor {mA.group(1)} = {pref32:.6f})")
        Fb = _pad_block(decker_fourier(m, FOURIER_SIGN[s]), 2 ** (nq - na))
        iota = np.zeros((2 ** nq, 2))
        iota[0, 0] = iota[1, 1] = 1
        W32 = (np.kron(A32.T, Fb.conj().T)
               @ _cnot(nq, ctrl=nq - 1, targ=na - 1).T @ iota)
        d32 = abs(W32 - decker_circuit(s)).max()
        assert d32 < LOOSE, \
            (f"{s}: the A parsed from eq:aunfactored (prefactor {pref32:.6f}, "
             f"roots {root}) does not rebuild Decker's circuit ({d32:.3e})")
        worst32 = max(worst32, d32)
    report.append(f"  32b. A rebuilt from eq:aunfactored's printed prefactor "
                  f"and matrix body and the paragraph's alpha/beta pair and "
                  f"gamma/delta pairing, roots positive: orthogonal, and "
                  f"kron(A^dag, F_b^dag) Q^dag iota reproduces decker_circuit "
                  f"on both A_5 solids (worst {worst32:.1e})")

    # --- 33. the worked E_1, symbolically and numerically -------------------
    worst33 = 0.0
    for s in SOLIDS:
        if s not in WORKED_E1:
            continue
        f = F[s]
        E1 = (I2 + sum(f["v"][0][a] * PAULI[a] for a in range(3))) / f["V"]
        worst33 = max(worst33, np.abs(E1 - f["E"][0]).max())
        # ... and the printed atoms, parsed out of the emitted string
        m33 = re.search(r"\$E_1 = \\tfrac1\{(\d+)\}\\pmat\{(.+?)\}\$",
                        " ".join(F[s]["emitted"]))
        assert m33, f"{s}: the worked E_1 does not parse"
        assert int(m33.group(1)) == f["V"], \
            f"{s}: the worked E_1 is scaled 1/{m33.group(1)}, not 1/{f['V']}"
        got33 = np.array([[_tex_complex(x) for x in row.split("&")]
                          for row in m33.group(2).split(r"\\")]) / f["V"]
        assert np.abs(got33 - f["E"][0]).max() < 1e-15, \
            (f"{s}: the printed E_1 is {got33}, the npz has {f['E'][0]}")
        worst33 = max(worst33, np.abs(got33 - f["E"][0]).max())
    report.append(f"  33. the worked E_1 on the two A_5 cards equals "
                  f"(1/V)(1 + n_1.sigma) and the npz, atom by atom parsed back "
                  f"out of the printed string (worst {worst33:.1e})")

    # --- 34. the pointer set ------------------------------------------------
    # The pointer column is what pays for the card's cuts, and a pointer
    # that does not resolve is a paragraph cut for nothing.  TEN live \refs
    # per card, nine on the tetrahedron (which has no atlas words to route):
    # one on the title line, two at each of the four band heads, and head 4's
    # third on the four cards that print atlas words.
    aux = PAPER / "bsc-thesis.aux"
    # A label the source declares but the .aux has not seen yet is live: the
    # .aux is one compile stale by construction, so a label added to
    # bsc-thesis.tex is exactly that case on the first build after it lands.
    known = set(re.findall(r"\\label\{([^}]*)\}", chapter))
    # Five pointers resolve to labels living in fragments the thesis \inputs
    # (atlas.tex's two group tables; the povm-atlas and the two decker
    # tables), which the source scan above cannot see and which a fresh
    # clone's missing .aux cannot rescue -- so read the \input'ed files
    # themselves.  ../code/ resolves against CODE, not PAPER: the controls
    # sandbox a copy of paper/ alone, and the fresh clone has both trees but
    # no .aux.
    for frag in re.findall(r"\\input\{([^}]*)\}", uncommented(chapter)):
        p = CODE / frag[len("../code/"):] if frag.startswith("../code/") \
            else PAPER / frag
        p = p if p.suffix == ".tex" else p.with_suffix(".tex")
        if p.is_file():
            known |= set(re.findall(r"\\label\{([^}]*)\}", p.read_text()))
    if aux.is_file():
        known |= set(re.findall(r"\\newlabel\{([^}]*)\}", aux.read_text()))
    inline = 0
    for s in SOLIDS:
        ptrs = re.findall(r"\\(?:ref|eqref)\{([^}]*)\}", " ".join(heads(F[s])))
        want = 9 if not F[s]["rays"] else 10
        assert len(ptrs) == want, \
            f"{s}: {len(ptrs)} pointers in the pointer column, not {want}"
        assert "fig:tetrahedron-recipe" in ptrs, \
            f"{s}: no route back to Figure 1.1"
        # every OTHER \ref on the card -- the A^dagger line's \eqref is the
        # only one -- must resolve as well: a pointer that does not resolve is
        # a paragraph that was cut for nothing.
        rest = [x for x in re.findall(r"\\(?:ref|eqref)\{([^}]*)\}",
                                      " ".join(F[s]["emitted"]))
                if x not in ptrs]
        inline += len(rest)
        miss = [x for x in ptrs + rest if x not in known]
        assert not miss, f"{s}: pointers with no label anywhere: {miss}"
    report.append(f"  34. ten pointers per card (nine on the tetrahedron) plus "
                  f"{inline} inline, every one a label declared in "
                  f"bsc-thesis.tex or a fragment it \\inputs (.aux fallback "
                  f"only), and Figure~\\ref{{fig:tetrahedron-recipe}} on all "
                  f"five")

    # --- 35. the hand-written body closes over the emitted data file --------
    # Nothing else joins the two halves of a card.  Check 12 builds `page` =
    # emitted + body and every literal it wants is already in `emitted`, so
    # its body half never bites; no other check reads a body for macro USE.  A
    # body that stops calling an emitted macro, or \inputs the wrong solid's
    # data file, would otherwise ship with every other check reporting pass.
    for s in SOLIDS:
        name = f"card_{SHORT[s]}_body.tex"
        body = uncommented(bodies.get(name, ""))
        if not body:
            report.append(f"  35. {s}: SKIPPED -- {name} does not exist yet")
            continue
        # TWO legal spellings, and exactly one of them: a scratch copy keeps
        # the data beside the body in paper/figures/, this tree keeps it in
        # code/data/, where every fragment the thesis \inputs lives.  What is
        # checked is that the body inputs ITS OWN solid's file, once, and no
        # other card's.
        ins = re.findall(r"\\input\{(?:\.\./code/data|figures)/(card_\w+_data)\}",
                         body)
        assert ins == [f"card_{SHORT[s]}_data"], \
            (f"{s}: {name} must \\input its OWN data file exactly once "
             f"(figures/ beside the body, or ../code/data/); "
             f"it inputs {ins}")
        defined = re.findall(r"\\newcommand\{\\(C[A-Za-z]*)\}\{(.*)",
                             "\n".join(F[s]["emitted"]))
        # a macro the emitter deliberately writes EMPTY (\CIdentB off the A_5
        # pair, \CRays on the tetrahedron) need not be called
        blob = "\n".join(F[s]["emitted"])
        used, empty = 0, 0
        for mac, _ in defined:
            body_of = re.search(r"\\newcommand\{\\%s\}\{" % mac, blob)
            assert body_of, f"{s}: {mac} vanished between emit and check"
            j = body_of.end()
            d, k = 1, j
            while d:
                d += (blob[k] == "{") - (blob[k] == "}")
                k += 1
            if not blob[j:k - 1].strip():
                empty += 1
                continue
            assert re.search(r"\\%s(?![A-Za-z])" % mac, body), \
                (f"{s}: {name} never calls \\{mac}, which "
                 f"card_{SHORT[s]}_data.tex defines with content -- the "
                 f"printed card would silently lose it")
            used += 1
        report.append(f"  35. {s}: {name} inputs its own data file and calls "
                      f"all {used} non-empty macros of the {used + empty} "
                      f"card_{SHORT[s]}_data.tex defines")

    # --- 36. every derived warning is a PREDICATE on the value it qualifies --
    # The two per-card kappa riders are retired (the note above kappa_num
    # has the reasons), but the discipline stays: Table D.4's printed kappa
    # must carry the sign of the recomputed value, head 3's em-dash gloss is
    # derived from the dead count, and a retired rider re-entering a label
    # is an abort, the way 19b keeps "after the alignment" out.
    for s in SOLIDS:
        f = F[s]
        one = " ".join(f["emitted"])
        k = kappa_num(f)
        kap = f["labels"][2]
        printed_neg = kap.lstrip("$").startswith("-")
        assert printed_neg == (k < 0), \
            f"{s}: Table D.4 prints {kap!r} and the recomputed kappa is {k}"
        # ... the noun is the estimator's, never eta's: "the calibration
        # scalar" is eta, noiseless 1/3, and kappa = 3 eta is the estimator
        # channel's multiplier
        assert "calibration" not in one, \
            f"{s}: a card calls kappa `the calibration'; that name is eta's"
        # ... and the riders' retirement holds
        assert "Negative:" not in one and "ill-conditioned" not in one, \
            f"{s}: a retired kappa rider is printed again"
        assert (HEAD_C_DEAD in one) == bool(f["dead"]), \
            (f"{s}: {len(f['dead'])} dead outcomes and the key's em-dash "
             f"gloss is {'printed' if HEAD_C_DEAD in one else 'absent'}")
    # ... and the free pairing AND the priced misread the kappa label now
    # prints are D.2's own claims, not the card's: if D.2 stops saying one,
    # the card must stop too.
    for lit in ("or neither, and keep Decker's pose and vertex list",
                r"$\kappa = 1$ and offset zero",
                r"his circuit with our list gives the $\kappa$ in"):
        assert " ".join(chapter.split()).count(" ".join(lit.split())) == 1, \
            (f"the kappa label leans on D.2 for its pairing and its priced "
             f"misread; D.2 no longer says {lit!r}")
    report.append("  36. Table D.4's kappa sign and head 3's em-dash gloss "
                  "are predicates on the derived values, the two per-card "
                  "riders stay retired, and no card prints `calibration'; "
                  "the free pairing and the priced misread the kappa label "
                  "prints are D.2's own sentences, checked there")

    # --- 12. the facts each card's surface hand-types -----------------------
    # The heads and labels are emitted and the rest of a card's surface is
    # hand-typed in its body file, so the check reads the PAGE -- body plus
    # emitted.  The controls corrupt these literals in the generator, so the
    # check keeps its power: a rewritten sentence is allowed, a drifted fact
    # is not.
    quoted = 0
    for s in SOLIDS:
        name = f"card_{SHORT[s]}_body.tex"
        f = F[s]
        prose = " ".join(uncommented(bodies.get(name, "")).split())
        page = " ".join((" ".join(f["emitted"]) + " " + prose).split())

        def says(t, why):
            assert t in page, \
                f"card {SHORT[s]} no longer says {t!r} -- {why}"

        says(card_title(f), "the title, and the series signature")
        for c in series_strip_row(s)[2]:
            says(c, "a strip cell of Figure 1.1's own row")
        says(STRIP_FOOTER, "the exactness conjunction, as the strip's footer")
        says(HEAD_A, "band 1's head, and Equation 2.1's form")
        says(HEAD_B, "band 2's head, and what makes the dashed box parse")
        says(HEAD_C + (HEAD_C_DEAD if f["dead"] else ""),
             "Table D.4's own register convention, and what --- means")
        says(HEAD_D_NONE if not f["rays"] else HEAD_D,
             "band 4's head, and the reading order of a ray")
        says(UG_RIDER, "the direction, g^{-1}, on every card")
        says(NAIMARK_GLOSS, "what Mtilde-dagger is, and where on the "
                            "page it is drawn")
        says(r"Run $U_R$ and read the key, or skip $U_R$ and read "
             r"Decker's own numbered list~"
             r"\cite{decker2004quantumcircuitssinglequbit}: either way "
             r"$\kappa = 1$, free.",
             "the free pairing with Decker's name and cite, the card's own "
             "protocol its first branch, printed beside the priced misread")
        quoted += 10 + len(series_strip_row(s)[2])
        # the caption is the one cold name on the page, and it lives in
        # bsc-thesis.tex; check it there once the float is in place
        if r"\input{figures/card_%s_body}" % SHORT[s] in chapter:
            assert chapter.count(r"\caption{%s}" % f["caption"]) == 1, \
                (f"{s}: the float's caption is not the shared one-line "
                 f"attribution with Decker's Figure {DECKER_FIG[s]}")
            quoted += 1
    report.append(f"  12. {quoted} facts verified verbatim across the five "
                  f"card surfaces (title, the seven strip cells, the exactness "
                  f"footer, four band heads, the g^-1 rider, the Naimark "
                  f"gloss, the caption)")
    return report


def caption_tokens(caption):
    r"""Every math token of a current caption that a card must still print.

    Inline math atoms, minus the ones that are pure cross-reference or pure
    prose scaffolding -- what is left is the parameter set: U_A, alpha, beta,
    the Fourier block, Q-dagger and its kets, Mtilde-dagger, B, C, u+-, v+-.
    """
    skip = {"$R$", "$B$", "$C$", "$I_2$", "$\\rho$", "$k$", "$V$", "$T^\\dagger$",
            "$U_R$", "$U_g$", "$\\alpha$", "$\\beta$", "$\\gamma$", "$\\delta$",
            "$-\\sigmaz$", "$4$", "$3$", "$2$", "$1$", "$5$"}
    out = []
    for m in re.finditer(r"\$[^$]+\$", caption):
        t = m.group(0)
        if t in skip or len(t) < 6:
            continue
        if t.startswith(r"$\alpha, \beta") or t.startswith(r"$\alpha,\beta"):
            continue
        out.append(t)
    return out


def chip_geometry(f):
    r"""(minimum chip separation in pt, the chipped vertices).

    Recomputed from the panel's own projection at the emitted \PanelScale, so a
    future re-pose or rescale that breaks legibility fails the build rather than
    shipping.  The camera is tdplot's {60}{120} and the screen basis is the one
    _povm_sphere_preamble writes down in \drawBladePoly; the chip offset is
    read off the panel's own ``!-Npt!`` where there is a panel.

    The offset is applied in SCREEN space, not in 3D -- ``($(vk)!-9pt!(O)$)``
    interpolates two points TikZ has already projected, so a vertex pointing
    near the camera has a short projected radius and its chip moves out along
    that short radius.  Computing the offset in 3D and projecting afterwards
    overstates the dodecahedron's near-end separation by nearly threefold.

    The screen basis itself is pinned to the camera by the two asserts below;
    the note there says why, and what it cost to find out.

    The rule: chip every vertex when the all-V minimum clears two chip
    diameters, otherwise one chip per axis at the near end -- and every
    colliding pair is one near and one far vertex, so the near-end rule
    removes the whole class.
    """
    view = np.array([0.75, 0.4330127, 0.5])          # the preamble's own
    ex = np.array([-0.5, 0.8660254, 0.0])            # (cos 120, sin 120, 0)
    ey = np.array([-0.4330127, -0.25, 0.8660254])
    # The screen basis is pinned to the camera rather than trusted: the panel
    # draws in tdplot's {60}{120} and \viewX,\viewY,\viewZ is the camera
    # normal, so a right-handed (ex, ey) MUST cross to it.  A basis that merely
    # looks like tdplot's can be the true one pre-composed with a turn about
    # z-hat -- which projects a ROTATED solid, agreeing on the octahedron and
    # the cube (both invariant under that turn) and disagreeing by up to 115pt
    # on the icosahedron and the dodecahedron.  Verified against four compiled
    # panels: with this basis every chip lands within 0.01pt of the position
    # \pgfgetlastxy reports.
    assert abs(ex @ view) < 1e-8 and abs(ey @ view) < 1e-8, \
        "the screen basis is not orthogonal to the panel's own camera"
    assert np.abs(np.cross(ex, ey) - view).max() < 1e-8, \
        "the screen basis does not cross to the panel's own camera"
    R_pt = f["panel_scale"] * 28.45276          # TikZ cm -> pt
    v = f["v"]
    V = len(v)

    def screen(k):
        p = R_pt * np.array([v[k] @ ex, v[k] @ ey])
        r = np.linalg.norm(p)
        assert r > 1.0, f"vertex {k+1} projects onto the centre; no chip ray"
        # per-vertex offsets, because they are not uniform and the difference
        # matters: a vertex pointing near the camera has a short projected
        # radius, so Figure 1.1 pushes its chip out 11pt instead of 9 to
        # keep it clear of a foreshortened blade, and the cube and the
        # dodecahedron inherit that remedy for exactly the same geometry.
        # Taking the minimum offset for all of them under-reports those
        # separations.
        return p * (1 + f["chip_offsets"].get(k + 1, f["chip_offset"]) / r)

    def minsep(ks):
        return min(np.linalg.norm(screen(i) - screen(j))
                   for a, i in enumerate(ks) for j in ks[a + 1:])

    allv = list(range(V))
    if minsep(allv) >= 19.0:
        return minsep(allv), [k + 1 for k in allv]
    near, seen = [], []
    for k in range(V):
        a = antipode(v, k)
        if a is None:
            near.append(k)
        elif k not in seen:
            seen += [k, a]
            near.append(k if v[k] @ view > v[a] @ view else a)
    return minsep(sorted(near)), sorted(k + 1 for k in near)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

HEADER = r"""%% Auto-generated by code/_build_povm_cards.py -- DO NOT EDIT.
%% The data blocks of the %s card of Appendix D
%% (paper/figures/card_%s_body.tex \input's this and calls them).  Every
%% number is derived from code/data/*.npz, Decker's circuits as
%% randomized_decker rebuilds them, Figure 1.1's own \RecipeSeriesStrip, and
%% Tables E.1, 5.2, D.1, D.3 and D.4, and is checked back against them before
%% this file is written; see the builder's docstring for the checks (and
%% _povm_cards_controls.py, which shows each of them failing on one corrupted
%% literal: 100 negative controls, all aborting, 0 stale).
%% Requires: booktabs, mathtools, hyperref, and the thesis macros \Id, \pmat,
%% \bra, \ket, \braket, \Tr, \C, \sigmavec, \TwoT, \TwoO, \TwoI, \phig,
%% \invphig, \sigmaz.  The body defines \NIS, \CTR, \PBOX, \COL, \FULLHEAD,
%% \BANDRULE and \TAB, which \CardSide and the tables use.
%% Regenerate with `cd code && uv run _build_povm_cards.py`.
"""


def build(out_dir, panel_scale=2.47):
    F = {}
    for s in SOLIDS:
        has_circ = (FIGURES / f"card_dec_{SHORT[s]}.tex").is_file()
        f = facts(s, has_card_files=has_circ)
        f["has_card_circuit"] = has_circ
        f["sphere_src"] = SPHERE_SRC[s]
        f["has_sphere"] = (FIGURES / f["sphere_src"]).is_file()
        f["has_body"] = (FIGURES / f"card_{SHORT[s]}_body.tex").is_file()
        f["chip_offset"] = 9.0
        f["chip_offsets"] = {}
        if f["has_sphere"]:
            src = (FIGURES / f["sphere_src"]).read_text()
            f["panel_scale"] = float(re.search(
                r"\\renewcommand\{\\PanelScale\}\{([\d.]+)\}", src).group(1))
            f["chip_offsets"] = {int(k): float(o) for k, o in re.findall(
                r"\\chipnode\{(\d+)\}\{\(\$\(\w+\)!-([\d.]+)pt!\(O\)\$\)\}", src)}
            assert f["chip_offsets"], \
                f"{s}: {f['sphere_src']} places no chip on a radial offset"
            f["chip_offset"] = min(f["chip_offsets"].values())
            f["panel_names_axes"] = r"{$\hat x$}" in src
        else:
            f["panel_scale"] = panel_scale
            f["panel_names_axes"] = True
        f["chips"] = chip_geometry(f)[1]
        f["chips_all"] = len(f["chips"]) == f["V"]
        # U_R's closed form: lifted from the FROZEN circuit, where it is what
        # the gate box drew before the box was named.  Never typed here.
        g = re.search(r"\\gate\{(T\^\\dagger|R_\{\\hat.*?)\} & \\ctrl",
                      f["frozen_src"])
        assert g, f"{s}: no reorientation gate in circuit_dec_{SHORT[s]}.tex"
        f["UR_tex"] = g.group(1)
        F[s] = f

    bodies = {}
    for s in SOLIDS:
        p = FIGURES / f"card_{SHORT[s]}_body.tex"
        if p.is_file():
            bodies[p.name] = p.read_text()
    for s in SOLIDS:
        f = F[s]
        f["caption"] = caption(f)
        # each block is emitted exactly ONCE and then reused: calling an
        # emitter twice quietly undid a mutation-based negative control
        hd, st, vt = heads(f), strip(f), vertex_blocks(f)
        ident, par, sd = identities(f), params(f), card_side(f)
        kr, kp, ry, cn = key_rectangle(f), kappa_label(f), rays(f), protocol(f)
        # what sets at \footnotesize on a card: the strip and the pointer
        # column, and nothing else.  Everything a reader retypes is \small or
        # larger, which is the type floor assertion 26 enforces.
        fn = st + [b for b in hd if r"\CPtr" in b]
        small = [b for b in hd if r"\CPtr" not in b] + vt + ident + par \
            + sd + kr + kp + ry + cn
        blocks = (hd + [""] + st + [""] + vt + [""] + ident + [""] + par
                  + [""] + sd + [""] + kr + [""] + kp + [""] + ry
                  + [""] + cn)
        f["emitted"] = blocks
        f["fn_blocks"] = fn                             # set at \footnotesize
        f["small_blocks"] = small
        f["text"] = (HEADER % (SOLID_ADJ[s], SHORT[s])).rstrip() + "\n\n" \
            + "\n".join(blocks) + "\n"

    report = []
    verify(F, bodies, report)
    return {s: F[s]["text"] for s in SOLIDS}, report


def main(argv=()):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="directory for card_<solid>_data.tex "
                         "(default code/data/; required when code/ is a "
                         "symlink)")
    ns = ap.parse_args(list(argv))
    # In a scratch copy code/ is a SYMLINK to the real repo, so the default
    # --out resolves through it and a bare run would write five files into the
    # repository.  Refuse the default there and make the destination explicit.
    if ns.out is None:
        assert not CODE.is_symlink() and (_HERE / "data").is_dir(), \
            ("--out is required: this tree's code/ is a symlink, so the "
             f"default would write through it to {DATA}")
        ns.out = str(DATA)
    out = Path(ns.out)
    assert out.is_dir(), f"--out {out} is not a directory"
    texts, report = build(out)
    for s in SOLIDS:
        (out / f"card_{SHORT[s]}_data.tex").write_text(texts[s])
    print(f"wrote {len(texts)} files to {out}")
    for line in report:
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
