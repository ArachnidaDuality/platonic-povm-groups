"""Build ``paper/figures/section_ladder.tex`` -- the Appendix C section ladder.

The figure draws the *parallel sections* of the three polytopes: slice each
binary polyhedral group at constant real part $w$ and the vertices you meet are
the axes of the rotations that section holds.  Everything drawn is derived here
from ``code/data/group_{2T,2O,2I}.npz`` -- the vertex coordinates, the face
lists, and the per-rung element counts -- so the figure cannot drift from
Table~C.1, which is read off the same group data.

What the sections ARE is pinned rather than typed: ``check_solids`` matches
every drawn solid against ``povm_properties``' POVM atlas, which is the first
code behind Appendix C.2's claim that each parallel section is a Platonic
solid *vertex for vertex the atlas*, and ``check_rungs`` then requires each
rung to draw the solid its own section actually is.  Between them they say
what the figure exists to say: 2O meets 2T's octahedron and cube at its own
latitudes, and 2I meets one icosahedron twice.

The *layout* is arithmetic, not taste.  Figure C.1 prints at the text width,
so the drawing's own width is a divisor: a centimetre of air between two
columns is paid for by shrinking every solid and every label on the page.  So
the columns are placed from the type metrics in ``MEASURED`` -- each gap is
exactly what the longest label reaching into it needs -- and the vertical
scale ``H`` falls out of ``tightest_pair()``, the closest two rungs that both
draw a solid, which is what caps how large an icon can be.  The figure then
checks the arithmetic back: every label is set through ``\\WCHECK`` carrying
the width assumed for it here, so a label that sets wider than the layout
allowed stops the LaTeX build instead of silently reaching into its neighbour.

Edit this builder, never ``paper/figures/section_ladder.tex`` (regeneration
overwrites manual edits).  The TikZ vertex-storage, back-face-culling and
Lambertian-shading machinery is lifted at build time from
``paper/figures/platonic_solids.tex``, so the solids here are lit exactly like
its panels (that figure is repository-only, not printed in the thesis); the
lift rewrites one thing, the face edge weight, which belongs to the size a
figure draws at rather than to the lighting.

Run with ``uv run _build_section_ladder.py``.  Afterwards recompile the
standalone figure so the PDF in the tree stays current::

    cd ../paper/figures && latexmk -pdf section_ladder.tex
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

import povm_properties

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT.parent / "paper" / "figures"
OUT = FIGURES / "section_ladder.tex"
MACHINERY = FIGURES / "platonic_solids.tex"

ORDER = {"2T": 24, "2O": 48, "2I": 120}

# --- type ------------------------------------------------------------------
# Sizes in pt, and larger than they look: the standalone page comes out 23.8 cm
# across and Figure C.1 prints it at the 15.9 cm text width, so type set here
# reaches the page at roughly 0.67 of its size -- a 14.4 pt column title lands
# as 10 pt against an 11 pt body, a 12 pt rung label as 8 pt.  Note which way
# ICON pushes that: smaller icons narrow the drawing, so they raise the
# residual scale and the type with it.  PREAMBLE's font macros are generated
# from these very numbers and the layout arithmetic below reads the same ones,
# so the type that is measured and the type that is set cannot come apart.
PT = 0.0351459804    # cm per TeX point
LABEL_PT = 12.0      # rung labels, and the pole labels
TICK_PT = 10.95      # the w / theta scale, and its header
TITLE_PT = 14.4      # the polytope name over a column
SMALL_PT = 10.95     # the covering map, and the order line under a column
FACE_LW = 0.55       # pt, the weight of a solid's face edges
SLACK = 0.03         # cm, what \WCHECK forgives MEASURED for being rounded

# Width of every string the figure sets, in cm per pt of type size, each one
# measured with \settowidth AT THE SIZE IT IS SET AT -- not at a reference size
# and scaled up.  Computer Modern is optically scaled, so that distinction is
# worth centimetres: cmr12 and cmr17 are their own designs and run narrower per
# point than cmr10, and in math neither the script sizes nor \scriptspace track
# the base at all.  Measured at 10 pt and read at these sizes, the table
# over-reserved by ~2% on the rung labels, ~3% on the titles and up to 8% on the
# tick math.  All of it safe-side, which is the trap: \WCHECK measures at the
# set size, so an over-estimate can never fire it, and the cost shows up only as
# gaps wider than the type needs -- in a figure the page scales by its width,
# that is the whole budget leaking quietly.
#
# So: RE-MEASURE whenever a *_PT above changes or a rung is relabelled.  The
# builder writes the document that does it -- ``--measure`` (see measure()).
# A string with no entry here fails this build; one that sets wider than its
# entry fails the LaTeX compile.  Five decimals leave about half a point of
# rounding to forgive, which is SLACK; the airs it guards are more than an
# order of magnitude wider, so nothing an eye could see hides inside it.
MEASURED = {
    # set at LABEL_PT
    "$q = 1$": 0.08002,
    "$q = -1$": 0.10735,
    "cube (8)": 0.12523,
    "octahedron (6)": 0.22273,
    "$+$ cuboctahedron (12)": 0.33266,
    "icosahedron (12)": 0.24873,
    "dodecahedron (20)": 0.28009,
    "icosidodecahedron (30)": 0.34528,
    # set at TICK_PT
    r"$w$\quad$\theta$": 0.07873,
    "$1$": 0.01757,
    r"$0^\circ$": 0.03282,
    "$-1$": 0.04491,
    r"$360^\circ$": 0.06797,
    r"$\phig/2$": 0.05449,
    r"$72^\circ$": 0.05039,
    r"$-\phig/2$": 0.08182,
    r"$288^\circ$": 0.06797,
    r"$1/\sqrt{2}$": 0.08201,
    r"$90^\circ$": 0.05039,
    r"$-1/\sqrt{2}$": 0.10934,
    r"$270^\circ$": 0.06797,
    "$1/2$": 0.05272,
    r"$120^\circ$": 0.06797,
    "$-1/2$": 0.08005,
    r"$240^\circ$": 0.06797,
    r"$\invphig/2$": 0.05649,
    r"$144^\circ$": 0.06797,
    r"$-\invphig/2$": 0.08383,
    r"$216^\circ$": 0.06797,
    "$0$": 0.01757,
    r"$180^\circ$": 0.06797,
    # set at TITLE_PT
    r"\textit{24-cell}": 0.10030,
    r"\textit{24-cell} and dual": 0.24369,
    r"\textit{600-cell}": 0.11787,
    # set at SMALL_PT
    r"$\TwoT \to \Tgroup$": 0.12839,
    r"$\lvert 2\mathcal{T} \rvert = 24$": 0.14718,
    r"$\TwoO \to \Ogroup$": 0.13016,
    r"$\lvert 2\mathcal{O} \rvert = 48$": 0.14806,
    r"$\TwoI \to \Igroup$": 0.11571,
    r"$\lvert 2\mathcal{I} \rvert = 120$": 0.15841,
}

# --- layout, in cm ---------------------------------------------------------
# Only ICON, CLEAR and the four airs are chosen; H and every x follow.
INNER = 2 * 1.4 * PT    # inner sep of a rung label's box, both sides
TICKBOX = 2 * 1.2 * PT  # inner sep of a scale label's box, both sides

ICON = 0.60             # circumradius of a solid icon
CLEAR = 0.22            # air left between the closest pair of icons
GUTTER = 1.55           # air between a column's longest label and the next column
GUTTER_L = 0.50         # air between the w / theta scale and the first column
LABEL_DX = ICON + 0.20  # axis -> left edge of an ordinary rung's label
PAIR_DX = ICON + 0.19   # axis -> centre of each icon of the paired rung
PAIR_DL = PAIR_DX + ICON + 0.15   # axis -> left edge of the paired rung's label
TICK_DX = 0.13          # axis -> a scale label, either side
TICK_LEN = 0.11         # half-length of a tick mark
DOT = 0.075             # radius of a pole dot
TITLE_DY = 0.70         # column head: the polytope name ...
COVER_DY = 0.22         # ... and the covering map under it
ORDER_DY = 0.52         # column foot: the order line
GUIDES = [1.0, 0.5, 0.0]  # the 24-cell's latitudes, shared by all three

DRAW = {3: r"\drawTri", 4: r"\drawQuad", 5: r"\drawPent"}

# 2O's equator is the one rung carrying two conjugacy classes, so it is the one
# rung that draws two solids, at -+PAIR_DX: macro name, printed name.
PAIR = (("oct", "octahedron"), ("cubocta", "cuboctahedron"))

POLE = {1: "$q = 1$", -1: "$q = -1$"}

# What each drawn solid must BE, not merely be called: vertex count and the
# multiset of face sizes.  The names are typed into COLUMNS and register(), and
# nothing else in the file would notice if one were wrong -- DRAW[len(f)] only
# rejects a face that is neither a 3-, 4- nor 5-gon, so a 12-vertex,
# 20-triangle solid labelled "dodecahedron" would draw without complaint.
SIGNATURE = {"cube":    (8, {4: 6}),
             "oct":     (6, {3: 8}),
             "ico":     (12, {3: 20}),
             "dod":     (20, {5: 12}),
             "icodod":  (30, {3: 20, 5: 12}),
             "cubocta": (12, {3: 8, 4: 6})}

# The atlas targets check_solids pins against: the four Platonic sections
# answer to a povm_properties solid directly, the two Archimedean ones to their
# dual pairs RECTIFIED -- which is what identifies them, and why each answers
# to both members of its pair.
ATLAS = {"cube": "cube", "oct": "octahedron",
         "ico": "icosahedron", "dod": "dodecahedron"}
RECTIFIED = {"cubocta": ("octahedron", "cube"),
             "icodod": ("icosahedron", "dodecahedron")}

# --------------------------------------------------------------- the geometry

def quaternions(group):
    """Recover (w, x, y, z) from the stored SU(2) matrices."""
    U = np.load(DATA / f"group_{group}.npz")["unitaries"]
    w = np.real(np.trace(U, axis1=1, axis2=2)) / 2
    x = np.imag(U[:, 0, 0])
    y = np.real(U[:, 0, 1])
    z = np.imag(U[:, 0, 1])
    return np.stack([w, x, y, z], axis=1)


def bloch_axis(q):
    """q = w + u_1 i + u_2 j + u_3 k  ->  Bloch axis n = -(u_3, u_2, u_1)."""
    v = q[1:]
    n = -np.array([v[2], v[1], v[0]])
    length = np.linalg.norm(n)
    return n / length if length > 1e-12 else n


def merge_coplanar(points, hull):
    """Group the hull's simplices into polygonal faces by their plane, then put
    each face's vertices in cyclic order so the polygon does not self-cross."""
    buckets = defaultdict(set)
    for equation, simplex in zip(hull.equations, hull.simplices):
        buckets[tuple(np.round(equation, 6))].update(int(i) for i in simplex)

    faces = []
    for plane, idxs in buckets.items():
        idxs = list(idxs)
        normal = np.array(plane[:3])
        centroid = points[idxs].mean(axis=0)
        ref = points[idxs[0]] - centroid
        ref /= np.linalg.norm(ref)
        perp = np.cross(normal, ref)
        angles = [np.arctan2((points[i] - centroid) @ perp,
                             (points[i] - centroid) @ ref) for i in idxs]
        faces.append([idxs[k] for k in np.argsort(angles)])
    return faces


def sections(group):
    """The sections at w >= 0; the ones below are their negatives."""
    q = quaternions(group)
    ws = q[:, 0]
    out = []
    for c in sorted({round(v, 9) for v in ws}, reverse=True):
        if c < -1e-9:
            continue
        members = q[np.abs(ws - c) < 1e-7]
        entry = dict(w=c, n=len(members), verts=None, faces=None)
        if abs(c) > 1 - 1e-9:  # a pole is a single point, not a polyhedron
            out.append(entry)
            continue
        pts = np.array([bloch_axis(m) for m in members])
        entry["verts"] = pts
        entry["faces"] = merge_coplanar(pts, ConvexHull(pts))
        out.append(entry)
    return out


def check_mirrors():
    """The figure draws one solid for both signs of w.  That is only honest
    because the section at -c is the negative of the one at +c (q and -q differ
    by a sign throughout) and every section solid here is centrally symmetric,
    so negating its vertices returns the same set.  Verify it rather than trust
    it -- if a future edit breaks the symmetry, the lower half would silently
    become a lie."""
    for group in ORDER:
        q = quaternions(group)
        ws = q[:, 0]
        for c in sorted({round(v, 9) for v in ws}):
            if not 1e-9 < c < 1 - 1e-9:
                continue
            up = {tuple(np.round(bloch_axis(m), 6) + 0.0) for m in q[np.abs(ws - c) < 1e-7]}
            down = {tuple(np.round(bloch_axis(m), 6) + 0.0) for m in q[np.abs(ws + c) < 1e-7]}
            assert up == down, f"{group}: sections at w = +/-{c:.6f} are not the same solid"


def match(A, B):
    """Pair the point set A against B one for one; return the two margins.

    A tolerance is unavoidable here -- these vertices arrive as float64 off the
    npz while the atlas's come out of SymPy -- so what is asserted is the
    MARGIN and not a threshold either side sits near: every vertex matches
    nearer than 1e-9 and every runner-up sits farther than 1e-9, and the two
    measured ends are returned so main() can print the gap they leave.

    The bijection is asserted too.  Without it this would be containment rather
    than set equality, and a count already checked elsewhere would be carrying
    the argument.
    """
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    assert len(A) == len(B), f"{len(A)} vertices against {len(B)}"
    worst, runner, hit = 0.0, np.inf, set()
    for row in A:
        d = np.linalg.norm(B - row, axis=1)
        k = int(np.argmin(d))
        worst, runner = max(worst, d[k]), min(runner, np.sort(d)[1])
        hit.add(k)
    assert len(hit) == len(B), "the pairing is not a bijection"
    assert worst < 1e-9 < runner, f"match {worst:.3e}, runner-up {runner:.3e}"
    return worst, runner


def atlas_solid(name):
    """povm_properties' vertices as floats, in the atlas's published order."""
    return np.array([[float(c) for c in v]
                     for v in povm_properties.numeric_vertices(name)])


def rectify(verts):
    """The normalized edge midpoints of a solid -- its rectification.

    Edges are read off the same convex hull and the same cyclic face ordering
    the figure draws with, and deduplicated on integer vertex indices, so this
    introduces no threshold of its own.
    """
    v = np.asarray(verts, dtype=float)
    edges = set()
    for f in merge_coplanar(v, ConvexHull(v)):
        for a, b in zip(f, f[1:] + f[:1]):
            edges.add((min(a, b), max(a, b)))
    mids = np.array([v[a] + v[b] for a, b in sorted(edges)])
    return mids / np.linalg.norm(mids, axis=1)[:, None]


def check_solids(store, faces):
    """Every section solid IS the solid it is named -- two pins, because a name
    makes two claims.  SIGNATURE settles the combinatorics (how many vertices,
    which faces).  Then the four Platonic sections are matched vertex for
    vertex against povm_properties' atlas and the two Archimedean ones against
    their dual pairs rectified (ATLAS, RECTIFIED).

    Returns the worst match and the tightest runner-up over every comparison.
    """
    worst, runner = 0.0, np.inf
    for name, pts in store.items():
        nv, degrees = SIGNATURE[name]
        got = defaultdict(int)
        for f in faces[name]:
            got[len(f)] += 1
        assert len(pts) == nv, f"{name}: {len(pts)} vertices, not {nv}"
        assert dict(got) == degrees, f"{name}: faces {dict(got)}, not {degrees}"

        if name in ATLAS:
            targets = [atlas_solid(ATLAS[name])]
        else:
            targets = [rectify(atlas_solid(p)) for p in RECTIFIED[name]]
        for target in targets:
            w, r = match(pts, target)
            worst, runner = max(worst, w), min(runner, r)
    return worst, runner


def check_rungs(S, store):
    """Every rung draws the solid its own section actually is.

    COLUMNS types one macro name per rung.  Four of the nine non-pole rungs
    name a macro registered from a DIFFERENT rung, and those four are the
    identifications the figure is FOR: 2O's w = 1/sqrt2 draws the octahedron 2T
    carries at its equator, 2O's w = 1/2 draws 2T's cube, 2I's w = sigma/2
    draws the icosahedron registered at w = tau/2, and 2O's equator draws the
    pair, whose octahedron half is again 2T's.  Each of the four is a typed
    name until it is measured here.

    The other five rungs draw the macro registered from that very rung, as does
    the pair's cuboctahedron half (solids() cuts it out of 2O's own equator):
    those comparisons are a set against itself, and establish nothing beyond
    the vertices being distinct.  What the cuboctahedron IS is settled by
    check_solids, not here.
    """
    byw = {g: {round(s["w"], 6): s for s in S[g]} for g in S}
    worst, runner = 0.0, np.inf
    for _, _, group, rungs in COLUMNS:
        for w, solid, _ in rungs:
            if solid is None:                     # a pole: no solid to identify
                continue
            verts = byw[group][round(w, 6)]["verts"]
            drawn = (np.vstack([store[nm] for nm, _ in PAIR])
                     if solid == "PAIR" else store[solid])
            wo, ru = match(verts, drawn)
            worst, runner = max(worst, wo), min(runner, ru)
    return worst, runner


def solids(S):
    """The distinct section polyhedra, keyed by the name the figure uses."""
    store, faces = {}, {}

    def register(name, entry):
        store[name], faces[name] = entry["verts"], entry["faces"]

    byw = {g: {round(s["w"], 6): s for s in S[g]} for g in S}
    register("cube", byw["2T"][0.5])
    register("oct", byw["2T"][0.0])
    register("ico", byw["2I"][0.809017])
    register("dod", byw["2I"][0.5])
    register("icodod", byw["2I"][0.0])

    # 2O's equator is one section carrying two classes: the six coordinate axes
    # (an octahedron, already stored) and twelve more (a cuboctahedron).
    equator = byw["2O"][0.0]["verts"]
    is_axis = np.sort(np.abs(equator), axis=1)[:, 1] < 1e-9
    cubocta = equator[~is_axis]
    store["cubocta"] = cubocta
    faces["cubocta"] = merge_coplanar(cubocta, ConvexHull(cubocta))
    return store, faces


# ----------------------------------------------------------------- the ladder

# Per column: title, covering map, group, and the rungs at w >= 0 as
# (w, solid, name).  Counts are read off the data at emit time, never typed.
COLUMNS = [
    (r"\textit{24-cell}", r"$\TwoT \to \Tgroup$", "2T",
     [(1.0, None, None),
      (0.5, "cube", "cube"),
      (0.0, "oct", "octahedron")]),
    (r"\textit{24-cell} and dual", r"$\TwoO \to \Ogroup$", "2O",
     [(1.0, None, None),
      (2 ** -0.5, "oct", "octahedron"),
      (0.5, "cube", "cube"),
      (0.0, "PAIR", None)]),
    (r"\textit{600-cell}", r"$\TwoI \to \Igroup$", "2I",
     [(1.0, None, None),
      (0.809017, "ico", "icosahedron"),
      (0.5, "dod", "dodecahedron"),
      (0.309017, "ico", "icosahedron"),
      (0.0, "icodod", "icosidodecahedron")]),
]

TICKS = [(1.0, "$1$"), (0.809017, r"$\phig/2$"), (2 ** -0.5, r"$1/\sqrt{2}$"),
         (0.5, "$1/2$"), (0.309017, r"$\invphig/2$"), (0.0, "$0$")]

LABEL = r"anchor=west, font=\labfont, align=left, fill=white, inner sep=1.4pt"
TICK = r"font=\tickfont, inner sep=1.2pt"


# ------------------------------------------------------- what the type needs

def wid(text, pt, box=0.0):
    """Width of a set string in cm, plus the inner sep of the box around it."""
    assert text in MEASURED, f"{text!r} has no measured width -- see MEASURED"
    return MEASURED[text] * pt + box


def order_label(group):
    return rf"$\lvert 2\mathcal{{{group[1]}}} \rvert = {ORDER[group]}$"


def theta_label(w):
    """The turn angle the latitude w stands for."""
    return rf"${2 * np.degrees(np.arccos(np.clip(w, -1, 1))):.0f}^\circ$"


def tick_label(lab, sgn):
    return lab if sgn > 0 else lab.replace("$", "$-", 1)


def signs(w):
    """The signs a latitude is drawn at -- the equator is its own mirror."""
    return [1, -1] if w > 1e-9 else [1]


def pair_lines(store):
    """The paired rung's label, one entry per set line."""
    return [("" if k == 0 else "$+$ ") + f"{name} ({len(store[nm])})"
            for k, (nm, name) in enumerate(PAIR)]


def tightest_pair():
    """The closest two rungs that both draw a solid, in units of w.

    This is what caps the icons: neighbours a gap g apart leave room for
    2*ICON + CLEAR only if g*H covers it, so the vertical scale follows from
    the icon size rather than being tuned against it.  Read off COLUMNS rather
    than typed, so a rung added to any column re-derives the ladder instead of
    quietly overlapping its neighbour.
    """
    return min(a - b
               for _, _, _, rungs in COLUMNS
               for (a, sa, _), (b, sb, _) in zip(rungs, rungs[1:])
               if sa is not None and sb is not None)


H = (2 * ICON + CLEAR) / tightest_pair()   # height of one unit of w


def column_extent(col, sizes, store):
    """How far a column's ink reaches left and right of its own axis, in cm.

    Every string the column sets is asked its width here, and main() then sets
    exactly these strings -- the arithmetic that places the columns and the
    type that lands between them are one list read twice.
    """
    title, cover, group, rungs = col
    left = right = 0.0
    for text, pt in ((title, TITLE_PT), (cover, SMALL_PT),
                     (order_label(group), SMALL_PT)):
        half = wid(text, pt) / 2            # head and foot are centred on the axis
        left, right = max(left, half), max(right, half)
    for w, solid, name in rungs:
        if solid is None:                   # a pole: a dot and its q = +-1
            reach = max(wid(POLE[s], LABEL_PT, INNER) for s in POLE)
            right = max(right, LABEL_DX + reach)
        elif solid == "PAIR":               # two icons, and a two-line label
            reach = max(wid(t, LABEL_PT, INNER) for t in pair_lines(store))
            left = max(left, PAIR_DX + ICON)
            right = max(right, PAIR_DL + reach)
        else:
            n = sizes[group][round(w, 6)]
            left = max(left, ICON)
            right = max(right, LABEL_DX + wid(f"{name} ({n})", LABEL_PT, INNER))
    return left, right


def ladder(sizes, store):
    """Where the three columns and the w / theta scale go.

    Each gap is the left column's longest label, plus GUTTER, plus whatever the
    right column reaches back with -- so the gaps come out unequal, which is
    the point: 2T's labels are short, and 2O's paired rung sets the longest
    line in the figure.
    """
    ext = [column_extent(c, sizes, store) for c in COLUMNS]
    xs = [0.0]
    for (_, r), (l, _) in zip(ext, ext[1:]):
        xs.append(xs[-1] + r + GUTTER + l)
    theta = max(wid(theta_label(sgn * w), TICK_PT, TICKBOX)
                for w, _ in TICKS for sgn in signs(w))
    tick = max(wid(tick_label(lab, sgn), TICK_PT, TICKBOX)
               for w, lab in TICKS for sgn in signs(w))
    axisx = -(ICON + GUTTER_L + TICK_DX + theta)
    return xs, axisx, ext, axisx - TICK_DX - tick


PREAMBLE = r"""% GENERATED by code/_build_section_ladder.py -- do not hand-edit.
% The vertex coordinates are Bloch axes n = -(u_3, u_2, u_1) of the parallel
% sections of the three polytopes, read straight off code/data/group_*.npz --
% the same source as Table C.1.  The shading machinery below is lifted from
% platonic_solids.tex so these solids are lit exactly like its panels (that
% figure is repository-only, not printed); only the face edge weight is
% rewritten, for icons drawn this much smaller.
\documentclass[border=5pt]{standalone}
\usepackage{tikz}
\usepackage{tikz-3dplot}
\usepackage{amsmath,amssymb}

\newcommand{\Tgroup}{\mathcal{T}}
\newcommand{\Ogroup}{\mathcal{O}}
\newcommand{\Igroup}{\mathcal{I}}
\newcommand{\TwoT}{2\Tgroup}
\newcommand{\TwoO}{2\Ogroup}
\newcommand{\TwoI}{2\Igroup}
\newcommand{\phig}{\tau}
\newcommand{\invphig}{\sigma}

% --- Type -----------------------------------------------------------------
% Generated from the pt constants the builder placed the columns with, so the
% arithmetic and the type on the page are the same numbers.
\newcommand{\labfont}{\fontsize{@LABEL@}{@LABELSKIP@}\selectfont}
\newcommand{\tickfont}{\fontsize{@TICK@}{@TICKSKIP@}\selectfont}
\newcommand{\titlefont}{\fontsize{@TITLE@}{@TITLESKIP@}\selectfont}
\newcommand{\smallfont}{\fontsize{@SMALL@}{@SMALLSKIP@}\selectfont}
\newcommand{\facelw}{@FACELW@pt}

% \WCHECK{width}{text} -- the label must set no wider than the builder assumed
% when it spaced the columns.  Wider means the layout arithmetic has gone
% stale, which on the page means one column's label reaching into the next; it
% stops the build here instead, where the cause is legible.
\newcommand{\WCHECK}[2]{%
  \begingroup\settowidth{\dimen0}{#2}%
  \ifdim\dimen0>#1\relax
    \errmessage{section ladder: \detokenize{#2} sets wider than #1, the width
      _build_section_ladder.py placed the columns with -- re-measure MEASURED}%
  \fi\endgroup}

% --- View and lighting (matched to platonic_solids.tex) --------------------
\tdplotsetmaincoords{60}{120}
\pgfmathsetmacro{\viewX}{sin(60)*sin(120)}
\pgfmathsetmacro{\viewY}{-sin(60)*cos(120)}
\pgfmathsetmacro{\viewZ}{cos(60)}
\pgfmathsetmacro{\lightX}{1/sqrt(6)}
\pgfmathsetmacro{\lightY}{1/sqrt(6)}
\pgfmathsetmacro{\lightZ}{2/sqrt(6)}
"""


def preamble():
    """PREAMBLE with the type sizes the layout was computed from filled in."""
    out = PREAMBLE
    for key, pt in (("LABEL", LABEL_PT), ("TICK", TICK_PT),
                    ("TITLE", TITLE_PT), ("SMALL", SMALL_PT)):
        out = out.replace(f"@{key}@", f"{pt:g}")
        out = out.replace(f"@{key}SKIP@", f"{pt * 1.2:g}")
    return out.replace("@FACELW@", f"{FACE_LW:g}")


def machinery():
    """platonic_solids.tex's vertex storage, culling and shading, lifted.

    One substitution on the way in.  platonic_solids.tex (repository-only, not
    printed) gives a solid a panel to itself; this figure draws sixteen icons a
    fraction of that size, and a face edge weight that suits the one is heavy
    on the other -- so the weight becomes a parameter here.  The lighting,
    which is what the two figures genuinely share, is untouched, and the assert
    makes a reshaped face drawer upstream fail here rather than silently keep
    the old weight.
    """
    src = MACHINERY.read_text()
    lifted = src[src.index("% --- Vertex storage"):
                 src.index("% --- Vertex tables")].rstrip()
    found = lifted.count("line width=0.4pt")
    assert found == 3, f"lifted face drawers changed shape: {found} widths, not 3"
    return lifted.replace("line width=0.4pt", r"line width=\facelw")


def build():
    """The figure's source, and the size and width of every string in it.

    Writes nothing.  main() writes the source and measure() writes a document
    measuring exactly these strings, so the set the layout reserved for and the
    set the type is measured over cannot come apart.
    """
    check_mirrors()
    S = {g: sections(g) for g in ORDER}
    store, faces = solids(S)
    margins = {"solids named": check_solids(store, faces),
               "rungs drawn": check_rungs(S, store)}
    sizes = {g: {round(s["w"], 6): s["n"] for s in S[g]} for g in ORDER}
    xs, axisx, ext, xleft = ladder(sizes, store)

    body, checks = [], {}

    def sets(text, pt, font):
        """Record the size and width the layout allowed a string, then hand it back."""
        checks[(font, text)] = (pt, wid(text, pt))
        return text

    # the latitude scale: w on the left, the turn angle it stands for on the right
    body.append(r"  % --- the latitude scale ---------------------------------------------")
    body.append(f"  \\draw[gray!55, line width=0.55pt] ({axisx:.3f},{-H:.3f}) -- "
                f"({axisx:.3f},{H:.3f});")
    body.append(f"  \\node[anchor=south, {TICK}] at ({axisx:.3f},{H + TITLE_DY:.3f}) "
                f"{{{sets(r'$w$\quad$\theta$', TICK_PT, r'\tickfont')}}};")
    for w, lab in TICKS:
        for sgn in signs(w):
            y, wv = H * sgn * w, sgn * w
            wlab = sets(tick_label(lab, sgn), TICK_PT, r"\tickfont")
            tlab = sets(theta_label(wv), TICK_PT, r"\tickfont")
            body.append(f"  \\draw[gray!55, line width=0.5pt] "
                        f"({axisx - TICK_LEN:.3f},{y:.3f}) -- "
                        f"({axisx + TICK_LEN:.3f},{y:.3f});")
            body.append(f"  \\node[anchor=east, {TICK}] at "
                        f"({axisx - TICK_DX:.3f},{y:.3f}) {{{wlab}}};")
            body.append(f"  \\node[anchor=west, {TICK}, gray!80] at "
                        f"({axisx + TICK_DX:.3f},{y:.3f}) {{{tlab}}};")
    body.append("")

    body.append(r"  % --- the 24-cell's latitudes, common to all three -------------------")
    for w in GUIDES:
        for sgn in signs(w):
            y = H * sgn * w
            body.append(f"  \\draw[gray!45, line width=0.45pt, "
                        f"dash pattern=on 1.3pt off 2.3pt] "
                        f"({-(ICON + 0.35):.3f},{y:.3f}) -- "
                        f"({xs[-1] + LABEL_DX:.3f},{y:.3f});")
    body.append("")

    # A rung at a latitude no other column has is the hardest thing in the
    # figure to find: the eye leaves the scale at, say, sigma/2 and crosses the
    # width of two columns with nothing to follow.  These give it a rail, drawn
    # fainter and finer than the 24-cell's and stopping at the icon it leads to
    # -- so it reads as a pointer to one rung, not as a latitude recurring in
    # all three columns, which is what the full-width dotting means.
    body.append(r"  % --- leaders to the rungs only one column has -----------------------")
    shared = {round(w, 6) for _, _, _, rungs in COLUMNS for w, solid, _ in rungs
              if solid is not None} - {round(w, 6) for w in GUIDES}
    for ci, (_, _, group, rungs) in enumerate(COLUMNS):
        for w, solid, _ in rungs:
            if solid is None or round(w, 6) not in shared:
                continue
            # nothing may sit at this latitude in another column, or the leader
            # would run through a solid rather than up to one
            elsewhere = [g for _, _, g, rs in COLUMNS if g != group
                         for x, sol, _ in rs if sol is not None and abs(x - w) < 1e-6]
            assert not elsewhere, f"{group}'s w = {w} is also a rung in {elsewhere}"
            for sgn in signs(w):
                y = H * sgn * w
                body.append(f"  \\draw[gray!32, line width=0.35pt, "
                            f"dash pattern=on 0.5pt off 2.5pt] "
                            f"({-(ICON + 0.35):.3f},{y:.3f}) -- "
                            f"({xs[ci] - ICON:.3f},{y:.3f});")
    body.append("")

    for ci, (title, cover, group, rungs) in enumerate(COLUMNS):
        cx, total = xs[ci], 0
        body.append(f"  % --- {group}: {title} " + "-" * 28)
        body.append(f"  \\node[anchor=south, align=center, font=\\titlefont] at "
                    f"({cx:.3f},{H + TITLE_DY:.3f}) "
                    f"{{{sets(title, TITLE_PT, r'\titlefont')}}};")
        body.append(f"  \\node[anchor=south, align=center, font=\\smallfont] at "
                    f"({cx:.3f},{H + COVER_DY:.3f}) "
                    f"{{{sets(cover, SMALL_PT, r'\smallfont')}}};")
        body.append(f"  \\draw[gray!45, line width=0.55pt] ({cx:.3f},{-H:.3f}) -- "
                    f"({cx:.3f},{H:.3f});")

        for w, solid, name in rungs:
            n = sizes[group][round(w, 6)]
            for sgn in signs(w):
                y = H * sgn * w
                total += n
                if solid is None:  # a pole
                    body.append(f"  \\fill[black!70] ({cx:.3f},{y:.3f}) circle ({DOT});")
                    body.append(f"  \\node[{LABEL}] at ({cx + LABEL_DX:.3f},{y:.3f}) "
                                f"{{{sets(POLE[sgn], LABEL_PT, r'\labfont')}}};")
                elif solid == "PAIR":  # one section, two conjugacy classes
                    for k, (nm, _) in enumerate(PAIR):
                        body.append(f"  \\begin{{scope}}[shift="
                                    f"{{({cx + (2 * k - 1) * PAIR_DX:.3f},{y:.3f})}}, "
                                    f"scale={ICON}, tdplot_main_coords]")
                        body.append(f"    \\solid{nm}")
                        body.append(r"  \end{scope}")
                    # the two classes must account for the section between them
                    assert sum(len(store[nm]) for nm, _ in PAIR) == n, \
                        f"{group}: pair rung splits {n} as {[len(store[nm]) for nm, _ in PAIR]}"
                    lines = [sets(t, LABEL_PT, r"\labfont") for t in pair_lines(store)]
                    body.append(f"  \\node[{LABEL}] at ({cx + PAIR_DL:.3f},{y:.3f}) "
                                f"{{{r' \\ '.join(lines)}}};")
                else:
                    body.append(f"  \\begin{{scope}}[shift={{({cx:.3f},{y:.3f})}}, "
                                f"scale={ICON}, tdplot_main_coords]")
                    body.append(f"    \\solid{solid}")
                    body.append(r"  \end{scope}")
                    body.append(f"  \\node[{LABEL}] at ({cx + LABEL_DX:.3f},{y:.3f}) "
                                f"{{{sets(f'{name} ({n})', LABEL_PT, r'\labfont')}}};")

        # the rungs must account for every element of the group
        assert total == ORDER[group], f"{group}: rungs sum to {total}, not {ORDER[group]}"
        body.append(f"  \\node[anchor=north, font=\\smallfont] at "
                    f"({cx:.3f},{-H - ORDER_DY:.3f}) "
                    f"{{{sets(order_label(group), SMALL_PT, r'\smallfont')}}};")
        body.append("")

    L = [preamble(), machinery(), ""]

    L.append(r"% --- Section vertex data --------------------------------------------")
    for name, pts in store.items():
        L.append(f"% {name}: {len(pts)} vertices, {len(faces[name])} faces")
        for k, p in enumerate(pts, start=1):
            L.append(f"\\PVert{{{name}}}{{{k}}}" + "".join(f"{{{v:.6f}}}" for v in p))
        L.append("")

    L.append(r"% --- One macro per section solid ------------------------------------")
    for name in store:
        L.append(f"\\newcommand{{\\solid{name}}}{{%")
        for f in faces[name]:
            L.append(f"  {DRAW[len(f)]}{{{name}}}" + "".join(f"{{{j + 1}}}" for j in f) + "%")
        L.append("}")
        L.append("")

    L.append(r"\begin{document}")
    L.append(r"% --- the widths the columns were spaced with, checked against the type")
    for (font, text), (_, cm) in checks.items():
        L.append(f"\\WCHECK{{{cm + SLACK:.4f}cm}}{{{font} {text}}}%")
    L.append(r"\begin{tikzpicture}[x=1cm, y=1cm]")
    L.append("")
    L.extend(body)
    L.append(r"\end{tikzpicture}")
    L.append(r"\end{document}")

    report = [f"  {g}: {[(round(x['w'], 3), x['n']) for x in S[g]]}  -> {ORDER[g]}"
              for g in ORDER]
    report += [f"  {what}: worst match {worst:.1e}, closest runner-up {runner:.4f}"
               for what, (worst, runner) in margins.items()]
    report.append(f"  layout: H = {H:.3f}, scale at {axisx:.3f}, columns at "
                  + ", ".join(f"{x:.3f}" for x in xs))
    report.append(f"  ladder reaches {xs[-1] + ext[-1][1] - xleft:.2f} cm across and "
                  f"{2 * H:.2f} cm pole to pole; {len(checks)} labels width-checked")
    return "\n".join(L) + "\n", checks, report


def measure(checks):
    """A standalone whose log carries the true width of every string set.

    MEASURED is size-specific on purpose (see its comment), so it has to be
    re-measured whenever a type size moves -- and a recipe left in a comment is
    a recipe that rots.  This writes the document instead, over exactly the
    strings build() just set, under the very font macros the figure uses::

        uv run _build_section_ladder.py --measure > /tmp/ladder.tex
        cd /tmp && pdflatex ladder.tex

    The log then carries one ``ROW <i> <width>pt`` per string, with <i> indexing
    the header this writes; MEASURED wants ``width_pt * 0.0351459804 / size_pt``.
    """
    rows = list(checks.items())
    L = [preamble().rstrip(),
         r"\newcommand{\M}[2]{\settowidth{\dimen0}{#2}\typeout{ROW #1 \the\dimen0}}",
         "",
         "% index   size  string"]
    L += [f"% {i:>5}  {pt:>5g}  {text}" for i, ((_, text), (pt, _)) in enumerate(rows)]
    L.append(r"\begin{document}")
    L += [f"\\M{{{i}}}{{{font} {text}}}%" for i, ((font, text), _) in enumerate(rows)]
    L += ["x", r"\end{document}"]
    return "\n".join(L) + "\n"


def main(argv=()):
    text, checks, report = build()
    if "--measure" in argv:
        print(measure(checks), end="")   # writes nothing; see measure()
        return
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT.parent)}")
    for line in report:
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
