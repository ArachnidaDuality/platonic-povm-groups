"""Negative controls for ``_build_povm_cards.py``: one per new assertion.

Each control corrupts exactly one literal -- in a sandbox copy of ``paper/``,
or by patching one function of the builder -- and requires that the builder
ABORTS and writes NOTHING.  An assertion nobody has watched fail is a comment.

Run: cd code && uv run python _povm_cards_controls.py
"""

import contextlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

import re

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "code"))

import _build_povm_cards as B          # noqa: E402

REAL_PAPER = B.PAPER
CONTROLS = []


class StaleControl(Exception):
    r"""The control's own literal has gone stale: it corrupts NOTHING.

    A control that corrupts nothing is not a control, and it must never be
    counted as a pass.  This exception is raised by every helper that looks
    for a literal, and the runner reports it as STALE wherever it surfaces --
    which matters, because a DEFERRED corruption (``patch_block``) does not
    run its search until the builder calls the patched emitter, i.e. from
    inside ``B.main()``, where an AssertionError is indistinguishable from
    the builder aborting on the corruption.
    """


# A deferred corruption must be seen to FIRE.  ``patch_block`` records that it
# was installed and, when the patched emitter actually runs on the named
# solid and the replacement lands, that it fired; the runner requires both.
DEFERRED = {"installed": [], "fired": []}


def control(name, assertion):
    def deco(fn):
        CONTROLS.append((name, assertion, fn))
        return fn
    return deco


def sandbox():
    """A writable copy of paper/, wired into the builder for one run."""
    d = Path(tempfile.mkdtemp(prefix="nc-"))
    shutil.copytree(REAL_PAPER, d / "paper")
    B.PAPER = d / "paper"
    B.FIGURES = B.PAPER / "figures"
    return d


def edit(rel, old, new):
    p = B.FIGURES / rel if rel.startswith("card") or rel.startswith("_") or \
        rel.startswith("circuit") else B.PAPER / rel
    t = p.read_text()
    if old not in t:
        raise StaleControl(f"the control's own literal is not in {rel}: "
                           f"{old!r}")
    p.write_text(t.replace(old, new, 1))


def sub(text, old, new):
    r"""``old`` -> ``new``, and FAIL if ``old`` is not there.

    A control whose own literal has gone stale stops controlling silently: it
    corrupts nothing, the builder writes five correct files, and the row reads
    "DID NOT ABORT" -- indistinguishable from a broken assertion.  One control
    did exactly that the moment assertion 26's type floor respelt
    ``\tfrac{1}{10}`` as ``(1/10)`` on the two A_5 cards.  So every
    substitution asserts its own literal, the way ``edit`` already does for
    files, and a stale control is a loud failure of the control suite rather
    than a quiet failure of the builder.
    """
    if old not in text:
        raise StaleControl(f"the control's own literal has gone stale: "
                           f"{old!r}")
    return text.replace(old, new)


def first_with(lines, lit):
    """``lines`` with ``lit`` corrupted in the FIRST line that carries it.

    The per-line stale guard is right for a one-line block and wrong for a
    tabular: sub() fires on the first line that does NOT match -- which is
    \begin{tabular} -- so the control aborted on itself and corrupted
    nothing.  This says what the caller meant: the literal is somewhere in the
    block (a stale literal is still a loud failure), and exactly one
    occurrence is corrupted, which is what a one-literal control is.
    """
    n = sum(lit in ln for ln in lines)
    if n < 1:
        raise StaleControl(f"the control's own literal is in no line: {lit!r}")
    out, done = [], False
    for ln in lines:
        if not done and lit in ln:
            out.append(ln.replace(lit, "\x00", 1)); done = True
        else:
            out.append(ln)
    return out


# Which emitter writes which \newcommand.  A macro whose emitter has been
# renamed or retired is a STALE control, not a passing one: without the guard
# below, `getattr` raises AttributeError, the runner counts it as "aborted",
# and a control that corrupts nothing reads as a pass.
BLOCK_EMITTER = {
    "CTitle": "heads", "CPtrStrip": "heads", "CHeadA": "heads",
    "CPtrA": "heads", "CHeadB": "heads", "CPtrB": "heads",
    "CHeadC": "heads", "CPtrC": "heads", "CHeadD": "heads", "CPtrD": "heads",
    "CStrip": "strip",
    "CardVertexTable": "vertex_blocks", "CardVertexBlockA": "vertex_blocks",
    "CardVertexBlockB": "vertex_blocks",
    "CIdent": "identities", "CIdentB": "identities",
    "CParams": "params", "CardSide": "card_side",
    "CardKey": "key_rectangle", "CKappa": "kappa_label",
    "CRays": "rays", "CCoin": "protocol",
}


def emitter(name):
    """``B.<name>``, or a loud STALE failure if the emitter has gone."""
    if not hasattr(B, name):
        raise StaleControl(f"the control's own literal has gone stale: "
                           f"{name} is not an emitter")
    return getattr(B, name)


def patch_block(macro, solid, old, new):
    """Corrupt one literal in ONE emitted block, after it is built.

    This is the shape that matters: the emitters' output is what the reader
    reads, and an assertion that re-derives the same value cannot see a
    corruption applied here.  Every assertion-29 control goes through it.
    """
    if macro not in BLOCK_EMITTER:
        raise StaleControl(f"the control's own literal has gone stale: "
                           f"no emitter writes {macro}")
    orig = emitter(BLOCK_EMITTER[macro])
    DEFERRED["installed"].append((macro, solid))

    def bad(*a, **k):
        f = a[0]
        out = orig(*a, **k)
        if f["solid"] != solid:
            return out
        got = first_with(out, old)          # raises StaleControl if absent
        DEFERRED["fired"].append((macro, solid))
        return [ln.replace("\x00", new) for ln in got]
    setattr(B, BLOCK_EMITTER[macro], bad)


# --- the controls ----------------------------------------------------------

@control("corrupt a kappa digit in randomized_labels.tex's parse", 18)
def _c1():
    orig = B.read_labels_table

    def bad():
        d = orig()
        top, bot, kap, rider, prem = d["octahedron"]
        d["octahedron"] = (top, bot, "+0.321976", rider, prem)
        return d
    B.read_labels_table = bad


@control("swap two key cells in Table D.4's parse", 15)
def _c2():
    orig = B.read_labels_table

    def bad():
        d = orig()
        top, bot, kap, rider, prem = d["octahedron"]
        bot = list(bot)
        bot[0], bot[1] = bot[1], bot[0]
        d["octahedron"] = (top, bot, kap, rider, prem)
        return d
    B.read_labels_table = bad


@control("drop the np.conj in the key rebuild", 15)
def _c3():
    orig = B.decker_vertices

    def bad(solid):
        d, live, W = orig(solid)
        # bloch_of_ket(W[k]) rather than bloch_of_ket(conj(W[k])): y flips
        d = np.array([[x[0], -x[1], x[2]] for x in d])
        return d, live, W
    B.decker_vertices = bad


@control("delete a dead outcome from the rule's prediction", 16)
def _c4():
    orig = B.read_fourier_block

    def bad(src):
        m, ell = orig(src)
        return (m + 1 if ell > 1 and m + 1 <= 2 ** ell else m), ell
    B.read_fourier_block = bad


@control("print g where the direction is g^{-1}", 11)
def _c5():
    orig = emitter("params")

    def bad(f):
        return [sub(x, r"relabels outcomes by $g^{-1}$",
                    r"relabels outcomes by $g$") for x in orig(f)]
    B.params = bad


@control("swap a coin word for an equal-cost mate", 19)
def _c6():
    orig = B.coset_representatives

    def bad(s, R, circuits):
        reps = list(orig(s, R, circuits))
        if len(reps) >= 2:
            reps[0], reps[1] = reps[1], reps[0]
        return reps
    B.coset_representatives = bad


@control("reverse one ray's index order (|1> vertex printed first)", 19)
def _c7():
    orig = B.facts

    def bad(solid, **kw):
        f = orig(solid, **kw)
        if f["rays"]:
            r = f["rays"][0]
            r["zero"], r["one"] = r["one"], r["zero"]
        return f
    B.facts = bad


@control("apply the alignment a second time to the coin's snapshots", 19)
def _c8():
    orig = B.alignment

    def bad(s):
        A, v = orig(s)
        return A, A @ v            # v0 pushed through A once more
    B.alignment = bad


@control("move \\gate{U_R} inside the solid \\gategroup", 14)
def _c9():
    sandbox()
    edit("card_dec_oct.tex",
         r"\gate{U_R} & \ctrl{-2}", r"\qw & \ctrl{-2}")
    edit("card_dec_oct.tex", r"\gategroup{1}{4}{3}{6}", r"\gategroup{1}{3}{3}{6}")


@control("shift a \\gategroup column index by 2 instead of 1", 14)
def _c10():
    sandbox()
    edit("card_dec_oct.tex", r"\gategroup{1}{4}{3}{6}", r"\gategroup{1}{5}{3}{7}")


@control("flip one angle's sign in the drawn U_R", 17)
def _c11():
    sandbox()
    edit("circuit_dec_oct.tex",
         r"R_{\hat y}(-\arccos(1/\sqrt3))", r"R_{\hat y}(\arccos(1/\sqrt3))")


@control("change a \\lstick count on the card circuit", 10)
def _c12():
    sandbox()
    edit("card_dec_oct.tex", r"\lstick{\rho}", r"\lstick{\sigma}")


@control("chip all twenty on the dodecahedron", 21)
def _c13():
    sandbox()
    src = (B.FIGURES / "card_oct_sphere.tex").read_text()
    src = sub(src, "{oct}", "{dod}")
    for k in range(7, 21):
        src = sub(src, r"  \chipnode{6}{($(v6)!-9pt!(O)$)}{6}",
                          r"  \chipnode{6}{($(v6)!-9pt!(O)$)}{6}" "\n"
                          r"  \chipnode{%d}{($(v%d)!-9pt!(O)$)}{%d}" % (k, k, k))
    (B.FIGURES / "card_dodec_sphere.tex").write_text(src)


@control("delete alpha's closed form from the side stack", 23)
def _c14():
    # alpha and beta are printed in the short-atom stack beside the circuit,
    # not in the parameter paragraph; the caption freeze follows them.
    patch_block("CardSide", "octahedron",
                r"\CTR{\hsize}{$\alpha = \sqrt{(3+\sqrt3)/18}$}\NIS{4pt}", "")


@control("drop the dilation clause from the strip's exactness footer", 12)
def _c15():
    # The thesis's exactness conjunction is printed once per card, as the
    # strip's spanned footer.  The clause that must not be compressed away is
    # "this dilation is not" -- it is what binds Theorem 1 to the circuit
    # printed 10cm below it; "exact, this dilation included" loses its host
    # and can be read as including the dilation among the EXACT things.
    patch_block("CStrip", "octahedron",
                "only the octahedral coin is exact --- this dilation is not",
                "only the octahedral coin is exact")


@control("reword head 3's register convention", 12)
def _c16():
    # The band heads are emitted, so check 12 reads the emitted surface.
    # "top wire first" is the half of the convention the key rectangle's own
    # heads cannot draw.
    patch_block("CHeadC", "octahedron",
                "Register $\\to$ vertex, top wire first",
                "Register $\\to$ vertex")


@control("put a \\tfrac into the one fragment that still sets footnotesize", 26)
def _c17():
    # Every fragment a reader retypes is set at \small, so the type floor
    # binds on the header strip and the pointer column alone -- and a \tfrac
    # there prints its numerals at 5.98pt.
    patch_block("CStrip", "octahedron", "no thesis gate set",
                r"no $\tfrac13$ gate set")


@control("re-type a \\PVert coordinate into the sphere stub", 22)
def _c18():
    sandbox()
    edit("card_oct_sphere.tex", r"\begin{document}",
         "\\PVert{oct}{1}{1}{0}{0}\n\\begin{document}")


@control("corrupt one printed effect atom", 2)
def _c19():
    orig = B.factored_effect

    def bad(solid, E):
        m = orig(solid, E)
        if solid == "octahedron" and m[0][0] == "1":
            m[0][0] = "2"
        return m
    B.factored_effect = bad


@control("corrupt one printed vertex tuple against Table E.1", 6)
def _c20():
    orig = B.factored_tuple

    def bad(solid, n):
        out = orig(solid, n)
        if solid == "octahedron" and out[0] == ("+", "1"):
            out[0] = ("-", "1")
        return out
    B.factored_tuple = bad



# --- the hand-typed LaTeX, the chapter head, and the panel's geometry ------


@control("corrupt EFFECT_SCALE's printed \\tfrac against its numeric half", 27)
def _c21():
    B.EFFECT_SCALE = dict(B.EFFECT_SCALE)
    B.EFFECT_SCALE["octahedron"] = (1 / 6, r"\tfrac15")


@control("corrupt a printed CNOT ket against _cnot's permutation", 27)
def _c22():
    B.PARAM_LINES = dict(B.PARAM_LINES)
    B.PARAM_LINES["octahedron"] = [sub(x, r"\ket{101}\mapsto\ket{001}",
                                       r"\ket{110}\mapsto\ket{001}")
                                   if r"\ket{101}" in x else x
                                   for x in B.PARAM_LINES["octahedron"]]


@control("corrupt the printed alpha/beta denominator", 27)
def _c23():
    B.SIDE_ATOMS = dict(B.SIDE_ATOMS)
    B.SIDE_ATOMS["octahedron"] = [sub(x, r"\sqrt{(3+\sqrt3)/18}",
                                      r"\sqrt{(3+\sqrt3)/12}")
                                  if r"(3+\sqrt3)/18" in x else x
                                  for x in B.SIDE_ATOMS["octahedron"]]


@control("corrupt a printed u_pm radical so B stops being unitary", 27)
def _c24():
    B.PARAM_LINES = dict(B.PARAM_LINES)
    B.PARAM_LINES["icosahedron"] = [
        sub(x, r"$u_\pm = \sqrt{1/2 \pm \sqrt{(2+\phig)/20}}$",
            r"$u_\pm = \sqrt{1/2 \pm \sqrt{(2+\phig)/10}}$")
        if r"u_\pm" in x else x for x in B.PARAM_LINES["icosahedron"]]


@control("leave the chapter head saying the U_g slot is omitted", 28)
def _c25():
    # All five card circuits draw the slot, so the head says "is drawn dashed
    # on each"; this control reverts it to the sentence the cards make false.
    sandbox()
    edit("bsc-thesis.tex", "is drawn dashed on each", "is omitted here")


@control("leave the chapter head on the transitional wording", 28)
def _c25b():
    # ... and the near miss: "where a card carries it" is true while some
    # circuits omit the slot and merely weak once all five draw it.
    # Assertion 28 counts the files, so it separates the two.
    sandbox()
    edit("bsc-thesis.tex", "is drawn dashed on each",
         "is drawn dashed where a card carries it")


@control("print a <0|E_k|0> cell to three places instead of four", 3)
def _c28():
    orig = emitter("vertex_table")

    def bad(f, ks, we):
        if f["solid"] != "octahedron":    # 1/6 to four places; the other two
            return orig(f, ks, we)        # V<=8 solids print other numbers
        return [ln.replace("\x00", "$0.167$")
                for ln in first_with(orig(f, ks, we), "$0.1667$")]
    B.vertex_table = bad


@control("rotate the chip model's screen basis off the panel's camera", 21)
def _c29():
    # the plausible-looking wrong basis: tdplot's, pre-composed with a quarter
    # turn about z-hat.  It agrees with the compiled panel on the octahedron
    # and the cube -- both invariant under that turn -- and is 115pt out on the
    # dodecahedron, which is what let it survive a first build looking checked.
    src = B.chip_geometry.__doc__
    body = ("    ex = np.array([-0.866, -0.5, 0.0])\n"
            "    ey = np.array([0.25, -0.433, 0.866])\n")
    text = "".join(open(B.__file__).read().split("def chip_geometry(f):")[1]
                   .split("\n", 1)[1])
    text = text[:text.index("\n\n\n")]
    text = sub(text, "    ex = np.array([-0.5, 0.8660254, 0.0])"
                        "            # (cos 120, sin 120, 0)\n"
                        "    ey = np.array([-0.4330127, -0.25, 0.8660254])\n",
                        body)
    ns = dict(B.__dict__)
    exec("def chip_geometry(f):\n" + text, ns)
    B.chip_geometry = ns["chip_geometry"]


@control("tilt the coin's alignment seed by five degrees", 19)
def _c30():
    # The coin's ray data is over-determined: nearest() insists on an EXACT
    # vertex (< 1e-9, runner-up > 0.1), so a tilt this large is caught by the
    # seed's own margin guard before the mixture sees it.  That is worth
    # knowing about the mixture check: it CONFIRMS the card's headline claim
    # rather than being the only thing standing behind it.
    orig = B.alignment
    c, s_ = np.cos(np.radians(5)), np.sin(np.radians(5))
    T = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

    def bad(v):
        A, v0 = orig(v)
        return A, T @ v0
    B.alignment = bad


@control("respell a u_pm radical to a DIFFERENT number", 23)
def _c31():
    # The corruption a string comparison cannot see.  Assertion 23 has to
    # admit respellings -- the type floor forces (1/10) in place of
    # \tfrac{1}{10} on the two A_5 cards -- and a respelling is exactly where
    # a wrong constant hides: this one is well-formed, sets beautifully, and
    # is off by a factor of ten.  Assertion 23b evaluates both closed forms
    # and compares NUMBERS, so it fires.
    orig = emitter("params")

    def bad(f):
        if f["solid"] != "icosahedron":
            return orig(f)
        return [sub(x, r"$u_\pm = \sqrt{1/2 \pm \sqrt{(2+\phig)/20}}$",
                    r"$u_\pm = \sqrt{1/2 \pm \sqrt{(2+\phig)/2}}$")
                for x in orig(f)]
    B.params = bad


@control("delete the chapter head's A-dagger factorization display", 23)
def _c32():
    # The A_5 pair leaves the shared four-stage factorization to the chapter
    # head.  That is a redirection, so assertion 23c checks the
    # target: delete the head's display and the two cards are pointing at
    # nothing.
    sandbox()
    edit("bsc-thesis.tex",
         "A^\\dagger \\;=\\; \\bigl(I_2 \\oplus (-\\sigmaz)\\bigr)\\,"
         "(I_2 \\otimes B)\\,\\mathsf{R}\\,(I_2 \\otimes C), \\qquad "
         "\\mathrm{diag}(-1, 1) = -\\sigmaz,",
         r"A^\dagger \;=\; \text{as published},")


@control("cut the A_5 card's pointer back to the chapter head", 23)
def _c33():
    # ... and the other half of the same redirection: the head may print it,
    # but a card that never points at it by \eqref has sent the reader
    # nowhere.  Both halves are required, so both are controlled.
    patch_block("CParams", "icosahedron",
                r"$A^\dagger$: Equation~\eqref{eq:adagger}",
                r"$A^\dagger$: four stages")


@control("draw the two A_5 card circuits at different wire spacings", 14)
def _c34():
    # Both A_5 circuits take the same wire-spacing cut, so the pair prints at
    # one spacing on facing pages.  The probe reads the drawing, not the
    # file's own header -- which writes the cut as "@R=1.5em -> 1.2em" and
    # would report the spacing the card was cut FROM.
    sandbox()
    edit("card_dec_icos.tex", r"\Qcircuit @C=1.4em @R=1.2em {",
         r"\Qcircuit @C=1.4em @R=1.5em {")


# --- the printed surface, the coin's order, and the routes -----------------

@control("state the coin's order the other way round (alignment first)", "19b")
def _c35():
    # The order is not decoration: the drawn word FIRST and the fixed
    # alignment after it is what measures R_g^T v0.  The reverse measures
    # A^T R_g^T zhat, which leaves the vertex set on four of ten dodecahedral
    # rays and three of four cubic ones.
    patch_block("CCoin", "cube",
                "apply it, then the alignment", "apply it after the alignment")


@control("derive the coin's rays with the alignment applied first", "19")
def _c36():
    # ... and the same error made in the DERIVATION rather than in the prose:
    # rays built from A^T R_g^T zhat instead of R_g^T v0.
    import numpy as _np
    orig = B.nearest

    def bad(v, w, what):
        if "coin |0> vertex" in what:
            A, _ = B.alignment(v)
            w = A.T @ (A @ w)              # a no-op if A is the identity ...
            w = A.T @ w                    # ... and the wrong order if not
        return orig(v, w, what)
    B.nearest = bad


@control("drop the A_5 card's route to Appendix A's table", "19c")
def _c37():
    # The route is head 4's third pointer, not a sentence.
    patch_block("CPtrD", "dodecahedron",
                r" $\cdot$ Table~\ref{tab:binary-icosahedral-group-2i}", "")


@control("print the atlas route twice on one card", "19c")
def _c38():
    patch_block("CPtrC", "cube",
                r"Table~\ref{tab:decker-labels}",
                r"Table~\ref{tab:binary-octahedral-group-2o}")


@control("drop the operator-order rider from a card with composite words",
         "19c")
def _c39():
    patch_block("CHeadD", "icosahedron",
                r"$\ket0$-vertex first, rightmost factor first",
                r"$\ket0$-vertex first")


@control("print the operator-order rider where every word is a single letter",
         "19c")
def _c39b():
    patch_block("CHeadD", "cube", r"$\ket0$-vertex first",
                r"$\ket0$-vertex first, rightmost factor first")


@control("drop the chip rule from a panel that chips half its vertices", 21)
def _c42():
    # The panel draws its axes as Figure 1.1's does and neither page names
    # them, so the control follows the one panel claim printed: on the two A_5
    # cards a chip marks the NEAR vertex of each axis, and nothing else on the
    # page prepares a reader for a sphere with half its vertices unnumbered.
    patch_block("CIdent", "dodecahedron",
                "; numbered at each axis's near vertex.", ".")


@control("swap two printed vertex tuples", 29)
def _c43():
    orig = emitter("vertex_blocks")

    def bad(f):
        out = orig(f)
        if f["solid"] != "cube":
            return out
        a = r"$(-1,\,-1,\,-1)$"
        b = r"$(+1,\,-1,\,-1)$"
        assert sum(a in ln for ln in out) == 1 and \
            sum(b in ln for ln in out) == 1, "the control's own literal"
        return [ln.replace(a, "@@").replace(b, a).replace("@@", b)
                for ln in out]
    B.vertex_blocks = bad


@control("corrupt one atom of a printed effect matrix", 29)
def _c44():
    patch_block("CardVertexTable", "octahedron",
                r"$\pmat{1 & -i \\ i & 1}$", r"$\pmat{1 & i \\ i & 1}$")


@control("swap two cells of the key rectangle", 29)
def _c46():
    patch_block("CardKey", "octahedron", "& 5 & 3 &", "& 3 & 5 &")


@control("mislabel a key column head", 29)
def _c47():
    patch_block("CardKey", "cube", r"$\mathtt{01}$ & $\mathtt{10}$",
                r"$\mathtt{10}$ & $\mathtt{01}$")


@control("swap the key's `upper` and `lower` corner heads", 29)
def _c47b():
    # The register convention is printed ON the object: the stub says which
    # bits the rows are and the spanned head which the columns are.  Swap
    # them and the rectangle reads bottom-wire-first.
    patch_block("CardKey", "dodecahedron",
                r"{\footnotesize upper} &", r"{\footnotesize lower} &")


@control("reverse a printed ray's |0>/|1> order", 29)
def _c48():
    patch_block("CRays", "octahedron", r"~$1$,\,$2$", r"~$2$,\,$1$")


@control("print the wrong design strength", 29)
def _c49():
    patch_block("CIdent", "icosahedron", r"a $5$-design", r"a $4$-design")


@control("print the wrong Tr E_k", 29)
def _c50():
    patch_block("CIdent", "octahedron", r"$\Tr E_k = \tfrac13$",
                r"$\Tr E_k = \tfrac16$")


@control("print the wrong V in the header strip", 29)
def _c51():
    # V is printed in Figure 1.1's own row, not in a sentence of the card's.
    patch_block("CStrip", "octahedron", "6 & $\\TwoO$ & 2 & 2 &",
                "7 & $\\TwoO$ & 2 & 2 &")


@control("print the wrong ancilla count in the header strip", 29)
def _c52():
    patch_block("CStrip", "cube", "8 & $\\TwoO$ & 2 & 0 &",
                "8 & $\\TwoO$ & 3 & 0 &")


@control("print a dead register value as live in the key", 29)
def _c53():
    # The dead RULE is drawn rather than stated: the dead outcomes are the
    # rectangle's empty cells, and 29 counts them against the dead set.
    patch_block("CardKey", "octahedron", "& 1 & --- \\\\", "& 1 & 7 \\\\")


@control("print the wrong alignment vertex", 29)
def _c54():
    patch_block("CCoin", "dodecahedron", r"($9 \mapsto \hat z$",
                r"($11 \mapsto \hat z$")


@control("flip a sign in the printed U_R", 29)
def _c55():
    # The card prints the closed form only, so the control flips the sign of
    # the angle -- the single most plausible silent error on a card, and 29
    # parses the PRINTED U_R back out as a rotation.
    patch_block("CParams", "icosahedron",
                r"$U_R = R_{\hat y}(\arccos(1/(\phig\sqrt3)))$",
                r"$U_R = R_{\hat y}(-\arccos(1/(\phig\sqrt3)))$")


@control("drop 'rightmost factor first' from the one product U_R", 29)
def _c56():
    patch_block("CParams", "octahedron", ", rightmost factor first,", ",")


@control("point kappa's band at Table D.3 instead of its definition", 29)
def _c57():
    patch_block("CPtrC", "cube", r"Table~\ref{tab:decker-labels}",
                r"Table~\ref{tab:decker-pose}")


# --- the strip, the shared literals, the re-homed A, E_1, the pointers -----

@control("corrupt one of Figure 1.1's column heads in the printed strip", 30)
def _c60():
    # The card's strip IS Figure 1.1's foot strip: the reader recognises the
    # row instead of reading it.  That only works if it really is the same
    # string, so it is compared cell by cell.
    patch_block("CStrip", "octahedron", "reorientation exact?",
                "reorientation exact")


@control("corrupt one of Figure 1.1's own cells in the printed strip", 30)
def _c61():
    patch_block("CStrip", "icosahedron", r"6 axes, no $\Phi$",
                r"6 axes, one $\Phi$")


@control("print the exactness footer differently on one card", 31)
def _c62():
    patch_block("CStrip", "cube",
                "only the octahedral coin is exact",
                "only the cubic coin is exact")


@control("give one card a different band head", 31)
def _c63():
    patch_block("CHeadB", "dodecahedron",
                "the dashed $U_g$ is an optional prepended rotation.",
                "the dashed prefix is an optional prepended rotation.")


@control("delete the chapter head's unfactored A", 32)
def _c64():
    # The two A_5 cards are laid out on the assumption that A is not on them
    # (measured: 43.81pt each).  Delete the head's display and they must
    # refuse to be written rather than point at nothing.
    sandbox()
    edit("bsc-thesis.tex",
         r"\begin{pmatrix}\alpha & \beta & \gamma & \delta",
         r"\begin{pmatrix}\alpha & \beta & \gamma & \delta_{\text{?}}")


@control("corrupt one atom of the worked E_1", 33)
def _c65():
    B.WORKED_E1 = dict(B.WORKED_E1)
    B.WORKED_E1["dodecahedron"] = sub(B.WORKED_E1["dodecahedron"],
        r"$E_1 = \tfrac1{20}\pmat{1+1/\sqrt3", r"$E_1 = \tfrac1{20}\pmat{1-1/\sqrt3")


@control("scale the worked E_1 by the wrong 1/V", 33)
def _c66():
    B.WORKED_E1 = dict(B.WORKED_E1)
    B.WORKED_E1["icosahedron"] = sub(B.WORKED_E1["icosahedron"],
        r"$E_1 = \tfrac1{12}", r"$E_1 = \tfrac1{10}")


@control("point the title line at a label that does not exist", 34)
def _c67():
    patch_block("CPtrStrip", "cube", r"Figure~\ref{fig:tetrahedron-recipe}",
                r"Figure~\ref{fig:tetrahedron-recipes}")


@control("drop the route back to Figure 1.1", 34)
def _c68():
    patch_block("CPtrStrip", "icosahedron",
                r"Figure~\ref{fig:tetrahedron-recipe}", "")


@control("corrupt a card title", 12)
def _c69():
    patch_block("CTitle", "cube", "The cubic POVM, end to end",
                "The cube POVM, end to end")


@control("change the caption's Decker figure number", 12)
def _c70():
    # The caption is the one cold name on the page and it lives in
    # bsc-thesis.tex; the generator owns the literal and checks it there.
    B.DECKER_FIG = dict(B.DECKER_FIG)
    B.DECKER_FIG["cube"] = 10


@control("leave D.2's opening claiming the captions record the relabelling",
         "28b")
def _c58():
    sandbox()
    edit("bsc-thesis.tex", "the outcome relabelling they record",
         "the outcome relabelling their captions record")


@control("set a numeral as a script in the one footnotesize fragment", "26b")
def _c59():
    # Two glyphs in the whole five-card set print below 7pt, both daggers, so
    # the rule is the strong one: NO numeral below 8.97pt anywhere.
    patch_block("CStrip", "cube", r"over Clifford${}+T$",
                r"over Clifford${}+T^{2}$")


@control("delete the readable companion of the strip's one sub-7pt glyph",
         "26b")
def _c59b():
    # The dagger of $T^\dagger$ is the only glyph on the five pages that
    # prints below 7pt, and what makes it safe is that Figure 1.1's own cell
    # prints the same T upright at 8.97pt two words later.
    patch_block("CStrip", "tetrahedron",
                r"$T^\dagger$, over Clifford${}+T$", r"$T^\dagger$")


# --- the body's closure over the data file, and the derived warnings -------

@control("delete an emitted block's call from the hand-written body", 35)
def _c71():
    # Nothing joined the two halves of a card until assertion 35: the data
    # file may be perfect and the body simply not call it, and every other
    # check reports pass while the printed page silently loses the strip.
    sandbox()
    edit("card_dodec_body.tex", r"\CStrip", r"\relax")


@control("point a body's \\input at another solid's data file", 35)
def _c73b():
    # The body's \input has two legal spellings -- figures/ where the data
    # file sits beside the body, ../code/data/ in this tree -- so the control
    # corrupts whichever the tree it runs in actually uses, never a hardcoded
    # one: a hardcoded spelling goes STALE in the other tree.
    sandbox()
    p = B.FIGURES / "card_dodec_body.tex"
    for stem in ("figures", "../code/data"):
        if r"\input{%s/card_dodec_data}" % stem in p.read_text():
            edit("card_dodec_body.tex",
                 r"\input{%s/card_dodec_data}" % stem,
                 r"\input{%s/card_icos_data}" % stem)
            return
    raise StaleControl("card_dodec_body.tex inputs no card_dodec_data")


@control("delete the kappa label's call from a body", 35)
def _c73():
    sandbox()
    edit("card_cube_body.tex", r"\CKappa", r"\relax")


@control("reprint a retired kappa rider", 36)
def _c74():
    # The two per-card kappa riders are retired (the builder's note above
    # kappa_num has the reasons: they described the UNcalibrated read beside
    # a calibrated "never a bias", and the sign one glossed the wrong
    # object).  The retirement is a ban, the way 19b keeps "after the
    # alignment" out, and this is its witness.
    patch_block("CKappa", "octahedron", "never a bias.",
                "never a bias. Negative: the estimate's sign flips.")


@control("call kappa `the calibration' on a card", 36)
def _c76():
    # "The calibration scalar" is eta (noiseless 1/3), the shrinkage BEFORE
    # the canonical dual's factor 3; kappa = 3 eta is the estimator channel's
    # multiplier.  A card calling kappa "the calibration" is read, by anyone
    # carrying the thesis's own dictionary, as a statement about eta.
    patch_block("CKappa", "cube", "never a bias.",
                "never a bias. The calibration is unchanged.")


@control("gloss the key's em-dashes on a card whose key has none", 36)
def _c77():
    patch_block("CHeadC", "cube", "top wire first",
                "top wire first; --- never fires")


@control("drop the em-dash gloss from a key that has dead cells", 36)
def _c78():
    patch_block("CHeadC", "octahedron", "; --- never fires", "")


@control("delete D.2's sentence licensing the free pairing", 36)
def _c79():
    # The kappa label prints the do-neither branch ("skip $U_R$ and read
    # Decker's own numbered list ... $\kappa = 1$, free").  That is D.2's
    # claim, not the card's; if D.2 stops making it the card must stop
    # printing it.
    sandbox()
    edit("bsc-thesis.tex",
         "or neither, and keep Decker's pose and vertex list",
         "or neither")


@control("price the mismatch and never print the free pairing", 29)
def _c80():
    # The failure this catches: a card that prices skipping U_R at up to
    # 10621.86x shots and never says that the published circuit read against
    # the published list is free leaves a card-only reader concluding that
    # U_R is compulsory.
    orig = emitter("kappa_label")

    def bad(f):
        (line,) = orig(f)
        m = re.match(r"(\\newcommand\{\\CKappa\}\{)(.*)\}$", line)
        assert m, "the kappa label no longer parses"
        body = sub(m.group(2), "Run $U_R$ and read the key, or "
                   "skip $U_R$ and read Decker's own numbered list~"
                   "\\cite{decker2004quantumcircuitssinglequbit}: either "
                   "way $\\kappa = 1$, free. ", "")
        return [m.group(1) + body + "}"]
    B.kappa_label = bad


@control("print the priced mismatch before the free pairing", 29)
def _c81():
    orig = emitter("kappa_label")

    def bad(f):
        (line,) = orig(f)
        m = re.match(r"(\\newcommand\{\\CKappa\}\{)(.*)\}$", line)
        assert m, "the kappa label no longer parses"
        free, priced = sub(m.group(2), "His circuit with our list",
                           "\x00His circuit with our list").split("\x00")
        return [m.group(1) + priced.rstrip() + " " + free.rstrip() + "}"]
    B.kappa_label = bad


@control("delete the chapter head's eq:adagger label", 32)
def _c82():
    # The two A_5 cards point at the four-stage display by \eqref.
    # Without the label the pointer is dead and the cards say nothing about
    # what A-dagger is.
    sandbox()
    edit("bsc-thesis.tex", r"\begin{equation} \label{eq:adagger}",
         r"\begin{equation}")


@control("delete the chapter head's eq:aunfactored label", 32)
def _c97():
    # The two A_5 cards point at the unfactored A by \eqref too.  Without the
    # label the pointer is dead.
    sandbox()
    edit("bsc-thesis.tex", r"\begin{equation} \label{eq:aunfactored}",
         r"\begin{equation}")


@control("break the gamma, delta pairing under the unfactored display", 32)
def _c98():
    # The paragraph pairs gamma_icos with delta_dodec by name; a reader at
    # the icosahedral card must find that pairing there, not infer it.
    sandbox()
    edit("bsc-thesis.tex", r"\gamma_\mathrm{icos}^2 = \delta_\mathrm{dodec}^2",
         r"\gamma_\mathrm{icos}^2 = \gamma_\mathrm{dodec}^2")


@control("cut the A_5 card's pointer to the unfactored display", 32)
def _c99():
    # The other half of the same redirection, as _c33 is for eq:adagger.
    patch_block("CParams", "dodecahedron",
                r"Equation~\eqref{eq:aunfactored}", r"the display above")


@control("corrupt one radicand of the unfactored display", 32)
def _c100():
    # Check 32's string half pins the display's shape -- label, rows, pointer,
    # the 1/sqrt2 in front -- and would pass this: alpha^2's radicand off by
    # ten.  Only the numeric half (32b), which rebuilds Decker's circuit from
    # the printed prefactor and radicands, can see it.
    sandbox()
    edit("bsc-thesis.tex",
         r"\alpha^2 = \tfrac{1}{2} + \tfrac{1}{30}\sqrt{75 + 30\sqrt{5}}",
         r"\alpha^2 = \tfrac{1}{2} + \tfrac{1}{30}\sqrt{75 + 20\sqrt{5}}")


@control("exchange the two gamma, delta pairing values", 32)
def _c101():
    # The pairing sentence with its two values swapped: every string check
    # passes (both pairings are still stated, by name) and A stays orthogonal
    # on both solids; only the rebuilt circuits disagree with decker_circuit.
    sandbox()
    edit("bsc-thesis.tex",
         r"\gamma_\mathrm{icos}^2 = \delta_\mathrm{dodec}^2 = \tfrac{1}{2} - ",
         r"\gamma_\mathrm{icos}^2 = \delta_\mathrm{dodec}^2 = \tfrac{1}{2} + ")
    edit("bsc-thesis.tex",
         r"\delta_\mathrm{icos}^2 = \gamma_\mathrm{dodec}^2 = \tfrac{1}{2} + ",
         r"\delta_\mathrm{icos}^2 = \gamma_\mathrm{dodec}^2 = \tfrac{1}{2} - ")


@control("corrupt the unfactored display's sqrt(m) to sqrt(2)", 32)
def _c102():
    # The visible sqrt(2/V) survives, so the string pin passes; the product
    # in front is then sqrt(4/V), not 1/sqrt2, and A is not orthogonal.
    sandbox()
    edit("bsc-thesis.tex", r"A \;=\; \sqrt{m}\,\sqrt{\tfrac{2}{V}}",
         r"A \;=\; \sqrt{2}\,\sqrt{\tfrac{2}{V}}")


@control("restate the strip's `no antipodes' cell as a sentence", 20)
def _c83():
    # Three statements of one fact within four centimetres: the strip cell,
    # head 4's "No coin over axes", and the sentence.  The strip carries it.
    patch_block("CCoin", "tetrahedron", "The dilation is forced;",
                "No antipodal pair: the dilation is forced;")


# --- the five draw notes ---------------------------------------------------

@control("print the two-jobs sentence on the octahedron's card", "19d")
def _c84():
    # The sentence is TRUE only where the bars meet.  On the octahedron it
    # would contradict the thesis's own counterexample ("the octahedron's
    # coin realizes its POVM and twirls nothing", Section 5.2.3 and F.3.2), so
    # 19d pins it to the icosahedron alone.
    patch_block("CCoin", "octahedron", " Random Pauli measurement.",
                " Random Pauli measurement. A full $\\TwoT$ draw does "
                "randomness's two jobs at once: it realizes the POVM and "
                "twirls the noise (Section~\\ref{sec:shadows:bills}).")


@control("drop the two-jobs sentence from the icosahedron", "19d")
def _c85():
    patch_block("CCoin", "icosahedron",
                " A full $\\TwoT$ draw does randomness's two jobs at once: "
                "it realizes the POVM and twirls the noise "
                "(Section~\\ref{sec:shadows:bills}).", "")


@control("drop the tetrahedron's cube sentence", "19d")
def _c87():
    # The one claim the thesis prints nowhere else: drawing 2T against
    # an aligned tetrahedron vertex is not a failure, it assembles the CUBE.
    # Without it the card's reader is left where bsc-thesis.tex leaves them,
    # with the exclusion and not its consequence.
    patch_block("CCoin", "tetrahedron",
                " Align a vertex and draw from $\\TwoT$ anyway: each readout "
                "reports an axis, not a vertex, and the eight effects "
                "assembled are the cube's "
                "(Figure~\\ref{fig:dec_cube_circuit}).", "")


@control("miscount the octahedron's coin in its own note", "19d")
def _c88():
    # The numeral is the PRINTED ray list above the sentence, and 19d derives
    # it from len(rays) rather than reading the word.  Three axes, three
    # words: "four" is the cube's card and would send a reader to a coin that
    # is not on this page.
    patch_block("CCoin", "octahedron",
                " Realization needs only these three;",
                " Realization needs only these four;")


@control("give the cube the dodecahedron's verdict", "19d")
def _c89():
    # The two jobs part ways in opposite directions on these two cards, and
    # the ladder is the point: on the cube it is the twirl that forces the
    # draw, on the dodecahedron realization.  Swapping them bills the full
    # 2I draw to the twirl when it is realization's.
    patch_block("CCoin", "cube",
                " Realization needs only these four; among groups it is the "
                "twirl that forces the full $\\TwoT$ draw "
                "(Section~\\ref{sec:shadows:bills}).",
                " Among groups, realization forces $\\TwoI$.")


@control("leave the cube without a draw note", "19d")
def _c90():
    # The octahedron and the cube share their rung, and the pin is written
    # against both: one card dropping the sentence is as much a failure as
    # one card gaining it.
    patch_block("CCoin", "cube",
                " among groups it is the twirl that forces the full "
                "$\\TwoT$ draw (Section~\\ref{sec:shadows:bills}).", "")


@control("cite the wrong Decker paper in the kappa label", 12)
def _c86():
    # Everything belonging to Decker carries a \cite, and the cite is part of
    # check 12's verbatim free-pairing literal, so a wrong key -- the 2005
    # POVM paper instead of the 2004 circuits paper the five captions cite --
    # is an abort, not a bibliography surprise.
    patch_block("CKappa", "tetrahedron",
                "decker2004quantumcircuitssinglequbit",
                "decker2005implementationgroupcovariantpositive")


@control("point the Naimark gloss at the dashed box", 12)
def _c91():
    # The gloss names the SOLID gategroup -- Decker's published block -- and
    # the page carries exactly two, head 2 having already named the dashed
    # U_g.  Pointed at the dashed one it reads as the claim that the optional
    # twirl is the Naimark extension unitary, so the shared literal is pinned
    # verbatim and a drifted noun is an abort.
    patch_block("CParams", "dodecahedron", "the solid box", "the dashed box")


# --- the flip completion ---------------------------------------------------

@control("drop the group scope from the dodecahedron's verdict", "19d")
def _c92():
    # Unscoped, "realization forces the 2I draw" is FALSE since
    # randomized_twojobs.check_flip_completion: the ten-word coin times the
    # Klein layer, forty elements and not a group, realizes the POVM and is
    # exactly depolarizing at T_zz for every (T, t) at half the Phi.  The
    # minimum is over SUBGROUPS on all three cards that state one, and 19d
    # pins the scope to exactly those three.
    patch_block("CCoin", "dodecahedron",
                " Among groups, realization forces $\\TwoI$.",
                " Realization forces $\\TwoI$.")


@control("move the dodecahedron's flip off the readout end", "19d")
def _c93():
    # The position is the whole content of the sentence.  Last, after the
    # fixed alignment, the conjugated flip fixes the seed axis and only signs
    # the snapshot, so realization is inherited from the coin and the four
    # flips average the readout row down to T_zz zhat^T.  Moved to the front,
    # before the alignment, the Klein orbit of v0 collapses to two axes, the
    # ten-word coin does not re-spread them, and the ensemble loses BOTH jobs
    # (hits 2..6, twirl gone by 5e-2 on the probe).  A card that misplaces it
    # prescribes a circuit that measures the wrong POVM.
    patch_block("CCoin", "dodecahedron",
                " $\\Id, X, Y, Z$ last ($X$ or $Y$ swaps the pair), these forty",
                " $\\Id, X, Y, Z$ first ($X$ or $Y$ swaps the pair), these forty")


# --- the flip's relabelling ------------------------------------------------

@control("drop the flip's relabelling from the dodecahedron's note", "19d")
def _c95():
    # Panel D reads pairs <0>-vertex first, and a drawn X or Y flips the Z
    # outcome, so on those two draws the pair reads reversed.  Without the
    # swap the four flips average the readout to (t/2) Id -- a measurement
    # that reports nothing -- and the card would prescribe it while still
    # saying "twirl".  19d builds the parenthesis from the gates' action on
    # zhat and requires it on the card.
    patch_block("CCoin", "dodecahedron",
                " last ($X$ or $Y$ swaps the pair), these forty",
                " last, these forty")


@control("name the wrong flips as the ones that swap the pair", "19d")
def _c96():
    # Z fixes zhat and X, Y negate it; the parenthesis printing {X, Z} would
    # tell the reader to reverse the pair on a draw that does not flip the
    # outcome, and to keep it on one that does.  The literal is derived, so
    # the wrong pair is an abort.
    patch_block("CCoin", "dodecahedron",
                "($X$ or $Y$ swaps the pair)", "($X$ or $Z$ swaps the pair)")


# --- the priced misread is D.2's claim too ---------------------------------

@control("delete D.2's sentence licensing the priced misread", 36)
def _c94():
    # The label's priced branch ("His circuit with our list, index for
    # index, is the mismatch") is D.2's "his circuit with our list gives
    # the $\kappa$ in Table D.4", plus the rider; if D.2 stops saying it the
    # card must stop pricing it.
    sandbox()
    edit("bsc-thesis.tex",
         "his circuit with our list gives the $\\kappa$ in",
         "a mismatch gives the $\\kappa$ in")


# --- the driver ------------------------------------------------------------

def run():
    import importlib
    rows = []
    for name, num, setup in CONTROLS:
        importlib.reload(B)
        B.PAPER, B.FIGURES = REAL_PAPER, REAL_PAPER / "figures"
        out = Path(tempfile.mkdtemp(prefix="ncout-"))
        DEFERRED["installed"], DEFERRED["fired"] = [], []
        # THE TWO PHASES ARE SEPARATE, and this is the whole staleness rule.
        # A control that corrupts nothing is not a control, and every way of
        # corrupting nothing raises in setup(): a literal that has gone stale
        # (AssertionError from sub/edit/emitter/first_with/patch_block), an
        # emitter the builder no longer has (AttributeError), a helper whose
        # own body is broken (NameError, TypeError, KeyError, IndexError).
        # Only exceptions raised from B.main() may count as an abort.
        #
        # Both halves of the rule matter.  An emitter renamed out from under
        # a control raises AttributeError before the corruption is installed,
        # and a helper with a broken body raises NameError, TypeError,
        # KeyError or IndexError in the same place; a generic
        # `except Exception -> aborted` counts either as a pass while the
        # suite corrupts nothing.  So it is not used here.
        try:
            setup()
        except StaleControl as e:
            verdict = "STALE: " + " ".join(str(e).split())[:88]
        except Exception as e:                       # noqa: BLE001
            msg = " ".join(str(e).split())
            verdict = f"STALE: setup raised {type(e).__name__}: " + msg[:80]
        else:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    B.main(["--out", str(out)])
                verdict = "DID NOT ABORT"
            except StaleControl as e:
                # a DEFERRED literal that has gone stale: it surfaces from
                # inside main(), and it is a dead control, not an abort
                verdict = "STALE: " + " ".join(str(e).split())[:88]
            except AssertionError as e:
                verdict = "aborted: " + " ".join(str(e).split())[:90]
            except Exception as e:                   # noqa: BLE001
                # the builder falling over ON the corruption is an abort too,
                # but it is labelled so a reader can see it was not a clean
                # assertion
                verdict = f"aborted ({type(e).__name__}): " + \
                    " ".join(str(e).split())[:80]
            # ... and a deferred corruption that was installed and never ran
            # corrupted nothing, however the run ended
            miss = [x for x in DEFERRED["installed"]
                    if x not in DEFERRED["fired"]]
            if miss and not verdict.startswith("STALE"):
                verdict = ("STALE: the corruption was installed and never "
                           "fired: %s" % (miss,))[:96]
        wrote = sorted(p.name for p in out.glob("*"))
        rows.append((name, num, verdict, wrote))
    B.PAPER, B.FIGURES = REAL_PAPER, REAL_PAPER / "figures"

    bad = 0
    print(f"{len(rows)} negative controls\n" + "-" * 78)
    for name, num, verdict, wrote in rows:
        ok = verdict.startswith("aborted") and not wrote
        bad += not ok
        print(f"[{'ok ' if ok else 'FAIL'}] a{num:<2} {name}")
        print(f"        {verdict}")
        if wrote:
            print(f"        WROTE {wrote}")
    print("-" * 78)
    print(f"{len(rows) - bad}/{len(rows)} controls aborted and wrote nothing")
    return bad


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
