"""
LaTeX fragments + figures for the shadows appendix (companion to Section 5.2; of these only shadow_gatenoise.tex is printed, as Table F.3 of Appendix F.3.4).

Reads data/shadow_experiments.npz (run shadow_experiments.py first) and reuses
the channel/state machinery of shadow_experiments.py for the exact prediction
curves.  Writes into data/:

  shadow_exp1.tex          MSE at the canonical dual (TFIM + Haar average),
                           with the exact-variance column the MC scatters
                           around (antipodal solids provably identical)
  shadow_exp2.tex          exact MSE ratios of optimized duals to the
                           canonical dual (MC agreement demoted to caption)
  shadow_gatenoise.tex     residual bias of the two randomized
                           implementations (randomized-projective and
                           twirled-native) under per-gate noise (the twirl's
                           assumption priced with the atlas's own circuits)
  shadow_robust_sweep.pdf  MSE vs depolarizing rate + decomposition at p = 0.1
  shadow_blindness.pdf     measured vs predicted bias under z-fixing channels
  shadow_nscaling.pdf      exact variance / optimization gain vs chain length

Table fragments are full table environments (booktabs, [H]) in the style of
povm_properties.py; the captions live here, so reword this script and
regenerate rather than editing the .tex.

    cd code && uv run shadow_report.py
"""

import os

import numpy as np

# Pinned before pyplot loads: matplotlib stamps the PDFs' CreationDate from
# SOURCE_DATE_EPOCH at savefig time, and the repo's contract is that a re-run
# reproduces every tracked artifact byte-identically.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shadow_experiments import (
    DATA, SOLIDS, DEPOL_RATES, BLIND_RATES, GATE_GAMMAS, NSCALE_NS,
    R_REPS, T_MAIN, T_EXP2, RC_CAL, N_HAAR, G_CRIT, GAMMA_DIL,
    chan_dephasing, chan_amp_damping,
    density, tfim_ground_state, site_bloch,
)

OBS_ALL = ["Z0", "X0", "Z0Z1", "X0X1", "Z0Z1Z2", "E_TFIM"]
OBS_EXP2 = ["Z0", "X0", "Z0Z1", "E_TFIM"]
OBS_TEX = {
    "Z0": "$Z_0$", "X0": "$X_0$", "Z0Z1": "$Z_0Z_1$", "X0X1": "$X_0X_1$",
    "Z0Z1Z2": "$Z_0Z_1Z_2$", "E_TFIM": "$H_{\\mathrm{TFIM}}$",
}
POVM_TEX = {"tetrahedron": "Tetrahedron", "octahedron": "Octahedron",
            "cube": "Cube", "icosahedron": "Icosahedron",
            "dodecahedron": "Dodecahedron"}
EXP2_LEVELS = [("observable", "observable-opt."), ("oracle", "oracle")]
EXP3_LEVELS = [("noiseless", "Noiseless canonical"),
               ("robust", "Robust canonical"),
               ("optimized", "Robust optimized"),
               ("oracle", "Robust per-qubit oracle")]
GATE_BLOCKS = [
    ("Randomized-projective, octahedral POVM", [
        ("2O", "$\\TwoO$", "min-depth", "uniform $\\gamma$")]),
    ("Randomized-projective, icosahedral POVM", [
        ("2I", "$\\TwoI$", "min-depth", "uniform $\\gamma$"),
        ("2I_phi3", "$\\TwoI$", "min-depth", "$3\\gamma$ on $\\Phi$"),
        ("2I_dij_phi3", "$\\TwoI$", "min-$\\Phi$", "$3\\gamma$ on $\\Phi$"),
        ("2T", "$\\TwoT$", "min-depth", "uniform $\\gamma$")]),
    ("Twirled-native, every POVM (SIC included)", [
        ("R2/icosahedron", "$\\TwoT$", "min-depth", "uniform $\\gamma$")]),
]


def gate_series(key):
    """The npz series a table row's *metadata* comes from.  The twirled-native
    rows share one: their estimator channel is provably solid-independent, so
    gatenoise/meta/ and gatenoise/indep/ are stored once under R2 rather than
    once per solid.  The per-row sweep is NOT collapsed -- gatenoise/{row}/
    {gamma} is stored for all five solids, which is what lets a second
    twirled-native row be printed without regenerating the npz."""
    return "R2" if key.startswith("R2/") else key


# every row key in printing order, the distinct metadata series behind them,
# and the protocol split.  Everything row-keyed below quantifies over these
# rather than over a hand-typed list or a positional slice of GATE_BLOCKS: a
# row added to any block is then picked up by every check and flag that
# applies to it, instead of silently escaping the ones it was not typed into.
GATE_ROWS = [row[0] for _, rows in GATE_BLOCKS for row in rows]
GATE_SERIES = list(dict.fromkeys(gate_series(k) for k in GATE_ROWS))
GATE_PROJECTIVE = [k for k in GATE_ROWS if gate_series(k) != "R2"]
# the two Phi-weighted rows are a variant of the plain 2I row above them --
# "3 gamma on Phi" is a modeling choice, magic cost read as noise cost, not a
# second measurement -- so they are indented under it and the table's spine
# reads 2O -> 2I -> 2T -> protocol swap.  Without this the eye's attractor,
# the largest residual in the table, is also its most hypothetical cell.
GATE_NESTED = {"2I_phi3", "2I_dij_phi3"}
# the twirled-native Z0 slope is EXACTLY zero, not merely small: randomized_
# scalars.py proves the linear coefficient vanishes identically in (x, y, z)
# and for every dilation strength (`lin[...] == 0`, in its
# check_gate_noise_residual).  A
# printed "-0.000" reads as a rounding artifact where a theorem belongs, so
# this cell prints 0.  The slope is still asserted below the print's own
# resolution, so the flag cannot outlive the fact.  Derived rather than typed,
# because the proof is solid-independent and so covers every twirled-native
# row: typed as a literal, a second such row -- the SIC in the table body --
# would print -0.000 beside the first row's 0 from data identical to nine
# decimals, under a caption calling the two rows the same.
GATE_EXACT_ZERO_Z = {k for k in GATE_ROWS if gate_series(k) == "R2"}
# GATE_NESTED is a layout judgement, so it alone stays keyed by hand, and a
# typo in it is SILENT: the indent just fails to happen while the caption goes
# on claiming it, and the page still typesets.  Same failure the exp-1 block
# guards below.  The subset test catches a key with no row; a row that should
# have been nested and was not is the direction no assert can reach.
assert GATE_NESTED <= set(GATE_ROWS), "stale row key"
# the twirled-native Z0 residual's exact gamma^2 coefficient (asserted below):
# randomized_implementations.py derives (4 sqrt(95) - 19)(z - 1)/244 for the
# dilated row symbolically, and the TFIM site-0 Bloch vector has z = 0.
R2_QUAD_EXACT = -(4 * np.sqrt(95) - 19) / 244        # -0.08191466...

HEADER = ("% Auto-generated by code/shadow_report.py from "
          "data/shadow_experiments.npz.\n"
          "% Include this file with \\input{...}.  Requires booktabs, float.\n"
          "% Regenerate: `uv run shadow_experiments.py && uv run "
          "shadow_report.py` in code/.\n\n")


def write(name, body):
    path = DATA / name
    path.write_text(HEADER + body)
    print(f"  -> {path}")


def main():
    d = np.load(DATA / "shadow_experiments.npz")

    # ----------------------------------------------------------------- exp 1
    # One "Computed" column serves a row that includes the tetrahedron, and the
    # caption tells the reader the tetrahedron shares that value everywhere
    # but one bolded cell.  Typed here as a pair of string equalities, that
    # cell would go unchecked -- the mathematics behind it is pinned in
    # shadow_experiments.py but the selection is not, so an added observable or
    # a renamed key would bold the wrong cell -- or none -- and still typeset.
    # Read it off the data instead, and require the answer to be the single
    # cell the caption goes on to name.  Margin: the eleven agreeing cells top
    # out at 1.4e-14 and the twelfth sits at 1.28 -- a gap spanning fourteen
    # orders, with the 1e-9 test about five orders above the agreeing end and
    # nine below the deviating one.  Both numbers come off the committed npz,
    # so they are fixed -- they move only if the npz does.
    bolded, margins = [], []

    def exp1_block(title, mse_of, exact_key):
        lines = [f"\\multicolumn{{8}}{{l}}{{\\itshape {title}}} \\\\"]
        for on in OBS_ALL:
            vals = np.array([mse_of(pn, on) for pn in SOLIDS])
            # CV over the antipodal four (vals[1:]): only their exact targets
            # are provably equal; the tetrahedron column is shown but scored
            # against its own exact value in the caption.
            cv = vals[1:].std(ddof=1) / vals[1:].mean()
            exact = d[f"exact/var/{exact_key}/{on}"][0]
            tetra = d[f"exact/var_tetra/{exact_key}/{on}"][0]
            gap = abs(tetra - exact)
            margins.append(gap)
            exact_cell = f"{exact:.2f}"
            if gap > 1e-9:                 # the tetrahedron misses this cell
                bolded.append((exact_key, on))
                exact_cell = f"$\\pmb{{{exact:.2f}}}$"
            cells = "".join(f" & {v * 1e3:.2f}" for v in vals)
            lines.append(f"  {OBS_TEX[on]}{cells} & {exact_cell}"
                         f" & {cv * 100:.1f}\\% \\\\")
        return lines

    rows = exp1_block("TFIM ground state ($h=1$)",
                      lambda pn, on: d[f"exp1/{pn}/TFIM/{on}"][2], "TFIM")
    rows += ["\\midrule"]
    rows += exp1_block(f"Haar average (same {N_HAAR} states)",
                       lambda pn, on: d[f"exp1_haar/{pn}/{on}"][0],
                       "haar_mean")
    assert bolded == [("haar_mean", "E_TFIM")], (
        "the caption names the bolded Haar-TFIM cell as the only one the "
        f"tetrahedron misses; the data bolds {bolded}"
    )
    agree = [m for m in margins if m <= 1e-9]
    deviate = [m for m in margins if m > 1e-9]
    assert max(agree) < 1e-12 and min(deviate) > 1.0, (
        "the agreeing cells and the deviating one are not cleanly separated "
        f"either side of the 1e-9 test: agree <= {max(agree):.1e}, "
        f"deviate >= {min(deviate):.1e}"
    )
    tetra_haar_e = d["exact/var_tetra/haar_mean/E_TFIM"][0]
    uni_haar_e = d["exact/var/haar_mean/E_TFIM"][0]
    body = "\n".join([
        "\\begin{table}[H]\\centering\\small",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{l rrrrr r r}",
        "\\toprule",
        " & \\multicolumn{5}{c}{MSE $\\times 10^{3}$} & & \\\\",
        "\\cmidrule(lr){2-6}",
        "Observable & Tetra. & Octa. & Cube & Icosa. & Dodec. "
        "& Computed & CV \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Experiment 1: mean squared error of the canonical-dual "
        f"shadow estimator (${T_MAIN}$ shots, ${R_REPS}$ replications, "
        "$n=4$ qubits), all five solids on the same states. Computed is the "
        "common single-shot variance of the four antipodal solids, computed "
        "rather than sampled, which at $10^3$ shots coincides "
        "numerically with $\\mathrm{MSE}\\times10^3$. CV, taken over the "
        "four antipodal columns, is pure Monte Carlo scatter around that "
        "common target. The tetrahedron---the one non-antipodal solid, "
        "hence the one with a nonvanishing third moment---differs only in "
        f"the bolded Haar-TFIM cell, {tetra_haar_e:.2f} against "
        f"{uni_haar_e:.2f}: a third-moment contribution takes two "
        "different Pauli letters meeting on a site (only the energy) and "
        "a state with complex amplitudes (only the Haar draws).}",
        "\\label{tab:shadow-exp1}",
        "\\end{table}",
    ])
    write("shadow_exp1.tex", body)

    # ----------------------------------------------------------------- exp 2
    # Rows = state block x antipodal solid; each observable is a two-column
    # (obs.-opt., oracle) strip, so the across-solid comparison the text
    # makes is read top to bottom.  The tetrahedron (unique dual, ratios
    # identically 1) is a caption sentence, not sixteen cells.
    def ratio_cell(pn, sn, on, ln):
        r = d[f"exact_ratio/{pn}/{sn}/{on}/{ln}"][0]
        return "$0$" if abs(r) < 1e-9 else f"{r:.3f}"

    blocks = []
    for sn, title in [("TFIM", "TFIM ground state ($h=1$), in-distribution"),
                      ("product", "product state, out-of-distribution")]:
        blocks.append(f"\\multicolumn{{9}}{{l}}{{\\itshape {title}}} \\\\")
        for pn in SOLIDS[1:]:
            cells = "".join(f" & {ratio_cell(pn, sn, on, ln)}"
                            for on in OBS_EXP2 for ln, _ in EXP2_LEVELS)
            blocks.append(f"  {POVM_TEX[pn]}{cells} \\\\")
        if sn == "TFIM":
            blocks.append("\\midrule")
    body = "\n".join([
        "\\begin{table}[H]\\centering\\small",
        "\\renewcommand{\\arraystretch}{1.05}",
        "\\begin{tabular}{l cc cc cc cc}",
        "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{OBS_TEX[on]}}}"
                           for on in OBS_EXP2) + " \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}"
        "\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
        "POVM" + " & obs.-opt. & oracle" * len(OBS_EXP2) + " \\\\",
        "\\midrule",
        *blocks,
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Experiment 2: variance of the optimized duals relative "
        "to the canonical dual, per observable (column pairs) at the two "
        "optimization levels. Every ratio is computed from "
        "the outcome tensors, not sampled; the Monte Carlo run alongside "
        f"(${T_EXP2}$ shots, ${R_REPS}$ replications) agrees with each "
        "cell within five standard errors, asserted in code. The "
        "observable-optimized dual is trained on TFIM "
        "($h\\in\\{0.3,0.5,0.7,1.0,1.5\\}$) and GHZ reduced states; the "
        "per-qubit oracle receives the true reduced state of each site---a "
        "ceiling, not a protocol. The tetrahedron ($V=d^2$: unique dual, "
        "all ratios $\\equiv 1$) is omitted.}",
        "\\label{tab:shadow-exp2}",
        "\\end{table}",
    ])
    write("shadow_exp2.tex", body)

    # ------------------------------------------------------------ gate noise
    g0 = GATE_GAMMAS[1]
    slopes = {}
    # linearity is measured per COLUMN, not on Z0 alone: the caption says
    # "for every randomized-projective row", and read as a reader parses
    # "row" that needs X0 too -- the plain-2I X0 cell drifts 15% over the
    # grid (its slope is -0.010, so the curvature is large relatively, not
    # absolutely; every other X0 cell is within 3.1%). Both columns are
    # measured and both figures print, the X0 one attributed to that cell --
    # and the attribution is asserted, not typed.
    lin_dev = {"Z0": 0.0, "X0": 0.0}
    lin_arg = {"Z0": None, "X0": None}
    for key in GATE_ROWS:
        row = d[f"gatenoise/{key}/{g0}"]
        s_z, s_x = row[4] / g0, row[3] / g0
        slopes[key] = (s_z, s_x)
        if key in GATE_PROJECTIVE:           # the R2 Z0 is second order
            for g in GATE_GAMMAS[2:]:
                for col, idx, s in (("Z0", 4, s_z), ("X0", 3, s_x)):
                    dev = abs(d[f"gatenoise/{key}/{g}"][idx] / g / s - 1.0)
                    if dev > lin_dev[col]:
                        lin_dev[col], lin_arg[col] = dev, key
    # lin_dev goes straight into the caption as "linear in gamma to within
    # N%", with nothing bounding N: the sentence would read the same at 400%,
    # where it would no longer be describing a linear response at all. The
    # measured values are 5% and 15%, so the bound below is loose by an order
    # (a third of one) and still catches the caption going false.
    assert max(lin_dev.values()) < 0.5, \
        f"the projective rows are not linear in gamma: {lin_dev}"
    # the caption's parenthetical: the X0 figure is set by the plain-2I row,
    # and that row's X0 slope is the smallest of the projective rows
    assert lin_arg["X0"] == "2I", f"the X0 drift is no longer the plain-2I row's: {lin_arg}"
    assert abs(slopes["2I"][1]) == min(abs(slopes[k][1]) for k in GATE_PROJECTIVE), \
        "the plain-2I X0 cell is no longer the near-zero one"
    # the octahedral projective row prints +0.250, and that cell has an exact
    # value, 1/4 (Richardson on the two smallest gammas gives 0.2500003). The
    # cell carries three decimals, so 5e-4 is where the print would move; the
    # measured deviation at g0 is 1.3e-4.
    assert abs(slopes["2O"][0] - 0.25) < 5e-4, \
        f"the 2O Z0 slope has left 1/4: {slopes['2O'][0]:.6f}"
    span_2i = d[f"gatenoise/span/2I/{g0}"] / g0
    span_2t = d[f"gatenoise/span/2T/{g0}"] / g0
    r2_quad = d[f"gatenoise/R2/icosahedron/{g0}"][4] / g0 ** 2
    # lin_dev exempts the R2 row as second order, which left nothing checking
    # that it IS second order, nor that the caption's single-gamma coefficient
    # is the right one.  Same shape as lin_dev, one order up: the gamma^2
    # coefficient must hold across the grid (measured drift 5%, so the same
    # order-loose bound), and at small gamma it must be R2_QUAD_EXACT.  At
    # g0 = 1e-3 the grid's own curvature already costs 8.2e-5 of that, so the
    # bound is 1e-3.
    quad_dev = 0.0
    for g in GATE_GAMMAS[2:]:
        quad_dev = max(quad_dev,
                       abs(d[f"gatenoise/R2/icosahedron/{g}"][4] / g ** 2
                           / r2_quad - 1.0))
    assert quad_dev < 0.5, f"the R2 row is not quadratic in gamma: {quad_dev:.2f}"
    assert abs(r2_quad - R2_QUAD_EXACT) < 1e-3, \
        f"the R2 gamma^2 coefficient has left its exact value: {r2_quad:.6f}"
    # F.3.4's prose and this caption repeat the table in words and rounded
    # literals; each is pinned here at the resolution the sentence carries,
    # and the comments below name the sentence each pin guards.  The cells
    # print from the same slopes, so a pin failing means the sentence, not
    # the table, has rotted.
    z = {k: slopes[k][0] for k in GATE_ROWS}
    x = {k: slopes[k][1] for k in GATE_ROWS}
    worst = max(abs(d[f"gatenoise/{k}/{g0}"][4]) for k in GATE_PROJECTIVE)
    assert f"{worst:.1e}" == "1.7e-03", worst           # "biases $Z_0$ by 1.7e-3"
    assert 1.8 < z["2I"] / z["2O"] < 2.3, z             # "about twice 2O's Z_0 bill"
    assert z["2I_phi3"] / z["2I"] > 2, z                # "more than doubles 2I's bill again"
    assert z["2I_dij_phi3"] / z["2I_phi3"] > 4 / 3, z   # "raises the bill again" (prose)
    # the four below are not repeated in prose -- the table's own cells
    # carry them -- and stand as data checks on those cells
    assert abs(x["2I_dij_phi3"] / x["2I_phi3"] - 1) < 0.1, x   # X_0 barely moves with it
    assert 0.45 < abs(z["2T"]) / z["2I"] < 0.6, z       # 2T's Z_0 bill is roughly half 2I's
    assert z["2T"] < 0 < z["2I"], z                     # and of the opposite sign
    assert f"{x['2I']:+.3f}" == "-0.010" and f"{x['2T']:+.3f}" == "+0.348", x   # the printed X_0 cells, -0.010 to +0.348
    # the caption's "wholly below it in magnitude" (the 2T row's alignment
    # span against the plain-2I one): disjoint in magnitude, the 2T one below
    assert np.abs(span_2t).max() < np.abs(span_2i).min(), (span_2t, span_2i)
    assert 0.45 < x["R2/icosahedron"] / x["2T"] < 0.55, x   # caption: "about half the projective 2T row's"
    assert f"{r2_quad:+.3f}" == "-0.082", r2_quad       # the caption interpolates it; the prose does not type it
    # the prose's depth/Phi comparatives ("deeper and carrying most of a Phi
    # per rotation", "with less Phi per rotation") and the two literals the
    # table's Phi column prints
    meta = {k: d[f"gatenoise/meta/{k}"] for k in ("2I", "2O", "2I_dij_phi3")}
    assert meta["2I"][1] > meta["2O"][1] and meta["2I"][3] > 0.5 > meta["2O"][3]
    assert f"{meta['2I_dij_phi3'][3]:.2f}" == "0.80" \
        and f"{meta['2I'][3]:.2f}" == "0.92", meta
    # the order-of-magnitude gap the caption prints, derived rather than
    # typed.  At g0 the projective rows are linear and the twirled-native row
    # quadratic, so the gap is |s_z| / (|r2_quad| g0) -- and it is not ONE
    # figure: 3.5 orders for 2O, 4.3 for the Phi-weighted rows.  Typed as a
    # single "four" it would hold only at the top of that range, while reading,
    # plural and unqualified, as a claim about all of it -- and a reader can
    # catch that from the table's own cells, since 2O prints +0.250 and the
    # coefficient below is in the same sentence.
    gaps = [np.log10(abs(slopes[k][0]) / (abs(r2_quad) * g0))
            for k in GATE_PROJECTIVE]
    ord_lo, ord_hi = round(min(gaps)), round(max(gaps))
    words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    # a KeyError here means the gap left the range the sentence can say in
    # words; ord_lo == ord_hi would make the caption's "to" a lie.
    ord_span = f"{words[ord_lo]} to {words[ord_hi]}"
    assert ord_lo < ord_hi, \
        f"the order gap no longer spans a range: {min(gaps):.2f}-{max(gaps):.2f}"
    # the caption interpolates ord_span rather than typing "three to four
    # orders below the projective rows", and the pin stays so that a data
    # change moves the sentence loudly rather than silently.
    assert ord_span == "three to four", \
        f"the F.3.4 prose says three to four orders; the data says {ord_span}"
    # the caption types the reference strength as $10^{-3}$, typography a
    # format string will not reproduce, so pin it instead of interpolating it.
    assert g0 == 1e-3, f"the caption's $10^{{-3}}$ is no longer g0: {g0}"
    # the two controls, printed as rows rather than left to the caption's last
    # sentence: they are this experiment's null, and putting them in the body
    # makes zero the table's baseline instead of an oddity in its last row.
    ctrl = max(abs(d[f"gatenoise/{k}/0.0"][[3, 4]]).max() for k in GATE_ROWS)
    ctrl_indep = max(abs(d[f"gatenoise/indep/{k}"]).max()   # both protocols
                     for k in GATE_SERIES)
    # both control rows print their cells as literals, so this assert is the
    # whole of what keeps them honest.  1e-12 is the caption's stated figure and
    # shadow_experiments.py's own assertion threshold; asserting it again here
    # survives that file's tolerance ever being loosened.
    assert max(ctrl, ctrl_indep) < 1e-12, \
        f"a control row is no longer zero: {max(ctrl, ctrl_indep):.2e}"
    rows = ["\\multicolumn{7}{l}{\\itshape Controls: every draw, every "
            "compilation, both protocols} \\\\",
            "  \\multicolumn{3}{l}{no gate noise ($\\gamma = 0$)} "
            "& & & $0$ & $0$ \\\\",
            "  \\multicolumn{3}{l}{gate-\\emph{independent} noise, any "
            "$\\gamma$} & & & $0$ & $0$ \\\\",
            "\\midrule"]
    for title, series in GATE_BLOCKS:
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\itshape {title}}} \\\\")
        for key, gtex, comp, noise in series:
            meta = gate_series(key)
            _, mean_dep, _, mean_phi = d[f"gatenoise/meta/{meta}"]
            s_z, s_x = slopes[key]
            phi_cell = "---" if mean_phi == 0 else f"{mean_phi:.2f}"
            draw = f"\\quad {gtex}" if key in GATE_NESTED else gtex
            if key in GATE_EXACT_ZERO_Z:
                assert abs(s_z) < 5e-4, \
                    f"the {key} Z0 slope is no longer below the print: {s_z:.6f}"
                z_cell = "$0$"
            else:
                z_cell = f"${s_z:+.3f}$"
            rows.append(f"  {draw} & {comp} & {noise} & "
                        f"{mean_dep:.2f} & {phi_cell} & "
                        f"${s_x:+.3f}$ & {z_cell} \\\\")
    body = "\n".join([
        "\\begin{table}[H]\\centering",
        "\\renewcommand{\\arraystretch}{1.15}",
        "\\begin{tabular}{lll cc rr}",
        "\\toprule",
        # |G| is dropped: it is a function of the Draw column, and printing
        # 24 beside 2O invites reading it as the order of the binary group
        # (48) when it is the order of the rotation group it covers.
        # Mean depth and mean Phi stay -- the min-magic row's point is
        # that it carries LESS Phi and leaves a LARGER residual, and
        # that is unreadable without the column.  $Z_0$ is rightmost: it is
        # the component every claim in the section argues over.
        " & & & Mean & Mean & \\multicolumn{2}{c}{Residual bias "
        "$/\\, \\gamma$} \\\\",
        "\\cmidrule(lr){6-7}",
        "Draw & Compiled & Noise & depth & $\\Phi$ & "
        "$X_0$ & $Z_0$ \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        # The caption is long by design, carrying the technicalities
        # F.3.4's prose leaves out -- the alignment-span brackets, the
        # "linear to within N%" clause, the alignment and dilation-channel
        # conventions, the two controls, the order-of-magnitude gap
        # (ord_span, interpolated), the first-order mechanism, the X0 ratio,
        # the disjoint alignment spans and the symbolic gamma = 0 controls --
        # while what the prose does carry (the two-condition claim,
        # solid-independence) is not repeated. The pins above guard those
        # caption sentences. The block header reads "Twirled-native, every
        # POVM (SIC included)" over a single printed row and the caption
        # attributes the zero slope to that row: deliberate, the zero being
        # solid-independent (GATE_EXACT_ZERO_Z above), so the header's scope
        # is the theorem's rather than the row count's.
        "\\caption{Residual bias of the $\\ket{0}$-calibrated estimator on the "
        "site-0 Bloch components of the critical transverse-field Ising (TFIM) "
        "ground state, $H_{\\mathrm{TFIM}} = -\\sum_i Z_i Z_{i+1} - h \\sum_i X_i$ "
        "on the four-qubit ring at $h = 1$, when amplitude damping of strength "
        "$\\gamma$ follows every elementary gate of the drawn word, per unit "
        f"$\\gamma$; across $\\gamma \\le {GATE_GAMMAS[-1]}$ every "
        "randomized-projective row is linear in $\\gamma$ to within "
        f"{100 * lin_dev['Z0']:.0f}\\% in $Z_0$ and "
        f"{100 * lin_dev['X0']:.0f}\\% in $X_0$ (the latter set by the "
        "plain-$\\TwoI$ row's near-zero cell). Atlas circuits "
        "(Appendices~\\ref{app:atlas} and~\\ref{app:synthesis}), min-depth (BFS) or "
        "min-$\\Phi$ (Dijkstra); in the indented rows the "
        "non-Clifford $\\Phi$ receives $3\\gamma$. "
        "\\emph{Randomized-projective rows}: $\\gamma$ after each gate of the "
        "drawn word, the alignment (to the vertex nearest $+\\hat z$; "
        "$\\hat z$ itself for $\\TwoO$) ideal; across the twelve icosahedral "
        "alignments the plain-$\\TwoI$ "
        f"$Z_0$ slope spans $[{span_2i[0]:+.3f},\\, {span_2i[1]:+.3f}]$ and "
        f"the $\\TwoT$ row's $[{span_2t[0]:+.3f},\\, {span_2t[1]:+.3f}]$, "
        "wholly below it in magnitude. "
        "\\emph{Twirled-native row}: $\\gamma$ after each gate of the same "
        "drawn $\\TwoT$ word, the fixed dilation modeled as a "
        "$g$-independent pre-measurement channel (amplitude damping of "
        f"strength ${GAMMA_DIL}$) that the twirl removes. Its $Z_0$ "
        "slope is exactly zero; the residual resumes at "
        f"${r2_quad:+.3f}\\,\\gamma^2$, {ord_span} orders below the "
        "projective rows at $\\gamma = 10^{-3}$. The reason, proved "
        "symbolically for every state and dilation strength "
        "(\\texttt{code/randomized\\_implementations.py}): to first order "
        "in $\\gamma$ the $\\TwoT$ words' estimator channel is a diagonal "
        "matrix, whose $zz$-entry the calibration absorbs, plus an offset "
        "with no $z$-component; the $\\TwoO$ words' offset has one. Its "
        "$X_0$ slope is about half the projective $\\TwoT$ row's. "
        "\\emph{Controls}: with the gate "
        "noise off, or present but gate-\\emph{independent}, every series "
        "vanishes to $10^{-12}$ under both protocols, asserted in code; at "
        "$\\gamma = 0$ the $\\TwoT$ and $\\TwoO$ draws' channels are ideal "
        "symbolically, the twirled-native one at every dilation strength.}",
        "\\label{tab:shadow-gatenoise}",
        "\\end{table}",
    ])
    write("shadow_gatenoise.tex", body)

    # ----------------------------------------------- figure 1: robust sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5),
                                   constrained_layout=True)
    markers = ["o", "s", "^", "d"]
    for (ln, ltex), m in zip(EXP3_LEVELS, markers):
        ax1.plot(DEPOL_RATES,
                 [d[f"exp3/{p}/E_TFIM/{ln}"][2] for p in DEPOL_RATES],
                 marker=m, label=ltex)
    ax1.set_yscale("log")
    ax1.set_xlabel("depolarizing rate $p$")
    ax1.set_ylabel(r"MSE of $\langle H_{\mathrm{TFIM}}\rangle$")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_title("(a) estimator hierarchy under noise", fontsize=10)

    x = np.arange(len(EXP3_LEVELS))
    b2 = [d[f"exp3/0.1/E_TFIM/{ln}"][0] for ln, _ in EXP3_LEVELS]
    var = [d[f"exp3/0.1/E_TFIM/{ln}"][1] for ln, _ in EXP3_LEVELS]
    ax2.bar(x, b2, color="#c44e52", label="bias$^2$")
    ax2.bar(x, var, bottom=b2, color="#4c72b0", label="variance")
    ax2.set_xticks(x, ["noiseless\ncanonical", "robust\ncanonical",
                       "robust\noptimized", "robust\noracle"], fontsize=8)
    ax2.set_ylabel(r"MSE of $\langle H_{\mathrm{TFIM}}\rangle$ at $p=0.1$")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_title("(b) bias converted to variance", fontsize=10)
    fig.savefig(DATA / "shadow_robust_sweep.pdf")
    plt.close(fig)
    print(f"  -> {DATA / 'shadow_robust_sweep.pdf'}")

    # -------------------------------------------- figure 2: blindness biases
    r0 = site_bloch(density(tfim_ground_state(G_CRIT)))[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True,
                             sharey=True)
    grid = np.linspace(0, 0.22, 100)
    for ax, cname, chan, title in [
            (axes[0], "dephasing", chan_dephasing, "(a) dephasing"),
            (axes[1], "amp-damping", chan_amp_damping,
             "(b) amplitude damping")]:
        for axis, label, color in [(0, "$X_0$", "#4c72b0"),
                                   (2, "$Z_0$", "#c44e52")]:
            pred = [(chan(p)[0] @ r0 + chan(p)[1] - r0)[axis] for p in grid]
            ax.plot(grid, pred, "--", color=color, lw=1,
                    label=f"{label} predicted")
            meas = [d[f"blind/{cname}/{p}"][1 if axis == 0 else 3]
                    for p in BLIND_RATES]
            ax.plot(BLIND_RATES, meas, "o", color=color,
                    label=f"{label} measured")
        etas = [d[f"blind/{cname}/{p}"][0] for p in BLIND_RATES]
        ax.annotate(rf"$\hat\eta = {np.mean(etas):.3f}$ at every $p$"
                    "\n(calibration reads “noiseless”)",
                    xy=(0.04, 0.96), xycoords="axes fraction", va="top",
                    fontsize=8)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("noise rate $p$")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("bias of the robust estimator")
    fig.savefig(DATA / "shadow_blindness.pdf")
    plt.close(fig)
    print(f"  -> {DATA / 'shadow_blindness.pdf'}")

    # ---------------------------------------------- figure 3: n-scaling
    ns = np.array(NSCALE_NS)
    var = np.array([d[f"nscale/{n}/var"] for n in NSCALE_NS])
    floor = np.array([d[f"nscale/{n}/floor"][0] for n in NSCALE_NS])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5),
                                   constrained_layout=True)
    levels = [("canonical", "#4c72b0", "o", 0),
              ("observable-opt.", "#dd8452", "s", 1),
              ("oracle", "#55a868", "^", 2)]
    for label, color, m, i in levels:
        ax1.plot(ns, var[:, i] / ns, marker=m, color=color, label=label)
    ax1.plot(ns, floor / ns, marker=".", ls="--", color="gray",
             label="dual-indep. floor")
    ax1.set_xlabel("chain length $n$")
    ax1.set_ylabel(r"single-shot $\mathrm{Var}[\hat H_{\mathrm{TFIM}}]\,/\,n$")
    ax1.set_xticks(ns)
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax1.set_title("(a) variance per site", fontsize=10)

    gain_obs = 100 * (1 - var[:, 1] / var[:, 0])
    gain_orc = 100 * (1 - var[:, 2] / var[:, 0])
    fl_share = 100 * floor / var[:, 0]
    ax2.plot(ns, gain_obs, marker="s", color="#dd8452",
             label="observable-opt.")
    ax2.plot(ns, gain_orc, marker="^", color="#55a868", label="oracle")
    ax2.plot(ns, fl_share, marker=".", ls="--", color="gray",
             label="floor share of variance")
    # pin the converged values at the last markers (the text quotes these)
    for ys, color in [(gain_obs, "#dd8452"), (gain_orc, "#55a868"),
                      (fl_share, "gray")]:
        ax2.annotate(f"{ys[-1]:.1f}%", (ns[-1], ys[-1]),
                     xytext=(6, 0), textcoords="offset points",
                     va="center", ha="left", fontsize=8, color=color)
    ax2.set_xlabel("chain length $n$")
    ax2.set_ylabel("gain over canonical (%)")
    ax2.set_xticks(ns)
    ax2.set_xlim(ns[0] - 1, ns[-1] + 2.4)
    ax2.set_ylim(0, 35)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(alpha=0.3)
    ax2.set_title("(b) optimization gain converges, not collapses",
                  fontsize=10)
    fig.savefig(DATA / "shadow_nscaling.pdf")
    plt.close(fig)
    print(f"  -> {DATA / 'shadow_nscaling.pdf'}")

    print("Done.")


if __name__ == "__main__":
    main()
