# Symmetry Groups of the Platonic Solid POVMs: Binary Polyhedral Groups as Quantum Circuits

A BSc thesis, and its code. **→ [`paper/bsc-thesis.pdf`](paper/bsc-thesis.pdf)** (pre-submission).

The POVMs on the Bloch sphere, in the browser: **→ [`extras/povms.html`](https://arachnidaduality.github.io/platonic-povm-groups/extras/povms.html)**.

Behind every Platonic-solid POVM stands a group: the rotations carrying the solid to itself,
lifted through the SU(2) → SO(3) double cover to one of the three *binary polyhedral groups*
2T, 2O, 2I — the exceptional finite subgroups of SU(2), and exactly the unitaries that permute
the measurement's outcomes. The thesis constructs those groups, synthesizes all 24 + 48 + 120
elements as explicit circuits over a minimal gate set, weighs what they cost, and then considers the
measurements in classical shadow estimation.

```
paper/   manuscript (LaTeX + PDF) and its standalone figures
code/    everything computational — fifteen scripts, six notebooks
extras/  Appendix D.1 at reading pace, an interactive POVM viewer, a Lean 4 check, a terminal toy
```

## Where the numbers come from

Table and data figures in the thesis are emitted by scripts.
Everything runs from `code/`, in the order below — the later
entries read `.npz` files the earlier ones write. Python is managed with
[uv](https://docs.astral.sh/uv/), which installs the pinned environment on first use, and one
command runs it all:

```bash
cd code
uv run everything.py
```

`everything.py` runs the table below in order, stops at the first failing check, and ends with
per-step timings and an artifact-drift report — the `git status` delta over the generated
files, so on an untouched tree its last line is *no artifact drift*. Any single row runs the
same way (`uv run main.py`).

| Run | Writes | Lands in the thesis | Time |
| --- | --- | --- | --- |
| `main.py` | `atlas.txt`, `atlas.tex`, `differing_rows.tex` | Appendix A — the atlas; Appendix B — Table B.1 | 31 s |
| `povm_properties.py` | `data/povm_atlas.tex`, `povm_properties.tex` | Appendix E | 8 s |
| `export_numpy.py` | `data/*.npz` — groups, gates, POVMs | *(feeds everything below)* | 21 s |
| `randomized_implementations.py` | `data/randomized_*.tex` | §5.2.3 and Appendices D, F.3 | 110 s |
| `shadow_experiments.py` | `data/shadow_experiments.npz` | *(feeds the report)* | 57 s |
| `shadow_report.py` | `data/shadow_gatenoise.tex` (+ 2 tables and 3 figures, repository only) | Appendix F.3.4, Table F.3 | < 1 s |
| `_build_section_ladder.py` | `paper/figures/section_ladder.tex` | Appendix C, Figure C.1 | < 1 s |
| `_build_recipe_figure.py` | `data/recipe_tet_data.tex` | Figure 1.1, the recipe page | < 1 s |
| `_build_povm_cards.py` | `data/card_*_data.tex` | Appendix D, Figures D.1–D.5 | < 1 s |
| `_povm_cards_controls.py` | nothing — 100 negative controls | — | 19 s |
| `numpy_atlas.py` | nothing — it checks (see below) | — | < 1 s |
| `phi_simulation_cost.py` | nothing — it checks | Appendix B.3 | < 1 s |
| `dial_settings.py` | nothing — it checks | Appendix E.2 | 17 s |
| `phi_star_copies.py` | nothing — it checks | — | 2 s |
| `weight_obstruction_escapes.py` | nothing — it checks | Appendix D.1 | 3 s |

The synthesis histograms Appendix B's lede points at come from `bpg-walkthrough.ipynb`
(`bpg_distributions.pdf`); they are not printed. The section-ladder
figure needs a LaTeX pass afterwards: `cd paper/figures && latexmk -pdf section_ladder.tex`.
`randomized_implementations.py` is the entry point of the eight-file `randomized_*` family —
run it, not the seven sibling modules that hold its bindings.

## How far the checking goes

**Every script is self-verifying.** Each asserts the claims it exists to support and exits
non-zero the moment one fails, so a clean exit *is* the result — nothing to eyeball.
Six of them write nothing at all and are pure checks.

**The main implementation is symbolic.** `main.py` and `povm_properties.py` work symbolically in
SymPy over ℚ(√2, √5, i); exact identity settles every comparison.

**A NumPy reimplementation exists.** `numpy_atlas.py` re-derives all three pillars — groups,
synthesis, POVMs — from scratch in plain numpy, no SymPy anywhere, then diffs value-for-value
against the symbolic output in `data/`. An independent reimplementation, and a way to read the
project without the symbolic layer.

**Full reproducibility.** A full sweep reproduces every tracked artifact identically —
`git status` stays silent, the seeded Monte-Carlo `.npz` and the matplotlib PDFs (which pin
their embedded creation date) included. A diff after a re-run is therefore always a real
regression and never noise — `everything.py`'s drift report ends by saying exactly that.

## Guided tours

Six notebooks in `code/`, one per script, each a narrated derivation with committed outputs — so
they read on GitHub without running anything. `uv run everything.py --notebooks` re-executes all
six in a throwaway copy of `code/`, leaving the committed outputs untouched; every walkthrough
ends in assertions, so here too a clean exit is the result.

| Notebook | Walks through |
| --- | --- |
| `bpg-walkthrough.ipynb` | the groups and their circuit synthesis (`main.py`) |
| `povm_walkthrough.ipynb` | the five POVMs, symbolically (`povm_properties.py`) |
| `numpy_walkthrough.ipynb` | the same three pillars in numpy (`numpy_atlas.py`) |
| `phi_walkthrough.ipynb` | what the golden gate Φ costs to simulate |
| `randomization_walkthrough.ipynb` | the two randomized protocols, with zero RNG |
| `shadow_walkthrough.ipynb` | the shadow-estimation study, end to end |

All but `bpg-walkthrough.ipynb` are generated by a `_build_*.py` beside them — **edit the
builder, never the notebook.** The same holds for every `.tex` fragment in `code/data/`:
regeneration overwrites manual edits, captions included.

## Using the atlas data

`code/data/` holds the numbers as `.npz`, and `everything.npz` bundles nearly all of it under prefixed keys (`export_numpy.py`'s manifest is the exact map).

```python
import numpy as np
d = np.load('code/data/group_2I.npz', allow_pickle=True)

d['unitaries']        # (120, 2, 2) complex — the group itself
d['bfs_sequences']    # (120,) str — shortest circuit for each element
d['dij_sequences']    # (120,) str — cheapest circuit, counting only Φ gates
```

Four synthesis strategies are stored in parallel, the cross of {shortest, cheapest} × {exact in
SU(2), up to global phase}: prefixes `bfs_`, `dij_`, `u2_`, `dij_u2_`, each with `_sequences`
and `_depths`, plus `_magic_costs` — the code's name throughout for what the thesis prints as
the *Φ-count* — and `_phases` where those apply. The `u2_` ones are what you
would actually run — global phase is unphysical — and are never longer than their SU(2)
counterparts.

Alongside: `group_{T,O,I}.npz` (the rotation groups as SO(3) matrices), `gates.npz` (the six
gates X, Z, H, S, F, Φ, their magic costs, and the gate set used per group), and
`povm_{tetrahedron,octahedron,cube,icosahedron,dodecahedron}.npz` (Bloch vertices, POVM
elements, and the shadow reconstruction coefficients *a*, *b* with ρ̂ = *a*·E_k + *b*·I).
`d.files` lists the keys of any of them.

## License and citation

Code, notebooks and data are MIT-licensed; the thesis, its figures and the exposé are CC BY 4.0 (see `NOTICE`). To cite, use `CITATION.cff` or GitHub's "Cite this repository" button.
