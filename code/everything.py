"""
Run everything, in one command:

    cd code && uv run everything.py                # the fifteen scripts
    cd code && uv run everything.py --notebooks    # ... plus the six walkthroughs

Each script runs as its own process, cwd = code/, in dependency order (the
README's table, then the pure checks); every one is self-verifying, so the
sweep stops at the first non-zero exit and a clean sweep IS the result.
Afterwards: a per-step timing table, then an artifact-drift report -- the
delta in `git status` over code/ and paper/figures/ across the run.  On an
untouched tree the delta is empty, because the repo re-derives itself
byte-identically; after a generator edit it names exactly the artifacts that
moved.  (One artifact needs a step git cannot see: when section_ladder.tex
drifts, its PDF wants `cd paper/figures && latexmk -pdf section_ladder.tex`.)

--notebooks executes the six committed walkthroughs in a throwaway copy of
code/, so their committed outputs are never rewritten (nbconvert's stdout
chunking is nondeterministic -- an in-tree execute would churn them).  Exit 0
is each walkthrough's whole contract: they all end in assertions, and the
generated ones hash-pin their source modules, so a notebook left stale by a
module edit fails its execute here and names the rebuild it wants.

The roster is a claim about the directory, and the claim is checked: every
*.py here must be a roster entry, a randomized_* binding (run through its
entry point), a _build_* generator, or this file.  A new script that fits no
class fails the sweep until it is placed -- nothing runs "everything" minus
a forgotten piece.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent

# Dependency order: writers first (later entries read data/*.npz written by
# earlier ones), then the write-nothing checks, which may import anything.
ROSTER = [
    "main.py",                       # atlas.txt, atlas.tex, differing_rows.tex -> App. A, App. B
    "povm_properties.py",            # povm_atlas.tex, povm_properties.tex -> App. E; povm_angle_multisets.tex (repository only)
    "export_numpy.py",               # data/*.npz                      (feeds everything below)
    "randomized_implementations.py", # data/randomized_*.tex           -> 5.2.3, App. D, F.3
    "shadow_experiments.py",         # data/shadow_experiments.npz     (feeds the report)
    "shadow_report.py",              # shadow_gatenoise.tex -> App. F.3.4 (+2 tables, 3 figures, repo only)
    "_build_section_ladder.py",      # paper/figures/section_ladder.tex -> App. C
    "_build_recipe_figure.py",       # data/recipe_tet_data.tex        -> Fig. 1.1
    "_build_povm_cards.py",          # data/card_*_data.tex            -> App. D cards
    "_povm_cards_controls.py",       # nothing -- the cards' 100 negative controls
    "numpy_atlas.py",                # nothing -- the numpy re-derivation, diffed
    "phi_simulation_cost.py",        # nothing -- mu(Phi) and the third level
    "dial_settings.py",              # nothing -- App. E.2's three dials
    "phi_star_copies.py",            # nothing -- the anchor's counts, Phi* and 2I's two copies
    "weight_obstruction_escapes.py", # nothing -- App. D.1's escapes and claims
]

NOTEBOOKS = [
    "bpg-walkthrough.ipynb",
    "povm_walkthrough.ipynb",
    "numpy_walkthrough.ipynb",
    "phi_walkthrough.ipynb",
    "randomization_walkthrough.ipynb",
    "shadow_walkthrough.ipynb",
]

BINDINGS = {f"randomized_{m}.py" for m in
            ("core", "field", "scalars", "twojobs", "obstruction",
             "decker", "fragments")}

COPY_SKIP = {".venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache",
             ".ruff_cache"}


def check_roster():
    missing = [s for s in ROSTER if not (CODE / s).exists()]
    if missing:
        sys.exit(f"roster names scripts that do not exist: {missing}")
    stray = [p.name for p in sorted(CODE.glob("*.py"))
             if p.name not in ROSTER
             and p.name not in BINDINGS
             and not p.name.startswith("_build_")
             and p.name != Path(__file__).name]
    if stray:
        sys.exit(f"unclassified script(s): {stray} -- everything.py runs "
                 "everything, so place each in ROSTER, BINDINGS, or _build_*")


def git_status():
    r = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--",
         "code", "paper/figures"],
        capture_output=True, text=True)
    return set(r.stdout.splitlines()) if r.returncode == 0 else None


def run_step(argv, label, cwd, timings):
    print(f"\n=== {label} " + "=" * max(0, 70 - len(label)), flush=True)
    t0 = time.monotonic()
    r = subprocess.run(argv, cwd=cwd)
    dt = time.monotonic() - t0
    timings.append((label, dt))
    if r.returncode != 0:
        print(f"\nFAILED after {dt:.0f}s: {label} (exit {r.returncode}) -- "
              "sweep stopped, later steps not run")
        sys.exit(r.returncode)


def run_notebooks(timings):
    tmp = Path(tempfile.mkdtemp(prefix="everything-nb-"))
    copy = tmp / "code"
    shutil.copytree(CODE, copy, ignore=shutil.ignore_patterns(*COPY_SKIP))
    print(f"\nnotebooks execute out of tree: {copy}")
    for nb in NOTEBOOKS:
        run_step([sys.executable, "-m", "nbconvert", "--to", "notebook",
                  "--execute", "--inplace", nb],
                 nb, copy, timings)
    shutil.rmtree(tmp)  # kept for inspection on failure (run_step exits first)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--notebooks", action="store_true",
                    help="also execute the six walkthroughs, out of tree")
    args = ap.parse_args()

    check_roster()
    before = git_status()
    t0 = time.monotonic()
    timings = []

    for script in ROSTER:
        run_step([sys.executable, script], script, CODE, timings)
    if args.notebooks:
        run_notebooks(timings)

    total = time.monotonic() - t0
    width = max(len(label) for label, _ in timings)
    print("\n" + "-" * (width + 12))
    for label, dt in timings:
        print(f"  {label:<{width}}  {int(dt) // 60}:{int(dt) % 60:02d}")
    print(f"  {'total':<{width}}  {int(total) // 60}:{int(total) % 60:02d}")

    after = git_status()
    if before is None or after is None:
        print("\nartifact drift: git unavailable, not checked")
    elif after - before:
        print("\nartifact drift (git status lines new since the sweep began):")
        for line in sorted(after - before):
            print(f"  {line}")
        if any("section_ladder.tex" in line for line in after - before):
            print("  -> section_ladder.tex drifted: recompile it with "
                  "`cd paper/figures && latexmk -pdf section_ladder.tex`")
    else:
        print("\nno artifact drift -- the sweep reproduced every tracked "
              "artifact byte-identically")


if __name__ == "__main__":
    main()
