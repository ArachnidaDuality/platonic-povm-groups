"""Builds code/shadow_walkthrough.ipynb from clean inline source.

Run with `cd code && uv run python _build_shadow_walkthrough.py`.

This script is the source of truth for the notebook. To edit a markdown
cell, modify the corresponding `md(r'''...''')` call below, then re-run this
script to regenerate the .ipynb. Do not edit the notebook directly --
regeneration will overwrite manual edits.

The notebook is the walkthrough for `code/shadow_experiments.py`, the
numerical study behind Section 5.2 and Appendix F.3 (only the two-protocol
study is in print; the rest is repository-only). Like the randomization
builder, code cells are not hand-copied: `lift()` and `lift_assign()`
extract function sources and module-level constants via `ast` at build
time, so the cells are verbatim by construction and regeneration tracks
module edits automatically. The notebook is fully self-contained -- it
re-runs the entire study by threading ONE shared RNG in the production
order (make_states -> exp1 -> exp2 -> exp3 -> blindness; everything else
is deterministic), while demo cells use local RNGs so the stream stays
pristine. The final cell rebuilds main()'s npz dict and diffs it against
the committed data/shadow_experiments.npz key for key, value for value
(max |diff| = 0). The notebook never writes the npz -- the script does.

Regenerating is only half the job: the builder emits empty outputs, so the
re-execute below is mandatory to keep the committed outputs current. Both
commands run from code/, and the notebook's own closing notes give the same
pair:

    uv run --with jupyter --with nbconvert jupyter nbconvert \
        --to notebook --execute --inplace shadow_walkthrough.ipynb
"""

import ast
from pathlib import Path

import nbformat

HERE = Path(__file__).parent
SRC = (HERE / "shadow_experiments.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)
LINES = SRC.splitlines()


def _func(name):
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(f"no module-level function {name!r}")


def lift(*names):
    """Verbatim source of module-level function defs, in the given order."""
    chunks = []
    for name in names:
        node = _func(name)
        chunks.append("\n".join(LINES[node.lineno - 1:node.end_lineno]))
    return "\n\n\n".join(chunks)


def lift_assign(*names):
    """Verbatim module-level assignments (with their leading comments),
    in module order."""
    chunks = []
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) in names for t in node.targets):
            start = node.lineno - 1
            while start - 1 >= 0 and LINES[start - 1].lstrip().startswith("#"):
                start -= 1
            chunks.append("\n".join(LINES[start:node.end_lineno]))
    return "\n\n".join(chunks)


nb = nbformat.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbformat.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbformat.v4.new_code_cell(text.strip("\n")))


# =============================================================================
# Cell 1 -- Title + intro
# =============================================================================
md(r'''
# Shadow Estimation with the Platonic-Solid POVMs — Walkthrough

A companion walkthrough for `code/shadow_experiments.py`, the numerical study behind Section 5.2
and Appendix F.3 of the thesis (only the two-protocol study is in print; the rest is
repository-only). A reader who works through this notebook instead of the script misses out on
**nothing**: every function of the module appears below verbatim (the builder lifts them
mechanically from the source at build time), every experiment is re-run in full, and the final
cell proves the point by rebuilding the study's entire output dictionary and diffing it against
the committed `data/shadow_experiments.npz` — key for key, value for value, to exactly zero. The
notebook writes nothing; the script owns the npz.

The whole appendix hangs off one skeleton:

> **1 lemma, 2 conjugations, 3 corners, 5 studies.**

One corollary of Schur's lemma. Two things a random rotation can wrap into a group average — the
readout *composed with* the noise (randomized-projective, scalar $T_{zz}$), or the noise *alone*
(twirled-native, scalar $\operatorname{tr}T/3$). Three implementation corners (native /
twirled-native / randomized-projective). Five numbered studies, one job each: the variance
landscape, the dual-frame optimization, the robust calibration, its blind spot, and the scaling
check. Hold the skeleton and everything else here is a corollary;
hold only the numbers and you hold nothing.

**References:**
- Huang, Kueng & Preskill, *Predicting many properties of a quantum system from very few measurements* (2020) — classical shadows, the $3^w$ sampling cost
- Nguyen, Bönsel, Steinberg & Gühne, *Optimising shadow tomography with generalised measurements* (2022) — Platonic-solid POVM shadows, the closest adjacent work
- Chen, Yu, Zeng & Flammia, *Robust shadow estimation* (2021) — calibration under a measurement twirl; their random-Clifford primitive is our octahedral protocol
- Innocenti et al., *Shadow tomography on general measurement frames* (2023) — the closed-form variance for 3-design measurements this study's Experiment 1 extends
- D'Ariano, Perinotti & Sacchi (2005) — the tetrahedral SIC as an *indecomposable* POVM
- Korhonen et al. (2025) — locally-optimal duals losing on composite observables (our 1.069 cell, arrived at uninvited)
- Caprotti et al. (2026); Mangini et al. (2025) — dual-optimization bounds, and what leaving the factorized class buys
- Wilkens et al. (2026) — local robust shadows run on trapped-ion hardware
- Brieger et al. (2025) — silent failure of miscalibrated robust estimators
- Jeanette et al. (2026) — blind calibration, the other exit from the blind spot
- Decker, Janzing & Beth (2004) — the native (Naimark) measurement circuits
- Elben et al., *The randomized measurement toolbox* (2022) — randomized-projective's literature home
''')


# =============================================================================
# Cell 2 -- The reframing + the map
# =============================================================================
md(r'''
## 0. One reframing, and the map

**A POVM is its Bloch-vertex array `s`** of shape $(V, 3)$. Everything in the study is a small
linear-algebra statement about `s`:

- *effects*: $E_k = \frac1V(\mathrm{Id} + \hat n_k\cdot\vec\sigma)$ — the array row-for-row;
- *probabilities*: $p(k) = \frac1V(1 + \hat n_k\cdot r)$ — affine in the state's Bloch vector;
- *dual frames*: pairs $(\alpha_k, \beta_k)$ with $D_k = \alpha_k\,\mathrm{Id} + \beta_k\cdot\vec\sigma$
  — unbiasedness is 16 linear equations in the $4V$ coordinates, variance a quadratic form, so
  optimizing the dual is a linearly-constrained quadratic program;
- *noise*: an affine Bloch map $r \to Tr + t$, pushed onto the effects in the Heisenberg picture;
- *$n$-qubit Born distributions*: one `einsum` of $\rho$ against per-wire effect tensors — so
  single-shot means and second moments are **finite sums evaluated exactly**, and every
  deterministic claim (unbiasedness, variance identities, bias factors, twirl scalars) is a
  theorem checked to machine precision, with Monte Carlo only measuring convergence *to targets
  already known*;
- *gate circuits*: affine Bloch maps too — so each randomized implementation's effective channel is
  an exact finite group average, no sampling anywhere.

| § | study | its one job |
|---|---|---|
| 3 | Experiment 1 + the exact landscape | does the choice of solid matter at the canonical dual? (provably: no) |
| 4 | the fourth-moment ladder | where the solid choice *does* reappear: the tails |
| 5 | Experiment 2 + exact ratios | what the $(4V{-}16)$-dimensional dual family buys (little, and not for the reason you'd guess) |
| 6 | Experiment 3 | robust calibration: the exact identity, and what the insurance premium costs |
| 7 | the blind spot + free diagnostic | where a $\ket{0}$-calibration is structurally blind, and the diagnostic hiding in its discards |
| 8 | the twirl + the two protocols | one lemma, two conjugations: $T_{zz}$ versus $\operatorname{tr}T/3$ |
| 9 | gate noise | the twirl's one assumption, priced with the atlas — and *why* the twirled-native residual is second order |
| 10 | $n$-scaling | nothing above is an $n=4$ artifact |
| 11 | the receipts | this notebook = the committed npz, exactly |

**The random-number discipline.** One shared `rng` is created in §2 and consumed *strictly in the
production order* of the script's `main()` — `make_states` → Experiment 1 → Experiment 2 →
Experiment 3 → blindness — and never touched anywhere else; every demonstration cell uses its own
local generator. That discipline is what makes §11's value-for-value replay possible. Everything
from §8 onward is deterministic (exact group averages; the two local generators inside
`twirl_check` and `two_protocol_twirl` are fixed-seed irreducibility witnesses, not Monte Carlo).
''')


# =============================================================================
# Cell 3 -- Setup: imports + constants
# =============================================================================
code(
    "# === Setup (lifted from shadow_experiments.py): imports + constants ===\n\n"
    "import numpy as np\n"
    "from pathlib import Path\n"
    "from scipy.sparse import coo_matrix\n"
    "from scipy.sparse.linalg import eigsh\n"
    "\n"
    "DATA = Path(\"data\")   # the one adaptation: the module resolves this "
    "next to itself\n\n"
    + lift_assign("RNG_SEED", "N_QUBITS", "R_REPS", "T_MAIN", "T_EXP2",
                  "RC_CAL", "N_HAAR", "G_CRIT", "TRAIN_G", "DEPOL_RATES",
                  "BLIND_RATES", "GATE_GAMMAS", "GAMMA_DIL", "INDEP_DELTAS",
                  "NSCALE_NS",
                  "SOLIDS", "ANTIPODAL", "I2", "PAULI", "AXIS")
)


# =============================================================================
# Cells 4-6 -- the POVM data
# =============================================================================
md(r'''
## 1. The five solids, as arrays

The vertices and effects come from the thesis's own symbolic exports (`data/povm_*.npz`, written
by `export_numpy.py` from `povm_properties.py`'s symbolic vertex orderings, in the atlas
orientation — the one where $\Phi$'s rotation axis lands *on* an icosahedron vertex). The loader
pins down the four facts that carry the whole study: every vertex is a unit vector, the vertex
sum is **zero** (the solid is centered), the vertex covariance is **isotropic**,
$\sum_k \hat n_k \hat n_k^\top = \frac{V}{3}\,\mathrm{Id}$ — the 2-design property — and effect
$E_k$ is built from vertex $\hat n_k$ at the *same index* $k$, which is the join every estimator
below relies on and the only one of the four a permutation of the rows can break. Completeness
of the effects is checked alongside.
''')

code(
    "# === POVM data (Bloch vertices + effects from the symbolic exports) ===\n\n"
    + lift("load_povms")
    + "\n\n\npovms = load_povms()"
)

code(r'''
# === Demo: the whole estimation story in three matvecs (local rng only) ===

rng_demo = np.random.default_rng(6)
s = povms["icosahedron"]["s"]
print("centered:  max|sum of vertices| =", np.abs(s.sum(axis=0)).max())
print("2-design:  max|s^T s - (V/3) Id| =",
      np.abs(s.T @ s - (len(s) / 3) * np.eye(3)).max())

r = rng_demo.standard_normal(3)
r *= rng_demo.random() / np.linalg.norm(r)      # a random state in the ball
p = (1 + s @ r) / len(s)                        # Born probabilities: affine in r
assert abs(p.sum() - 1) < 1e-12 and p.min() > 0
print("\nE[sampled vertex] =", np.round(p @ s, 8))
print("            r / 3 =", np.round(r / 3, 8))
print("\nthe 1/3 shrinkage is the 2-design identity at work; tripling the "
      "sampled vertex\nundoes it -- that rescaling IS the canonical dual, "
      "and it is unbiased by geometry")
''')


# =============================================================================
# Cells 7-9 -- states, observables, the stream
# =============================================================================
md(r'''
## 2. States, observables, and the shared stream

Four qubits throughout. The observables: single Paulis $Z_0, X_0$, the strings $Z_0Z_1$,
$X_0X_1$, $Z_0Z_1Z_2$, and the critical TFIM energy
$H = -\sum_i Z_iZ_{i+1} - h\sum_i X_i$ at $h = 1$. The states: the TFIM ground state, GHZ, a
generic product state, and ten Haar-random states shared across all five POVMs. Everything any
estimator reports is checked against exact values computed from the dense operators.

The next cell defines the machinery; the one after it starts **the shared Monte Carlo stream**
(seed `20260612`). From here on, that stream is consumed only by the five marked production
cells, in order.
''')

code(
    "# === Observables and states ===\n\n"
    + lift("kron_all", "term_operator", "obs_operator", "exact_value",
           "tfim_terms")
    + "\n\n\n"
    + lift_assign("OBSERVABLES")
    + "\n\n\n"
    + lift("tfim_ground_state", "density", "make_states", "site_bloch")
)

code(r'''
# === THE SHARED STREAM -- created once, consumed strictly in production order ===
# (make_states -> exp1 -> exp2 -> exp3 -> blindness; demos use local rngs.)

rng = np.random.default_rng(RNG_SEED)
states, haar = make_states(rng)                                # [stream 1/5]

print(f"TFIM h=1 ground energy: "
      f"{exact_value(states['TFIM'], OBSERVABLES['E_TFIM']):.6f}")
print(f"TFIM site-0 Bloch vector r0 = "
      f"{np.round(site_bloch(states['TFIM'])[0], 5)}   "
      "(the gate-noise study's test vector)")
''')


# =============================================================================
# Cells 10-17 -- machinery: noise, Born, duals, estimators
# =============================================================================
md(r'''
## 2a. Noise, Born tensors, duals, estimators

**Noise** is an affine Bloch map $r \to Tr + t$ applied i.i.d. per qubit just before the
measurement. Four channels appear: depolarizing ($T = (1-p)\,\mathrm{Id}$), dephasing
($T = \mathrm{diag}(1{-}2p, 1{-}2p, 1)$), amplitude damping
($T = \mathrm{diag}(\sqrt{1{-}\gamma}, \sqrt{1{-}\gamma}, 1{-}\gamma)$, $t = \gamma\hat z$ — the
only one with a shift), and a *tilted* depolarizing channel for the diagnostic of §7. In the
Heisenberg picture the noise moves onto the effects:
$\tilde E_k = \frac1V[(1 + \hat n_k\cdot t)\,\mathrm{Id} + (T^\top \hat n_k)\cdot\vec\sigma]$.

**Born distributions.** The $n$-qubit outcome distribution of the product POVM is an explicit
$(V,V,V,V)$ tensor — one `einsum`. That tensor is the study's method in miniature: any
*single-shot* moment of any estimator is a finite sum over it, evaluated term by term. Sampling
(`sample_outcomes`) exists only to measure convergence speed.
''')

code(
    "# === Noise channels: affine Bloch maps, applied to the effects ===\n\n"
    + lift("chan_depolarizing", "chan_dephasing", "chan_amp_damping",
           "chan_tilted_depolarizing", "noisy_effects")
)

code(
    "# === Born distributions and sampling ===\n\n"
    + lift("born_tensor", "sample_outcomes", "single_qubit_probs")
)

md(r'''
### The dual family: 16 equations in $4V$ unknowns

A single-qubit dual frame is $D_k = \alpha_k\,\mathrm{Id} + \beta_k\cdot\vec\sigma$: four numbers
per outcome, $4V$ in all. Unbiasedness — the frame condition
$\sum_k \operatorname{tr}[E_k\rho]\,D_k = \rho$ for every $\rho$ — is a $4\times4$ reconstruction
map set entry-by-entry to the identity: **16 linear equations** (`frame_system`). What remains is
an affine family of valid duals of dimension $4V - 16$: a *point* for the tetrahedron
($V = d^2 = 4$, the control), $8/16/32/64$ dimensions for the four overcomplete solids. The
canonical dual $\alpha = \frac12$, $\beta = \frac32\hat n_k$ is one member; `optimize_dual` finds
the variance-minimizing member for one Pauli letter by solving the quadratic program — since the
mean is pinned by the constraints, minimizing the second moment *is* minimizing the variance, and
the QP reduces to a linear solve in the null space of the constraint matrix.
''')

code(
    "# === Dual frames: canonical, robust, and the variance-minimizing QP ===\n\n"
    + lift("canonical_dual", "robust_canonical_dual", "frame_system",
           "optimize_dual", "dual_frame_residual")
)

code(r'''
# === Demo: how big is each dual family? (deterministic) ===

for name, povm in povms.items():
    s, V = povm["s"], povm["V"]
    A, b = frame_system(s)
    rank = np.linalg.matrix_rank(A, tol=1e-10)
    alpha, beta = canonical_dual(s)
    v_can = np.concatenate([alpha, beta.ravel()])
    assert np.abs(A @ v_can - b).max() < 1e-12   # the canonical dual qualifies
    print(f"  {name:12s} V = {V:2d}:  4V = {4 * V:3d} coordinates,  "
          f"rank(A) = {rank},  free dimensions = {4 * V - rank}")
print("\nthe tetrahedron's family is a single point (unique dual, the control);"
      "\nExperiment 2 searches the others")
''')

md(r'''
### Estimator conventions

A *dual assignment* is a lookup table `lut[i, a, k]`: the per-shot value site $i$ contributes when
reading Pauli letter $a$ off outcome $k$. With the canonical dual that value is
$3(\hat n_k)_a$; identity sites contribute exactly $1$ (every dual here keeps
$\operatorname{tr}D_k = 1$), so they fall away for free. A composite observable is estimated
term by term, each term multiplying its sites' letter-values — and `exact_estimator_mean` /
`exact_second_moment` evaluate the same contractions *exactly* over the outcome tensor, which is
what turns every unbiasedness and variance claim below into a deterministic assert.

One deliberate convention, against the more obvious alternative: duals are pure classical
post-processing, so each term of a composite observable uses the dual matched to *its own letter at
each site*, rather than routing each qubit to a single dual for all terms (that of its most
frequent letter). Single-term observables coincide either way; the TFIM energy does not.
Unbiasedness is unaffected and the optimized levels can only improve.
''')

code(
    "# === Dual assignments (LUTs) and estimator evaluation ===\n\n"
    + lift("lut_uniform", "lut_canonical", "lut_robust_canonical",
           "lut_from_letter_duals", "shot_estimates", "exact_estimator_mean",
           "exact_second_moment", "rep_stats")
)


# =============================================================================
# Cells 18-20 -- Experiment 1
# =============================================================================
md(r'''
## 3. Experiment 1 — the canonical dual is blind to the choice of solid

**The job:** measure fixed states with all five solids at the canonical dual — no noise, no
optimization — and ask whether the choice of solid matters. The table invites a ranking; the
theorem says there is none to be had.

**The theorem (the site-cases identity).** Expand the estimator's second moment over ordered pairs
of Pauli strings; each pair contributes $\operatorname{tr}[\rho\bigotimes_i M_i]$ with one
single-qubit operator per site, fixed by which strings touch it:

$$M_i = \begin{cases}
\mathrm{Id} & \text{neither string touches site } i,\\[2pt]
\sum_k 3(\hat n_k)_a E_k = \sigma_a & \text{one does, with letter } a,\\[2pt]
\sum_k 9(\hat n_k)_a(\hat n_k)_b E_k = 3\delta_{ab}\,\mathrm{Id} + 9\,S_{abc}\,\sigma_c
  & \text{both do, with letters } a, b,
\end{cases}$$

where $S_{abc} = \frac1V\sum_k (\hat n_k)_a(\hat n_k)_b(\hat n_k)_c$ is the vertex set's
third-moment tensor. The middle case is the frame condition read backwards — it holds for *every*
valid dual, which §10 will lean on. So the vertices enter the variance through their moments of
order **at most three**: order 1 vanishes (centered), order 2 is universal (2-design), and $S$
vanishes for every antipodal vertex set. Consequences, all asserted exactly:

- octahedron, cube, icosahedron, dodecahedron: **identical variance on every (state, observable)**;
- a lone weight-$w$ Pauli string sits at $\operatorname{Var} = 3^w - \langle P\rangle^2$ exactly,
  for *all five* solids (the tetrahedron's $(\hat n)_a^2 = \frac13$ kills its same-letter third
  moments) — the familiar $3^w$ Pauli-shadow cost, here an identity rather than a bound, owned by
  the whole family;
- the tetrahedron can deviate **only** where all three conditions meet: a term pair sharing a site
  with two *different* letters (among our observables: only the TFIM energy, where $ZZ$ meets
  $X$), a state whose amplitudes are not all real (the surviving operator carries a lone
  $\sigma_y$ — so not TFIM or GHZ, only the Haar draws), and a nonvanishing $S$ (only the
  tetrahedron). One cell of the whole table.

**Counterfactual** (worth holding): a vertex set that is a 2-design but *not centered* breaks the
middle case — $\sum_k 3(\hat n_k)_a E_k$ picks up a multiple of the identity, so the canonical
formula is no longer unbiased, and the $\ket{0}$-calibration of §6 reads
$\bar n_z + \frac13 \neq \frac13$ *on a clean channel*. Centering is not decoration; it is load-bearing.
''')

code(
    "# === Experiment 1: the Monte Carlo landscape ===\n\n"
    + lift("experiment_1")
    + "\n\n\nexp1, exp1_haar = experiment_1(povms, states, haar, rng)"
      "        # [stream 2/5]"
)

code(
    "# === The exact landscape: the theorem the Monte Carlo illustrates ===\n\n"
    + lift_assign("K_BODY")
    + "\n\n\n"
    + lift("exact_backbone")
    + "\n\n\nvar_exact, exact_out = exact_backbone(povms, states, haar, "
      "exp1, exp1_haar)"
)


# =============================================================================
# Cells 21-23 -- Fourth moments
# =============================================================================
md(r'''
## 4. The fourth-moment ladder — where the solid choice reappears

One moment order up, the landscape stops being flat. For a single Pauli letter the single-shot
estimate is $3(\hat n_k)_a$, so
$\mathbb E[\hat o^4] = 81\cdot\frac1V\sum_k(\hat n_k)_a^4$ — and the state-dependent part is a
*fifth*-order odd moment, killed by antipodality (or by $(\hat n)_a^4 = \frac19$ for the
tetrahedron). So the fourth moment is **state-independent** and genuinely solid-dependent:

$$\mathbb E[\hat o^4] = 9 \;(\text{tetrahedron, cube}), \qquad 27 \;(\text{octahedron}), \qquad
\tfrac{81}{5} = 16.2 \;(\text{icosahedron, dodecahedron}).$$

The 5-designs sit *pinned at the uniform-sphere value in any orientation* — design strength fixes
the tails at the sphere's own weight; it does not minimize them. The extremes are worth feeling
physically: the cube's vertices all have $|(\hat n)_a| = \frac1{\sqrt3}$, so measuring a letter
returns $\pm\sqrt3$ *always* — the lightest possible tails at variance 3 — while the octahedron
(= standard Pauli-6 shadows) returns $0$ with probability exactly $\frac23$ **on any state** (its
equator is centered) and a $\pm3$ spike otherwise. Same mean, same variance, very different tails
— a reason to prefer the cube that no variance table can show, and the demo after the check makes
the distributions explicit.
''')

code(
    "# === The ladder, exactly ===\n\n"
    + lift_assign("FOURTH_EXPECTED")
    + "\n\n\n"
    + lift("fourth_moment_ladder")
    + "\n\n\nfour_out = fourth_moment_ladder(povms, states, haar)"
)

code(r'''
# === Demo: same variance, different tails (local rng only) ===

rng_demo = np.random.default_rng(23)
r = rng_demo.standard_normal(3)
r *= rng_demo.random() / np.linalg.norm(r)          # a random state in the ball
for name in ("cube", "octahedron", "icosahedron"):
    s = povms[name]["s"]
    q = (1 + s @ r) / len(s)                        # Born probabilities
    vals = 3.0 * s[:, 2]                            # per-shot values, letter Z
    uq, inv = np.unique(np.round(vals, 9), return_inverse=True)
    probs = np.bincount(inv, weights=q)
    dist = ",  ".join(f"{u:+.3f} w.p. {p:.4f}" for u, p in zip(uq, probs))
    print(f"  {name:12s} {dist}")
    print(f"  {'':12s} E[o] = {q @ vals:+.4f}   "
          f"Var = {q @ vals ** 2 - (q @ vals) ** 2:.4f}   "
          f"E[o^4] = {q @ vals ** 4:.4f}")
print("\nthe octahedron's 0 has probability exactly 2/3 on ANY state "
      "(its equator sums to zero);\nthe cube never leaves +-sqrt(3)")
''')


# =============================================================================
# Cells 24-27 -- Experiment 2
# =============================================================================
md(r'''
## 5. Experiment 2 — optimizing the dual

**The job:** the canonical dual wasted none of the solids' redundancy — but the $(4V-16)$-dimensional
dual family is still a resource. Minimize the variance over it and see what overcompleteness buys.

The optimization runs **per Pauli letter**: the estimator only ever reads one letter per site, its
mean is pinned by the frame condition, so its second moment — a quadratic form in the dual's
coordinates, weighted by the outcome distribution — is the whole objective. The training
distribution enters that objective *only through the mean Bloch vector* of its reduced states
(the objective is linear in the distribution); for the TFIM sweep $h \in \{0.3,\dots,1.5\}$ plus
GHZ that mean points along $x$. Three levels:

- **canonical** — the baseline;
- **observable-optimized** — trained on the TFIM+GHZ distribution; a deployable protocol;
- **oracle** — handed the true reduced state of each site, per site. A *ceiling, not a protocol*:
  it can overfit the dual to the state being measured (watch the product-state $X_0$ cell reach
  exactly zero variance — a degenerate dual perfectly aligned with a pure $\ket{+}$ site).

Two sanity anchors before trusting any gains: feeding the QP the *uniform* distribution returns
the canonical dual for every solid (Haar-optimality — so gains live entirely in the prior), and
the tetrahedron's ratios are $1$ identically (nowhere to move).
''')

code(
    "# === Experiment 2: training, per-letter duals, Monte Carlo ===\n\n"
    + lift("training_bloch_mean", "letter_duals", "oracle_lut")
    + "\n\n\n"
    + lift("experiment_2")
    + "\n\n\nexp2 = experiment_2(povms, states, rng)        # [stream 3/5]"
)

md(r'''
### The exact ratios, and the two cells worth staring at

The trained duals are deterministic functions of the training data, so every table entry is an
**exact** variance ratio via the outcome tensors; the Monte Carlo is merely asserted to agree
within five standard errors.

**The gains do not scale with the dual dimension.** On the in-distribution energy the *cube*
(16 free parameters) beats both 5-designs — the icosahedron (32) and the dodecahedron (64) — at
both levels. A quarter to a third off, at best, from redundancy that grew eightfold: the dual
family is a rapidly saturating resource.

**The backfire cell.** The octahedron's per-letter *oracle* on the TFIM energy lands at ratio
$1.069$ — *worse than canonical*, exactly. No contradiction: the per-letter objective minimizes
each letter's own second moment, but a composite observable's variance also carries **cross-term
covariances the objective never sees**, and on the octahedron the letter-optimal duals land on
the wrong side of them. Korhonen et al. construct two-qubit instances of exactly this; in the
Platonic family it arrives uninvited, as a computed number.

*Orientation note.* These gains are properties of the atlas orientation (they depend on how the
solid sits relative to the states being measured): re-pose a solid and its number moves, the
icosahedron's $\sim$15% included. What is orientation-invariant is the structure: convergence
of the gain, the saturation of the family, and §10's class ceiling.
''')

code(
    "# === Exact Experiment-2 ratios (deterministic duals -> exact table) ===\n\n"
    + lift("exact_exp2_ratios")
    + "\n\n\nratio_out = exact_exp2_ratios(povms, states, exp2)"
)


# =============================================================================
# Cells 28-29 -- Experiment 3
# =============================================================================
md(r'''
## 6. Experiment 3 — robust calibration

**The job:** turn on per-qubit depolarizing noise at rate $p$ and run the robust protocol:
calibrate the effective shrinkage on $\ket{0}$, then invert $\hat\eta$ instead of $\frac13$.

**What the calibration learns is an exact identity.** On $\ket{0}$ the mean sampled vertex's
$z$-component is
$\mathbb E[(\hat n_k)_z] = \frac1V\sum_k (\hat n_k)_z + \frac{1-p}{V}\sum_k(\hat n_k)_z^2 =
0 + \frac{1-p}3$ — the zero-sum and covariance identities again. One matvec, asserted for every
rate. Equally deterministic is the damage when you *don't* calibrate: the noiseless dual on noisy
outcomes shrinks every weight-$w$ term by exactly $(1-p)^w$ — a bias, so **no number of shots
helps**.

**Calibration is insurance.** It converts that bias into variance. The premium is the sampling
error of $\hat\eta$ itself, which the division writes multiplicatively into every estimate — at
$p = 0$ the robust estimator pays about **60% extra MSE before there is any noise to correct**
(0.050 → 0.080 on the energy). The premium is generated by $\operatorname{Var}[\hat\eta] \propto
1/(4R_C)$ and amortizes accordingly; past the crossover near $p \approx 0.02$ the trade wins at
every rate. Re-optimizing the dual *on the calibrated effects* (same QP, now fed
$\hat\eta$-rescaled vertices) claws back a further $\sim$7% at $p = 0.1$ — optimization survives
noise, but shrinks.
''')

code(
    "# === Experiment 3: calibration + the estimator hierarchy under noise ===\n\n"
    + lift("calibrate")
    + "\n\n\n"
    + lift("experiment_3")
    + "\n\n\nexp3 = experiment_3(povms, states[\"TFIM\"], rng)"
      "        # [stream 4/5]"
)


# =============================================================================
# Cells 30-32 -- Blindness
# =============================================================================
md(r'''
## 7. The blind spot, and the free diagnostic

**The geometric one-liner:** the calibration probes the channel through a single state, so any
channel that **fixes $\ket{0}$** — dephasing and amplitude damping both do — is invisible to it:
$\hat\eta$ stays pinned at $\frac13$, reading "noiseless", and the robust estimator silently
reverts to the canonical one, reporting the *noisy* expectation values as though clean.

What it reports is itself exactly predictable: pushing the channel through the estimator, the
reported value of $\sigma_a$ on a site with Bloch vector $r$ is $(Tr + t)_a$, a residual bias of
$(Tr + t - r)_a$ — the cell asserts the measured biases sit on that prediction at every rate.
**Silence, not error, is the failure mode**: the wrong number ships under a clean confidence
interval, and no statistics on the estimation side can flag it.

**The free diagnostic.** The calibration shots already contain a full 3-vector: the mean sampled
vertex converges to $\hat v = \frac13(T\hat z + t)$, the calibrated image of the whole readout
axis. The protocol keeps its $z$-part ($\hat\eta$) and discards the rest — but the transverse
remainder $\hat v_\perp$ flags any channel that *tilts* the $z$-axis (a coherent misrotation
folded into readout, which is what drifting calibration produces): the tilted-depolarizing test
below reads $\hat\eta = 0.313$ — impersonating honest depolarizing noise at $p\approx0.06$ —
while $|\hat v_\perp| = 0.065$ busts it. The diagnostic's own blind spot: channels that fix the
whole $z$-axis *direction* — amplitude damping maps $\hat z \mapsto \hat z$ exactly, so it is
invisible to $\hat\eta$ **and** to $\hat v_\perp$. Narrower, at zero experimental cost; not
closed. Closing it structurally is §8's job.
''')

code(
    "# === Calibration blindness + the anisotropy diagnostic ===\n\n"
    + lift("blindness")
    + "\n\n\nblind, aniso = blindness(povms, states[\"TFIM\"], rng)"
      "        # [stream 5/5]"
)

md(r'''
*The shared stream is now retired.* Everything below — the twirl checks, the two-protocol
channels, the gate-noise study, the scaling study — is deterministic: exact finite group
averages and exact linear algebra, asserted at $10^{-12}$.
''')


# =============================================================================
# Cells 33-38 -- The twirl: one lemma, two conjugations
# =============================================================================
md(r'''
## 8. The twirl: one lemma, two conjugations

**The lemma (the only one).** Let a finite rotation group $G \subset SO(3)$ act irreducibly on
$\mathbb R^3$ — no invariant line. Then for any matrix $X$ and any vector $u$:

$$\frac1{|G|}\sum_{g\in G} R_g^\top X\, R_g = \frac{\operatorname{tr}X}{3}\,\mathrm{Id},
\qquad\qquad \frac1{|G|}\sum_{g\in G} R_g^\top u = 0.$$

(The average commutes with every $R_h$ — re-index the sum — so Schur forces a scalar; the average
preserves the trace, which fixes the scalar. The vector average spans an invariant subspace,
which must be $\{0\}$.) The tetrahedral, octahedral and icosahedral rotation groups all act
irreducibly — which is why a $T$-draw can twirl at all.

**The five-liner, twice.** A protocol's *estimator channel* is the affine map $r \mapsto Mr + m$
from the state's Bloch vector to the mean sampled snapshot vertex (ideal: $r/3$).

*Randomized-projective* — draw $g$, apply $U_g$, apply the fixed alignment $A$ (vertex $v$ nearest
$+\hat z$ → $\hat z$), let the noise $r \to Tr+t$ act, measure $Z$:
1. outcome $(g, b)$ post-processes into the snapshot $b\,R_g^\top v$;
2. its probability is $\frac12\bigl(1 + b\,[T(AR_g r) + t]_z\bigr)$ — **the same $g$ sits in the
   probability and in the snapshot**;
3. summing over $b = \pm1$: $M = \mathbb E_g\, R_g^\top (vw^\top) R_g$ with
   $w = A^\top T^\top\hat z$ — the twirled object is the rank-one **composition**
   $\hat z\hat z^\top T$ of readout projector and noise, conjugated into the solid's frame;
4. the lemma: $M = \frac{\operatorname{tr}[\hat z\hat z^\top T]}{3}\,\mathrm{Id} =
   \frac{T_{zz}}{3}\,\mathrm{Id}$ — the trace picks out the noise's entry **along the readout
   axis**;
5. the offset $m = (\hat z\cdot t)\,\mathbb E_g R_g^\top v = 0$ — the vector average dies by
   irreducibility.

*Twirled-native* — draw $g$, apply $U_g$, run the untouched Naimark circuit, relabel by $g$:
1. outcome $(g, k)$ post-processes into $R_g^\top \hat n_k$;
2. its probability is $\frac1V\bigl(1 + \hat n_k\cdot(TR_g r + t)\bigr)$ — the readout ranges over
   all $V$ effects **whatever the draw**: nothing downstream of the noise depends on $g$;
3. summing over $k$ first contracts the vertex covariance
   $\sum_k \hat n_k\hat n_k^\top = \frac V3\,\mathrm{Id}$, leaving
   $M = \mathbb E_g\, R_g^\top \frac{T}{3} R_g$ — the twirled object is **the noise alone**;
4. the lemma: $M = \frac{\operatorname{tr}T/3}{3}\,\mathrm{Id}$ — the isotropic mean, which has
   forgotten every axis;
5. the offset dies **twice over**, and the two deaths are different mathematical facts: its POVM
   part rides on $\sum_k \hat n_k = 0$ — the *solid* is centered — and its noise-shift part on
   $\mathbb E_g R_g^\top = 0$ — the *group* leaves no invariant vector.

Step 3 never used antipodality, so the twirled-native column exists for **all five solids — the
tetrahedral SIC included**. Decomposability is a requirement of the projective route only.

**The physicist's version (no formulas).** The two protocols differ in exactly one place: *who
decides the readout axis.* In randomized-projective the same coin that rotates the state also
decides — through the fixed $\hat z$ readout — which lab-frame direction of the noise gets probed:
draw and readout are perfectly correlated, the correlation sits *inside* the average, and what
survives is the noise as seen along the readout axis, $T_{zz}$. In twirled-native the readout is
the same fixed POVM every shot; there is nothing for the draw to correlate with, the noise is
seen whole, and only its isotropic part survives, $\operatorname{tr}T/3$. That correlation is not
a nuisance to be argued away — it *is* the difference between the protocols.
''')

md(r'''
### Boxes and wires: where the theorems reach

```text
native:                 rho ──[noise N]──[Naimark dilation]── outcome k
twirled-native:         rho ──[U_g]──[noise N]──[Naimark dilation]── k, relabel by g
randomized-projective:  rho ──[U_g]──[noise N]──[A]──[measure Z]── b, post-process by (g, b)
```

The theorems cover any noise $N$ that is **measurement-side** (between the drawn rotation and the
readout) and **independent of the draw** — for such $N$ the twirl is exact, whatever $N$ is. Not
covered: noise *inside the drawn word's own gates* (correlated with $g$ — that is §9, and Schur
has nothing to say about it), and state-preparation noise (that deforms $\rho$ itself: shadows
faithfully report the noisy state — it is not a measurement error at all). For twirled-native the
dilation's own noise is $g$-independent by construction — the dilation is the same circuit every
shot — so it sits inside the theorem's reach; the *model* of it (a pre-measurement channel on the
data qubit rather than an ancilla-side deformation) is a modeling choice, isolated in §9.
''')

code(
    "# === The twirl check: the projective channel is exactly depolarizing "
    "at T_zz/3 ===\n\n"
    + lift("align_to_z", "twirl_check")
    + "\n\n\ntwirl_check(povms)"
)

code(
    "# === The two protocols, side by side, on every solid and both draws ===\n\n"
    + lift_assign("COVARIANCE_GROUP", "T_PROBE", "t_PROBE")
    + "\n\n\n"
    + lift("channel_R1", "channel_R2", "orbit_hits")
    + "\n\n\n"
    + lift("two_protocol_twirl")
    + "\n\n\ntp_out = two_protocol_twirl(povms)"
)

md(r'''
### Reading the table

**The octahedron corner is the correlation story stripped bare.** Three separate facts conspire
there: $\hat z$ is already a vertex (so $A = \mathrm{Id}$), the native measurement is the uniform
mixture of the three Pauli bases (so no dilation), and the covariance group $O$ is exactly the
Bloch image of the single-qubit Clifford group $2O$. Both circuits collapse into "measure a
uniformly random Pauli" — standard randomized-Pauli shadows — *and the scalars still differ*:
$T_{zz} = 0.62$ against $\operatorname{tr}T/3 = 0.72$ on the shared probe. All that survives the
collapse is bookkeeping: whether the lab readout axis is fixed ($\hat z$, with the draw
re-orienting the state) or effectively ranges over the six vertices untied to the draw. Each
conspiring fact fails for every other solid: $\hat z$ is not a vertex of the cube, icosahedron or
dodecahedron (alignment nontrivial — and, by the field obstruction, *inexact*), and every other
solid's native circuit needs its dilation.

**The dodecahedron row splits the two jobs of the draw.** Under the minimal $T$-draw its channel
is *still exactly depolarizing* and the estimator unbiased — the twirl job never fails — but the
orbit covers only 6 of its 10 vertex axes: eight vertices are never sampled, so the draw
*realizes the wrong measurement*. Realization, not twirling, is what forces the dodecahedron up
to the full $2I$ draw (and its $0.8\,\Phi$ per shot, on the projective route alone).

**The blind spot does not survive either randomization.** Dephasing and amplitude damping — the
channels that pin the native calibration at $\frac13$ — twirl to $\eta$ readings the calibration
*can* see under twirled-native (e.g. $0.2889$ and $0.3108$ at rate $0.1$); and the projective
route twirls readout-axis dephasing ($T_{zz} = 1$) to *no noise at all*.

**Novelty ledger** (what is whose): the single-qubit core of the projective channel is Chen et
al.'s robust-shadow primitive — their uniformly-random Clifford + $Z$ readout *is* the
$2O$ protocol, their fidelity parameter is $\eta$. Ours: the two-channel distinction
($T_{zz}$ vs $\operatorname{tr}T/3$ — that the estimator-channel factor *identifies the protocol*),
the twirl that keeps the SIC, $2T$-universality with the realize/twirl split, and §9's gate-noise
pricing. Nguyen et al. (2022), the nearest neighbor, use Platonic transitivity to simplify the
canonical dual and model readout noise on the dilation ancillas — no group-averaging of noise
anywhere.
''')

code(r'''
# === Demo: the five-liner mechanized, the two deaths, and a reducible counterexample ===

zhat = np.array([0.0, 0.0, 1.0])
R_I = np.load(DATA / "group_I.npz")["rotations"]
s_ico = povms["icosahedron"]["s"]
v = s_ico[np.argmax(s_ico[:, 2])]
A = align_to_z(v)

# R1's twirled object: the rank-one composition, conjugated into the solid's frame
X1 = A.T @ np.outer(zhat, zhat) @ T_PROBE @ A
tw1 = np.einsum("gji,jk,gkl->il", R_I, X1, R_I) / len(R_I)
assert np.abs(tw1 - (np.trace(X1) / 3) * np.eye(3)).max() < 1e-12
print(f"R1: twirl(zz^T T) = (tr/3) Id,  tr = T_zz = {np.trace(X1):.6f}")

# R2's twirled object: the noise alone
tw2 = np.einsum("gji,jk,gkl->il", R_I, T_PROBE, R_I) / len(R_I)
assert np.abs(tw2 - (np.trace(T_PROBE) / 3) * np.eye(3)).max() < 1e-12
print(f"R2: twirl(T)      = (tr/3) Id,  tr/3 = {np.trace(T_PROBE) / 3:.6f}")

# the offset's two deaths, separately
print(f"death 1 (solid):  max|sum of vertices| = "
      f"{np.abs(s_ico.sum(axis=0)).max():.1e}")
print(f"death 2 (group):  max|E_g R_g|         = "
      f"{np.abs(R_I.mean(axis=0)).max():.1e}")

# counterfactual: a REDUCIBLE twirl group -- the z-axis stabilizer inside O.
# Schur's step dies and the channel is no longer one scalar.
R_O = np.load(DATA / "group_O.npz")["rotations"]
stab = np.array([R for R in R_O if abs((R @ zhat)[2]) > 0.999])
print(f"\nz-axis stabilizer in O: {len(stab)} rotations "
      "(reducible -- it fixes the z line)")
twr = np.einsum("gji,jk,gkl->il", stab, T_PROBE, stab) / len(stab)
print("its twirl of the probe noise:")
print(np.round(twr, 6))
print("two scalars, not one: the x/y and z blocks average separately, so a "
      "single\ncalibrated eta cannot describe the channel -- the one-scalar "
      "claims of the robust\nprotocol die exactly at the lemma's "
      "'irreducibly'. (The dodecahedron's D5 story\nin "
      "randomized_implementations.py is this same failure inside I.)")
''')


# =============================================================================
# Cells 39-43 -- Gate noise
# =============================================================================
md(r'''
## 9. Gate noise — the twirl's assumption, priced with the atlas

The twirl rests on one assumption: the noise must be **independent of the drawn rotation**. On
hardware each $g$ is its own circuit, so noise arrives per elementary gate, in an amount and
orientation *correlated with the draw* — and Schur's lemma is silent about a correlated average.
The assumption is not the same size for the two protocols: the projective route's whole circuit
varies with $g$; the twirled-native circuit varies only in its drawn $2T$ prefix — a Clifford
word of depth $\le 2$ — the dilation being the same circuit every shot.

**What the study computes.** One circuit per rotation from the atlas; amplitude damping of
strength $\gamma$ after every elementary gate of the drawn word; the estimator's channel averaged
*exactly* over the group; reported is the residual bias left on the TFIM site-0 Bloch components
after the $\ket{0}$-calibration — the error that survives the protocol's own correction, per unit
$\gamma$. The noise accounting per series:

- **projective rows** (`2O`, `2I`, `2I_phi3`, `2I_dij_phi3`, `2T`): damping on the drawn word
  only; the alignment is held ideal — a fixed gate's noise is $g$-independent and twirls cleanly,
  so charging it would only blur the correlated signal being measured. Because the noise *is*
  $g$-correlated, the residual depends on which vertex the alignment picks: the min/max over all
  vertex choices is stored alongside the convention's value.
- **`R2` rows**: damping on the same drawn $2T$ words; the fixed dilation is *modeled* as one
  $g$-independent pre-measurement amplitude-damping channel ($\Gamma = 0.05$) on the data qubit.
  A `R2bare` variant with no dilation channel isolates that modeling choice (it moves the
  calibration reading only; the residual shifts by under 2%).

**The two exact anchors** — the answer to "isn't the zero-residual claim circular, since you
modeled the dilation as removable?": at $\gamma = 0$ the residual is zero *against the visibly
noisy modeled dilation* — proving the twirl removes **whatever is $g$-independent**, which is a
theorem, not an assumption of convenience ($g$-independence of a fixed circuit is a statement
about the hardware layout, and the general theorem covers *any* such channel, ancilla-side
deformations included; only the location of the model is a choice, and `R2bare` prices it). And
gate-*independent* noise of any strength still twirls to an exact scalar with zero residual,
whatever the compilation. What the model is *not*: a hardware forecast. One species, one rate —
an illustration of the assumption's price, run on the atlas's own words.
''')

code(
    "# === Gate-noise study: per-gate damping on the atlas circuits ===\n\n"
    + lift("bloch_rotation", "parse_sequence")
    + "\n\n\n"
    + lift("gate_noise_twirl")
    + "\n\n\ngate_out = gate_noise_twirl(states, povms)"
)

md(r'''
### Reading the residuals

Every projective series is **linear in $\gamma$**, and the slope tracks how the noise correlates
with the draw — not any single cost column: $2I$ (mean depth 2.43, $0.92\,\Phi$ per rotation)
pays about twice $2O$ (depth 1.71, Clifford); tripling the damping on $\Phi$ alone — magic cost
read as noise cost — more than doubles $2I$'s bill; and recompiling min-magic (Dijkstra)
*raises* the $Z_0$ residual by about a third while carrying **less** $\Phi$ (0.80 vs 0.92) — a
cheaper magic bill is not a cheaper correlation bill. Swapping the icosahedron's full-$2I$ draw
for the minimal $2T$ one — same POVM, same alignment vertex, twelve all-Clifford words of depth
$\le 2$ — roughly **halves** the correlated exposure (and flips the sign: that is a property of
the convention vertex, as the alignment-span rows show; the magnitude is not).

The `R2` row is the headline: its $Z_0$ residual is **second order** ($-0.082\,\gamma^2$ — four
orders below the projective rows at $\gamma = 10^{-3}$), its $X_0$ slope about half the projective
$2T$ row's, and the channel is **provably solid-independent** — the drawn words never see which
POVM the dilation implements, so the five per-solid rows are asserted equal to $10^{-12}$ and one
bill prices every solid. For the tetrahedral SIC, which no projective protocol admits, these are
its first noise numbers in the thesis.

That second order is not luck, and the next two cells prove why.
''')

md(r'''
### Cracking the second order

Expand the noisy word channel to first order in $\gamma$. Amplitude damping after one gate
contributes the generator pair $D = \mathrm{diag}(\frac12, \frac12, 1)$ (matrix decay) and
$\hat z$ (displacement), inserted at that point of the word: writing the word's rotation as
$R_g = S\,P$ — suffix times prefix at the insertion — the first-order channel error is a sum of
insertions $S\,D\,P$ and displacements $S\,\hat z$. Three structural facts finish it:

1. **The estimator sees each insertion through its prefix alone.** Twirled-native post-processing
   applies $R_g^\top$ to the snapshot, and orthogonality cancels the suffix:
   $R_g^\top (S\,D\,P) = P^\top D\,P$ (with the modeled dilation in front,
   $R_g^\top T_N S\,D\,P = P^\top (S^\top T_N S)\,D\,P$ — the suffix survives only inside a
   diagonal sandwich). The first-order average is a sum over the **prefix multiset** of the twelve
   drawn words.
2. **Nothing first-order can tilt axes.** Every rotation in $T$ is a *signed coordinate
   permutation*, and $D$, $T_N$ are diagonal — so every term above is diagonal: at order $\gamma$
   the twirled channel error can **rescale axes and displace along axes, but never mix them**.
   ($O$ is also all signed permutations, so this holds for the octahedral projective row too —
   diagonality alone is not the protocol difference.)
3. **A scalar $z$-calibration absorbs a diagonal $z$-error exactly.** Writing the first-order
   channel as $M \approx \eta_0\mathrm{Id} + \gamma M_1$, $m \approx \gamma m_1$ with $M_1$
   diagonal, the calibrated residual's linear $Z_0$ coefficient collapses to
   $$\frac{d}{d\gamma}\Bigl[\text{residual}_z\Bigr]_{\gamma=0} =
   \frac{(m_1)_z\,(1 - z_0)}{\eta_0}$$
   — the matrix part cancels *identically* between numerator and calibration. Only an axis-aligned
   **$z$-displacement** could survive at first order — and for the twelve atlas words the $\pm$
   prefix contributions along $z$ cancel **exactly**: $(m_1)_z = 0$, verified by direct count
   below. So the $Z_0$ residual starts at $\gamma^2$.

What stays linear, and why, completes the picture. The *transverse* displacement does not cancel
($m_1 \propto (1,1,0)$), and no scalar calibration exists to absorb it: the $X_0$ slope is
$(m_1)_x/\eta_0$ — for the bare variant exactly $\frac{1/18}{1/3} = \frac16$ — plus a whisker of
$\Gamma$-induced diagonal anisotropy. And the **projective** route stays linear in $Z_0$ because
its $g$-correlated snapshot *weight* deposits a first-order $z$-displacement: for the octahedral
draw — where step 2's diagonality argument applies just as well — the entire $+0.250$ slope is
displacement, $(m_1)_z = \frac1{12}$ exactly, and $3 \times \frac1{12} = 0.25$. The protocol
difference at first order is *where the displacement goes*: the projective correlation pushes it
onto the readout axis; the twirled-native relabeling average cancels it there.
''')

code(r'''
# === Demo: the first-order error, assembled by hand from the word prefixes ===

gates_npz = np.load(DATA / "gates.npz")
su2 = {("Φ" if str(n) == "Phi" else str(n)): U
       for n, U in zip(gates_npz["names"], gates_npz["su2"])}
rot = {n: bloch_rotation(U) for n, U in su2.items()}
d2t = np.load(DATA / "group_2T.npz")
Us, seqs, score = d2t["unitaries"], d2t["bfs_sequences"], d2t["bfs_depths"]
reps = {}
for i in range(len(Us)):        # one circuit per rotation, as in the study
    key = tuple(np.round(bloch_rotation(Us[i]), 9).ravel())
    if key not in reps or score[i] < score[reps[key]]:
        reps[key] = i
idx = sorted(reps.values())
words = [parse_sequence(seqs[i]) for i in idx]
Rlist = [bloch_rotation(Us[i]) for i in idx]
print("the twelve drawn words:",
      [" ".join(b + ("†" if dg else "") for b, dg in tk) or "I"
       for tk in words])

# fact: every rotation in T is a signed coordinate permutation
assert all(set(np.unique(np.abs(np.round(R, 12)))) <= {0.0, 1.0}
           for R in Rlist)

zhat = np.array([0.0, 0.0, 1.0])
D_gen = np.diag([0.5, 0.5, 1.0])            # amplitude damping: -dT/dgamma
TN, tN = chan_amp_damping(GAMMA_DIL)        # the modeled dilation channel
M1 = np.zeros((3, 3)); m1 = np.zeros(3)     # first-order error, dilated
m1_bare = np.zeros(3)                       # and the bare displacement
for tk, Rg in zip(words, Rlist):
    P = np.eye(3)
    for base, dag in reversed(tk):          # gates in time order
        P = (rot[base].T if dag else rot[base]) @ P
        S = Rg @ P.T                        # suffix: S P = R_g
        mid = S.T @ TN @ S                  # diagonal (signed permutations)
        M1 -= P.T @ mid @ D_gen @ P / (3 * 12)
        m1 += P.T @ mid @ zhat / (3 * 12)
        m1_bare += P.T @ zhat / (3 * 12)

off = np.abs(M1 - np.diag(np.diag(M1))).max()
print(f"\nM1 off-diagonal = {off}  (exactly diagonal)")
print(f"diag(M1) = {np.round(np.diag(M1), 6)}")
print(f"m1 = {np.round(m1, 6)}   (m1)_z = {m1[2]}  (exact cancellation)")
assert off == 0.0 and m1[2] == 0.0

# the scalar calibration absorbs the diagonal z-error: linear slopes
r0 = site_bloch(states["TFIM"])[0]
eta0 = np.trace(TN) / 9
eta1 = M1[2, 2] + m1[2]
lin = (M1 @ r0 + m1 - r0 * eta1) / eta0
g = GATE_GAMMAS[1]
row = gate_out[f"gatenoise/R2/icosahedron/{g}"]
print(f"\npredicted linear slopes:  X0 = {lin[0]:+.6f}   Z0 = {lin[2]:+.6f}")
print(f"study, at gamma={g}:      X0/g = {row[3] / g:+.6f}   "
      f"Z0/g = {row[4] / g:+.6f}   Z0/g^2 = {row[4] / g ** 2:+.4f}")
assert lin[2] == 0.0 and abs(lin[0] - row[3] / g) < 1e-3
print(f"bare closed form: X0 slope = 3*(m1_bare)_x = {3 * m1_bare[0]:.6f} "
      f"= 1/6;  study: {gate_out[f'gatenoise/R2bare/{g}'][3] / g:+.6f}")

# contrast: the projective route on the SAME words -- the snapshot weight is
# g-correlated, and its first-order z-displacement is what stays linear
def r1_channel(gamma):
    M = np.zeros((3, 3)); m = np.zeros(3)
    v = s_ico[np.argmax(s_ico[:, 2])]
    for tk, Rg in zip(words, Rlist):
        T, t = np.eye(3), np.zeros(3)
        for base, dag in reversed(tk):
            R = rot[base].T if dag else rot[base]
            T, t = R @ T, R @ t
            if gamma > 0:
                Tn, tn = chan_amp_damping(gamma)
                T, t = Tn @ T, Tn @ t + tn
        M += np.outer(Rg.T @ v, T.T @ v) / 12
        m += (Rg.T @ v) * (v @ t) / 12
    return M, m

eps = 1e-7
m1_proj = (r1_channel(eps)[1] - r1_channel(0.0)[1]) / eps
print(f"\nprojective 2T draw (icosahedron vertex): first-order z-displacement "
      f"= {m1_proj[2]:+.6f}")
print(f"study's 2T Z0 slope: {gate_out[f'gatenoise/2T/{g}'][4] / g:+.6f}   "
      "(nonzero displacement -> linear; the scalar cannot absorb it)")
''')


# =============================================================================
# Cells 44-45 -- n-scaling
# =============================================================================
md(r'''
## 10. $n$-scaling — nothing above is an $n = 4$ artifact

**The job:** rule out that Experiment 2's gains evaporate — or explode — as the chain grows.

**Why this is computable exactly at $n = 16$**, where the outcome tensor would have $12^{16}$
entries: the variance is a sum over *term pairs*, and each pair's expectation touches only the
reduced state on the union of the two supports — at most 4 sites for the TFIM. Sparse-Lanczos
ground states plus $\le 4$-site reduced density matrices make every dual level's single-shot
variance an exact number at any $n$ we care to reach (the $n = 4$ values are asserted equal to
the full outcome-tensor route).

**The dual-independent floor.** For a support-*disjoint* pair the per-site operators reduce to
plain Paulis — the middle case of the site-cases identity, i.e. the frame condition, valid for
*every* dual — so disjoint-pair covariances are identical for all duals, and optimization can
only touch the shared-site block. That floor is exactly the long-range-correlation share one
might fear would dominate at criticality; it never exceeds $1.7\%$ of the variance at any size
computed.

**The verdict:** the per-site variance is flat within two percent ($12.41 \to 12.18$), and the
optimization gain **converges, not collapses** — oracle $15.6\% \to 15.1\%$, observable-optimized
$\to 14.1\%$. The $n = 4$ tables were representative all along; what caps the gain is the
per-site quantum limit of the *factorized estimator class*, not anything the chain grows into.

And the honest defense of that mildly negative headline — "fifteen percent, so what?": the
ceiling is a *structural* fact about the scalable class (leaving it — joint or tensor-network
duals — buys orders of magnitude, at exponential or network cost), and knowing the exact
accounting changes what a deployer does: pick the cube for tails (§4), expect no ranking at the
canonical dual (§3), budget the calibration premium (§6), keep $\hat v_\perp$ (§7), calibrate
the scalar *of the protocol actually run* (§8), and prefer the twirled-native route when gate
noise dominates (§9).
''')

code(
    "# === n-scaling: exact variance out to n = 16 ===\n\n"
    + lift("tfim_terms_n", "tfim_ground_sparse", "tfim_ground_energy_exact",
           "reduced_rho", "site_pair_ops", "exact_energy_variance")
    + "\n\n\n"
    + lift("nscaling")
    + "\n\n\nnscale_out = nscaling(povms, states)"
)


# =============================================================================
# Cells 46-47 -- The receipts
# =============================================================================
md(r'''
## 11. The receipts

The cells above *are* the module — lifted mechanically at build time — but a committed notebook
can drift after later module edits. This cell closes the gap behaviorally and numerically:

1. **behavioral spot-checks**: import the production module and assert our (identical-by-lift)
   primitives return bit-equal results on shared inputs;
2. **the full replay**: rebuild the exact dictionary `main()` writes — same loops, same key
   grammar — and compare against the committed `data/shadow_experiments.npz`: the key sets must
   match exactly, and every value must agree to **zero** (not tolerance — the notebook threaded
   the same seed through the same operations in the same order).

The notebook never writes the npz; regenerating it is the script's job
(`uv run shadow_experiments.py`, deterministic).
''')

code(r'''
# === Anti-drift: this notebook == the committed npz, exactly ===

import shadow_experiments as se

# behavioral spot-checks across the pipeline
_p = povms["icosahedron"]
assert np.array_equal(born_tensor(states["TFIM"], _p["E"]),
                      se.born_tensor(states["TFIM"], _p["E"]))
_a, _b = optimize_dual(_p["s"], np.full(12, 1 / 12), 2)
_a2, _b2 = se.optimize_dual(_p["s"], np.full(12, 1 / 12), 2)
assert np.array_equal(_a, _a2) and np.array_equal(_b, _b2)
_R = np.load(DATA / "group_I.npz")["rotations"]
for f_nb, f_se in ((channel_R1, se.channel_R1), (channel_R2, se.channel_R2)):
    M_nb, m_nb = f_nb(_p["s"], _R, T_PROBE, t_PROBE)
    M_se, m_se = f_se(_p["s"], _R, se.T_PROBE, se.t_PROBE)
    assert np.array_equal(M_nb, M_se) and np.array_equal(m_nb, m_se)
assert np.array_equal(site_bloch(states["GHZ"]), se.site_bloch(states["GHZ"]))
print("behavioral spot-checks against the production module: OK")

# rebuild main()'s output dict -- same loops, same key grammar
out = {}
for (pn, sn, on), st in exp1.items():
    out[f"exp1/{pn}/{sn}/{on}"] = np.array(
        [st["bias2"], st["var"], st["mse"], st["se_mse"]])
for (pn, on), v in exp1_haar.items():
    out[f"exp1_haar/{pn}/{on}"] = np.array([v])
for (pn, sn, on, ln), st in exp2.items():
    out[f"exp2/{pn}/{sn}/{on}/{ln}"] = np.array(
        [st["bias2"], st["var"], st["mse"], st["se_mse"]])
for key, st in exp3.items():
    if key[1] == "eta":
        out[f"exp3/eta/{key[0]}"] = np.array(st)
    else:
        p_rate, on, ln = key
        out[f"exp3/{p_rate}/{on}/{ln}"] = np.array(
            [st["bias2"], st["var"], st["mse"], st["se_mse"]])
for (cn, p_rate), row in blind.items():
    out[f"blind/{cn}/{p_rate}"] = np.array(
        [row["eta"], *row["X0"], *row["Z0"]])
for label, v in aniso.items():
    out[f"blind/aniso/{label}"] = v
for extra in (exact_out, four_out, ratio_out, tp_out, gate_out, nscale_out):
    out.update(extra)

committed = np.load(DATA / "shadow_experiments.npz")
assert set(out) == set(committed.files), \
    sorted(set(out) ^ set(committed.files))[:10]
fams = {}
for k in committed.files:
    fams[k.split("/")[0]] = fams.get(k.split("/")[0], 0) + 1
print("families: " + ", ".join(f"{f} ({n})" for f, n in sorted(fams.items())))
worst = max(np.abs(out[k] - committed[k]).max() for k in committed.files)
print(f"\nnpz keys: {len(committed.files)}  (all present, none extra)")
print(f"value-for-value: max |notebook - committed| = {worst}")
assert worst == 0.0
print("\nThe committed npz and this notebook agree exactly. "
      "The notebook wrote nothing.")
''')


# =============================================================================
# Cell 48 -- Closing notes
# =============================================================================
md(r'''
## Closing notes

**What was shown.**

| § | finding |
|---|---|
| 3 | at the canonical dual the four antipodal solids are provably identical on every (state, observable) — the variance sees only vertex moments $\le 3$; single weight-$w$ strings sit at $3^w - \langle P\rangle^2$ exactly; the tetrahedron escapes in exactly one cell, and it takes three conditions at once |
| 4 | design strength pins the tails at the sphere's weight but does not minimize them: the cube ($\pm\sqrt3$ always) is lightest, the octahedron — standard Pauli-6 shadows — heaviest |
| 5 | dual optimization is a per-letter QP; gains are real, modest, and do not scale with $V$ (the cube beats both 5-designs); the per-letter oracle can lose exactly (1.069 — cross-term covariances the objective never sees); the oracle is a ceiling, not a protocol |
| 6 | $\mathbb E[\hat\eta] = (1-p)/3$ is an identity; calibration converts an incurable bias ($(1-p)^w$, exact) into a variance premium (~60% at $p=0$) that amortizes as $1/R_C$ |
| 7 | a $\ket 0$-calibration is structurally blind to channels fixing $\ket 0$; the reported bias is exactly $(Tr+t-r)_a$; $\hat v_\perp$ is a free tilt diagnostic; amplitude damping evades both; the failure mode is silence |
| 8 | one lemma, two conjugations: readout∘noise → $T_{zz}/3$ (randomized-projective), noise alone → $(\operatorname{tr}T/3)/3$ (twirled-native, offset dead twice over, SIC included); the estimator-channel factor identifies the protocol (0.62 vs 0.72 on one probe); $2T$ is the universal minimal twirl and the dodecahedron's full-$2I$ bill is a *realization* cost |
| 9 | gate noise breaks the twirl exactly as far as the drawn words: projective residuals are linear and track the draw–noise correlation (min-magic recompilation *raises* the bill); the $2T$ draw halves the icosahedron's exposure; the twirled-native $Z_0$ residual is second order — proven: prefix conjugation + signed permutations ⇒ diagonal first-order error, which one scalar absorbs along $z$ and the word set's displacement balance finishes — and its one row prices all five solids, the SIC's first noise numbers |
| 10 | everything holds at $n = 16$, exactly: gains converge (15.1%/14.1%), the dual-independent floor stays $\le 1.7\%$, and the ceiling is the factorized class itself |

**The study's assumptions, collected** (each is load-bearing somewhere above): the per-qubit
*factorized* estimator class (§3, §10 — the ceiling); i.i.d. measurement-side noise (§6);
the depolarizing family for the one-scalar calibration — *assumed* under the native
implementation, *enforced* under either randomized one (§7, §8); a trusted $\ket 0$ probe
(§6; relaxing it is Jeanette et al.'s blind calibration); the twirl's $g$-independence
(§9 prices its failure); clean classical post-processing throughout.

**The three corners** (the appendix's closing trade): *native* — no random gates, keeps the SIC,
but the depolarizing model is an assumption and the calibration has a structural blind spot;
*twirled-native* — one all-Clifford depth-$\le2$ word per shot on top of the dilation turns the
model into a theorem, keeps the SIC, and confines the correlated exposure to that word;
*randomized-projective* — sheds the dilation, but one draw must realize *and* twirl: no SIC, an
inexact alignment everywhere but the octahedron, and the dodecahedron's $0.8\,\Phi$ per shot.
Which corner wins is a property of the hardware; the mathematics fixes the menu.

**Numbers worth holding cold** (everything else: know *where it lives*, not its digits): the
ideal shrinkage $1/3$; *that* the two protocols read different scalars off the same noise (0.62
vs 0.72 as an existence proof); the ~15% factorized-optimization ceiling; the dodecahedron's
$0.8\,\Phi$ per shot; the $2T$ draw = twelve all-Clifford words of depth $\le 2$.

**Self-test.** This notebook is built to be read closed-book: take each claim above,
regenerate it on paper, and only then read the derivation back. One of them — *why is
the twirled-native $Z_0$ residual second order in $\gamma$?* — is the one §9 works out in full,
mechanism and verification.

To run the production script end to end (writes the npz; deterministic):

```
cd code && uv run shadow_experiments.py
```

To regenerate this notebook after editing the builder (never edit the .ipynb directly):

```
cd code && uv run python _build_shadow_walkthrough.py
uv run --with jupyter --with nbconvert jupyter nbconvert --to notebook --execute --inplace shadow_walkthrough.ipynb
```
''')


# =============================================================================
# Assemble and write
# =============================================================================
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

out = HERE / "shadow_walkthrough.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Wrote {out} ({len(cells)} cells)")
