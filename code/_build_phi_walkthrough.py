r"""Builds code/phi_walkthrough.ipynb from clean inline source.

Run with `cd code && uv run python _build_phi_walkthrough.py`.

This script is the source of truth for the notebook. To edit a cell, modify
the corresponding `md(r'''...''')` or `code(r'''...''')` call below, then
re-run this script to regenerate the .ipynb. Do not edit the notebook
directly -- regeneration will overwrite manual edits.

The notebook is the walkthrough for `code/phi_simulation_cost.py`, the script
backing the classical-simulation-cost claims of Appendix B.3. Code cells carry
copies of that script's functions so a reader can recognize them one-for-one,
and the `lifts=` argument to `code()` is what keeps "verbatim" true: each copy
is checked against `inspect.getsource` at build time. Cell 17 of 26 imports the
production module and runs its `main()` as an anti-drift cross-check; nine
bonus cells then redo the three claims in the field, with no tolerance
anywhere.

After regenerating, re-execute in place so the stored outputs stay current:

    uv run --with jupyter --with nbconvert jupyter nbconvert \
        --to notebook --execute --inplace phi_walkthrough.ipynb
"""

import inspect
from pathlib import Path

import nbformat

import phi_simulation_cost as psc

nb = nbformat.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbformat.v4.new_markdown_cell(text.strip("\n")))


def code(text, lifts=()):
    """Add a code cell; `lifts` names production defs it must contain verbatim.

    A copy that no longer matches `inspect.getsource` stops the build, so the
    notebook cannot quietly disagree with `phi_simulation_cost.py`.
    """
    text = text.strip("\n")
    for name in lifts:
        src = inspect.getsource(getattr(psc, name)).rstrip("\n")
        assert src in text, (
            f"{name} has drifted from phi_simulation_cost.py; re-copy the "
            f"production source into this builder:\n\n{src}\n"
        )
    cells.append(nbformat.v4.new_code_cell(text))


# =============================================================================
# Cell 1 (md) -- Title + intro
# =============================================================================
md(r'''
# Golden-Gate Simulation Cost — Walkthrough

A companion walkthrough for `code/phi_simulation_cost.py`, the script behind the
classical-simulation-cost claims in Appendix B.3 of the thesis. The thesis argues that the golden
gate $\Phi$ is, in resource terms, *almost free*: a Clifford${}+\Phi$ circuit can be classically
simulated in time exponential **only** in the number $m$ of $\Phi$ gates, and the base of that
exponential is itself a golden-ratio quantity.

The relevant machinery is the **sum-over-Cliffords** simulator of Bravyi et al. (*Quantum* **3**,
181, 2019). Its runtime is $\mu(U)^{2m}\,\mathrm{poly}(n)$, where the base is the **gate extent**
$\mu(U)$ — the smallest $\ell_1$-norm over ways of writing the gate as a linear combination of
single-qubit Cliffords. This notebook verifies the three claims that paragraph and footnote rest
on:

1. **An explicit decomposition (upper bound).** The two-term expression
   $$\Phi \;=\; \frac{\sigma(1+i)}{2}\,S^3 \;+\; \frac{1}{\sqrt2}\,ZH$$
   is *exact*, so $\mu(\Phi) \le \sigma/\sqrt2 + 1/\sqrt2 = \tau/\sqrt2$.
2. **Optimality (matching lower bound).** No Clifford decomposition does better: basis pursuit over
   all $24$ projective Cliffords lands on the same $\ell_1$-norm, and a **dual certificate** closes
   the duality gap to $\sim 10^{-12}$. Hence $\mu(\Phi) = \tau/\sqrt2$ *exactly*, and the per-$\Phi$
   simulation cost is $\mu(\Phi)^2 = \tau^2/2 \approx 1.31$. The same machinery recovers the known
   $\mu(T) = \cos\frac\pi8 + (\sqrt2-1)\sin\frac\pi8$ as a positive control.
3. **$\Phi$ is outside the third Clifford level.** $\Phi P \Phi^\dagger$ is non-Clifford for
   $P = X, Y, Z$ (where $T$ passes the analogous test), so the $T$-style trick of injecting a
   gate with a Clifford correction does not apply to $\Phi$ — its magic, though minimal, is genuine.

Everything is `float64` numpy. The algebraic numbers involved are well separated, and the certified
duality gap sits orders of magnitude below the spacing of candidate closed forms, so floating point
is safe here. That is a claim about *comparisons*, and separation is the whole of its argument; the
rounding **key** below is a different object and needs a second argument besides, which is where the
next cell goes. The numbers printed below should match the stdout of
`cd code && uv run phi_simulation_cost.py`.

**References:**
- Bravyi, Browne, Calpin, Campbell, Gosset, Howard, *Simulation of quantum circuits by low-rank stabilizer decompositions* (Quantum 3, 181, 2019)
- Gottesman & Chuang, *Quantum teleportation is a universal computational primitive* (1999) — the Clifford hierarchy
- Conway & Smith, *On Quaternions and Octonions* (2003) — the $\tau$ / $\sigma$ convention
''')


# =============================================================================
# Cell 2 (code) -- Setup: constants, gates, Phi
# =============================================================================
code(r'''
# === Imports, constants, and gates (lifted from phi_simulation_cost.py) ===

import numpy as np

TAU = (1 + np.sqrt(5)) / 2          # golden ratio (Conway & Smith tau)
SIGMA = TAU - 1                     # inverse golden ratio (sigma = 1/tau)

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

# Golden gate, Equation (eq:golden-gate) of the thesis.
PHI = 0.5 * np.array(
    [[TAU + 1j / TAU, 1], [-1, TAU - 1j / TAU]], dtype=complex
)

print(f"tau   = {TAU:.6f}")
print(f"sigma = {SIGMA:.6f}   (tau * sigma = {TAU * SIGMA:.6f}, expect 1)")
print(f"1/tau = {1 / TAU:.6f}  (= sigma, since i/tau = i*sigma in Phi)")
print()
print("Phi =")
print(np.round(PHI, 6))
print("Phi is unitary:", np.allclose(PHI @ PHI.conj().T, I2),
      " det(Phi) =", np.round(np.linalg.det(PHI), 6))
''')


# =============================================================================
# Cell 3 (md) -- The projective Clifford group and the rounding grid
# =============================================================================
md(r'''
## The projective Clifford group, by rounding

Every quantity below is phrased against the **single-qubit Clifford group**: the $24$ unitaries
generated by $S$ and $H$, taken *modulo global phase* (so $U$ and $e^{i\theta}U$ are the same
element). Global phase is unphysical, and — crucially for the gate extent — rescaling a Clifford by
a phase rescales its coefficient by the inverse phase, leaving the $\ell_1$-norm unchanged. So the
extent is well defined even though we only ever hold phase representatives.

To make "modulo phase" computable we use the same trick as the numpy walkthrough: **identity by
rounding**. `proj_key` maps a unitary to a canonical, hashable tuple by (i) dividing out the
determinant to force $\det = 1$ up to sign, (ii) rotating the first large entry to be positive real
to kill the residual phase, and (iii) rounding to $9$ decimals. Two unitaries should collide under
`proj_key` exactly when they are equal up to global phase — which, as in the numpy walkthrough,
takes *two* arguments and not one.

**Collision** is ruled out by separation, and the margin is enormous: distinct projective Cliffords
are $0.707$ apart against a grid of $10^{-9}$.

**Splitting** — one unitary reaching two different keys, because float noise carried a component
across a rounding boundary — is ruled out by the distance to that boundary, and here the group on
its own is misleading. Clifford entries come from $\{0, \pm\tfrac12, \pm\tfrac1{\sqrt2}, \pm1\}$,
whose tightest member $1/\sqrt2 = 0.707106781\,|\,1865\ldots$ clears a boundary by
$3.13 \times 10^{-10}$. But `proj_key` is not applied only to Cliffords: Claim 3 keys
$\Phi P \Phi^\dagger$, and $\Phi$ drags $\sqrt5$ in with it, so *those* keys carry
$\tau/2 = 0.809016994\,|\,3749\ldots$ and $\sigma/2$ and clear a boundary by only
$1.25 \times 10^{-10}$ — against a measured float error of $2.0 \times 10^{-16}$, so a margin of
$6 \times 10^5$. **Which entry binds is set by the call site, not by the group**, and the tightest
call site is the one carrying the thesis claim.

Both margins are real and both were measured — but a measurement is still standing in for a
decision, and the bonus section at the end replaces the whole arrangement with one.

The group itself is then a breadth-first closure of $\langle S, H\rangle$: keep left-multiplying the
frontier by $S$ and $H$, keying each result, until no new key appears. It terminates at exactly $24$.
''')


# =============================================================================
# Cell 4 (code) -- proj_key + clifford_group (verbatim) + build
# =============================================================================
code(r'''
# === proj_key and the Clifford closure (verbatim from the script) ===

def proj_key(U):
    """Canonical key of a unitary modulo global phase (float-rounding grid)."""
    U = U / np.linalg.det(U) ** 0.5                 # det 1 up to sign
    flat = U.flatten()
    k = np.argmax(np.abs(flat) > 1e-9)
    U = U * (np.abs(flat[k]) / flat[k])             # first big entry > 0 real
    return tuple(np.round(U.flatten(), 9).tolist())


def clifford_group():
    """The 24 projective single-qubit Cliffords as closure of <S, H>."""
    group = {proj_key(I2): I2}
    frontier = [I2]
    while frontier:
        new = []
        for U in frontier:
            for G in (S, H):
                V = G @ U
                key = proj_key(V)
                if key not in group:
                    group[key] = V
                    new.append(V)
        frontier = new
    return group


cliffords = clifford_group()
assert len(cliffords) == 24, f"expected 24 Cliffords, got {len(cliffords)}"
print(f"projective Clifford group <S, H>: {len(cliffords)} elements")

# A phase rescaling leaves the key unchanged -- that is what makes the extent
# (an l1 norm over these representatives) independent of phase choice.
print("proj_key(S) == proj_key(exp(1.2i) * S):",
      proj_key(S) == proj_key(np.exp(1.2j) * S))
''', lifts=("proj_key", "clifford_group"))


# =============================================================================
# Cell 5 (md) -- Phi is in SU(2) and non-Clifford
# =============================================================================
md(r'''
## $\Phi$ sits in $\mathrm{SU}(2)$ and is non-Clifford

Two quick prerequisites. The golden gate is genuinely a special-unitary single-qubit gate
($\Phi\Phi^\dagger = \mathbf I$, $\det\Phi = 1$), and it is genuinely *not* a Clifford — otherwise the
whole magic-cost discussion would be vacuous. The non-Clifford check is a one-liner: $\Phi$ is a
Clifford iff its `proj_key` is one of the $24$ keys we just enumerated.
''')


# =============================================================================
# Cell 6 (code) -- Phi SU(2) + non-Clifford
# =============================================================================
code(r'''
# === Phi is in SU(2) and is not a Clifford ===

def is_clifford(U, cliffords):
    return proj_key(U) in cliffords


# Guards in this notebook measure the deviation and bound it explicitly:
# allclose and isclose keep rtol=1e-5 alongside any atol, and it scales with the
# expected operand, not the residual.  Residuals here are 1.3e-17 and 1.2e-16.
d_uni = np.abs(PHI @ PHI.conj().T - I2).max()
d_det = abs(np.linalg.det(PHI) - 1)
assert d_uni < 1e-12, f"Phi is not unitary (max |dev| = {d_uni:.2e})"
assert d_det < 1e-12, f"det Phi != 1 (|dev| = {d_det:.2e})"
assert not is_clifford(PHI, cliffords), "Phi must be non-Clifford"
print("[ok] Phi is in SU(2) and is not a Clifford gate")
''', lifts=("is_clifford",))


# =============================================================================
# Cell 7 (md) -- Claim 3: the Clifford hierarchy
# =============================================================================
md(r'''
## Claim 3 — $\Phi$ lies outside the third Clifford level

The **Clifford hierarchy** (Gottesman & Chuang, 1999) is the nested family $\mathcal C^{(1)} \subset
\mathcal C^{(2)} \subset \mathcal C^{(3)} \subset \cdots$ defined by
$$\mathcal C^{(1)} = \text{Pauli group}, \qquad
  \mathcal C^{(k+1)} = \bigl\{\, U : U P U^\dagger \in \mathcal C^{(k)} \ \text{ for every Pauli } P \,\bigr\}.$$
So $\mathcal C^{(2)}$ is the Clifford group (gates that send Paulis to Paulis), and the **third level**
$\mathcal C^{(3)}$ is the set of gates that send every Pauli into the *Clifford* group. The $T$ gate is
the canonical inhabitant of $\mathcal C^{(3)}$, and that membership is exactly what powers gate
teleportation / magic-state injection: a $\mathcal C^{(3)}$ gate can be applied by consuming a magic
state and applying a **Clifford** correction conditioned on the measurement outcome.

$\Phi$ fails this. We verify the per-Pauli statement quoted in the thesis, which is *stronger* than
mere non-membership: $\Phi P \Phi^\dagger$ is non-Clifford for $P = X, Y, Z$ (non-membership
only needs *one* Pauli to escape). $T$ serves as the positive control — $T P T^\dagger$ is Clifford
for every $P$ (e.g. $TXT^\dagger = (X+Y)/\sqrt2$, a Clifford). The consequence: there is no
$T$-style injection-with-Clifford-correction for $\Phi$, so its magic cost — though never more than
one gate — is real and specific to the icosahedral gate set.
''')


# =============================================================================
# Cell 8 (code) -- level-3 per-Pauli table
# =============================================================================
code(r'''
# === Clifford-hierarchy level-3 test: T passes, Phi fails on X, Y, Z ===

print(f"{'gate':4s}  {'Pauli':5s}  U P U^dag is Clifford?")
print("-" * 36)
for name, U, expected in (("T", T, True), ("Phi", PHI, False)):
    for pname, P in (("X", X), ("Y", Y), ("Z", Z)):
        conj_clifford = is_clifford(U @ P @ U.conj().T, cliffords)
        assert conj_clifford == expected, \
            f"{name}: level-3 test surprised us on Pauli {pname}"
        print(f"{name:4s}  {pname:5s}  {conj_clifford}")
print()
print("[ok] T lies in the third level of the Clifford hierarchy;")
print("     Phi P Phi^dag is non-Clifford for P = X, Y, Z")
''')


# =============================================================================
# Cell 9 (md) -- The gate extent
# =============================================================================
md(r'''
## The gate extent, and why it sets the simulation base

A sum-over-Cliffords simulator writes each non-Clifford gate as a linear combination of Cliffords,
$U = \sum_i c_i C_i$, and propagates the corresponding superposition of stabilizer states. The
resource that controls the runtime is the **extent** $\xi(U) = \bigl(\sum_i |c_i|\bigr)^2$ — a
*squared* $\ell_1$-norm — and extents multiply across gates, so $m$ copies of one gate contribute a
factor $\xi(U)^m$. Minimizing the (un-squared) $\ell_1$-norm over decompositions gives the **gate
extent**
$$\mu(U) \;=\; \min\Bigl\{\, \textstyle\sum_i |c_i| \;:\; U = \sum_i c_i\,C_i,\ \ C_i \ \text{Clifford} \,\Bigr\},
  \qquad \xi(U) = \mu(U)^2.$$
Writing $\xi = \mu^2$ turns $\xi(\Phi)^m$ into $\mu(\Phi)^{2m}$ — *that* is where the exponent $2m$
comes from — so a circuit with $m$ copies of $\Phi$ simulates in time $\mu(\Phi)^{2m}\,\mathrm{poly}(n)$.

Computing $\mu$ is a **convex optimization**. Vectorize each of the $24$ Cliffords into a column of a
$4\times24$ matrix $A$, and vectorize the target gate into $b$. Then
$$\mu(U) \;=\; \min_{c}\ \|c\|_1 \quad\text{subject to}\quad A c = b,$$
the classic *basis-pursuit* problem. Because it is convex, the explicit decomposition of Claim 1
gives an upper bound and the dual of Claim 2 gives a matching lower bound — together pinning $\mu$
exactly.
''')


# =============================================================================
# Cell 10 (md) -- Claim 1
# =============================================================================
md(r'''
### Claim 1 — the two-term decomposition (upper bound)

The footnote's explicit decomposition is
$$\Phi \;=\; \frac{\sigma(1+i)}{2}\,S^3 \;+\; \frac{1}{\sqrt2}\,ZH.$$
Both $S^3 = S^\dagger$ and $ZH$ are Cliffords, so this is a *two-term* Clifford decomposition. Its
$\ell_1$-norm is
$$\Bigl|\tfrac{\sigma(1+i)}{2}\Bigr| + \Bigl|\tfrac{1}{\sqrt2}\Bigr|
  = \frac{\sigma}{\sqrt2} + \frac{1}{\sqrt2}
  = \frac{\sigma + 1}{\sqrt2}
  = \frac{\tau}{\sqrt2},$$
using the defining identity $\tau = \sigma + 1$. (The first modulus is $\sigma\,|1+i|/2 =
\sigma\sqrt2/2 = \sigma/\sqrt2$.) Hence $\mu(\Phi) \le \tau/\sqrt2 \approx 1.144$. The cell checks the
decomposition reproduces $\Phi$ to machine precision. It is an exact identity over
$\mathbb{Q}(\sqrt2,\sqrt5,i)$, though, and the bonus section settles it there instead.
''')


# =============================================================================
# Cell 11 (code) -- verify the decomposition
# =============================================================================
code(r'''
# === Claim 1: the explicit two-term decomposition is exact ===

S3 = S @ S @ S
decomp = (SIGMA * (1 + 1j) / 2) * S3 + (1 / np.sqrt(2)) * (Z @ H)
# residuals 1.2e-16 (decomposition) and 0.0 (mu)
d_dec = np.abs(decomp - PHI).max()
assert d_dec < 1e-14, f"decomposition != Phi (max |dev| = {d_dec:.2e})"

mu_claimed = SIGMA / np.sqrt(2) + 1 / np.sqrt(2)
d_mu = abs(mu_claimed - TAU / np.sqrt(2))          # residual 0.0
assert d_mu < 1e-14, f"sigma/sqrt2 + 1/sqrt2 != tau/sqrt2 (|dev| = {d_mu:.2e})"

print("Phi reconstructed from (sigma(1+i)/2) S^3 + (1/sqrt2) ZH:")
print(np.round(decomp, 6))
print(f"\nl1 norm of this decomposition = sigma/sqrt2 + 1/sqrt2 = {mu_claimed:.12f}")
print(f"tau/sqrt2                                              = {TAU / np.sqrt(2):.12f}")
print("[ok] decomposition is exact, so mu(Phi) <= tau/sqrt2")
''')


# =============================================================================
# Cell 12 (md) -- Claim 2
# =============================================================================
md(r'''
### Claim 2 — nothing beats it (a primal–dual certificate)

An upper bound alone doesn't prove $\tau/\sqrt2$ is *optimal*. We pin it from both sides.

**Primal (upper bound).** We solve $\min\|c\|_1$ s.t. $Ac=b$ by **IRLS / FOCUSS**: iteratively
reweighted least squares. Reweighting the least-squares solve by the current $|c|$ drives small
coefficients toward zero, and the fixed point is the $\ell_1$ minimizer. Since basis pursuit is
convex, this fixed point is the *global* optimum — and we certify it independently anyway.

**Dual (lower bound).** Basis pursuit has the dual
$$\max_{y}\ \operatorname{Re}\langle y, b\rangle \quad\text{subject to}\quad \|A^\dagger y\|_\infty \le 1.$$
*Weak duality* says any dual-feasible $y$ satisfies $\operatorname{Re}\langle y,b\rangle \le \|c\|_1$
for every primal-feasible $c$ — so a feasible $y$ is a certified lower bound on $\mu$. We build one
from complementary slackness on the support of the primal solution ($A_{\mathrm{supp}}^\dagger y =
\operatorname{sign}(c_{\mathrm{supp}})$) and rescale it into the feasible region. When the resulting
lower bound meets the primal $\|c\|_1$ — here to a gap of $\sim 10^{-12}$ — the value is **certified
optimal**, independently of whether IRLS actually converged. A gap of $10^{-12}$ identifies the
closed form beyond reasonable doubt; the bonus section removes the doubt, by rounding this witness
and re-checking both dual conditions in the field.

Running this for both gates recovers $\mu(\Phi) = \tau/\sqrt2$ and the known control
$\mu(T) = \cos\frac\pi8 + (\sqrt2-1)\sin\frac\pi8$.
''')


# =============================================================================
# Cell 13 (code) -- basis pursuit + dual certificate (verbatim) + run
# =============================================================================
code(r'''
# === Claim 2: basis pursuit (primal) and the dual certificate (verbatim) ===

def min_l1_over_cliffords(A, target, iters=5000, eps=1e-13):
    """Basis pursuit min ||c||_1 s.t. A c = vec(target), by IRLS (FOCUSS).

    The problem is convex, so the fixed point is the global optimum; the
    caller independently certifies it via the dual anyway.
    """
    b = target.flatten()
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    for _ in range(iters):
        w = np.abs(c) + eps
        AW = A * w
        y = np.linalg.solve(AW @ A.conj().T + 1e-15 * np.eye(4), b)
        c = w * (A.conj().T @ y)
    # Every guard in this module measures its deviation and bounds it: allclose
    # and isclose keep rtol=1e-5 alongside any atol, and it scales with the
    # expected operand, not the residual. Neither is called anywhere below.
    d_feas = np.abs(A @ c - b).max()      # residual ~8.7e-16
    assert d_feas < 1e-12, f"IRLS lost feasibility (max |dev| = {d_feas:.2e})"
    return c


def certified_extent(A, target, c):
    """Lower-bound mu via the dual: max Re<y,b> s.t. ||A^H y||_inf <= 1.

    A dual-feasible y is built from the support of the primal solution c
    (A_supp^H y = sign(c_supp)), rescaled into the feasible region. Weak
    duality then gives Re<y,b> <= mu for every decomposition, so a tiny
    primal-dual gap certifies optimality of ||c||_1.
    """
    b = target.flatten()
    support = np.abs(c) > 1e-8
    signs = c[support] / np.abs(c[support])
    y, *_ = np.linalg.lstsq(A[:, support].conj().T, signs, rcond=None)
    y = y / max(1.0, np.abs(A.conj().T @ y).max())   # force dual feasibility
    return np.real(np.vdot(y, b))


# Each Clifford becomes a column of the 4 x 24 design matrix A.
A = np.column_stack([C.flatten() for C in cliffords.values()])

report = {}
print(f"{'gate':4s}  {'mu (extent)':>14s}  {'closed form':>14s}  {'duality gap':>11s}  support")
print("-" * 62)
for name, U, mu_closed in (
    ("T", T, np.cos(np.pi / 8) + (np.sqrt(2) - 1) * np.sin(np.pi / 8)),
    ("Phi", PHI, TAU / np.sqrt(2)),
):
    c = min_l1_over_cliffords(A, U)
    mu = np.abs(c).sum()
    lower = certified_extent(A, U, c)
    assert mu - lower < 1e-9, f"{name}: duality gap {mu - lower:.2e}"
    assert abs(mu - mu_closed) < 1e-9, \
        f"{name}: mu = {mu} != closed form {mu_closed}"   # LP residual ~9.7e-13
    report[name] = mu
    nterms = int((np.abs(c) > 1e-8).sum())
    print(f"{name:4s}  {mu:>14.12f}  {mu_closed:>14.12f}  {mu - lower:>11.1e}  "
          f"{nterms} Cliffords")
print()
print("[ok] mu(Phi) = tau/sqrt2 and mu(T) = cos(pi/8) + (sqrt2-1) sin(pi/8),")
print("     both certified optimal by the dual (gaps ~1e-12).")
''', lifts=("min_l1_over_cliffords", "certified_extent"))


# =============================================================================
# Cell 14 (md) -- per-gate simulation cost
# =============================================================================
md(r'''
## The headline: per-gate simulation cost

With $\mu(\Phi) = \tau/\sqrt2$ certified, every number in the footnote is now just arithmetic.

**Per-gate cost.** Squaring the gate extent gives the extent $\xi(\Phi)$ — the actual per-$\Phi$
factor:
$$\xi(\Phi) \;=\; \mu(\Phi)^2 \;=\; \Bigl(\tfrac{\tau}{\sqrt2}\Bigr)^2 \;=\; \frac{\tau^2}{2}
  \;=\; 1.3090\ldots \;\approx\; 1.31.$$
(Numerically $\tau^2 = 2.618\ldots$, halved is $1.309\ldots$ — this is the $\approx 1.31$ of the
footnote.)

**Exponential form.** Across $m$ golden gates the factor compounds to $\xi(\Phi)^m = (\tau^2/2)^m$.
Taking $\log_2$,
$$\Bigl(\tfrac{\tau^2}{2}\Bigr)^m \;=\; 2^{\,m\log_2(\tau^2/2)} \;=\; 2^{(2\log_2\tau - 1)\,m}
  \;\approx\; 2^{0.39\,m},$$
where $\log_2(\tau^2/2) = 2\log_2\tau - 1 \approx 0.388$ (the $-1$ is $\log_2\tfrac12$). So $1.31$ and
$2^{0.39}$ are the *same* per-gate number written two ways: $2^{0.388} \approx 1.31$. A Clifford${}+\Phi$
circuit with $m$ golden gates therefore simulates classically in $2^{0.39m}\,\mathrm{poly}(n)$ time,
and via the double-coset decomposition a circuit of Cliffords plus $k$ arbitrary elements of $2I$ is
capped at the same $2^{0.39k}\,\mathrm{poly}(n)$.

For comparison the $T$ gate costs $\mu(T)^2 = 4 - 2\sqrt2 \approx 1.17$, i.e. $2^{0.23}$ per gate: $\Phi$ is
slightly steeper than $T$, but the same order — the price of one golden gate is comparable to one $T$.
''')


# =============================================================================
# Cell 15 (code) -- gamma + mu^2
# =============================================================================
code(r'''
# === Per-gate simulation cost exponents and ratios ===

gamma_phi = 2 * np.log2(report["Phi"])
gamma_t = 2 * np.log2(report["T"])

# gamma = 2 log2 mu, so mu(Phi)^2 = tau^2/2 and mu(T)^2 = 4 - 2 sqrt2.  The
# control gets one too: a control pinned less hard than the thing it controls is
# not controlling it.  Residuals ~2.4e-12, set by the LP's precision on mu.
d_phi = abs(gamma_phi - (2 * np.log2(TAU) - 1))
d_t = abs(gamma_t - np.log2(4 - 2 * np.sqrt(2)))
assert d_phi < 1e-9, f"gamma_Phi off 2 log2 tau - 1 by {d_phi:.2e}"
assert d_t < 1e-9, f"gamma_T off log2(4 - 2 sqrt2) by {d_t:.2e}"

# Per-gate cost = extent = mu^2, in plain form (~1.31) and exponent form (2^0.39).
print(f"mu(Phi)^2 = (tau/sqrt2)^2 = tau^2 / 2 = {report['Phi']**2:.6f}   (per-gate cost; ~1.31)")
print(f"mu(T)^2   = 4 - 2 sqrt2               = {report['T']**2:.6f}   (per-gate cost; ~1.17)")
print()
print(f"Phi per-gate exponent: 2^{gamma_phi:.4f}  (= 2^(2 log2 tau - 1),   off {d_phi:.1e})")
print(f"T   per-gate exponent: 2^{gamma_t:.4f}  (= 2^log2(4 - 2 sqrt2), off {d_t:.1e})")
print()
# The two readings are one number: 2^0.388... reproduces 1.309...
d_rt = abs(2 ** gamma_phi - report["Phi"] ** 2)
assert d_rt < 1e-12, f"2^gamma_Phi != mu(Phi)^2 by {d_rt:.2e}"
print(f"consistency: 2^{gamma_phi:.4f} = {2 ** gamma_phi:.6f} = mu(Phi)^2   (same number, two forms)")
print(f"total cost for m Phi gates: 2^({gamma_phi:.4f} m) poly(n)")
''')


# =============================================================================
# Cell 16 (md) -- cross-check against the production script
# =============================================================================
md(r'''
## Cross-check against the production script

Everything above was staged by hand from the script's pieces — the copies of its functions are checked
against `inspect.getsource` when this notebook is built, so "verbatim" above is enforced rather than
merely claimed. What is left for run time is agreement of the *values*: we import the production
module, confirm our golden gate is byte-identical to its, that its `proj_key` and ours return the same
key for $\Phi$, and that its Clifford closure still has 24 elements — then let its `main()` reproduce
the canonical verification report. If the module's stdout matches the numbers printed above, the
hand-staged walkthrough and the production script agree.

(`phi_simulation_cost.py` writes nothing; `main()` raises on the first failed claim and otherwise
prints its `[ok]` report.)
''')


# =============================================================================
# Cell 17 (code) -- import + anti-drift + main()
# =============================================================================
code(r'''
# === Diff our hand-staged objects against the production module, then run it ===

import phi_simulation_cost as psc

# Anti-drift guard: the cells above are lifted from phi_simulation_cost.py, so
# its primitives must agree with what we rebuilt here exactly -- same arithmetic
# run twice, so the only honest residual is 0.0, not a tolerance.
d_phi = np.abs(psc.PHI - PHI).max()
assert d_phi == 0.0, f"Phi drifted (max |dev| = {d_phi:.2e})"
assert proj_key(PHI) == psc.proj_key(PHI), "proj_key drifted"
assert len(psc.clifford_group()) == 24, "Clifford closure drifted"

print("anti-drift checks passed; running the production report:\n")
psc.main()
''')


# =============================================================================
# Cell 18 (md) -- Bonus: the same claims without a tolerance
# =============================================================================
md(r"""
---

## Bonus — the same three claims, without a tolerance

Everything above is float64, and every verdict it reached is correct. But look at what *kind* of
statement each claim is. Claim 1 is an algebraic **identity**. Claim 3 is a discrete **membership**
question — is this matrix one of 24? Neither has an approximation anywhere in its statement, so
neither needs one in its proof. Even Claim 2's conclusion is exact: $\mu(\Phi)$ either *is*
$\tau/\sqrt2$ or it is not.

What makes the lift cheap is that the whole file lives in one small number field. $H$ contributes
$1/\sqrt2$, $S$ and $T$ contribute $i$, and $\Phi$ contributes $\tau$, so every matrix here has
entries in $\mathbb{Q}(\sqrt2,\sqrt5,i)$. Splitting real and imaginary parts lands both in
$\mathbb{Q}(\sqrt2,\sqrt5)$ — a degree-4 field with basis $\{1,\sqrt2,\sqrt5,\sqrt{10}\}$, which is
exactly what `main.py`'s `_to_basis` reduces a number to. That 4-tuple is a **canonical form**: two
expressions denoting the same number give the same tuple, always, and the tuple is *hashable*.

Two properties, and both do work below. **Canonical** is what lets a tuple replace a rounding grid
as a dictionary key. **Fail-loud** is what makes the field declaration a checked claim rather than
an assumption: hand `_to_basis` a $\sqrt3$ and it raises instead of guessing.
""")


# =============================================================================
# Cell 19 (code) -- canonical keys replace the rounding grid
# =============================================================================
code(r"""
# === Identity by decision, not by rounding ===

import sympy as sp
from sympy import Matrix, Rational, sqrt, eye
from main import _to_basis, _is_zero

# Two spellings of one number in Q(sqrt2, sqrt5). Rounding agrees to 9 places;
# the canonical form agrees exactly, and would agree at any depth of nesting.
a, b = sqrt(2) * (1 + sqrt(5)) / 2, sqrt(2) / 2 + sqrt(10) / 2
print(f"a = {a}\nb = {b}")
print(f"  same 4-tuple over (1, sqrt2, sqrt5, sqrt10): {_to_basis(a)} == {_to_basis(b)}"
      f"  ->  {_to_basis(a) == _to_basis(b)}")

# The exact closure of <S, H>, keyed canonically, against the float closure.
exact = psc.exact_clifford_group()
_, hit, miss = psc.check_exact_cliffords(cliffords)
print(f"\nexact projective Clifford group: {len(exact)} elements")
print(f"  matched to the float group up to phase, |tr(A^dag B)| = {hit:.12f} (must be 2)")
print(f"  next-best overlap for any other pair  = {miss:.6f} (= sqrt2)")
print("  -- so the correspondence is a bijection with room to spare on both sides")
""")


# =============================================================================
# Cell 20 (md) -- Claims 1 and 3, decided
# =============================================================================
md(r"""
### Claims 1 and 3, decided rather than measured

With canonical keys in hand both claims collapse to one-liners. Claim 3 asks whether
$\Phi P \Phi^\dagger$ is one of the 24 — a `in` test against a `dict` whose keys are now decisions.
Claim 1 asks whether a difference of matrices is zero — `_is_zero` on each entry.

Worth noting how far from the boundary the float version was: $T$'s conjugates land within
$2\times10^{-16}$ of a Clifford, and $\Phi$'s sit $0.33$ to $0.51$ away from the nearest one. The
float verdict was never in danger. It was simply never *proved*, and that is a different complaint.
""")


# =============================================================================
# Cell 21 (code) -- the exact level-3 table
# =============================================================================
code(r"""
# === Claim 3, exactly: is U P U^dag a Clifford? ===

print(f"{'gate':>5s}  {'X':>12s} {'Y':>12s} {'Z':>12s}")
for name, U in (("T", psc.T_SYM), ("Phi", psc.PHI_SYM)):
    row = []
    for P in (psc.X_SYM, psc.Y_SYM, psc.Z_SYM):
        row.append(psc.exact_is_clifford(sp.expand(U * P * U.conjugate().T), exact))
    print(f"{name:>5s}  " + " ".join(f"{str(v):>12s}" for v in row))
print("\nT conjugates every Pauli into the Cliffords; Phi conjugates no X, Y, Z.")

# === Claim 1, exactly ===
decomp = ((psc.SIG_SYM * (1 + sp.I) / 2) * psc.S_SYM ** 3
          + (1 / sqrt(2)) * (psc.Z_SYM * psc.H_SYM))
print(f"\ndecomposition - Phi is entrywise zero: "
      f"{all(_is_zero(e) for e in sp.expand(decomp - psc.PHI_SYM))}")
print(f"sigma/sqrt2 + 1/sqrt2 - tau/sqrt2 == 0: "
      f"{_is_zero(sp.expand(psc.SIG_SYM / sqrt(2) + 1 / sqrt(2) - psc.TAU_SYM / sqrt(2)))}")
""")


# =============================================================================
# Cell 22 (md) -- the certificate, in the field
# =============================================================================
md(r"""
### Claim 2 — from a certificate to a theorem

The dual witness $y$ that IRLS produced is a float vector, but it rounds to something startlingly
simple:
$$y \;=\; \Bigl(\tfrac{\sqrt2}{3} + \tfrac{i\sqrt2}{6},\ \ \tfrac{\sqrt2}{6},\ \
  -\tfrac{\sqrt2}{6},\ \ \tfrac{\sqrt2}{3} - \tfrac{i\sqrt2}{6}\Bigr).$$
Once rounded, the search has done its job and can be discarded — this is the same **float-search /
exact-confirm** discipline `povm_properties.py` uses for its face finder. Both dual conditions are
then re-asked in the field:

* **Feasibility**, $|\langle C_j, y\rangle| \le 1$ for all 24 Cliffords. This is an *order* question,
  not an equality, so it takes the other tool — a sign decided by interval refinement. It is asked of
  the *squares*, and those turn out rational: $|\langle C_j, y\rangle|^2 \in \{0, \tfrac19, \tfrac29,
  \tfrac49, \tfrac89, 1\}$, two of them tight. (The moduli themselves are not rational — the same
  multiset unsquared contains $\tfrac{\sqrt2}{3}$ and $\tfrac{2\sqrt2}{3}$.)
* **The objective**, $\operatorname{Re}\langle y, \operatorname{vec}\Phi\rangle = \tau/\sqrt2$ — an
  *equality* question, so it goes to the field.

Weak duality then gives $\mu(\Phi) \ge \tau/\sqrt2$ for **every** Clifford decomposition, and Claim 1
gives $\le$. The two meet, so $\mu(\Phi) = \tau/\sqrt2$ is a theorem with a certificate a reader can
check by hand — six rational squares and one identity.

One detail is more interesting than it looks. The $T$ control's optimum, $\cos\frac\pi8 +
(\sqrt2-1)\sin\frac\pi8$, is **not** in $\mathbb{Q}(\sqrt2,\sqrt5)$: $\cos\frac\pi8$ generates
$\mathbb{Q}\bigl(\sqrt{2-\sqrt2}\bigr)$, a different degree-4 field. $T$ is an eighth root of unity's
gate and $\Phi$ is a golden one, and their extents remember it.
""")


# =============================================================================
# Cell 23 (code) -- the exact certificate, both gates
# =============================================================================
code(r"""
# === Claim 2, exactly: upper bound and lower bound in the field ===

for name, target in (("Phi", psc.PHI_SYM), ("T", psc.T_SYM)):
    terms = psc.check_exact_decomposition(name, target, exact)
    tight, slack = psc.check_exact_extent(name, target, exact)
    print(f"mu({name}) = {psc._EXTENT_LABEL[name]}")
    print(f"   upper: an exact {terms}-term Clifford decomposition of that l1 norm")
    print(f"   lower: a dual witness, {tight} of 24 constraints tight, largest slack {slack}")

# The feasibility values for Phi, as a multiset -- the whole certificate.
mags = []
for C in exact.values():
    ip = sp.expand(sum(sp.conjugate(list(C)[i]) * psc.DUAL_WITNESS["Phi"][i]
                       for i in range(4)))
    re, im = ip.as_real_imag()
    mags.append(sp.nsimplify(sp.expand(re ** 2 + im ** 2)))
print(f"\n|<C,y>|^2 over all 24 Cliffords: {sorted(set(mags), key=float)}")
print(f"   every one <= 1, and {mags.count(sp.Integer(1))} attain it")
""")


# =============================================================================
# Cell 24 (md) -- fail-loud
# =============================================================================
md(r"""
### The half that is easy to forget

A canonical form on its own is only half a standard. The other half is that it **raises** on input
outside its field rather than returning a plausible answer — which is what turns "these gates lie in
$\mathbb{Q}(\sqrt2,\sqrt5,i)$" from an assumption in a comment into a claim the code checks on every
call. Mistype a gate and nothing silently passes.

The two fields refusing each other's generators is the same effect the POVM walkthrough shows for
the icosahedron and dodecahedron, one level up.
""")


# =============================================================================
# Cell 25 (code) -- fail-loud demo
# =============================================================================
code(r"""
# === Out-of-field input raises; it does not round ===

for label, fn in (
    ("sqrt(3) into the atlas's field", lambda: _to_basis(sqrt(3))),
    ("sqrt(3) into exact_proj_key", lambda: psc.exact_proj_key(Matrix([[sqrt(3), 0], [0, 1]]))),
    ("sqrt(5) into the T control's field", lambda: psc._T_FIELD.from_sympy(sqrt(5))),
    ("sqrt(2) into the T control's field", lambda: psc._T_FIELD.from_sympy(sqrt(2))),
):
    try:
        fn()
        print(f"  accepted        {label}")
    except Exception as e:
        print(f"  {type(e).__name__:<15s} {label}")
print("\n(sqrt2 IS in Q(sqrt(2 - sqrt2)), so that one is accepted -- the test")
print(" distinguishes membership, not spelling.)")
""")


# =============================================================================
# Cell 26 (md) -- Closing notes
# =============================================================================
md(r'''
## Closing notes

**What was shown.** Three claims, each settled twice — once numerically, once in the field:

| Claim | Statement | Numerically | Exactly |
|---|---|---|---|
| 1 | $\Phi = \frac{\sigma(1+i)}{2}S^3 + \frac{1}{\sqrt2}ZH$ | agreement to $10^{-16}$ | entrywise zero over $\mathbb{Q}(\sqrt2,\sqrt5,i)$ |
| 2 | $\mu(\Phi) = \tau/\sqrt2$ (and $\mu(T)$ as control) | basis pursuit + dual certificate, gap $\sim10^{-12}$ | exact decomposition above, exact dual witness below |
| 3 | $\Phi$ outside the third Clifford level | membership on a $10^{-9}$ rounding grid | membership on canonical hashable keys |

Together they back the footnote's reading of $2I$ as *almost Clifford*: the gate extent
$\mu(\Phi)=\tau/\sqrt2$ makes the per-gate classical-simulation cost $\tau^2/2\approx1.31$
(i.e. $2^{0.39}$), comparable to — if slightly above — the $T$ gate's $1.17$, while Claim 3 confirms
that this minimal magic is nonetheless genuine and gate-set-specific.

**A note on the two routes.** The float pipeline is kept, not superseded, and the two are asserted
to agree — agreement between two independently derived answers is worth more than either alone, and
the exact witness has nothing to confirm if the search that proposed it stops running. The division
of labour is the useful part to carry away: **the float layer searches, the exact layer decides.**
IRLS finds a dual witness; rounding it and re-checking the two dual conditions in the field is what
turns "$\mu(\Phi)$ is within $10^{-12}$ of $\tau/\sqrt2$" into "$\mu(\Phi)$ is $\tau/\sqrt2$".

Note also which tool answers which question. A number field decides **equality** and nothing else —
so dual *feasibility*, being an inequality, needs a sign test instead, and the two halves of the same
certificate are settled by two different mechanisms. Neither is a better version of the other.

To run the production script end-to-end:

```
cd code && uv run phi_simulation_cost.py
```
''')


# =============================================================================
# Assemble and write
# =============================================================================
nb.cells = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "pygments_lexer": "ipython3",
}

out = Path(__file__).parent / "phi_walkthrough.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f"Wrote {out} ({len(cells)} cells)")
