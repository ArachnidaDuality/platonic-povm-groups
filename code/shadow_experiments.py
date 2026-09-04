"""
Shadow estimation with the Platonic-solid POVMs -- numerical demonstration.

The numerical study behind the Discussion chapter (Section 5.2) and its
appendix (F.3).  Only the two-protocol study and the gate-noise series it
carries are printed there; the landscape, dual-optimization and calibration
studies below are repository-only.

The reframing that carries the file: a POVM is its Bloch-vertex array `s`
(V, 3).  Effects, probabilities, dual frames, noise and gate circuits are all
small linear-algebra statements about `s`, and the n-qubit Born distribution
is one einsum of rho against the per-wire effect tensors -- so means, second
moments, bias factors and twirl scalars are evaluated *exactly*, and the
Monte Carlo only ever measures convergence to targets already known.

Map of the checks.  Each function's docstring is the account of its own:

  experiment_1          variance landscape at the canonical dual: five POVMs,
                        Pauli observables and the TFIM energy, on structured
                        and Haar-random 4-qubit states; no optimization, no
                        noise.
  exact_backbone        the theorem that landscape illustrates, verified
                        directly: the variance sees the vertices only through
                        moments of order <= 3.
  fourth_moment_ladder  where the choice of solid does reappear: the tails.
  experiment_2          dual-frame optimization (a linearly-constrained QP)
  exact_exp2_ratios     at three levels; every table ratio also exact.
  experiment_3          robust |0> calibration under depolarizing noise, and
  blindness             its structural blind spot -- channels fixing |0> --
                        plus the free anisotropy diagnostic.
  twirl_check           randomized-projective's exact estimator channel is
                        depolarizing at eta = T_zz/3 (Schur).
  two_protocol_twirl    Section 4.2.4's two protocols as exact estimator
                        channels, per solid and per draw; cross-checked
                        against randomized_implementations.py.
  gate_noise_twirl      the twirl's one assumption -- noise independent of
                        the drawn rotation g -- dropped, and priced with the
                        atlas's own circuits.
  nscaling              exact estimator variance for n = 4..16: nothing above
                        is an n = 4 artifact.

CONVENTION, since the two modules differ: eta here is the thesis's
CALIBRATION SCALAR (ideal = Id/3, so noiseless eta = 1/3, as everywhere else
in this file and in Appendix F.3.1).  randomized_implementations.py reports
the same physics as kappa = 3 eta, the ESTIMATOR CHANNEL's multiplier,
ideal 1.  Values cross-check after that factor; premia, being ratios, need
no conversion.

Dual routing (deliberate): duals are pure post-processing, so for a
composite observable each term here uses the dual matched to its own Pauli
letter at each site, rather than one dual per qubit chosen by that qubit's
most frequent letter -- unbiasedness is unaffected and the optimized levels
can only improve.  The two routings agree on single-term observables; on
the TFIM energy they do not.

Writes data/shadow_experiments.npz: the Monte-Carlo families exp1/, exp1_haar/,
exp2/, exp3/, blind/ (blind/aniso/ included -- two of its three columns are
drawn from the shared stream, only the third is exact), and the exact families
exact/, exact4/, exact_ratio/, twirl1/, twirl2/, gatenoise/, nscale/.  No .tex
output.

WARNING: any edit must keep the pre-existing npz keys BIT-IDENTICAL -- the
Monte-Carlo stream is untouchable.  The one rng is consumed strictly in
main()'s order (make_states -> experiment_1 -> experiment_2 -> experiment_3 ->
blindness); inserting, removing or reordering a single draw rewrites every
sampled number downstream of it, and shadow_report.py's tables and figures
with them.

    cd code && uv run shadow_experiments.py

Exits 0 with a report if every check passes; raises on the first failure.
"""

import time

import numpy as np
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

DATA = Path(__file__).parent / "data"

RNG_SEED = 20260612
N_QUBITS = 4
R_REPS = 2000         # Monte Carlo replications per (POVM, state, setting)
T_MAIN = 1000         # shots per replication (Experiments 1 and 3)
T_EXP2 = 5000         # shots per replication (Experiment 2 headline)
RC_CAL = 1000         # calibration shots per replication (Experiment 3)
N_HAAR = 10           # Haar-random states in Experiment 1
G_CRIT = 1.0          # TFIM transverse field (critical point)
TRAIN_G = (0.3, 0.5, 0.7, 1.0, 1.5)   # training TFIM fields (+ GHZ)
DEPOL_RATES = (0.0, 0.01, 0.05, 0.1, 0.2)
BLIND_RATES = (0.01, 0.05, 0.1, 0.2)
GATE_GAMMAS = (0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
GAMMA_DIL = 0.05      # twirled-native series: modeled dilation pre-channel
INDEP_DELTAS = (0.01, 0.05, 0.2)   # gate-INDEPENDENT control: swept, not pinned
NSCALE_NS = (4, 6, 8, 10, 12, 14, 16)

SOLIDS = ("tetrahedron", "octahedron", "cube", "icosahedron", "dodecahedron")
ANTIPODAL = SOLIDS[1:]          # vertex sets closed under n -> -n

I2 = np.eye(2, dtype=complex)
PAULI = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
AXIS = {"X": 0, "Y": 1, "Z": 2}


# ---------------------------------------------------------------------------
# POVM data
# ---------------------------------------------------------------------------
def load_povms():
    """Load Bloch vertices + effects from the symbolic-derived exports."""
    povms = {}
    for name in SOLIDS:
        d = np.load(DATA / f"povm_{name}.npz")
        s, E = d["vertices"], d["elements"]
        V = len(s)
        # Deviations are measured and bounded, not handed to allclose, which
        # keeps rtol=1e-5 alongside any atol and scales it with the expected
        # operand.  Worst over the five solids: 1.1e-16 (norm), 1.8e-15
        # (2-design), 2.2e-16 (completeness).  The zero-sum check keeps
        # allclose: its reference is 0.0, so atol really is the bound.
        d_norm = np.abs(np.linalg.norm(s, axis=1) - 1.0).max()
        assert d_norm < 1e-12, f"{name}: vertices not unit norm " \
            f"(max |dev| = {d_norm:.2e})"
        assert np.allclose(s.sum(axis=0), 0.0, atol=1e-12), name
        d_design = np.abs(s.T @ s - (V / 3) * np.eye(3)).max()      # 2-design
        assert d_design < 1e-12, f"{name}: not a spherical 2-design " \
            f"(max |dev| = {d_design:.2e})"
        d_complete = np.abs(E.sum(axis=0) - I2).max()            # completeness
        assert d_complete < 1e-12, f"{name}: POVM not complete " \
            f"(max |dev| = {d_complete:.2e})"
        # E[k] is built from s[k], at the same index.  The three checks above
        # are permutation-invariant sums; this is the only one row order breaks.
        E_from_s = (I2[None] + np.einsum(
            "kn,nab->kab", s, np.stack([PAULI["X"], PAULI["Y"], PAULI["Z"]]))) / V
        d_align = np.abs(E - E_from_s).max()
        assert d_align < 1e-12, f"{name}: effects not row-for-row aligned with " \
            f"vertices (max |dev| = {d_align:.2e})"
        povms[name] = {"V": V, "s": s, "E": E}
    print(f"  POVMs loaded: " + ", ".join(
        f"{n} (V={p['V']})" for n, p in povms.items()))
    return povms


# ---------------------------------------------------------------------------
# Observables and states
# ---------------------------------------------------------------------------
def kron_all(ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def term_operator(sites):
    """Dense operator for a Pauli term given as [(site, letter), ...]."""
    letters = ["I"] * N_QUBITS
    for i, a in sites:
        letters[i] = a
    return kron_all([PAULI[a] for a in letters])


def obs_operator(obs):
    return sum(c * term_operator(t) for c, t in obs)


def exact_value(rho, obs):
    val = np.trace(rho @ obs_operator(obs))
    assert abs(val.imag) < 1e-12
    return val.real


def tfim_terms(g):
    """4-qubit periodic TFIM:  H = -sum ZZ - g sum X."""
    terms = [(-1.0, [(i, "Z"), ((i + 1) % N_QUBITS, "Z")])
             for i in range(N_QUBITS)]
    terms += [(-g, [(i, "X")]) for i in range(N_QUBITS)]
    return terms


def tfim_ground_energy_exact(n, g):
    """Closed-form ground energy of the periodic TFIM, by Jordan-Wigner.

    H = -sum_i Z_i Z_{i+1} - g sum_i X_i on a ring of n sites is free fermions.
    For even n the ground state sits in the even-parity (Neveu-Schwarz) sector,
    whose momenta are the half-integers k = (2m+1) pi / n, and

        E_0 = -(1/2) sum_k eps_k,   eps_k = 2 sqrt(1 + g^2 - 2 g cos k).

    This exists to be an INDEPENDENT VERIFIER of the two ground states this
    file builds, and it shares no mechanism with either: it constructs no
    Hamiltonian, diagonalizes nothing, and samples nothing.  Both routes are
    checked against it -- the dense eigh at n = 4, whose state anchors every
    number in Appendix F and whose energy the thesis quotes as -5.23, and the
    sparse Lanczos across n = 4..16 behind the n-scaling figure.
    """
    ks = (np.pi * (2 * m + 1) / n for m in range(n))
    return -0.5 * sum(2 * np.sqrt(1 + g ** 2 - 2 * g * np.cos(k)) for k in ks)


OBSERVABLES = {
    "Z0":     [(1.0, [(0, "Z")])],
    "X0":     [(1.0, [(0, "X")])],
    "Z0Z1":   [(1.0, [(0, "Z"), (1, "Z")])],
    "X0X1":   [(1.0, [(0, "X"), (1, "X")])],
    "Z0Z1Z2": [(1.0, [(0, "Z"), (1, "Z"), (2, "Z")])],
    "E_TFIM": tfim_terms(G_CRIT),
}


def tfim_ground_state(g):
    H = obs_operator(tfim_terms(g))
    _, vecs = np.linalg.eigh(H)
    return vecs[:, 0]


def density(psi):
    return np.outer(psi, psi.conj())


def make_states(rng):
    ghz = np.zeros(16, dtype=complex)
    ghz[0] = ghz[15] = 1 / np.sqrt(2)
    plus = np.array([1, 1]) / np.sqrt(2)
    plus_i = np.array([1, 1j]) / np.sqrt(2)
    zero, one = np.array([1, 0]), np.array([0, 1])
    product = kron_all([plus, zero, plus_i, one]).astype(complex)
    states = {
        "TFIM": density(tfim_ground_state(G_CRIT)),
        "GHZ": density(ghz),
        "product": density(product),
    }
    haar = []
    for _ in range(N_HAAR):
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        haar.append(density(psi / np.linalg.norm(psi)))
    return states, haar


def site_bloch(rho):
    """Per-site Bloch vectors (N_QUBITS, 3) of the reduced states."""
    r = np.empty((N_QUBITS, 3))
    for i in range(N_QUBITS):
        for a in "XYZ":
            r[i, AXIS[a]] = exact_value(rho, [(1.0, [(i, a)])])
    return r


# ---------------------------------------------------------------------------
# Noise channels: affine Bloch maps r -> T r + t, applied i.i.d. per qubit
# before the measurement (equivalently, adjoint-applied to the effects).
# ---------------------------------------------------------------------------
def chan_depolarizing(p):
    return (1 - p) * np.eye(3), np.zeros(3)


def chan_dephasing(p):
    """rho -> (1-p) rho + p Z rho Z."""
    return np.diag([1 - 2 * p, 1 - 2 * p, 1.0]), np.zeros(3)


def chan_amp_damping(g):
    return np.diag([np.sqrt(1 - g), np.sqrt(1 - g), 1 - g]), \
        np.array([0.0, 0.0, g])


def chan_tilted_depolarizing(p, theta):
    """Depolarizing composed with a rotation about x: tilts the z-axis."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return (1 - p) * R, np.zeros(3)


def noisy_effects(s, T, t):
    """Heisenberg-picture effects E~_k = (1/V)[(1 + s_k.t) Id + (T^T s_k).sigma]."""
    V = len(s)
    c = 1.0 + s @ t
    s_eff = s @ T                       # rows are T^T s_k
    E = (c[:, None, None] * I2
         + s_eff[:, 0, None, None] * PAULI["X"]
         + s_eff[:, 1, None, None] * PAULI["Y"]
         + s_eff[:, 2, None, None] * PAULI["Z"]) / V
    return E


# ---------------------------------------------------------------------------
# Born distributions and sampling
# ---------------------------------------------------------------------------
def born_tensor(rho, E):
    """Joint outcome distribution (V,V,V,V) for i.i.d. per-qubit effects E."""
    p = np.einsum("ijklmnop,ami,bnj,cok,dpl->abcd",
                  rho.reshape([2] * (2 * N_QUBITS)), E, E, E, E)
    assert np.abs(p.imag).max() < 1e-12
    p = p.real
    assert p.min() > -1e-12 and abs(p.sum() - 1) < 1e-10
    return np.clip(p, 0.0, None)


def sample_outcomes(p, n_shots, rng):
    """(n_shots, N_QUBITS) outcome indices drawn from the joint tensor."""
    cum = np.cumsum(p.ravel())
    cum[-1] = 1.0
    idx = np.searchsorted(cum, rng.random(n_shots), side="right")
    return np.stack(np.unravel_index(idx, p.shape), axis=1)


def single_qubit_probs(s, T, t, r0):
    """Outcome distribution of one noisy POVM on a product-state qubit r0."""
    q = (1.0 + s @ (T @ r0 + t)) / len(s)
    assert q.min() > -1e-12 and abs(q.sum() - 1) < 1e-12
    return np.clip(q, 0.0, None)


# ---------------------------------------------------------------------------
# Dual frames.  D_k = alpha_k Id + beta_k . sigma, packed v = [alpha, beta].
# The frame condition  sum_k tr(E_k rho) D_k = rho  (for all rho), with
# effects E_k = (1/V)(Id + s_k . sigma), is the 16-equation linear system
# built in frame_system().  The canonical dual is alpha = 1/2, beta = 3/2 s.
# ---------------------------------------------------------------------------
def canonical_dual(s):
    return 0.5 * np.ones(len(s)), 1.5 * s


def robust_canonical_dual(s, eta):
    """Inverts effective shrinkage eta instead of 1/3 (eta = 1/3 is noiseless)."""
    return 0.5 * np.ones(len(s)), s / (2 * eta)


def frame_system(s_eff):
    """Constraint system A v = b for duals of effects (1/V)(Id + s_eff.sigma)."""
    V = len(s_eff)
    A = np.zeros((16, 4 * V))
    b = np.zeros(16)
    A[0, :V] = 1.0                       # sum alpha = V/2
    b[0] = V / 2
    for j in range(3):                   # sum alpha_k s_eff_kj = 0
        A[1 + j, :V] = s_eff[:, j]
    for i in range(3):                   # sum_k beta_ki = 0
        A[4 + i, V + i::3] = 1.0
    row = 7
    for i in range(3):                   # sum_k beta_ki s_eff_kj = (V/2) d_ij
        for j in range(3):
            A[row, V + i::3] = s_eff[:, j]
            b[row] = V / 2 if i == j else 0.0
            row += 1
    return A, b


def optimize_dual(s_eff, pbar, axis):
    """Variance-minimizing dual for one Pauli letter.

    Minimizes the second moment sum_k pbar_k tr(sigma_axis D_k)^2 over the
    affine family of frame duals (the mean is fixed by unbiasedness, so this
    is the variance).  Quadratic objective + linear constraints = particular
    solution + null-space linear solve.
    """
    V = len(s_eff)
    A, b = frame_system(s_eff)
    v_p, *_ = np.linalg.lstsq(A, b, rcond=None)
    _, sv, Vt = np.linalg.svd(A)
    rank = int((sv > 1e-10 * sv[0]).sum())
    N = Vt[rank:].T                            # null-space basis (4V, 4V-rank)
    q = np.zeros(4 * V)                        # objective: tr(sigma_a D_k) = 2 beta_ka
    q[V + axis::3] = 4.0 * pbar
    M = N.T @ (q[:, None] * N)
    rhs = -N.T @ (q * v_p)
    c, *_ = np.linalg.lstsq(M, rhs, rcond=None)
    v = v_p + N @ c
    assert np.linalg.norm(A @ v - b) < 1e-9
    return v[:V], v[V:].reshape(V, 3)


def dual_frame_residual(alpha, beta, s):
    """max |sum_k tr(E_k rho) D_k - rho| over random states (unbiasedness)."""
    rng = np.random.default_rng(5)
    V = len(s)
    worst = 0.0
    for _ in range(5):
        r = rng.standard_normal(3)
        r /= np.linalg.norm(r) / rng.random()       # random point in the ball
        rho = (I2 + r[0] * PAULI["X"] + r[1] * PAULI["Y"]
               + r[2] * PAULI["Z"]) / 2
        pk = (1.0 + s @ r) / V
        recon = sum(
            pk[k] * (alpha[k] * I2 + beta[k, 0] * PAULI["X"]
                     + beta[k, 1] * PAULI["Y"] + beta[k, 2] * PAULI["Z"])
            for k in range(V))
        worst = max(worst, np.abs(recon - rho).max())
    return worst


# ---------------------------------------------------------------------------
# Estimator evaluation.  A "dual assignment" is an array lut (N_QUBITS, 3, V):
# lut[i, a, k] = tr(sigma_a D^{(i,a)}_k), the per-shot value contributed by
# site i when reading Pauli letter a from outcome k.  Identity sites are
# skipped: every dual used here keeps tr D_k = 1 (canonical alpha = 1/2),
# so identity sites contribute exactly 1 with zero variance.
# ---------------------------------------------------------------------------
def lut_uniform(x_axes):
    """Same per-letter values on every site; x_axes has shape (3, V)."""
    return np.broadcast_to(x_axes, (N_QUBITS,) + x_axes.shape)


def lut_canonical(s):
    return lut_uniform(3.0 * s.T)


def lut_robust_canonical(s, eta):
    return lut_uniform(s.T / eta)


def lut_from_letter_duals(betas):
    """betas: dict axis -> (V, 3) optimized beta for that letter."""
    V = next(iter(betas.values())).shape[0]
    x = np.empty((3, V))
    for a in range(3):
        x[a] = 2.0 * betas[a][:, a]
    return lut_uniform(x)


def shot_estimates(outcomes, obs, lut):
    total = np.zeros(len(outcomes))
    for coeff, sites in obs:
        v = np.full(len(outcomes), coeff)
        for i, a in sites:
            v *= lut[i, AXIS[a]][outcomes[:, i]]
        total += v
    return total


def exact_estimator_mean(p, obs, lut):
    """E[single-shot estimate], summed exactly over the outcome tensor."""
    V = p.shape[0]
    total = 0.0
    for coeff, sites in obs:
        f = [np.ones(V)] * N_QUBITS
        for i, a in sites:
            f = list(f)
            f[i] = lut[i, AXIS[a]]
        total += coeff * np.einsum("abcd,a,b,c,d", p, *f)
    return total


def exact_second_moment(p, obs, lut):
    """E[(single-shot estimate)^2], summed exactly over the outcome tensor.

    Same contraction as exact_estimator_mean, applied to every ordered pair
    of terms: a site hit by both terms multiplies its two letter-values.
    Together with the exact mean this makes single-shot variances -- and so
    every variance identity in the appendix -- deterministic statements.
    """
    V = p.shape[0]
    total = 0.0
    for c1, sites1 in obs:
        for c2, sites2 in obs:
            f = [np.ones(V) for _ in range(N_QUBITS)]
            for i, a in sites1:
                f[i] = f[i] * lut[i, AXIS[a]]
            for i, a in sites2:
                f[i] = f[i] * lut[i, AXIS[a]]
            total += c1 * c2 * np.einsum("abcd,a,b,c,d", p, *f)
    return total


def rep_stats(flat, truth):
    """bias^2 / variance / MSE of the T-shot mean across R replications."""
    means = flat.reshape(R_REPS, -1).mean(axis=1)
    err2 = (means - truth) ** 2
    return {
        "bias2": (means.mean() - truth) ** 2,
        "var": means.var(ddof=1),
        "mse": err2.mean(),
        "se_mse": err2.std(ddof=1) / np.sqrt(R_REPS),
    }


# ---------------------------------------------------------------------------
# Experiment 1: variance landscape at the canonical dual
# ---------------------------------------------------------------------------
def experiment_1(povms, states, haar, rng):
    print("\nExperiment 1: variance landscape, canonical dual "
          f"(shots={T_MAIN}, reps={R_REPS})")
    obs_names = list(OBSERVABLES)
    truths = {(sn, on): exact_value(rho, OBSERVABLES[on])
              for sn, rho in states.items() for on in obs_names}

    mse = {}        # (povm, state, obs) -> stats
    haar_mse = {}   # (povm, obs) -> mean MSE over Haar states
    for pname, povm in povms.items():
        lut = lut_canonical(povm["s"])
        for sname, rho in states.items():
            p = born_tensor(rho, povm["E"])
            for on in obs_names:   # deterministic unbiasedness, no sampling
                em = exact_estimator_mean(p, OBSERVABLES[on], lut)
                assert abs(em - truths[(sname, on)]) < 1e-9, (pname, sname, on)
            out = sample_outcomes(p, R_REPS * T_MAIN, rng)
            for on in obs_names:
                mse[(pname, sname, on)] = rep_stats(
                    shot_estimates(out, OBSERVABLES[on], lut),
                    truths[(sname, on)])
        acc = {on: [] for on in obs_names}
        for rho in haar:
            p = born_tensor(rho, povm["E"])
            out = sample_outcomes(p, R_REPS * T_MAIN, rng)
            for on in obs_names:
                acc[on].append(rep_stats(
                    shot_estimates(out, OBSERVABLES[on], lut),
                    exact_value(rho, OBSERVABLES[on]))["mse"])
        for on in obs_names:
            haar_mse[(pname, on)] = np.mean(acc[on])
    print("  exact unbiasedness of the canonical dual: OK "
          "(all POVMs x states x observables, deterministic)")

    def table(title, getter):
        print(f"\n  MSE, {title}:")
        print("    " + f"{'':10s}" + "".join(f"{n[:12]:>14s}" for n in SOLIDS))
        for on in obs_names:
            print(f"    {on:10s}" + "".join(
                f"{getter(pn, on):14.5f}" for pn in SOLIDS))

    table("TFIM ground state (h=1)", lambda pn, on: mse[(pn, 'TFIM', on)]["mse"])
    table("Haar average", lambda pn, on: haar_mse[(pn, on)])

    print("\n  Haar average, ratio to octahedron / CV across POVMs:")
    for on in obs_names:
        vals = np.array([haar_mse[(pn, on)] for pn in SOLIDS])
        ratios = vals / haar_mse[("octahedron", on)]
        cv = vals.std(ddof=1) / vals.mean()
        print(f"    {on:10s}" + "".join(f"{r:14.3f}" for r in ratios)
              + f"   CV = {cv:5.1%}")
    return mse, haar_mse


# ---------------------------------------------------------------------------
# Exact landscape: the theorem behind Experiment 1 (no sampling, no RNG)
# ---------------------------------------------------------------------------
K_BODY = {"Z0": 1, "X0": 1, "Z0Z1": 2, "X0X1": 2, "Z0Z1Z2": 3}


def exact_backbone(povms, states, haar, exp1, exp1_haar):
    """Exact single-shot variances at the canonical dual, and the proposition.

    With the canonical dual, a site both terms touch contributes the operator
    sum_k (3 n_a)(3 n_b) E_k = 3 delta_ab Id + (9/V sum_k n_a n_b n_c) sigma_c,
    a site one term touches contributes exactly sigma_a (frame condition), so
    the variance sees the vertices only through moments of order <= 3: order 1
    vanishes (centered), order 2 is universal (2-design), order 3 vanishes for
    antipodal vertex sets.  Hence octahedron, cube, icosahedron, dodecahedron
    have identical variance on every (state, observable); a single k-body
    Pauli string sits at exactly 3^k - <P>^2 for all five solids (the
    tetrahedron's n_a^2 = 1/3 kills its same-letter third moments); and the
    tetrahedron can deviate only where a term pair shares a site with two
    different letters -- here only the TFIM energy, and only on states with
    the right correlations (the Haar states and the product state; TFIM and
    GHZ have real amplitudes, which zero the relevant <..sigma_y..> terms).
    The Experiment-1 Monte Carlo is then asserted to sit within 5 SE of the
    exact values it merely illustrates.
    """
    print("\nExact landscape at the canonical dual (deterministic):")
    obs_names = list(OBSERVABLES)
    state_items = list(states.items()) \
        + [(f"haar{i}", rho) for i, rho in enumerate(haar)]
    var = {}
    for pname, povm in povms.items():
        lut = lut_canonical(povm["s"])
        for sname, rho in state_items:
            p = born_tensor(rho, povm["E"])
            for on in obs_names:
                truth = exact_value(rho, OBSERVABLES[on])
                var[(pname, sname, on)] = \
                    exact_second_moment(p, OBSERVABLES[on], lut) - truth ** 2

    worst_anti = max(
        max(var[(pn, sn, on)] for pn in ANTIPODAL)
        - min(var[(pn, sn, on)] for pn in ANTIPODAL)
        for sn, _ in state_items for on in obs_names)
    assert worst_anti < 1e-10
    print(f"  antipodal solids identical on every (state, observable): "
          f"max spread = {worst_anti:.1e}: OK")

    worst_3k = max(
        abs(var[(pn, sn, on)] - (3.0 ** k - exact_value(rho, OBSERVABLES[on]) ** 2))
        for sn, rho in state_items for on, k in K_BODY.items() for pn in SOLIDS)
    assert worst_3k < 1e-10
    print(f"  single k-body strings: Var = 3^k - <P>^2 exactly, all five "
          f"solids: max |diff| = {worst_3k:.1e}: OK")

    dev = {sn: var[("tetrahedron", sn, "E_TFIM")]
           - var[("octahedron", sn, "E_TFIM")] for sn, _ in state_items}
    assert abs(dev["TFIM"]) < 1e-10 and abs(dev["GHZ"]) < 1e-10
    haar_devs = [dev[f"haar{i}"] for i in range(N_HAAR)]
    print("  tetrahedron deviations (third moment), E_TFIM only: "
          f"TFIM/GHZ = 0, product = {dev['product']:+.4f}, "
          f"Haar in [{min(haar_devs):+.4f}, {max(haar_devs):+.4f}]")
    print(f"    e.g. haar0: {var[('tetrahedron', 'haar0', 'E_TFIM')]:.4f} "
          f"(tetra) vs {var[('octahedron', 'haar0', 'E_TFIM')]:.4f} (others)")

    for sn in states:
        for on in obs_names:
            for pn in SOLIDS:
                st = exp1[(pn, sn, on)]
                assert abs(st["mse"] - var[(pn, sn, on)] / T_MAIN) \
                    < 5 * st["se_mse"] + 1e-12, (pn, sn, on)
    for on in obs_names:
        for pn in SOLIDS:
            ex = np.mean([var[(pn, f"haar{i}", on)]
                          for i in range(N_HAAR)]) / T_MAIN
            assert abs(exp1_haar[(pn, on)] - ex) / ex < 0.05, (pn, on)
    print("  Experiment-1 Monte Carlo within 5 SE of the exact values "
          "(TFIM/GHZ/product cells; Haar means within 5%): OK")

    out = {}
    for sn in list(states) + ["haar_mean"]:
        for on in obs_names:
            if sn == "haar_mean":
                uni = np.mean([var[("octahedron", f"haar{i}", on)]
                               for i in range(N_HAAR)])
                tet = np.mean([var[("tetrahedron", f"haar{i}", on)]
                               for i in range(N_HAAR)])
            else:
                uni, tet = var[("octahedron", sn, on)], \
                    var[("tetrahedron", sn, on)]
            out[f"exact/var/{sn}/{on}"] = np.array([uni])
            out[f"exact/var_tetra/{sn}/{on}"] = np.array([tet])
    return var, out


# ---------------------------------------------------------------------------
# Fourth-moment ladder: where the solid choice reappears, exactly
# ---------------------------------------------------------------------------
FOURTH_EXPECTED = {"tetrahedron": 9.0, "cube": 9.0, "octahedron": 27.0,
                   "icosahedron": 81.0 / 5.0, "dodecahedron": 81.0 / 5.0}


def fourth_moment_ladder(povms, states, haar):
    """Exact single-shot fourth moments of the canonical estimator.

    For a single Pauli sigma_a the single-shot estimate is f = 3 n_a, so
    E[f^4] = 81 (1/V) sum_k n_a^4 + 81 (1/V) sum_k n_a^4 (n_k . r).  The
    state-dependent term is a fifth-order odd moment and vanishes for every
    solid (antipodality; for the tetrahedron n_a^4 = 1/9 is constant), so
    E[f^4] is state-independent: 9 for tetrahedron and cube, 27 for the
    octahedron, 81/5 for icosahedron and dodecahedron -- whose 5-design
    strength pins them at the spherical value.  Design strength fixes the
    tails; it does not minimize them (Pauli-6/octahedron is heaviest, the
    cube lightest).
    """
    print("\nFourth-moment ladder: E[o^4] of the single-shot canonical "
          "estimator, single Paulis")
    state_items = list(states.items()) \
        + [(f"haar{i}", rho) for i, rho in enumerate(haar)]
    out = {}
    worst = 0.0
    for pname, povm in povms.items():
        s, V = povm["s"], povm["V"]
        vals = 81.0 * (s ** 4).mean(axis=0)
        assert vals.max() - vals.min() < 1e-12, pname
        assert abs(vals[0] - FOURTH_EXPECTED[pname]) < 1e-12, pname
        for sname, rho in state_items:      # state independence, exactly
            for r0 in site_bloch(rho):
                q = (1.0 + s @ r0) / V
                for a in range(3):
                    worst = max(worst, abs(q @ (3.0 * s[:, a]) ** 4 - vals[a]))
        out[f"exact4/{pname}"] = np.array([vals[0]])
        print(f"  {pname:12s} E[o^4] = {vals[0]:7.4f}   "
              f"(= 81/V sum_k n_a^4; every axis, site, state)")
    assert worst < 1e-12
    print(f"  state-independence across the full suite: "
          f"max |diff| = {worst:.1e}: OK")
    return out


# ---------------------------------------------------------------------------
# Experiment 2: dual-frame optimization (noiseless)
# ---------------------------------------------------------------------------
def training_bloch_mean():
    """Mean Bloch vector of the training reduced states (TFIM sweep + GHZ).

    The QP objective is linear in the training distribution, which for
    effects (1/V)(Id + s.sigma) depends on the reduced states only through
    their mean Bloch vector.
    """
    vecs = [site_bloch(density(tfim_ground_state(g))) for g in TRAIN_G]
    ghz = np.zeros(16, dtype=complex)
    ghz[0] = ghz[15] = 1 / np.sqrt(2)
    vecs.append(site_bloch(density(ghz)))           # = 0 (maximally mixed)
    return np.concatenate(vecs).mean(axis=0)


def letter_duals(s, rbar, s_eff=None):
    """Optimized dual per Pauli letter for training distribution rbar."""
    if s_eff is None:
        s_eff = s
    pbar = (1.0 + s @ rbar) / len(s)
    return {a: optimize_dual(s_eff, pbar, a)[1] for a in range(3)}


def oracle_lut(s, site_r, s_eff=None, noisy_site_r=None):
    """Per-(site, letter) duals fed the true reduced state of each site."""
    if s_eff is None:
        s_eff = s
    if noisy_site_r is None:
        noisy_site_r = site_r
    V = len(s)
    lut = np.empty((N_QUBITS, 3, V))
    for i in range(N_QUBITS):
        pbar = (1.0 + s @ noisy_site_r[i]) / V
        for a in range(3):
            _, beta = optimize_dual(s_eff, pbar, a)
            lut[i, a] = 2.0 * beta[:, a]
    return lut


def experiment_2(povms, states, rng):
    print("\nExperiment 2: dual-frame optimization, noiseless "
          f"(shots={T_EXP2}, reps={R_REPS})")
    rbar = training_bloch_mean()
    print(f"  training-set mean Bloch vector: "
          f"({rbar[0]:.4f}, {rbar[1]:.4f}, {rbar[2]:.4f})")
    obs_names = ["Z0", "X0", "Z0Z1", "E_TFIM"]
    eval_states = {"TFIM": states["TFIM"], "product": states["product"]}

    # Haar-optimality: uniform training distribution recovers the canonical
    # dual (overcomplete solids), and the tetrahedron has no freedom at all.
    for pname, povm in povms.items():
        s, V = povm["s"], povm["V"]
        for a in range(3):
            _, beta = optimize_dual(s, np.full(V, 1 / V), a)
            assert np.abs(beta[:, a] - 1.5 * s[:, a]).max() < 1e-8, (pname, a)
    print("  Haar-optimality: uniform-distribution QP returns the canonical "
          "dual (all solids): OK")

    results = {}
    for pname, povm in povms.items():
        s = povm["s"]
        luts = {"canonical": lut_canonical(s),
                "observable": lut_from_letter_duals(letter_duals(s, rbar))}
        if pname == "tetrahedron":     # V = d^2: the dual family is a point
            assert np.abs(luts["observable"] - luts["canonical"]).max() < 1e-8
        for braw in letter_duals(s, rbar).values():
            assert dual_frame_residual(0.5 * np.ones(len(s)), braw, s) < 1e-9
        for sname, rho in eval_states.items():
            luts["oracle"] = oracle_lut(s, site_bloch(rho))
            p = born_tensor(rho, povm["E"])
            out = sample_outcomes(p, R_REPS * T_EXP2, rng)
            for on in obs_names:
                truth = exact_value(rho, OBSERVABLES[on])
                for lname, lut in luts.items():
                    em = exact_estimator_mean(p, OBSERVABLES[on], lut)
                    assert abs(em - truth) < 1e-9, (pname, sname, on, lname)
                    results[(pname, sname, on, lname)] = rep_stats(
                        shot_estimates(out, OBSERVABLES[on], lut), truth)
    print("  exact unbiasedness of optimized duals: OK (deterministic)")

    for on in ["Z0", "Z0Z1"]:   # tetrahedron control: ratios exactly 1
        r = results[("tetrahedron", "TFIM", on, "oracle")]["mse"] \
            / results[("tetrahedron", "TFIM", on, "canonical")]["mse"]
        assert abs(r - 1.0) < 1e-12
    print("  tetrahedron control: MSE ratio = 1 exactly: OK")

    for sname in eval_states:
        label = "TFIM h=1 (in-distribution)" if sname == "TFIM" \
            else "product state (out-of-distribution)"
        print(f"\n  MSE ratio to canonical, {label}:")
        print("    " + f"{'POVM':14s}{'level':12s}"
              + "".join(f"{on:>12s}" for on in obs_names))
        for pname in SOLIDS:
            for lname in ["observable", "oracle"]:
                row = []
                for on in obs_names:
                    a = results[(pname, sname, on, lname)]
                    c = results[(pname, sname, on, "canonical")]
                    ratio = a["mse"] / c["mse"]
                    se = ratio * np.hypot(a["se_mse"] / a["mse"],
                                          c["se_mse"] / c["mse"])
                    row.append(f"{ratio:7.2f} ({se:.2f})")
                print(f"    {pname:14s}{lname:12s}" + "".join(
                    f"{r:>12s}" for r in row))
    print("  (parenthesised: delta-method standard error; shots are shared "
          "across levels, so these are conservative)")
    return results


# ---------------------------------------------------------------------------
# Exact Experiment-2 ratios: the duals are deterministic, so the table is too
# ---------------------------------------------------------------------------
def exact_exp2_ratios(povms, states, exp2):
    """Exact MSE ratios for every Experiment-2 cell (no sampling, no RNG).

    The optimized duals are fixed functions of the training data, so each
    cell's single-shot variance -- hence each ratio to canonical -- is an
    exact number via exact_second_moment.  The Monte Carlo of Experiment 2
    is asserted to agree within 5 delta-method SE (conservative: all levels
    share the same shots).  Two cells worth naming: the octahedron's oracle
    X-dual on the product state has *exactly* zero variance (ratio 0), and
    the octahedron's per-letter oracle on the TFIM energy is exactly *worse*
    than canonical (ratio > 1) -- per-letter optimization minimizes each
    letter's own second moment, not the cross-term covariances a composite
    observable also carries.
    """
    print("\nExact Experiment-2 ratios (deterministic duals):")
    rbar = training_bloch_mean()
    obs_names = ["Z0", "X0", "Z0Z1", "E_TFIM"]
    eval_states = {"TFIM": states["TFIM"], "product": states["product"]}
    out = {}
    worst_z = 0.0
    for pname, povm in povms.items():
        s = povm["s"]
        luts = {"canonical": lut_canonical(s),
                "observable": lut_from_letter_duals(letter_duals(s, rbar))}
        for sname, rho in eval_states.items():
            luts["oracle"] = oracle_lut(s, site_bloch(rho))
            p = born_tensor(rho, povm["E"])
            for on in obs_names:
                truth = exact_value(rho, OBSERVABLES[on])
                v = {ln: exact_second_moment(p, OBSERVABLES[on], lut)
                     - truth ** 2 for ln, lut in luts.items()}
                for ln in ("observable", "oracle"):
                    exact_r = v[ln] / v["canonical"]
                    out[f"exact_ratio/{pname}/{sname}/{on}/{ln}"] = \
                        np.array([exact_r])
                    if pname == "tetrahedron":     # unique dual: ratio is 1
                        assert abs(exact_r - 1.0) < 1e-7, (sname, on, ln)
                    a = exp2[(pname, sname, on, ln)]
                    c = exp2[(pname, sname, on, "canonical")]
                    if exact_r < 1e-9:             # exact zero-variance cell
                        assert a["mse"] < 1e-9 * c["mse"], (pname, sname, on)
                        continue
                    mc_r = a["mse"] / c["mse"]
                    se = mc_r * np.hypot(a["se_mse"] / a["mse"],
                                         c["se_mse"] / c["mse"])
                    worst_z = max(worst_z, abs(mc_r - exact_r) / se)
    assert worst_z < 5.0
    print(f"  MC ratios within 5 SE of exact on all cells "
          f"(worst |z| = {worst_z:.2f}): OK")
    print("  tetrahedron control: exact ratio = 1 (unique dual): OK")
    assert out["exact_ratio/octahedron/product/X0/oracle"][0] < 1e-9
    korh = out["exact_ratio/octahedron/TFIM/E_TFIM/oracle"][0]
    assert korh > 1.0
    print(f"  octahedron X0 / product / oracle: exactly zero variance: OK")
    print(f"  octahedron E_TFIM / TFIM / oracle: exact ratio = {korh:.3f} > 1 "
          "(per-letter optimization can lose on composite observables)")
    return out


# ---------------------------------------------------------------------------
# Experiment 3: robust calibration under noise
# ---------------------------------------------------------------------------
def calibrate(s, T, t, n_reps, n_shots, rng):
    """|0> calibration: per-rep mean sampled vertex (3-vector); eta = z part."""
    q = single_qubit_probs(s, T, t, np.array([0.0, 0.0, 1.0]))
    cum = np.cumsum(q)
    cum[-1] = 1.0
    idx = np.searchsorted(cum, rng.random((n_reps, n_shots * N_QUBITS)),
                          side="right")
    return s[idx].mean(axis=1)        # (n_reps, 3)


def experiment_3(povms, rho, rng):
    print("\nExperiment 3: robust calibration, icosahedron, TFIM h=1 "
          f"(shots={T_MAIN}, reps={R_REPS}, R_C={RC_CAL}, depolarizing sweep)")
    povm = povms["icosahedron"]
    s, V = povm["s"], povm["V"]
    obs_names = ["E_TFIM", "Z0Z1", "X0"]
    truths = {on: exact_value(rho, OBSERVABLES[on]) for on in obs_names}
    rbar = training_bloch_mean()
    site_r = site_bloch(rho)
    levels = ["noiseless", "robust", "optimized", "oracle"]
    results = {}

    # the calibration target is an exact identity, not a statistical fact:
    # E[eta_hat] = z-component of E[sampled vertex] = (1-p)/3, one matvec
    zhat = np.array([0.0, 0.0, 1.0])
    for p_rate in DEPOL_RATES:
        v = single_qubit_probs(s, *chan_depolarizing(p_rate), zhat) @ s
        assert abs(v[2] - (1 - p_rate) / 3) < 1e-12, p_rate
        assert np.hypot(v[0], v[1]) < 1e-12, p_rate
    print("  exact identity E[eta_hat] = (1-p)/3 (and E[v_perp] = 0), "
          "every rate: OK")

    for p_rate in DEPOL_RATES:
        T3, t3 = chan_depolarizing(p_rate)
        pt = born_tensor(rho, noisy_effects(s, T3, t3))
        out = sample_outcomes(pt, R_REPS * T_MAIN, rng)
        eta_hat = calibrate(s, T3, t3, R_REPS, RC_CAL, rng)[:, 2]

        # deterministic checks: eta target and the (1-p)^k bias factor of
        # the noiseless dual on noisy outcomes
        eta_target = (1 - p_rate) / 3
        lut0 = lut_canonical(s)
        for on, k_body in [("Z0", 1), ("Z0Z1", 2), ("E_TFIM", None)]:
            em = exact_estimator_mean(pt, OBSERVABLES[on], lut0)
            if k_body is not None:
                truth = exact_value(rho, OBSERVABLES[on])
                assert abs(em - (1 - p_rate) ** k_body * truth) < 1e-9

        est = {(on, ln): np.empty(R_REPS * T_MAIN)
               for on in obs_names for ln in levels}
        for r in range(R_REPS):
            sl = slice(r * T_MAIN, (r + 1) * T_MAIN)
            eta = eta_hat[r]
            s_eff = 3 * eta * s                # depolarizing-model effects
            luts = {
                "noiseless": lut0,
                "robust": lut_robust_canonical(s, eta),
                "optimized": lut_from_letter_duals(
                    letter_duals(s, 3 * eta * rbar, s_eff=s_eff)),
                "oracle": oracle_lut(s, site_r, s_eff=s_eff,
                                     noisy_site_r=(1 - p_rate) * site_r),
            }
            for on in obs_names:
                for ln in levels:
                    est[(on, ln)][sl] = shot_estimates(
                        out[sl], OBSERVABLES[on], luts[ln])
        for on in obs_names:
            for ln in levels:
                results[(p_rate, on, ln)] = rep_stats(est[(on, ln)],
                                                      truths[on])
        results[(p_rate, "eta")] = (eta_hat.mean(), eta_target)

    # calibration convergence at large R_C
    for p_rate in DEPOL_RATES:
        T3, t3 = chan_depolarizing(p_rate)
        eta = calibrate(s, T3, t3, 1, 5000, rng)[0, 2]
        se = 1 / np.sqrt(3 * 5000 * N_QUBITS)   # Var[s_z] <= 1/3
        assert abs(eta - (1 - p_rate) / 3) < 5 * se, p_rate
    print("  calibration convergence (R_C=5000): "
          "|eta_hat - (1-p)/3| < 5 SE for all rates: OK")
    print("  noiseless-dual bias factor (1-p)^k: exact (deterministic): OK")

    for p_rate in DEPOL_RATES:
        m, tgt = results[(p_rate, "eta")]
        print(f"    p={p_rate:<5} eta_hat = {m:.4f}  target = {tgt:.4f}")

    print(f"\n  bias^2 / variance / MSE at p=0.1 (icosahedron):")
    for on in ["E_TFIM", "Z0Z1"]:
        print(f"    {on}  (truth {truths[on]:+.4f})")
        for ln in levels:
            st = results[(0.1, on, ln)]
            print(f"      {ln:12s} bias2 {st['bias2']:9.5f}   "
                  f"var {st['var']:9.5f}   mse {st['mse']:9.5f}")

    print("\n  MSE for E_TFIM across the sweep:")
    print("    " + f"{'level':14s}"
          + "".join(f"{f'p={p}':>10s}" for p in DEPOL_RATES))
    for ln in levels:
        print(f"    {ln:14s}" + "".join(
            f"{results[(p, 'E_TFIM', ln)]['mse']:10.4f}"
            for p in DEPOL_RATES))

    # at p = 0 every level must be unbiased (bias^2 within MC error of 0)
    for on in obs_names:
        for ln in levels:
            st = results[(0.0, on, ln)]
            assert st["bias2"] < 25 * st["var"] / R_REPS, (on, ln)
    print("  p=0: all four levels unbiased within MC error: OK")
    return results


# ---------------------------------------------------------------------------
# Calibration blindness + the free anisotropy diagnostic
# ---------------------------------------------------------------------------
def blindness(povms, rho, rng):
    print("\nCalibration blindness (icosahedron, robust-canonical estimator, "
          f"TFIM h=1, shots={T_MAIN}, reps={R_REPS})")
    povm = povms["icosahedron"]
    s = povm["s"]
    site_r = site_bloch(rho)
    channels = {"dephasing": chan_dephasing, "amp-damping": chan_amp_damping}
    zhat = np.array([0.0, 0.0, 1.0])
    results = {}

    for cname, chan in channels.items():
        for p_rate in BLIND_RATES:
            T3, t3 = chan(p_rate)
            # the channel fixes |0>, so calibration sees "noiseless": exact.
            # As in experiment_3 the target is an identity, not a statistical
            # fact -- E[eta_hat] is the z part of E[sampled vertex], pinned at
            # 1/3 at every rate -- so it is computed, one matvec, not narrated
            assert abs((T3 @ zhat + t3)[2] - 1.0) < 1e-12
            v_cal = single_qubit_probs(s, T3, t3, zhat) @ s
            assert abs(v_cal[2] - 1 / 3) < 1e-12, (cname, p_rate)
            assert np.hypot(v_cal[0], v_cal[1]) < 1e-12, (cname, p_rate)
            eta_hat = calibrate(s, T3, t3, R_REPS, RC_CAL, rng)[:, 2]
            # and the sampled mean meets it: R_REPS x RC_CAL x N_QUBITS draws
            # of s_z with Var[s_z] <= 1/3, so 5 SE = 1.0e-3, against a worst
            # measured deviation of 2.4e-4 over the eight rows
            se_eta = 1 / np.sqrt(3 * R_REPS * RC_CAL * N_QUBITS)
            assert abs(eta_hat.mean() - 1 / 3) < 5 * se_eta, (cname, p_rate)
            pt = born_tensor(rho, noisy_effects(s, T3, t3))
            out = sample_outcomes(pt, R_REPS * T_MAIN, rng)
            row = {"eta": eta_hat.mean()}
            for on, i, a in [("X0", 0, "X"), ("Z0", 0, "Z")]:
                obs = OBSERVABLES.get(on) or [(1.0, [(i, a)])]
                est = np.empty(R_REPS * T_MAIN)
                for r in range(R_REPS):
                    sl = slice(r * T_MAIN, (r + 1) * T_MAIN)
                    est[sl] = shot_estimates(
                        out[sl], obs, lut_robust_canonical(s, eta_hat[r]))
                bias = est.reshape(R_REPS, -1).mean(axis=1).mean() \
                    - site_r[i, AXIS[a]]
                # eta_hat -> 1/3, so the estimator silently reports the
                # *noisy* expectation: predicted bias = (T r + t - r)_a
                pred = (T3 @ site_r[i] + t3 - site_r[i])[AXIS[a]]
                assert abs(bias - pred) < 0.01, (cname, p_rate, on)
                row[on] = (bias, pred)
            results[(cname, p_rate)] = row

    print("    channel       p     eta_hat   bias X0 (pred)      bias Z0 (pred)")
    for (cname, p_rate), row in results.items():
        bx, px = row["X0"]
        bz, pz = row["Z0"]
        print(f"    {cname:12s}{p_rate:5.2f}   {row['eta']:.4f}   "
              f"{bx:+.4f} ({px:+.4f})   {bz:+.4f} ({pz:+.4f})")
    print("  eta_hat pinned at 1/3 (exact, and the sampled mean within 5 SE) "
          "while biases\n  match the Heisenberg-picture prediction: OK")

    print("\n  Anisotropy diagnostic: the calibration shots already contain "
          "the full 3-vector\n  v_hat -> (1/3) T z_hat; "
          "channels tilting the z-axis are flagged for free:")
    tests = {
        "depolarizing p=0.1": chan_depolarizing(0.1),
        "dephasing p=0.1": chan_dephasing(0.1),
        "amp-damping p=0.1": chan_amp_damping(0.1),
        "tilted-depol p=0.05, theta=0.2": chan_tilted_depolarizing(0.05, 0.2),
    }
    diag = {}
    for label, (T3, t3) in tests.items():
        v = calibrate(s, T3, t3, 1, 5000, rng)[0]
        v_exact = (T3 @ zhat + t3) / 3
        perp = np.hypot(v[0], v[1])
        diag[label] = (v, v_exact, perp)
        print(f"    {label:32s} eta_hat = {v[2]:7.4f}   "
              f"|v_perp| = {perp:.4f}   (exact {np.hypot(*v_exact[:2]):.4f})")
    assert diag["tilted-depol p=0.05, theta=0.2"][2] > 0.05
    assert all(diag[k][2] < 0.02 for k in list(tests)[:3])
    print("  z-axis-fixing channels: |v_perp| ~ 0; tilted channel flagged: OK")

    short = {"depolarizing p=0.1": "depol", "dephasing p=0.1": "dephasing",
             "amp-damping p=0.1": "ampdamp",
             "tilted-depol p=0.05, theta=0.2": "tilted"}
    aniso = {short[k]: np.array([diag[k][0][2], diag[k][2],
                                 np.hypot(*diag[k][1][:2])]) for k in tests}
    return results, aniso


# ---------------------------------------------------------------------------
# Twirl check: randomized implementation makes any noise depolarizing
# ---------------------------------------------------------------------------
def align_to_z(v):
    """Rotation A with A v = zhat (Rodrigues; v a unit vector)."""
    zhat = np.array([0.0, 0.0, 1.0])
    c, ax = v @ zhat, np.cross(v, zhat)
    s = np.linalg.norm(ax)
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    ax = ax / s
    K = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]],
                  [-ax[1], ax[0], 0.0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def twirl_check(povms):
    """Randomized-projective: the exact estimator channel is depolarizing.

    The protocol draws g uniformly from the covariance group, applies U_g,
    then one fixed alignment A taking a vertex axis v to the readout axis
    zhat (the octahedron already has zhat as a vertex; the icosahedron's
    zhat is an *edge* axis, so it genuinely needs one), lets the
    measurement-side noise r -> T r + t act, and measures the z basis;
    outcome (g, b) is post-processed into the snapshot vertex b R_g^T v.
    Summing probability x snapshot over (g, b) gives the estimator's
    channel exactly, and the twirled object is the *composition*
    zhat zhat^T T of readout projector and noise -- readout and conjugation
    share the same g -- not the noise alone.  Schur's lemma collapses it
    all the same: the two rotation groups act irreducibly on the Bloch
    space, so averaging any matrix M over conjugation gives (tr M / 3) Id
    and any vector averages to zero, but the trace picks out
    tr(zhat zhat^T T) = T_zz.  The channel is depolarizing with shrinkage
    eta = T_zz / 3, the noise's diagonal entry along the readout axis
    (Chen et al.'s single-qubit fidelity f), NOT (tr T / 3) / 3 -- that
    scalar twirls the noise alone and is the OTHER protocol's, twirled-
    native, derived in two_protocol_twirl below.  Dephasing along the
    readout axis is the eta = 1/3 case: twirled into no noise at all.
    The |0> calibration reads exactly this eta, so the robust inversion
    is exactly unbiased -- asserted on a random state.

    Checked to 1e-12 with the SO(3) groups from data/group_{O,I}.npz: the
    raw Schur lemma on a random matrix and vector (a probability-one
    witness of irreducibility, the twirl defect being linear in M), the
    orbit {+- R_g^T v} covering the solid's vertex set uniformly
    (randomized-projective samples the POVM it claims to), and the
    channel + calibration for dephasing, amplitude damping, and a random
    affine map.
    """
    print("\nTwirl check: randomized-projective's exact estimator "
          "channel\n(group-averaged over the atlas rotations, alignment "
          "included) is depolarizing:")
    rng = np.random.default_rng(11)
    zhat = np.array([0.0, 0.0, 1.0])
    tests = {
        "dephasing p=0.13": chan_dephasing(0.13),
        "amp-damping g=0.21": chan_amp_damping(0.21),
        "random affine": (rng.standard_normal((3, 3)) * 0.3,
                          rng.standard_normal(3) * 0.2),
    }
    for gname, solid in (("O", "octahedron"), ("I", "icosahedron")):
        R = np.load(DATA / f"group_{gname}.npz")["rotations"]

        # Schur lemma, raw: twirl(M) = (tr M / 3) Id, avg R^T u = 0
        M = rng.standard_normal((3, 3))
        u = rng.standard_normal(3)
        M_avg = np.einsum("gji,jk,gkl->il", R, M, R) / len(R)
        assert np.abs(M_avg - (np.trace(M) / 3) * np.eye(3)).max() < 1e-12
        assert np.abs(np.einsum("gji,j->i", R, u) / len(R)).max() < 1e-12

        # alignment: identify the readout axis with a vertex axis
        s = povms[solid]["s"]
        v = s[np.argmax(s[:, 2])]        # convention: first vertex nearest +z
        if gname == "O":
            assert np.abs(v - zhat).max() < 1e-12       # already aligned
        else:                            # unaligned z is not a vertex axis
            assert np.linalg.norm(s - zhat, axis=1).min() > 0.1
        A = align_to_z(v)

        # the orbit of the readout axis is the vertex set, uniformly
        orbit = np.einsum("gji,j->gi", R, v)            # R_g^T v = R_g^T A^T z
        orbit = np.concatenate([orbit, -orbit])
        counts = (np.linalg.norm(orbit[:, None, :] - s[None], axis=2)
                  < 1e-9).sum(axis=0)
        assert (counts == 2 * len(R) // len(s)).all(), gname
        print(f"  group {gname} ({len(R)} rotations): Schur lemma on random "
              f"(M, u): OK; orbit of the\n    readout axis = the {len(s)} "
              f"{solid} vertices, {counts[0]}x each: OK")

        for label, (T3, t3) in tests.items():
            # channel: E_g outer(R^T v, R^T w) with w = (T A)^T z; shift
            # carries the mean sampled vertex E_g[R^T v] = 0 (centered)
            w = A.T @ T3.T @ zhat
            ch = np.einsum("gji,j,gkl,k->il", R, v, R, w) / len(R)
            m = (zhat @ t3) * np.einsum("gji,j->i", R, v) / len(R)
            assert np.abs(ch - (T3[2, 2] / 3) * np.eye(3)).max() < 1e-12, \
                (gname, label)
            assert np.abs(m).max() < 1e-12, (gname, label)
            eta = (ch @ zhat + m)[2]     # what the |0> calibration reads
            r = rng.standard_normal(3) * 0.5
            assert np.abs((ch @ r + m) / eta - r).max() < 1e-12
            print(f"    {label:20s} eta = T_zz/3 = {eta:+.6f}   "
                  f"((tr T/3)/3 would read {np.trace(T3) / 9:+.6f})")
    print("  channel scalar + shift zero to 1e-12; calibration reads eta; "
          "inversion unbiased: OK")


# ---------------------------------------------------------------------------
# Two protocols: randomized-projective vs twirled-native, as exact channels
# ---------------------------------------------------------------------------
COVARIANCE_GROUP = {"tetrahedron": "T", "octahedron": "O", "cube": "O",
                    "icosahedron": "I", "dodecahedron": "I"}

# the generic probe noise of randomized_implementations.py, verbatim, so the
# two modules' exact scalars cross-check value for value: T_zz = 0.62,
# tr(T)/3 = 0.72 (genericity re-asserted in two_protocol_twirl)
T_PROBE = np.array([[0.83, 0.06, -0.11],
                    [-0.04, 0.71, 0.09],
                    [0.12, -0.07, 0.62]])
t_PROBE = np.array([0.05, -0.03, 0.17])


def channel_R1(s, R, T, t):
    """Exact estimator channel of the randomized-projective protocol.

    Draw g uniformly from the rotations R, apply U_g, apply the fixed
    alignment A (vertex nearest +z -> zhat), let the measurement-side noise
    r -> T r + t act, read out Z; outcome (g, b) is post-processed into the
    snapshot vertex b R_g^T v.  Returns (M, m) of the mean sampled vertex,
    E[vertex] = M r + m -- this module's convention (ideal channel Id/3;
    randomized_implementations.py includes the canonical dual's factor 3,
    so its channels are 3x these).  Same expressions as twirl_check().
    """
    zhat = np.array([0.0, 0.0, 1.0])
    v = s[np.argmax(s[:, 2])]
    A = align_to_z(v)
    w = A.T @ T.T @ zhat
    M = np.einsum("gji,j,gkl,k->il", R, v, R, w) / len(R)
    m = (zhat @ t) * np.einsum("gji,j->i", R, v) / len(R)
    return M, m


def channel_R2(s, R, T, t):
    """Exact estimator channel of the twirled-native protocol.

    Draw g, apply U_g, measure the NATIVE (Naimark) POVM -- all V effects
    -- and relabel by g in post-processing: snapshot vertex R_g^T n_k with
    probability (1/V)(1 + n_k . (T R_g r + t)).  Same mean-sampled-vertex
    convention as channel_R1; no alignment, no decomposability -- every
    solid admitted.  The vertex sums are kept explicit (rather than
    substituting the 2-design identity), so the computed channel is
    per-solid evidence, not a closed form restated.
    """
    C = s.T @ s / len(s)                     # = Id/3: the 2-design identity
    M = np.einsum("gji,jk,kl,glm->im", R, C, T, R) / len(R)
    m = np.einsum("gji,j->i", R, s.mean(axis=0) + C @ t) / len(R)
    return M, m


def orbit_hits(s, R):
    """How often the signed orbit {+-R_g^T v} of the alignment vertex hits
    each vertex.  Uniform counts = the draw's coin samples the POVM's
    vertex set uniformly (the projective route realizes the POVM); a zero
    means that vertex is never measured at all."""
    v = s[np.argmax(s[:, 2])]
    orbit = np.einsum("gji,j->gi", R, v)
    orbit = np.concatenate([orbit, -orbit])
    d = np.linalg.norm(orbit[:, None, :] - s[None], axis=2)
    assert (d.min(axis=1) < 1e-9).all()      # every draw lands on a vertex
    return (d < 1e-9).sum(axis=0)


def two_protocol_twirl(povms):
    """The two randomized implementations as exact estimator channels.

    Section 4.2.4 distinguishes two protocols that both carried the name
    "randomized implementation"; this check computes both on this module's
    estimator side, giving the appendix its own citable numbers (the
    symbolic gate for the underlying claims is
    randomized_implementations.py; the probe noise here is that module's
    verbatim, so the scalars cross-check value for value).

    Checked to 1e-12, per solid and per draw (covariance group and the
    uniform T draw): randomized-projective is exactly depolarizing at
    eta = T_zz/3 for the four antipodal solids -- the readout is drawn
    with the same g that conjugates the noise, so only the readout-axis
    entry survives -- and twirled-native at eta = (tr T/3)/3 for ALL FIVE,
    the tetrahedral SIC included; offsets die; the calibrated inversion is
    exactly unbiased; a local-rng random affine map re-witnesses each
    equality with probability one.  The T draw is the universal minimal
    twirl, and its orbit realizes octahedron/cube/icosahedron vertex-
    uniformly (x4/x3/x2) while covering only 6 of the dodecahedron's 10
    axes -- eight vertices never sampled, so the projective route's single
    draw does not realize the dodecahedral POVM there (the channel itself
    stays exactly depolarizing: it is a perfectly unbiased *different*
    measurement).  Payoff for the blindness study: dephasing and amplitude
    damping -- the fixers of |0> that pin the native calibration at 1/3 --
    twirl to eta readings visibly != 1/3 under twirled-native, and
    randomized-projective twirls readout-axis dephasing (T_zz = 1) to no
    noise at all.
    """
    print("\nTwo-protocol twirl: randomized-projective vs twirled-native "
          "estimator channels,\nexact per solid and draw (same probe noise "
          "as randomized_implementations.py):")
    # genericity: the two candidate factors differ, offset nonzero, T aniso
    assert abs(T_PROBE[2, 2] - np.trace(T_PROBE) / 3) > 0.05
    assert np.linalg.norm(t_PROBE) > 0.05
    assert np.linalg.norm(T_PROBE - np.trace(T_PROBE) / 3 * np.eye(3)) > 0.05

    rng = np.random.default_rng(7)           # local: irreducibility witness
    T_rand = rng.standard_normal((3, 3)) * 0.3
    t_rand = rng.standard_normal(3) * 0.2
    groups = {g: np.load(DATA / f"group_{g}.npz")["rotations"]
              for g in ("T", "O", "I")}
    r_test = np.array([0.31, -0.42, 0.53])   # fixed state inside the ball
    eta1 = T_PROBE[2, 2] / 3                 # R1 reading: T_zz / 3
    eta2 = np.trace(T_PROBE) / 9             # R2 reading: (tr T / 3) / 3

    # the cross-check the docstrings narrate, computed rather than asserted
    # in prose: the probe IS that module's, its two scalars are these two
    # after the factor 3, and its channels -- an independent expression that
    # carries the canonical dual's factor 3 -- are exactly 3x these.  Worst
    # deviation 1.1e-15 (icosahedron, both protocols).
    import randomized_implementations as ri  # local: import order untouched
    assert np.array_equal(T_PROBE, ri.T_NOISE), "probe T drifted apart"
    assert np.array_equal(t_PROBE, ri.t_NOISE), "probe t drifted apart"
    assert abs(3 * eta1 - ri.T_NOISE[2, 2]) < 1e-12           # R1: kappa
    assert abs(3 * eta2 - np.trace(ri.T_NOISE) / 3) < 1e-12   # R2: kappa
    s_x, R_x = povms["icosahedron"]["s"], groups["I"]
    for here, there in ((channel_R1, ri.channel_R1),
                        (channel_R2, ri.channel_R2)):
        M_h, m_h = here(s_x, R_x, T_PROBE, t_PROBE)
        M_t, m_t = there(s_x, R_x, ri.T_NOISE, ri.t_NOISE)
        assert np.abs(M_t - 3 * M_h).max() < 1e-12, here.__name__
        assert np.abs(m_t - 3 * m_h).max() < 1e-12, here.__name__
    print(f"  probe factors (3 eta): T_zz = {3 * eta1:.6f} "
          f"(randomized-projective) vs tr T/3 = {3 * eta2:.6f} "
          "(twirled-native)")

    out = {}
    rows = []
    for name in SOLIDS:
        s = povms[name]["s"]
        for draw, R in (("cov", groups[COVARIANCE_GROUP[name]]),
                        ("T", groups["T"])):
            M, m = channel_R2(s, R, T_PROBE, t_PROBE)
            aniso = np.abs(M - M[0, 0] * np.eye(3)).max()
            assert abs(M[0, 0] - eta2) < 1e-12 and aniso < 1e-12, (name, draw)
            assert np.abs(m).max() < 1e-12, (name, draw)
            assert np.abs((M @ r_test + m) / eta2 - r_test).max() < 1e-12
            Mr, mr = channel_R2(s, R, T_rand, t_rand)
            assert np.abs(Mr - (np.trace(T_rand) / 9)
                          * np.eye(3)).max() < 1e-12, (name, draw)
            assert np.abs(mr).max() < 1e-12, (name, draw)
            out[f"twirl2/{draw}/{name}"] = np.array(
                [M[0, 0], aniso, np.abs(m).max()])
        if name in ANTIPODAL:
            RG = groups[COVARIANCE_GROUP[name]]
            hits = orbit_hits(s, RG)         # the covariance draw realizes
            assert hits.min() == hits.max() == 2 * len(RG) // len(s), name
            M, m = channel_R1(s, RG, T_PROBE, t_PROBE)
            assert np.abs(M - eta1 * np.eye(3)).max() < 1e-12, name
            assert np.abs(m).max() < 1e-12, name
            assert np.abs((M @ r_test + m) / eta1 - r_test).max() < 1e-12
            Mr, mr = channel_R1(s, RG, T_rand, t_rand)
            assert np.abs(Mr - (T_rand[2, 2] / 3)
                          * np.eye(3)).max() < 1e-12, name
            assert np.abs(mr).max() < 1e-12, name
            out[f"twirl1/cov/{name}"] = np.array(
                [M[0, 0], np.abs(m).max(), hits[0]])
            # the minimal draw: T twirls every antipodal solid at the same
            # T_zz -- but its orbit realizes only octa/cube/icosa
            Mt, mt = channel_R1(s, groups["T"], T_PROBE, t_PROBE)
            assert np.abs(Mt - eta1 * np.eye(3)).max() < 1e-12, name
            assert np.abs(mt).max() < 1e-12, name
            th = orbit_hits(s, groups["T"])
            if name == "dodecahedron":       # 6/10 axes: realization fails
                assert sorted(set(th.tolist())) == [0, 2]
                assert int((th > 0).sum()) == 12
                torbit = "6/10 axes only -- NOT this POVM"
            else:
                assert th.min() == th.max() == 24 // len(s), name
                torbit = f"uniform x{th[0]}"
            out[f"twirl1/minT/{name}"] = np.array(
                [Mt[0, 0], th.min(), th.max(), int((th > 0).sum())])
            r1_cell = f"{M[0, 0]:.6f}"
        else:
            r1_cell, torbit = "-- (no antipodes)", "-- (R2 needs no orbit)"
        rows.append(f"  {name:14s}{r1_cell:>20s}{eta2:>12.6f}   {torbit}")
    print(f"  {'solid':14s}{'R1 eta = T_zz/3':>20s}{'R2 eta':>12s}   "
          "T-draw orbit")
    print("\n".join(rows))
    print("  [ok] R1 exactly depolarizing at T_zz/3 (antipodal four); R2 at "
          "(tr T/3)/3 for all\n  five (SIC included); offsets 0, inversion "
          "unbiased, both draws, to 1e-12")

    # the blindness study's channels, twirled: the blind spot closed exactly
    print("  blind-spot channels twirled (native |0> calibration reads 1/3 "
          "for every rate):")
    s_ico, s_tet = povms["icosahedron"]["s"], povms["tetrahedron"]["s"]
    zhat = np.array([0.0, 0.0, 1.0])
    for cname, chan in (("dephasing", chan_dephasing),
                        ("amp-damping", chan_amp_damping)):
        for p in BLIND_RATES:
            T3, t3 = chan(p)
            assert abs((T3 @ zhat + t3)[2] - 1.0) < 1e-12   # fixes |0>
            e2 = np.trace(T3) / 9
            for s_any in (s_ico, s_tet):     # solid-blind, SIC included
                M, m = channel_R2(s_any, groups["T"], T3, t3)
                assert np.abs(M - e2 * np.eye(3)).max() < 1e-12, (cname, p)
                assert np.abs(m).max() < 1e-12, (cname, p)
                assert np.abs((M @ r_test + m) / e2 - r_test).max() < 1e-12
            e1 = T3[2, 2] / 3
            M1, m1 = channel_R1(s_ico, groups["T"], T3, t3)
            assert np.abs(M1 - e1 * np.eye(3)).max() < 1e-12, (cname, p)
            assert np.abs(m1).max() < 1e-12, (cname, p)
            out[f"twirl2/blind/{cname}/{p}"] = np.array([M[0, 0]])
            out[f"twirl1/blind/{cname}/{p}"] = np.array([M1[0, 0]])
        readings = "  ".join(
            f"p={p}: {out[f'twirl2/blind/{cname}/{p}'][0]:.4f}"
            for p in BLIND_RATES)
        print(f"    {cname:12s} R2 eta {readings}")
    assert all(abs(out[f"twirl1/blind/dephasing/{p}"][0] - 1 / 3) < 1e-12
               for p in BLIND_RATES)         # T_zz = 1: twirled to no noise
    print("    (R1: dephasing has T_zz = 1 -- twirled to no noise at all; "
          "amp-damping reads (1-g)/3)")
    print("  [ok] each protocol's calibration sees the channels the native "
          "one cannot")
    return out


# ---------------------------------------------------------------------------
# Gate noise: the twirl's assumption, exercised with the atlas's own circuits
# ---------------------------------------------------------------------------
def bloch_rotation(U):
    """SO(3) rotation of conjugation by U: R_ij = tr(sigma_i U sigma_j U†)/2."""
    P = [PAULI["X"], PAULI["Y"], PAULI["Z"]]
    return np.array([[0.5 * np.trace(P[i] @ U @ P[j] @ U.conj().T).real
                      for j in range(3)] for i in range(3)])


def parse_sequence(seq):
    """Atlas sequence string -> [(gate, dagger), ...], operator order.

    Sequences are space-separated with 'I' for the identity; 'F X' means
    F.X, i.e. X is applied first (main.py's convention).  A trailing dagger
    marks the SU(2) inverse.
    """
    if seq == "I":
        return []
    return [(t.rstrip("†"), t.endswith("†")) for t in seq.split()]


def gate_noise_twirl(states, povms):
    """Per-gate noise breaks the exact twirl; the residual, quantified.

    twirl_check() assumes one noise channel independent of the drawn rotation
    g.  On hardware each g is a circuit from the atlas, so noise arrives per
    elementary gate and is *correlated with g* -- Schur's lemma no longer
    applies.  Per group element we compose the exact affine Bloch channel
    (T_g, t_g) of its synthesized circuit with amplitude damping gamma after
    every gate, then average the *estimator's* channel exactly -- readout
    probability times snapshot vertex, summed over (g, b), as in
    twirl_check():

        r -> M r + m,   M = E_g outer(R_g^T v, T_g^T v),
                        m = E_g (R_g^T v) (v . t_g),

    with v the vertex axis the fixed alignment identifies with the readout
    (v = z for the octahedron; the first vertex nearest +z for the
    icosahedron; the alignment itself is taken ideal -- its noise would be
    g-independent and twirl cleanly).  Reported: the anisotropy of M, the
    displacement |m|, the calibration reading eta = z.(M z + m) (ideal 1/3),
    and the exact residual bias (M r0 + m)/eta - r0 surviving the scalar
    |0> calibration.  Because the noise is g-correlated, the residual also
    depends on *which* vertex the alignment picks: the min/max of the Z0
    residual over all vertex choices is stored alongside the convention's
    value.  Series: min-depth (BFS) circuits for 2O and 2I at uniform
    gamma, plus 2I with Phi three times noisier -- magic cost read as noise
    cost -- compiled min-depth (BFS, up to two Phi per circuit) vs
    min-magic (Dijkstra, at most one); the "2T" series is the minimal
    projective draw -- the icosahedron realized and twirled by the uniform
    T draw (12 rotations, every word Clifford at depth <= 2, orbit = all
    12 vertices x2, same alignment vertex as the 2I series, so the
    residual drop is attributable to the draw alone); and the "R2" series
    prices the twirled-native protocol on the same T words.

    Noise accounting (a row is uninterpretable without it): every
    projective row (2O/2I/2T) damps every elementary gate of the DRAWN
    word only -- the alignment stays ideal as above.  The R2 rows damp
    every gate of the drawn T word and MODEL the fixed dilation as one
    g-independent pre-measurement amplitude-damping channel (GAMMA_DIL) on
    the data qubit: real dilation noise deforms the effects ancilla-side;
    the general twirl theorem covers any g-independent channel, and the
    pre-channel is the modeling choice.  It twirls out exactly -- at
    gamma = 0 the residual is zero against a visibly noisy dilation
    (eta = tr T_N/9, calibrated away) -- and the R2 estimator channel is
    provably solid-independent (the drawn words never see the solid), so
    the five per-solid rows are asserted equal to 1e-12: the SIC's noise
    bill equals the icosahedron's, and the whole g-correlated exposure
    sits in depth-<=2 Clifford words.  A bare variant (no dilation
    channel) isolates the modeling choice's effect.  Sanity limits:
    gamma = 0 recovers the ideal channel M = Id/3 with zero residual (for
    R2, M = (tr T_N/9) Id), and gate-independent noise of any strength
    still twirls to an exact scalar with zero residual, whatever the
    compilation -- both to machine precision.
    """
    print("\nGate-noise twirl study: per-gate amplitude damping on the "
          "atlas circuits\n(noise correlated with g -- the exact twirl "
          "assumption dropped):")
    gates = np.load(DATA / "gates.npz")
    su2 = {("Φ" if str(n) == "Phi" else str(n)): U
           for n, U in zip(gates["names"], gates["su2"])}
    rot = {n: bloch_rotation(U) for n, U in su2.items()}
    zhat = np.array([0.0, 0.0, 1.0])
    r0 = site_bloch(states["TFIM"])[0]     # the blindness study's test vector

    def circuit_channel(tokens, gamma, phi_mult):
        """Affine Bloch channel of the circuit, noise after every gate."""
        T, t = np.eye(3), np.zeros(3)
        for base, dag in reversed(tokens):      # rightmost gate acts first
            R = rot[base].T if dag else rot[base]
            T, t = R @ T, R @ t
            g = gamma * (phi_mult if base == "Φ" else 1.0)
            if g > 0.0:
                Tn, tn = chan_amp_damping(g)
                T, t = Tn @ T, Tn @ t + tn
        return T, t

    def load_series(gname, mode):
        """One circuit per SO(3) rotation: min depth (bfs) or min (magic,
        depth) (dij) within each +/-q pair; replay-checked against the
        stored unitaries."""
        d = np.load(DATA / f"group_{gname}.npz")
        Us, seqs = d["unitaries"], d[f"{mode}_sequences"]
        score = d["bfs_depths"] if mode == "bfs" else \
            list(zip(d["dij_magic_costs"], d["dij_depths"]))
        reps = {}
        for i in range(len(Us)):
            key = tuple(np.round(bloch_rotation(Us[i]), 9).ravel())
            if key not in reps or score[i] < score[reps[key]]:
                reps[key] = i
        idx = sorted(reps.values())
        assert len(idx) == len(Us) // 2, gname
        Rs, toks, phis = [], [], []
        for i in idx:
            tk = parse_sequence(seqs[i])
            U = I2.copy()
            for base, dag in tk:               # operator order replay
                U = U @ (su2[base].conj().T if dag else su2[base])
            assert np.abs(U - Us[i]).max() < 1e-10, (gname, mode, i)
            Rs.append(bloch_rotation(Us[i]))
            toks.append(tk)
            phis.append(sum(1 for b, _ in tk if b == "Φ"))
        return np.stack(Rs), toks, np.array(phis)

    def estimator_channel(Rs, chans, v):
        """(M, m) of the randomized estimator for alignment vertex v."""
        M = np.mean([np.outer(R.T @ v, T.T @ v)
                     for R, (T, _) in zip(Rs, chans)], axis=0)
        m = np.mean([(R.T @ v) * (v @ t)
                     for R, (_, t) in zip(Rs, chans)], axis=0)
        return M, m

    def residual(M, m):
        """Post-|0>-calibration bias on r0, and the calibration reading."""
        eta = (M @ zhat + m)[2]
        return (M @ r0 + m) / eta - r0, eta

    series = {"2O": ("2O", "bfs", 1.0), "2I": ("2I", "bfs", 1.0),
              "2I_phi3": ("2I", "bfs", 3.0),
              "2I_dij_phi3": ("2I", "dij", 3.0),
              "2T": ("2T", "bfs", 1.0)}
    verts = {"2O": povms["octahedron"]["s"], "2I": povms["icosahedron"]["s"],
             "2T": povms["icosahedron"]["s"]}
    stash = {}                                 # the T words, reused by R2
    out = {}
    for name, (gname, mode, phi_mult) in series.items():
        Rs, toks, phis = load_series(gname, mode)
        depths = np.array([len(tk) for tk in toks])
        if mode == "dij":                      # the magic-cost dichotomy
            assert phis.max() <= 1, name
        out[f"gatenoise/meta/{name}"] = np.array(
            [len(Rs), depths.mean(), depths.max(), phis.mean()])
        s = verts[gname]
        v = s[np.argmax(s[:, 2])]              # alignment convention
        if gname == "2O":
            assert np.abs(v - zhat).max() < 1e-12   # z already a vertex
        if name == "2T":                       # the minimal projective draw
            assert depths.max() <= 2 and int(phis.max()) == 0, name
            assert (orbit_hits(s, Rs) == 2).all()   # all 12 vertices, x2
            stash["T"] = (Rs, toks, depths, phis)

        # sanity: gate-independent noise after the whole (ideal) circuit
        # still twirls to an exact scalar with zero residual (Schur).  The
        # compilation cannot enter: the damping lands after the whole ideal
        # circuit, so `toks` is never read below and the three icosahedral
        # rows store bitwise-identical values.  The control row's "every
        # compilation" is therefore true by construction, not by this sweep.
        # Stored as well as asserted: Appendix F's table prints the zero as a
        # literal, so what keeps the page honest is shadow_report.py
        # re-asserting the stored value before it typesets -- a guard that
        # sits downstream of this file, and so survives THIS file's tolerance
        # being loosened.  The row claims ANY strength, so the check sweeps
        # strengths rather than pinning one and leaving Schur to cover the
        # difference.
        worst = np.zeros(2)
        for delta in INDEP_DELTAS:
            Tn, tn = chan_amp_damping(delta)
            M0, m0 = estimator_channel(Rs, [(Tn @ R, tn) for R in Rs], v)
            assert np.abs(M0 - (v @ Tn @ v / 3) * np.eye(3)).max() < 1e-12
            assert np.abs(m0).max() < 1e-12
            bias0 = residual(M0, m0)[0]
            assert np.abs(bias0).max() < 1e-12, (name, delta)
            worst = np.maximum(worst, np.abs(bias0[[0, 2]]))
        out[f"gatenoise/indep/{name}"] = worst    # worst |(X0, Z0)| over the sweep

        for gamma in GATE_GAMMAS:
            chans = [circuit_channel(tk, gamma, phi_mult) for tk in toks]
            M, m = estimator_channel(Rs, chans, v)
            aniso = np.abs(M - (np.trace(M) / 3) * np.eye(3)).max()
            disp = np.linalg.norm(m)
            bias, eta = residual(M, m)
            out[f"gatenoise/{name}/{gamma}"] = np.array(
                [aniso, disp, eta, bias[0], bias[2]])
            span = [residual(*estimator_channel(Rs, chans, vk))[0][2]
                    for vk in s]
            out[f"gatenoise/span/{name}/{gamma}"] = np.array(
                [min(span), max(span)])
        g0 = out[f"gatenoise/{name}/0.0"]
        assert np.abs(g0[[0, 1, 3, 4]]).max() < 1e-12
        assert abs(g0[2] - 1 / 3) < 1e-12
        print(f"  {name:12s} rotations {len(Rs):3d}, mean depth "
              f"{depths.mean():.2f}, mean Phi {phis.mean():.2f}: "
              "gamma=0 -> ideal, gate-independent -> Schur: OK")

    # ---- the twirled-native series: the same per-gate damping on the T
    # words; the fixed dilation MODELED as one g-independent pre-measurement
    # channel (amplitude damping, GAMMA_DIL) on the data qubit, which the
    # twirl removes exactly -- so what remains is the drawn words' bill,
    # and the estimator channel is provably solid-independent.
    Rs, toks, depths, phis = stash["T"]
    TN, tN = chan_amp_damping(GAMMA_DIL)
    eta_dil = np.trace(TN) / 9               # the gamma = 0 reading
    out["gatenoise/meta/R2"] = np.array(
        [len(Rs), depths.mean(), depths.max(), phis.mean()])
    for gamma in GATE_GAMMAS:
        chans = [circuit_channel(tk, gamma, 1.0) for tk in toks]
        M_cf = np.mean([R.T @ TN @ T                 # closed form: Id/3
                        for R, (T, _) in zip(Rs, chans)], axis=0) / 3.0
        ref = None
        for sname in SOLIDS:
            sv = povms[sname]["s"]
            C = sv.T @ sv / len(sv)          # the solid enters here only --
            M = np.mean([R.T @ C @ TN @ T    # -- and contracts to Id/3
                         for R, (T, _) in zip(Rs, chans)], axis=0)
            m = np.mean([R.T @ (sv.mean(axis=0) + C @ (TN @ t + tN))
                         for R, (_, t) in zip(Rs, chans)], axis=0)
            assert np.abs(M - M_cf).max() < 1e-12, (sname, gamma)
            aniso = np.abs(M - (np.trace(M) / 3) * np.eye(3)).max()
            bias, eta = residual(M, m)
            row = np.array([aniso, np.linalg.norm(m), eta, bias[0], bias[2]])
            out[f"gatenoise/R2/{sname}/{gamma}"] = row
            if ref is None:
                ref = row
            assert np.abs(row - ref).max() < 1e-12, (sname, gamma)
        # bare variant: no dilation channel -- the modeling choice isolated
        Mb = np.mean([R.T @ T for R, (T, _) in zip(Rs, chans)], axis=0) / 3.0
        mb = np.mean([R.T @ t for R, (_, t) in zip(Rs, chans)], axis=0) / 3.0
        biasb, etab = residual(Mb, mb)
        out[f"gatenoise/R2bare/{gamma}"] = np.array(
            [np.abs(Mb - (np.trace(Mb) / 3) * np.eye(3)).max(),
             np.linalg.norm(mb), etab, biasb[0], biasb[2]])
    # the twirled-native counterpart of the projective control above, and what
    # lets Appendix F's second control row say "both protocols" rather than
    # "the projective series": at gamma = 0 the drawn words are ideal and the
    # only noise present is the dilation, which is g-independent by
    # construction -- so sweeping ITS strength is the same test.
    sv = povms["icosahedron"]["s"]
    C = sv.T @ sv / len(sv)
    worst = np.zeros(2)
    for delta in INDEP_DELTAS:
        Td, td = chan_amp_damping(delta)
        M = np.mean([R.T @ C @ Td @ R for R in Rs], axis=0)
        m = np.mean([R.T @ (sv.mean(axis=0) + C @ td) for R in Rs], axis=0)
        bias0 = residual(M, m)[0]
        assert np.abs(bias0).max() < 1e-12, delta
        worst = np.maximum(worst, np.abs(bias0[[0, 2]]))
    out["gatenoise/indep/R2"] = worst
    g0 = out["gatenoise/R2/icosahedron/0.0"]
    assert np.abs(g0[[0, 1, 3, 4]]).max() < 1e-12    # the exact anchor:
    assert abs(g0[2] - eta_dil) < 1e-12              # dilation twirls out
    print(f"  {'R2':12s} rotations {len(Rs):3d}, mean depth "
          f"{depths.mean():.2f}, mean Phi {phis.mean():.2f}: gamma=0 -> "
          f"zero residual against the\n{'':15s}dilation channel "
          f"(eta = tr T_N/9 = {eta_dil:.4f}, calibrated away): OK; "
          "channel\n" + " " * 15 + "solid-independent (all five rows equal "
          "to 1e-12, SIC included): OK")

    print("    series        " + "".join(f"{f'g={g}':>11s}"
                                         for g in GATE_GAMMAS[1:]))
    for name in series:
        row = [out[f"gatenoise/{name}/{g}"][4] for g in GATE_GAMMAS[1:]]
        print(f"    {name:12s}  " + "".join(f"{b:+11.5f}" for b in row)
              + "   (residual Z0 bias)")
    row = [out[f"gatenoise/R2/icosahedron/{g}"][4] for g in GATE_GAMMAS[1:]]
    print(f"    {'R2 (any)':12s}  " + "".join(f"{b:+11.5f}" for b in row)
          + "   (residual Z0 bias)")
    g1 = GATE_GAMMAS[1]
    print(f"    alignment span of the Z0 residual at g={g1} "
          "(min .. max over vertex choices; R2 has no alignment):")
    for name in series:
        lo, hi = out[f"gatenoise/span/{name}/{g1}"]
        print(f"    {name:12s}  {lo:+.6f} .. {hi:+.6f}")
    return out


# ---------------------------------------------------------------------------
# n-scaling: exact estimator variance of the TFIM energy vs chain length
# ---------------------------------------------------------------------------
def tfim_terms_n(n, g):
    """Periodic TFIM terms at arbitrary size (tfim_terms is N_QUBITS-bound)."""
    terms = [(-1.0, [(i, "Z"), ((i + 1) % n, "Z")]) for i in range(n)]
    terms += [(-g, [(i, "X")]) for i in range(n)]
    return terms


def tfim_ground_sparse(n, g):
    """Ground energy and state of the periodic TFIM, sparse Lanczos."""
    dim = 1 << n
    basis = np.arange(dim)
    diag = np.zeros(dim)
    for i in range(n):                     # -sum Z_i Z_{i+1}: diagonal
        zi = 1 - 2 * ((basis >> (n - 1 - i)) & 1)
        zj = 1 - 2 * ((basis >> (n - 1 - (i + 1) % n)) & 1)
        diag -= zi * zj
    rows, cols, vals = [basis], [basis], [diag]
    for i in range(n):                     # -g sum X_i: bit flips
        rows.append(basis)
        cols.append(basis ^ (1 << (n - 1 - i)))
        vals.append(np.full(dim, -g))
    H = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(dim, dim)).tocsr()
    v0 = np.full(dim, dim ** -0.5)         # deterministic start (H stoquastic
    w, v = eigsh(H, k=1, which="SA", v0=v0)  # -> overlap with ground state)
    return w[0], v[:, 0]


def reduced_rho(psi, n, sites):
    """Reduced density matrix of |psi> on the given (sorted) sites."""
    rest = [i for i in range(n) if i not in sites]
    M = psi.reshape([2] * n).transpose(list(sites) + rest)
    M = M.reshape(2 ** len(sites), -1)
    return M @ M.conj().T


def site_pair_ops(s, x):
    """Per-site 2x2 operators of the factorized estimator.

    x has shape (3, V): x[a, k] is the per-shot value for Pauli letter a on
    outcome k.  single[a] = sum_k x[a,k] E_k is what a site one term touches
    contributes -- the frame condition makes it exactly sigma_a, for *every*
    valid dual (asserted).  pair[a, b] = sum_k x[a,k] x[b,k] E_k is the
    dual-dependent operator for a site both terms touch.
    """
    V = len(s)
    E = np.stack([(I2 + s[k, 0] * PAULI["X"] + s[k, 1] * PAULI["Y"]
                   + s[k, 2] * PAULI["Z"]) / V for k in range(V)])
    single = np.einsum("ak,kij->aij", x, E)
    pair = np.einsum("ak,bk,kij->abij", x, x, E)
    for a, name in enumerate("XYZ"):       # frame condition, exactly
        assert np.abs(single[a] - PAULI[name]).max() < 1e-11
    return single, pair


def exact_energy_variance(psi, n, terms, s, x, rho_cache):
    """Exact single-shot variance of the factorized energy estimator.

    Var = sum_{t,t'} c_t c_t' (E[o_t o_t'] - <P_t><P_t'>), with each
    E[o_t o_t'] = Tr[rho_S (x) per-site ops] on the <= 4-site union S of the
    two supports.  Support-disjoint pairs reduce to the plain quantum
    covariance <P_t P_t'> - <P_t><P_t'> for every valid dual (the frame
    condition again; asserted en route), so that part of the variance -- the
    'floor', returned alongside -- is dual-independent: optimization can only
    touch the overlapping block.
    """
    single, pair = site_pair_ops(s, x)
    sigma = {a: PAULI[nm] for a, nm in enumerate("XYZ")}

    def rho_S(S):
        if S not in rho_cache:
            rho_cache[S] = reduced_rho(psi, n, list(S))
        return rho_cache[S]

    supports = [dict((i, AXIS[a]) for i, a in sites) for _, sites in terms]
    coeffs = np.array([c for c, _ in terms])
    means = np.empty(len(terms))
    for ti, sup in enumerate(supports):
        S = tuple(sorted(sup))
        means[ti] = np.trace(rho_S(S) @ kron_all(
            [sigma[sup[i]] for i in S])).real
    e_mean = coeffs @ means

    second, floor, worst = 0.0, 0.0, 0.0
    for ti, s1 in enumerate(supports):
        for tj, s2 in enumerate(supports):
            S = tuple(sorted(set(s1) | set(s2)))
            ops = [pair[s1[i], s2[i]] if (i in s1 and i in s2)
                   else sigma[s1[i]] if i in s1 else sigma[s2[i]]
                   for i in S]
            val = np.trace(rho_S(S) @ kron_all(ops)).real
            second += coeffs[ti] * coeffs[tj] * val
            if not set(s1) & set(s2):      # disjoint: dual-independent
                pval = np.trace(rho_S(S) @ kron_all(
                    [sigma[(s1 | s2)[i]] for i in S])).real
                worst = max(worst, abs(val - pval))
                floor += coeffs[ti] * coeffs[tj] * (pval - means[ti] * means[tj])
    assert worst < 1e-10
    return second - e_mean ** 2, floor, e_mean


def nscaling(povms, states):
    """Exact variance of the factorized energy estimator, n = 4 .. 16.

    Resolves the 'everything is at n = 4' caveat by computing the icosahedral
    estimator's exact single-shot variance on the critical TFIM energy per
    dual level (canonical / observable-optimized / per-site oracle) as the
    chain grows.  The per-site variance and the optimization gain both
    converge fast -- the gain to a constant ~15%, not to zero -- and the
    dual-independent covariance floor stays a small share, so what caps the
    gain is the per-site quantum limit of the factorized class, not
    long-range correlations.  Cross-checks: at n = 4 the marginal route
    equals the full outcome-tensor route on all three levels, and the
    canonical-dual curve is identical for octahedron and icosahedron
    (the exact-landscape proposition, at every n).
    """
    print("\nn-scaling: exact variance of the factorized TFIM-energy "
          f"estimator, icosahedron,\nn = {NSCALE_NS[0]}..{NSCALE_NS[-1]} "
          "(sparse ground states; canonical / observable-opt / oracle):")
    s_ico = povms["icosahedron"]["s"]
    s_oct = povms["octahedron"]["s"]

    def x_canonical(s):
        return 3.0 * s.T.copy()

    def x_letter(s, rbar):
        b = letter_duals(s, rbar)
        return np.stack([2.0 * b[a][:, a] for a in range(3)])

    out = {}
    print(f"    {'n':>2s} {'E0':>10s} {'<X>':>8s} {'Var can':>10s} "
          f"{'Var obs':>10s} {'Var orc':>10s} {'floor':>8s} {'gain':>7s}")
    for n in NSCALE_NS:
        terms = tfim_terms_n(n, G_CRIT)
        e0, psi = tfim_ground_sparse(n, G_CRIT)
        # Lanczos converged on the ground state, not on a low excited one --
        # the free-fermion energy is an independent witness at every n.
        assert abs(e0 - tfim_ground_energy_exact(n, G_CRIT)) < 1e-9, (n, e0)
        cache = {}
        xs = []
        for g in TRAIN_G:                  # training mean, as in Experiment 2
            _, pg = tfim_ground_sparse(n, g)
            r1 = reduced_rho(pg, n, [0])
            xs.append([np.trace(r1 @ PAULI[a]).real for a in "XYZ"])
        rbar = np.mean(xs + [[0.0, 0.0, 0.0]], axis=0)      # + GHZ (= 0)
        r1 = reduced_rho(psi, n, [0])
        r_true = np.array([np.trace(r1 @ PAULI[a]).real for a in "XYZ"])

        var_can, floor, e_mean = exact_energy_variance(
            psi, n, terms, s_ico, x_canonical(s_ico), cache)
        var_obs, floor2, _ = exact_energy_variance(
            psi, n, terms, s_ico, x_letter(s_ico, rbar), cache)
        var_orc, floor3, _ = exact_energy_variance(
            psi, n, terms, s_ico, x_letter(s_ico, r_true), cache)
        assert abs(floor - floor2) < 1e-9 and abs(floor - floor3) < 1e-9
        assert abs(e_mean - e0) < 1e-8
        var_oct, _, _ = exact_energy_variance(
            psi, n, terms, s_oct, x_canonical(s_oct), cache)
        assert abs(var_oct - var_can) < 1e-9       # antipodal universality

        if n == 4:                         # cross-route + cross-module checks
            rho4 = states["TFIM"]
            assert abs(e0 - exact_value(rho4, OBSERVABLES["E_TFIM"])) < 1e-9
            p4 = born_tensor(rho4, povms["icosahedron"]["E"])
            for x, v_marg in [(x_canonical(s_ico), var_can),
                              (x_letter(s_ico, rbar), var_obs),
                              (x_letter(s_ico, r_true), var_orc)]:
                v_full = exact_second_moment(
                    p4, OBSERVABLES["E_TFIM"], lut_uniform(x)) - e0 ** 2
                assert abs(v_full - v_marg) < 1e-9
            print("    n=4: marginal route == full outcome-tensor route "
                  "(all three levels): OK")

        gain = 1.0 - var_orc / var_can
        out[f"nscale/{n}/var"] = np.array([var_can, var_obs, var_orc])
        out[f"nscale/{n}/floor"] = np.array([floor])
        out[f"nscale/{n}/state"] = np.array([e0, r_true[0]])
        print(f"    {n:>2d} {e0:>10.5f} {r_true[0]:>8.5f} {var_can:>10.4f} "
              f"{var_obs:>10.4f} {var_orc:>10.4f} {floor:>8.4f} {gain:>6.1%}")

    g4 = 1 - out["nscale/4/var"][2] / out["nscale/4/var"][0]
    g16 = 1 - out["nscale/16/var"][2] / out["nscale/16/var"][0]
    print(f"  frame condition / disjoint-pair dual-independence at every "
          "size: OK (asserted)\n  oracle gain converges: "
          f"{g4:.1%} (n=4) -> {g16:.1%} (n=16); floor share "
          f"{out['nscale/16/floor'][0] / out['nscale/16/var'][0]:.1%} at n=16")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)
    print("Shadow estimation with Platonic-solid POVMs "
          f"(n={N_QUBITS}, seed={RNG_SEED})")
    povms = load_povms()
    states, haar = make_states(rng)
    e_tfim = exact_value(states["TFIM"], OBSERVABLES["E_TFIM"])
    e_exact = tfim_ground_energy_exact(N_QUBITS, G_CRIT)
    # 1e-9, the same bound the sparse route carries in nscaling().  Residual is
    # 6.2e-15 (~7 ulps on |E| = 5.226) but comes out of LAPACK's eigh, which a
    # different build or thread count can move by an order.
    assert abs(e_tfim - e_exact) < 1e-9, (e_tfim, e_exact)
    print(f"  TFIM h=1 ground energy: {e_tfim:.6f} "
          f"(free fermions: {e_exact:.6f}, |diff| {abs(e_tfim - e_exact):.1e})")

    exp1, exp1_haar = experiment_1(povms, states, haar, rng)
    _, exact_out = exact_backbone(povms, states, haar, exp1, exp1_haar)
    four_out = fourth_moment_ladder(povms, states, haar)
    exp2 = experiment_2(povms, states, rng)
    ratio_out = exact_exp2_ratios(povms, states, exp2)
    exp3 = experiment_3(povms, states["TFIM"], rng)
    blind, aniso = blindness(povms, states["TFIM"], rng)
    twirl_check(povms)
    tp_out = two_protocol_twirl(povms)
    gate_out = gate_noise_twirl(states, povms)
    nscale_out = nscaling(povms, states)

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
    for extra in (exact_out, four_out, ratio_out, tp_out, gate_out,
                  nscale_out):
        out.update(extra)
    np.savez_compressed(DATA / "shadow_experiments.npz", **out)
    print(f"\n  -> {DATA / 'shadow_experiments.npz'} ({len(out)} keys)")
    print(f"\nAll checks passed.  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
