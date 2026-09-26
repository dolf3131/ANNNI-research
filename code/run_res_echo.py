"""Resolution experiment: echo readout, ref5, ESPRIT order 24, no filter.

Order 24 was selected on noisy data at matched Gamma, where it recovered every
field above 2*pi/T against two of six for the Hadamard test. Gamma is measured
at each error rate rather than extrapolated, because Gamma/p drifts by a third
over the range studied.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pennylane as qml
from scipy.sparse.linalg import eigsh

from paths import RESULTS, MANUSCRIPT
from annni import hamiltonian
from circuits import echo_signal, ref5, sample_probability
from spectral import gap_from_echo

N, KAPPA, DT, SHOTS = 6, 0.2, 0.30, 10_000


ORDER, RELF = 24, 0.0
P_LIST = [0.00015, 0.00037, 0.00075, 0.0015, 0.0030]
HS = np.round(np.concatenate([np.arange(0.30, 0.80, 0.06),
                              np.arange(0.85, 1.61, 0.15)]), 3)
SEEDS = range(5)
LOG_SNR = np.log(np.sqrt(SHOTS))



PSI0 = ref5(N)


import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figstyle


def measure_gamma(p, nt=160):
    t = DT * np.arange(nt)
    c = np.abs(echo_signal(N, KAPPA, 0.7, nt, DT) - 1 / 2**N)
    d = np.abs(echo_signal(N, KAPPA, 0.7, nt, DT, p) - 1 / 2**N)
    ok = c > 0.05
    sl, ic = np.polyfit(t[ok], np.log(d[ok] / c[ok]), 1)
    return -float(sl), float(np.std(np.log(d[ok] / c[ok]) - (sl * t[ok] + ic)))


print(f"echo resolution: ref5, order={ORDER}, filter={RELF}", flush=True)
PLOT_ONLY = "--plot-only" in sys.argv

if PLOT_ONLY:
    d = np.load(RESULTS / "res_echo.npz")
    HS, gaps = d["HS"], d["gaps"]
    P_LIST = list(d["P_LIST"])
    res = {p: d[f"est_{p}"] for p in P_LIST}
    meta = {p: (float(g), 0, float(d[f"dE_{p}"])) for p, g in zip(P_LIST, d["gammas"])}
    print(f"replotting res_echo.npz: {len(P_LIST)} rates, {len(HS)} fields")
else:
    cal = {}
    for p in P_LIST:
        g, r = measure_gamma(p)
        cal[p] = g
        print(f"  p={p:.5f} Gamma={g:.4f} (ratio {g/p:6.1f}) resid {r:.3f}", flush=True)

    gaps = np.array([np.diff(np.sort(eigsh(
        hamiltonian(N, KAPPA, h).sparse_matrix(wire_order=range(N)).tocsc().real,
        k=2, which="SA", return_eigenvectors=False))[:2])[0] for h in HS])

    res, meta = {}, {}
    for p in P_LIST:
        gamma = cal[p]
        nt = int(np.clip(LOG_SNR / gamma / DT, 48, 1400))
        dE = 2 * np.pi / (nt * DT)
        meta[p] = (gamma, nt, dE)
        est, t0 = [], time.time()
        for h in HS:
            base = echo_signal(N, KAPPA, h, nt, DT, p)
            est.append(np.median([gap_from_echo(sample_probability(base, SHOTS, np.random.default_rng(1000 + s)),
                                           DT, ORDER, RELF) for s in SEEDS]))
        res[p] = np.array(est)
        print(f"p={p:.5f} Gamma={gamma:.4f} nt={nt:5d} dE={dE:.4f} [{(time.time()-t0)/60:.1f}m]",
              flush=True)
        np.savez(RESULTS / "res_echo.npz", HS=HS, gaps=gaps, P_LIST=P_LIST,
                 gammas=[cal[q] for q in P_LIST],
                 **{f"est_{q}": res[q] for q in res}, **{f"dE_{q}": meta[q][2] for q in meta})


print()
agree = total = 0
for p in P_LIST:
    gamma, nt, dE = meta[p]
    good = np.abs(res[p] - gaps) < 0.15 * gaps + 0.02
    pred = gaps > dE
    print(f"Gamma={gamma:.4f} dE={dE:.4f} | above {good[pred].mean():5.0%} ({pred.sum():2d})"
          f" | below {good[~pred].mean():5.0%} ({(~pred).sum():2d})")
    agree += (good == pred).sum(); total += len(good)
print(f"\ncriterion predicts the outcome at {agree}/{total} ({agree/total:.0%})")

figstyle.use()
fig, (ax, sx) = plt.subplots(2, 1, figsize=(figstyle.COLUMN, 3.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.5], "hspace": 0.08})
cols = plt.cm.viridis(np.linspace(0.12, 0.85, len(P_LIST)))
ax.semilogy(HS, gaps, "-", color="0.15", lw=1.4, label=r"exact $E_1-E_0$")
for c, p in zip(cols, P_LIST):
    ax.axhline(meta[p][2], color=c, ls="--", lw=0.9)
    ax.axvline(float(np.interp(meta[p][2], gaps, HS)), color=c, ls=":", lw=0.7)
ax.set_ylabel("level spacing"); ax.tick_params(labelbottom=False)
ax.legend(fontsize=7, loc="lower right")
for row, (c, p) in enumerate(zip(cols, P_LIST)):
    good = np.abs(res[p] - gaps) < 0.15 * gaps + 0.02
    y = np.full(len(HS), len(P_LIST) - 1 - row, float)
    sx.plot(HS[good], y[good], "o", color=c, ms=3.6)
    sx.plot(HS[~good], y[~good], "o", mfc="white", mec=c, ms=3.6, mew=0.8)
    sx.axvline(float(np.interp(meta[p][2], gaps, HS)), color=c, ls=":", lw=0.7)
sx.set_yticks(range(len(P_LIST)))
sx.set_yticklabels([rf"$\Gamma$={meta[p][0]:.3f}" for p in reversed(P_LIST)], fontsize=6)
sx.set_ylim(-0.7, len(P_LIST) - 0.3); sx.set_xlabel("$h$"); sx.tick_params(axis="y", length=0)
fig.savefig(MANUSCRIPT / "resolvability_echo.pdf", bbox_inches="tight")
print(f"wrote {MANUSCRIPT / 'resolvability_echo.pdf'}")
