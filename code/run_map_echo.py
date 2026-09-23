"""Phase map: echo readout, ref5, ESPRIT order 16, no amplitude filter.

Settings chosen on the noiseless 30-point validation grid (92% of sites above
2*pi/T recovered to 5%). The noisy experiment uses order 24 instead: the model
order has to match the signal-to-noise ratio, so it is selected separately on
data matched to each condition.
"""
import sys
import time
from pathlib import Path
import numpy as np
import pennylane as qml
from scipy.sparse.linalg import eigsh

from paths import RESULTS, MANUSCRIPT
from annni import hamiltonian, ferro_para_boundary, antiphase_boundary
from circuits import echo_signal, ref5
from spectral import gap_from_echo

N, DT, NT = 6, 0.30, 800
DE = 2 * np.pi / (NT * DT)





PSI0 = ref5(N)


import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import figstyle


ORDER, RELF = 16, 0.0
KS = np.linspace(0.0, 1.0, 21)
HS = np.linspace(0.05, 1.70, 21)
CK = RESULTS / "map_echo.npz"

PLOT_ONLY = "--plot-only" in sys.argv

if PLOT_ONLY:
    d = np.load(CK)
    circ, ref = d["circ"], d["ref"]
    print(f"replotting {CK.name}, {circ.shape} grid, nt={int(d['NT'])}")
else:
    circ = np.zeros((len(HS), len(KS))); ref = np.zeros_like(circ); start = 0
    if CK.exists():
        d = np.load(CK)
        if d["circ"].shape == circ.shape:
            circ, ref, start = d["circ"].copy(), d["ref"].copy(), int(d["rows_done"])

    print(f"echo phase map: {len(HS)}x{len(KS)}, nt={NT}, order={ORDER}, dE={DE:.4f}", flush=True)
    t0 = time.time()
    for a in range(start, len(HS)):
        h = HS[a]
        for b, k in enumerate(KS):
            circ[a, b] = gap_from_echo(echo_signal(N, k, h, NT, DT), DT, ORDER, RELF)
            m = hamiltonian(N, k, h).sparse_matrix(wire_order=range(N)).tocsc().real
            w = np.sort(eigsh(m, k=2, which="SA", return_eigenvectors=False))
            ref[a, b] = w[1] - w[0]
        el = time.time() - t0; n = a - start + 1
        print(f"  h={h:.3f} [{el/60:.1f}m, eta {el/n*(len(HS)-a-1)/60:.0f}m]", flush=True)
        np.savez(CK, circ=circ, ref=ref, KS=KS, HS=HS, N=N, DT=DT, NT=NT, DE=DE,
                 ORDER=ORDER, rows_done=a + 1)


ok = np.abs(circ - ref) < 0.05 * ref
print(f"\nwithin 5%: {ok.sum()}/{ok.size} ({ok.mean():.0%})")
print(f"  gap > dE : {ok[ref > DE].mean():.0%}   gap < dE : {ok[ref < DE].mean():.0%}")

figstyle.use()
fig, axes = plt.subplots(1, 2, figsize=(figstyle.FULLWIDTH, 2.9), constrained_layout=True)
kk, ka = np.linspace(0.001, 0.499, 200), np.linspace(0.5, 1, 100)
for ax, dat, t in ((axes[0], circ, "gap from the echo circuit"),
                   (axes[1], ref, r"exact $E_1-E_0$ (benchmark)")):
    im = ax.pcolormesh(KS, HS, np.clip(dat, 1e-4, None), cmap="magma",
                       norm=LogNorm(1e-4, 3), shading="nearest", rasterized=True)
    ax.plot(kk, ferro_para_boundary(kk), "w--", lw=1.0)
    ax.plot(ka, antiphase_boundary(ka), "w:", lw=1.1)
    ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("$h$"); ax.set_title(t, fontsize=9)
    fig.colorbar(im, ax=ax)
axes[0].plot([], [], "w--", lw=1.0, label="analytic boundaries")
axes[0].legend(fontsize=6.5, loc="upper left")
fig.savefig(MANUSCRIPT / "phase_map_echo.pdf", bbox_inches="tight")
print(f"wrote {MANUSCRIPT / 'phase_map_echo.pdf'}")
