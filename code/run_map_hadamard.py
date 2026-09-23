"""Phase map from the Hadamard test, with the settings fixed by tune.py.

  reference ref5 (plus, ferromagnet, period-4, two domain walls)
  ESPRIT order 32, amplitude filter 0.01
  fourth-order Suzuki, dt = 0.30, nt = 800, N = 6

Chosen because that combination scored 97% of validation points within 5% and
100% of those above 2*pi/T, on a plateau rather than a spike -- order 48 scored
higher but collapsed when the filter moved either way.
"""
import time
from pathlib import Path
import numpy as np
import pennylane as qml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse.linalg import eigsh

from paths import RESULTS
from annni import hamiltonian, ferro_para_boundary, antiphase_boundary
from circuits import hadamard_signal, ref5
from spectral import gap_from_levels
import figstyle


N, DT, NT = 6, 0.30, 800
ORDER, RELF = 32, 0.01
KS = np.linspace(0.0, 1.0, 21)
HS = np.linspace(0.05, 1.70, 21)
DE = 2 * np.pi / (NT * DT)
CK = RESULTS / "map_hadamard.npz"





PSI0 = ref5(N)


circ = np.zeros((len(HS), len(KS)))
ref = np.zeros_like(circ)
start = 0
if CK.exists():
    d = np.load(CK)
    if d["circ"].shape == circ.shape:
        circ, ref, start = d["circ"].copy(), d["ref"].copy(), int(d["rows_done"])
        print(f"resuming from row {start}", flush=True)

print(f"N={N} grid {len(HS)}x{len(KS)} nt={NT} order={ORDER} filter={RELF} dE={DE:.4f}",
      flush=True)
t0 = time.time()
for a in range(start, len(HS)):
    h = HS[a]
    for b, k in enumerate(KS):
        circ[a, b] = gap_from_levels(hadamard_signal(N, k, h, NT, DT), DT, ORDER, RELF)
        m = hamiltonian(N, k, h).sparse_matrix(wire_order=range(N)).tocsc().real
        w = np.sort(eigsh(m, k=2, which="SA", return_eigenvectors=False))
        ref[a, b] = w[1] - w[0]
    el = time.time() - t0
    n = a - start + 1
    print(f"  h={h:.3f} [{el/60:.1f}m, {el/n/60:.1f}m/row, eta {el/n*(len(HS)-a-1)/60:.0f}m]",
          flush=True)
    np.savez(CK, circ=circ, ref=ref, KS=KS, HS=HS, N=N, DT=DT, NT=NT, DE=DE,
             ORDER=ORDER, RELF=RELF, rows_done=a + 1)

ok = np.abs(circ - ref) < 0.05 * ref
print(f"\nwithin 5%: {ok.sum()}/{ok.size} ({ok.mean():.0%})")
print(f"  gap > dE : {ok[ref > DE].mean():.0%}   gap < dE : {ok[ref < DE].mean():.0%}")


figstyle.use()
fig, axes = plt.subplots(1, 2, figsize=(figstyle.FULLWIDTH, 2.9), constrained_layout=True)
kk, ka = np.linspace(0.001, 0.499, 200), np.linspace(0.5, 1, 100)
for ax, dat, t in ((axes[0], circ, "gap from the Hadamard-test circuit"),
                   (axes[1], ref, r"exact $E_1-E_0$ (benchmark)")):
    im = ax.pcolormesh(KS, HS, np.clip(dat, 1e-4, None), cmap="magma",
                       norm=LogNorm(1e-4, 3), shading="nearest", rasterized=True)
    ax.plot(kk, ferro_para_boundary(kk), "w--", lw=1.0)
    ax.plot(ka, antiphase_boundary(ka), "w:", lw=1.1)
    ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("$h$"); ax.set_title(t, fontsize=9)
    fig.colorbar(im, ax=ax)
axes[0].plot([], [], "w--", lw=1.0, label="analytic boundaries")
axes[0].legend(fontsize=6.5, loc="upper left")
fig.savefig(RESULTS / "phase_map_hadamard.pdf", bbox_inches="tight")
print(f"wrote {RESULTS / 'phase_map_hadamard.pdf'}")
