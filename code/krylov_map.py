"""Where Krylov works and where it does not, as a map over (subspace size, Trotter step).

Three pairings of the two matrices the method needs:

  consistent   S and H both read from the Trotterised propagator
  mismatched   S from the Trotterised propagator, H from exact evolution
  exact        both from exact evolution

The middle one is the mistake that produced our earlier, wrong conclusion that
Krylov is fragile to Trotter error. It is not: with consistent matrix elements
it converges to the gap of the propagator it was actually given. Mixing them
diverges, and the divergence grows with the subspace dimension because that is
where the overlap matrix becomes ill-conditioned enough to amplify the
inconsistency.
"""

import numpy as np
import pennylane as qml
from scipy.linalg import eigh, toeplitz
from scipy.sparse.linalg import expm_multiply, eigsh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from annni import hamiltonian
from circuits import ref5, s4
from spectral import esprit
import figstyle
from paths import RESULTS, MANUSCRIPT

N, KAPPA, H_FIELD = 6, 0.2, 0.66
DEPTHS = np.array([12, 18, 24, 36, 48, 72, 96, 150])
STEPS = np.array([0.40, 0.30, 0.20, 0.15, 0.10, 0.075, 0.05])
TOTAL_TIME = 120.0
ESPRIT_ORDER = 24


def trotter_propagator(dt):
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev)
    def c():
        s4(N, KAPPA, H_FIELD, dt)
        return qml.state()

    return qml.matrix(c)()


def records(dt, nt, trotterised):
    "g(t) and <psi0|H U(t)|psi0> from one propagator, exact or Trotterised."
    hs = hamiltonian(N, KAPPA, H_FIELD).sparse_matrix(wire_order=range(N)).tocsc()
    psi = ref5(N)
    u = trotter_propagator(dt) if trotterised else None
    g = np.empty(nt, dtype=complex)
    hser = np.empty(nt, dtype=complex)
    cur = psi.copy()
    for i in range(nt):
        g[i] = psi.conj() @ cur
        hser[i] = psi.conj() @ (hs @ cur)
        cur = u @ cur if trotterised else expm_multiply(-1j * dt * hs, cur)
    return g, hser


def krylov_gap(g, hser, depth, thresh=1e-10):
    s = toeplitz(np.conj(g[:depth]), g[:depth])
    hm = toeplitz(np.conj(hser[:depth]), hser[:depth])
    s, hm = (s + s.conj().T) / 2, (hm + hm.conj().T) / 2
    w, v = eigh(s)
    keep = w > thresh * w.max()
    if keep.sum() < 2:
        return np.nan
    x = v[:, keep] / np.sqrt(w[keep])
    e = np.sort(eigh(x.conj().T @ hm @ x, eigvals_only=True))
    return float(e[1] - e[0])


def condition_number(g, depth):
    s = toeplitz(np.conj(g[:depth]), g[:depth])
    w = np.linalg.eigvalsh((s + s.conj().T) / 2)
    return float(abs(w).max() / max(abs(w).min(), 1e-300))


def run():
    hs = hamiltonian(N, KAPPA, H_FIELD).sparse_matrix(wire_order=range(N)).tocsc().real
    w = np.sort(eigsh(hs, k=2, which="SA", return_eigenvectors=False))
    exact = float(w[1] - w[0])

    shape = (len(STEPS), len(DEPTHS))
    out = {k: np.full(shape, np.nan) for k in ("consistent", "mismatched", "exact")}
    cond = np.full(shape, np.nan)
    esp = np.full(len(STEPS), np.nan)

    for a, dt in enumerate(STEPS):
        nt = int(TOTAL_TIME / dt)
        g_t, h_t = records(dt, nt, True)
        g_e, h_e = records(dt, nt, False)
        esp[a] = abs(np.diff(np.sort(esprit(g_t, dt, ESPRIT_ORDER))[:2])[0] - exact) / exact
        for b, d in enumerate(DEPTHS):
            if d > nt:
                continue
            out["consistent"][a, b] = abs(krylov_gap(g_t, h_t, d) - exact) / exact
            out["mismatched"][a, b] = abs(krylov_gap(g_t, h_e, d) - exact) / exact
            out["exact"][a, b] = abs(krylov_gap(g_e, h_e, d) - exact) / exact
            cond[a, b] = condition_number(g_t, d)
        print(f"  dt={dt:.3f} done", flush=True)

    np.savez(RESULTS / "krylov_map.npz", exact_gap=exact, DEPTHS=DEPTHS, STEPS=STEPS,
             cond=cond, esprit=esp, **out)
    return out, cond, esp, exact


def plot(out, cond, esp, exact):
    figstyle.use()
    fig = plt.figure(figsize=(figstyle.FULLWIDTH, 4.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.85],
                          width_ratios=[1, 1, 1, 0.055],
                          hspace=0.42, wspace=0.18, left=0.07, right=0.95,
                          top=0.93, bottom=0.10)

    panels = [("consistent", "$S$, $H$ both Trotterised"),
              ("mismatched", "$S$ Trotterised, $H$ exact"),
              ("exact", "$S$, $H$ both exact")]
    norm = LogNorm(1e-6, 1e3)
    for i, (key, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        im = ax.pcolormesh(np.arange(len(DEPTHS)), np.arange(len(STEPS)),
                           np.clip(out[key], 1e-6, 1e3), cmap="RdYlGn_r",
                           norm=norm, shading="nearest")
        ax.set_xticks(range(len(DEPTHS))); ax.set_xticklabels(DEPTHS, fontsize=6)
        ax.set_xlabel("subspace dimension $D$")
        ax.set_title(title, fontsize=8)
        if i == 0:
            ax.set_yticks(range(len(STEPS)))
            ax.set_yticklabels([f"{d:g}" for d in STEPS], fontsize=6)
            ax.set_ylabel(r"Trotter step $\delta t$")
        else:
            ax.set_yticks(range(len(STEPS))); ax.set_yticklabels([])
    fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]), label="relative gap error")

    ax = fig.add_subplot(gs[1, :])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(DEPTHS, out["mismatched"][1], "s-", ms=4, color="#c0392b",
            label=r"mismatched pair, $\delta t=0.30$")
    ax.plot(DEPTHS, out["consistent"][1], "^-", ms=4, color="#1f6fb4",
            label=r"consistent pair, $\delta t=0.30$")
    ax.plot(DEPTHS, out["exact"][1], "o-", ms=4, color="#2e8b57",
            label=r"both exact, $\delta t=0.30$")
    ax.axhline(esp[1], color="0.35", ls="--", lw=1.1,
               label=f"ESPRIT, order {ESPRIT_ORDER}")
    ax.set_xlabel("subspace dimension $D$")
    ax.set_ylabel("relative gap error")
    ax.set_xticks(DEPTHS); ax.set_xticklabels(DEPTHS, fontsize=7)
    ax.minorticks_off()
    ax.grid(alpha=0.25, lw=0.4)

    cx = ax.twinx()
    cx.set_yscale("log")
    cx.plot(DEPTHS, cond[1], ":", lw=1.4, color="0.45")
    cx.set_ylabel(r"$\mathrm{cond}(S)$", color="0.45", fontsize=8)
    cx.tick_params(axis="y", colors="0.45", labelsize=7)

    h1, l1 = ax.get_legend_handles_labels()
    h1.append(plt.Line2D([], [], ls=":", lw=1.4, color="0.45"))
    l1.append(r"$\mathrm{cond}(S)$, right axis")
    ax.legend(h1, l1, fontsize=6.5, ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), frameon=False)

    fig.savefig(MANUSCRIPT / "krylov_map.pdf", bbox_inches="tight")
    print(f"wrote {MANUSCRIPT / 'krylov_map.pdf'}")


if __name__ == "__main__":
    plot(*run())
