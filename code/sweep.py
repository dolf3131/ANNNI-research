"""Map the ANNNI phase diagram on a (kappa, h) grid by exact diagonalisation.

Writes phase_diagram.npz (raw data) and phase_diagram.png (four panels:
three order parameters plus a model-free fidelity-drop map).
"""

from pathlib import Path

import numpy as np

import figstyle

figstyle.use()
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from paths import RESULTS, MANUSCRIPT
from annni import order_parameters


DATA = RESULTS / "phase_diagram.npz"

N = 6
KAPPAS = np.linspace(0.0, 1.0, 41)
FIELDS = np.linspace(0.0, 2.0, 41)


def ferro_para_boundary(kappa):
    """Standard ANNNI estimate, valid for 0 < kappa < 0.5. It runs from h = 1 at
    kappa = 0 (where the model is just the transverse-field Ising chain) down to
    h = 0 at the multiphase point kappa = 1/2."""
    k = np.clip(kappa, 1e-6, 0.5)
    return (1.0 - k) / k * (1.0 - np.sqrt((1.0 - 3.0 * k + 4.0 * k**2) / (1.0 - k)))


def antiphase_boundary(kappa):
    """Antiphase -> floating, the usual h ~ 1.05 sqrt(kappa - 1/2) fit."""
    return 1.05 * np.sqrt(np.clip(kappa - 0.5, 0.0, None))


def run():
    shape = (len(FIELDS), len(KAPPAS))
    ferro = np.zeros(shape)
    anti = np.zeros(shape)
    para = np.zeros(shape)
    states = np.zeros(shape + (2**N,))

    for a, h in enumerate(FIELDS):
        for b, kappa in enumerate(KAPPAS):
            op = order_parameters(N, kappa, h)
            ferro[a, b], anti[a, b], para[a, b] = op["ferro"], op["antiphase"], op["para"]
            states[a, b] = op["state"]
        print(f"h = {h:.2f} done", flush=True)

    # Model-free boundary detector: 1 - |<psi(h)|psi(h + dh)>| spikes at a
    # transition, without assuming which order parameter is the right one.
    # eigsh fixes no global sign, hence the absolute value.
    overlap = np.abs(np.einsum("ijk,ijk->ij", states[:-1], states[1:]))
    fidelity_drop = np.vstack([1.0 - overlap, np.full((1, len(KAPPAS)), np.nan)])
    # At h = 0 the ground state is exactly degenerate (Z2, and the antiphase
    # sector on top of that), so eigsh returns an arbitrary vector from the
    # degenerate subspace and the overlap against h = dh is meaningless. The
    # order parameters are built from symmetry-invariant correlators and stay
    # valid there; only this panel has to drop the row.
    if FIELDS[0] == 0.0:
        fidelity_drop[0] = np.nan

    np.savez(DATA, n=N, kappas=KAPPAS, fields=FIELDS,
             ferro=ferro, antiphase=anti, para=para, fidelity_drop=fidelity_drop)
    return ferro, anti, para, fidelity_drop


def plot(ferro, anti, para, fidelity_drop):
    # Raw 1 - |overlap| is O(dh**2) and only resolves the sharp ferro/antiphase
    # boundary. The fidelity susceptibility chi = 2(1-|overlap|)/dh**2 is the
    # standard normalisation and picks up the continuous transitions too.
    dh = FIELDS[1] - FIELDS[0]
    chi = 2.0 * fidelity_drop / dh**2

    panels = [
        (ferro, "Ferromagnetic order  $S(k{=}0)$", "viridis"),
        (anti, r"Antiphase order  $S(k{=}\pi/2)$", "magma"),
        (para, r"Paramagnetic order  $\langle X\rangle$", "cividis"),
        (chi, r"Fidelity susceptibility  $\chi_F$  (log scale)", "inferno"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(figstyle.FULLWIDTH, 5.0),
                             constrained_layout=True)
    extent = [KAPPAS[0], KAPPAS[-1], FIELDS[0], FIELDS[-1]]
    fine = np.linspace(0, 1, 200)

    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        norm = LogNorm(*np.nanpercentile(data, [5, 100])) if data is chi else None
        # rasterised: the pixel data would otherwise bloat the vector file,
        # while the axes, ticks and labels stay as text.
        im = ax.imshow(data, origin="lower", aspect="auto", extent=extent,
                       cmap=cmap, norm=norm, rasterized=True)
        ax.plot(fine[fine < 0.5], ferro_para_boundary(fine[fine < 0.5]), "w--", lw=1.1)
        ax.plot(fine[fine > 0.5], antiphase_boundary(fine[fine > 0.5]), "w:", lw=1.3)
        ax.axvline(0.5, color="w", ls="-", lw=0.5, alpha=0.45)
        ax.set_title(title)
        ax.set_xlabel(r"$\kappa$")
        ax.set_ylabel("$h$")
        cb = fig.colorbar(im, ax=ax)
        cb.outline.set_linewidth(0.5)

    fig.savefig(MANUSCRIPT / "phase_diagram.pdf", dpi=300)
    print(f"wrote {MANUSCRIPT / 'phase_diagram.pdf'}")


def load():
    d = np.load(DATA)
    return d["ferro"], d["antiphase"], d["para"], d["fidelity_drop"]


if __name__ == "__main__":
    import sys
    plot(*(load() if "--plot-only" in sys.argv else run()))
