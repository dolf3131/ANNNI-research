"""Choose the estimator settings, on data matched to the condition they will be used on.

This is one script because the choice is one 2x2: two readouts, with and
without noise. Running it as four separate scripts is how a model order picked
on clean signals reached a noisy experiment, where it recovered 12% of the
fields instead of 100%.

    python code/tune.py echo clean        noiseless order for the phase map
    python code/tune.py echo noisy        order for the resolution experiment
    python code/tune.py hadamard clean    the same two for the other readout,
    python code/tune.py hadamard noisy      which is how Table III is built

The clean arms also compare the three-component reference against the
five-component one, which is where the domain walls earn their place.
"""

import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh

from annni import hamiltonian
from circuits import echo_signal, hadamard_signal, ref5, sample_probability, sample_pm1
from spectral import gap_from_echo, gap_from_levels

N, KAPPA, DT, SHOTS = 6, 0.2, 0.30, 10_000
ORDERS = (12, 16, 24, 32, 48)
THRESHOLDS = (0.0, 0.01, 0.05)

CLEAN_NT = 800
CLEAN_KS = np.linspace(0.0, 1.0, 6)
CLEAN_HS = np.linspace(0.3, 1.5, 5)

NOISY_GAMMA = 0.104                 # matched across readouts, since Gamma is what
NOISY_HS = [0.42, 0.54, 0.66, 0.78, 0.90, 1.05, 1.25, 1.50]   # enters the resolution
SEEDS = range(5)


def ref3(n):
    "The reference before the domain walls were added: even under reflection."
    v = np.ones(2**n, dtype=complex) / 2 ** (n / 2)
    for pattern in ("0" * n, ("0011" * n)[:n]):
        v[int(pattern, 2)] += 1.0
    return v / np.linalg.norm(v)


def exact_gap(kappa, h):
    m = hamiltonian(N, kappa, h).sparse_matrix(wire_order=range(N)).tocsc().real
    w = np.sort(eigsh(m, k=2, which="SA", return_eigenvectors=False))
    return float(w[1] - w[0])


def scan_clean(readout):
    signal, gap = ((echo_signal, gap_from_echo) if readout == "echo"
                   else (hadamard_signal, gap_from_levels))
    pts = [(float(k), float(h)) for k in CLEAN_KS for h in CLEAN_HS]
    dE = 2 * np.pi / (CLEAN_NT * DT)
    print(f"{readout}, noiseless: {len(pts)} points, nt={CLEAN_NT}, dE={dE:.4f}", flush=True)

    refs = {"ref3": ref3(N), "ref5": ref5(N)}
    sigs, exact, t0 = {}, {}, time.time()
    for i, (k, h) in enumerate(pts):
        exact[(k, h)] = exact_gap(k, h)
        for name, psi in refs.items():
            sigs[(name, k, h)] = signal(N, k, h, CLEAN_NT, DT, psi0=psi)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pts)} [{(time.time()-t0)/60:.1f}m]", flush=True)

    print(f"\n{'reference':>10} {'order':>6} {'threshold':>10} | {'all':>6} {'above dE':>9}")
    for name in refs:
        for order in ORDERS:
            for rel in THRESHOLDS:
                ok, above = [], []
                for k, h in pts:
                    v = gap(sigs[(name, k, h)], DT, order, rel)
                    good = np.isfinite(v) and abs(v - exact[(k, h)]) < 0.05 * exact[(k, h)]
                    ok.append(good)
                    if exact[(k, h)] > dE:
                        above.append(good)
                print(f"{name:>10} {order:>6} {rel:>10.2f} | "
                      f"{np.mean(ok):5.0%} {np.mean(above):8.0%}")


def scan_noisy(readout):
    if readout == "echo":
        signal, gap, sample, gamma_per_p = echo_signal, gap_from_echo, sample_probability, 134.0
    else:
        signal, gap, sample, gamma_per_p = hadamard_signal, gap_from_levels, sample_pm1, 104.0
    p = NOISY_GAMMA / gamma_per_p
    nt = int(np.log(np.sqrt(SHOTS)) / NOISY_GAMMA / DT)
    dE = 2 * np.pi / (nt * DT)
    print(f"{readout}, noisy: p={p:.5f}, Gamma~{NOISY_GAMMA}, nt={nt}, dE={dE:.4f}", flush=True)

    base, exact, t0 = {}, {}, time.time()
    for h in NOISY_HS:
        base[h] = signal(N, KAPPA, h, nt, DT, p)
        exact[h] = exact_gap(KAPPA, h)
        print(f"  h={h} [{(time.time()-t0)/60:.1f}m]", flush=True)

    print(f"\n{'order':>6} {'threshold':>10} {'recovered':>11}   per-field error %")
    for order in ORDERS:
        for rel in THRESHOLDS:
            errs, ok = [], []
            for h in NOISY_HS:
                vals = [gap(sample(base[h], SHOTS, np.random.default_rng(s)), DT, order, rel)
                        for s in SEEDS]
                v = np.median(vals)
                errs.append(100 * abs(v - exact[h]) / exact[h] if np.isfinite(v) else np.nan)
                ok.append(np.isfinite(v) and abs(v - exact[h]) < 0.15 * exact[h] + 0.02)
            print(f"{order:>6} {rel:>10.2f} {np.mean(ok):10.0%}   " +
                  " ".join(f"{e:5.0f}" if np.isfinite(e) else "  nan" for e in errs))


if __name__ == "__main__":
    args = sys.argv[1:]
    readout = next((a for a in args if a in ("echo", "hadamard")), "echo")
    if "noisy" in args:
        scan_noisy(readout)
    else:
        scan_clean(readout)
