"""Classical post-processing for the spectral estimator.

ESPRIT recovers the frequencies of a sum of decaying complex exponentials from
the shift-invariance of the Hankel matrix. Energies come from arg(z)/-dt, which
is blind to the decay rate, so a uniform envelope cancels.

`order` is required, not defaulted. The right model order depends on the
signal-to-noise ratio of the record it is applied to -- 16 works on the
noiseless maps here, 24 under noise -- and defaulting it is how a noiseless
choice silently reached noisy data and cost a run.
"""

import numpy as np


def esprit(g, dt, order):
    """Frequencies of a sum of decaying complex exponentials, from the
    shift-invariance of the Hankel matrix. Energies are arg(z)/-dt, which is
    blind to the decay rate.

    ponytail: fixed model order. Selecting it from the Hankel singular-value
    spectrum was tried and was measurably worse -- it over-fits the sampling
    noise once decoherence shortens the record. Stabilising the large-Gamma
    regime needs real denoising (Cadzow / Fourier-denoised DMD), not order
    selection."""
    m = len(g) // 2
    H0 = np.array([g[i:i + m] for i in range(len(g) - m)])
    u, _, _ = np.linalg.svd(H0, full_matrices=False)
    U = u[:, :order]
    z = np.linalg.eigvals(np.linalg.pinv(U[:-1]) @ U[1:])
    return np.sort(np.angle(z) / -dt)


def _filtered(g, dt, order, rel):
    """ESPRIT frequencies, optionally keeping only the amplitudes that matter.

    A least-squares fit of the mode amplitudes separates real lines from the
    ones ESPRIT conjures out of sampling noise. `rel = 0` skips the fit.
    """
    e = esprit(g, dt, order)
    if rel > 0:
        t = dt * np.arange(len(g))
        a = np.abs(np.linalg.lstsq(np.exp(-1j * np.outer(t, e)), g, rcond=None)[0])
        e = np.sort(e[a > rel * a.max()])
    return e


def gap_from_echo(signal, dt, order, rel=0.0):
    """E1 - E0 from an echo record.

    The echo spectrum holds level *differences*, so the gap is the smallest
    positive frequency in it. DC is excluded by the 5e-3 floor.
    """
    e = _filtered(np.asarray(signal, dtype=complex), dt, order, rel)
    e = np.sort(e[e > 5e-3])
    return float(e[0]) if len(e) else np.nan


def gap_from_levels(signal, dt, order, rel=0.0):
    """E1 - E0 from a Hadamard-test record, which carries absolute energies,
    so the two lowest levels are identified by ordering rather than by size."""
    e = np.sort(_filtered(np.asarray(signal, dtype=complex), dt, order, rel))
    return float(e[1] - e[0]) if len(e) > 1 else np.nan
