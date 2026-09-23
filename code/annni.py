"""1D ANNNI model: ground states, order parameters, and the (kappa, h) phase diagram.

    H = -J sum_i Z_i Z_{i+1}  +  J*kappa sum_i Z_i Z_{i+2}  -  J*h sum_i X_i

Open chain, J = 1. kappa > 0 makes the next-nearest-neighbour coupling
antiferromagnetic, which frustrates the ferromagnetic nearest-neighbour term.

Three phases are expected:
    ferromagnetic   small kappa, small h   -> structure factor peaks at k = 0
    antiphase       kappa > 0.5, small h   -> period-4 order, peaks at k = pi/2
    paramagnetic    large h                -> <X> -> 1

Exact diagonalisation here is the ground truth. VQE and the noisy study are
benchmarked against it.
"""

import numpy as np
import pennylane as qml
from scipy.sparse.linalg import eigsh


def hamiltonian(n, kappa, h):
    """ANNNI Hamiltonian on an open chain of n spins."""
    coeffs, terms = [], []
    for i in range(n - 1):
        coeffs.append(-1.0)
        terms.append(qml.Z(i) @ qml.Z(i + 1))
    for i in range(n - 2):
        coeffs.append(kappa)
        terms.append(qml.Z(i) @ qml.Z(i + 2))
    for i in range(n):
        coeffs.append(-h)
        terms.append(qml.X(i))
    return qml.Hamiltonian(coeffs, terms)


def ground_state(n, kappa, h):
    """Lowest eigenpair. The state is real, so sign is arbitrary -- callers
    that compare states must take the absolute overlap."""
    mat = hamiltonian(n, kappa, h).sparse_matrix(wire_order=range(n)).real
    energy, vec = eigsh(mat, k=1, which="SA")
    return float(energy[0]), np.asarray(vec[:, 0], dtype=float)


def _spin_table(n):
    """(2**n, n) array of +-1 spin values, row index = computational basis state."""
    bits = ((np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)) & 1)
    return 1.0 - 2.0 * bits


def zz_correlators(state, n):
    """<Z_i Z_j> for every pair. Diagonal in the computational basis, so this is
    just a weighted outer product of the spin table."""
    s = _spin_table(n)
    p = state**2
    return s.T @ (p[:, None] * s)


def transverse_magnetisation(state, n):
    """(1/n) sum_i <X_i>. X_i flips bit i, so the expectation is an overlap of
    the state with its bit-flipped self."""
    idx = np.arange(2**n)
    total = sum(float(state @ state[idx ^ (1 << (n - 1 - i))]) for i in range(n))
    return total / n


def structure_factor(corr, k):
    """S(k) = (1/n) sum_ij cos(k(i-j)) <Z_i Z_j>, normalised to [0, 1]-ish by n."""
    n = corr.shape[0]
    i = np.arange(n)
    phase = np.cos(k * (i[:, None] - i[None, :]))
    return float((phase * corr).sum()) / n**2


def order_parameters(n, kappa, h):
    """Everything needed to label one point of the phase diagram."""
    energy, state = ground_state(n, kappa, h)
    corr = zz_correlators(state, n)
    return {
        "energy": energy,
        "state": state,
        "ferro": structure_factor(corr, 0.0),
        "antiphase": structure_factor(corr, np.pi / 2),
        "para": transverse_magnetisation(state, n),
    }


def demo():
    """Assert the three phases show up where the literature says they do."""
    n = 8

    ferro = order_parameters(n, kappa=0.0, h=0.0)
    assert ferro["ferro"] > 0.9, ferro["ferro"]
    assert ferro["antiphase"] < 0.1, ferro["antiphase"]

    para = order_parameters(n, kappa=0.0, h=3.0)
    assert para["para"] > 0.9, para["para"]
    assert para["ferro"] < 0.2, para["ferro"]

    anti = order_parameters(n, kappa=1.0, h=0.0)
    assert anti["antiphase"] > anti["ferro"], (anti["antiphase"], anti["ferro"])
    assert anti["antiphase"] > 0.3, anti["antiphase"]

    # The multiphase point kappa = 0.5 at h = 0 separates ferro from antiphase.
    assert order_parameters(n, 0.3, 0.0)["ferro"] > order_parameters(n, 0.7, 0.0)["ferro"]

    print("ok: ferro, paramagnetic and antiphase limits all reproduce")


if __name__ == "__main__":
    demo()


def ferro_para_boundary(kappa):
    """Ferromagnet-paramagnet line, valid for 0 < kappa < 1/2. It runs from
    h = 1 at kappa = 0, where the model is the transverse-field Ising chain,
    down to h = 0 at the multiphase point."""
    k = np.clip(kappa, 1e-6, 0.5)
    return (1.0 - k) / k * (1.0 - np.sqrt((1.0 - 3.0 * k + 4.0 * k**2) / (1.0 - k)))


def antiphase_boundary(kappa):
    "Antiphase-floating line, the usual h ~ 1.05 sqrt(kappa - 1/2) fit."
    return 1.05 * np.sqrt(np.clip(kappa - 0.5, 0.0, None))
