"""The circuit pieces every run shares.

Only what is provably identical across the scripts lives here. The readout
itself does not: the echo measures a projector and returns |g|^2, the Hadamard
test measures an ancilla and returns g, their shot noise is binomial on
different quantities, and the gap is read out of their spectra by different
rules. Those stay with the script that uses them, because they are genuinely
different measurements rather than copies of one.
"""

import numpy as np
import pennylane as qml

SUZUKI = 1.0 / (4.0 - 4.0 ** (1 / 3))


def s2(n, kappa, h, dt):
    """One second-order Strang step of exp(-i H dt), unitary only.

    For a term c*P the evolution is exp(-i c dt P): IsingZZ(2 c dt) for P = ZZ
    and RX(2 c dt) for P = X. ANNNI has c = -1 on nearest bonds, +kappa on
    next-nearest and -h on the field, which is split into two half layers.
    """
    for i in range(n):
        qml.RX(-h * dt, wires=i)
    for i in range(n - 1):
        qml.IsingZZ(-2.0 * dt, wires=[i, i + 1])
    for i in range(n - 2):
        qml.IsingZZ(2.0 * kappa * dt, wires=[i, i + 2])
    for i in range(n):
        qml.RX(-h * dt, wires=i)


def s4(n, kappa, h, dt):
    """Fourth-order Suzuki: five second-order substeps, so five times the gates."""
    for f in (SUZUKI, SUZUKI, 1 - 4 * SUZUKI, SUZUKI, SUZUKI):
        s2(n, kappa, h, f * dt)


def ref5(n):
    """|+>^n plus the ferromagnet, the period-four state and two domain walls.

    The domain walls are load-bearing. H commutes with the spin flip and with
    reflection about the chain centre; without them the reference is even under
    both, and E1 is reflection-odd over much of the diagram, so the gap is
    invisible at any resolution.
    """
    v = np.ones(2**n, dtype=complex) / 2 ** (n / 2)
    for pattern in ("0" * n, ("0011" * n)[:n], ("000111" * n)[:n], ("001111" * n)[:n]):
        v[int(pattern, 2)] += 1.0
    return v / np.linalg.norm(v)


def noisy_s4(n, kappa, h, dt, p=0.0):
    """Fourth-order step with depolarizing after each second-order substep.

    At p = 0 this is exactly s4. The channel is charged per substep rather than
    per Trotter step so that the fourth-order product pays for its five times
    the gates.
    """
    for f in (SUZUKI, SUZUKI, 1 - 4 * SUZUKI, SUZUKI, SUZUKI):
        s2(n, kappa, h, f * dt)
        if p > 0:
            for i in range(n):
                qml.DepolarizingChannel(p, wires=i)


def controlled_noisy_s4(n, kappa, h, dt, p=0.0):
    """The same step controlled on wire n, for the Hadamard test.

    The noise sits outside the control: a channel is not a unitary and cannot be
    controlled, and physically the register decoheres whatever the ancilla is
    doing. The ancilla gets a channel of its own because it is entangled with
    the register through every controlled gate.
    """
    for f in (SUZUKI, SUZUKI, 1 - 4 * SUZUKI, SUZUKI, SUZUKI):
        qml.ctrl(s2, control=n)(n, kappa, h, f * dt)
        if p > 0:
            for i in range(n):
                qml.DepolarizingChannel(p, wires=i)
            qml.DepolarizingChannel(p, wires=n)


def _snapshots(circ, dev):
    sn = qml.snapshots(qml.QNode(circ, dev))()
    return np.array([float(v) for k, v in sn.items() if k != "execution_results"])


def echo_signal(n, kappa, h, nt, dt, p=0.0, psi0=None):
    """Return probability after each step: |<psi_0|U(t)|psi_0>|^2, no ancilla.

    On hardware this is prepare, evolve, undo the preparation, count all-zeros;
    the projector measures exactly that. Snapshots read it after every step in
    one run, which is O(nt) where re-running per time point would be O(nt^2).
    """
    psi0 = ref5(n) if psi0 is None else psi0
    dev = qml.device("default.mixed" if p > 0 else "default.qubit", wires=n)

    def circ():
        qml.StatePrep(psi0, wires=range(n))
        ob = qml.Projector(psi0, wires=range(n))
        qml.Snapshot(measurement=qml.expval(ob))
        for _ in range(nt - 1):
            noisy_s4(n, kappa, h, dt, p)
            qml.Snapshot(measurement=qml.expval(ob))
        return qml.expval(ob)

    return _snapshots(circ, dev)


def hadamard_signal(n, kappa, h, nt, dt, p=0.0, psi0=None):
    """g(t) = <psi_0|U(t)|psi_0> from an ancilla in |+> controlling U.

    Its reduced state is [[1, g*], [g, 1]]/2, so <X> = Re g and <Y> = Im g.
    """
    psi0 = ref5(n) if psi0 is None else psi0
    dev = qml.device("default.mixed" if p > 0 else "default.qubit", wires=n + 1)
    out = []
    for part in ("re", "im"):
        def circ(part=part):
            qml.StatePrep(psi0, wires=range(n))
            qml.Hadamard(wires=n)
            ob = qml.X(n) if part == "re" else qml.Y(n)
            qml.Snapshot(measurement=qml.expval(ob))
            for _ in range(nt - 1):
                controlled_noisy_s4(n, kappa, h, dt, p)
                qml.Snapshot(measurement=qml.expval(ob))
            return qml.expval(ob)
        out.append(_snapshots(circ, dev))
    return out[0] + 1j * out[1]


def sample_probability(prob, shots, rng):
    "Readout for the echo, which returns a probability."
    return rng.binomial(shots, np.clip(prob, 0, 1)) / shots


def sample_pm1(g, shots, rng):
    "Readout for the Hadamard test, whose observables have eigenvalues +-1."
    def draw(v):
        return 2 * rng.binomial(shots, np.clip((1 + v) / 2, 0, 1)) / shots - 1
    return draw(g.real) + 1j * draw(g.imag)
