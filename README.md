# Decoherence-limited spectroscopy of the 1D ANNNI phase diagram

Q-SITE Hackathon 2026 — Open Challenge, Scientific track.

Jeongbin Jo · Department of Physics, and School of Mathematics and Computing
(Computational Science and Engineering), Yonsei University · jeongbin033@yonsei.ac.kr

**Notebook:** [`annni_circuits.ipynb`](annni_circuits.ipynb) — the circuit study end to end, self-contained

This repository holds the code. The write-up is submitted separately; the
numbers quoted below are from it, and every one of them is reproducible from
what is here.

---

## What this is

The challenge asks for two things: map the phase diagram of the 1D ANNNI model in
the (κ, h) plane with PennyLane, and study how depolarizing noise distorts the
phase boundaries.

The second half turns out to be the interesting one, because in the two most
natural readings of it the answer is that noise does not distort the boundaries
at all.

- A depolarizing channel applied to the exact ground state rescales every
  two-point correlator by one common factor — confirmed to six digits against
  (1−4p/3)². Every order parameter shrinks by the same amount, so their relative
  ordering, and the boundary with it, is untouched.
- Applied as a decay envelope on the time signal it leaves the estimated
  energies unbiased, because the estimator reads the *phase* of its spectral
  roots and the envelope acts only on their modulus.

What noise actually destroys is the length of record that carries information.
Decoherence and finite sampling together cap the usable evolution time at
`T_max = ln√M / Γ`, and so the spectral resolution at `δE ≈ 2π/T_max`. A phase
boundary can be located only where the relevant gap clears δE.

Over five decoherence rates that criterion is **exact in one direction**: of
sixteen field values whose gap lies below δE, not one was ever recovered. Above
δE recovery runs at 92–93% while the noise is weak and degrades as the record
shortens.

## Method

Loschmidt echo readout — prepare, evolve, undo the preparation, count the
all-zeros outcome — on fourth-order Suzuki evolution. No ancilla and no
controlled gates. Depolarizing noise is a channel after every two-qubit gate;
readout is binomial on the return probability. Gaps come from ESPRIT, which is
classical post-processing, exactly as it would be on hardware.

| | |
|---|---|
| system size | N = 6 qubits |
| evolution | fourth-order Suzuki, δt = 0.30 |
| reference state | \|+⟩⊗ᴺ + ferromagnet + period-4 + two domain walls |
| estimator | ESPRIT, order 16 noiseless / 24 under noise |
| shots | 10⁴ per time point |

Exact diagonalisation appears only as a labelled benchmark.

## Running it

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

    python code/annni.py             # self-check: the three phase limits
    jupyter lab annni_circuits.ipynb # the circuit study, ~25 min end to end

To regenerate the figures:

    python code/sweep.py             # exact-diagonalisation phase diagram
    python code/run_res_echo.py      # the resolvability figure, ~50 min
    python code/run_map_echo.py      # the circuit phase map, ~18 min

Each writes its data to `results/` and its figure to `manuscript/`, creating
both if needed. Neither directory is tracked here, so a fresh clone recomputes
rather than redrawing; once a run has completed, `--plot-only` on either script
redraws from the cached `.npz` without recomputing.

`code/tune.py` reproduces the scans behind the tables; it takes the readout and
the noise condition as arguments, because that choice is one 2×2:

    python code/tune.py echo clean       python code/tune.py hadamard clean
    python code/tune.py echo noisy       python code/tune.py hadamard noisy

## Layout

    code/         everything executable
    results/      created by a run: the .npz it wrote   (not tracked)
    logs/         created by a run: its stdout          (not tracked)
    manuscript/   figure output                          (not tracked)

Inside `code/`: `annni.py` (model, exact diagonalisation, analytic boundaries),
`circuits.py` (Trotter steps, reference state, both readouts, both shot
samplers), `spectral.py` (ESPRIT and the two rules for reading a gap out of it),
`figstyle.py`, then one script per figure and `tune.py` for the tables.

## Why not Krylov

Real-time Krylov consumes the same record, and its overlap and Hamiltonian
matrices are Toeplitz in the time index, so it costs no more to measure. It is
ruled out here for two reasons, neither of them accuracy:

- it cannot run on the echo at all — the overlap matrix needs the complex
  amplitude and |g|² has lost the phase;
- it needs a second observable, ⟨ψ₀|H U(t)|ψ₀⟩ = i dg/dt, which is one circuit
  per Pauli term (fifteen at N = 6) or a derivative of a noisy signal.

Where it can run it is perfectly sound. `python code/krylov_map.py` maps its
error over subspace dimension and Trotter step: given both matrices from the
same propagator it converges to a plateau set by the Trotter bias, and given
both from exact evolution it reaches machine precision. At equal model
dimension ESPRIT is still ahead — 2.6×10⁻⁴ against 1.3×10⁻² at δt = 0.30,
D = 24 — but Krylov catches up by D ≈ 48.

**The trap we fell into** is worth repeating. Building S from the Trotterised
record and H from exact evolution — two propagators, one matrix each — diverges
by five orders of magnitude. It looks fine at small D and blows up only as the
subspace is enlarged, which is exactly what one does to improve a variational
method. We first read that as "Krylov is fragile to Trotter error". It is not.
We had mixed our matrix elements.

## Three things that had to be right

Each of these was wrong at first, and each produced a plot that looked plausible.
They are in the paper because they are the substance of making the measurement
work.

**The reference state must break two symmetries.** H commutes with the spin flip
∏ᵢXᵢ *and* with reflection about the chain centre. A reference even under both
has exactly zero overlap with odd eigenstates — measured at 6×10⁻²⁶ — and E₁ is
reflection-odd over much of the diagram, so the gap is invisible at any
resolution. Adding two domain-wall components fixes it, and lifts recovery above
δE from 73% to 92%.

**The model order must be chosen on data matched to its use.** Order 32 is
optimal on noiseless records and recovers 12% of noisy fields. The order that
works under noise is 24. Picking it on clean data and applying it to noisy data
cost a 90-minute run.

**Γ must be measured, not extrapolated.** Γ/p drifts from 72 to 47 across the
range studied, so inverting a linear fit puts the record length, and therefore
δE, at the wrong value.

## Limits, stated plainly

The scaling law |h*−h_c| ~ Γ^(1/zν) follows from the criterion but is **not**
tested here: recovery above δE is already down to 22% at Γ = 0.14, so the usable
window is roughly Γ ≲ 0.05, too narrow to fit a power law through. Widening it
is a signal-processing problem — denoising the Hankel subspace — not a circuit
one.

System size is bounded from the other side. The echo line for E₁−E₀ carries
weight |c₀|²|c₁|², the *product* of two occupations, and at 10⁴ shots it crosses
the sampling floor between N = 8 and N = 10. An earlier map at N = 10 recovered
47% of the sites above δE where N = 6 recovers 87%; no amount of estimator
tuning gets that back.
