"""Slides for the submission video, built from the paper's own figures."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from annni import hamiltonian, ferro_para_boundary, antiphase_boundary  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "video" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

W, H, DPI = 12.8, 7.2, 150          # 1920x1080
BG, FG, DIM, HI = "#0d1117", "#e6edf3", "#8b949e", "#58a6ff"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": DIM, "ytick.color": DIM})


def blank():
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
    return fig


def save(fig, n):
    fig.savefig(OUT / f"{n:02d}.png", facecolor=BG, dpi=DPI)
    plt.close(fig)
    print(f"  slide {n:02d}")


def text_slide(n, title, lines, sub=None, highlight=()):
    fig = blank()
    fig.text(0.08, 0.80, title, fontsize=40, weight="bold", va="top")
    if sub:
        fig.text(0.08, 0.705, sub, fontsize=20, color=DIM, va="top")
    y = 0.58
    for i, line in enumerate(lines):
        col = HI if i in highlight else FG
        fig.text(0.08, y, line, fontsize=26, color=col, va="top")
        y -= 0.115
    save(fig, n)


def figure_slide(n, title, pdf, caption, crop=None):
    "Embed a paper figure by rasterising the PDF page."
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pdftoppm", "-png", "-r", "260", str(ROOT / "manuscript" / pdf),
                        f"{td}/p"], check=True)
        img = plt.imread(sorted(Path(td).glob("p*.png"))[0])
    fig = blank()
    fig.text(0.06, 0.93, title, fontsize=32, weight="bold", va="top")
    ax = fig.add_axes([0.06, 0.16, 0.88, 0.70])
    ax.imshow(img); ax.axis("off")
    fig.text(0.06, 0.09, caption, fontsize=19, color=DIM, va="top")
    save(fig, n)


text_slide(1, "Decoherence sets a resolution limit,\nnot a bias",
           ["Jeongbin Jo — Yonsei University",
            "Q-SITE Hackathon 2026 · Open Challenge, Scientific track"],
           sub="Spectroscopy of the 1D ANNNI phase diagram on a quantum circuit")

text_slide(2, "The question",
           ["Map the ANNNI phase diagram in the $(\\kappa, h)$ plane with PennyLane.",
            "Then: how does depolarizing noise distort the phase boundaries?",
            "",
            "The premise deserves checking before it is used."],
           highlight=(3,))

text_slide(3, "Two obvious answers, both wrong",
           ["A channel on the exact ground state",
            "    every correlator rescales by one factor — verified to six digits",
            "",
            "A decay envelope on the time signal",
            "    energies stay unbiased — the estimator reads phase, not modulus"],
           sub="Neither moves the boundary at all")

text_slide(4, "What survives is resolution",
           ["Decoherence and finite sampling truncate the usable record:",
            "",
            "        $T_{\\max} = \\ln\\sqrt{M}\\,/\\,\\Gamma$        $\\delta E \\simeq 2\\pi/T_{\\max}$",
            "",
            "A boundary is locatable only where the gap clears $\\delta E$."],
           highlight=(4,))

figure_slide(5, "The criterion holds — in one direction exactly",
             "resolvability_echo.pdf",
             "Five decoherence rates, fifteen fields. Below $\\delta E$: 16 attempts, "
             "0 recovered. Above: 93% at weak noise.")

figure_slide(6, "Phase map from the circuit",
             "phase_map_echo.pdf",
             "Loschmidt echo, no ancilla, $N=6$, $21\\times21$. "
             "308/441 sites to 5%; 87% of those above $\\delta E$.")

text_slide(7, "Three things that had to be right",
           ["Reference state must break spin-flip AND reflection symmetry",
            "    otherwise the tracked level is invisible — overlap $6\\times10^{-26}$",
            "Model order must be chosen under the noise it will meet",
            "    the noiseless optimum recovers 12% of noisy fields",
            "$\\Gamma$ must be measured, not extrapolated — $\\Gamma/p$ drifts 72 → 47"],
           sub="Each was wrong at first, and each produced a plausible-looking wrong plot")

text_slide(8, "What we do not claim",
           ["The scaling law  $|h^*-h_c| \\sim \\Gamma^{1/z\\nu}$  follows from the criterion",
            "but is not tested: recovery is 22% already at $\\Gamma = 0.14$,",
            "so the usable window $\\Gamma \\lesssim 0.05$ is too narrow to fit through.",
            "",
            "github.com/dolf3131/ANNNI-research"],
           highlight=(4,))

print(f"\nwrote {len(list(OUT.glob('*.png')))} slides to {OUT}")
