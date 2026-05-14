import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import pennylane as qp
from dataclasses import dataclass
from typing import Callable


# ── data container ─────────────────────────────────────────────────────────────

@dataclass
class NoiseResult:
    noise_prob: float
    fidelity: float
    max_err: float
    avg_err: float


# ── exact reference ────────────────────────────────────────────────────────────

def exact_dft_matrix(n_qubits: int) -> np.ndarray:
    N = 2 ** n_qubits
    omega = np.exp(2j * np.pi / N)
    return np.array([[omega ** (j * k) for k in range(N)]
                     for j in range(N)]) / np.sqrt(N)


def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.dot(np.conj(a), b)) ** 2)


def ideal_qft_output(n_qubits: int, input_state: np.ndarray) -> np.ndarray:
    """Exact (noiseless) QFT via matrix multiply."""
    return exact_dft_matrix(n_qubits) @ input_state


# ── noisy circuit builders ─────────────────────────────────────────────────────

def qft_with_depolarizing(n_qubits: int, input_state: np.ndarray,
                          p: float) -> np.ndarray:
    dev = qp.device("default.mixed", wires=n_qubits)

    @qp.qnode(dev)
    def circuit():
        qp.StatePrep(input_state, wires=range(n_qubits))
        for wire in range(n_qubits):
            qp.Hadamard(wires=wire)
            qp.DepolarizingChannel(p, wires=wire)
            for ctrl in range(wire + 1, n_qubits):
                qp.ControlledPhaseShift(np.pi / 2 ** (ctrl - wire),
                                         wires=[ctrl, wire])
                qp.DepolarizingChannel(p, wires=ctrl)
                qp.DepolarizingChannel(p, wires=wire)
        for i in range(n_qubits // 2):
            j = n_qubits - 1 - i
            qp.SWAP(wires=[i, j])
            qp.DepolarizingChannel(p, wires=i)
            qp.DepolarizingChannel(p, wires=j)
        return qp.state()

    return np.array(circuit())


def qft_with_bitflip(n_qubits: int, input_state: np.ndarray,
                     p: float) -> np.ndarray:
    dev = qp.device("default.mixed", wires=n_qubits)

    @qp.qnode(dev)
    def circuit():
        qp.StatePrep(input_state, wires=range(n_qubits))
        for wire in range(n_qubits):
            qp.Hadamard(wires=wire)
            qp.BitFlip(p, wires=wire)
            for ctrl in range(wire + 1, n_qubits):
                qp.ControlledPhaseShift(np.pi / 2 ** (ctrl - wire),
                                         wires=[ctrl, wire])
                qp.BitFlip(p, wires=ctrl)
                qp.BitFlip(p, wires=wire)
        for i in range(n_qubits // 2):
            j = n_qubits - 1 - i
            qp.SWAP(wires=[i, j])
            qp.BitFlip(p, wires=i)
            qp.BitFlip(p, wires=j)
        return qp.state()

    return np.array(circuit())


def qft_with_phaseflip(n_qubits: int, input_state: np.ndarray,
                       p: float) -> np.ndarray:
    dev = qp.device("default.mixed", wires=n_qubits)

    @qp.qnode(dev)
    def circuit():
        qp.StatePrep(input_state, wires=range(n_qubits))
        for wire in range(n_qubits):
            qp.Hadamard(wires=wire)
            qp.PhaseFlip(p, wires=wire)
            for ctrl in range(wire + 1, n_qubits):
                qp.ControlledPhaseShift(np.pi / 2 ** (ctrl - wire),
                                         wires=[ctrl, wire])
                qp.PhaseFlip(p, wires=ctrl)
                qp.PhaseFlip(p, wires=wire)
        for i in range(n_qubits // 2):
            j = n_qubits - 1 - i
            qp.SWAP(wires=[i, j])
            qp.PhaseFlip(p, wires=i)
            qp.PhaseFlip(p, wires=j)
        return qp.state()

    return np.array(circuit())


def qft_with_amplitude_damping(n_qubits: int, input_state: np.ndarray,
                                gamma: float) -> np.ndarray:
    dev = qp.device("default.mixed", wires=n_qubits)

    @qp.qnode(dev)
    def circuit():
        qp.StatePrep(input_state, wires=range(n_qubits))
        for wire in range(n_qubits):
            qp.Hadamard(wires=wire)
            qp.AmplitudeDamping(gamma, wires=wire)
            for ctrl in range(wire + 1, n_qubits):
                qp.ControlledPhaseShift(np.pi / 2 ** (ctrl - wire),
                                         wires=[ctrl, wire])
                qp.AmplitudeDamping(gamma, wires=ctrl)
                qp.AmplitudeDamping(gamma, wires=wire)
        for i in range(n_qubits // 2):
            j = n_qubits - 1 - i
            qp.SWAP(wires=[i, j])
            qp.AmplitudeDamping(gamma, wires=i)
            qp.AmplitudeDamping(gamma, wires=j)
        return qp.state()

    return np.array(circuit())


def qft_combined_noise(n_qubits: int, input_state: np.ndarray,
                       p_dep: float, gamma: float) -> np.ndarray:
    dev = qp.device("default.mixed", wires=n_qubits)

    @qp.qnode(dev)
    def circuit():
        qp.StatePrep(input_state, wires=range(n_qubits))
        for wire in range(n_qubits):
            qp.Hadamard(wires=wire)
            qp.DepolarizingChannel(p_dep, wires=wire)
            qp.AmplitudeDamping(gamma, wires=wire)
            for ctrl in range(wire + 1, n_qubits):
                qp.ControlledPhaseShift(np.pi / 2 ** (ctrl - wire),
                                         wires=[ctrl, wire])
                for w in [ctrl, wire]:
                    qp.DepolarizingChannel(p_dep, wires=w)
                    qp.AmplitudeDamping(gamma, wires=w)
        for i in range(n_qubits // 2):
            j = n_qubits - 1 - i
            qp.SWAP(wires=[i, j])
            for w in [i, j]:
                qp.DepolarizingChannel(p_dep, wires=w)
                qp.AmplitudeDamping(gamma, wires=w)
        return qp.state()

    return np.array(circuit())


# ── fidelity from density matrix ───────────────────────────────────────────────

def dm_fidelity(ideal_state: np.ndarray, rho: np.ndarray) -> float:
    return float(np.real(ideal_state.conj() @ rho @ ideal_state))


def dm_to_state_max_err(ideal_state: np.ndarray, rho: np.ndarray) -> tuple:
    noisy_probs = np.real(np.diag(rho))
    ideal_probs = np.abs(ideal_state) ** 2
    err = np.abs(noisy_probs - ideal_probs)
    return float(err.max()), float(err.mean())


# ── sweep runner ───────────────────────────────────────────────────────────────

def sweep_noise(label: str,
                n_qubits: int,
                input_state: np.ndarray,
                ideal_out: np.ndarray,
                noise_fn: Callable,
                probs: list,
                prob_label: str = "p") -> list[NoiseResult]:
    print(f"\n{'='*65}")
    print(f"  {label}  (n_qubits={n_qubits})")
    print('='*65)
    print(f"  {prob_label:>8}  {'fidelity':>10}  {'max_err':>10}  {'avg_err':>10}")

    results = []
    for p in probs:
        rho = noise_fn(p)
        fid = dm_fidelity(ideal_out, rho)
        max_e, avg_e = dm_to_state_max_err(ideal_out, rho)
        results.append(NoiseResult(p, fid, max_e, avg_e))
        print(f"  {p:>8.4f}  {fid:>10.6f}  {max_e:>10.2e}  {avg_e:>10.2e}")

    return results


# ── threshold analysis ─────────────────────────────────────────────────────────

def find_fidelity_threshold(results: list[NoiseResult],
                             threshold: float = 0.99) -> float | None:
    for r in results:
        if r.fidelity < threshold:
            return r.noise_prob
    return None


def print_thresholds(all_results: dict[str, list[NoiseResult]],
                     thresholds: list[float] = [0.999, 0.99, 0.95, 0.90]) -> None:
    print(f"\n{'='*65}")
    print("  FIDELITY THRESHOLD SUMMARY")
    print('='*65)
    header = f"  {'noise_model':<22}" + "".join(f"  F>{t:.3f}" for t in thresholds)
    print(header)
    print('-'*65)
    for label, results in all_results.items():
        row = f"  {label:<22}"
        for t in thresholds:
            p = find_fidelity_threshold(results, t)
            row += f"  {f'p={p:.4f}' if p is not None else 'never':>8}"
        print(row)


# ── plotting ───────────────────────────────────────────────────────────────────

COLORS = {
    "depolarizing": "#e05c5c",
    "bit_flip":     "#e09a3a",
    "phase_flip":   "#4fa8d5",
    "amp_damping":  "#6abf69",
    "combined":     "#9b72cf",
}
DISPLAY_LABELS = {
    "depolarizing": "Depolarizing",
    "bit_flip":     "Bit-flip",
    "phase_flip":   "Phase-flip",
    "amp_damping":  "Amplitude damping (T1)",
    "combined":     "Combined (dep. + T1)",
}


def _arrays(results: list[NoiseResult]):
    ps  = np.array([r.noise_prob for r in results])
    fid = np.array([r.fidelity   for r in results])
    mxe = np.array([r.max_err    for r in results])
    ave = np.array([r.avg_err    for r in results])
    return ps, fid, mxe, ave


def plot_results(all_results: dict[str, list[NoiseResult]],
                 scaling_data: list[tuple[int, float]],
                 fixed_p: float,
                 save_path: str = "qft_noise_analysis.png") -> None:
    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor":   "#1a1d27",
        "axes.edgecolor":   "#3a3d4d",
        "axes.labelcolor":  "#c8ccd8",
        "axes.titlecolor":  "#e8ecf4",
        "xtick.color":      "#8a8fa8",
        "ytick.color":      "#8a8fa8",
        "grid.color":       "#2a2d3d",
        "grid.linestyle":   "--",
        "grid.alpha":       0.6,
        "text.color":       "#c8ccd8",
        "font.family":      "monospace",
        "legend.facecolor": "#1a1d27",
        "legend.edgecolor": "#3a3d4d",
        "legend.labelcolor":"#c8ccd8",
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("QFT Fidelity Under Noise  —  Key Lessons",
                 fontsize=14, fontweight="bold", color="#e8ecf4", y=0.98)

    # ── Panel 1: Fidelity vs noise strength ───────────────────────────────────
    ax1 = axes[0, 0]
    ax1.set_title("① Fidelity vs. Noise Strength", fontweight="bold")
    ax1.set_xlabel("Noise probability p")
    ax1.set_ylabel("Fidelity  F(|ψ_ideal⟩, ρ_noisy)")
    ax1.grid(True)

    for key, results in all_results.items():
        ps, fid, _, _ = _arrays(results)
        ax1.plot(ps, fid, "o-", color=COLORS[key],
                 label=DISPLAY_LABELS[key], linewidth=2, markersize=5)

    for thresh, ls in [(0.999, ":"), (0.99, "--"), (0.95, "-.")]:
        ax1.axhline(thresh, color="#ffffff", alpha=0.25, linestyle=ls,
                    linewidth=0.9, label=f"F = {thresh}")

    ax1.set_ylim(0.5, 1.02)
    ax1.legend(fontsize=7.5, loc="lower left")
    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.3f}"))

    # ── Panel 2: Max error on log scale ───────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.set_title("② Max Probability Error vs. Noise Strength", fontweight="bold")
    ax2.set_xlabel("Noise probability p")
    ax2.set_ylabel("Max |p_noisy − p_ideal|")
    ax2.grid(True)
    ax2.set_yscale("log")

    for key, results in all_results.items():
        ps, _, mxe, _ = _arrays(results)
        mask = ps > 0   # skip p=0: error ~1e-16 distorts log scale
        ax2.plot(ps[mask], mxe[mask], "s-", color=COLORS[key],
                 label=DISPLAY_LABELS[key], linewidth=2, markersize=5)

    ax2.legend(fontsize=7.5)

    combined_ps, _, combined_mxe, _ = _arrays(all_results["combined"])
    ax2.annotate(
        "Combined noise\ndegrades fastest",
        xy=(combined_ps[-1], combined_mxe[-1]),
        xytext=(0.06, 0.25),
        arrowprops=dict(arrowstyle="->", color=COLORS["combined"]),
        color=COLORS["combined"], fontsize=8,
    )

    # ── Panel 3: Filled fidelity collapse ─────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.set_title("③ Fidelity Collapse by Model\n"
                  "(shaded area = robustness to noise)", fontweight="bold")
    ax3.set_xlabel("Noise probability p")
    ax3.set_ylabel("Fidelity")
    ax3.grid(True)

    for key, results in all_results.items():
        ps, fid, _, _ = _arrays(results)
        ax3.fill_between(ps, fid, alpha=0.15, color=COLORS[key])
        ax3.plot(ps, fid, "-", color=COLORS[key],
                 label=DISPLAY_LABELS[key], linewidth=2)

    ax3.axhline(0.99, color="#ffffff", alpha=0.4, linestyle="--",
                linewidth=1, label="F = 0.99 threshold")
    ax3.set_ylim(0.5, 1.02)
    ax3.legend(fontsize=7.5)
    ax3.text(
        0.105, 0.54,
        "Phase-flip is\nparticularly damaging\nto QFT — it destroys\nphase coherence",
        color=COLORS["phase_flip"], fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1d27",
                  edgecolor=COLORS["phase_flip"], alpha=0.8),
    )
 
    # ── Panel 4: Fidelity vs qubit count ──────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_title(
        f"④ Fidelity Decay with Qubit Count\n(depolarizing, p = {fixed_p})",
        fontweight="bold")
    ax4.set_xlabel("Number of qubits")
    ax4.set_ylabel("Fidelity")
    ax4.grid(True)

    ns   = np.array([r[0] for r in scaling_data])
    fids = np.array([r[1] for r in scaling_data])

    ax4.plot(ns, fids, "o-", color=COLORS["depolarizing"],
             linewidth=2.5, markersize=9,
             markerfacecolor="#0f1117", markeredgewidth=2)

    for n, f in zip(ns, fids):
        ax4.annotate(f"{f:.4f}", xy=(n, f), xytext=(4, 6),
                     textcoords="offset points", fontsize=8, color="#e8ecf4")

    # Exponential trend fit
    coeffs = np.polyfit(ns, np.log(np.clip(fids, 1e-9, None)), 1)
    ns_fine = np.linspace(ns.min(), ns.max(), 100)
    ax4.plot(ns_fine, np.exp(np.polyval(coeffs, ns_fine)),
             "--", color="#ffffff", alpha=0.3, linewidth=1.5,
             label="Exponential trend")

    ax4.set_xticks(ns)
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=8)
    ax4.text(
        2.05, 0.12,
        "More qubits → more gates\n→ more noise accumulation\n"
        "→ exponential fidelity loss\n(QFT depth scales as O(n²))",
        color="#c8ccd8", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1d27",
                  edgecolor="#3a3d4d", alpha=0.9),
    )

    # Shared legend strip at bottom
    legend_handles = [
        Line2D([0], [0], color=COLORS[k], linewidth=2,
               label=DISPLAY_LABELS[k])
        for k in COLORS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, 0.01), framealpha=0.5)

    plt.tight_layout(rect=[0.0, 0.06, 1.0, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nSaved → {save_path}")
    plt.show()


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    N_QUBITS = 3
    N = 2 ** N_QUBITS

    # Use |1⟩ as test state (clean known transform)
    input_state = np.zeros(N, dtype=complex)
    input_state[1] = 1.0
    ideal_out = ideal_qft_output(N_QUBITS, input_state)

    PROBS = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20]
    PROBS_COMBINED = [p / 2 for p in PROBS]

    all_results = {}

    all_results["depolarizing"] = sweep_noise(
        "DEPOLARIZING NOISE", N_QUBITS, input_state, ideal_out,
        lambda p: qft_with_depolarizing(N_QUBITS, input_state, p),
        PROBS,
    )
    all_results["bit_flip"] = sweep_noise(
        "BIT-FLIP NOISE", N_QUBITS, input_state, ideal_out,
        lambda p: qft_with_bitflip(N_QUBITS, input_state, p),
        PROBS,
    )
    all_results["phase_flip"] = sweep_noise(
        "PHASE-FLIP NOISE", N_QUBITS, input_state, ideal_out,
        lambda p: qft_with_phaseflip(N_QUBITS, input_state, p),
        PROBS,
    )
    all_results["amp_damping"] = sweep_noise(
        "AMPLITUDE DAMPING (T1-like)", N_QUBITS, input_state, ideal_out,
        lambda g: qft_with_amplitude_damping(N_QUBITS, input_state, g),
        PROBS, prob_label="γ",
    )
    all_results["combined"] = sweep_noise(
        "COMBINED (depolarizing + T1)", N_QUBITS, input_state, ideal_out,
        lambda p: qft_combined_noise(N_QUBITS, input_state, p, p),
        PROBS_COMBINED, prob_label="p_each",
    )

    print_thresholds(all_results)

    FIXED_P = 0.01
    print(f"\n{'='*65}")
    print(f"  QUBIT SCALING AT p={FIXED_P} (depolarizing)")
    print('='*65)
    print(f"  {'n_qubits':>8}  {'N':>5}  {'fidelity':>10}  {'max_err':>10}")
    scaling_data = []
    for n in range(2, 6):
        Nn = 2 ** n
        s = np.zeros(Nn, dtype=complex); s[1] = 1.0
        ideal = ideal_qft_output(n, s)
        rho = qft_with_depolarizing(n, s, FIXED_P)
        fid = dm_fidelity(ideal, rho)
        max_e, _ = dm_to_state_max_err(ideal, rho)
        scaling_data.append((n, fid))
        print(f"  {n:>8}  {Nn:>5}  {fid:>10.6f}  {max_e:>10.2e}")

    plot_results(all_results, scaling_data, fixed_p=FIXED_P)
