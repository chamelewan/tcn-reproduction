"""Compare TCN and recurrent baselines on Copy Memory from observed logs and result JSON files."""

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tcn-log", type=Path, required=True)
    parser.add_argument("--baseline", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    text = args.tcn_log.read_text(encoding="utf-8")
    tcn_losses = [float(v) for v in re.findall(r"Test set: Average loss: ([0-9.]+)", text)]
    tcn_last10 = float(re.search(r"FINAL_LAST10_ACCURACY: ([0-9.]+)%", text).group(1))
    tcn_params = int(re.search(r"FINAL_MODEL_PARAMETERS: (\d+)", text).group(1))
    blank_len = int(re.search(r"blank_len=(\d+)", text).group(1))
    seq_len = int(re.search(r"seq_len=(\d+)", text).group(1))
    memoryless = seq_len * math.log(8) / (blank_len + 2 * seq_len)

    colors = {"TCN": "#1166aa", "GRU": "#c46500", "LSTM": "#2e8b57", "RNN": "#888888"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(range(1, len(tcn_losses) + 1), tcn_losses, color=colors["TCN"], lw=2,
                 label=f"TCN ({tcn_params:,} params)")
    for name, path in args.baseline:
        data = json.loads(Path(path).read_text())
        hist = data["history"]
        epochs = [h["epoch"] for h in hist]
        axes[0].plot(epochs, [h["test_loss"] for h in hist], color=colors.get(name), lw=2,
                     label=f"{name} ({data['parameters']:,} params)")
        axes[1].plot(epochs, [h["last10_accuracy"] for h in hist], color=colors.get(name), lw=2,
                     label=f"{name}: final {hist[-1]['last10_accuracy']:.2f}%")
    axes[0].axhline(memoryless, color="gray", ls="--", label=f"Memoryless guess ({memoryless:.4f})")
    axes[0].set(title=f"Copy Memory T={blank_len}: test loss", xlabel="Epoch",
                ylabel="Cross-entropy (log scale)", yscale="log")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="lower left", fontsize=9)

    n_epochs = len(tcn_losses)
    axes[1].scatter([n_epochs], [tcn_last10], color=colors["TCN"], s=60, zorder=3,
                    label=f"TCN: final {tcn_last10:.2f}% (measured at end only)")
    axes[1].axhline(100 / 8, color="gray", ls="--", label="Random guess among 8 digits (12.5%)")
    axes[1].set(title="Last-10 digit accuracy", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(-3, 105))
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="center right", fontsize=9)

    seed = re.search(r"seed=(\d+)", text).group(1)
    fig.suptitle(f"Copy Memory T={blank_len}, seed {seed}, {n_epochs} epochs: TCN vs self-built recurrent baselines", fontsize=13)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
