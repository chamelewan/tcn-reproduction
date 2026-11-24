"""Plot the observed Copy Memory test loss and overall accuracy from a raw log."""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8")
    records = re.findall(
        r"Test set: Average loss: ([0-9.]+)  \|  Accuracy: ([0-9.]+)", text
    )
    if not records:
        parser.error("No evaluation records found in the log")

    epochs = range(1, len(records) + 1)
    losses = [float(loss) for loss, _ in records]
    accuracies = [float(accuracy) for _, accuracy in records]
    tail_match = re.search(r"FINAL_LAST10_ACCURACY: ([0-9.]+)%", text)
    tail_accuracy = float(tail_match.group(1)) if tail_match else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, losses, color="#1166aa", linewidth=2)
    axes[0].scatter([len(losses)], [losses[-1]], color="#1166aa", zorder=3)
    axes[0].set(title="Copy Memory T=1000: test loss", xlabel="Epoch", ylabel="Cross-entropy (log scale)")
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].annotate(f"Final: {losses[-1]:.6f}", (len(losses), losses[-1]),
                     xytext=(-8, 16), textcoords="offset points", ha="right")

    zero_baseline_accuracy = 100 * 1010 / 1020
    axes[1].plot(epochs, accuracies, color="#c46500", linewidth=2, label="Overall accuracy")
    axes[1].axhline(zero_baseline_accuracy, color="gray", linestyle="--",
                    label="All-zero accuracy (99.02%)")
    axes[1].set(title="Overall accuracy needs context", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].set_ylim(98.9, 100.05)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="lower right")
    if tail_accuracy is not None:
        axes[1].text(0.03, 0.88, f"Final last-10 accuracy: {tail_accuracy:.2f}%",
                     transform=axes[1].transAxes, fontsize=10,
                     bbox={"facecolor": "white", "edgecolor": "#bbbbbb"})

    fig.suptitle("TCN Copy Memory reproduction, seed 1111, 50 epochs", fontsize=14)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
