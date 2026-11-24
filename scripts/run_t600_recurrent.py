"""Run a recurrent Adding Problem baseline against a pinned upstream TCN checkout.

Example in Colab after cloning the upstream repository:
    python scripts/run_t600_recurrent.py --repo /content/TCN --model GRU --seed 1111

These are our own single-layer baselines, not the paper authors' recurrent code.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HIDDEN_SIZES = {"GRU": 150, "LSTM": 130, "RNN": 262}


class AddingRecurrent(nn.Module):
    def __init__(self, kind: str, hidden_size: int):
        super().__init__()
        layer = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}[kind]
        self.kind = kind
        self.recurrent = layer(input_size=2, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, channel, time) -> (batch, time, channel)
        _, state = self.recurrent(x)
        hidden = state[0] if self.kind == "LSTM" else state
        return self.output(hidden[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/content/TCN"))
    parser.add_argument("--model", choices=HIDDEN_SIZES, required=True)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--outdir", type=Path, default=Path("/content/TCN/reproduction_logs"))
    args = parser.parse_args()

    if not (args.repo / "TCN" / "adding_problem" / "utils.py").is_file():
        parser.error(f"Upstream TCN checkout not found: {args.repo}")
    sys.path.insert(0, str(args.repo))
    from TCN.adding_problem.utils import data_generator

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train = data_generator(50000, 600)
    x_test, y_test = data_generator(1000, 600)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    model = AddingRecurrent(args.model, HIDDEN_SIZES[args.model]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    n_params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        constant_one_mse = F.mse_loss(torch.ones_like(y_test), y_test).item()

    args.outdir.mkdir(parents=True, exist_ok=True)
    stem = f"adding_{args.model.lower()}_T600_seed{args.seed}_{args.epochs}epochs"
    results_path = args.outdir / f"{stem}_results.json"
    checkpoint_path = args.outdir / f"{stem}_checkpoint.pt"
    metrics = []
    print(f"model={args.model} seed={args.seed} params={n_params} device={device}", flush=True)
    print(f"constant_one_test_mse={constant_one_mse:.9f}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for start in range(0, len(x_train), 32):
            end = min(start + 32, len(x_train))
            optimizer.zero_grad()
            loss = F.mse_loss(model(x_train[start:end]), y_train[start:end])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_mse = F.mse_loss(model(x_test), y_test).item()
        metrics.append({"epoch": epoch, "test_mse": test_mse})
        print(f"epoch={epoch} test_mse={test_mse:.9f}", flush=True)

        result = {
            "task": "Adding Problem",
            "model": f"single-layer PyTorch {args.model} with linear output",
            "upstream_commit": "2f8c2b817050206397458dfd1f5a25ce8a32fe65",
            "sequence_length": 600,
            "seed": args.seed,
            "hidden_size": HIDDEN_SIZES[args.model],
            "parameter_count": n_params,
            "train_samples": 50000,
            "test_samples": 1000,
            "batch_size": 32,
            "optimizer": "Adam",
            "learning_rate": 0.004,
            "constant_one_test_mse": constant_one_mse,
            "test_mse_by_epoch": metrics,
        }
        results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "completed_epochs": epoch,
            },
            checkpoint_path,
        )

    print(f"results={results_path}\ncheckpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
