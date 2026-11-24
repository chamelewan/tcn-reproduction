
"""Copy Memory recurrent baselines (GRU/LSTM) using the original TCN data generator."""
import argparse, json, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default="/content/TCN")
parser.add_argument("--model", choices=["GRU", "LSTM", "RNN"], required=True)
parser.add_argument("--hidden", type=int, required=True)
parser.add_argument("--blank_len", type=int, default=1000)
parser.add_argument("--seq_len", type=int, default=10)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--clip", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=1111)
parser.add_argument("--out_prefix", required=True)
args = parser.parse_args()
print(args, flush=True)

sys.path.insert(0, args.repo)
from TCN.copy_memory.utils import data_generator

# TCN 실행과 같은 순서: torch 시드 → NumPy 시드로 학습·테스트 데이터 생성 → 모델 생성
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)
T, seq_len, n_classes = args.blank_len, args.seq_len, 10
train_x, train_y = data_generator(T, seq_len, 10000)
test_x, test_y = data_generator(T, seq_len, 1000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_x, train_y, test_x, test_y = (t.to(device) for t in (train_x, train_y, test_x, test_y))


class RecurrentBaseline(nn.Module):
    def __init__(self, cell, hidden):
        super().__init__()
        self.rnn = getattr(nn, cell)(1, hidden, batch_first=True)
        self.linear = nn.Linear(hidden, n_classes)

    def forward(self, x):
        out, _ = self.rnn(x.unsqueeze(-1))
        return self.linear(out)


model = RecurrentBaseline(args.model, args.hidden).to(device)
params = sum(p.numel() for p in model.parameters())
print("MODEL_PARAMETERS:", params, flush=True)
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)


def evaluate():
    model.eval()
    with torch.no_grad():
        logits = model(test_x)
        loss = F.cross_entropy(logits.reshape(-1, n_classes), test_y.reshape(-1)).item()
        pred = logits.argmax(-1)
        overall = (pred == test_y).float().mean().item() * 100
        last10 = (pred[:, -seq_len:] == test_y[:, -seq_len:]).float().mean().item() * 100
        last10_ce = F.cross_entropy(logits[:, -seq_len:].reshape(-1, n_classes),
                                    test_y[:, -seq_len:].reshape(-1)).item()
    return loss, overall, last10, last10_ce


history = []
started = time.perf_counter()
for ep in range(1, args.epochs + 1):
    model.train()
    ep_start = time.perf_counter()
    for start in range(0, train_x.size(0), args.batch_size):
        x = train_x[start:start + args.batch_size]
        y = train_y[start:start + args.batch_size]
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).reshape(-1, n_classes), y.reshape(-1))
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # backward 뒤 적용
        optimizer.step()
    loss, overall, last10, last10_ce = evaluate()
    history.append({"epoch": ep, "test_loss": loss, "overall_accuracy": overall,
                    "last10_accuracy": last10, "last10_cross_entropy": last10_ce,
                    "epoch_seconds": time.perf_counter() - ep_start})
    print(f"\nTest set: Average loss: {loss:.8f}  |  Accuracy: {overall:.4f}", flush=True)
    print(f"| Epoch {ep:3d} | last10 accuracy {last10:.4f} | last10 CE {last10_ce:.8f} "
          f"| {time.perf_counter() - ep_start:.1f}s", flush=True)

final = history[-1]
print("FINAL_MODEL_PARAMETERS:", params)
print(f"FINAL_OVERALL_ACCURACY: {final['overall_accuracy']:.4f}%")
print(f"FINAL_LAST10_ACCURACY: {final['last10_accuracy']:.4f}%")
print(f"FINAL_LAST10_CROSS_ENTROPY: {final['last10_cross_entropy']:.8f}")
print(f"TRAIN_SECONDS: {time.perf_counter() - started:.1f}")
json.dump({"args": vars(args), "parameters": params, "history": history},
          open(args.out_prefix + "_results.json", "w"), indent=2)
torch.save({"model_state_dict": model.state_dict(), "args": vars(args),
            "upstream_commit": "2f8c2b817050206397458dfd1f5a25ce8a32fe65"},
           args.out_prefix + "_checkpoint.pt")
