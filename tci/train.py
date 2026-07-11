"""Config-driven training of the GNN troubled-cell detector.

Usage:
    python -m tci.train configs/gnn1d.yaml [--out runs/my-run]

Every ablation (architecture, features, data mode, mesh randomization, ...)
is a new YAML file, not new code.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

DEFAULTS = {
    "seed": 0,
    "data": {
        "mode": "exact",  # exact | numerical
        "n_samples": 2000,
        "N": 1,
        "k_range": [50, 150],
        "max_disc": 5,
        "n_fourier": 15,
        "crop": False,
        "val_fraction": 0.2,
    },
    "model": {"conv": "gat", "hidden": 32, "heads": 8, "dropout": 0.2, "layers": 2},
    "train": {"epochs": 60, "lr": 1e-3, "batch_size": 32, "threshold": 0.1},
}


def _merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and k in base else v
    return out


def label_metrics(y_true, y_prob, threshold):
    y_pred = y_prob > threshold
    tp = float(np.sum(y_pred & y_true))
    fp = float(np.sum(y_pred & ~y_true))
    fn = float(np.sum(~y_pred & y_true))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    accuracy = float(np.mean(y_pred == y_true))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def make_dataset(cfg, seed):
    from tci.data.generate import generate_exact_samples, generate_numerical_samples

    kwargs = dict(
        n_samples=cfg["n_samples"],
        N=cfg["N"],
        k_range=tuple(cfg["k_range"]),
        max_disc=cfg["max_disc"],
        n_fourier=cfg["n_fourier"],
        crop=cfg["crop"],
        seed=seed,
    )
    if cfg["mode"] == "exact":
        return generate_exact_samples(**kwargs)
    if cfg["mode"] == "numerical":
        return generate_numerical_samples(**kwargs)
    raise ValueError(f"unknown data mode {cfg['mode']!r}")


def train(config, out_dir):
    import torch
    from torch_geometric.loader import DataLoader

    from tci.data.graphs import sample_to_data
    from tci.models import GNNDetector

    cfg = _merge(DEFAULTS, config)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    samples = make_dataset(cfg["data"], seed)
    dataset = [sample_to_data(s) for s in samples]
    n_val = max(1, int(len(dataset) * cfg["data"]["val_fraction"]))
    train_set, val_set = dataset[n_val:], dataset[:n_val]
    print(
        f"dataset: {len(train_set)} train / {len(val_set)} val graphs "
        f"({time.time() - t0:.1f}s to generate)"
    )

    in_dim = dataset[0].x.shape[1]
    model = GNNDetector(in_dim=in_dim, **cfg["model"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(train_set, batch_size=cfg["train"]["batch_size"], shuffle=True)

    threshold = cfg["train"]["threshold"]
    history = []
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total = 0.0
        for batch in loader:
            opt.zero_grad()
            logits = model(batch.x, batch.edge_index)
            loss = loss_fn(logits, batch.y)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * batch.num_graphs
        train_loss = total / len(train_set)

        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for d in val_set:
                probs.append(torch.sigmoid(model(d.x, d.edge_index)).numpy())
                trues.append(d.y.numpy().astype(bool))
        metrics = label_metrics(
            np.concatenate(trues), np.concatenate(probs), threshold
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        if epoch % 5 == 0 or epoch == cfg["train"]["epochs"] - 1:
            print(
                f"epoch {epoch:3d}  loss {train_loss:.4f}  "
                f"P {metrics['precision']:.3f}  R {metrics['recall']:.3f}  "
                f"F1 {metrics['f1']:.3f}"
            )

    model.save(out / "model.pt")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    (out / "metrics.json").write_text(json.dumps(history, indent=2))
    print(f"saved model + metrics to {out}")
    return model, history[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML config file")
    parser.add_argument("--out", default=None, help="output dir (default runs/<name>)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text()) or {}
    out = args.out or Path("runs") / Path(args.config).stem
    train(config, out)


if __name__ == "__main__":
    main()
