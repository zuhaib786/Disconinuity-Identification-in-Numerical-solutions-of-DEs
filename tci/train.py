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
        "mode": "exact",  # exact | numerical | euler_riemann
        "n_samples": 2000,
        "N": 1,
        "k_range": [50, 150],
        "max_disc": 5,
        "n_fourier": 15,
        "crop": False,
        "val_fraction": 0.2,
        "domain": [0.0, 1.0],
        "a_range": [0.5, 1.0],
        "t_range": [0.1, 0.3],
        "cfl": 0.375,
    },
    "model": {
        "type": "gnn",  # gnn | mlp (fixed-stencil Ray-Hesthaven-style baseline)
        "conv": "gat",
        "hidden": 32,
        "heads": 8,
        "dropout": 0.2,
        "layers": 2,
    },
    "train": {
        "epochs": 60,
        "lr": 1e-3,
        "batch_size": 32,
        "threshold": 0.1,
        "device": "auto",  # auto | cpu | cuda | cuda:N
    },
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
    from tci.data.generate import (
        generate_euler_riemann_samples,
        generate_exact_samples,
        generate_numerical_samples,
    )

    n_samples = int(cfg["n_samples"])
    N = int(cfg["N"])
    k_range = tuple(cfg["k_range"])
    max_disc = int(cfg["max_disc"])
    n_fourier = int(cfg["n_fourier"])
    crop = bool(cfg["crop"])
    domain = tuple(cfg.get("domain", (0.0, 1.0)))
    if cfg["mode"] == "exact":
        return generate_exact_samples(
            n_samples, N, k_range, domain, max_disc, n_fourier, seed, crop
        )
    if cfg["mode"] == "numerical":
        return generate_numerical_samples(
            n_samples,
            N,
            k_range,
            domain,
            max_disc,
            n_fourier,
            seed,
            crop,
            a_range=tuple(cfg.get("a_range", (0.5, 1.0))),
            t_range=tuple(cfg.get("t_range", (0.1, 0.3))),
            cfl=cfg.get("cfl", 0.375),
        )
    if cfg["mode"] == "euler_riemann":
        euler_kwargs = {
            key: cfg[key]
            for key in (
                "rho_range",
                "velocity_range",
                "pressure_range",
                "diaphragm_range",
                "t_range",
                "cfl",
                "max_attempts",
            )
            if key in cfg
        }
        for key in (
            "rho_range",
            "velocity_range",
            "pressure_range",
            "diaphragm_range",
            "t_range",
        ):
            if key in euler_kwargs:
                euler_kwargs[key] = tuple(euler_kwargs[key])
        return generate_euler_riemann_samples(
            n_samples,
            N,
            k_range,
            domain,
            seed,
            crop,
            **euler_kwargs,
        )
    if cfg["mode"] == "exact2d":
        from tci.data.generate2d import generate_exact_2d_samples

        return generate_exact_2d_samples(
            n_samples=n_samples,
            domain=(tuple(cfg.get("xlim", (0.0, 1.0))), tuple(cfg.get("ylim", (0.0, 1.0)))),
            mesh_type=cfg.get("mesh_type", "delaunay"),
            n_interior_range=tuple(cfg.get("n_interior_range", (50, 150))),
            boundary_divisions=tuple(cfg.get("boundary_divisions", (8, 8))),
            structured_range=tuple(cfg.get("structured_range", (6, 12))),
            curves=tuple(cfg.get("curves", ("line", "circle"))),
            coefficient_sigma=cfg.get("coefficient_sigma", 1.0),
            seed=seed,
        )
    if cfg["mode"] in ("numerical2d", "mixed2d"):
        from tci.data.generate2d import (
            generate_mixed_2d_samples,
            generate_numerical_2d_samples,
        )

        mesh_range_2d = tuple(cfg.get("mesh_range", (8, 12)))
        curves_2d = tuple(cfg.get("curves", ("line", "circle")))
        time_range_2d = tuple(cfg.get("time_range", (0.02, 0.2)))
        limited_fraction = float(cfg.get("limited_fraction", 0.0))
        cfl_2d = float(cfg.get("cfl", 0.15))
        max_steps = int(cfg.get("max_steps", 1500))
        max_seconds_per_sample = float(cfg.get("max_seconds_per_sample", 5.0))
        max_generation_seconds = float(cfg.get("max_generation_seconds", 600.0))
        if cfg["mode"] == "numerical2d":
            return generate_numerical_2d_samples(
                n_samples,
                mesh_range_2d,
                curves_2d,
                time_range_2d,
                limited_fraction,
                cfl_2d,
                max_steps,
                max_seconds_per_sample,
                max_generation_seconds,
                seed,
            )
        return generate_mixed_2d_samples(
            n_samples,
            exact_fraction=cfg.get("exact_fraction", 0.25),
            seed=seed,
            mesh_range=mesh_range_2d,
            curves=curves_2d,
            time_range=time_range_2d,
            limited_fraction=limited_fraction,
            cfl=cfl_2d,
            max_steps=max_steps,
            max_seconds_per_sample=max_seconds_per_sample,
            max_generation_seconds=max_generation_seconds,
        )
    raise ValueError(f"unknown data mode {cfg['mode']!r}")


def _epoch_loop(model, opt, loss_fn, loader, batches_fn, eval_fn, cfg, n_train):
    """Shared train/eval loop; returns per-epoch history."""
    import torch

    threshold = cfg["train"]["threshold"]
    history = []
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total = None
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            logits, target, weight = batches_fn(batch)
            loss = loss_fn(logits, target)
            loss.backward()
            opt.step()
            weighted_loss = loss.detach() * weight
            total = weighted_loss if total is None else total + weighted_loss
        assert total is not None
        train_loss = float(total.cpu()) / n_train

        model.eval()
        with torch.no_grad():
            y_true, y_prob = eval_fn()
        metrics = label_metrics(y_true, y_prob, threshold)
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        if epoch % 5 == 0 or epoch == cfg["train"]["epochs"] - 1:
            print(
                f"epoch {epoch:3d}  loss {train_loss:.4f}  "
                f"P {metrics['precision']:.3f}  R {metrics['recall']:.3f}  "
                f"F1 {metrics['f1']:.3f}"
            )
    return history


def train(config, out_dir):
    import torch

    cfg = _merge(DEFAULTS, config)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")

    requested_device = str(cfg["train"].get("device", "auto"))
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    cfg["train"]["resolved_device"] = str(device)
    print(f"training device: {device}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    samples = make_dataset(cfg["data"], seed)
    n_val = max(1, int(len(samples) * cfg["data"]["val_fraction"]))
    print(
        f"dataset: {len(samples) - n_val} train / {n_val} val samples "
        f"({time.time() - t0:.1f}s to generate)"
    )

    model_cfg = dict(cfg["model"])
    model_type = model_cfg.pop("type", "gnn")
    if model_type == "gnn":
        model, history = _train_gnn(samples, n_val, model_cfg, cfg, device)
    elif model_type == "mlp":
        model, history = _train_mlp(samples, n_val, model_cfg, cfg, device)
    else:
        raise ValueError(f"unknown model type {model_type!r}")

    model.cpu()
    model.save(out / "model.pt")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    (out / "metrics.json").write_text(json.dumps(history, indent=2))
    print(f"saved model + metrics to {out}")
    return model, history[-1]


def _train_gnn(samples, n_val, model_cfg, cfg, device):
    import torch
    from torch_geometric.loader import DataLoader

    from tci.data.graphs import sample2d_to_data, sample_to_data
    from tci.models import GNNDetector

    convert = sample2d_to_data if hasattr(samples[0], "mesh") else sample_to_data
    dataset = [convert(s) for s in samples]
    train_set, val_set = dataset[n_val:], dataset[:n_val]

    first_x = dataset[0].x
    assert first_x is not None
    model = GNNDetector(in_dim=first_x.shape[1], **model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    def batches_fn(batch):
        batch = batch.to(device, non_blocking=device.type == "cuda")
        return model(batch.x, batch.edge_index), batch.y, batch.num_graphs

    def eval_fn():
        probs, trues = [], []
        for batch in val_loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            probs.append(torch.sigmoid(model(batch.x, batch.edge_index)))
            trues.append(batch.y)
        return (
            torch.cat(trues).cpu().numpy().astype(bool),
            torch.cat(probs).cpu().numpy(),
        )

    history = _epoch_loop(
        model, opt, loss_fn, loader, batches_fn, eval_fn, cfg, len(train_set)
    )
    return model, history


def _train_mlp(samples, n_val, model_cfg, cfg, device):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from tci.data.features import stencil_features, stencil_features2d
    from tci.models import MLPDetector

    # Only the MLP-relevant hyperparameters apply.
    model_cfg = {k: v for k, v in model_cfg.items() if k in ("hidden", "dropout")}

    def to_xy(subset):
        if hasattr(subset[0], "mesh"):
            X = np.concatenate(
                [stencil_features2d(s.u, s.mesh) for s in subset]
            )
        else:
            X = np.concatenate([stencil_features(s.u) for s in subset])
        y = np.concatenate([s.labels for s in subset]).astype(np.float32)
        return torch.from_numpy(X), torch.from_numpy(y)

    X_val, y_val = to_xy(samples[:n_val])
    X_tr, y_tr = to_xy(samples[n_val:])

    model = MLPDetector(in_dim=X_tr.shape[1], **model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=cfg["train"]["batch_size"] * 64,  # rows, not graphs
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    X_val_device = X_val.to(device, non_blocking=device.type == "cuda")
    y_val_numpy = y_val.numpy().astype(bool)

    def batches_fn(batch):
        xb, yb = batch
        xb = xb.to(device, non_blocking=device.type == "cuda")
        yb = yb.to(device, non_blocking=device.type == "cuda")
        return model(xb), yb, len(xb)

    def eval_fn():
        prob = torch.sigmoid(model(X_val_device)).cpu().numpy()
        return y_val_numpy, prob

    history = _epoch_loop(
        model, opt, loss_fn, loader, batches_fn, eval_fn, cfg, len(X_tr)
    )
    return model, history


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
