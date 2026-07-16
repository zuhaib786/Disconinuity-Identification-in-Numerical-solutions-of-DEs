"""Config-driven training of the GNN troubled-cell detector.

Usage:
    python -m tci.train configs/gnn1d.yaml [--out runs/my-run]

Every ablation (architecture, features, data mode, mesh randomization, ...)
is a new YAML file, not new code.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

DEFAULTS = {
    "seed": 0,
    "data": {
        "seed": 0,
        "split_seed": 0,
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
        "seed": 0,
        "epochs": 60,
        "lr": 1e-3,
        "batch_size": 32,
        "threshold": 0.1,
        "selection_metric": "f1",
        "patience": 10,
        "min_delta": 0.0,
        "ece_bins": 10,
        "label_halo": 0,
        "device": "auto",  # auto | cpu | cuda | cuda:N
    },
}

TRAINING_PROTOCOL_VERSION = "corrected-v2"


def _merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = _merge(base[k], v) if isinstance(v, dict) and k in base else v
    return out


def resolve_config(config):
    """Merge defaults while preserving the legacy top-level seed contract."""

    raw = config or {}
    cfg = _merge(DEFAULTS, raw)
    legacy_seed = int(raw.get("seed", DEFAULTS["seed"]))
    raw_data = raw.get("data") or {}
    raw_train = raw.get("train") or {}
    cfg["seed"] = legacy_seed
    cfg["data"]["seed"] = int(raw_data.get("seed", legacy_seed))
    cfg["data"]["split_seed"] = int(raw_data.get("split_seed", legacy_seed))
    cfg["train"]["seed"] = int(raw_train.get("seed", legacy_seed))
    return cfg


def _pr_auc(y_true, y_prob):
    """Average precision, i.e. the step integral under the PR curve."""

    positives = int(np.sum(y_true))
    if positives == 0:
        return 0.0
    order = np.argsort(-y_prob, kind="mergesort")
    truth = y_true[order]
    probability = y_prob[order]
    distinct = np.where(np.diff(probability))[0]
    threshold_indices = np.concatenate([distinct, [len(probability) - 1]])
    true_positives = np.cumsum(truth)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positives
    recall_increment = np.diff(np.concatenate([[0.0], recall]))
    return float(np.sum(recall_increment * precision))


def _expected_calibration_error(y_true, y_prob, n_bins):
    if n_bins < 1:
        raise ValueError("ece_bins must be positive")
    bins = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    error = 0.0
    for bin_index in range(n_bins):
        selected = bins == bin_index
        if np.any(selected):
            error += float(np.mean(selected)) * abs(
                float(np.mean(y_prob[selected])) - float(np.mean(y_true[selected]))
            )
    return error


def label_metrics(y_true, y_prob, threshold, ece_bins=10):
    y_true = np.asarray(y_true, dtype=bool)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape or y_true.ndim != 1:
        raise ValueError("y_true and y_prob must be one-dimensional arrays of equal shape")
    if not np.all(np.isfinite(y_prob)) or np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("y_prob must contain finite probabilities in [0, 1]")
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
        "pr_auc": _pr_auc(y_true, y_prob),
        "ece": _expected_calibration_error(y_true, y_prob, int(ece_bins)),
    }


def dataset_id(samples):
    """Hash generated sample content, including mesh topology when present."""

    digest = hashlib.sha256()

    def update_array(name, values):
        array = np.ascontiguousarray(values)
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())

    digest.update(str(len(samples)).encode())
    for index, sample in enumerate(samples):
        digest.update(index.to_bytes(8, "little"))
        update_array("u", sample.u)
        update_array("labels", sample.labels)
        if getattr(sample, "aux_labels", None) is not None:
            update_array("aux_labels", sample.aux_labels)
        if getattr(sample, "x", None) is not None:
            update_array("x", sample.x)
        if hasattr(sample, "disc_locs"):
            update_array("disc_locs", sample.disc_locs)
        if hasattr(sample, "mesh"):
            update_array("mesh.points", sample.mesh.points)
            update_array("mesh.cells", sample.mesh.cells)
            metadata = {
                "curve": sample.curve,
                "parameters": sample.parameters,
                "source": sample.source,
                "time": sample.time,
                "trajectory_id": sample.trajectory_id,
            }
            digest.update(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
            )
    return digest.hexdigest()


def deterministic_split(n_samples, val_fraction, split_seed, data_identifier, groups=None):
    """Split sample indices, optionally keeping whole groups on one side.

    ``groups`` holds one key per sample; snapshots of one `data-v3` trajectory
    share a key so a smeared state cannot leak across the split.  Passing
    ``None`` reproduces the legacy per-sample permutation exactly, so the
    frozen Phase 2--4 split IDs are unchanged.
    """
    if n_samples < 2:
        raise ValueError("training requires at least two samples")
    n_val = max(1, int(n_samples * float(val_fraction)))
    if n_val >= n_samples:
        raise ValueError("val_fraction leaves no training samples")
    rng = np.random.default_rng(int(split_seed))
    if groups is None:
        permutation = rng.permutation(n_samples)
        val_indices = permutation[:n_val]
        train_indices = permutation[n_val:]
    else:
        if len(groups) != n_samples:
            raise ValueError("groups must hold one key per sample")
        members = {}
        for index, key in enumerate(groups):
            members.setdefault(key, []).append(index)
        keys = list(members)
        validation = []
        for position in rng.permutation(len(keys)):
            if len(validation) >= n_val:
                break
            validation.extend(members[keys[int(position)]])
        selected = set(validation)
        if not selected or len(selected) >= n_samples:
            raise ValueError("grouped split leaves one side empty; adjust val_fraction")
        val_indices = np.asarray(sorted(selected), dtype=np.int64)
        train_indices = np.asarray(
            [index for index in range(n_samples) if index not in selected], dtype=np.int64
        )
    digest = hashlib.sha256()
    digest.update(data_identifier.encode())
    digest.update(np.asarray(train_indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(val_indices, dtype=np.int64).tobytes())
    return train_indices, val_indices, digest.hexdigest()


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
    if cfg["mode"] == "data-v3":
        from tci.data.generate2d_v3 import load_or_generate

        spec = dict(cfg.get("data_v3") or {})
        spec["n_samples"] = n_samples
        spec["seed"] = seed
        return load_or_generate(spec, cache_dir=spec.pop("cache_dir", "runs/data-v3"))
    raise ValueError(f"unknown data mode {cfg['mode']!r}")


def split_groups(samples):
    """Group keys that keep every snapshot of one trajectory on one side."""
    return [
        f"trajectory-{sample.trajectory_id}"
        if getattr(sample, "trajectory_id", -1) >= 0
        else f"sample-{index}"
        for index, sample in enumerate(samples)
    ]


def _epoch_loop(model, opt, loss_fn, loader, batches_fn, eval_fn, cfg, n_train):
    """Train with declared validation selection and restore the best state."""
    import torch

    threshold = cfg["train"]["threshold"]
    ece_bins = int(cfg["train"]["ece_bins"])
    selection_metric = str(cfg["train"]["selection_metric"])
    patience = int(cfg["train"]["patience"])
    min_delta = float(cfg["train"]["min_delta"])
    if cfg["train"]["epochs"] < 1:
        raise ValueError("epochs must be positive")
    if patience < 1:
        raise ValueError("patience must be positive")
    if min_delta < 0.0:
        raise ValueError("min_delta must be nonnegative")
    minimize = selection_metric in {"ece", "train_loss"}
    history = []
    best_state = None
    best_epoch = None
    best_value = None
    epochs_without_improvement = 0
    stopping_reason = "max_epochs"
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
        metrics = label_metrics(y_true, y_prob, threshold, ece_bins=ece_bins)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        if selection_metric not in row:
            raise ValueError(
                f"unknown selection_metric {selection_metric!r}; "
                f"choose from {sorted(row)}"
            )
        value = float(row[selection_metric])
        improved = best_value is None or (
            value < best_value - min_delta
            if minimize
            else value > best_value + min_delta
        )
        if improved:
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
            best_epoch = epoch
            best_value = value
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(row)
        if epoch % 5 == 0 or epoch == cfg["train"]["epochs"] - 1:
            print(
                f"epoch {epoch:3d}  loss {train_loss:.4f}  "
                f"P {metrics['precision']:.3f}  R {metrics['recall']:.3f}  "
                f"F1 {metrics['f1']:.3f}  AP {metrics['pr_auc']:.3f}  "
                f"ECE {metrics['ece']:.3f}"
            )
        if epochs_without_improvement >= patience:
            stopping_reason = "early_stopping"
            break
    assert best_state is not None and best_epoch is not None and best_value is not None
    model.load_state_dict(best_state)
    for row in history:
        row["is_best"] = row["epoch"] == best_epoch
    summary = {
        "final_epoch": history[-1]["epoch"],
        "best_epoch": best_epoch,
        "best_selection_value": best_value,
        "selection_metric": selection_metric,
        "stopping_reason": stopping_reason,
    }
    history[-1].update(summary)
    return history, summary


def train(config, out_dir):
    import torch

    cfg = resolve_config(config)
    data_seed = cfg["data"]["seed"]
    split_seed = cfg["data"]["split_seed"]
    train_seed = cfg["train"]["seed"]

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
    samples = make_dataset(cfg["data"], data_seed)
    data_identifier = dataset_id(samples)
    groups = split_groups(samples) if cfg["data"]["mode"] == "data-v3" else None
    train_indices, val_indices, split_identifier = deterministic_split(
        len(samples), cfg["data"]["val_fraction"], split_seed, data_identifier, groups
    )
    train_samples = [samples[int(index)] for index in train_indices]
    val_samples = [samples[int(index)] for index in val_indices]
    print(
        f"dataset: {len(train_samples)} train / {len(val_samples)} val samples "
        f"({time.time() - t0:.1f}s to generate)"
    )

    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
        torch.set_float32_matmul_precision("high")

    model_cfg = dict(cfg["model"])
    model_type = model_cfg.pop("type", "gnn")
    feature_schema = model_cfg.pop("feature_schema", None)
    edge_dim = model_cfg.get("edge_dim")
    if feature_schema is None:
        is_2d = str(cfg["data"]["mode"]).endswith("2d")
        if model_type == "gnn":
            feature_schema = "ordered-global-v1" if is_2d else "ordered-global-1d-v1"
        else:
            feature_schema = "fixed-stencil-2d-v1" if is_2d else "fixed-stencil-1d-v1"
    cfg["model"]["feature_schema"] = feature_schema
    cfg["model"]["edge_dim"] = edge_dim
    if model_type == "gnn":
        model, history, selection = _train_gnn(
            train_samples, val_samples, model_cfg, cfg, device
        )
    elif model_type == "mlp":
        model, history, selection = _train_mlp(
            train_samples, val_samples, model_cfg, cfg, device
        )
    else:
        raise ValueError(f"unknown model type {model_type!r}")

    protocol = {
        "version": TRAINING_PROTOCOL_VERSION,
        "data_id": data_identifier,
        "split_id": split_identifier,
        "data_seed": data_seed,
        "split_seed": split_seed,
        "train_seed": train_seed,
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(val_samples),
    }
    if cfg["data"]["mode"] == "data-v3":
        from tci.data.generate2d_v3 import resolve_spec, spec_id

        spec = resolve_spec(
            {**(cfg["data"].get("data_v3") or {}), "n_samples": cfg["data"]["n_samples"], "seed": data_seed}
        )
        spec.pop("cache_dir", None)
        protocol["data_v3"] = {
            "ladder": spec["ladder"],
            "spec_id": spec_id(spec),
            "label": spec["label"],
            "mixture": spec["mixture"],
            "grouped_split": "trajectory_id",
            "positive_rate": float(
                np.mean([np.mean(sample.labels) for sample in samples])
            ),
        }
    cfg["protocol"] = protocol
    history[-1].update(protocol)
    checkpoint_metadata = {
        "feature_schema": feature_schema,
        "edge_dim": edge_dim,
        "best_epoch": selection["best_epoch"],
        "final_epoch": selection["final_epoch"],
        "stopping_reason": selection["stopping_reason"],
        "selection_metric": selection["selection_metric"],
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "label_policy": selection.get("label_policy", {"training_hops": 0}),
        **{key: protocol[key] for key in ("data_id", "split_id", "data_seed", "split_seed", "train_seed")},
    }
    model.cpu()
    model.checkpoint_metadata = checkpoint_metadata
    model.save(out / "model.pt")
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    (out / "metrics.json").write_text(json.dumps(history, indent=2))
    print(f"saved model + metrics to {out}")
    return model, history[-1]


def _train_gnn(train_samples, val_samples, model_cfg, cfg, device):
    import torch
    from torch_geometric.loader import DataLoader

    from tci.data.graphs import expand_labels_by_hops, sample2d_to_data, sample_to_data
    from tci.models import GNNDetector

    if hasattr(train_samples[0], "mesh"):
        feature_schema = cfg["model"]["feature_schema"]
        label_halo = cfg["train"].get("label_halo", 0)
        if not isinstance(label_halo, int) or label_halo < 0:
            raise ValueError("train.label_halo must be a nonnegative integer")

        def convert(sample, training):
            labels = (
                expand_labels_by_hops(sample.labels, sample.mesh, label_halo)
                if training and label_halo
                else sample.labels
            )
            return sample2d_to_data(
                sample, feature_schema=feature_schema, labels=labels
            )
    else:
        label_halo = cfg["train"].get("label_halo", 0)
        if label_halo:
            raise ValueError("train.label_halo is currently supported only for 2D graphs")

        def convert(sample, training):
            return sample_to_data(sample)

    train_set = [convert(sample, True) for sample in train_samples]
    val_set = [convert(sample, False) for sample in val_samples]

    first_x = train_set[0].x
    assert first_x is not None
    first_edge_attr = getattr(train_set[0], "edge_attr", None)
    inferred_edge_dim = None if first_edge_attr is None else first_edge_attr.shape[1]
    if model_cfg.get("edge_dim") != inferred_edge_dim:
        raise ValueError(
            f"schema {cfg['model']['feature_schema']!r} produces edge_dim="
            f"{inferred_edge_dim}, config declares {model_cfg.get('edge_dim')}"
        )
    model = GNNDetector(in_dim=first_x.shape[1], **model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(
        train_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(cfg["train"]["seed"]),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    def batches_fn(batch):
        batch = batch.to(device, non_blocking=device.type == "cuda")
        return (
            model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None)),
            batch.y,
            batch.num_graphs,
        )

    def eval_fn():
        probs, trues = [], []
        for batch in val_loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            probs.append(
                torch.sigmoid(
                    model(batch.x, batch.edge_index, getattr(batch, "edge_attr", None))
                )
            )
            trues.append(batch.y)
        return (
            torch.cat(trues).cpu().numpy().astype(bool),
            torch.cat(probs).cpu().numpy(),
        )

    history, selection = _epoch_loop(
        model, opt, loss_fn, loader, batches_fn, eval_fn, cfg, len(train_set)
    )
    original_train_positive = sum(int(np.sum(sample.labels)) for sample in train_samples)
    training_positive = sum(int(torch.sum(data.y).item()) for data in train_set)
    training_cells = sum(int(data.y.numel()) for data in train_set)
    validation_positive = sum(int(np.sum(sample.labels)) for sample in val_samples)
    validation_cells = sum(len(sample.labels) for sample in val_samples)
    selection["label_policy"] = {
        "kind": "face-adjacency halo" if label_halo else "original binary labels",
        "training_hops": int(label_halo),
        "training_original_positive_cells": original_train_positive,
        "training_target_positive_cells": training_positive,
        "training_cell_count": training_cells,
        "training_original_positive_fraction": original_train_positive / training_cells,
        "training_target_positive_fraction": training_positive / training_cells,
        "validation_hops": 0,
        "validation_positive_cells": validation_positive,
        "validation_cell_count": validation_cells,
        "validation_positive_fraction": validation_positive / validation_cells,
    }
    return model, history, selection


def _train_mlp(train_samples, val_samples, model_cfg, cfg, device):
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

    X_val, y_val = to_xy(val_samples)
    X_tr, y_tr = to_xy(train_samples)

    model = MLPDetector(in_dim=X_tr.shape[1], **model_cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=cfg["train"]["batch_size"] * 64,  # rows, not graphs
        shuffle=True,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(cfg["train"]["seed"]),
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

    history, selection = _epoch_loop(
        model, opt, loss_fn, loader, batches_fn, eval_fn, cfg, len(X_tr)
    )
    return model, history, selection


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
