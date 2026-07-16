import json

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from tci.models import GNNDetector  # noqa: E402
from tci.train import (  # noqa: E402
    TRAINING_PROTOCOL_VERSION,
    _epoch_loop,
    deterministic_split,
    label_metrics,
    resolve_config,
    train,
)


def test_protocol_config_uses_legacy_seed_as_fallback():
    legacy = resolve_config({"seed": 7})
    assert legacy["data"]["seed"] == 7
    assert legacy["data"]["split_seed"] == 7
    assert legacy["train"]["seed"] == 7

    explicit = resolve_config(
        {
            "seed": 7,
            "data": {"seed": 1, "split_seed": 2},
            "train": {"seed": 3},
        }
    )
    assert explicit["data"]["seed"] == 1
    assert explicit["data"]["split_seed"] == 2
    assert explicit["train"]["seed"] == 3


def test_label_metrics_include_pr_auc_and_ece():
    metrics = label_metrics(
        np.array([True, False, True, False]),
        np.array([0.9, 0.8, 0.7, 0.1]),
        threshold=0.5,
        ece_bins=10,
    )
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["pr_auc"] == pytest.approx(5.0 / 6.0)
    assert metrics["ece"] == pytest.approx(0.325)


def test_deterministic_split_depends_on_split_seed_and_data_id():
    train_a, val_a, split_a = deterministic_split(20, 0.2, 4, "data-a")
    train_b, val_b, split_b = deterministic_split(20, 0.2, 4, "data-a")
    _, _, split_c = deterministic_split(20, 0.2, 5, "data-a")
    _, _, split_d = deterministic_split(20, 0.2, 4, "data-b")
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(val_a, val_b)
    assert split_a == split_b
    assert split_a != split_c
    assert split_a != split_d


def test_epoch_loop_restores_best_state_and_stops_early():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    x = torch.ones(1, 1)
    target = torch.ones(1)
    evaluated_states = []

    def batches_fn(_batch):
        return model(x).squeeze(1), target, 1

    def eval_fn():
        evaluated_states.append(model.weight.detach().clone())
        if len(evaluated_states) == 1:
            return np.array([True, False]), np.array([0.9, 0.1])
        return np.array([True, False]), np.array([0.1, 0.1])

    cfg = resolve_config(
        {
            "train": {
                "epochs": 10,
                "selection_metric": "f1",
                "patience": 2,
            }
        }
    )
    history, summary = _epoch_loop(
        model,
        optimizer,
        loss_fn,
        [None],
        batches_fn,
        eval_fn,
        cfg,
        n_train=1,
    )
    assert summary == {
        "final_epoch": 2,
        "best_epoch": 0,
        "best_selection_value": 1.0,
        "selection_metric": "f1",
        "stopping_reason": "early_stopping",
    }
    assert len(history) == 3
    assert torch.equal(model.weight, evaluated_states[0])
    assert [row["is_best"] for row in history] == [True, False, False]


def test_checkpoint_metadata_roundtrip_and_legacy_load(tmp_path):
    model = GNNDetector(in_dim=2, hidden=4, heads=1)
    metadata = {
        "feature_schema": "ordered-global-v1",
        "edge_dim": None,
        "best_epoch": 3,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
    }
    model.save(tmp_path / "new.pt", metadata=metadata)
    loaded = GNNDetector.load(tmp_path / "new.pt")
    assert loaded.checkpoint_metadata == metadata
    model.eval()
    x = torch.rand(5, 2)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]])
    with torch.no_grad():
        assert torch.equal(model(x, edge_index), loaded(x, edge_index))

    torch.save(
        {"hparams": model.hparams, "state_dict": model.state_dict()},
        tmp_path / "legacy.pt",
    )
    legacy = GNNDetector.load(tmp_path / "legacy.pt")
    assert legacy.checkpoint_metadata == {}


def test_training_seeds_share_data_and_split_ids(tmp_path):
    base = {
        "data": {
            "seed": 11,
            "split_seed": 12,
            "n_samples": 16,
            "k_range": [12, 16],
            "val_fraction": 0.25,
        },
        "model": {"hidden": 4, "heads": 1, "dropout": 0.0},
        "train": {"epochs": 2, "batch_size": 4, "patience": 2},
    }
    runs = []
    for seed in (0, 1):
        config = json.loads(json.dumps(base))
        config["train"]["seed"] = seed
        out = tmp_path / f"seed{seed}"
        model, _ = train(config, out)
        saved_config = yaml.safe_load((out / "config.yaml").read_text())
        history = json.loads((out / "metrics.json").read_text())
        loaded = GNNDetector.load(out / "model.pt")
        assert loaded.checkpoint_metadata["best_epoch"] == history[-1]["best_epoch"]
        assert loaded.checkpoint_metadata["train_seed"] == seed
        assert model.checkpoint_metadata == loaded.checkpoint_metadata
        assert {"pr_auc", "ece"} <= history[-1].keys()
        runs.append((saved_config, history[-1]))

    assert runs[0][0]["protocol"]["data_id"] == runs[1][0]["protocol"]["data_id"]
    assert runs[0][0]["protocol"]["split_id"] == runs[1][0]["protocol"]["split_id"]
    assert runs[0][1]["data_id"] == runs[1][1]["data_id"]
    assert runs[0][1]["split_id"] == runs[1][1]["split_id"]
    assert runs[0][1]["train_seed"] != runs[1][1]["train_seed"]
