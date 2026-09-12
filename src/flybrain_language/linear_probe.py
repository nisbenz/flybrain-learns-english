from __future__ import annotations

import copy

import numpy as np
import torch


def _measure(model: torch.nn.Module, features: np.ndarray, labels: np.ndarray) -> dict:
    with torch.no_grad():
        logits = model(torch.from_numpy(features))
        targets = torch.from_numpy(labels)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        predictions = logits.argmax(1)
    return {
        "accuracy": float((predictions == targets).float().mean()),
        "cross_entropy": float(loss),
        "predictions": predictions.numpy(),
    }


def fit_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    seed: int,
    epochs: int = 200,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Fit a bias-free two-way readout and select solely on validation loss."""
    torch.manual_seed(seed)
    model = torch.nn.Linear(train_features.shape[1], 2, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
    x = torch.from_numpy(train_features)
    y = torch.from_numpy(train_labels)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        validation = _measure(model, validation_features, validation_labels)
        if validation["cross_entropy"] < best_loss:
            best_loss = validation["cross_entropy"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    training = _measure(model, train_features, train_labels)
    validation = _measure(model, validation_features, validation_labels)
    for result in (training, validation):
        result.pop("predictions")
    return best_state, {
        "train": training,
        "validation": validation,
        "best_epoch": best_epoch,
        "trainable_parameters": model.weight.numel(),
    }


def evaluate_probe(
    state: dict[str, torch.Tensor], features: np.ndarray, labels: np.ndarray,
) -> dict:
    model = torch.nn.Linear(features.shape[1], 2, bias=False)
    model.load_state_dict(state)
    result = _measure(model, features, labels)
    predictions = result.pop("predictions")
    hits = predictions == labels
    result["per_class_accuracy"] = {
        str(label): float(hits[labels == label].mean()) for label in (0, 1)
    }
    result["predictions"] = predictions.tolist()
    return result
