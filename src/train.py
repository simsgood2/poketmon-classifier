import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_dataset
from model import build_model, configure_trainable_layers
from utils import plot_history, save_class_names, save_history, set_seed


def load_experiment(config_path, experiment_name):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    common = config["common"]
    experiment = config["experiments"][experiment_name]
    return {**common, **experiment, "name": experiment_name}


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in tqdm(loader, leave=False):
        images = images.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    return avg_loss, acc


def evaluate_metrics(model, loader, device):
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in tqdm(loader, leave=False):
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_targets.extend(targets.tolist())

    return {
        "accuracy": accuracy_score(all_targets, all_preds),
        "precision_macro": precision_score(all_targets, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_targets, all_preds, average="macro", zero_division=0),
        "f1_macro": f1_score(all_targets, all_preds, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    cfg = load_experiment(args.config, args.experiment)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = build_dataset(cfg["data_dir"], "train", cfg["image_size"])
    val_dataset = build_dataset(cfg["data_dir"], "val", cfg["image_size"])
    test_dataset = build_dataset(cfg["data_dir"], "test", cfg["image_size"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    model = build_model(cfg["backbone"], len(train_dataset.classes), cfg["pretrained"])
    model = configure_trainable_layers(
        model,
        cfg["backbone"],
        cfg["freeze_backbone"],
        cfg["unfreeze_last_block"],
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    run_dir = Path(cfg["output_dir"]) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_class_names(train_dataset.classes, run_dir / "class_names.json")

    best_val_acc = 0.0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": cfg["backbone"],
                    "num_classes": len(train_dataset.classes),
                    "class_names": train_dataset.classes,
                    "config": cfg,
                },
                run_dir / "best_model.pt",
            )

    save_history(history, run_dir / "history.csv")
    plot_history(history, run_dir / "learning_curve.png")

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_metrics(model, test_loader, device)
    metrics["best_val_acc"] = best_val_acc
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

