import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_dataset
from model import build_model
from utils import load_class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/pokemon")
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    class_names = load_class_names(checkpoint_path.parent / "class_names.json")
    dataset = build_dataset(args.data_dir, "test", args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.backbone, len(class_names), pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu().tolist()
            y_pred.extend(preds)
            y_true.extend(targets.tolist())

    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred).tolist()
    output = {
        "classification_report": report,
        "confusion_matrix": matrix,
    }
    (checkpoint_path.parent / "evaluation.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

