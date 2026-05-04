import argparse
import random
import shutil
from pathlib import Path


def copy_split(files, target_dir):
    target_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, target_dir / file.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/raw")
    parser.add_argument("--target", default="data/pokemon")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source)
    target = Path(args.target)
    random.seed(args.seed)

    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for class_dir in sorted([p for p in source.iterdir() if p.is_dir()]):
        files = [p for p in class_dir.iterdir() if p.suffix.lower() in image_exts]
        random.shuffle(files)
        train_end = int(len(files) * args.train_ratio)
        val_end = train_end + int(len(files) * args.val_ratio)

        splits = {
            "train": files[:train_end],
            "val": files[train_end:val_end],
            "test": files[val_end:],
        }
        for split, split_files in splits.items():
            copy_split(split_files, target / split / class_dir.name)

        print(f"{class_dir.name}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")


if __name__ == "__main__":
    main()

