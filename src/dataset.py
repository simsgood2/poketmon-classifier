from pathlib import Path

from torchvision import datasets, transforms


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_dataset(data_dir: str, split: str, image_size: int):
    root = Path(data_dir) / split
    if not root.exists():
        raise FileNotFoundError(f"Dataset split not found: {root}")
    return datasets.ImageFolder(root=root, transform=build_transforms(image_size, train=split == "train"))

