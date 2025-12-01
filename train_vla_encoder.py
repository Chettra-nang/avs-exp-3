"""
Fine-tune CLIP ViT-B/32 vision encoder with a supervised head on the VLA dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from transformers import CLIPModel

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CLASS_NAMES = ["Clear", "Hold", "Yield"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune CLIP vision encoder on VLA dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/vla_highway/train"))
    parser.add_argument("--output-path", type=Path, default=Path("models/vla_classifier.pt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_datasets(root: Path, val_split: float, seed: int) -> Tuple[Dataset, Dataset, Dict[str, int]]:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    dataset = datasets.ImageFolder(root=root, transform=transform)
    total_len = len(dataset)
    val_len = int(total_len * val_split)
    train_len = total_len - val_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=generator)
    return train_ds, val_ds, dataset.class_to_idx


def freeze_clip_layers(model: CLIPModel) -> None:
    # Keep text encoder frozen but fully unfreeze the vision tower for fine-tuning.
    for param in model.text_model.parameters():
        param.requires_grad = False

    for param in model.vision_model.parameters():
        param.requires_grad = True

    if hasattr(model, "visual_projection"):
        for param in model.visual_projection.parameters():
            param.requires_grad = True


class VLAClassifier(nn.Module):
    def __init__(self, clip_model: CLIPModel, num_classes: int) -> None:
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = self.clip.get_image_features(pixel_values=pixel_values)
        return self.classifier(image_features)


def evaluate(
    model: VLAClassifier,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    yield_idx: int,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    yield_correct = 0
    yield_total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            yield_mask = labels == yield_idx
            yield_total += yield_mask.sum().item()
            if yield_mask.any():
                yield_correct += (preds[yield_mask] == labels[yield_mask]).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    yield_accuracy = yield_correct / max(yield_total, 1) if yield_total > 0 else 0.0
    return avg_loss, accuracy, yield_accuracy


def train() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_ds, val_ds, class_to_idx = build_datasets(args.data_dir, args.val_split, args.seed)
    num_classes = len(class_to_idx)
    missing = [cls for cls in CLASS_NAMES if cls not in class_to_idx]
    if missing:
        raise ValueError(f"Missing expected classes in dataset: {missing}")
    yield_idx = class_to_idx["Yield"]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    freeze_clip_layers(clip_model)

    model = VLAClassifier(clip_model, num_classes=num_classes).to(device)
    class_weights = torch.tensor([1.0, 1.2, 4.0], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.1
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    best_yield_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_loss, val_acc, yield_acc = evaluate(model, val_loader, device, criterion, yield_idx)

        print(
            f"Epoch {epoch:02d}: Train Loss {train_loss:.4f} | Train Acc {train_acc:.3f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.3f} | Yield Acc {yield_acc:.3f}"
        )

        if yield_acc > best_yield_acc:
            best_yield_acc = yield_acc
            torch.save({"model_state": model.state_dict(), "class_to_idx": class_to_idx}, args.output_path)
            print(f"Saved best Yield model ({best_yield_acc:.3f}) to {args.output_path}")
            if best_yield_acc >= 0.80:
                backup_path = args.output_path.parent / "vla_80plus.pt"
                torch.save({"model_state": model.state_dict(), "class_to_idx": class_to_idx}, backup_path)
                print(f"Yield >= 80%, also saved backup to {backup_path}")


if __name__ == "__main__":
    train()
