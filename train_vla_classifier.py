# train_vla_classifier.py
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# CONFIG
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-5
LABELS = ["Clear", "Hold", "Yield"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABELS)}

class HighwayDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.meta_path = self.data_dir / "metadata.jsonl"
        self.samples = []
        with self.meta_path.open("r") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = self.data_dir / item["file_name"]
        image = Image.open(img_path).convert("RGB")
        label = LABEL_TO_IDX.get(item["label"], 0)
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, torch.tensor(label, dtype=torch.long)

class VLAClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip.parameters():
            param.requires_grad = False  # Freeze by default
        # Unfreeze last vision layer and post layernorm for adaptation
        for param in self.clip.vision_model.encoder.layers[-1].parameters():
            param.requires_grad = True
        for param in self.clip.vision_model.post_layernorm.parameters():
            param.requires_grad = True
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, len(LABELS))
        )

    def forward(self, pixel_values):
        features = self.clip.get_image_features(pixel_values=pixel_values)
        return self.head(features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/ego_vla_v2")
    parser.add_argument("--output-path", type=str, default="models/multi/vla_classifier.pt")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}...")

    dataset = HighwayDataset(args.data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_labels = [dataset.samples[i]["label"] for i in train_set.indices]
    class_counts = {label: 0 for label in LABELS}
    for label in train_labels:
        class_counts[label] += 1
    weights = np.array([1.0 / class_counts[label] for label in train_labels], dtype=np.float32)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    
    model = VLAClassifier().to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_yield_acc = 0.0
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        for pixels, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            pixels, labels = pixels.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(pixels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct = 0
        total = 0
        per_class_correct = {idx: 0 for idx in range(len(LABELS))}
        per_class_total = {idx: 0 for idx in range(len(LABELS))}
        with torch.no_grad():
            for pixels, labels in val_loader:
                pixels, labels = pixels.to(device), labels.to(device)
                outputs = model(pixels)
                preds = torch.argmax(outputs, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                for cls_idx in range(len(LABELS)):
                    mask = labels == cls_idx
                    per_class_total[cls_idx] += mask.sum().item()
                    if mask.any():
                        per_class_correct[cls_idx] += (preds[mask] == cls_idx).sum().item()

        val_acc = correct / total
        hold_idx = LABEL_TO_IDX["Hold"]
        yield_idx = LABEL_TO_IDX["Yield"]
        hold_acc = per_class_correct[hold_idx] / per_class_total[hold_idx] if per_class_total[hold_idx] else 0.0
        yield_acc = per_class_correct[yield_idx] / per_class_total[yield_idx] if per_class_total[yield_idx] else 0.0
        print(
            f"Epoch {epoch+1}: Val Acc: {val_acc:.4f} | HOLD ACC: {hold_acc:.4f} | "
            f"YIELD ACC: {yield_acc:.4f}"
        )

        if yield_acc > best_yield_acc:
            best_yield_acc = yield_acc
            torch.save(model.state_dict(), args.output_path)
            print(f"Saved Best Model (Yield {yield_acc:.4f}) -> {args.output_path}")

if __name__ == "__main__":
    main()
