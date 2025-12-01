"""
Validation utilities for the synthetic VLA highway-env dataset.
Performs logic checks, distribution analysis, and visual audits.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

CLASSES = ("Yield", "Hold", "Clear")


def load_metadata(path: Path) -> List[Dict]:
    entries: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


def logic_audit(entries: Sequence[Dict]) -> None:
    yield_entries = [e for e in entries if e["text"] == "Yield"]
    errors = [e for e in yield_entries if float(e["ground_truth_dist"]) >= 50.0]
    rate = (len(errors) / len(yield_entries)) if yield_entries else 0.0
    print(f"Logic Error Rate: {rate:.4f} ({len(errors)}/{len(yield_entries)} Yield samples violating dist<50)")


def distribution_check(entries: Sequence[Dict], out_dir: Path) -> None:
    counts = Counter(e["text"] for e in entries)
    total = sum(counts.values())
    ratios = {cls: (counts[cls] / total) if total else 0.0 for cls in CLASSES}
    print("Class ratios:", ", ".join(f"{cls}={ratios[cls]:.3f}" for cls in CLASSES))

    yield_ratio = ratios.get("Yield", 0.0)
    if yield_ratio < 0.2 or yield_ratio > 0.5:
        print("Warning: Yield class outside recommended 20%-50% range.")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(CLASSES, [counts[cls] for cls in CLASSES], color=["#4c72b0", "#55a868", "#c44e52"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    fig.tight_layout()
    hist_path = out_dir / "class_hist.png"
    fig.savefig(hist_path)
    plt.close(fig)
    print(f"Saved class histogram to {hist_path}")


def sample_images_by_class(entries: Sequence[Dict], k: int, rng: random.Random) -> Dict[str, List[str]]:
    by_class: Dict[str, List[str]] = defaultdict(list)
    for e in entries:
        by_class[e["text"]].append(e["file_name"])
    samples: Dict[str, List[str]] = {}
    for cls in CLASSES:
        paths = by_class.get(cls, [])
        if not paths:
            samples[cls] = []
            continue
        if len(paths) <= k:
            samples[cls] = paths
        else:
            samples[cls] = rng.sample(paths, k)
    return samples


def visual_audit(entries: Sequence[Dict], data_dir: Path, out_path: Path, seed: int) -> None:
    rng = random.Random(seed)
    samples = sample_images_by_class(entries, k=5, rng=rng)

    first_path = None
    for paths in samples.values():
        if paths:
            first_path = paths[0]
            break
    if first_path is None:
        print("No images available for visual audit.")
        return

    first_img = Image.open(data_dir / first_path).convert("RGB")
    w, h = first_img.size
    caption_h = 16
    cols = 5
    rows = len(CLASSES)
    grid = Image.new("RGB", (cols * w, rows * (h + caption_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.load_default()

    for row, cls in enumerate(CLASSES):
        for col in range(cols):
            paths = samples.get(cls, [])
            if col >= len(paths):
                continue
            img = Image.open(data_dir / paths[col]).convert("RGB")
            grid.paste(img, (col * w, row * (h + caption_h)))
            caption = f"{cls}: {Path(paths[col]).name}"
            text_y = row * (h + caption_h) + h
            draw.text((col * w + 2, text_y), caption, fill=(0, 0, 0), font=font)

    grid.save(out_path)
    print(f"Saved visual audit grid to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate VLA highway-env dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/vla_highway"), help="Dataset root directory.")
    parser.add_argument("--metadata", type=Path, default=None, help="Path to metadata.jsonl (defaults to <data-dir>/metadata.jsonl).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for visual sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = args.metadata or (args.data_dir / "metadata.jsonl")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.jsonl not found at {meta_path}")

    entries = load_metadata(meta_path)
    print(f"Loaded {len(entries)} metadata entries from {meta_path}")

    logic_audit(entries)
    distribution_check(entries, args.data_dir)

    audit_grid_path = args.data_dir / "audit_grid.png"
    visual_audit(entries, data_dir=args.data_dir, out_path=audit_grid_path, seed=args.seed)


if __name__ == "__main__":
    main()
