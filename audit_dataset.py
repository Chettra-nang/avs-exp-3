import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def audit_dataset(data_dir: Path, n_samples: int = 10) -> None:
    meta_path = data_dir / "metadata.jsonl"
    if not meta_path.exists():
        print(f"No metadata found at {meta_path}")
        return

    with meta_path.open("r") as f:
        records = [json.loads(line) for line in f]

    yields = [r for r in records if r.get("label") == "Yield"]
    clears = [r for r in records if r.get("label") == "Clear"]

    print(f"Total: {len(records)} | Yield: {len(yields)} | Clear: {len(clears)}")

    samples_y = random.sample(yields, min(len(yields), n_samples)) if yields else []
    samples_c = random.sample(clears, min(len(clears), n_samples)) if clears else []

    cols = max(len(samples_y), len(samples_c), 1)
    fig, axes = plt.subplots(2, cols, figsize=(4 * cols, 6))
    fig.suptitle("Top: Yield (Ambulance behind) | Bottom: Clear")

    for i in range(cols):
        ax_y = axes[0, i] if cols > 1 else axes[0]
        ax_c = axes[1, i] if cols > 1 else axes[1]
        ax_y.axis("off")
        ax_c.axis("off")

        if i < len(samples_y):
            record = samples_y[i]
            img_path = data_dir / record["file_name"]
            if img_path.exists():
                ax_y.imshow(Image.open(img_path))
            ax_y.set_title(f"Yield | Ag:{record['agent_id']}")

        if i < len(samples_c):
            record = samples_c[i]
            img_path = data_dir / record["file_name"]
            if img_path.exists():
                ax_c.imshow(Image.open(img_path))
            ax_c.set_title(f"Clear | Ag:{record['agent_id']}")

    plt.tight_layout()
    out_path = data_dir / "audit_grid.png"
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved audit grid to {out_path}")


if __name__ == "__main__":
    audit_dataset(Path("data/ego_vla_v2"))
