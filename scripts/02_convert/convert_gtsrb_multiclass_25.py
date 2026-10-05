import pandas as pd
from pathlib import Path
from PIL import Image
import shutil
import random

random.seed(42)

SRC_CSV = Path("outputs/gtsrb_labels_with_names.csv")
OUT_DIR = Path("data/processed/gtsrb_25_multiclass")
TARGET_SIZE = (25, 25)

# Use None to keep all available images per class
# Or set to an int like 1000 if you want to cap classes
MAX_PER_CLASS = None

def clear_dir(p: Path):
    if p.exists():
        if p.is_symlink():
            raise RuntimeError(f"{p} is a symlink. Point OUT_DIR to the real folder instead.")
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def main():
    if not SRC_CSV.exists():
        print("ERROR: Missing:", SRC_CSV)
        return

    clear_dir(OUT_DIR)

    df = pd.read_csv(SRC_CSV)
    df["label_name"] = df["label_name"].str.lower()

    # ALL classes from GTSRB
    target_labels = sorted(df["label_name"].unique())

    for lbl in target_labels:
        (OUT_DIR / lbl).mkdir(parents=True, exist_ok=True)

    total = 0

    print(f"Total classes found: {len(target_labels)}\n")

    for lbl in target_labels:
        rows = df[df["label_name"] == lbl]["image_path"].tolist()
        random.shuffle(rows)

        if MAX_PER_CLASS is not None:
            rows = rows[:min(MAX_PER_CLASS, len(rows))]

        converted = 0
        for p in rows:
            src = Path(p)
            if not src.exists():
                continue

            try:
                with Image.open(src) as im:
                    im = im.convert("RGB").resize(TARGET_SIZE)
                    out = OUT_DIR / lbl / (src.stem + ".jpg")
                    im.save(out, format="JPEG", quality=95)
                converted += 1
            except Exception as e:
                print("Skip error:", src, e)

        print(f"[GTSRB] {lbl}: converted {converted}")
        total += converted

    print("\nDONE")
    print("Output:", OUT_DIR)
    print("Total converted:", total)

if __name__ == "__main__":
    main()