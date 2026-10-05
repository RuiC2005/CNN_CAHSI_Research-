from pathlib import Path
import random
import pandas as pd
from PIL import Image

# Reproducibility
RNG_SEED = 42
random.seed(RNG_SEED)

TARGET_SIZE = (25, 25)  # (width, height)
OUT_ROOT = Path("data/processed_25")

# Binary mapping
LABEL_MAP = {"yield": 0, "stop": 1}
KEEP = set(LABEL_MAP.keys())

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".tif", ".tiff", ".webp"}

def ensure_dirs():
    for split in ["train", "val", "test"]:
        for cls in ["stop", "yield"]:
            (OUT_ROOT / split / cls).mkdir(parents=True, exist_ok=True)

def to_rgb_jpeg_25(in_path: Path, out_path: Path):
    with Image.open(in_path) as im:
        # Normalize color channels: RGBA/L/etc -> RGB
        im = im.convert("RGB")
        # Resize to 25x25
        im = im.resize(TARGET_SIZE)
        # Save as JPEG
        im.save(out_path, format="JPEG", quality=95, optimize=True)

def load_gtsrb_stop_yield():
    """
    Uses your gtsrb_labels_with_names.csv which already has label_name.
    """
    df = pd.read_csv("outputs/gtsrb_labels_with_names.csv")
    # Keep only stop/yield
    df["label_name"] = df["label_name"].str.lower()
    df = df[df["label_name"].isin(["stop", "yield"])].copy()
    return df[["image_path", "label_name"]]

def load_my_data_stop_yield():
    """
    User-provided labels for your own photos.
    """
    p = Path("outputs/my_data_labels.csv")
    if not p.exists():
        print("[WARN] outputs/my_data_labels.csv not found. Skipping my_data.")
        return pd.DataFrame(columns=["image_path", "label_name"])
    df = pd.read_csv(p)
    df["label_name"] = df["label_name"].str.lower()
    df = df[df["label_name"].isin(["stop", "yield"])].copy()
    return df[["image_path", "label_name"]]

def split_rows(rows, train=0.8, val=0.1, test=0.1):
    assert abs(train + val + test - 1.0) < 1e-9
    rows = rows[:]
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * train)
    n_val = int(n * val)
    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train + n_val]
    test_rows = rows[n_train + n_val:]
    return train_rows, val_rows, test_rows

def copy_split(rows, split_name):
    """
    rows: list of (image_path, label_name)
    """
    counts = {"stop": 0, "yield": 0}
    for src, label in rows:
        srcp = Path(src)
        if srcp.suffix.lower() not in IMAGE_EXTS:
            continue
        if not srcp.exists():
            continue

        # name output deterministically
        # keep original stem but add a random token to avoid collisions
        token = random.randint(0, 10**9)
        outp = OUT_ROOT / split_name / label / f"{srcp.stem}_{token}.jpg"
        to_rgb_jpeg_25(srcp, outp)
        counts[label] += 1
    return counts

def main():
    ensure_dirs()

    gtsrb = load_gtsrb_stop_yield()
    myd = load_my_data_stop_yield()

    print("[INFO] GTSRB stop/yield rows:", len(gtsrb))
    print("[INFO] my_data stop/yield rows:", len(myd))

    # Combine
    all_df = pd.concat([gtsrb, myd], ignore_index=True)
    all_df = all_df.sample(frac=1.0, random_state=RNG_SEED).reset_index(drop=True)

    # Split per class to keep balance
    results = {}
    for cls in ["stop", "yield"]:
        cls_rows = list(all_df[all_df["label_name"] == cls].itertuples(index=False, name=None))
        tr, va, te = split_rows(cls_rows)
        results.setdefault("train", []).extend(tr)
        results.setdefault("val", []).extend(va)
        results.setdefault("test", []).extend(te)

    # Shuffle within split
    for s in results:
        random.shuffle(results[s])

    train_counts = copy_split(results["train"], "train")
    val_counts = copy_split(results["val"], "val")
    test_counts = copy_split(results["test"], "test")

    print("\n[DONE] Wrote processed JPEG 25x25 dataset to:", OUT_ROOT)
    print("Train counts:", train_counts)
    print("Val counts:", val_counts)
    print("Test counts:", test_counts)

if __name__ == "__main__":
    main()
