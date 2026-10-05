from pathlib import Path
import random, shutil
import pandas as pd

random.seed(42)

# Source data (already converted to 25x25 jpg)
GTSRB_ROOT = Path("data/gtsrb_25_multiclass")

# Optional: your phone photos that you converted to 25x25 and labeled
MY_LABELS_CSV = Path("outputs/my_data_labels.csv")

# Output dataset (this is what the training script expects)
OUT = Path("data/processed_25_multiclass")

GTSRB_ROOT = Path("data/processed/gtsrb_25_multiclass")

CLASSES = sorted([d.name for d in GTSRB_ROOT.iterdir() if d.is_dir()])

# If you truly want only the 3, use:
# CLASSES = ["no entry", "speed limit (30km/h)", "speed limit (50km/h)"]

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # test is remaining 0.15

def clear_out():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

def ensure_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)

def safe_copy(src: Path, dst_dir: Path):
    # avoid filename collisions by adding a random token
    token = random.randint(0, 10**9)
    dst = dst_dir / f"{src.stem}_{token}{src.suffix}"
    shutil.copy(src, dst)

def split_list(items):
    items = list(items)
    random.shuffle(items)
    n = len(items)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return train, val, test

def load_my_data():
    """Returns dict: {class_name: [Path,...]} from outputs/my_data_labels.csv if present."""
    my = {c: [] for c in CLASSES}
    if not MY_LABELS_CSV.exists():
        return my

    df = pd.read_csv(MY_LABELS_CSV)
    if df.empty:
        return my

    df["label_name"] = df["label_name"].astype(str).str.strip().str.lower()

    for _, row in df.iterrows():
        lbl = row["label_name"]
        p = Path(row["image_path"])
        if lbl in my and p.exists():
            my[lbl].append(p)
    return my

def main():
    clear_out()
    ensure_dirs()

    # Collect GTSRB images per class
    gtsrb = {}
    for cls in CLASSES:
        cls_dir = GTSRB_ROOT / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir} (Did you convert it?)")
        gtsrb[cls] = list(cls_dir.glob("*.jpg"))

    my = load_my_data()

    print("[INFO] Source counts:")
    total = 0
    for cls in CLASSES:
        g = len(gtsrb[cls])
        m = len(my[cls])
        total += (g + m)
        print(f" - {cls:22s}  gtsrb={g:4d}  my_data={m:3d}")

    summary = {s: {c: 0 for c in CLASSES} for s in ["train", "val", "test"]}

    for cls in CLASSES:
        all_imgs = gtsrb[cls] + my[cls]
        train, val, test = split_list(all_imgs)

        for p in train:
            safe_copy(p, OUT / "train" / cls)
            summary["train"][cls] += 1
        for p in val:
            safe_copy(p, OUT / "val" / cls)
            summary["val"][cls] += 1
        for p in test:
            safe_copy(p, OUT / "test" / cls)
            summary["test"][cls] += 1

    print("\n[DONE] Built dataset at:", OUT)

    print("\nSplit Summary")
    print("-" * 72)
    print(f"{'Class':25s} {'Train':>10s} {'Val':>10s} {'Test':>10s}")
    print("-" * 72)

    for cls in CLASSES:
        print(
            f"{cls:25s} "
            f"{summary['train'][cls]:10d} "
            f"{summary['val'][cls]:10d} "
            f"{summary['test'][cls]:10d}"
        )

    print("-" * 72)

    total_train = sum(summary["train"].values())
    total_val = sum(summary["val"].values())
    total_test = sum(summary["test"].values())

    print(
        f"{'TOTAL':25s} "
        f"{total_train:10d} "
        f"{total_val:10d} "
        f"{total_test:10d}"
    )
    print("-" * 72)

if __name__ == "__main__":
    main()