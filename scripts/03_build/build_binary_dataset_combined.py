from pathlib import Path
import pandas as pd
import random
import shutil

random.seed(42)

# Sources
GTSRB_BAL = Path("data/gtsrb_25_balanced")
MY_CSV = Path("outputs/my_data_labels.csv")

# Output dataset
OUT = Path("data/processed_25_binary")

CLASSES = ["stop", "yield"]

def clear_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def make_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (OUT / split / cls).mkdir(parents=True, exist_ok=True)

def split_rows(rows, train=0.8, val=0.1, test=0.1):
    rows = rows[:]
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * train)
    n_val = int(n * val)
    return rows[:n_train], rows[n_train:n_train+n_val], rows[n_train+n_val:]

def safe_copy(src: Path, dst_dir: Path):
    token = random.randint(0, 10**9)
    dst = dst_dir / f"{src.stem}_{token}{src.suffix}"
    shutil.copy(src, dst)

def main():
    clear_dir(OUT)
    make_dirs()

    # ---- Load balanced GTSRB images directly from folders ----
    gtsrb_paths = []
    for cls in CLASSES:
        gtsrb_paths += [(p, cls) for p in (GTSRB_BAL / cls).glob("*.jpg")]

    # ---- Load my_data from CSV, keep only stop/yield ----
    df = pd.read_csv(MY_CSV)
    df["label_name"] = df["label_name"].str.lower()
    df = df[df["label_name"].isin(CLASSES)].copy()
    my_paths = [(Path(p), lbl) for p, lbl in zip(df["image_path"], df["label_name"])]

    print("[INFO] Source counts:")
    print(" - GTSRB balanced:", len(gtsrb_paths))
    print(" - my_data stop/yield:", len(my_paths))
    print(" - total:", len(gtsrb_paths) + len(my_paths))

    # Split per class (keeps class distribution stable across splits)
    summary = {s: {c: 0 for c in CLASSES} for s in ["train","val","test"]}

    for cls in CLASSES:
        rows = [p for p, y in gtsrb_paths if y == cls] + [p for p, y in my_paths if y == cls]
        tr, va, te = split_rows(rows)

        for p in tr:
            safe_copy(p, OUT / "train" / cls)
            summary["train"][cls] += 1
        for p in va:
            safe_copy(p, OUT / "val" / cls)
            summary["val"][cls] += 1
        for p in te:
            safe_copy(p, OUT / "test" / cls)
            summary["test"][cls] += 1

    print("\n[DONE] Built combined dataset at:", OUT)
    for split in ["train","val","test"]:
        print(split, summary[split])

if __name__ == "__main__":
    main()
