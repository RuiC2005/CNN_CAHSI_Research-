import pandas as pd 
from pathlib import Path
import shutil
import random

random.seed(42)

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

def safe_copy(src_path: str, dst_dir: Path):
    src = Path(src_path)
    if not src.exists():
        return False
    # avoid collisions
    token = random.randint(0, 10**9)
    dst = dst_dir / f"{src.stem}_{token}{src.suffix}"
    shutil.copy(src, dst)
    return True

def main():
    clear_dir(OUT)
    make_dirs()

    # --- GTSRB stop/yield (already labeled + has names) ---
    g = pd.read_csv("outputs/gtsrb_labels_with_names.csv")
    g["label_name"] = g["label_name"].str.lower()
    g = g[g["label_name"].isin(CLASSES)][["image_path", "label_name"]]

    # --- my_data stop/yield (your labels) ---
    m = pd.read_csv("outputs/my_data_labels.csv")
    m["label_name"] = m["label_name"].str.lower()
    m = m[m["label_name"].isin(CLASSES)][["image_path", "label_name"]]

    df = pd.concat([g, m], ignore_index=True)

    print("[INFO] Rows to use:")
    print(" - GTSRB:", len(g))
    print(" - my_data:", len(m))
    print(" - total:", len(df))

    # split per class so it's balanced across splits
    summary = {s: {c: 0 for c in CLASSES} for s in ["train","val","test"]}

    for cls in CLASSES:
        rows = df[df["label_name"] == cls]["image_path"].tolist()
        tr, va, te = split_rows(rows)

        for p in tr:
            if safe_copy(p, OUT / "train" / cls):
                summary["train"][cls] += 1
        for p in va:
            if safe_copy(p, OUT / "val" / cls):
                summary["val"][cls] += 1
        for p in te:
            if safe_copy(p, OUT / "test" / cls):
                summary["test"][cls] += 1

    print("\n[DONE] Built dataset at:", OUT)
    for split in ["train","val","test"]:
        print(split, summary[split])

if __name__ == "__main__":
    main()
