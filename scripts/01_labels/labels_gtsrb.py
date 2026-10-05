from pathlib import Path
import pandas as pd

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".tif", ".tiff"}

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

def build_rel_lookup(images, base: Path):
    """
    Build multiple keys -> absolute path to robustly match CSV filenames.
    Keys include:
      - tail filename (e.g., 00001.png)
      - path relative to base (posix)
      - path relative to base/archive (posix)
    """
    lookup = {}
    for p in images:
        try:
            rel = p.relative_to(base).as_posix()
        except Exception:
            rel = p.as_posix()

        lookup[p.name] = p  # tail
        lookup[rel] = p     # relative to data/GTSRB
        # also allow matching relative to archive
        try:
            rel2 = p.relative_to(base / "archive").as_posix()
            lookup[rel2] = p
        except Exception:
            pass
    return lookup

def normalize_path_str(s: str):
    # Normalize slashes and strip leading ./ if present
    s = str(s).replace("\\", "/").lstrip("./")
    return s

def label_from_csv(csv_path: Path, gtsrb_base: Path, images, default_label_col="ClassId"):
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        # Common for GT-final_test.csv depending on packaging; skip if empty
        print(f"[GTSRB] WARNING: {csv_path.name} appears empty/unreadable; skipping.")
        return pd.DataFrame(columns=["image_path","label_id","source_csv"]), pd.DataFrame(columns=["csv_file","csv_filename"]), None, None
    except Exception:
        # fallback for semicolon-delimited CSVs (common in official GTSRB)
        df = pd.read_csv(csv_path, sep=";")

    cols = {c.lower(): c for c in df.columns}

    # Try to infer filename column
    filename_candidates = ["filename", "path", "image", "img"]
    fname_col = None
    for c in filename_candidates:
        if c in cols:
            fname_col = cols[c]
            break
    if fname_col is None:
        raise ValueError(f"Could not find a filename column in {csv_path} with columns {list(df.columns)}")

    # Try to infer label column
    label_candidates = ["classid", "class_id", "label", "class"]
    label_col = None
    for c in label_candidates:
        if c in cols:
            label_col = cols[c]
            break

    # Some CSVs might not have labels (e.g., Test.csv sometimes is unlabeled).
    if label_col is None:
        label_col = default_label_col if default_label_col in df.columns else None

    img_lookup = build_rel_lookup(images, gtsrb_base)

    out_rows = []
    missing_rows = []

    for _, r in df.iterrows():
        raw = normalize_path_str(r[fname_col])
        tail = raw.split("/")[-1]

        # Try different matching keys
        match = None
        for key in (raw, tail):
            if key in img_lookup:
                match = img_lookup[key]
                break

        if match is None:
            missing_rows.append({"csv_file": str(csv_path), "csv_filename": raw})
            continue

        label_val = None
        if label_col is not None and label_col in df.columns:
            label_val = r[label_col]

        out_rows.append({
            "image_path": str(match),
            "label_id": None if pd.isna(label_val) else int(label_val),
            "source_csv": str(csv_path.name),
        })

    return pd.DataFrame(out_rows), pd.DataFrame(missing_rows), fname_col, label_col

def main():
    project_root = Path(__file__).resolve().parents[1]
    gtsrb_base = project_root / "data" / "GTSRB"
    out_root = project_root / "outputs"
    out_root.mkdir(exist_ok=True)

    images = list_images(gtsrb_base)
    print(f"[GTSRB] images found on disk: {len(images)}")

    csvs = [
        gtsrb_base / "archive" / "Train.csv",
        gtsrb_base / "archive" / "Test.csv",
        gtsrb_base / "archive" / "Test" / "GT-final_test.csv",
    ]
    csvs = [c for c in csvs if c.exists()]
    print(f"[GTSRB] CSVs found: {len(csvs)}")
    for c in csvs:
        print(" -", c)

    labeled_all = []
    missing_all = []

    for c in csvs:
        df_lab, df_miss, fname_col, label_col = label_from_csv(c, gtsrb_base, images)
        print(f"[GTSRB] {c.name}: labeled rows = {len(df_lab)} | missing filename rows = {len(df_miss)} "
              f"| filename_col='{fname_col}' | label_col='{label_col}'")
        labeled_all.append(df_lab)
        missing_all.append(df_miss)

    df_labels = pd.concat(labeled_all, ignore_index=True) if labeled_all else pd.DataFrame(columns=["image_path","label_id","source_csv"])
    df_missing = pd.concat(missing_all, ignore_index=True) if missing_all else pd.DataFrame(columns=["csv_file","csv_filename"])

    # Deduplicate by image_path (some images may appear in multiple CSVs)
    df_labels = df_labels.drop_duplicates(subset=["image_path"], keep="first")

    # Determine unlabeled images on disk (no row in labels file)
    labeled_set = set(df_labels["image_path"].astype(str))
    df_unlabeled = pd.DataFrame(
        [{"image_path": str(p)} for p in images if str(p) not in labeled_set]
    )

    df_labels.to_csv(out_root / "gtsrb_labels.csv", index=False)
    df_unlabeled.to_csv(out_root / "gtsrb_unlabeled.csv", index=False)
    df_missing.to_csv(out_root / "gtsrb_csv_missing_filenames.csv", index=False)

    print("\n[GTSRB] FINAL")
    print(f" - labeled images: {len(df_labels)}")
    print(f" - unlabeled images on disk: {len(df_unlabeled)}")
    print(f" - CSV rows that referenced missing files: {len(df_missing)}")
    print("Wrote:")
    print(" - outputs/gtsrb_labels.csv")
    print(" - outputs/gtsrb_unlabeled.csv")
    print(" - outputs/gtsrb_csv_missing_filenames.csv")

if __name__ == "__main__":
    main()
