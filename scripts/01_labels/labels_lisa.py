from pathlib import Path
import pandas as pd

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

def normalize_path(s: str):
    return str(s).replace("\\", "/").lstrip("./")

def read_box_csv(csv_path: Path):
    """
    LISA frameAnnotationsBOX.csv is often semicolon-delimited.
    Try comma first, then fallback to semicolon if it collapses to 1 column.
    """
    try:
        df = pd.read_csv(csv_path)
        if df.shape[1] <= 1:
            df = pd.read_csv(csv_path, sep=";")
    except Exception:
        df = pd.read_csv(csv_path, sep=";")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def pick_column(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def main():
    project_root = Path(__file__).resolve().parents[1]
    lisa_root = project_root / "data" / "LISA"
    out_root = project_root / "outputs"
    out_root.mkdir(exist_ok=True)

    images = list_images(lisa_root)
    print(f"[LISA] images found on disk: {len(images)}")

    box_csvs = list(lisa_root.rglob("frameAnnotationsBOX.csv"))
    print(f"[LISA] BOX annotation files found: {len(box_csvs)}")
    if not box_csvs:
        print("[LISA] ERROR: No frameAnnotationsBOX.csv files found.")
        return

    # Tail filename -> list of matching paths (rare duplicates possible)
    tail_to_paths = {}
    for p in images:
        tail_to_paths.setdefault(p.name, []).append(p)

    all_rows = []
    missing_ann_files = 0
    parsed_files = 0

    for csv_path in box_csvs:
        df = read_box_csv(csv_path)
        cols = set(df.columns)

        # LISA actual column names (as seen in your output)
        fname_col = pick_column(cols, ["filename", "file", "image", "img", "path"])
        label_col = pick_column(cols, ["annotation tag", "annotation", "label", "signtype", "sign", "type", "class"])

        x1_col = pick_column(cols, ["upper left corner x", "x1", "xmin", "left"])
        y1_col = pick_column(cols, ["upper left corner y", "y1", "ymin", "top"])
        x2_col = pick_column(cols, ["lower right corner x", "x2", "xmax", "right"])
        y2_col = pick_column(cols, ["lower right corner y", "y2", "ymax", "bottom"])

        if not all([fname_col, label_col, x1_col, y1_col, x2_col, y2_col]):
            print(f"[LISA] WARNING: Still unexpected columns in {csv_path}")
            print("Columns:", list(df.columns))
            continue

        parsed_files += 1

        for _, r in df.iterrows():
            raw_name = normalize_path(r[fname_col])
            tail = raw_name.split("/")[-1]

            candidates = tail_to_paths.get(tail, [])
            if not candidates:
                missing_ann_files += 1
                continue

            img_path = candidates[0]

            all_rows.append({
                "image_path": str(img_path),
                "label_name": str(r[label_col]),
                "xmin": float(r[x1_col]),
                "ymin": float(r[y1_col]),
                "xmax": float(r[x2_col]),
                "ymax": float(r[y2_col]),
                "source_csv": str(csv_path),
            })

    df_boxes = pd.DataFrame(all_rows)

    if not df_boxes.empty:
        grouped = (
            df_boxes.groupby("image_path")["label_name"]
            .apply(lambda s: "|".join(sorted(set(map(str, s)))))
            .reset_index()
            .rename(columns={"label_name": "labels_in_image"})
        )
    else:
        grouped = pd.DataFrame(columns=["image_path", "labels_in_image"])

    labeled_set = set(grouped["image_path"].astype(str))
    df_unlabeled = pd.DataFrame([{"image_path": str(p)} for p in images if str(p) not in labeled_set])

    df_boxes.to_csv(out_root / "lisa_boxes.csv", index=False)
    grouped.to_csv(out_root / "lisa_image_labels.csv", index=False)
    df_unlabeled.to_csv(out_root / "lisa_unlabeled.csv", index=False)

    print("\n[LISA] FINAL")
    print(f" - annotation BOX files parsed: {parsed_files}/{len(box_csvs)}")
    print(f" - sign instances (boxes): {len(df_boxes)}")
    print(f" - images with >=1 label: {len(grouped)}")
    print(f" - images unlabeled (no box rows): {len(df_unlabeled)}")
    print(f" - annotation rows with missing image match: {missing_ann_files}")
    print("Wrote:")
    print(" - outputs/lisa_boxes.csv")
    print(" - outputs/lisa_image_labels.csv")
    print(" - outputs/lisa_unlabeled.csv")

if __name__ == "__main__":
    main()
