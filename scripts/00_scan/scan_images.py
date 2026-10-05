from pathlib import Path
import pandas as pd
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".tif", ".tiff"}

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS]

def scan_images(image_paths):
    rows = []
    bad = 0
    for p in image_paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
                mode = im.mode  # 'RGB', 'L', 'RGBA', ...
                channels = {
                    "1": 1, "L": 1, "P": 1,
                    "RGB": 3, "YCbCr": 3,
                    "RGBA": 4, "CMYK": 4
                }.get(mode, None)

                rows.append({
                    "image_path": str(p),
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "channels": channels,
                    "suffix": p.suffix.lower(),
                })
        except Exception:
            bad += 1

    df = pd.DataFrame(rows)
    return df, bad

def summarize(df: pd.DataFrame, bad_count: int, name: str):
    if df.empty:
        print(f"[{name}] No images found.")
        return
    print(f"\n[{name}] images scanned: {len(df)} | bad/unreadable: {bad_count}")
    print(f"[{name}] width  min/med/max : {df['width'].min()} / {int(df['width'].median())} / {df['width'].max()}")
    print(f"[{name}] height min/med/max : {df['height'].min()} / {int(df['height'].median())} / {df['height'].max()}")
    print(f"[{name}] modes: {df['mode'].value_counts().to_dict()}")
    print(f"[{name}] channels: {df['channels'].value_counts(dropna=False).to_dict()}")

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    out_root = project_root / "outputs"
    out_root.mkdir(exist_ok=True)

    # Scan GTSRB
    gtsrb_root = data_root / "GTSRB"
    gtsrb_imgs = list_images(gtsrb_root)
    df_g, bad_g = scan_images(gtsrb_imgs)
    summarize(df_g, bad_g, "GTSRB")
    df_g.to_csv(out_root / "gtsrb_image_audit.csv", index=False)

    # Scan LISA
    lisa_root = data_root / "LISA"
    lisa_imgs = list_images(lisa_root)
    df_l, bad_l = scan_images(lisa_imgs)
    summarize(df_l, bad_l, "LISA")
    df_l.to_csv(out_root / "lisa_image_audit.csv", index=False)

    print("\nWrote:")
    print(" - outputs/gtsrb_image_audit.csv")
    print(" - outputs/lisa_image_audit.csv")

    # Scan MY_DATA
    my_data_root = data_root / "my_data"
    my_data_imgs = list_images(my_data_root)
    df_m, bad_m = scan_images(my_data_imgs)
    summarize(df_m, bad_m, "MY_DATA")
    df_m.to_csv(out_root / "my_data_image_audit.csv", index=False)

if __name__ == "__main__":
    main()


