import pandas as pd
from pathlib import Path
from PIL import Image
import shutil

SRC_CSV = Path("outputs/gtsrb_labels_with_names.csv")
OUT_DIR = Path("data/gtsrb_25_binary")

TARGET_SIZE = (25, 25)

CLASSES = {
    "stop": 14,
    "yield": 13
}

def clear_dir(p):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def main():
    clear_dir(OUT_DIR)

    for cls in CLASSES.keys():
        (OUT_DIR / cls).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SRC_CSV)

    # filter only stop and yield
    df = df[df["label_name"].str.lower().isin(CLASSES.keys())]

    print("Total stop/yield rows:", len(df))

    converted = 0

    for _, row in df.iterrows():
        src_path = Path(row["image_path"])
        label = row["label_name"].lower()

        if not src_path.exists():
            continue

        try:
            with Image.open(src_path) as im:
                im = im.convert("RGB")
                im = im.resize(TARGET_SIZE)

                out_name = src_path.stem + ".jpg"
                out_path = OUT_DIR / label / out_name

                im.save(out_path, format="JPEG", quality=95)
                converted += 1

        except Exception as e:
            print("Error:", src_path, e)

    print("Converted:", converted)
    print("Output:", OUT_DIR)

if __name__ == "__main__":
    main()
