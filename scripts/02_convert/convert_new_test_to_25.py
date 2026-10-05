# scripts/convert_new_test_to_25.py

from pathlib import Path
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

INPUT = Path("data/new_test_images")
OUTPUT = Path("data/new_test_images_25")
OUTPUT.mkdir(parents=True, exist_ok=True)

for img_path in INPUT.glob("*"):
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = im.resize((25, 25))
            out_path = OUTPUT / (img_path.stem + ".jpg")
            im.save(out_path, format="JPEG")
            print("Converted:", out_path.name)
    except Exception as e:
        print("Skipped:", img_path.name)