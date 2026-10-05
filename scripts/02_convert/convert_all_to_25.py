from pathlib import Path
from PIL import Image
import pillow_heif

# Enable HEIC support
pillow_heif.register_heif_opener()

INPUT_DIR = Path("data/my_data/images")
OUTPUT_DIR = Path("data/my_data_25")

TARGET_SIZE = (25, 25)
VALID_EXTS = {".heic", ".jpg", ".jpeg", ".png"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

count = 0
skipped = 0

for img_path in sorted(INPUT_DIR.iterdir()):
    if not img_path.is_file():
        continue
    if img_path.suffix.lower() not in VALID_EXTS:
        skipped += 1
        continue

    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")      # ensures 3 channels
            im = im.resize(TARGET_SIZE)
            
            out_name = img_path.stem + ".jpg"
            out_path = OUTPUT_DIR / out_name
            
            im.save(out_path, format="JPEG", quality=95, optimize=True)
            count += 1

    except Exception as e:
        print("Error processing", img_path.name, ":", e)

print("\nDone.")
print("Converted:", count)
print("Skipped:", skipped)
print("Output folder:", OUTPUT_DIR)
