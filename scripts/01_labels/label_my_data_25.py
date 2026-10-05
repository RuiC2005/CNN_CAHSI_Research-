from pathlib import Path
import csv

INPUT_DIR = Path("data/my_data_25")
CSV_PATH = Path("outputs/my_data_labels.csv")

rows = []

print("Label images: type stop, yield, or other\n")

for img_path in sorted(INPUT_DIR.glob("*.jpg")):
    while True:
        label = input(f"{img_path.name} -> ").strip().lower()
        if label in {"stop", "yield", "other"}:
            rows.append({
                "image_path": str(img_path),
                "label_name": label
            })
            break
        print("Type stop, yield, or other only.")

with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "label_name"])
    writer.writeheader()
    writer.writerows(rows)

print("\nDone.")
print("Wrote:", CSV_PATH)
