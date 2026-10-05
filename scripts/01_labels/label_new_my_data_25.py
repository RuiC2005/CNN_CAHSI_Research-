from pathlib import Path
import csv

IMAGES_DIR = Path("data/my_data_25")
CSV_PATH = Path("outputs/my_data_labels.csv")

VALID = {"stop", "yield", "other", "skip"}

def load_existing_paths():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.DictReader(f)
        return set(row["image_path"] for row in reader if "image_path" in row)

def append_rows(rows):
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label_name"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

def main():
    existing = load_existing_paths()
    all_imgs = sorted(IMAGES_DIR.glob("*.jpg"))

    new_imgs = [p for p in all_imgs if str(p) not in existing]

    print(f"Total processed images: {len(all_imgs)}")
    print(f"Already labeled in CSV: {len(existing)}")
    print(f"NEW images to label: {len(new_imgs)}\n")

    if not new_imgs:
        print("Nothing new to label. You're done.")
        return

    rows = []
    for p in new_imgs:
        while True:
            label = input(f"{p.name} -> label (stop/yield/other/skip): ").strip().lower()
            if label in VALID:
                break
            print("Please type: stop, yield, other, or skip")

        if label == "skip":
            continue

        rows.append({"image_path": str(p), "label_name": label})

    if rows:
        append_rows(rows)

    print("\nDONE")
    print(f"Added rows: {len(rows)}")
    print(f"Wrote/updated: {CSV_PATH}")

if __name__ == "__main__":
    main()
