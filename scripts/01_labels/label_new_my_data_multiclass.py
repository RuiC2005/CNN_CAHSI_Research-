from pathlib import Path
import csv

IMAGES_DIR = Path("data/my_data_25")
CSV_PATH = Path("outputs/my_data_labels.csv")

TARGET_LABELS = [
    "stop",
    "yield",
    "no entry",
    "speed limit (30km/h)",
    "speed limit (50km/h)",
    "priority road",
]

VALID = set(TARGET_LABELS) | {"skip"}

def load_existing_paths():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, "r", newline="") as f:
        r = csv.DictReader(f)
        return set(row["image_path"] for row in r)

def append_rows(rows):
    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label_name"])
        if not file_exists:
            w.writeheader()
        w.writerows(rows)

def main():
    existing = load_existing_paths()
    imgs = sorted(IMAGES_DIR.glob("*.jpg"))
    new_imgs = [p for p in imgs if str(p) not in existing]

    print("Allowed labels:")
    for t in TARGET_LABELS:
        print(" -", t)
    print("\nNEW images to label:", len(new_imgs))

    rows = []
    for p in new_imgs:
        while True:
            label = input(f"{p.name} -> label (or 'skip'): ").strip().lower()
            if label in VALID:
                break
            print("Invalid label. Use one of TARGET_LABELS exactly, or 'skip'.")
        if label == "skip":
            continue
        rows.append({"image_path": str(p), "label_name": label})

    if rows:
        append_rows(rows)

    print("Added:", len(rows))
    print("CSV:", CSV_PATH)

if __name__ == "__main__":
    main()