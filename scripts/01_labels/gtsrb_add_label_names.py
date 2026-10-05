from pathlib import Path
import pandas as pd

# Standard GTSRB class names (0..42)
GTSRB_LABELS = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

def main():
    project_root = Path(__file__).resolve().parents[1]
    labels_path = project_root / "outputs" / "gtsrb_labels.csv"
    out_path = project_root / "outputs" / "gtsrb_labels_with_names.csv"

    df = pd.read_csv(labels_path)

    if "label_id" not in df.columns:
        raise ValueError("Expected column 'label_id' in outputs/gtsrb_labels.csv")

    df["label_name"] = df["label_id"].map(GTSRB_LABELS)

    # Sanity check: any unmapped IDs?
    missing = df["label_name"].isna().sum()
    if missing:
        bad_ids = sorted(df.loc[df["label_name"].isna(), "label_id"].unique().tolist())
        print("[GTSRB] WARNING: Missing label names for IDs:", bad_ids)

    df.to_csv(out_path, index=False)
    print(f"[GTSRB] Wrote: {out_path}")
    print(f"[GTSRB] Rows: {len(df)} | Missing label_name: {missing}")

if __name__ == "__main__":
    main()
