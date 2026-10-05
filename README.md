# Traffic Sign Audit

CNN-based traffic sign classification research, conducted through CAHSI with Dr. Nuri Yilmazer (Texas A&M University-Kingsville) and presented as a poster at GMiS 2026 (Albuquerque).

The project audits and labels images from three sources (GTSRB, the LISA Traffic Sign Dataset, and custom real-world phone photos), converts them to 25x25 RGB, and trains small Keras CNNs for two tasks:

- **Binary:** stop vs. yield
- **Multiclass:** all 43 GTSRB sign classes (99.2% validation accuracy)

## Project structure

```
traffic_sign_audit/
├── data/                          # datasets (not tracked)
│   ├── GTSRB/                     # raw GTSRB download
│   ├── LISA/                      # raw LISA download
│   ├── my_data/images/            # custom photos (HEIC/JPG/PNG)
│   ├── my_data_25/                # custom photos resized to 25x25
│   ├── gtsrb_25_binary/           # GTSRB stop/yield at 25x25
│   ├── gtsrb_25_balanced/         # class-balanced stop/yield
│   ├── processed/gtsrb_25_multiclass/
│   ├── processed_25_binary/       # train/val/test for binary model
│   ├── processed_25_multiclass/   # train/val/test for multiclass model
│   ├── new_test_images/           # unseen images for prediction
│   └── new_test_images_25/
├── outputs/
│   ├── labels/                    # label CSVs per dataset
│   ├── reports/                   # image audit CSVs (size, mode, channels)
│   └── models/                    # trained .keras models + training plots
├── outputs_backup/                # snapshot of an earlier outputs/ folder
├── scripts/
│   ├── paths.py                   # shared project paths
│   ├── 00_scan/                   # audit image files
│   ├── 01_labels/                 # build / add labels
│   ├── 02_convert/                # resize to 25x25 RGB JPEG
│   ├── 03_build/                  # balance and split into train/val/test
│   ├── 04_train/                  # train models
│   └── 05_predict/                # run inference
└── venv/
```

## Setup

Requires Python 3.13.

```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow pandas numpy pillow pillow-heif matplotlib
```

Versions used: TensorFlow 2.20, Keras 3.13, pandas 3.0, NumPy 2.4, Pillow 12.1, pillow-heif 1.2, Matplotlib 3.10.

Download the datasets into `data/`:

- **GTSRB:** German Traffic Sign Recognition Benchmark, into `data/GTSRB/`
- **LISA:** LISA Traffic Sign Dataset, into `data/LISA/`
- **Custom photos:** into `data/my_data/images/` (HEIC is supported)

## Pipeline

Run every script **from the project root**, since most scripts use paths relative to it (for example `data/my_data_25`).

### 0. Audit images

```bash
python scripts/00_scan/scan_images.py
```

Scans GTSRB, LISA, and custom data, reporting width, height, color mode, channel count, and unreadable files. Writes `*_image_audit.csv` reports.

### 1. Labels

```bash
python scripts/01_labels/labels_gtsrb.py            # match GTSRB CSV annotations to image files
python scripts/01_labels/gtsrb_add_label_names.py   # map ClassId 0-42 to sign names
python scripts/01_labels/labels_lisa.py             # parse LISA frameAnnotationsBOX.csv
```

Custom photos are labeled interactively in the terminal (run after step 2 creates `data/my_data_25/`):

```bash
python scripts/01_labels/label_my_data_25.py               # stop / yield / other
python scripts/01_labels/label_new_my_data_25.py           # label only new images
python scripts/01_labels/label_new_my_data_multiclass.py   # multiclass label set
```

### 2. Convert to 25x25

```bash
python scripts/02_convert/convert_all_to_25.py             # custom photos -> data/my_data_25
python scripts/02_convert/convert_gtsrb_stop_yield_25.py   # GTSRB stop (14) / yield (13)
python scripts/02_convert/convert_gtsrb_multiclass_25.py   # all 43 GTSRB classes
python scripts/02_convert/convert_new_test_to_25.py        # unseen test images
```

All images are converted to 3-channel RGB and saved as JPEG.

### 3. Build datasets

Binary (stop vs. yield):

```bash
python scripts/03_build/balance_gtsrb_binary.py           # undersample to equal class sizes
python scripts/03_build/build_binary_dataset_combined.py  # GTSRB + custom, 80/10/10 split
```

Multiclass:

```bash
python scripts/03_build/build_multiclass_dataset_combined.py   # 70/15/15 split
```

Splits use `random.seed(42)` for reproducibility.

### 4. Train

```bash
python scripts/04_train/train_binary_cnn_25.py       # -> outputs/models/stop_vs_yield_cnn_25x25.keras
python scripts/04_train/train_multiclass_cnn_25.py   # -> outputs/models/gtsrb_multiclass_cnn_25x25.keras
```

The multiclass script also saves `accuracy_plot.png` and `loss_plot.png` to `outputs/models/`.

`train_3class_mlp.py` is an earlier fully connected baseline kept for comparison.

### 5. Predict

```bash
python scripts/05_predict/predict_one.py path/to/image.jpg
python scripts/05_predict/predict_batch_new.py   # runs on data/new_test_images_25/
```

Both use the binary stop vs. yield model and print the predicted class with the sigmoid probability of "yield".

## Model architecture

Both CNNs share the same backbone, sized for 25x25x3 input:

| Layer | Details |
|---|---|
| Rescaling | 1/255 |
| Conv2D + MaxPool | 16 filters, 3x3, ReLU |
| Conv2D + MaxPool | 32 filters, 3x3, ReLU |
| Conv2D + MaxPool | 64 filters, 3x3, ReLU |
| Flatten + Dense | 64, ReLU |
| Dropout | 0.3 |
| Output | 1 sigmoid (binary) or 43 softmax (multiclass) |

Training: Adam (lr 1e-3), batch size 64, up to 30 epochs, early stopping on validation loss (patience 5, best weights restored).

## Results

| Model | Classes | Validation accuracy |
|---|---|---|
| Multiclass CNN | 43 | 99.2% |

## Known issues

- `scan_images.py` and `labels_lisa.py` locate the project root with `Path(__file__).resolve().parents[1]`. Since the scripts now live one folder deeper, this resolves to `scripts/` instead of the project root. Use `parents[2]` or import `ROOT` from `scripts/paths.py`.
- `build_multiclass_dataset_combined.py` defines `GTSRB_ROOT` twice; the second value (`data/processed/gtsrb_25_multiclass`) is the one in effect.
- Some scripts write label CSVs to `outputs/` while others read from `outputs/labels/`. Check the path at the top of a script if it reports a missing file.

## Acknowledgments

Research supported by CAHSI, with mentorship from Dr. Nuri Yilmazer (Texas A&M University-Kingsville). Datasets: GTSRB (Stallkamp et al.) and the LISA Traffic Sign Dataset (Møgelmose et al.).
