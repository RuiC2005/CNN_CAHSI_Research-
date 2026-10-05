from pathlib import Path
import random
import shutil

random.seed(42)

ROOT = Path("data/gtsrb_25_binary")
STOP_DIR = ROOT / "stop"
YIELD_DIR = ROOT / "yield"

BALANCED_ROOT = Path("data/gtsrb_25_balanced")

def clear_dir(p):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def main():
    clear_dir(BALANCED_ROOT)
    (BALANCED_ROOT / "stop").mkdir(parents=True)
    (BALANCED_ROOT / "yield").mkdir(parents=True)

    stop_imgs = list(STOP_DIR.glob("*.jpg"))
    yield_imgs = list(YIELD_DIR.glob("*.jpg"))

    n = min(len(stop_imgs), len(yield_imgs))

    stop_sample = random.sample(stop_imgs, n)
    yield_sample = random.sample(yield_imgs, n)

    for img in stop_sample:
        shutil.copy(img, BALANCED_ROOT / "stop")

    for img in yield_sample:
        shutil.copy(img, BALANCED_ROOT / "yield")

    print("Balanced dataset created:")
    print("Stop:", n)
    print("Yield:", n)
    print("Location:", BALANCED_ROOT)

if __name__ == "__main__":
    main()
