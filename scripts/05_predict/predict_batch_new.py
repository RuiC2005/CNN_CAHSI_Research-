from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "outputs/models/stop_vs_yield_cnn_25x25.keras"
CLASS_NAMES = ["stop", "yield"]

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(img_path):
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im = im.resize((25, 25))
        arr = np.array(im, dtype=np.float32)
        return np.expand_dims(arr, axis=0)

test_dir = Path("data/new_test_images_25")

correct = 0
total = 0

for img_path in test_dir.glob("*.jpg"):
    x = preprocess(img_path)
    prob_yield = float(model.predict(x, verbose=0)[0][0])
    pred = "yield" if prob_yield >= 0.5 else "stop"

    print(f"{img_path.name} → {pred} ({prob_yield:.3f})")

    total += 1

print("\nDone. Predicted on", total, "images.")