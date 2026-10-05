import sys
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "outputs/models/stop_vs_yield_cnn_25x25.keras"
IMG_SIZE = (25, 25)

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess(p: Path):
    with Image.open(p) as im:
        im = im.convert("RGB").resize(IMG_SIZE)
        arr = np.array(im, dtype=np.float32)   # keep 0..255
        return np.expand_dims(arr, axis=0)     # (1,25,25,3)

p = Path(sys.argv[1])
x = preprocess(p)

prob_yield = float(model.predict(x, verbose=0)[0][0])  # sigmoid
pred = "yield" if prob_yield >= 0.5 else "stop"

print("pred:", pred, "prob_yield:", prob_yield)