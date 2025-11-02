
"""
Usage:
    python predict.py path/to/image.jpg
"""

import sys
import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

MODEL_PATH = "keras_Model.h5"   # your exported model file
LABELS_PATH = "labels.txt"      # one label per line
IMAGE_SIZE = (224, 224)         # adjust to model input (width, height)
NORMALIZE_TYPE = "neg1_to_1"    # options: "0_to_1" or "neg1_to_1"

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def preprocess_image(img_path, target_size=IMAGE_SIZE, normalize=NORMALIZE_TYPE):
    img = Image.open(img_path).convert("RGB")
    img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32)
    if normalize == "0_to_1":
        arr = arr / 255.0
    elif normalize == "neg1_to_1":
        arr = (arr / 127.5) - 1.0
    else:
        raise ValueError("Unsupported normalize type")
    return np.expand_dims(arr, axis=0)  # batch dimension

def predict(image_path):
    # load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # load labels
    labels = load_labels(LABELS_PATH)

    # preprocess
    x = preprocess_image(image_path)

    # inference
    probs = model.predict(x)[0]   # shape (num_classes,)
    top_idx = int(np.argmax(probs))
    top_label = labels[top_idx] if top_idx < len(labels) else str(top_idx)
    top_prob = float(probs[top_idx])

    # full sorted results
    sorted_results = sorted(
        [(labels[i] if i < len(labels) else str(i), float(p)) for i, p in enumerate(probs)],
        key=lambda t: t[1], reverse=True
    )

    return top_label, top_prob, sorted_results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(1)
    image_path = sys.argv[1]
    label, prob, all_scores = predict(image_path)
    print(f"Top: {label} ({prob:.4f})")
    print("All scores:")
    for l, p in all_scores:
        print(f"  {l}: {p:.4f}")
