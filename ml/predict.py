"""Predict a BellaBox product category from one image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

IMAGE_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    image = tf.io.read_file(str(path))
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize_with_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1])
    return image.numpy()[None, ...]


def main() -> None:
    args = parse_args()
    model = tf.keras.models.load_model(args.model)
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    probabilities = model.predict(load_image(args.image), verbose=0)[0]
    top_k = min(max(args.top_k, 1), len(labels))
    indices = np.argsort(probabilities)[::-1][:top_k]
    result = {
        "image": str(args.image),
        "predictions": [
            {"label": labels[int(index)], "probability": round(float(probabilities[index]), 6)}
            for index in indices
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
