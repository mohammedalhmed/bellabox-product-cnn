"""Train a product-category CNN from BellaBox's reviewed manifest."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("ml_outputs/bellabox"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--fine-tune-layers", type=int, default=40)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def read_manifest(data_dir: Path) -> list[dict[str, str]]:
    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}. Run build_dataset.py first.")
    rows: list[dict[str, str]] = []
    with manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = Path(row["local_path"])
            if row["download_status"] == "ok" and path.exists():
                rows.append(row)
    if not rows:
        raise ValueError("No usable downloaded images found in manifest.csv.")
    return rows


def grouped_stratified_split(rows: list[dict[str, str]], val_size: float, test_size: float):
    groups: dict[str, str] = {}
    for row in rows:
        groups.setdefault(row["product_id"], row["label"])
    group_ids = np.array(sorted(groups))
    group_labels = np.array([groups[group_id] for group_id in group_ids])
    if len(group_ids) < 6:
        raise ValueError("Need at least 6 distinct products for grouped train/val/test split.")

    try:
        train_groups, temp_groups = train_test_split(
            group_ids,
            test_size=val_size + test_size,
            random_state=SEED,
            stratify=group_labels,
        )
        temp_labels = np.array([groups[group_id] for group_id in temp_groups])
        relative_test = test_size / (val_size + test_size)
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=relative_test,
            random_state=SEED,
            stratify=temp_labels,
        )
    except ValueError as error:
        logging.warning("Stratified group split fallback: %s", error)
        rng = np.random.default_rng(SEED)
        shuffled = group_ids.copy()
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_size))
        n_val = max(1, round(len(shuffled) * val_size))
        test_groups = shuffled[:n_test]
        val_groups = shuffled[n_test : n_test + n_val]
        train_groups = shuffled[n_test + n_val :]

    split_by_group = {
        group_id: "train" for group_id in train_groups
    } | {group_id: "validation" for group_id in val_groups} | {
        group_id: "test" for group_id in test_groups
    }
    split_rows = {"train": [], "validation": [], "test": []}
    for row in rows:
        split_rows[split_by_group[row["product_id"]]].append(row)

    if not split_rows["train"] or not split_rows["validation"] or not split_rows["test"]:
        raise ValueError("Grouped split produced an empty partition; add more products.")
    train_ids = {row["product_id"] for row in split_rows["train"]}
    val_ids = {row["product_id"] for row in split_rows["validation"]}
    test_ids = {row["product_id"] for row in split_rows["test"]}
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    return split_rows


def make_dataset(rows: list[dict[str, str]], label_to_id: dict[str, int], training: bool, batch_size: int):
    paths = np.array([row["local_path"] for row in rows])
    labels = np.array([label_to_id[row["label"]] for row in rows], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(rows), seed=SEED, reshuffle_each_iteration=True)

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize_with_pad(image, IMAGE_SIZE[0], IMAGE_SIZE[1])
        image = tf.cast(image, tf.float32)
        return image, label

    return dataset.map(load_image, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)


def build_model(num_classes: int, learning_rate: float):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="product_image")
    x = tf.keras.layers.RandomFlip("horizontal")(inputs)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.RandomZoom(0.12)(x)
    x = tf.keras.layers.RandomContrast(0.1)(x)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="class_probabilities"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="bellabox_product_cnn")
    compile_model(model, learning_rate)
    return model, base


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.08),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


class MacroF1Callback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, validation_labels: np.ndarray):
        super().__init__()
        self.validation_data = validation_data
        self.validation_labels = validation_labels

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        predictions = self.model.predict(self.validation_data, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)
        logs["val_macro_f1"] = float(
            f1_score(self.validation_labels, predicted_labels, average="macro", zero_division=0)
        )
        logging.info("epoch=%s val_macro_f1=%.4f", epoch + 1, logs["val_macro_f1"])


def make_callbacks(output_dir: Path, validation_data, validation_labels: np.ndarray, patience: int, backup_dir: Path):
    return [
        MacroF1Callback(validation_data, validation_labels),
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "best.keras",
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_dir / "last.keras",
            save_best_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1", mode="max", factor=0.3, patience=max(1, patience // 2), min_lr=1e-7
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=str(backup_dir)),
        tf.keras.callbacks.CSVLogger(output_dir / "training_log.csv", append=True),
    ]


def plot_history(histories: list[dict[str, list[float]]], output_dir: Path) -> None:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.items():
            merged.setdefault(key, []).extend(values)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(merged.get("loss", []), label="train")
    axes[0].plot(merged.get("val_loss", []), label="validation")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(merged.get("accuracy", []), label="train")
    axes[1].plot(merged.get("val_accuracy", []), label="validation")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(figure)


def evaluate(model, test_data, test_rows, labels, output_dir: Path) -> dict:
    true = np.array([labels.index(row["label"]) for row in test_rows])
    probabilities = model.predict(test_data, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, predicted, average="macro", zero_division=0
    )
    report = classification_report(true, predicted, target_names=labels, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    matrix = confusion_matrix(true, predicted, labels=list(range(len(labels))))
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=list(range(len(labels))), yticks=list(range(len(labels))),
        xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True",
        title="BellaBox Product Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, matrix[row_index, column_index], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    metrics = {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "test_images": len(test_rows),
        "test_products": len({row["product_id"] for row in test_rows}),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    set_seed()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.data_dir)
    split_rows = grouped_stratified_split(rows, args.val_size, args.test_size)
    labels = sorted({row["label"] for row in rows})
    label_to_id = {label: index for index, label in enumerate(labels)}
    for split_name, split in split_rows.items():
        logging.info(
            "%s: %s images, %s products, class counts=%s",
            split_name, len(split), len({row["product_id"] for row in split}), Counter(row["label"] for row in split),
        )
    (args.output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "split_summary.json").write_text(
        json.dumps(
            {
                split: {
                    "images": len(rows_for_split),
                    "products": len({row["product_id"] for row in rows_for_split}),
                    "class_counts": dict(Counter(row["label"] for row in rows_for_split)),
                }
                for split, rows_for_split in split_rows.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    train_data = make_dataset(split_rows["train"], label_to_id, True, args.batch_size)
    validation_data = make_dataset(split_rows["validation"], label_to_id, False, args.batch_size)
    test_data = make_dataset(split_rows["test"], label_to_id, False, args.batch_size)
    validation_labels = np.array([label_to_id[row["label"]] for row in split_rows["validation"]])
    train_labels = np.array([label_to_id[row["label"]] for row in split_rows["train"]])
    weights = compute_class_weight("balanced", classes=np.arange(len(labels)), y=train_labels)
    class_weights = {index: float(weight) for index, weight in enumerate(weights)}
    (args.output_dir / "class_weights.json").write_text(json.dumps(class_weights, indent=2), encoding="utf-8")

    model, backbone = build_model(len(labels), args.learning_rate)
    stage1_done = args.output_dir / "stage1_done.json"
    histories: list[dict[str, list[float]]] = []
    if args.resume and stage1_done.exists() and (args.output_dir / "best.keras").exists():
        logging.info("Stage 1 marker found; loading best checkpoint and continuing.")
        model = tf.keras.models.load_model(args.output_dir / "best.keras")
        backbone = next(layer for layer in model.layers if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower())
    else:
        stage1_callbacks = make_callbacks(
            args.output_dir, validation_data, validation_labels, args.patience,
            args.output_dir / "backup_stage1",
        )
        history = model.fit(
            train_data, validation_data=validation_data, epochs=max(3, args.epochs // 2),
            class_weight=class_weights, callbacks=stage1_callbacks, verbose=1,
        )
        histories.append(history.history)
        model.save(args.output_dir / "stage1_last.keras")
        stage1_done.write_text(json.dumps({"completed": True}), encoding="utf-8")

    for layer in backbone.layers[:-args.fine_tune_layers]:
        layer.trainable = False
    for layer in backbone.layers[-args.fine_tune_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    compile_model(model, args.fine_tune_learning_rate)
    stage2_callbacks = make_callbacks(
        args.output_dir, validation_data, validation_labels, args.patience,
        args.output_dir / "backup_stage2",
    )
    remaining_epochs = max(1, args.epochs - max(3, args.epochs // 2))
    history = model.fit(
        train_data, validation_data=validation_data, epochs=remaining_epochs,
        class_weight=class_weights, callbacks=stage2_callbacks, verbose=1,
    )
    histories.append(history.history)
    best_path = args.output_dir / "best.keras"
    if best_path.exists():
        model = tf.keras.models.load_model(best_path)
    model.save(args.output_dir / "final_model.keras")
    plot_history(histories, args.output_dir)
    metrics = evaluate(model, test_data, split_rows["test"], labels, args.output_dir)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({"seed": SEED, "image_size": IMAGE_SIZE, **vars(args)}, default=str, indent=2),
        encoding="utf-8",
    )
    logging.info("Final test metrics: %s", metrics)
    logging.info("Saved model and reports to %s", args.output_dir)


if __name__ == "__main__":
    main()
