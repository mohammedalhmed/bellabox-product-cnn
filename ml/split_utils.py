"""Pure-Python grouped split utilities for BellaBox training."""
from __future__ import annotations

import random
from collections import defaultdict


def grouped_stratified_split_strict(
    rows: list[dict[str, str]],
    val_size: float,
    test_size: float,
    *,
    seed: int,
) -> dict[str, list[dict[str, str]]]:
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be > 0 and sum to less than 1")

    product_labels: dict[str, str] = {}
    for row in rows:
        product_id = row["product_id"]
        label = row["label"]
        previous = product_labels.get(product_id)
        if previous is not None and previous != label:
            raise ValueError(
                f"Product {product_id!r} appears under conflicting labels: {previous!r} and {label!r}"
            )
        product_labels[product_id] = label

    products_by_label: dict[str, list[str]] = defaultdict(list)
    for product_id, label in product_labels.items():
        products_by_label[label].append(product_id)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}

    for label in sorted(products_by_label):
        product_ids = sorted(products_by_label[label])
        if len(product_ids) < 3:
            raise ValueError(
                f"Class {label!r} needs at least 3 distinct products so train, validation, "
                "and test each contain the class."
            )

        rng.shuffle(product_ids)
        count = len(product_ids)
        n_val = max(1, round(count * val_size))
        n_test = max(1, round(count * test_size))

        # Always preserve at least one training product.
        while n_val + n_test > count - 1:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                raise ValueError(f"Unable to split class {label!r} safely")

        test_ids = product_ids[:n_test]
        val_ids = product_ids[n_test : n_test + n_val]
        train_ids = product_ids[n_test + n_val :]

        for product_id in train_ids:
            assignment[product_id] = "train"
        for product_id in val_ids:
            assignment[product_id] = "validation"
        for product_id in test_ids:
            assignment[product_id] = "test"

    split_rows: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for row in rows:
        split_rows[assignment[row["product_id"]]].append(row)

    for label in products_by_label:
        for split_name, split in split_rows.items():
            if not any(row["label"] == label for row in split):
                raise AssertionError(f"{label!r} missing from {split_name} split")

    return split_rows
