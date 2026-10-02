import unittest

from ml.split_utils import grouped_stratified_split_strict


def rows_for(label: str, products: int, images_per_product: int = 2):
    rows = []
    for product_index in range(products):
        product_id = f"{label}-{product_index}"
        for image_index in range(images_per_product):
            rows.append(
                {
                    "product_id": product_id,
                    "label": label,
                    "local_path": f"/tmp/{product_id}-{image_index}.jpg",
                }
            )
    return rows


class GroupedSplitTests(unittest.TestCase):
    def test_four_products_per_class_are_present_in_every_split(self):
        rows = rows_for("makeup", 4) + rows_for("skincare", 4)
        split = grouped_stratified_split_strict(rows, 0.15, 0.15, seed=42)

        for label in ("makeup", "skincare"):
            for part in ("train", "validation", "test"):
                self.assertTrue(any(row["label"] == label for row in split[part]))

        product_sets = {
            part: {row["product_id"] for row in part_rows}
            for part, part_rows in split.items()
        }
        self.assertTrue(product_sets["train"].isdisjoint(product_sets["validation"]))
        self.assertTrue(product_sets["train"].isdisjoint(product_sets["test"]))
        self.assertTrue(product_sets["validation"].isdisjoint(product_sets["test"]))

    def test_conflicting_product_labels_are_rejected(self):
        rows = [
            {"product_id": "1", "label": "a", "local_path": "a.jpg"},
            {"product_id": "1", "label": "b", "local_path": "b.jpg"},
        ]
        with self.assertRaises(ValueError):
            grouped_stratified_split_strict(rows, 0.15, 0.15, seed=42)

    def test_invalid_split_sizes_are_rejected(self):
        with self.assertRaises(ValueError):
            grouped_stratified_split_strict(rows_for("a", 4), 0.6, 0.5, seed=42)


if __name__ == "__main__":
    unittest.main()
