"""Build a grouped image-classification dataset from a Salla products Excel export.

The Excel file is the source of truth: product categories become labels and the
image URLs in the `صورة المنتج` column become the input images. The script does
not use the store sitemap.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Iterable

from openpyxl import load_workbook
from PIL import Image

IMAGE_URL_RE = re.compile(r"https?://[^\s,;]+", re.IGNORECASE)
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products-xlsx", type=Path, required=True, help="Salla products export")
    parser.add_argument("--output-dir", type=Path, default=Path("data/bellabox"))
    parser.add_argument("--header-row", type=int, default=2, help="1-indexed Excel header row")
    parser.add_argument(
        "--category-level", type=int, choices=(1, 2, 3), default=2,
        help="Number of category path levels used as the label",
    )
    parser.add_argument("--max-products", type=int, default=0, help="0 means all product rows")
    parser.add_argument("--max-images-per-product", type=int, default=3)
    parser.add_argument("--min-images-per-class", type=int, default=20)
    parser.add_argument("--min-products-per-class", type=int, default=4)
    parser.add_argument(
        "--drop-small-classes", action="store_true",
        help="Exclude classes below the minimum and save them in excluded_classes.json",
    )
    parser.add_argument("--download", action="store_true", help="Download image files")
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.set_defaults(download=True)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def scalar_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def fetch_bytes(url: str, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "BellaBox-Product-CNN-Dataset/2.0 (+https://github.com/mohammedalhmed/bellabox-product-cnn)",
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception as error:  # network errors vary by Colab runtime
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Unable to download {url}: {last_error}") from last_error


def find_column(headers: list[str], expected: str) -> int:
    normalized_expected = expected.strip()
    for index, header in enumerate(headers):
        if header.strip() == normalized_expected:
            return index
    raise ValueError(
        f"Required Excel column {expected!r} was not found. Available columns: {headers[:12]}"
    )


def parse_image_urls(value: object) -> list[str]:
    text = html.unescape(scalar_text(value)).replace("\n", ",")
    urls: list[str] = []
    for match in IMAGE_URL_RE.findall(text):
        url = match.rstrip("\"' )]")
        if url and url not in urls:
            urls.append(url)
    return urls


def category_parts(value: object) -> list[str]:
    """Select the deepest valid Salla category path from a category cell."""
    raw_value = html.unescape(scalar_text(value))
    candidates: list[list[str]] = []
    for raw_candidate in raw_value.split(","):
        parts = [part.strip() for part in raw_candidate.split(">") if part.strip()]
        if parts and parts[0] == "كل المنتجات":
            parts = parts[1:]
        if parts:
            candidates.append(parts)
    return max(candidates, key=len) if candidates else []


def read_excel_records(path: Path, header_row: int, category_level: int, max_products: int) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Products Excel file does not exist: {path}")
    if path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError("--products-xlsx must be an .xlsx or .xlsm file")
    if header_row < 1:
        raise ValueError("--header-row is 1-indexed and must be at least 1")

    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = workbook[workbook.sheetnames[0]]
    rows = sheet.iter_rows(values_only=True)
    headers: list[str] | None = None
    for row_number, row in enumerate(rows, start=1):
        if row_number == header_row:
            headers = [scalar_text(value) for value in row]
            break
    if headers is None:
        raise ValueError(f"Excel file does not contain header row {header_row}")

    id_index = find_column(headers, "No.")
    name_index = find_column(headers, "أسم المنتج")
    category_index = find_column(headers, "تصنيف المنتج")
    image_index = find_column(headers, "صورة المنتج")
    records: list[dict[str, object]] = []
    for row_number, row in enumerate(rows, start=header_row + 1):
        values = list(row)
        if not any(value is not None and scalar_text(value) for value in values):
            continue
        product_id = scalar_text(values[id_index] if id_index < len(values) else "")
        product_name = scalar_text(values[name_index] if name_index < len(values) else "")
        category_path = category_parts(values[category_index] if category_index < len(values) else "")
        image_urls = parse_image_urls(values[image_index] if image_index < len(values) else "")
        if not product_id or not product_name or not category_path or not image_urls:
            continue
        label = " > ".join(category_path[:category_level])
        records.append(
            {
                "product_id": product_id,
                "product_name": product_name,
                "category_path": " > ".join(category_path),
                "label": label,
                "image_urls": image_urls,
                "source_row": row_number,
            }
        )
        if max_products > 0 and len(records) >= max_products:
            break
    workbook.close()
    return records


def safe_suffix(url: str) -> str:
    suffix = Path(urllib.parse.urlsplit(url).path).suffix.lower()
    return suffix if suffix in SUPPORTED_SUFFIXES else ".jpg"


def validate_image(path: Path) -> tuple[int, int, str]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    if width < 64 or height < 64:
        raise ValueError(f"image is too small: {width}x{height}")
    return width, height, mode


def download_image(url: str, destination: Path, retries: int) -> tuple[str, int, int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(fetch_bytes(url, retries=retries))
    try:
        width, height, mode = validate_image(destination)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return "ok", width, height, mode


def write_manifest(path: Path, rows: Iterable[dict[str, object]]) -> None:
    fieldnames = [
        "product_id", "product_name", "category_path", "label", "source_row",
        "image_url", "local_path", "download_status", "width", "height", "mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_root = args.output_dir / "images"
    logging.info("Reading Salla products Excel: %s", args.products_xlsx)
    records = read_excel_records(
        args.products_xlsx, args.header_row, args.category_level, args.max_products
    )
    logging.info("Found %s products with categories and image URLs", len(records))

    estimated_images: Counter[str] = Counter()
    estimated_products: Counter[str] = Counter()
    for record in records:
        label = str(record["label"])
        estimated_images[label] += min(len(record["image_urls"]), args.max_images_per_product)
        estimated_products[label] += 1
    estimated_small_classes = {
        label: {"images": estimated_images[label], "products": estimated_products[label]}
        for label in sorted(estimated_images)
        if estimated_images[label] < args.min_images_per_class
        or estimated_products[label] < args.min_products_per_class
    }
    if estimated_small_classes and args.drop_small_classes:
        excluded_labels = set(estimated_small_classes)
        records = [record for record in records if str(record["label"]) not in excluded_labels]
        logging.warning("Excluding small classes: %s", ", ".join(sorted(excluded_labels)))
    elif estimated_small_classes:
        logging.warning(
            "Small classes detected. Use --drop-small-classes to exclude them: %s",
            ", ".join(sorted(estimated_small_classes)),
        )

    rows: list[dict[str, object]] = []
    image_counts: Counter[str] = Counter()
    product_counts: Counter[str] = Counter()
    for index, record in enumerate(records, start=1):
        product_id = str(record["product_id"])
        label = str(record["label"])
        image_urls = list(record["image_urls"])[: args.max_images_per_product]
        successful_images_for_product = 0
        for image_index, image_url in enumerate(image_urls, start=1):
            local_path = image_root / label / f"{product_id}_{image_index}{safe_suffix(str(image_url))}"
            status = "not_downloaded"
            width = height = ""
            mode = ""
            if args.download:
                try:
                    status, width, height, mode = download_image(
                        str(image_url), local_path, retries=args.retries
                    )
                    successful_images_for_product += 1
                except Exception as error:
                    status = f"error:{type(error).__name__}"
                    local_path.unlink(missing_ok=True)
                    logging.warning("Skipping image %s: %s", image_url, error)
            rows.append(
                {
                    "product_id": product_id,
                    "product_name": record["product_name"],
                    "category_path": record["category_path"],
                    "label": label,
                    "source_row": record["source_row"],
                    "image_url": image_url,
                    "local_path": str(local_path),
                    "download_status": status,
                    "width": width,
                    "height": height,
                    "mode": mode,
                }
            )
            if args.download:
                time.sleep(max(args.delay_seconds, 0))
        counted_images = successful_images_for_product if args.download else len(image_urls)
        image_counts[label] += counted_images
        if counted_images > 0:
            product_counts[label] += 1
        if index % 100 == 0 or index == len(records):
            logging.info("Processed %s/%s products", index, len(records))

    manifest_path = args.output_dir / "manifest.csv"
    write_manifest(manifest_path, rows)
    summary = {
        "source_type": "salla_products_xlsx",
        "source_file": args.products_xlsx.name,
        "sheet": "first worksheet",
        "header_row": args.header_row,
        "category_level": args.category_level,
        "products_read": len(records),
        "images_manifested": len(rows),
        "usable_images_by_label": dict(image_counts),
        "products_by_label": dict(product_counts),
        "download_enabled": args.download,
        "min_images_per_class": args.min_images_per_class,
        "min_products_per_class": args.min_products_per_class,
        "excluded_classes": estimated_small_classes if args.drop_small_classes else {},
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("Dataset summary: %s", json.dumps(summary, ensure_ascii=False))

    small_classes = {
        label: {"images": image_counts[label], "products": product_counts[label]}
        for label in sorted(image_counts)
        if image_counts[label] < args.min_images_per_class
        or product_counts[label] < args.min_products_per_class
    }
    if len(image_counts) < 2:
        raise SystemExit("Dataset needs at least two usable classes.")
    if small_classes:
        raise SystemExit(
            "Some classes are below the minimum. Increase data or choose a broader "
            "--category-level. Details: " + json.dumps(small_classes, ensure_ascii=False)
        )
    logging.info(
        "Dataset is ready: %s classes, %s products, %s usable images",
        len(image_counts), sum(product_counts.values()), sum(image_counts.values()),
    )


if __name__ == "__main__":
    main()
