"""Build a grouped image-classification dataset from BellaBox's public sitemap.

The script intentionally uses the store's own product sitemap instead of a
teaching dataset. It records every decision in manifest.csv so labels can be
reviewed before training.
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
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image

BASE_URL = "https://bellaboxksa.com"
SITEMAP_URL = f"{BASE_URL}/sitemap.xml"
SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
IMAGE_NS = {"img": "http://www.google.com/schemas/sitemap-image/1.1"}
PRODUCT_ID_RE = re.compile(r"/p(\d+)(?:[/?#]|$)", re.IGNORECASE)

# Rules are intentionally explicit and easy to audit in the manifest. The
# first matching rule wins; more specific product terms come before broad ones.
LABEL_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "hair_care",
        (
            "العناية بالشعر", "الشعر", "شامبو", "بلسم", "زيت الشعر", "قناع الشعر",
            "hair", "shampoo", "conditioner", "hair oil", "hair mask",
        ),
    ),
    (
        "makeup",
        (
            "مكياج", "مكياج", "شفاه", "روج", "مسكرة", "كحل", "ظلال", "فاونديشن",
            "احمر خدود", "برايمر", "makeup", "lipstick", "lip gloss", "mascara",
            "eyeliner", "eyeshadow", "foundation", "concealer", "blush",
        ),
    ),
    (
        "fragrance",
        ("عطر", "عطور", "مباراة", "perfume", "fragrance", "eau de", "body mist"),
    ),
    (
        "body_care",
        (
            "العناية بالجسم", "الجسم", "اليد", "القدم", "مقشر الجسم", "body", "hand cream",
            "foot", "scrub", "body lotion", "body wash",
        ),
    ),
    (
        "skin_care",
        (
            "العناية بالبشرة", "البشرة", "الوجه", "غسول الوجه", "سيروم", "مرطب الوجه",
            "skin", "face", "serum", "moisturizer", "cleanser", "toner", "sunscreen",
        ),
    ),
    (
        "health",
        (
            "صحة", "فيتامين", "مكمل", "health", "vitamin", "supplement", "wellness",
        ),
    ),
    (
        "tools_accessories",
        (
            "فرشاة", "اسفنجة", "أداة", "اداة", "اكسسوار", "إكسسوار", "brush", "sponge",
            "tool", "accessor",
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/bellabox"))
    parser.add_argument("--sitemap-url", default=SITEMAP_URL)
    parser.add_argument("--max-products", type=int, default=0, help="0 means all products")
    parser.add_argument("--max-images-per-product", type=int, default=3)
    parser.add_argument("--min-images-per-class", type=int, default=10)
    parser.add_argument("--include-other", action="store_true")
    parser.add_argument("--download", action="store_true", help="Download image files")
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.set_defaults(download=True)
    parser.add_argument("--delay-seconds", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def fetch_bytes(url: str, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "BellaBox-Product-CNN-Dataset/1.0 (+https://github.com/mohammedalhmed/bellabox-product-cnn)",
                    "Accept": "application/xml,text/xml,image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                },
            )
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except Exception as error:  # network errors vary by Colab runtime
            last_error = error
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Unable to download {url}: {last_error}") from last_error


def xml_root(url: str, retries: int) -> ET.Element:
    return ET.fromstring(fetch_bytes(url, retries=retries))


def child_text(node: ET.Element, tag: str, namespaces: dict[str, str]) -> str:
    child = node.find(tag, namespaces)
    return (child.text or "").strip() if child is not None and child.text else ""


def collect_product_records(sitemap_url: str, retries: int) -> list[dict[str, object]]:
    root = xml_root(sitemap_url, retries)
    sitemap_links = [
        (node.text or "").strip()
        for node in root.findall(".//sm:sitemap/sm:loc", SITEMAP_NS)
        if node.text
    ]
    if not sitemap_links:
        sitemap_links = [sitemap_url]

    records: list[dict[str, object]] = []
    for child_sitemap in sitemap_links:
        if "blog" in child_sitemap.lower():
            continue
        child = xml_root(child_sitemap, retries)
        for url_node in child.findall(".//sm:url", SITEMAP_NS):
            product_url = child_text(url_node, "sm:loc", SITEMAP_NS)
            match = PRODUCT_ID_RE.search(product_url)
            if not match:
                continue
            image_urls = [
                (image_node.text or "").strip()
                for image_node in url_node.findall("img:image/img:loc", IMAGE_NS)
                if image_node.text
            ]
            if not image_urls:
                continue
            title = urllib.parse.unquote(urllib.parse.urlsplit(product_url).path.rsplit("/p", 1)[0])
            title = re.sub(r"^/", "", title).replace("-", " ").strip()
            records.append(
                {
                    "product_id": match.group(1),
                    "product_url": product_url,
                    "raw_title": html.unescape(title),
                    "image_urls": image_urls,
                }
            )
    unique: dict[str, dict[str, object]] = {}
    for record in records:
        unique.setdefault(str(record["product_id"]), record)
    return list(unique.values())


def normalize_text(value: str) -> str:
    value = html.unescape(value).casefold()
    value = re.sub(r"[\u064B-\u065F\u0670]", "", value)
    value = re.sub(r"[\s_–—]+", " ", value)
    return value.strip()


def infer_label(title: str, include_other: bool) -> str | None:
    normalized = normalize_text(title)
    for label, terms in LABEL_RULES:
        if any(normalize_text(term) in normalized for term in terms):
            return label
    return "other" if include_other else None


def safe_suffix(url: str) -> str:
    suffix = Path(urllib.parse.urlsplit(url).path).suffix.lower()
    return suffix if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"} else ".jpg"


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
        "product_id", "product_url", "raw_title", "label", "image_url", "local_path",
        "download_status", "width", "height", "mode",
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
    logging.info("Reading product sitemap: %s", args.sitemap_url)
    records = collect_product_records(args.sitemap_url, retries=args.retries)
    if args.max_products > 0:
        records = records[: args.max_products]
    rows: list[dict[str, object]] = []
    counters: Counter[str] = Counter()

    for index, record in enumerate(records, start=1):
        title = str(record["raw_title"])
        label = infer_label(title, include_other=args.include_other)
        if label is None:
            counters["unlabeled"] += 1
            continue
        product_id = str(record["product_id"])
        image_urls = list(record["image_urls"])[: args.max_images_per_product]
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
                except Exception as error:
                    status = f"error:{type(error).__name__}"
                    local_path.unlink(missing_ok=True)
                    logging.warning("Skipping image %s: %s", image_url, error)
            rows.append(
                {
                    "product_id": product_id,
                    "product_url": record["product_url"],
                    "raw_title": title,
                    "label": label,
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
        counters[label] += len(image_urls)
        if index % 50 == 0 or index == len(records):
            logging.info("Processed %s/%s products", index, len(records))

    manifest_path = args.output_dir / "manifest.csv"
    write_manifest(manifest_path, rows)
    summary = {
        "source_sitemap": args.sitemap_url,
        "products_seen": len(records),
        "images_manifested": len(rows),
        "counts_by_label": dict(counters),
        "download_enabled": args.download,
        "min_images_per_class": args.min_images_per_class,
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("Dataset summary: %s", json.dumps(summary, ensure_ascii=False))

    usable = Counter(
        row["label"]
        for row in rows
        if row["download_status"] == "ok" or not args.download
    )
    too_small = {
        label: count for label, count in usable.items() if count < args.min_images_per_class
    }
    if len(usable) < 2:
        raise SystemExit(
            "Dataset needs at least two usable classes. Review manifest.csv and label rules."
        )
    if too_small:
        raise SystemExit(
            "Some classes are below --min-images-per-class: "
            + json.dumps(too_small, ensure_ascii=False)
        )
    logging.info("Dataset is ready: %s classes, %s usable images", len(usable), sum(usable.values()))


if __name__ == "__main__":
    main()
