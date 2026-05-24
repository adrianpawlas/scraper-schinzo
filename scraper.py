"""
Shinzo Brand Scraper
Scrapes all products from eu.shinzobrand.com, generates embeddings, uploads to Supabase.
Smart upsert: skips unchanged products, removes stale ones after 2 missed runs.
"""

import json
import io
import os
import re
import time
from datetime import datetime, timezone

import requests
import torch
from PIL import Image
from supabase import create_client, Client
from transformers import AutoProcessor, AutoModel

BASE_URL = "https://eu.shinzobrand.com"
SOURCE = "scraper-schinzo"
BRAND = "Schinzo Brand"

COLLECTIONS = [
    ("head-wear", "Headwear"),
    ("jumpers-jackets", "Jumpers, Jackets"),
    ("t-shirts", "Shirting"),
    ("knitwear", "Knitwear"),
    ("jeans-bottoms", "Jeans, Bottoms"),
    ("accessories", "Accessories"),
]

API_PARAMS = "?limit=250&currency=CZK"

SUPABASE_URL = os.environ.get(
    "SUPABASE_URL",
    "https://yqawmzggcgpeyaaynrjk.supabase.co",
)
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
)

RATES_FROM_EUR = {
    "EUR": 1.0, "USD": 1.1595, "CZK": 24.289, "PLN": 4.242,
    "GBP": 0.86418, "SEK": 10.8695, "NOK": 10.7295, "DKK": 7.4731,
    "HUF": 359.08, "RON": 5.249, "CHF": 0.9119, "AUD": 1.6283,
    "CAD": 1.6002, "AED": 4.258, "JPY": 184.53, "CNY": 7.8791,
    "INR": 110.9585, "MXN": 20.1045, "TRY": 53.0067, "BRL": 5.8165,
    "HKD": 9.0865, "SGD": 1.4846, "NZD": 1.9817, "KRW": 1759.6,
    "THB": 37.875, "ILS": 3.3569, "ISK": 143.6, "PHP": 71.45,
    "MYR": 4.6009, "IDR": 20516.54, "ZAR": 19.1045,
}

CURRENCY_PRIORITY = ["EUR", "USD", "CZK", "PLN", "GBP", "SEK", "NOK", "DKK", "HUF", "RON", "CHF", "AUD", "CAD"]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def load_model():
    print("Loading SigLIP model...")
    model_name = "google/siglip-base-patch16-384"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return processor, model


def convert_to_all_currencies(price_czk: float) -> str:
    eur = price_czk / RATES_FROM_EUR["CZK"]
    parts = []
    for currency in CURRENCY_PRIORITY:
        rate = RATES_FROM_EUR.get(currency)
        if rate is None:
            continue
        converted = round(eur * rate, 2)
        parts.append(f"{converted}{currency}")
    return ", ".join(parts)


def clean_html(html_text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("&amp;", "&").replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    return text


def fetch_all_products_from_collections() -> dict:
    handle_data: dict[str, dict] = {}
    for col_handle, cat_name in COLLECTIONS:
        url = f"{BASE_URL}/collections/{col_handle}/products.json{API_PARAMS}"
        try:
            resp = requests.get(url, timeout=30, headers=HEADERS)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ERROR fetching {col_handle}: {e}")
            continue

        products = data.get("products", [])
        print(f"  {col_handle}: {len(products)} products")
        for prod in products:
            h = prod["handle"]
            if h not in handle_data:
                handle_data[h] = {
                    "categories": set(),
                    "collection_title": prod.get("title", ""),
                    "collection_vendor": prod.get("vendor", "Shinzo"),
                    "collection_product_type": prod.get("product_type", ""),
                    "collection_tags": prod.get("tags", ""),
                    "collection_variants": [],
                    "collection_images": prod.get("images", []),
                }
            handle_data[h]["categories"].add(cat_name)
            for v in prod.get("variants", []):
                handle_data[h]["collection_variants"].append(v)
    return handle_data


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json",
}


def fetch_product_json(handle: str) -> dict | None:
    url = f"{BASE_URL}/products/{handle}.json?currency=CZK"
    try:
        resp = requests.get(url, timeout=30, headers=HEADERS)
        resp.raise_for_status()
        return resp.json()["product"]
    except Exception as e:
        print(f"    ERROR fetching product {handle}: {e}")
        return None


def get_image_embedding(processor, model, image_url: str) -> list[float] | None:
    try:
        resp = requests.get(image_url, timeout=30, headers=HEADERS)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
        return outputs.pooler_output[0].cpu().tolist()
    except Exception as e:
        print(f"    ERROR embedding image {image_url}: {e}")
        return None


def get_text_embedding(processor, model, text: str) -> list[float] | None:
    try:
        tokens = processor.tokenizer(text, truncation=True, max_length=60, return_tensors="pt")
        inputs = {"input_ids": tokens["input_ids"].to(device)}
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
        return outputs.pooler_output[0].cpu().tolist()
    except Exception as e:
        print(f"    ERROR embedding text: {e}")
        return None


def build_product_data(handle: str, col_data: dict, full_product: dict | None,
                       existing: dict | None = None) -> dict | None:
    title = col_data["collection_title"]
    product_type = col_data["collection_product_type"]
    tags_raw = col_data["collection_tags"]
    if isinstance(tags_raw, str):
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    else:
        tags = list(tags_raw) if tags_raw else []

    vendor = col_data.get("collection_vendor", "Shinzo")

    body_html = (full_product or {}).get("body_html", "")
    description = clean_html(body_html) if body_html else None
    if not description and existing:
        try:
            existing_meta = json.loads(existing["metadata"]) if isinstance(existing["metadata"], str) else existing["metadata"]
            description = existing_meta.get("description") or existing.get("description", "")
        except (json.JSONDecodeError, TypeError, KeyError):
            description = existing.get("description", "")

    variants = col_data["collection_variants"]
    if not variants:
        return None

    prices_czk = []
    sale_prices_czk = []
    sale_detected = False
    for v in variants:
        price_str = v.get("price", "0")
        try:
            price_f = float(price_str)
        except (ValueError, TypeError):
            continue
        prices_czk.append(price_f)

        cap = v.get("compare_at_price")
        if cap is not None and cap != "":
            try:
                cap_f = float(cap)
                if cap_f > 0 and abs(cap_f - price_f) > 0.01:
                    sale_detected = True
                    sale_prices_czk.append(price_f)
            except (ValueError, TypeError):
                pass

    if not prices_czk:
        return None

    min_price_czk = min(prices_czk)

    price_str = convert_to_all_currencies(min_price_czk)

    sale_value = None
    if sale_detected and sale_prices_czk:
        sale_value = f"{min(sale_prices_czk):.2f}CZK"

    sizes_raw = [v.get("title", "").strip() for v in variants if v.get("title", "").strip()]
    seen = set()
    sizes = []
    for s in sizes_raw:
        if s not in seen:
            seen.add(s)
            sizes.append(s)

    images = (full_product or {}).get("images") or col_data.get("collection_images") or []
    if not images:
        return None

    primary_image_url = images[0]["src"]
    additional_image_urls = [img["src"] for img in images[1:]]
    additional_images_str = " , ".join(additional_image_urls) if additional_image_urls else None

    product_url = f"{BASE_URL}/products/{handle}"

    categories = list(col_data["categories"])
    if product_type and product_type not in categories:
        categories.append(product_type)
    category_str = ", ".join(categories) if categories else None

    gender = None

    size_str = ", ".join(sizes) if sizes else None

    available_count = sum(1 for v in variants if v.get("available"))

    metadata_parts = {
        "title": title,
        "description": description,
        "sizes": sizes,
        "price_czk": min_price_czk,
        "category": category_str,
        "tags": tags,
        "vendor": vendor,
        "product_type": product_type,
        "sku": variants[0].get("sku", ""),
        "available_variants": available_count,
        "total_variants": len(variants),
        "missed_runs": 0,
    }
    if sale_detected:
        metadata_parts["on_sale"] = True
        metadata_parts["sale_price_czk"] = sale_value

    metadata_json = json.dumps(metadata_parts, ensure_ascii=False)

    info_text = (
        f"Title: {title}. Description: {description}. "
        f"Category: {category_str}. Price: {price_str}. "
        f"Sizes: {size_str}. Tags: {', '.join(tags)}. "
        f"Gender: {gender or 'unisex'}. Brand: {BRAND}."
    )

    now = datetime.now(timezone.utc).isoformat()

    record = {
        "id": handle,
        "source": SOURCE,
        "product_url": product_url,
        "image_url": primary_image_url,
        "brand": BRAND,
        "title": title,
        "description": description,
        "category": category_str,
        "gender": gender,
        "created_at": now,
        "metadata": metadata_json,
        "size": size_str,
        "second_hand": False,
        "country": None,
        "tags": tags if tags else None,
        "price": price_str,
        "sale": sale_value,
        "additional_images": additional_images_str,
        "info_text": info_text,
    }

    return record


def has_changed(existing: dict, new_data: dict) -> bool:
    fields = ["title", "price", "sale", "category", "image_url", "additional_images", "size"]
    for field in fields:
        if existing.get(field) != new_data.get(field):
            return True

    existing_tags = existing.get("tags")
    new_tags = new_data.get("tags")
    if existing_tags != new_tags:
        return True

    existing_meta_str = existing.get("metadata", "{}")
    new_meta_str = new_data.get("metadata", "{}")
    try:
        existing_meta = json.loads(existing_meta_str) if isinstance(existing_meta_str, str) else existing_meta_str
        new_meta = json.loads(new_meta_str) if isinstance(new_meta_str, str) else new_meta_str
    except (json.JSONDecodeError, TypeError):
        return True

    for key in ("description", "available_variants", "sku", "on_sale"):
        if existing_meta.get(key) != new_meta.get(key):
            return True

    return False


def upsert_batch(supabase: Client, records: list[dict], batch_size: int = 50, max_retries: int = 3):
    failed_log = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        for attempt in range(1, max_retries + 1):
            try:
                batch_clean = []
                for rec in batch:
                    r = {k: v for k, v in rec.items() if k != "info_text"}
                    batch_clean.append(r)
                supabase.table("products").upsert(batch_clean, on_conflict="id").execute()
                break
            except Exception as e:
                print(f"    Batch upsert error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    ids = [r.get("id", "?") for r in batch]
                    msg = f"[{datetime.now(timezone.utc).isoformat()}] Failed after {max_retries} retries: {ids} | {e}"
                    failed_log.append(msg)
                    print(f"    GIVING UP on batch: {ids}")
    if failed_log:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "failed_products.log")
        with open(log_path, "a") as f:
            for entry in failed_log:
                f.write(entry + "\n")


def main():
    processor, model = load_model()
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    print("\nStage 1: Collecting all product handles from collections...")
    handle_data = fetch_all_products_from_collections()
    current_handles = set(handle_data.keys())
    print(f"Total unique products found: {len(current_handles)}")

    print("\nStage 2: Fetching product details and comparing with DB...")
    response = supabase.table("products").select("*").eq("source", SOURCE).execute()
    existing_map: dict[str, dict] = {}
    for row in response.data:
        existing_map[row["id"]] = row

    to_upsert = []
    new_ids = []
    updated_ids = []
    unchanged_ids = []
    failed_ids = []

    for i, handle in enumerate(sorted(current_handles), 1):
        col_data = handle_data[handle]
        print(f"  [{i}/{len(current_handles)}] {handle}...")

        full_product = fetch_product_json(handle)
        existing = existing_map.get(handle)

        product_data = build_product_data(handle, col_data, full_product, existing)
        if product_data is None:
            print(f"    FAILED to build record")
            failed_ids.append(handle)
            continue

        if existing is None:
            print(f"    NEW product")
            image_emb = get_image_embedding(processor, model, product_data["image_url"])
            if image_emb is None:
                print(f"    FAILED image embedding")
                failed_ids.append(handle)
                continue
            text_emb = get_text_embedding(processor, model, product_data["info_text"])
            product_data["image_embedding"] = image_emb
            product_data["info_embedding"] = text_emb
            to_upsert.append(product_data)
            new_ids.append(handle)
            time.sleep(0.5)
            print(f"    OK - {product_data['title'][:60]}")

        elif has_changed(existing, product_data):
            print(f"    CHANGED - regenerating embeddings")
            image_emb = get_image_embedding(processor, model, product_data["image_url"])
            if image_emb is None:
                print(f"    FAILED image embedding")
                failed_ids.append(handle)
                continue
            text_emb = get_text_embedding(processor, model, product_data["info_text"])
            product_data["image_embedding"] = image_emb
            product_data["info_embedding"] = text_emb
            to_upsert.append(product_data)
            updated_ids.append(handle)
            time.sleep(0.5)
            print(f"    OK - {product_data['title'][:60]}")

        else:
            print(f"    UNCHANGED - skipping")
            unchanged_ids.append(handle)

    total_new = len(new_ids)
    total_updated = len(updated_ids)
    total_unchanged = len(unchanged_ids)
    total_failed = len(failed_ids)

    print(f"\nStage 3: Batch-upserting {len(to_upsert)} products...")
    if to_upsert:
        upsert_batch(supabase, to_upsert, batch_size=50, max_retries=3)

    print(f"\nStage 4: Resetting missed_runs for seen products...")
    reset_ids = []
    for handle in current_handles:
        existing = existing_map.get(handle)
        if existing is None:
            continue
        meta_str = existing.get("metadata", "{}")
        try:
            meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
        except (json.JSONDecodeError, TypeError):
            meta = {}
        if meta.get("missed_runs", 0) != 0:
            meta["missed_runs"] = 0
            try:
                supabase.table("products").update(
                    {"metadata": json.dumps(meta, ensure_ascii=False)}
                ).eq("id", handle).execute()
                reset_ids.append(handle)
            except Exception as e:
                print(f"  ERROR resetting missed_runs for {handle}: {e}")
    if reset_ids:
        print(f"  Reset missed_runs for {len(reset_ids)} products")

    print(f"\nStage 5: Stale product cleanup...")
    deleted_ids = []
    missed_ids = []
    for handle, existing in existing_map.items():
        if handle in current_handles:
            continue
        meta_str = existing.get("metadata", "{}")
        try:
            meta = json.loads(meta_str) if isinstance(meta_str, str) else meta_str
        except (json.JSONDecodeError, TypeError):
            meta = {}
        missed = meta.get("missed_runs", 0) + 1
        if missed >= 2:
            try:
                supabase.table("products").delete().eq("id", handle).execute()
                deleted_ids.append(handle)
                print(f"  DELETED {handle} (missed {missed} runs)")
            except Exception as e:
                print(f"  ERROR deleting {handle}: {e}")
        else:
            meta["missed_runs"] = missed
            try:
                supabase.table("products").update(
                    {"metadata": json.dumps(meta, ensure_ascii=False)}
                ).eq("id", handle).execute()
                missed_ids.append(handle)
                print(f"  STALE {handle} (missed {missed}/2)")
            except Exception as e:
                print(f"  ERROR updating missed_runs for {handle}: {e}")

    print(f"\n{'='*50}")
    print(f"  RUN COMPLETE")
    print(f"  New products added:      {total_new}")
    print(f"  Products updated:        {total_updated}")
    print(f"  Products unchanged:      {total_unchanged}")
    print(f"  Stale products deleted:  {len(deleted_ids)}")
    print(f"  Stale products (miss 1): {len(missed_ids)}")
    print(f"  Failed to process:       {total_failed}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
