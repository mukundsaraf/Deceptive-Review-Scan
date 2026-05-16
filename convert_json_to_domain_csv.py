"""
Stage A — Amazon JSON → Domain CSV Converter
Converts x.json (JSONL format) into x.csv
matching the universal schema.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime


def convert_json(json_path: str, output_csv: str) -> pd.DataFrame:
    rows = []

    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)

                review_text  = data.get("reviewText", "") or ""
                review_title = data.get("summary", "")   or ""
                rating       = float(data.get("overall", 3))
                verified     = int(bool(data.get("verified", False)))

                vote = str(data.get("vote", "0")).replace(",", "")
                try:
                    helpful_votes = int(vote)
                except ValueError:
                    helpful_votes = 0

                images = data.get("image", [])
                image_count = len(images) if isinstance(images, list) else 0
                has_images  = int(image_count > 0)

                reviewer_id = data.get("reviewerID", "")
                product_id  = data.get("asin", "")
                unix_time   = data.get("unixReviewTime", 0) or 0

                year, month = 2019, 6
                if unix_time:
                    dt    = datetime.utcfromtimestamp(unix_time)
                    year  = dt.year
                    month = dt.month

                style_present = int("style" in data)

                rows.append({
                    "review_title":               review_title,
                    "review_text":                review_text,
                    "review_rating":              rating,
                    "number_of_helpful":          helpful_votes,
                    "number_of_photos":           image_count,
                    "has_photos":                 has_images,
                    "is_campaign_product":        0,
                    "review_year":                year,
                    "review_month":               month,
                    "reviewer_classified_fake":   0,   # unknown — unlabeled
                    "fake_review_product":        0,   # unknown — unlabeled
                    "review_is_removed_by_amazon": 0,
                    "verified_purchase":          verified,
                    "image_count":                image_count,
                    "reviewer_id":                reviewer_id,
                    "product_id":                 product_id,
                    "unix_review_time":           unix_time,
                    "style_present":              style_present,
                })

            except Exception:
                continue

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[Stage A] Saved: {output_csv}  ({len(df):,} rows)")
    return df


if __name__ == "__main__":
    convert_json("Software.json", "Software.csv")
