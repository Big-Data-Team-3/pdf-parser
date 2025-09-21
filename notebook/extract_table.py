#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF-only extractor for the two tables on the 'SHARE REPURCHASES AND DIVIDENDS' page (MSFT 10-K 2022).
- Top table: monthly share repurchases (Q4 FY22)
- Bottom table: dividends (Q4 FY22)

Approach:
1) Find the page by scanning for the page title (or you can pass --page to force).
2) Crop two regions using stable anchors:
   - Repurchases: between the "Period" header and the paragraph starting with "All share repurchases..."
   - Dividends: between the header line ("Declaration Date ... Amount") and the paragraph starting with "We returned ..."
3) Rebuild columns from word x-positions, then split mixed cells like "9,124,963 $ 289.34" into separate columns.

Outputs:
- <outdir>/share_repurchases_q4_2022.csv
- <outdir>/dividends_q4_2022.csv
"""

from __future__ import annotations
from pathlib import Path
import argparse
import re
from typing import List, Tuple
import numpy as np
import pandas as pd
import pdfplumber


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the input PDF")
    parser.add_argument("--outdir", default="output_tables", help="Directory to write CSVs")
    parser.add_argument("--page", type=int, default=None, help="Force a specific 1-based page number (optional)")
    return parser.parse_args()


# ---------- helpers: geometry -> rows/columns ----------

def cluster_rows(words, y_tol: float = 3.0):
    """Group words into rows by approximate y (top)."""
    rows = {}
    for w in words:
        k = round(w["top"] / y_tol) * y_tol
        rows.setdefault(k, []).append(w)
    ordered = []
    for k in sorted(rows):
        ordered.append(sorted(rows[k], key=lambda ww: ww["x0"]))
    return ordered

def infer_column_centers(rows, max_gap: float = 40.0) -> List[float]:
    """Rough clustering of x0 positions into column centers."""
    xs = []
    for r in rows:
        for w in r:
            xs.append(w["x0"])
    if not xs:
        return []
    xs = sorted(xs)
    centers = []
    cur = [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] <= max_gap:
            cur.append(x)
        else:
            centers.append(sum(cur) / len(cur))
            cur = [x]
    centers.append(sum(cur) / len(cur))
    return centers

def assign_to_nearest_centers(rows, centers: List[float]) -> List[List[str]]:
    """Assign each word to nearest column center; return row strings per column."""
    table = []
    for r in rows:
        cols = {i: [] for i, _ in enumerate(centers)}
        for w in r:
            idx = int(np.argmin([abs(w["x0"] - c) for c in centers]))
            cols[idx].append(w["text"])
        table.append([" ".join(cols[i]).strip() for i in range(len(centers))])
    return table

def extract_table_in_bbox(page, bbox) -> List[List[str]]:
    """Crop, row-cluster, infer columns, and build a raw table (list of row lists)."""
    words = page.crop(bbox).extract_words(use_text_flow=True, keep_blank_chars=False)
    rows = cluster_rows(words, y_tol=3.0)
    centers = infer_column_centers(rows, max_gap=50.0)
    if not centers:
        return []
    table = assign_to_nearest_centers(rows, centers)
    # drop empty rows
    table = [r for r in table if any(c.strip() for c in r)]
    return table


# ---------- anchors -> crop boxes for page 38 ----------

def find_anchor_y(page, term: str) -> float | None:
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    for w in words:
        if w["text"] == term:
            return w["top"]
    return None

def crop_repurchases_region(page) -> Tuple[float, float, float, float]:
    """Use 'Period' (header) and 'All' (first paragraph below table) anchors."""
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    y_top = None
    y_bottom = None
    for w in words:
        if w["text"] == "Period":
            y_top = w["top"] - 8
            break
    for w in words:
        if w["text"] == "All":
            y_bottom = w["top"] - 6
            break
    if y_top is None or y_bottom is None:
        raise RuntimeError("Could not find anchors for the repurchases table.")
    return (30, y_top, page.width - 30, y_bottom)

def crop_dividends_region(page) -> Tuple[float, float, float, float]:
    """Use 'Declaration' (header) and 'We' (paragraph below table) anchors."""
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    y_top = None
    y_bottom = None
    for w in words:
        if w["text"] == "Declaration":
            y_top = w["top"] - 8
            break
    for w in words:
        if w["text"] == "We":
            y_bottom = w["top"] - 6
            break
    if y_top is None or y_bottom is None:
        raise RuntimeError("Could not find anchors for the dividends table.")
    return (30, y_top, page.width - 30, y_bottom)


# ---------- clean & structure: specific to these tables ----------

NUM_RX = re.compile(r'[\d,]+(?:\.\d+)?')

def parse_repurchases_table(raw_rows: List[List[str]]) -> pd.DataFrame:
    """Split the 3-column raw into a 5-column tidy DataFrame."""
    out = []
    for row in raw_rows:
        left = row[0].strip() if len(row) > 0 else ""
        c2   = row[1].strip() if len(row) > 1 else ""
        c3   = row[2].strip() if len(row) > 2 else ""
        combo = " ".join(row).lower()

        # Skip header fragments / units lines
        if left.lower() == "period" or "in millions" in combo or "share" in combo and left == "":
            continue

        # Total line (no label text, just repeated totals in two numeric cols)
        if left == "" and NUM_RX.search(c2) and NUM_RX.search(c3):
            t2 = NUM_RX.findall(c2)
            t3 = NUM_RX.findall(c3)
            out.append(["Total", (t2[0] if t2 else ""), "", (t3[0] if t3 else ""), ""])
            continue

        # Normal monthly row → split c2 & c3 into two columns each
        t2 = NUM_RX.findall(c2)
        t3 = NUM_RX.findall(c3)
        if len(t2) >= 2 and len(t3) >= 2:
            out.append([
                left,
                t2[0],          # Total Number of Shares Purchased
                t2[-1],         # Average Price Paid Per Share
                t3[0],          # Shares Purchased as Part of Plans/Programs
                t3[-1],         # Approximate Dollar Value (In millions)
            ])

    cols = [
        "Period",
        "Total Number of Shares Purchased",
        "Average Price Paid Per Share",
        "Total Number of Shares Purchased as Part of Publicly Announced Plans or Programs",
        "Approximate Dollar Value of Shares That May Yet Be Purchased Under the Plans or Programs (In millions)",
    ]
    return pd.DataFrame(out, columns=cols)

def parse_dividends_table(raw_rows: List[List[str]]) -> pd.DataFrame:
    """Split the last column into Dividend Per Share and Amount (In millions)."""
    rows = []
    for row in raw_rows:
        if not row or all(not (c or "").strip() for c in row):
            continue
        line = " ".join(row).lower()
        if "declaration date" in line or "in millions" in line:
            continue
        # Expect row: [Declaration, Record, Payment, "$ 0.62 $ 4,627"]
        if len(row) >= 4 and re.search(r"[A-Za-z]+\s+\d{1,2},\s+\d{4}", row[0]):
            nums = NUM_RX.findall(row[-1])
            div, amt = (nums[0] if nums else ""), (nums[-1] if len(nums) >= 2 else "")
            rows.append([row[0], row[1], row[2], div, amt])

    return pd.DataFrame(rows, columns=[
        "Declaration Date",
        "Record Date",
        "Payment Date",
        "Dividend Per Share",
        "Amount (In millions)",
    ])


# ---------- main logic ----------

def find_target_page(pdf) -> int | None:
    for i, page in enumerate(pdf.pages, start=1):
        txt = page.extract_text() or ""
        if "SHARE REPURCHASES AND DIVIDENDS" in txt:
            return i
    return None

def main():
    args = parse_args()
    pdf_path = Path(args.pdf)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise SystemExit(f"❌ PDF not found: {pdf_path}")

    with pdfplumber.open(str(pdf_path)) as pdf:
        page_num = args.page or find_target_page(pdf)
        if page_num is None:
            raise SystemExit("❌ Could not locate the 'SHARE REPURCHASES AND DIVIDENDS' page.")
        page = pdf.pages[page_num - 1]

        # --- Repurchases ---
        rep_bbox = crop_repurchases_region(page)
        rep_raw = extract_table_in_bbox(page, rep_bbox)
        rep_df = parse_repurchases_table(rep_raw)
        rep_out = outdir / "share_repurchases_q4_2022.csv"
        rep_df.to_csv(rep_out, index=False)

        # --- Dividends ---
        div_bbox = crop_dividends_region(page)
        div_raw = extract_table_in_bbox(page, div_bbox)
        div_df = parse_dividends_table(div_raw)
        div_out = outdir / "dividends_q4_2022.csv"
        div_df.to_csv(div_out, index=False)

    print("✅ Wrote:")
    print(" -", rep_out)
    print(" -", div_out)


if __name__ == "__main__":
    main()
