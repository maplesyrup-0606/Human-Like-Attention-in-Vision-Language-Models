#!/usr/bin/env python3
"""
Merge SBERT summary JSON and GEM metrics JSON into a single CSV.

- SBERT: keep only 'mean' per method.
- GEM:
  * scalar metrics kept as-is (e.g., 'bleu', 'CIDEr')
  * dict metrics reduced to F1-type value:
      - 'bertscore' -> take 'f1'
      - 'rouge1', 'rouge2', 'rougeL', 'rougeLsum' -> take 'fmeasure'
  * ignore housekeeping keys: 'predictions_file', 'references_file', 'N'

Usage:
  python merge_metrics.py --sbert sbert.json --gem gem.json --out merged.csv
"""
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Any

HOUSEKEEPING_KEYS = {"predictions_file", "references_file", "N"}

# Desired output columns in order
GEM_COLUMNS = [
    ("bleu", "bleu"),
    ("bertscore", "bertscore_f1"),      # take 'f1'
    # ("CIDEr", "cider"),                 # keep numeric, normalize name to 'cider'
    ("rouge1", "rouge1_fmeasure"),      # take 'fmeasure'
    ("rouge2", "rouge2_fmeasure"),      # take 'fmeasure'
    ("rougeL", "rougeL_fmeasure"),      # take 'fmeasure'
    ("rougeLsum", "rougeLsum_fmeasure") # take 'fmeasure'
]

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)

def extract_gem_for_method(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reduce a GEM metrics dict for one method to target fields.
    """
    out = {}

    # Map through desired columns with rules
    for metric_name, out_name in GEM_COLUMNS:
        if metric_name not in entry:
            out[out_name] = None
            continue

        val = entry[metric_name]
        if isinstance(val, dict):
            # Choose f1 for bertscore, fmeasure for rouge*
            if metric_name == "bertscore":
                out[out_name] = val.get("f1", None)
            else:
                out[out_name] = val.get("fmeasure", None)
        else:
            # Scalar (e.g., bleu, CIDEr)
            if metric_name == "CIDEr":
                out[out_name] = val  # rename to 'cider' column
            else:
                out[out_name] = val
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sbert", type=Path, help="Path to SBERT JSON")
    ap.add_argument("--gem", type=Path, help="Path to GEM metrics JSON")
    ap.add_argument("--out", type=Path, help="Path to output CSV")
    args = ap.parse_args()

    # args.sbert = Path("~/NSERC/data/generated_captions/jul18_samples/semantics/mscoco/sbert_f1_summary.json").expanduser()
    # args.gem = Path("~/NSERC/data/generated_captions/jul18_samples/semantics/mscoco/gem_summary.json").expanduser()
    # args.out = Path("~/NSERC/data/generated_captions/jul18_samples/semantics/mscoco/overall.csv").expanduser()
    args.sbert = Path("~/NSERC/data/generated_captions/jul18_samples/semantics/cub/sbert_f1_summary.json").expanduser()
    args.gem = Path("~/NSERC/data/generated_captions/jul18_samples/semantics/cub/gem_summary.json").expanduser()
    args.out = Path("~/NSERC/data/generated_captions/jul18_samples/semantics/cub/overall.csv").expanduser()

    sbert = load_json(args.sbert)          # {method: {"mean": float, "std": float}}
    gem = load_json(args.gem)              # {method: {...metrics...}}

    # Union of method names so we don't silently drop anything
    methods = sorted(set(sbert.keys()) | set(gem.keys()))

    header = ["method", "sbert_mean"] + [col for _, col in GEM_COLUMNS]

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for m in methods:
            row = {"method": m}

            # SBERT mean
            sbert_entry = sbert.get(m)
            row["sbert_mean"] = (sbert_entry or {}).get("mean", None)

            # GEM metrics (reduced)
            gem_entry = gem.get(m)
            if gem_entry is not None:
                # Strip housekeeping keys if present
                gem_clean = {k: v for k, v in gem_entry.items() if k not in HOUSEKEEPING_KEYS}
                row.update(extract_gem_for_method(gem_clean))
            else:
                # Fill GEM cols with None if missing
                for _, col in GEM_COLUMNS:
                    row[col] = None

            writer.writerow(row)

    print(f"Saved merged CSV to: {args.out}")

if __name__ == "__main__":
    main()