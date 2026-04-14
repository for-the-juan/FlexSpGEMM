#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py
---------------
One-command entry point for the full data preparation pipeline.

This script:
  1) Runs probe9 and probeC on all matrices in test/train/val   (step1)
  2) Expands rows with hardware params and best_tile/best_tc    (step2)

Final outputs in data_prepare/data_get/:
  - train.csv, val.csv, test.csv   (454-column, ready for LightGBM training)
  - probe.csv                      (timing-report style for test matrices)

Usage:
    python3 run_pipeline.py                # full pipeline
    python3 run_pipeline.py --step 1       # only step1
    python3 run_pipeline.py --step 2       # only step2 (requires step1 output)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def remove_gpu_mode_columns():
    """Remove _gpu and _mode columns from main train/val/test.csv files."""
    from pipeline_utils import OUTPUT_DIR, build_final_header
    final_header = build_final_header()
    
    for split in ("train", "val", "test"):
        csv_path = OUTPUT_DIR / "{}.csv".format(split)
        if not csv_path.exists():
            continue
        
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        # Rewrite without _gpu and _mode columns
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=final_header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        
        print("[cleanup] {} : removed _gpu/_mode columns".format(csv_path.name))


def main():
    parser = argparse.ArgumentParser(
        description="FlexSpGEMM data preparation pipeline")
    parser.add_argument(
        "--step", type=int, default=0, choices=[0, 1, 2],
        help="Run only a specific step (0 = both, default)")
    args = parser.parse_args()

    print("=" * 70)
    print("  FlexSpGEMM Data Preparation Pipeline")
    print("=" * 70)

    t0 = time.time()

    if args.step in (0, 1):
        from step1_build_probe_features import run_step1
        run_step1()

    if args.step in (0, 2):
        from step2_expand_labels import run_step2, split_by_gpu_mode
        run_step2()
        # Split data into A100_AA, A100_AAT, H200_AA, H200_AAT folders
        split_by_gpu_mode()
        # Remove _gpu and _mode columns from main CSVs
        remove_gpu_mode_columns()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  Pipeline finished in {:.1f} seconds".format(elapsed))
    print("=" * 70)

    # Summary
    from pipeline_utils import OUTPUT_DIR, build_final_header
    for name in ("test.csv", "train.csv", "val.csv", "probe.csv"):
        p = OUTPUT_DIR / name
        if p.exists():
            import csv as csv_mod
            with p.open("r", encoding="utf-8") as f:
                reader = csv_mod.reader(f)
                header = next(reader)
                count = sum(1 for _ in reader)
            print("  {} : {} cols x {} rows".format(name, len(header), count))
        else:
            print("  {} : NOT FOUND".format(name))


if __name__ == "__main__":
    main()
