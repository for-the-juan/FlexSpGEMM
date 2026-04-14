#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2_expand_labels.py
----------------------
Task 2: Read step1 intermediate CSVs, expand each matrix into training rows
         (AA/AAT × A100/H200), add hardware params and best_tile/best_tc labels
         from prime_data benchmark results.

The final output CSVs have the exact same 454-column schema as the reference
all_data/{train,val,test}_dataset.csv.

Usage (called by run_pipeline.py, or standalone):
    python3 step2_expand_labels.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_utils as pu


# ═══════════════════════════════════════════════════════════════════════════════
# Build prime_data lookup indices (one-time)
# ═══════════════════════════════════════════════════════════════════════════════

def build_all_indices():
    """Return dict of lookup indices for A100/H200 × AA/AAT."""
    print("[step2] Building prime_data indices ...")
    idx = {
        "A100_AA": pu.build_a100_aa_index(),
        "A100_AAT": pu.build_a100_aat_index(),
        "H200_AA": pu.build_h200_index(pu.H200_AA_DIR),
        "H200_AAT": pu.build_h200_index(pu.H200_AAT_DIR),
    }
    for key, val in idx.items():
        print("  {}: {} entries".format(key, len(val)))
    return idx


def lookup_best(name, gpu, mode, indices):
    """Look up best_tile/best_tc for a matrix from prime_data.
    Returns (best_tile, best_tc) or None if not found / symmetric."""
    key = "{}_{}".format(gpu, mode)
    index = indices.get(key, {})
    entry = index.get(name)
    if entry is None:
        return None

    if gpu == "A100" and mode == "AA":
        return pu.parse_a100_aa_file(entry)
    elif gpu == "A100" and mode == "AAT":
        return pu.parse_a100_aat_file(entry)
    elif gpu.startswith("H200"):
        return pu.parse_h200_log_dir(entry)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Row expansion: one intermediate row -> 2 or 4 final rows
# ═══════════════════════════════════════════════════════════════════════════════

def expand_row(stage1_row, hw_data, indices):
    """Expand one stage1 intermediate row into final training rows."""
    name = stage1_row.get("matrix_name", "")
    is_sym = stage1_row.get("symmetric", "").lower() == "yes"

    combos = []
    for gpu in ("A100", "H200"):
        combos.append((gpu, "AA"))
        if not is_sym:
            combos.append((gpu, "AAT"))

    final_rows = []
    for gpu, mode in combos:
        row = {}
        row["matrix_name"] = name
        # Internal fields for splitting (not written to CSV)
        row["_gpu"] = gpu
        row["_mode"] = mode
        row["rows"] = stage1_row.get("rows", "")
        row["cols"] = stage1_row.get("cols", "")
        row["nnz"] = stage1_row.get("nnz", "")
        row["density"] = stage1_row.get("density", "")
        row["avg_nnz_per_row"] = stage1_row.get("avg_nnz_per_row", "")

        # p9 A features — always the same
        for tile in pu.P9_TILE_ORDER:
            for f in pu.P9_FIELDS:
                key = "p9_{}_{}".format(tile, f)
                row[key] = stage1_row.get(key, "")

        # p9 AT features
        if mode == "AA":
            # For AA mode: p9_AT = copy of p9 (A matrix explored twice)
            for tile in pu.P9_TILE_ORDER:
                for f in pu.P9_FIELDS:
                    src = "p9_{}_{}".format(tile, f)
                    dst = "p9_AT_{}_{}".format(tile, f)
                    row[dst] = stage1_row.get(src, "")
        else:
            # For AAT mode: use real AT probe features
            for tile in pu.P9_TILE_ORDER:
                for f in pu.P9_FIELDS:
                    key = "p9_AT_{}_{}".format(tile, f)
                    row[key] = stage1_row.get(key, "")

        # pC features: AA mode uses probeC AA, AAT mode uses probeC AAT
        src_prefix = "pC_AA" if mode == "AA" else "pC_AAT"
        for tile in pu.PC_TILE_ORDER:
            for f in pu.PC_FIELDS:
                src_key = "{}_{}_{}".format(src_prefix, tile, f)
                dst_key = "pC_{}_{}".format(tile, f)
                row[dst_key] = stage1_row.get(src_key, "")

        # Hardware params
        hw = hw_data.get(gpu, {})
        for col in pu.HARDWARE_COLS:
            row[col] = hw.get(col, "")

        # Labels from prime_data
        result = lookup_best(name, gpu, mode, indices)
        if result is not None:
            row["best_tile"] = result[0]
            row["best_tc"] = result[1]
        else:
            row["best_tile"] = ""
            row["best_tc"] = ""

        final_rows.append(row)

    return final_rows


# ═══════════════════════════════════════════════════════════════════════════════
# Also update probe.csv with best_tile/best_tc labels
# ═══════════════════════════════════════════════════════════════════════════════

def update_probe_csv(indices):
    """Fill in best_tile/best_tc/best_combo in probe.csv using prime_data."""
    probe_csv = pu.OUTPUT_DIR / "probe.csv"
    if not probe_csv.exists():
        print("[step2] probe.csv not found, skipping update")
        return

    with probe_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames)
        rows = list(reader)

    updated = 0
    for row in rows:
        name = row.get("matrix_name", "")
        gpu = row.get("gpu", "")
        mode = row.get("mode", "")
        result = lookup_best(name, gpu, mode, indices)
        if result is not None:
            row["best_tile"] = result[0]
            row["best_tc"] = result[1]
            row["best_combo"] = "{}_{}".format(result[0], result[1]) if result[0] else ""
            updated += 1

    with probe_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print("[step2] probe.csv updated: {} rows with labels".format(updated))


# ═══════════════════════════════════════════════════════════════════════════════
# Split data by GPU and mode
# ═══════════════════════════════════════════════════════════════════════════════

def split_by_gpu_mode():
    """Split train/val/test.csv into A100_AA, A100_AAT, H200_AA, H200_AAT folders."""
    print("\n" + "=" * 70)
    print("  Splitting data by GPU and mode")
    print("=" * 70)

    final_header = pu.build_final_header()
    splits = ("train", "val", "test")
    gpu_modes = ["A100_AA", "A100_AAT", "H200_AA", "H200_AAT"]

    # Create output directories
    for gm in gpu_modes:
        gm_dir = pu.OUTPUT_DIR / gm
        gm_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        csv_path = pu.OUTPUT_DIR / "{}.csv".format(split)
        if not csv_path.exists():
            print("[split] {} not found, skipping".format(csv_path))
            continue

        # Read all rows (with _gpu and _mode fields)
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)

        # Group by gpu_mode
        grouped = {gm: [] for gm in gpu_modes}
        for row in all_rows:
            gpu = row.get("_gpu", "")
            mode = row.get("_mode", "")
            key = "{}_{}".format(gpu, mode)
            if key in grouped:
                grouped[key].append(row)

        # Write split CSVs (without _gpu and _mode columns)
        for gm in gpu_modes:
            gm_dir = pu.OUTPUT_DIR / gm
            out_csv = gm_dir / "{}.csv".format(split)
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=final_header, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(grouped[gm])
            print("[split] {}/{} : {} rows".format(gm, split, len(grouped[gm])))

    print("\n[split] Split complete!")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def run_step2():
    print("\n" + "=" * 70)
    print("  Step 2: Expand labels and hardware params")
    print("=" * 70)

    hw_data = pu.read_hardware()
    indices = build_all_indices()
    final_header = pu.build_final_header()
    # Header with _gpu and _mode for intermediate file (used for splitting)
    intermediate_header = final_header + ["_gpu", "_mode"]

    for split in ("test", "train", "val"):
        stage1_csv = pu.STAGE1_DIR / "{}.csv".format(split)
        if not stage1_csv.exists():
            print("[step2] {} not found, skipping".format(stage1_csv))
            continue
        final_csv = pu.OUTPUT_DIR / "{}.csv".format(split)

        # Read stage1 intermediate
        with stage1_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            stage1_rows = list(reader)

        print("\n[step2] {} : {} matrices".format(split, len(stage1_rows)))

        final_rows = []
        no_label = 0
        for idx, s1_row in enumerate(stage1_rows, 1):
            expanded = expand_row(s1_row, hw_data, indices)
            for row in expanded:
                if not row.get("best_tile"):
                    no_label += 1
            final_rows.extend(expanded)

        # Write final expanded CSV with temporary _gpu/_mode fields for splitting.
        # Timing columns from step1 are intentionally not written into final output.
        with final_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=intermediate_header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(final_rows)

        print("[step2] {} done: {} rows (no_label={}) -> {}".format(
            split, len(final_rows), no_label, final_csv))

    print("\n[step2] Step 2 complete!")


if __name__ == "__main__":
    run_step2()
