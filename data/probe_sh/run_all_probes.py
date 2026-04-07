#!/usr/bin/env python3
"""
run_all_probes.py
-----------------
Batch probe (probe9/tile_probe) and predict (probeC AA + AAT) for all matrices
in test/train/val splits, then append results and timing to the corresponding CSV files.

Usage:
    python3 run_all_probes.py                       # process all 3 splits
    python3 run_all_probes.py --splits test          # process test only
    python3 run_all_probes.py --splits test val      # process test + val
    python3 run_all_probes.py --timeout 300          # custom timeout per matrix
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path("/home/stu1/donghangcheng/code/FlexSpGEMM/data")
PROBE9_EXEC = Path("/home/stu1/marui/probe9_v2/tile_probe")
PROBEC_EXEC = Path("/home/stu1/marui/probeC_v2/probeC")
TIMEOUT_SEC = 600

SPLITS = {
    "test": {
        "matrix_dir": BASE_DIR / "test",
        "csv_path": BASE_DIR / "test_matrices.csv",
    },
    "train": {
        "matrix_dir": BASE_DIR / "train",
        "csv_path": BASE_DIR / "train_matrices.csv",
    },
    "val": {
        "matrix_dir": BASE_DIR / "val",
        "csv_path": BASE_DIR / "val_matrices.csv",
    },
}

# ─── Column definitions ─────────────────────────────────────────────────────

PROBE9_TILE_ORDER = [
    "8x8", "16x8", "8x16", "16x16", "16x32", "32x16", "32x32", "8x32", "32x8",
]
PROBE9_FIELDS = [
    "numtile", "tile_density", "nnz_per_tile_avg", "nnz_per_tile_max",
    "nnz_per_tile_min", "nnz_per_tile_std", "nnz_per_tile_cv",
    "tile_fill_avg", "tile_fill_max",
    "tiles_per_row_avg", "tiles_per_row_max", "tiles_per_row_min",
    "tiles_per_row_std", "tiles_per_row_cv", "empty_row_ratio",
    "tiles_per_col_avg", "tiles_per_col_max", "tiles_per_col_min",
    "tiles_per_col_std", "tiles_per_col_cv", "empty_col_ratio",
    "hist_1", "hist_2_4", "hist_4_8", "hist_8_16", "hist_16_32",
    "hist_32_64", "hist_64_128", "hist_128_plus",
    "nnz_per_row_max", "nnz_per_row_std", "nnz_per_row_skewness",
    "nnz_per_col_max", "nnz_per_col_avg", "nnz_per_col_std",
]

PROBEC_TILE_ORDER = ["8", "16", "32"]
PROBEC_FIELDS = [
    "sml", "lrg", "dns", "ful", "numblkC", "total_flops",
    "avg_matchedcnt", "max_matchedcnt", "max_flops_per_tile",
    "tpc_C_avg", "tpc_C_max", "tpc_C_std", "tpc_C_empty_ratio",
    "est_nnz_avg", "est_nnz_max", "est_nnz_std",
]

TIMING_FIELDS = [
    "probe9_wall_ms", "probe9_load_ms", "probe9_probe_ms",
    "probeC_AA_wall_ms", "probeC_AA_load_ms", "probeC_AA_build_ms",
    "probeC_AA_estimate_ms",
    "probeC_AAT_wall_ms", "probeC_AAT_load_ms", "probeC_AAT_build_ms",
    "probeC_AAT_estimate_ms",
]


# ─── Field name builders ────────────────────────────────────────────────────

def probe9_output_fields():
    cols = []
    for tile in PROBE9_TILE_ORDER:
        for field in PROBE9_FIELDS:
            cols.append("p9_{}_{}".format(tile, field))
    return cols


def probec_output_fields(prefix):
    cols = []
    for tile in PROBEC_TILE_ORDER:
        for field in PROBEC_FIELDS:
            cols.append("{}_{}_{}".format(prefix, tile, field))
    return cols


def all_new_fields():
    return (TIMING_FIELDS
            + probe9_output_fields()
            + probec_output_fields("pC_AA")
            + probec_output_fields("pC_AAT"))


# ─── Low-level helpers ──────────────────────────────────────────────────────

def run_command(cmd, timeout_sec):
    """Run a command with timeout. Returns (rc, stdout, stderr, wall_ms)."""
    started = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout_sec,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        return result.returncode, result.stdout, result.stderr, wall_ms
    except subprocess.TimeoutExpired:
        wall_ms = (time.perf_counter() - started) * 1000.0
        return -1, "", "TIMEOUT", wall_ms


def extract_float(pattern, text):
    match = re.search(pattern, text)
    return match.group(1) if match else ""


def parse_csv_section(text, marker):
    """Extract CSV header + rows from the section starting with *marker*."""
    if marker not in text:
        return [], []

    tail = text.split(marker, 1)[1]
    csv_lines = []
    for raw in tail.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("===") or line.startswith("===="):
            break
        if line.startswith("="):
            break
        if "," in line:
            csv_lines.append(line)

    if not csv_lines:
        return [], []

    header = [x.strip() for x in csv_lines[0].split(",")]
    rows = []
    for line in csv_lines[1:]:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != len(header):
            continue
        rows.append(dict(zip(header, parts)))
    return header, rows


# ─── Output parsers ─────────────────────────────────────────────────────────

def parse_probe9_output(stdout, wall_ms):
    data = {
        "probe9_wall_ms": "{:.3f}".format(wall_ms),
        "probe9_load_ms": extract_float(r"Load time:\s*([\d.]+)\s*ms", stdout),
        "probe9_probe_ms": extract_float(r"Probe time:\s*([\d.]+)\s*ms", stdout),
    }

    _, rows = parse_csv_section(stdout, "=== CSV Format Features ===")
    by_tile = {row.get("tile_size", ""): row for row in rows}

    for tile in PROBE9_TILE_ORDER:
        row = by_tile.get(tile, {})
        for field in PROBE9_FIELDS:
            data["p9_{}_{}".format(tile, field)] = row.get(field, "")

    return data


def parse_probec_output(stdout, wall_ms, time_prefix, feature_prefix):
    data = {
        "{}_wall_ms".format(time_prefix): "{:.3f}".format(wall_ms),
        "{}_load_ms".format(time_prefix): extract_float(r"Load time:\s*([\d.]+)\s*ms", stdout),
        "{}_build_ms".format(time_prefix): extract_float(r"Build time:\s*([\d.]+)\s*ms", stdout),
        "{}_estimate_ms".format(time_prefix): extract_float(r"Estimate time:\s*([\d.]+)\s*ms", stdout),
    }

    _, rows = parse_csv_section(stdout, "=== CSV ===")
    by_tile = {row.get("tile_m", ""): row for row in rows}

    for tile in PROBEC_TILE_ORDER:
        row = by_tile.get(tile, {})
        for field in PROBEC_FIELDS:
            data["{}_{}_{}".format(feature_prefix, tile, field)] = row.get(field, "")

    return data


# ─── Per-matrix driver ──────────────────────────────────────────────────────

def empty_result_row():
    return {field: "" for field in all_new_fields()}


def process_matrix(matrix_path, timeout_sec):
    """Run probe9, probeC AA, probeC AAT on one matrix. Returns dict of all new columns."""
    result = empty_result_row()

    # 1) probe9 / tile_probe
    rc, stdout, stderr, wall_ms = run_command(
        [str(PROBE9_EXEC), str(matrix_path)], timeout_sec)
    if rc == 0 and stdout:
        result.update(parse_probe9_output(stdout, wall_ms))
    else:
        msg = (stderr.strip() or str(rc))[:120]
        print("[WARN] probe9 failed: {} :: {}".format(matrix_path.name, msg))

    # 2) probeC AA
    rc, stdout, stderr, wall_ms = run_command(
        [str(PROBEC_EXEC), str(matrix_path)], timeout_sec)
    if rc == 0 and stdout:
        result.update(parse_probec_output(stdout, wall_ms, "probeC_AA", "pC_AA"))
    else:
        msg = (stderr.strip() or str(rc))[:120]
        print("[WARN] probeC AA failed: {} :: {}".format(matrix_path.name, msg))

    # 3) probeC AAT
    rc, stdout, stderr, wall_ms = run_command(
        [str(PROBEC_EXEC), "--aat", str(matrix_path)], timeout_sec)
    if rc == 0 and stdout:
        result.update(parse_probec_output(stdout, wall_ms, "probeC_AAT", "pC_AAT"))
    else:
        msg = (stderr.strip() or str(rc))[:120]
        print("[WARN] probeC AAT failed: {} :: {}".format(matrix_path.name, msg))

    return result


# ─── CSV update logic ───────────────────────────────────────────────────────

def build_index(rows):
    """Map matrix name -> list of row indices (by Name and Original_Name)."""
    index = {}
    for i, row in enumerate(rows):
        for key in (row.get("Name", "").strip(), row.get("Original_Name", "").strip()):
            if not key:
                continue
            index.setdefault(key, []).append(i)
    return index


def ensure_tools_exist():
    missing = []
    for path in (PROBE9_EXEC, PROBEC_EXEC):
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing executable(s): " + ", ".join(missing))


def update_split(split_name, timeout_sec):
    """Process all matrices in one split and update the CSV in place."""
    matrix_dir = SPLITS[split_name]["matrix_dir"]
    csv_path = SPLITS[split_name]["csv_path"]
    backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")

    matrices = sorted(matrix_dir.glob("*.mtx"))
    print("[{}] matrices={} csv={}".format(split_name, len(matrices), csv_path.name))

    # Read existing CSV
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        original_fields = list(reader.fieldnames or [])

    # Determine new columns (avoid duplicating if script is re-run)
    new_fields = [field for field in all_new_fields() if field not in original_fields]
    fieldnames = original_fields + new_fields

    # Initialize new columns to empty
    for row in rows:
        for field in new_fields:
            row.setdefault(field, "")

    # Build name -> row index mapping
    row_index = build_index(rows)

    # Process each matrix
    for idx, matrix_path in enumerate(matrices, 1):
        matrix_name = matrix_path.stem
        matches = row_index.get(matrix_name, [])
        if not matches:
            print("[WARN] no CSV row matched matrix: {}".format(matrix_name))
            continue

        sys.stdout.write("[{}] {}/{} {} ...".format(split_name, idx, len(matrices), matrix_name))
        sys.stdout.flush()
        result = process_matrix(matrix_path, timeout_sec)
        for row_i in matches:
            rows[row_i].update(result)
        print(" OK")

    # Backup original CSV, then overwrite
    shutil.copy2(str(csv_path), str(backup_path))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("[{}] done  -> {}".format(split_name, csv_path))
    print("[{}] backup -> {}".format(split_name, backup_path))


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch probe9 + probeC (AA/AAT) for FlexSpGEMM datasets")
    parser.add_argument(
        "--splits", nargs="+", default=["test", "train", "val"],
        choices=list(SPLITS.keys()),
        help="Which splits to process (default: all)")
    parser.add_argument(
        "--timeout", type=int, default=TIMEOUT_SEC,
        help="Timeout per matrix per tool in seconds (default: {})".format(TIMEOUT_SEC))
    args = parser.parse_args()

    ensure_tools_exist()
    for split_name in args.splits:
        update_split(split_name, args.timeout)

    print("\nAll done!")


if __name__ == "__main__":
    main()
