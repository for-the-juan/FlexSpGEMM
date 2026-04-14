#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1_build_probe_features.py
-----------------------------
Task 1: Run probe9 (tile_probe) and probeC on every matrix in test/train/val,
         produce per-matrix intermediate CSVs with all probe features and timing.
         Also produce probe.csv for test matrices (timing-report style).

Important:
  - train.csv / val.csv / test.csv in data_get/ are final outputs and must not
    be polluted by step1 timing columns.
  - step1 intermediate CSVs are therefore written into stage1_intermediate/.
  - probe.csv stores probe/build/estimate timings only; load time is excluded.

Usage (called by run_pipeline.py, but can also be run standalone):
    python3 step1_build_probe_features.py
"""

import csv
import sys
from pathlib import Path

# Ensure this script's directory is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline_utils as pu


# ─── Per-matrix probe runner ───────────────────────────────────────────────

def probe_one_matrix(mtx_path):
    """Run probe9 A, probe9 AT, probeC AA, probeC AAT on one matrix.
    Returns a dict with all intermediate columns."""
    name = mtx_path.stem
    row = {"matrix_name": name}

    # 1) probe9 A
    rc, stdout, stderr, wall_ms = pu.run_command([str(pu.PROBE9_EXEC), str(mtx_path)])
    if rc == 0 and stdout:
        p9_data, info = pu.parse_probe9(stdout)
        row.update(p9_data)
        row["rows"] = info["rows"]
        row["cols"] = info["cols"]
        row["nnz"] = info["nnz"]
        row["density"] = info["density"]
        row["avg_nnz_per_row"] = info["avg_nnz_per_row"]
        row["symmetric"] = info["symmetric"]
        row["probe9_A_wall_ms"] = "{:.3f}".format(wall_ms)
        row["probe9_A_probe_ms"] = info["probe_time_ms"]
    else:
        row["probe9_A_wall_ms"] = "{:.3f}".format(wall_ms)
        print("[WARN] probe9 A failed: {} :: {}".format(name, (stderr.strip() or str(rc))[:80]))

    # 2) probe9 AT
    is_symmetric = row.get("symmetric", "").lower() == "yes"
    at_path = pu.MTX_T_DIR / "{}.mtx".format(name)

    if is_symmetric:
        # Copy A features as AT features
        for tile in pu.P9_TILE_ORDER:
            for f in pu.P9_FIELDS:
                row["p9_AT_{}_{}".format(tile, f)] = row.get("p9_{}_{}".format(tile, f), "")
        row["probe9_AT_wall_ms"] = "0"
        row["probe9_AT_probe_ms"] = "0"
    elif at_path.exists():
        rc, stdout, stderr, wall_ms = pu.run_command([str(pu.PROBE9_EXEC), str(at_path)])
        if rc == 0 and stdout:
            at_data = pu.parse_probe9_at(stdout)
            row.update(at_data)
            row["probe9_AT_wall_ms"] = "{:.3f}".format(wall_ms)
            row["probe9_AT_probe_ms"] = pu._extract(r"Probe time: ([\d.]+) ms", stdout)
        else:
            row["probe9_AT_wall_ms"] = "{:.3f}".format(wall_ms)
            print("[WARN] probe9 AT failed: {} :: {}".format(name, (stderr.strip() or str(rc))[:80]))
    else:
        print("[WARN] No transpose file for non-symmetric matrix: {}".format(name))

    # 3) probeC AA
    rc, stdout, stderr, wall_ms = pu.run_command([str(pu.PROBEC_EXEC), str(mtx_path)])
    if rc == 0 and stdout:
        pc_data, pc_timing = pu.parse_probec(stdout, "pC_AA")
        row.update(pc_data)
        row["probeC_AA_wall_ms"] = "{:.3f}".format(wall_ms)
        row["probeC_AA_load_ms"] = pc_timing["load_ms"]
        row["probeC_AA_build_ms"] = pc_timing["build_ms"]
        row["probeC_AA_estimate_ms"] = pc_timing["estimate_ms"]
    else:
        row["probeC_AA_wall_ms"] = "{:.3f}".format(wall_ms)
        print("[WARN] probeC AA failed: {} :: {}".format(name, (stderr.strip() or str(rc))[:80]))

    # 4) probeC AAT
    rc, stdout, stderr, wall_ms = pu.run_command([str(pu.PROBEC_EXEC), "--aat", str(mtx_path)])
    if rc == 0 and stdout:
        pc_data, pc_timing = pu.parse_probec(stdout, "pC_AAT")
        row.update(pc_data)
        row["probeC_AAT_wall_ms"] = "{:.3f}".format(wall_ms)
        row["probeC_AAT_load_ms"] = pc_timing["load_ms"]
        row["probeC_AAT_build_ms"] = pc_timing["build_ms"]
        row["probeC_AAT_estimate_ms"] = pc_timing["estimate_ms"]
    else:
        row["probeC_AAT_wall_ms"] = "{:.3f}".format(wall_ms)
        print("[WARN] probeC AAT failed: {} :: {}".format(name, (stderr.strip() or str(rc))[:80]))

    return row


# ─── Intermediate CSV schema ──────────────────────────────────────────────

def build_stage1_header():
    """Return the column names for step1 intermediate CSV."""
    cols = ["matrix_name", "rows", "cols", "nnz", "density", "avg_nnz_per_row", "symmetric"]
    # probe9 A
    for tile in pu.P9_TILE_ORDER:
        for f in pu.P9_FIELDS:
            cols.append("p9_{}_{}".format(tile, f))
    # probe9 AT
    for tile in pu.P9_TILE_ORDER:
        for f in pu.P9_FIELDS:
            cols.append("p9_AT_{}_{}".format(tile, f))
    # probeC AA (for probe.csv and internal use)
    for tile in pu.PC_TILE_ORDER:
        for f in pu.PC_FIELDS:
            cols.append("pC_AA_{}_{}".format(tile, f))
    # probeC AAT
    for tile in pu.PC_TILE_ORDER:
        for f in pu.PC_FIELDS:
            cols.append("pC_AAT_{}_{}".format(tile, f))
    # timing
    cols.extend(pu.STAGE1_TIMING_COLS)
    return cols


# ─── probe.csv builder (timing only) ──────────────────────────────────────

PROBE_CSV_HEADER = [
    "matrix_name", "mode", "gpu",
    "A_probe_ms", "C_build_ms", "C_estimate_ms", "C_probe_total_ms",
    "lightgbm_decision_ms", "csr2tile_ms", "pipeline_overhead_ms",
]


def build_probe_csv_header():
    return list(PROBE_CSV_HEADER)


def expand_test_to_probe_rows(stage1_row, hw_data):
    """Expand one test matrix's stage1 data into probe.csv rows (timing only)."""
    rows = []
    name = stage1_row.get("matrix_name", "")
    is_sym = stage1_row.get("symmetric", "").lower() == "yes"

    combos = []
    for gpu in ("A100", "H200"):
        combos.append((gpu, "AA"))
        if not is_sym:
            combos.append((gpu, "AAT"))

    for gpu, mode in combos:
        r = {"matrix_name": name, "mode": mode, "gpu": gpu}
        if mode == "AA":
            # A_probe/C_build/C_estimate only keep pure probe computation time.
            # They intentionally exclude matrix load time and process wall time.
            r["A_probe_ms"] = stage1_row.get("probe9_A_probe_ms", "")
            r["C_build_ms"] = stage1_row.get("probeC_AA_build_ms", "")
            r["C_estimate_ms"] = stage1_row.get("probeC_AA_estimate_ms", "")
        else:
            r["A_probe_ms"] = stage1_row.get("probe9_AT_probe_ms", "")
            r["C_build_ms"] = stage1_row.get("probeC_AAT_build_ms", "")
            r["C_estimate_ms"] = stage1_row.get("probeC_AAT_estimate_ms", "")
        try:
            r["C_probe_total_ms"] = "{:.3f}".format(
                float(r.get("C_build_ms") or 0) + float(r.get("C_estimate_ms") or 0))
        except (ValueError, TypeError):
            r["C_probe_total_ms"] = ""
        r["lightgbm_decision_ms"] = ""
        r["csr2tile_ms"] = ""
        r["pipeline_overhead_ms"] = ""
        rows.append(r)
    return rows



# ─── Main ──────────────────────────────────────────────────────────────────

def run_step1():
    print("=" * 70)
    print("  Step 1: Build probe features")
    print("=" * 70)

    # 0) Check executables
    for exe in (pu.PROBE9_EXEC, pu.PROBEC_EXEC):
        if not exe.exists():
            print("[ERROR] Missing executable: {}".format(exe))
            sys.exit(1)

    # 1) Check transpose matrix directory
    if not pu.MTX_T_DIR.is_dir():
        print("[ERROR] Transpose matrix directory not found: {}".format(pu.MTX_T_DIR))
        print("        Please ensure transposed matrices are placed in mtx_T/ before running.")
        sys.exit(1)
    t_count = len(list(pu.MTX_T_DIR.glob("*.mtx")))
    print("[step1] Transpose matrices available: {}".format(t_count))

    # 2) Read hardware
    hw_data = pu.read_hardware()

    # 3) Build output dirs
    pu.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pu.STAGE1_DIR.mkdir(parents=True, exist_ok=True)

    stage1_header = build_stage1_header()
    probe_header = build_probe_csv_header()
    probe_rows = []

    for split in ("test", "train", "val"):
        mtx_dir = pu.MTX_DIRS[split]
        matrices = sorted(mtx_dir.glob("*.mtx"))
        out_csv = pu.STAGE1_DIR / "{}.csv".format(split)

        print("\n[step1] Processing {} ({} matrices) -> {}".format(split, len(matrices), out_csv))

        all_rows = []
        for idx, mtx_path in enumerate(matrices, 1):
            sys.stdout.write("  [{}/{}] {} ...".format(idx, len(matrices), mtx_path.stem))
            sys.stdout.flush()
            row = probe_one_matrix(mtx_path)
            all_rows.append(row)
            print(" OK")

            # For test set, also build probe.csv rows
            if split == "test":
                probe_rows.extend(expand_test_to_probe_rows(row, hw_data))

        # Write stage1 intermediate CSV
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=stage1_header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

        print("[step1] {} done: {} rows -> {}".format(split, len(all_rows), out_csv))

    # Write probe.csv
    probe_csv = pu.OUTPUT_DIR / "probe.csv"
    with probe_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=probe_header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(probe_rows)
    print("\n[step1] probe.csv: {} rows -> {}".format(len(probe_rows), probe_csv))
    print("[step1] intermediate CSVs -> {}".format(pu.STAGE1_DIR))

    print("\n[step1] Step 1 complete!")


if __name__ == "__main__":
    run_step1()
