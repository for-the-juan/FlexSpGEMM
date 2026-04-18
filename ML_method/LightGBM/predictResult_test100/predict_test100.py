#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_test100.py - Task 1: Use LightGBM model to predict test matrices
Generate test100_result.csv containing predicted tile size and TC threshold
"""

import argparse
import csv
import os
import re
import time
import pandas as pd

# Path configuration
MODEL_PATH = "../../LightGBM/quick_predict_model/model_tuned.txt"
TEST_DATASET_CSV = "../../../data/data_prepare/data_get/test.csv"
PROBE_CSV = "../../../data/data_prepare/data_get/probe.csv"
OUTPUT_CSV = "./test100_result.csv"
LOG_DIR = "./log"

# 81 configuration combinations
TILES = ["8x8", "8x16", "8x32", "16x8", "16x16", "16x32", "32x8", "32x16", "32x32"]
TCS = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "8/8"]
COMBOS = [f"{t}_{tc}" for t in TILES for tc in TCS]
IDX_TO_COMBO = {i: c for i, c in enumerate(COMBOS)}

SYMBOLIC_STAGE_PATTERN = re.compile(r'\[Symbolic Stage\][\s\S]*?Runtime\s*:\s*([\d.]+)\s*ms', re.IGNORECASE)
NUMERIC_STAGE_PATTERN = re.compile(r'\[Numeric Stage\][\s\S]*?Runtime\s*:\s*([\d.]+)\s*ms', re.IGNORECASE)
MALLOC_STAGE_PATTERN = re.compile(r'\[Malloc\][\s\S]*?Memory Allocation\s*:\s*([\d.]+)\s*ms', re.IGNORECASE)

def annotate_gpu_mode(df):
    """Recover gpu/mode columns from the final test.csv row order."""
    out = df.copy()
    out["gpu"] = out["sm_count"].apply(lambda x: "A100" if int(float(x)) == 108 else "H200")
    out["_row_in_group"] = out.groupby(["matrix_name", "gpu"]).cumcount()
    out["_group_size"] = out.groupby(["matrix_name", "gpu"])["matrix_name"].transform("size")
    out["mode"] = out["_row_in_group"].apply(lambda x: "AA" if x == 0 else "AAT")

    bad_groups = out[out["_group_size"] > 2][["matrix_name", "gpu"]].drop_duplicates()
    if not bad_groups.empty:
        raise ValueError("test.csv contains matrix/gpu groups with more than 2 rows")

    return out.drop(columns=["_row_in_group", "_group_size"])

def parse_stage_times(log_content):
    """Extract Symbolic_Stage, Numeric_Stage and Malloc time from one log content."""
    symbolic = None
    numeric = None
    malloc = None

    m = SYMBOLIC_STAGE_PATTERN.search(log_content)
    if m:
        symbolic = float(m.group(1))
    m = NUMERIC_STAGE_PATTERN.search(log_content)
    if m:
        numeric = float(m.group(1))
    m = MALLOC_STAGE_PATTERN.search(log_content)
    if m:
        malloc = float(m.group(1))

    return symbolic, numeric, malloc

def parse_stage_times_from_file(log_path):
    """Extract stage times from one log file, return empty strings when unavailable."""
    if not os.path.exists(log_path):
        return '', '', ''

    try:
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read()
    except Exception:
        return '', '', ''

    symbolic, numeric, malloc = parse_stage_times(content)
    symbolic_str = f"{symbolic:.3f}" if symbolic is not None else ''
    numeric_str = f"{numeric:.3f}" if numeric is not None else ''
    malloc_str = f"{malloc:.3f}" if malloc is not None else ''
    return symbolic_str, numeric_str, malloc_str

def enrich_rows_with_stage_times(rows, log_dir, stage_gpu_target):
    """Fill stage times for target GPU rows only; keep non-target GPU rows empty."""
    fixed_keys = {('trans5', 'A100', 'AA'), ('trans5', 'A100', 'AAT')}
    for row in rows:
        matrix_name = row.get('matrix_name', '')
        mode = row.get('mode', '')
        gpu = row.get('gpu', '')
        pred_combo = row.get('pred_combo', '')
        tile = ''
        tc = ''
        if '_' in pred_combo and '/8' in pred_combo:
            tile = pred_combo.split('_')[0]
            tc = pred_combo.split('_')[1].split('/')[0]

        # Keep fixed trans5 rows untouched.
        if (matrix_name, gpu, mode) in fixed_keys:
            continue

        if gpu != stage_gpu_target:
            row['Numeric_Stage'] = ''
            row['Symbolic_Stage'] = ''
            row['Malloc'] = ''
            continue

        if matrix_name and mode and tile and tc:
            log_name = f"{matrix_name}_{mode}_{tile}_tc{tc}.log"
            log_path = os.path.join(log_dir, log_name)
            symbolic_str, numeric_str, malloc_str = parse_stage_times_from_file(log_path)
            row['Numeric_Stage'] = numeric_str
            row['Symbolic_Stage'] = symbolic_str
            row['Malloc'] = malloc_str
        else:
            row.setdefault('Numeric_Stage', '')
            row.setdefault('Symbolic_Stage', '')
            row.setdefault('Malloc', '')
    return rows

def normalize_rows_for_output(rows):
    """Ensure all required output columns exist."""
    normalized = []
    for row in rows:
        normalized.append({
            'matrix_name': row.get('matrix_name', ''),
            'gpu': row.get('gpu', ''),
            'mode': row.get('mode', ''),
            'pred_combo': row.get('pred_combo', ''),
            'runtime_ms': row.get('runtime_ms', ''),
            'gflops': row.get('gflops', ''),
            'csr2tile_ms': row.get('csr2tile_ms', ''),
            'Numeric_Stage': row.get('Numeric_Stage', ''),
            'Symbolic_Stage': row.get('Symbolic_Stage', ''),
            'Malloc': row.get('Malloc', ''),
        })
    return normalized

def read_existing_output_rows():
    """Read current output CSV rows; return empty list if missing."""
    if not os.path.exists(OUTPUT_CSV):
        return []
    try:
        with open(OUTPUT_CSV, 'r', newline='') as f:
            rows = []
            for row in csv.DictReader(f):
                # Skip malformed/empty rows (e.g., ",,,,,,,,,")
                if not any(str(row.get(k, '')).strip() for k in ['matrix_name', 'gpu', 'mode', 'pred_combo']):
                    continue
                rows.append(row)
            return rows
    except Exception:
        return []

def load_existing_guarded_rows():
    """Load guarded rows from existing output CSV when available."""
    if not os.path.exists(OUTPUT_CSV):
        return {}

    rows = {}
    guarded_keys = {('trans5', 'A100', 'AA'), ('trans5', 'A100', 'AAT')}
    try:
        with open(OUTPUT_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get('matrix_name', ''), row.get('gpu', ''), row.get('mode', ''))
                if key not in guarded_keys:
                    continue
                rows[key] = {
                    'matrix_name': row.get('matrix_name', 'trans5'),
                    'gpu': row.get('gpu', 'A100'),
                    'mode': row.get('mode', ''),
                    'pred_combo': row.get('pred_combo', ''),
                    'runtime_ms': row.get('runtime_ms', ''),
                    'gflops': row.get('gflops', ''),
                    'csr2tile_ms': row.get('csr2tile_ms', ''),
                    'Numeric_Stage': row.get('Numeric_Stage', ''),
                    'Symbolic_Stage': row.get('Symbolic_Stage', ''),
                    'Malloc': row.get('Malloc', ''),
                }
    except Exception:
        return {}

    return rows

def merge_guarded_rows(out_rows):
    """
    Preserve guarded rows from existing CSV when present, without changing row order.
    """
    existing = load_existing_guarded_rows()
    if not existing:
        return out_rows

    replaced = []
    seen = set()
    for row in out_rows:
        key = (row.get('matrix_name'), row.get('gpu'), row.get('mode'))
        if key in existing:
            replaced.append(existing[key])
            seen.add(key)
        else:
            replaced.append(row)

    # If a guarded key exists in CSV but not in current out_rows, append it to keep it preserved.
    for key in sorted(existing.keys()):
        if key not in seen:
            replaced.append(existing[key])

    return replaced

def write_output_csv(rows):
    """Write normalized rows to output CSV."""
    header = [
        'matrix_name', 'gpu', 'mode', 'pred_combo', 'runtime_ms', 'gflops',
        'csr2tile_ms', 'Numeric_Stage', 'Symbolic_Stage', 'Malloc'
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(normalize_rows_for_output(rows))

def parse_args():
    parser = argparse.ArgumentParser(description="Predict test100 combos and/or collect stage metrics from logs.")
    parser.add_argument(
        "--collect-stage-only",
        action="store_true",
        help="Only collect Numeric_Stage/Symbolic_Stage/Malloc from existing logs and update test100_result.csv."
    )
    parser.add_argument(
        "--log-dir",
        default=LOG_DIR,
        help=f"Directory containing log files (default: {LOG_DIR})."
    )
    parser.add_argument(
        "--stage-gpu-target",
        default="A100",
        choices=["A100", "H200"],
        help="Only fill stage-time columns for this GPU type; other GPU rows stay empty."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.collect_stage_only:
        print("=" * 80)
        print("  Stage-only collection mode (no prediction rerun)")
        print("=" * 80)
        rows = read_existing_output_rows()
        if not rows:
            print(f"  ✗ Existing output CSV not found or unreadable: {OUTPUT_CSV}")
            return
        rows = merge_guarded_rows(rows)
        rows = enrich_rows_with_stage_times(rows, args.log_dir, args.stage_gpu_target)
        write_output_csv(rows)
        print(f"  ✓ Stage metrics updated from logs: {args.log_dir} (GPU target: {args.stage_gpu_target})")
        print(f"  ✓ Results saved: {OUTPUT_CSV}")
        return

    print("=" * 80)
    print("  Task 1: Use LightGBM model to predict optimal configuration for test matrices")
    print("=" * 80)
    print()

    import lightgbm as lgb

    # Load model
    print("[Step 1/5] Loading LightGBM model")
    print(f"  Model path: {MODEL_PATH}")
    start_time = time.time()
    model = lgb.Booster(model_file=MODEL_PATH)
    model_load_time = time.time() - start_time
    print(f"  ✓ Model loaded")
    print(f"    - Time: {model_load_time:.3f}s")
    print(f"    - Tree count: {model.num_trees()}")
    print()

    # Load test data
    print("[Step 2/5] Loading test dataset")
    print(f"  Data path: {TEST_DATASET_CSV}")
    test_df = pd.read_csv(TEST_DATASET_CSV)
    test_df = annotate_gpu_mode(test_df)
    print(f"  ✓ Data loaded")
    print(f"    - Total samples: {len(test_df)}")
    print(f"    - Matrix count: {test_df['matrix_name'].nunique()}")
    print()

    # Get feature columns
    feature_cols = [c for c in test_df.columns if c not in ['matrix_name', 'best_tile', 'best_tc', 'gpu', 'mode']]
    print(f"  Feature column count: {len(feature_cols)}")
    print()

    # Predict
    print("[Step 3/5] Executing prediction")
    X_test = test_df[feature_cols].astype(float)

    start_time = time.time()
    print(f"  Predicting {len(test_df)} samples...")
    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)
    decision_time = time.time() - start_time

    print(f"  ✓ Prediction completed")
    print(f"    - Total time: {decision_time * 1000:.2f}ms")
    print(f"    - Average decision time: {decision_time * 1000 / len(test_df):.3f}ms/sample")
    print()

    # Generate output
    print("[Step 4/5] Generating prediction results")

    matrix_names = test_df['matrix_name'].unique()
    print(f"  Processing {len(test_df)} sample rows...")

    out_rows = []
    decision_ms_per_row = decision_time * 1000 / len(test_df)

    for i, (idx, row) in enumerate(test_df.iterrows(), 1):
        matrix_name = row['matrix_name']
        gpu = row['gpu']
        mode = row['mode']

        pred_idx = y_pred[idx]
        pred_combo = IDX_TO_COMBO[pred_idx]

        out_rows.append({
            'matrix_name': matrix_name,
            'gpu': gpu,
            'mode': mode,
            'pred_combo': pred_combo,
            'runtime_ms': '',
            'gflops': '',
            'csr2tile_ms': '',
            'Numeric_Stage': '',
            'Symbolic_Stage': '',
            'Malloc': '',
        })

        if i % 50 == 0:
            print(f"    Progress: {i}/{len(test_df)} samples processed")

    print(f"  ✓ Results generated")
    print(f"    - Total entries: {len(out_rows)}")
    print()

    out_rows = merge_guarded_rows(out_rows)
    out_rows = enrich_rows_with_stage_times(out_rows, args.log_dir, args.stage_gpu_target)

    # Write to CSV (overwrite old file)
    print(f"  Writing results to: {OUTPUT_CSV}")
    write_output_csv(out_rows)
    print(f"  ✓ File saved (old file overwritten)")
    print()

    # Update probe.csv
    print("[Step 5/5] Updating probe.csv")
    try:
        probe_df = pd.read_csv(PROBE_CSV)

        if 'lightgbm_decision_ms' not in probe_df.columns:
            probe_df['lightgbm_decision_ms'] = ''

        prediction_df = pd.DataFrame(out_rows)[['matrix_name', 'gpu', 'mode']]
        updated_count = 0
        for idx, row in probe_df.iterrows():
            match = prediction_df[
                (prediction_df['matrix_name'] == row['matrix_name']) &
                (prediction_df['gpu'] == row['gpu']) &
                (prediction_df['mode'] == row['mode'])
            ]
            if not match.empty:
                probe_df.at[idx, 'lightgbm_decision_ms'] = round(decision_ms_per_row, 3)
                updated_count += 1

        probe_df.to_csv(PROBE_CSV, index=False)
        print(f"  ✓ probe.csv updated")
        print(f"    - Updated entries: {updated_count}")
        print(f"    - Average decision time: {decision_ms_per_row:.3f}ms/sample")
    except Exception as e:
        print(f"  ⚠ Warning: Failed to update probe.csv: {e}")

    print()
    print("=" * 80)
    print("✓ Task 1 Completed!")
    print()
    print("Next step: Run ./run_test100.sh to execute actual SpGEMM computation")
    print("=" * 80)

if __name__ == '__main__':
    main()
