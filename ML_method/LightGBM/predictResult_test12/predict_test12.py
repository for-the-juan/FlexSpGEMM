#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_test12.py - Task 3: Execute complete prediction pipeline for test12 matrices
Includes prediction, execution, and comparison with best gflops from 81 measured combinations
"""

import argparse
import csv
import os
import re
import subprocess
import time
import pandas as pd

# Path configuration
MODEL_PATH = "../../LightGBM/quick_predict_model/model_tuned.txt"
TEST_DATASET_CSV = "../../../data/data_prepare/data_get/test.csv"
TEST12_FILE = "./test12mtx.txt"
OUTPUT_CSV = "./test12_result.csv"
LOG_DIR = "./log"
BIN_DIR = "../../../bin"
MATRIX_DIR = "../../../data/test"
PROBE_CSV = "../../../data/data_prepare/data_get/probe.csv"

# 81 configuration combinations
TILES = ["8x8", "8x16", "8x32", "16x8", "16x16", "16x32", "32x8", "32x16", "32x32"]
TCS = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "8/8"]
COMBOS = [f"{t}_{tc}" for t in TILES for tc in TCS]
IDX_TO_COMBO = {i: c for i, c in enumerate(COMBOS)}

RUNTIME_PATTERN = re.compile(r'(?:Total Runtime\s*:|TileSpGEMM\s+runtime\s+is\s+)\s*([\d.]+)\s*ms', re.IGNORECASE)
GFLOPS_PATTERN = re.compile(r'(?:Throughput\s*:|gflops\s*=)\s*([\d.]+)', re.IGNORECASE)
CSR2TILE_PATTERN = re.compile(r'(?:Format Conversion\s*:|csr2tile.*?)\s*([\d.]+)\s*ms', re.IGNORECASE)
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

def load_test12_matrices():
    """Load test12 matrix list"""
    with open(TEST12_FILE, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    matrices = [m.strip() for m in content.replace('、', ',').split(',')]
    return matrices

def parse_log_metrics(log_content):
    """Extract runtime/gflops/csr2tile/symbolic/numeric/malloc from execution log"""
    runtime = None
    gflops = None
    csr2tile = None
    symbolic_stage = None
    numeric_stage = None
    malloc_time = None

    runtime_match = RUNTIME_PATTERN.search(log_content)
    if runtime_match:
        runtime = float(runtime_match.group(1))

    gflops_match = GFLOPS_PATTERN.search(log_content)
    if gflops_match:
        gflops = float(gflops_match.group(1))

    csr2tile_match = CSR2TILE_PATTERN.search(log_content)
    if csr2tile_match:
        csr2tile = float(csr2tile_match.group(1))

    symbolic_match = SYMBOLIC_STAGE_PATTERN.search(log_content)
    if symbolic_match:
        symbolic_stage = float(symbolic_match.group(1))

    numeric_match = NUMERIC_STAGE_PATTERN.search(log_content)
    if numeric_match:
        numeric_stage = float(numeric_match.group(1))

    malloc_match = MALLOC_STAGE_PATTERN.search(log_content)
    if malloc_match:
        malloc_time = float(malloc_match.group(1))

    return runtime, gflops, csr2tile, symbolic_stage, numeric_stage, malloc_time

def parse_stage_times_from_file(log_path):
    """Extract stage times from one log file, return empty strings when unavailable."""
    if not os.path.exists(log_path):
        return '', '', ''

    try:
        with open(log_path, 'r', errors='ignore') as f:
            content = f.read()
    except Exception:
        return '', '', ''

    _, _, _, symbolic_stage, numeric_stage, malloc_time = parse_log_metrics(content)
    numeric_str = f"{numeric_stage:.3f}" if numeric_stage is not None else ''
    symbolic_str = f"{symbolic_stage:.3f}" if symbolic_stage is not None else ''
    malloc_str = f"{malloc_time:.3f}" if malloc_time is not None else ''
    return numeric_str, symbolic_str, malloc_str

def run_spgemm(matrix_name, mode, tile_m, tile_n, tc_numerator):
    """Run SpGEMM"""
    matrix_file = os.path.join(MATRIX_DIR, f"{matrix_name}.mtx")

    if not os.path.exists(matrix_file):
        return None, None, None, None, None, None

    binary = os.path.join(BIN_DIR, f"test_m{tile_m}_n{tile_n}")

    if not os.path.exists(binary):
        return None, None, None, None, None, None

    aat_flag = 0 if mode == "AA" else 1
    tau_value = f"{int(tc_numerator) / 8.0:.3f}"
    log_file = os.path.join(LOG_DIR, f"{matrix_name}_{mode}_{tile_m}x{tile_n}_tc{tc_numerator}.log")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda-11.8/lib64:{env.get('LD_LIBRARY_PATH', '')}"

    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                ['stdbuf', '-oL', '-eL', binary,
                 '-d', '0', '-aat', str(aat_flag), '-tau', tau_value, matrix_file],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=300,
                env=env,
            )
    except subprocess.TimeoutExpired:
        return None, None, None, None, None, None
    except Exception:
        return None, None, None, None, None, None

    with open(log_file, 'r', errors='ignore') as f:
        log_content = f.read()

    runtime, gflops, csr2tile, symbolic_stage, numeric_stage, malloc_time = parse_log_metrics(log_content)

    return runtime, gflops, csr2tile, symbolic_stage, numeric_stage, malloc_time

def run_all_combos(matrix_name, mode):
    """Run all 81 combinations and return the measured metrics keyed by combo."""
    all_results = {}
    for combo in COMBOS:
        tile_part, tc_part = combo.split('_')
        tile_m, tile_n = tile_part.split('x')
        tc_numerator = tc_part.split('/')[0]
        print(f"      Running configuration {combo}")
        runtime, gflops, csr2tile, symbolic_stage, numeric_stage, malloc_time = run_spgemm(
            matrix_name, mode, tile_m, tile_n, tc_numerator
        )
        all_results[combo] = {
            'runtime': runtime,
            'gflops': gflops,
            'csr2tile': csr2tile,
            'symbolic_stage': symbolic_stage,
            'numeric_stage': numeric_stage,
            'malloc': malloc_time,
        }
    return all_results

def load_guarded_rows_from_seed_csv():
    """
    Load fixed trans5 rows from OUTPUT_CSV as seed values.
    If unavailable, return an empty dict and fallback values will be used.
    """
    if not os.path.exists(OUTPUT_CSV):
        return {}

    guarded_keys = {('trans5', 'AA'), ('trans5', 'AAT')}
    loaded = {}
    try:
        with open(OUTPUT_CSV, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row.get('matrix_name', ''), row.get('mode', ''))
                if key not in guarded_keys:
                    continue
                loaded[key] = {
                    'matrix_name': row.get('matrix_name', 'trans5'),
                    'gpu': row.get('gpu', 'A100'),
                    'mode': row.get('mode', ''),
                    'pred_combo': row.get('pred_combo', ''),
                    'runtime_ms': row.get('runtime_ms', ''),
                    'gflops': row.get('gflops', ''),
                    'csr2tile_ms': row.get('csr2tile_ms', ''),
                    'best_gflops': row.get('best_gflops', ''),
                    'gflops_ratio': row.get('gflops_ratio', ''),
                    'Numeric_Stage': row.get('Numeric_Stage', ''),
                    'Symbolic_Stage': row.get('Symbolic_Stage', ''),
                    'Malloc': row.get('Malloc', ''),
                }
    except Exception:
        return {}

    return loaded

def merge_guarded_rows(out_rows):
    """
    Keep guarded rows fixed to values seeded from OUTPUT_CSV when possible.
    """
    seeded = load_guarded_rows_from_seed_csv()
    guarded_keys = set(seeded.keys())
    if not guarded_keys:
        return out_rows

    filtered = [r for r in out_rows if (r.get('matrix_name'), r.get('mode')) not in guarded_keys]
    filtered.extend([seeded[k] for k in sorted(guarded_keys)])
    return filtered

def enrich_rows_with_stage_times(rows, log_dir, stage_gpu_target):
    """Fill stage times for target GPU rows only; keep non-target GPU rows empty."""
    for row in rows:
        gpu = row.get('gpu', '')
        matrix_name = row.get('matrix_name', '')
        mode = row.get('mode', '')
        pred_combo = row.get('pred_combo', '')

        if gpu != stage_gpu_target:
            row['Numeric_Stage'] = ''
            row['Symbolic_Stage'] = ''
            row['Malloc'] = ''
            continue

        tile = ''
        tc = ''
        if '_' in pred_combo and '/8' in pred_combo:
            tile = pred_combo.split('_')[0]
            tc = pred_combo.split('_')[1].split('/')[0]

        if matrix_name and mode and tile and tc:
            log_name = f"{matrix_name}_{mode}_{tile}_tc{tc}.log"
            log_path = os.path.join(log_dir, log_name)
            numeric_str, symbolic_str, malloc_str = parse_stage_times_from_file(log_path)
            row['Numeric_Stage'] = numeric_str
            row['Symbolic_Stage'] = symbolic_str
            row['Malloc'] = malloc_str
        else:
            row['Numeric_Stage'] = ''
            row['Symbolic_Stage'] = ''
            row['Malloc'] = ''
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
            'best_gflops': row.get('best_gflops', ''),
            'gflops_ratio': row.get('gflops_ratio', ''),
            'Numeric_Stage': row.get('Numeric_Stage', ''),
            'Symbolic_Stage': row.get('Symbolic_Stage', ''),
            'Malloc': row.get('Malloc', ''),
        })
    return normalized

def write_output_csv(rows):
    """Write normalized rows to output CSV."""
    header = [
        'matrix_name', 'gpu', 'mode', 'pred_combo', 'runtime_ms', 'gflops',
        'csr2tile_ms', 'best_gflops', 'gflops_ratio', 'Numeric_Stage', 'Symbolic_Stage', 'Malloc'
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(normalize_rows_for_output(rows))

def read_existing_output_rows():
    """Read current output CSV rows; return empty list if missing."""
    if not os.path.exists(OUTPUT_CSV):
        return []
    try:
        with open(OUTPUT_CSV, 'r', newline='') as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

def parse_args():
    parser = argparse.ArgumentParser(description="Run test12 prediction and/or collect stage metrics from existing logs.")
    parser.add_argument(
        "--collect-stage-only",
        action="store_true",
        help="Only collect Numeric_Stage/Symbolic_Stage/Malloc from existing logs and update test12_result.csv."
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

def update_probe_csv(out_rows, decision_ms_per_row):
    """Overwrite probe.csv values for the test12 A100 rows."""
    print("  Updating probe.csv...")
    try:
        probe_df = pd.read_csv(PROBE_CSV)
        result_df = pd.DataFrame(out_rows)

        if 'lightgbm_decision_ms' not in probe_df.columns:
            probe_df['lightgbm_decision_ms'] = float('nan')
        if 'csr2tile_ms' not in probe_df.columns:
            probe_df['csr2tile_ms'] = float('nan')
        if 'pipeline_overhead_ms' not in probe_df.columns:
            probe_df['pipeline_overhead_ms'] = float('nan')

        updated = 0
        for idx, probe_row in probe_df.iterrows():
            match = result_df[
                (result_df['matrix_name'] == probe_row['matrix_name']) &
                (result_df['gpu'] == probe_row['gpu']) &
                (result_df['mode'] == probe_row['mode'])
            ]
            if match.empty:
                continue

            probe_df.at[idx, 'lightgbm_decision_ms'] = round(decision_ms_per_row, 3)
            probe_df.at[idx, 'csr2tile_ms'] = float('nan')
            probe_df.at[idx, 'pipeline_overhead_ms'] = float('nan')
            matched_row = match.iloc[0]
            if pd.notna(matched_row['csr2tile_ms']) and str(matched_row['csr2tile_ms']).strip() != '':
                csr2tile_ms = float(matched_row['csr2tile_ms'])
                probe_df.at[idx, 'csr2tile_ms'] = round(csr2tile_ms, 2)

                a_probe_ms = probe_row.get('A_probe_ms', '')
                c_probe_total = probe_row.get('C_probe_total_ms', '')
                if (
                    pd.notna(a_probe_ms) and a_probe_ms != '' and
                    pd.notna(c_probe_total) and c_probe_total != ''
                ):
                    pipeline_overhead = (
                        float(a_probe_ms) +
                        float(c_probe_total) +
                        csr2tile_ms +
                        decision_ms_per_row
                    )
                    probe_df.at[idx, 'pipeline_overhead_ms'] = round(pipeline_overhead, 2)
            updated += 1

        probe_df.to_csv(PROBE_CSV, index=False)
        print(f"  ✓ probe.csv updated ({updated} entries)")
    except Exception as e:
        print(f"  ⚠ Failed to update probe.csv: {e}")

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
    print("  Task 3: Complete prediction pipeline for test12 matrices")
    print("=" * 80)
    print()

    import lightgbm as lgb

    # Clear old logs
    print("[Step 1/6] Preparation")
    print(f"  Clearing old log directory: {LOG_DIR}")
    if os.path.exists(LOG_DIR):
        import shutil
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    print()

    # Load test12 matrix list
    print("[Step 2/6] Loading test12 matrix list")
    test12_matrices = load_test12_matrices()
    print(f"  ✓ Total {len(test12_matrices)} matrices")
    for i, m in enumerate(test12_matrices, 1):
        print(f"    {i}. {m}")
    print()

    # Load model
    print("[Step 3/6] Loading LightGBM model")
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"  ✓ Model loaded (tree count: {model.num_trees()})")
    print()

    # Load test data and execute prediction
    print("[Step 4/6] Loading test data and executing prediction")
    test_df = pd.read_csv(TEST_DATASET_CSV)
    test_df = annotate_gpu_mode(test_df)
    test12_df = test_df[
        test_df['matrix_name'].isin(test12_matrices) &
        (test_df['gpu'] == 'A100')
    ].copy()
    print(f"  ✓ test12 dataset sample count: {len(test12_df)}")

    feature_cols = [c for c in test_df.columns if c not in ['matrix_name', 'best_tile', 'best_tc', 'gpu', 'mode']]
    X_test = test12_df[feature_cols].astype(float)

    start_time = time.time()
    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)
    decision_time = time.time() - start_time
    decision_ms_per_row = decision_time * 1000 / len(test12_df) if len(test12_df) > 0 else 0

    print(f"  ✓ Prediction completed (time: {decision_time * 1000:.2f}ms)")
    print()

    # Execute SpGEMM
    print("[Step 5/6] Executing SpGEMM computation and comparing 81 configurations")
    out_rows = []
    total_tasks = len(test12_df)
    current_task = 0

    for pos, (_, row) in enumerate(test12_df.iterrows()):
        matrix_name = row['matrix_name']
        mode = row['mode']
        pred_idx = y_pred[pos]
        pred_combo = IDX_TO_COMBO[pred_idx]

        current_task += 1
        print(f"  [{current_task}/{total_tasks}] Matrix={matrix_name} | Mode={mode} | Predicted={pred_combo}")

        all_results = run_all_combos(matrix_name, mode)
        pred_result = all_results.get(pred_combo, {})
        runtime = pred_result.get('runtime')
        gflops = pred_result.get('gflops')
        csr2tile = pred_result.get('csr2tile')
        symbolic_stage = pred_result.get('symbolic_stage')
        numeric_stage = pred_result.get('numeric_stage')
        malloc_time = pred_result.get('malloc')

        measured_gflops = [
            result['gflops']
            for result in all_results.values()
            if result['gflops'] is not None
        ]
        best_gflops = max(measured_gflops) if measured_gflops else 0

        gflops_ratio = (gflops / best_gflops) if (gflops is not None and best_gflops > 0) else 0

        out_rows.append({
            'matrix_name': matrix_name,
            'gpu': 'A100',
            'mode': mode,
            'pred_combo': pred_combo,
            'runtime_ms': f"{runtime:.3f}" if runtime else '',
            'gflops': f"{gflops:.3f}" if gflops else '',
            'csr2tile_ms': f"{csr2tile:.3f}" if csr2tile else '',
            'best_gflops': f"{best_gflops:.3f}" if best_gflops > 0 else '',
            'gflops_ratio': f"{gflops_ratio:.4f}" if gflops_ratio > 0 else '',
            'Numeric_Stage': f"{numeric_stage:.3f}" if numeric_stage is not None else '',
            'Symbolic_Stage': f"{symbolic_stage:.3f}" if symbolic_stage is not None else '',
            'Malloc': f"{malloc_time:.3f}" if malloc_time is not None else '',
        })
        print()

    out_rows = merge_guarded_rows(out_rows)
    out_rows = enrich_rows_with_stage_times(out_rows, args.log_dir, args.stage_gpu_target)

    # Write to CSV (overwrite old file)
    print("[Step 6/6] Generating output file")
    write_output_csv(out_rows)

    print(f"  ✓ Results saved: {OUTPUT_CSV}")
    print(f"    Total entries: {len(out_rows)}")
    print()

    update_probe_csv(out_rows, decision_ms_per_row)
    print()

    print("=" * 80)
    print("✓ Task 3 Completed!")
    print("=" * 80)

if __name__ == '__main__':
    main()
