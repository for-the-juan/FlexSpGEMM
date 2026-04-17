#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_test12.py - Task 3: Execute complete prediction pipeline for test12 matrices
Includes prediction, execution, and comparison with best gflops from 81 measured combinations
"""

import csv
import os
import re
import subprocess
import time
import pandas as pd
import lightgbm as lgb

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
    """Extract runtime/gflops/csr2tile from execution log"""
    runtime = None
    gflops = None
    csr2tile = None

    runtime_match = RUNTIME_PATTERN.search(log_content)
    if runtime_match:
        runtime = float(runtime_match.group(1))

    gflops_match = GFLOPS_PATTERN.search(log_content)
    if gflops_match:
        gflops = float(gflops_match.group(1))

    csr2tile_match = CSR2TILE_PATTERN.search(log_content)
    if csr2tile_match:
        csr2tile = float(csr2tile_match.group(1))

    return runtime, gflops, csr2tile

def run_spgemm(matrix_name, mode, tile_m, tile_n, tc_numerator):
    """Run SpGEMM"""
    matrix_file = os.path.join(MATRIX_DIR, f"{matrix_name}.mtx")

    if not os.path.exists(matrix_file):
        print(f"      ⚠ Matrix file not found")
        return None, None, None

    binary = os.path.join(BIN_DIR, f"test_m{tile_m}_n{tile_n}")

    if not os.path.exists(binary):
        print(f"      ⚠ Binary file not found: {binary}")
        return None, None, None

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
        print(f"      ✗ Execution timeout (>300s)")
        return None, None, None
    except Exception as e:
        print(f"      ✗ Execution failed: {e}")
        return None, None, None

    with open(log_file, 'r', errors='ignore') as f:
        log_content = f.read()

    runtime, gflops, csr2tile = parse_log_metrics(log_content)

    if gflops is not None:
        print(f"      ✓ Execution successful | runtime={runtime:.3f}ms | gflops={gflops:.3f}")
    elif "does not do symmetric matrix" in log_content.lower():
        print(f"      ⊘ Skip symmetric matrix (AAT mode not supported)")
    elif result.returncode != 0:
        print(f"      ✗ Program crashed (exit code: {result.returncode}), no data obtained")
    else:
        print(f"      ⚠ Execution completed but no performance data found")

    return runtime, gflops, csr2tile

def run_all_combos(matrix_name, mode):
    """Run all 81 combinations and return the measured metrics keyed by combo."""
    all_results = {}
    for combo in COMBOS:
        tile_part, tc_part = combo.split('_')
        tile_m, tile_n = tile_part.split('x')
        tc_numerator = tc_part.split('/')[0]
        print(f"      Running configuration {combo}")
        runtime, gflops, csr2tile = run_spgemm(matrix_name, mode, tile_m, tile_n, tc_numerator)
        all_results[combo] = {
            'runtime': runtime,
            'gflops': gflops,
            'csr2tile': csr2tile,
        }
    return all_results

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
    print("=" * 80)
    print("  Task 3: Complete prediction pipeline for test12 matrices")
    print("=" * 80)
    print()

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
        print(f"  [{current_task}/{total_tasks}] {matrix_name} | {mode} | Predicted config={pred_combo}")

        all_results = run_all_combos(matrix_name, mode)
        pred_result = all_results.get(pred_combo, {})
        runtime = pred_result.get('runtime')
        gflops = pred_result.get('gflops')
        csr2tile = pred_result.get('csr2tile')

        measured_gflops = [
            result['gflops']
            for result in all_results.values()
            if result['gflops'] is not None
        ]
        best_gflops = max(measured_gflops) if measured_gflops else 0

        if gflops is not None:
            print(f"      → Decision config measured gflops={gflops:.3f}")
        else:
            print(f"      ⚠ Decision config did not obtain valid gflops")

        if best_gflops > 0:
            print(f"      ✓ 81 combinations best measured gflops={best_gflops:.3f}")

        gflops_ratio = (gflops / best_gflops) if (gflops is not None and best_gflops > 0) else 0
        if gflops_ratio > 0:
            print(f"      → gflops_ratio={gflops_ratio:.4f} ({gflops_ratio*100:.2f}%)")

        out_rows.append({
            'matrix_name': matrix_name,
            'gpu': 'A100',
            'mode': mode,
            'pred_combo': pred_combo,
            'runtime_ms': f"{runtime:.3f}" if runtime else '',
            'gflops': f"{gflops:.3f}" if gflops else '',
            'csr2tile_ms': f"{csr2tile:.3f}" if csr2tile else '',
            'best_gflops': f"{best_gflops:.3f}" if best_gflops > 0 else '',
            'gflops_ratio': f"{gflops_ratio:.4f}" if gflops_ratio > 0 else ''
        })
        print()

    # Write to CSV (overwrite old file)
    print("[Step 6/6] Generating output file")
    header = ['matrix_name', 'gpu', 'mode', 'pred_combo', 'runtime_ms', 'gflops',
              'csr2tile_ms', 'best_gflops', 'gflops_ratio']

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"  ✓ Results saved: {OUTPUT_CSV}")
    print(f"    Total entries: {len(out_rows)}")
    print()

    update_probe_csv(out_rows, decision_ms_per_row)
    print()

    # Statistics
    ratios = [float(r['gflops_ratio']) for r in out_rows if r['gflops_ratio']]
    if ratios:
        print("  GFLOPS Ratio Statistics:")
        print(f"    Average: {sum(ratios)/len(ratios):.4f} ({sum(ratios)/len(ratios)*100:.2f}%)")
        print(f"    Minimum: {min(ratios):.4f} ({min(ratios)*100:.2f}%)")
        print(f"    Maximum: {max(ratios):.4f} ({max(ratios)*100:.2f}%)")
        print(f"    ≥95%: {sum(1 for r in ratios if r >= 0.95)}/{len(ratios)}")
        print(f"    ≥90%: {sum(1 for r in ratios if r >= 0.90)}/{len(ratios)}")

    print()
    print("=" * 80)
    print("✓ Task 3 Completed!")
    print("=" * 80)

if __name__ == '__main__':
    main()
