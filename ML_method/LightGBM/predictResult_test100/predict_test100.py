#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_test100.py - Task 1: Use LightGBM model to predict test matrices
Generate test100_result.csv containing predicted tile size and TC threshold
"""

import csv
import time
import pandas as pd
import lightgbm as lgb

# Path configuration
MODEL_PATH = "../../LightGBM/quick_predict_model/model_tuned.txt"
TEST_DATASET_CSV = "../../../data/data_prepare/data_get/test.csv"
PROBE_CSV = "../../../data/data_prepare/data_get/probe.csv"
OUTPUT_CSV = "./test100_result.csv"

# 81 configuration combinations
TILES = ["8x8", "8x16", "8x32", "16x8", "16x16", "16x32", "32x8", "32x16", "32x32"]
TCS = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "8/8"]
COMBOS = [f"{t}_{tc}" for t in TILES for tc in TCS]
IDX_TO_COMBO = {i: c for i, c in enumerate(COMBOS)}

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

def main():
    print("=" * 80)
    print("  Task 1: Use LightGBM model to predict optimal configuration for test matrices")
    print("=" * 80)
    print()

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
            'csr2tile_ms': ''
        })

        if i % 50 == 0:
            print(f"    Progress: {i}/{len(test_df)} samples processed")

    print(f"  ✓ Results generated")
    print(f"    - Total entries: {len(out_rows)}")
    print()

    # Write to CSV (overwrite old file)
    print(f"  Writing results to: {OUTPUT_CSV}")
    header = ['matrix_name', 'gpu', 'mode', 'pred_combo', 'runtime_ms', 'gflops', 'csr2tile_ms']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)
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
