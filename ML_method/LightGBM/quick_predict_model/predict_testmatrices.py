#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_testset.py - Generate decision CSV for test matrices using tuned LightGBM model
"""

import csv, os, re, json
import numpy as np
import lightgbm as lgb
import pandas as pd

BASE_DIR   = "/home/stu1/donghangcheng/lightGBM_A_AT_0327100"
DATA_DIR    = "/home/stu1/donghangcheng/lightGBM_A_AT_0327100/all_data"
MODEL_PATH  = "./model_tuned.txt"
OUTPUT_CSV  = "./100_testmatrices_results.csv"

# gflops data sources
A100_AA_GFLOPS  = "/home/stu1/donghangcheng/finalLightgbm_baseline/gflops_data/a100_gflops_all.csv"
H200_AA_GFLOPS  = "/home/stu1/donghangcheng/finalLightgbm_baseline/gflops_data/h200_gflops_all.csv"
A100_AAT_SRC    = "/home/stu1/donghangcheng/lightGBM_A_AT_0327100/A100_data_AAT/training_dataset_aat_a100"
H200_AAT_SRC    = "/home/stu1/donghangcheng/lightGBM_A_AT_0327100/H200_data_AAT/training_dataset_aat_h200"

# 81 configs
TILES = ["8x8","8x16","8x32","16x8","16x16","16x32","32x8","32x16","32x32"]
TCS   = ["0/8","1/8","2/8","3/8","4/8","5/8","6/8","7/8","8/8"]
COMBOS = [f"{t}_{tc}" for t in TILES for tc in TCS]
COMBO_TO_IDX = {c: i for i, c in enumerate(COMBOS)}
IDX_TO_COMBO = {i: c for c, i in COMBO_TO_IDX.items()}


def build_aa_gflops(csv_path):
    """Load existing AA gflops CSV -> {matrix_name: {combo: gflops}}"""
    gf = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            name = row['matrix_name']
            gf[name] = {combo: float(row.get(combo, 0)) for combo in COMBOS}
    return gf


def build_a100_aat_gflops():
    """Extract gflops from A100 AAT txt files -> {matrix_name: {combo: gflops}}"""
    gf = {}
    config_re = re.compile(r'配置\s+\d+/81:\s+tile=(\d+)x(\d+),\s+TC=(\d+/8)')
    gflops_re2 = re.compile(r'(?:CUDA\s+)?TileSpGEMM\s+runtime\s+is\s+([\d.]+)\s*ms,\s*gflops\s*=\s*([\d.]+)')
    
    for cat_dir in sorted(os.listdir(A100_AAT_SRC)):
        cat_path = os.path.join(A100_AAT_SRC, cat_dir)
        if not os.path.isdir(cat_path): 
            continue
        for fname in os.listdir(cat_path):
            if not fname.endswith('.txt'): 
                continue
            name = fname.replace('.txt', '')
            fpath = os.path.join(cat_path, fname)
            vals = {}
            with open(fpath) as f:
                content = f.read()
            # Log format: "配置 X/81: tile=MxN, TC=T/8" followed by gflops output
            configs = list(config_re.finditer(content))
            if configs:
                for i, m in enumerate(configs):
                    start = m.end()
                    end = configs[i+1].start() if i+1 < len(configs) else len(content)
                    section = content[start:end]
                    gm = gflops_re2.search(section)
                    if gm:
                        combo = f"{m.group(1)}x{m.group(2)}_{m.group(3)}"
                        vals[combo] = float(gm.group(2))
            else:
                # Tabular format fallback
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('#') or line.startswith('tile_m') or line.startswith('-'):
                        continue
                    parts = line.split('|')
                    if len(parts) < 2: 
                        continue
                    cfg = parts[0].strip().split()
                    perf = parts[1].strip().split()
                    if len(cfg) >= 4 and len(perf) >= 6:
                        combo_key = f"{cfg[0]}x{cfg[1]}_{cfg[3]}"
                        gflops_str = perf[5]
                        if gflops_str != '-':
                            vals[combo_key] = float(gflops_str)
            gf[name] = vals
    return gf


def build_h200_aat_gflops():
    """Extract gflops from H200 AAT log files -> {matrix_name: {combo: gflops}}"""
    gf = {}
    gflops_re = re.compile(r'(?:CUDA\s+)?TileSpGEMM\s+runtime\s+is\s+([\d.]+)\s*ms,\s*gflops\s*=\s+([\d.]+)')
    fname_re = re.compile(r'aat\d+_m(\d+)_n(\d+)_tc(\d+)\.log')
    
    for mdir_name in sorted(os.listdir(H200_AAT_SRC)):
        mdir = os.path.join(H200_AAT_SRC, mdir_name)
        if not os.path.isdir(mdir): 
            continue
        vals = {}
        for fn in os.listdir(mdir):
            fm = fname_re.match(fn)
            if not fm: 
                continue
            combo = f"{fm.group(1)}x{fm.group(2)}_{fm.group(3)}/8"
            with open(os.path.join(mdir, fn)) as f:
                content = f.read()
            gm = gflops_re.search(content)
            if gm:
                vals[combo] = float(gm.group(2))
        gf[mdir_name] = vals
    return gf


def main():
    print("=" * 70)
    print("  Generate decision results for 100 matrices using tuned model")
    print("=" * 70)

    # Load model
    print(f"\n[1/4] Loading model: {MODEL_PATH}")
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"  Model iterations: {model.num_trees()}")

    # Load gflops tables
    print("\n[2/4] Loading gflops data...")
    a100_aa_gf = build_aa_gflops(A100_AA_GFLOPS)
    print(f"  A100 AA: {len(a100_aa_gf)} matrices")
    h200_aa_gf = build_aa_gflops(H200_AA_GFLOPS)
    print(f"  H200 AA: {len(h200_aa_gf)} matrices")
    a100_aat_gf = build_a100_aat_gflops()
    print(f"  A100 AAT: {len(a100_aat_gf)} matrices")
    h200_aat_gf = build_h200_aat_gflops()
    print(f"  H200 AAT: {len(h200_aat_gf)} matrices")

    # Load test data
    print("\n[3/4] Loading test data...")
    aa_test_df = pd.read_csv(os.path.join(BASE_DIR, "all_data_AA", "test_dataset.csv"))
    aat_test_df = pd.read_csv(os.path.join(BASE_DIR, "all_data_AAT", "test_dataset.csv"))
    test_df = pd.read_csv("/home/stu1/donghangcheng/code/FlexSpGEMM/data/test_dataset.csv")
    
    aa_count = len(aa_test_df)
    print(f"  AA test set: {aa_count} samples")
    print(f"  AAT test set: {len(aat_test_df)} samples")
    print(f"  Total test set: {len(test_df)} samples")

    # Feature columns
    feature_cols = [c for c in test_df.columns if c not in ['matrix_name', 'best_tile', 'best_tc']]

    # Predict
    X_test = test_df[feature_cols].astype(float)
    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)

    # Get all matrix names from test set
    test_names = test_df['matrix_name'].unique().tolist()
    print(f"  Test matrices: {len(test_names)}")

    # Build index: (matrix_name, gpu, mode) -> row_idx
    row_index = {}
    for i, (_, row) in enumerate(test_df.iterrows()):
        gpu = 'A100' if row['sm_count'] == 108 else 'H200'
        mode = 'AA' if i < aa_count else 'AAT'
        row_index[(row['matrix_name'], gpu, mode)] = i

    # Baseline config
    BASELINE_COMBO = "16x16_0/8"

    # Output
    print("\n[4/4] Generating decision results...")
    out_rows = []
    header = [
        "matrix_name", "gpu", "mode",
        "lightgbm_pred_combo", "lightgbm_pred_gflops",
        "best_combo", "best_gflops", "gflops_ratio",
        "tilespgemm_16x16_gflops", "speedup_vs_tilespgemm_16x16"
    ]

    for name in test_names:
        for gpu in ['A100', 'H200']:
            for mode in ['AA', 'AAT']:
                key = (name, gpu, mode)
                if key not in row_index:
                    continue

                idx = row_index[key]
                pred_combo = IDX_TO_COMBO[y_pred[idx]]

                # Select gflops table
                if mode == 'AA':
                    gf_table = a100_aa_gf if gpu == 'A100' else h200_aa_gf
                else:
                    gf_table = a100_aat_gf if gpu == 'A100' else h200_aat_gf

                if name not in gf_table:
                    continue

                gf_all = gf_table[name]
                if not gf_all:
                    continue

                # Predicted gflops
                pred_gf = gf_all.get(pred_combo, 0) if isinstance(gf_all, dict) else 0

                # Best gflops
                if isinstance(gf_all, dict) and len(gf_all) > 0:
                    best_combo = max(gf_all, key=gf_all.get)
                    best_gf = gf_all[best_combo]
                else:
                    best_gf = 0
                    best_combo = ''

                # gflops ratio
                ratio = pred_gf / best_gf if best_gf > 0 else 0

                # tilespgemm 16x16 baseline
                baseline_gf = gf_all.get(BASELINE_COMBO, 0) if isinstance(gf_all, dict) else 0
                speedup = pred_gf / baseline_gf if baseline_gf > 0 else 0

                out_rows.append({
                    'matrix_name': name,
                    'gpu': gpu,
                    'mode': mode,
                    'lightgbm_pred_combo': pred_combo,
                    'lightgbm_pred_gflops': f"{pred_gf:.6f}",
                    'best_combo': best_combo,
                    'best_gflops': f"{best_gf:.6f}",
                    'gflops_ratio': f"{ratio:.6f}",
                    'tilespgemm_16x16_gflops': f"{baseline_gf:.6f}",
                    'speedup_vs_tilespgemm_16x16': f"{speedup:.6f}",
                })

    # Write CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nOutput: {OUTPUT_CSV}")
    print(f"Total entries: {len(out_rows)}")

    # Statistics
    aa_rows = [r for r in out_rows if r['mode'] == 'AA']
    aat_rows = [r for r in out_rows if r['mode'] == 'AAT']
    a100_rows = [r for r in out_rows if r['gpu'] == 'A100']
    h200_rows = [r for r in out_rows if r['gpu'] == 'H200']
    print(f"  AA: {len(aa_rows)}, AAT: {len(aat_rows)}")
    print(f"  A100: {len(a100_rows)}, H200: {len(h200_rows)}")

    # gflops ratio statistics
    ratios = [float(r['gflops_ratio']) for r in out_rows if float(r['best_gflops']) > 0]
    print(f"\n  Gflops Ratio Statistics (n={len(ratios)}):")
    print(f"    >=100%: {sum(1 for r in ratios if r >= 1.0)/len(ratios)*100:.1f}%")
    print(f"    >= 95%: {sum(1 for r in ratios if r >= 0.95)/len(ratios)*100:.1f}%")
    print(f"    >= 90%: {sum(1 for r in ratios if r >= 0.90)/len(ratios)*100:.1f}%")
    print(f"    Average ratio: {np.mean(ratios):.4f}")

    # Detailed statistics
    for label, subset in [("AA_A100", [r for r in out_rows if r['mode']=='AA' and r['gpu']=='A100']),
                           ("AA_H200", [r for r in out_rows if r['mode']=='AA' and r['gpu']=='H200']),
                           ("AAT_A100", [r for r in out_rows if r['mode']=='AAT' and r['gpu']=='A100']),
                           ("AAT_H200", [r for r in out_rows if r['mode']=='AAT' and r['gpu']=='H200'])]:
        rs = [float(r['gflops_ratio']) for r in subset if float(r['best_gflops']) > 0]
        if rs:
            print(f"    {label} (n={len(rs)}): acc={np.mean(rs):.4f}, >=95%={sum(1 for r in rs if r>=0.95)/len(rs)*100:.1f}%, >=90%={sum(1 for r in rs if r>=0.90)/len(rs)*100:.1f}%")

    print("\nDone!")


if __name__ == '__main__':
    main()