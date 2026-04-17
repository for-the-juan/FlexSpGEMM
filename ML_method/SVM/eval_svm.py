#!/usr/bin/env python3
"""
Load a trained SVM checkpoint and make predictions on a specified dataset split.
Only prints Representative 12 matrix results with avg_ratio.

Usage:
    python predict_from_checkpoint.py --checkpoint svm_checkpoint.pkl --split test
"""

import argparse
import csv
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# Custom kernel functions - must be defined for pickle to load the model
# ============================================================================

def constant_kernel(X, Y):
    """Constant kernel - all samples appear equally similar."""
    return np.ones((X.shape[0], Y.shape[0])) * 0.5


def random_kernel(X, Y):
    """Random kernel - assigns random similarity values."""
    np.random.seed(42)
    return np.random.rand(X.shape[0], Y.shape[0])


def inverse_rbf_kernel(X, Y):
    """Inverse RBF kernel - distance is proportional to similarity."""
    gamma = 0.1
    result = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            dist_sq = np.sum((X[i] - Y[j]) ** 2)
            result[i, j] = np.exp(gamma * dist_sq)
    return result


# Default data root
DEFAULT_DATA_ROOT = Path("../../data/data_prepare/data_get")
CACHE_DIR = DEFAULT_DATA_ROOT / ".gflops_cache"

# GFLOPS data paths
A100_AA_GFLOPS = Path("../../data/data_prepare/prime_data/a100_gflops_all.csv")
H200_AA_GFLOPS = Path("../../data/data_prepare/prime_data/h200_gflops_all.csv")
A100_AAT_GFLOPS = Path("../../data/data_prepare/prime_data/a100_aat_gflops_all.csv")
H200_AAT_GFLOPS = Path("../../data/data_prepare/prime_data/h200_aat_gflops_all.csv")

# Combo mappings
TILES = ["8x8", "8x16", "8x32", "16x8", "16x16", "16x32", "32x8", "32x16", "32x32"]
TCS = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "8/8"]
COMBOS = [f"{tile}_{tc}" for tile in TILES for tc in TCS]
COMBO_TO_IDX = {combo: idx for idx, combo in enumerate(COMBOS)}
IDX_TO_COMBO = {idx: combo for idx, combo in enumerate(COMBOS)}

# Representative 12 matrices
REPRESENTATIVE_12 = [
    "hangGlider_4",
    "webbase-1M",
    "pkustk12",
    "Goodwin_095",
    "af_shell10",
    "s3rmq4m1",
    "rma10",
    "nemeth12",
    "TSOPF_FS_b300_c2",
    "trans5",
    "heart3",
    "gupta3",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Load SVM checkpoint and make predictions.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test", help="Dataset split.")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_ROOT), help="Data root directory.")
    parser.add_argument(
        "--test12_result_csv",
        type=str,
        default="../LightGBM/predictResult_test12/test12_result.csv",
        help="Path to test12_result.csv for best_gflops lookup",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="../LightGBM/predictResult_test12/log",
        help="Directory containing FlexSpGEMM log files",
    )
    return parser.parse_args()


def build_aa_gflops(csv_path: Path):
    gf = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            matrix_name = row.get("matrix_name", "").strip()
            if not matrix_name:
                continue
            vals = []
            for combo in COMBOS:
                v = row.get(combo, "")
                if v is None or v.strip() == "" or v.strip().upper() == "N/A":
                    vals.append(0.0)
                else:
                    try:
                        vals.append(float(v.strip()))
                    except ValueError:
                        vals.append(0.0)
            gf[matrix_name] = vals
    return gf


def load_or_build_gflops_tables():
    cache_path = CACHE_DIR / "gflops_tables.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    tables = {
        "a100_aa": build_aa_gflops(A100_AA_GFLOPS),
        "h200_aa": build_aa_gflops(H200_AA_GFLOPS),
        "a100_aat": build_aa_gflops(A100_AAT_GFLOPS),
        "h200_aat": build_aa_gflops(H200_AAT_GFLOPS),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(tables, f)
    return tables


def load_checkpoint(checkpoint_path: Path):
    with checkpoint_path.open("rb") as f:
        return pickle.load(f)


def load_split(data_dir: Path, split_name: str):
    aa_df = pd.read_csv(data_dir / "all_data_AA" / f"{split_name}.csv").copy()
    aa_df["source_type"] = "aa"
    aat_df = pd.read_csv(data_dir / "all_data_AAT" / f"{split_name}.csv").copy()
    aat_df["source_type"] = "aat"
    return pd.concat([aa_df, aat_df], ignore_index=True)


def sanitize_numeric(df: pd.DataFrame, feature_cols):
    out = df.copy()
    out.loc[:, feature_cols] = out.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out


def choose_table_key(row):
    gpu_prefix = "a100" if int(row["sm_count"]) == 108 else "h200"
    return f"{gpu_prefix}_{row['source_type']}"


def format_ratio(value):
    return f"{value * 100:.2f}%"


def parse_gflops_from_log(log_path: str) -> float | None:
    """Parse GFlops throughput from log file."""
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Throughput" in line and "GFlops" in line:
                    # Extract number before "GFlops"
                    match = re.search(r"(\d+\.?\d*)\s*GFlops", line)
                    if match:
                        return float(match.group(1))
    except FileNotFoundError:
        pass
    return None


def load_best_gflops_table(csv_path: str) -> dict[str, float]:
    """Load best_gflops from test12_result.csv, keyed by matrix_name (AA mode only)."""
    table: dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            matrix_name = (row.get("matrix_name") or "").strip()
            mode = (row.get("mode") or "").strip()
            best_gflops_str = (row.get("best_gflops") or "").strip()
            if not matrix_name or mode != "AA":
                continue
            if best_gflops_str and best_gflops_str.upper() != "N/A":
                try:
                    table[matrix_name] = float(best_gflops_str)
                except ValueError:
                    continue
    return table


def main():
    args = parse_args()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = load_checkpoint(checkpoint_path)
    model = checkpoint["model"]
    scaler = checkpoint["scaler"]
    feature_cols = checkpoint["feature_cols"]
    
    # Load data
    data_dir = Path(args.data_dir)
    df = load_split(data_dir, args.split)
    df = sanitize_numeric(df, feature_cols)
    X = df[feature_cols].to_numpy(dtype=np.float64)
    
    # Predict
    X_scaled = scaler.transform(X)
    pred_indices = model.predict(X_scaled)
    
    # Load GFLOPS tables
    gflops_tables = load_or_build_gflops_tables()
    
    # Calculate ratios for each row
    records = []
    for row, pred_idx in zip(df.to_dict("records"), pred_indices):
        table_key = choose_table_key(row)
        matrix_name = row["matrix_name"]
        gflops_all = gflops_tables.get(table_key, {}).get(matrix_name)
        if not gflops_all:
            continue
        best_gflops = max(gflops_all)
        if best_gflops <= 0:
            continue
        pred_combo = IDX_TO_COMBO[int(pred_idx)]
        records.append({
            "matrix_name": matrix_name,
            "pred_combo": pred_combo,
        })
    
    result_df = pd.DataFrame(records)
    
    # Calculate matrix-level pred_combo
    matrix_summary = result_df.groupby("matrix_name", as_index=False).agg(pred_combo=("pred_combo", "first"))
    
    # Filter Representative 12
    rep_summary = matrix_summary[matrix_summary["matrix_name"].isin(REPRESENTATIVE_12)].copy()
    rep_summary = rep_summary.set_index("matrix_name").reindex(REPRESENTATIVE_12).reset_index()
    
    # Load best_gflops from test12_result.csv
    best_gflops_table = load_best_gflops_table(args.test12_result_csv)
    
    # Enrich rep_summary with pred_gflops from log and best_gflops from csv
    output_rows = []
    for _, row in rep_summary.iterrows():
        matrix_name = row["matrix_name"]
        pred_combo = row["pred_combo"]
        
        # Parse pred_combo to construct log filename
        # pred_combo format: "32x32_8/8" -> log filename: "{matrix_name}_AA_32x32_tc8.log"
        pred_gflops = None
        best_gflops = None
        gflops_ratio = None
        
        if pd.notna(pred_combo) and pred_combo != "-":
            # Parse combo: "32x32_8/8" -> tile="32x32", tc="8"
            parts = pred_combo.split("_")
            if len(parts) == 2:
                tile = parts[0]  # "32x32"
                tc_part = parts[1]  # "8/8"
                tc = tc_part.split("/")[0]  # "8"
                
                # Construct log filename
                log_filename = f"{matrix_name}_AA_{tile}_tc{tc}.log"
                log_path = f"{args.log_dir}/{log_filename}"
                pred_gflops = parse_gflops_from_log(log_path)
        
        # Get best_gflops from table
        best_gflops = best_gflops_table.get(matrix_name)
        
        # Calculate ratio
        if pred_gflops is not None and best_gflops is not None and best_gflops > 0:
            gflops_ratio = pred_gflops / best_gflops
        
        output_rows.append({
            "matrix_name": matrix_name,
            "pred_combo": pred_combo if pd.notna(pred_combo) else "-",
            "pred_gflops": pred_gflops,
            "best_gflops": best_gflops,
            "gflops_ratio": gflops_ratio,
        })
    
    output_df = pd.DataFrame(output_rows)
    
    # Save results to CSV
    output_path = Path(__file__).resolve().parent / "svm_12_results.csv"
    output_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()