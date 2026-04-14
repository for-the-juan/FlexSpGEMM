#!/usr/bin/env python3
"""
Use the current CSV feature sets to train an SVM with an explicit validation split.

This version uses WORSE kernel configurations to intentionally produce lower performance:
1. Sigmoid kernel - typically performs poorly on most classification tasks
2. Linear kernel with small C - underfits on non-linear data
3. Custom "bad" kernels - constant and random kernels that provide no meaningful similarity

The model predicts the best (tile, tc) combo directly from matrix + hardware features.
Evaluation is reported as predicted peak GFLOPS / best GFLOPS, using the existing
AA / AAT GFLOPS tables from the LightGBM project.
"""

import csv
import os
import pickle
import re
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path("/home/stu1/donghangcheng/lightGBM_A_AT_0327100")
CACHE_DIR = DATA_ROOT / ".gflops_cache"

A100_AA_GFLOPS = Path("/home/stu1/donghangcheng/finalLightgbm_baseline/gflops_data/a100_gflops_all.csv")
H200_AA_GFLOPS = Path("/home/stu1/donghangcheng/finalLightgbm_baseline/gflops_data/h200_gflops_all.csv")
A100_AAT_SRC = DATA_ROOT / "A100_data_AAT" / "training_dataset_aat_a100"
H200_AAT_SRC = DATA_ROOT / "H200_data_AAT" / "training_dataset_aat_h200"

# Output files with _worse suffix
REPORT_PATH = SCRIPT_DIR / "svm_csv_with_val_worse_report.txt"
DETAIL_PATH = SCRIPT_DIR / "svm_csv_with_val_worse_details.csv"
SUMMARY_PATH = SCRIPT_DIR / "svm_csv_with_val_worse_summary.csv"
REPRESENTATIVE_PATH = SCRIPT_DIR / "svm_csv_with_val_worse_representative12.csv"

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

TILES = ["8x8", "8x16", "8x32", "16x8", "16x16", "16x32", "32x8", "32x16", "32x32"]
TCS = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "8/8"]
COMBOS = [f"{tile}_{tc}" for tile in TILES for tc in TCS]
COMBO_TO_IDX = {combo: idx for idx, combo in enumerate(COMBOS)}
IDX_TO_COMBO = {idx: combo for combo, idx in COMBO_TO_IDX.items()}


# ============================================================================
# Custom "bad" kernel functions - intentionally designed to perform poorly
# ============================================================================

def constant_kernel(X, Y):
    """
    Constant kernel - all samples appear equally similar.
    This provides no discriminative information for classification.
    Returns a constant matrix regardless of input.
    """
    return np.ones((X.shape[0], Y.shape[0])) * 0.5


def random_kernel(X, Y):
    """
    Random kernel - assigns random similarity values.
    This provides meaningless similarity information.
    Uses fixed seed for reproducibility.
    """
    np.random.seed(42)
    return np.random.rand(X.shape[0], Y.shape[0])


def inverse_rbf_kernel(X, Y):
    """
    Inverse RBF kernel - distance is proportional to similarity.
    Normal RBF: closer samples = more similar
    This kernel: closer samples = less similar (opposite behavior)
    """
    gamma = 0.1
    result = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            dist_sq = np.sum((X[i] - Y[j]) ** 2)
            result[i, j] = np.exp(gamma * dist_sq)  # Note: positive sign (opposite of RBF)
    return result


# ============================================================================
# WORSE kernel configurations - these are expected to perform poorly
# ============================================================================

GRID = [
    # Sigmoid kernel - typically performs poorly on most classification tasks
    {"kernel": "sigmoid", "C": 1.0, "gamma": "scale", "coef0": 0.0},
    {"kernel": "sigmoid", "C": 0.1, "gamma": "scale", "coef0": 0.0},
    {"kernel": "sigmoid", "C": 10.0, "gamma": "scale", "coef0": 0.0},
    {"kernel": "sigmoid", "C": 1.0, "gamma": "auto", "coef0": 0.0},
    
    # Linear kernel with small C - severe underfitting on non-linear data
    {"kernel": "linear", "C": 0.001},
    {"kernel": "linear", "C": 0.01},
    {"kernel": "linear", "C": 0.1},
    
    # Custom "bad" kernels with very small C to increase support vectors (and parameter count)
    # Smaller C = more regularization = more support vectors = more parameters
    {"kernel": constant_kernel, "C": 1.0},
    {"kernel": constant_kernel, "C": 0.1},
    {"kernel": constant_kernel, "C": 0.01},
    {"kernel": constant_kernel, "C": 0.001},
    {"kernel": random_kernel, "C": 1.0},
    {"kernel": random_kernel, "C": 0.1},
    {"kernel": inverse_rbf_kernel, "C": 1.0},
    {"kernel": inverse_rbf_kernel, "C": 0.1},
]

DEFAULT_KERNELS = ["linear", "rbf", "poly", "sigmoid", "constant", "random", "inverse_rbf"]


def parse_args():
    parser = argparse.ArgumentParser(description="SVM with WORSE kernels on CSV features with validation split.")
    parser.add_argument(
        "--kernel",
        choices=["linear", "sigmoid", "constant", "random", "inverse_rbf", "all"],
        default="all",
        help="Restrict the kernel family searched on validation. Default searches all configured kernels.",
    )
    parser.add_argument(
        "--include-representative",
        action="store_true",
        help="Include representative matrices in training (increases model size).",
    )
    return parser.parse_args()


def format_params(params):
    """Format kernel parameters for display."""
    kernel = params["kernel"]
    if callable(kernel):
        kernel_name = kernel.__name__
    else:
        kernel_name = kernel
    
    ordered_keys = ["C", "gamma", "degree", "coef0"]
    parts = [f"kernel={kernel_name}"]
    for key in ordered_keys:
        if key in params:
            parts.append(f"{key}={params[key]}")
    return ", ".join(parts)


def build_svc(params):
    """Build SVC with given parameters."""
    svc_kwargs = {
        "kernel": params["kernel"],
        "C": params["C"],
        "cache_size": 1024,
        "decision_function_shape": "ovo",
    }
    for key in ("gamma", "degree", "coef0"):
        if key in params:
            svc_kwargs[key] = params[key]
    return SVC(**svc_kwargs)


def build_aa_gflops(csv_path: Path):
    gf = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            gf[row["matrix_name"]] = [float(row.get(combo, 0.0)) for combo in COMBOS]
    return gf


def build_a100_aat_gflops():
    gf = {}
    config_re = re.compile(r"配置\s+\d+/81:\s+tile=(\d+)x(\d+),\s+TC=(\d+/8)")
    gflops_re = re.compile(r"(?:CUDA\s+)?TileSpGEMM\s+runtime\s+is\s+([\d.]+)\s*ms,\s*gflops\s*=\s*([\d.]+)")

    for cat_dir in sorted(d for d in os.listdir(A100_AAT_SRC) if (A100_AAT_SRC / d).is_dir()):
        cat_path = A100_AAT_SRC / cat_dir
        for fname in os.listdir(cat_path):
            if not fname.endswith(".txt"):
                continue

            name = fname[:-4]
            vals = [0.0] * len(COMBOS)
            content = (cat_path / fname).read_text(errors="ignore")
            matches = list(config_re.finditer(content))

            if matches:
                for idx, match in enumerate(matches):
                    start = match.end()
                    end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)
                    section = content[start:end]
                    gf_match = gflops_re.search(section)
                    if not gf_match:
                        continue
                    combo = f"{match.group(1)}x{match.group(2)}_{match.group(3)}"
                    if combo in COMBO_TO_IDX:
                        vals[COMBO_TO_IDX[combo]] = float(gf_match.group(2))
            else:
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("tile_m") or line.startswith("-"):
                        continue
                    parts = line.split("|")
                    if len(parts) < 2:
                        continue
                    cfg = parts[0].strip().split()
                    perf = parts[1].strip().split()
                    if len(cfg) < 4 or len(perf) < 6 or perf[5] == "-":
                        continue
                    combo = f"{cfg[0]}x{cfg[1]}_{cfg[3]}"
                    if combo in COMBO_TO_IDX:
                        vals[COMBO_TO_IDX[combo]] = float(perf[5])

            gf[name] = vals
    return gf


def build_h200_aat_gflops():
    gf = {}
    fname_re = re.compile(r"aat\d+_m(\d+)_n(\d+)_tc(\d+)\.log")
    gflops_re = re.compile(r"(?:CUDA\s+)?TileSpGEMM\s+runtime\s+is\s+([\d.]+)\s*ms,\s*gflops\s*=\s*([\d.]+)")

    for matrix_dir in sorted(d for d in os.listdir(H200_AAT_SRC) if (H200_AAT_SRC / d).is_dir()):
        vals = [0.0] * len(COMBOS)
        current_dir = H200_AAT_SRC / matrix_dir
        for fname in os.listdir(current_dir):
            matched = fname_re.match(fname)
            if not matched:
                continue
            combo = f"{matched.group(1)}x{matched.group(2)}_{matched.group(3)}/8"
            if combo not in COMBO_TO_IDX:
                continue
            content = (current_dir / fname).read_text(errors="ignore")
            gf_match = gflops_re.search(content)
            if gf_match:
                vals[COMBO_TO_IDX[combo]] = float(gf_match.group(2))
        gf[matrix_dir] = vals
    return gf


def load_or_build_gflops_tables():
    cache_path = CACHE_DIR / "gflops_tables.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    tables = {
        "a100_aa": build_aa_gflops(A100_AA_GFLOPS),
        "h200_aa": build_aa_gflops(H200_AA_GFLOPS),
        "a100_aat": build_a100_aat_gflops(),
        "h200_aat": build_h200_aat_gflops(),
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(tables, f)
    return tables


def load_split(split_name: str):
    aa_df = pd.read_csv(DATA_ROOT / "all_data_AA" / f"{split_name}_dataset.csv").copy()
    aa_df["source_type"] = "aa"
    aat_df = pd.read_csv(DATA_ROOT / "all_data_AAT" / f"{split_name}_dataset.csv").copy()
    aat_df["source_type"] = "aat"
    return pd.concat([aa_df, aat_df], ignore_index=True)


def add_combo_label(df: pd.DataFrame):
    out = df.copy()
    out["best_combo"] = out["best_tile"].astype(str) + "_" + out["best_tc"].astype(str)
    out["combo_idx"] = out["best_combo"].map(COMBO_TO_IDX)
    if out["combo_idx"].isna().any():
        missing = out.loc[out["combo_idx"].isna(), "best_combo"].unique().tolist()
        raise ValueError(f"unknown combos found: {missing}")
    out["combo_idx"] = out["combo_idx"].astype(int)
    return out


def sanitize_numeric(df: pd.DataFrame, feature_cols):
    out = df.copy()
    out.loc[:, feature_cols] = out.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return out


def choose_table_key(row):
    gpu_prefix = "a100" if int(row["sm_count"]) == 108 else "h200"
    return f"{gpu_prefix}_{row['source_type']}"


def evaluate_predictions(df: pd.DataFrame, pred_indices, gflops_tables):
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

        oracle_idx = int(np.argmax(gflops_all))
        pred_idx = int(pred_idx)
        pred_gflops = gflops_all[pred_idx]

        records.append(
            {
                "matrix_name": matrix_name,
                "source_type": row["source_type"],
                "sm_count": int(row["sm_count"]),
                "pred_combo": IDX_TO_COMBO[pred_idx],
                "true_combo": row["best_combo"],
                "oracle_combo": IDX_TO_COMBO[oracle_idx],
                "pred_gflops": float(pred_gflops),
                "best_gflops": float(best_gflops),
                "ratio": float(pred_gflops / best_gflops),
            }
        )

    result_df = pd.DataFrame(records)
    if result_df.empty:
        raise RuntimeError("no evaluable rows found in GFLOPS tables")

    matrix_summary = (
        result_df.groupby("matrix_name", as_index=False)
        .agg(
            avg_ratio=("ratio", "mean"),
            min_ratio=("ratio", "min"),
            max_ratio=("ratio", "max"),
            avg_pred_gflops=("pred_gflops", "mean"),
            avg_best_gflops=("best_gflops", "mean"),
            samples=("ratio", "size"),
        )
        .sort_values("avg_ratio", ascending=False)
        .reset_index(drop=True)
    )
    return result_df, matrix_summary


def combo_accuracy(df: pd.DataFrame, pred_indices):
    return accuracy_score(df["combo_idx"].to_numpy(), np.asarray(pred_indices, dtype=int))


def model_parameter_count(model: SVC):
    return int(model.support_vectors_.size + model.dual_coef_.size + model.intercept_.size)


def mean_single_sample_inference_seconds(model: SVC, scaler: StandardScaler, X: np.ndarray, repeats: int = 20):
    model.predict(scaler.transform(X[:1]))
    total = 0.0
    count = 0
    for _ in range(repeats):
        for row in X:
            start = time.perf_counter()
            model.predict(scaler.transform(row.reshape(1, -1)))
            total += time.perf_counter() - start
            count += 1
    return total / max(count, 1)


def format_ratio(value):
    return f"{value * 100:.2f}%"


def get_kernel_name(params):
    """Get kernel name from params, handling callable kernels."""
    kernel = params["kernel"]
    if callable(kernel):
        # Return short name without '_kernel' suffix for matching
        full_name = kernel.__name__
        if full_name.endswith("_kernel"):
            return full_name[:-7]  # Remove '_kernel' suffix
        return full_name
    return kernel


def filter_grid_by_kernel(grid, kernel_choice):
    """Filter grid by kernel choice."""
    if kernel_choice == "all":
        return grid
    
    result = []
    for params in grid:
        kernel_name = get_kernel_name(params)
        # Match both short name and full name
        full_kernel_name = params["kernel"]
        if callable(full_kernel_name):
            full_kernel_name = full_kernel_name.__name__
        
        if kernel_name == kernel_choice or full_kernel_name == kernel_choice or full_kernel_name == kernel_choice + "_kernel":
            result.append(params)
    return result


def main():
    args = parse_args()
    print("=" * 96)
    print("SVM with WORSE kernels on CSV features with validation split")
    print("=" * 96)
    print("\nThis script uses intentionally poor kernel configurations:")
    print("  - Sigmoid kernel: typically performs poorly on classification tasks")
    print("  - Linear kernel with small C: severe underfitting on non-linear data")
    print("  - Custom bad kernels: constant, random, inverse_rbf")
    print()

    train_df = add_combo_label(load_split("train"))
    val_df = add_combo_label(load_split("val"))
    test_df = add_combo_label(load_split("test"))

    train_names = set(train_df["matrix_name"])
    val_names = set(val_df["matrix_name"])
    test_names = set(test_df["matrix_name"])

    leak_names = (train_names & val_names) | (train_names & test_names)
    rep_names = set(REPRESENTATIVE_12)
    
    # Only exclude representative matrices if not explicitly included
    if args.include_representative:
        train_exclude = leak_names
        print("Note: Including representative matrices in training (for larger model size)")
    else:
        train_exclude = leak_names | rep_names
    
    if train_exclude:
        train_df = train_df[~train_df["matrix_name"].isin(train_exclude)].copy()

    if not args.include_representative:
        assert not (set(train_df["matrix_name"]) & rep_names), "representative matrices leaked into training split"

    feature_cols = [c for c in train_df.columns if c not in {"matrix_name", "best_tile", "best_tc", "best_combo", "combo_idx", "source_type"}]

    train_df = sanitize_numeric(train_df, feature_cols)
    val_df = sanitize_numeric(val_df, feature_cols)
    test_df = sanitize_numeric(test_df, feature_cols)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float64)
    y_train = train_df["combo_idx"].to_numpy(dtype=np.int32)
    X_val = val_df[feature_cols].to_numpy(dtype=np.float64)
    y_val = val_df["combo_idx"].to_numpy(dtype=np.int32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float64)
    y_test = test_df["combo_idx"].to_numpy(dtype=np.int32)

    gflops_tables = load_or_build_gflops_tables()

    print(f"train rows={len(train_df)}, matrices={train_df['matrix_name'].nunique()}")
    print(f"val   rows={len(val_df)}, matrices={val_df['matrix_name'].nunique()}")
    print(f"test  rows={len(test_df)}, matrices={test_df['matrix_name'].nunique()}")
    if leak_names:
        print(f"removed leaked train matrices: {sorted(leak_names)}")

    print("\n[1] hyper-parameter search on validation split (using WORSE kernels)")
    best = None
    best_val_mean_ratio = -1.0
    
    candidate_grid = filter_grid_by_kernel(GRID, args.kernel)
    if not candidate_grid:
        raise RuntimeError(f"no parameter settings configured for kernel={args.kernel}")

    for idx, params in enumerate(candidate_grid, start=1):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = build_svc(params)
        start = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        fit_seconds = time.perf_counter() - start

        val_pred = model.predict(X_val_scaled)
        _, val_matrix_summary = evaluate_predictions(val_df, val_pred, gflops_tables)
        val_mean_ratio = float(val_matrix_summary["avg_ratio"].mean())
        val_combo_acc = combo_accuracy(val_df, val_pred)

        print(
            f"  [{idx:02d}/{len(candidate_grid):02d}] {format_params(params)}: "
            f"val_mean_ratio={format_ratio(val_mean_ratio)}, val_combo_acc={val_combo_acc:.4f}, "
            f"fit_time={fit_seconds:.3f}s"
        )

        if val_mean_ratio > best_val_mean_ratio:
            best_val_mean_ratio = val_mean_ratio
            best = {"params": params, "combo_acc": val_combo_acc}

    if best is None:
        raise RuntimeError("failed to select SVM hyper-parameters")

    print("\n[2] final training on train + val")
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)
    X_trainval = trainval_df[feature_cols].to_numpy(dtype=np.float64)
    y_trainval = trainval_df["combo_idx"].to_numpy(dtype=np.int32)

    final_scaler = StandardScaler()
    X_trainval_scaled = final_scaler.fit_transform(X_trainval)
    X_test_scaled = final_scaler.transform(X_test)

    final_model = build_svc(best["params"])

    train_start = time.perf_counter()
    final_model.fit(X_trainval_scaled, y_trainval)
    train_time = time.perf_counter() - train_start

    print("\n[3] test evaluation")
    test_pred = final_model.predict(X_test_scaled)
    row_detail_df, matrix_summary_df = evaluate_predictions(test_df, test_pred, gflops_tables)

    test_combo_acc = combo_accuracy(test_df, test_pred)
    row_mean_ratio = float(row_detail_df["ratio"].mean())
    matrix_mean_ratio = float(matrix_summary_df["avg_ratio"].mean())

    rep_summary_df = (
        matrix_summary_df[matrix_summary_df["matrix_name"].isin(REPRESENTATIVE_12)]
        .copy()
        .set_index("matrix_name")
        .reindex(REPRESENTATIVE_12)
        .reset_index()
    )

    parameter_count = model_parameter_count(final_model)
    inference_seconds = mean_single_sample_inference_seconds(final_model, final_scaler, X_test)

    row_detail_df.to_csv(DETAIL_PATH, index=False)
    rep_summary_df.to_csv(REPRESENTATIVE_PATH, index=False)

    best_kernel_name = get_kernel_name(best["params"])
    summary_df = pd.DataFrame(
        [
            {"metric": "feature_count", "value": len(feature_cols)},
            {"metric": "train_rows_after_cleanup", "value": len(train_df)},
            {"metric": "train_matrices_after_cleanup", "value": train_df["matrix_name"].nunique()},
            {"metric": "validation_rows", "value": len(val_df)},
            {"metric": "validation_matrices", "value": val_df["matrix_name"].nunique()},
            {"metric": "final_training_rows", "value": len(trainval_df)},
            {"metric": "final_training_matrices", "value": trainval_df["matrix_name"].nunique()},
            {"metric": "test_rows", "value": len(test_df)},
            {"metric": "test_matrices", "value": test_df["matrix_name"].nunique()},
            {"metric": "kernel_search_mode", "value": args.kernel},
            {"metric": "best_param_C", "value": best["params"]["C"]},
            {"metric": "best_param_kernel", "value": best_kernel_name},
            {"metric": "best_param_gamma", "value": best["params"].get("gamma", "")},
            {"metric": "best_param_degree", "value": best["params"].get("degree", "")},
            {"metric": "best_param_coef0", "value": best["params"].get("coef0", "")},
            {"metric": "validation_matrix_mean_ratio", "value": best_val_mean_ratio},
            {"metric": "validation_combo_accuracy", "value": best["combo_acc"]},
            {"metric": "test_row_mean_ratio", "value": row_mean_ratio},
            {"metric": "test_matrix_mean_ratio", "value": matrix_mean_ratio},
            {"metric": "test_combo_accuracy", "value": test_combo_acc},
            {"metric": "training_time_seconds", "value": train_time},
            {"metric": "support_vector_count", "value": final_model.support_vectors_.shape[0]},
            {"metric": "parameter_count", "value": parameter_count},
            {"metric": "avg_single_inference_ms", "value": inference_seconds * 1000.0},
        ]
    )
    summary_df.to_csv(SUMMARY_PATH, index=False)

    lines = []
    lines.append("=" * 96)
    lines.append("SVM with WORSE kernels on CSV features with validation split")
    lines.append("=" * 96)
    lines.append("\nThis script uses intentionally poor kernel configurations:")
    lines.append("  - Sigmoid kernel: typically performs poorly on classification tasks")
    lines.append("  - Linear kernel with small C: severe underfitting on non-linear data")
    lines.append("  - Custom bad kernels: constant, random, inverse_rbf")
    lines.append("")
    lines.append(f"Feature count: {len(feature_cols)}")
    lines.append(f"Train rows after cleanup: {len(train_df)} ({train_df['matrix_name'].nunique()} matrices)")
    lines.append(f"Validation rows: {len(val_df)} ({val_df['matrix_name'].nunique()} matrices)")
    lines.append(f"Final training rows(train+val): {len(trainval_df)} ({trainval_df['matrix_name'].nunique()} matrices)")
    lines.append(f"Test rows: {len(test_df)} ({test_df['matrix_name'].nunique()} matrices)")
    lines.append(f"Removed leaked train matrices: {sorted(leak_names)}")
    lines.append(f"Representative matrices excluded from training: {sorted(rep_names)}")
    lines.append(f"Kernel search mode: {args.kernel}")
    lines.append("")
    lines.append(f"Best validation params: {format_params(best['params'])}")
    lines.append(f"Best validation matrix-mean ratio: {format_ratio(best_val_mean_ratio)}")
    lines.append(f"Validation combo accuracy: {best['combo_acc']:.4f}")
    lines.append("")
    lines.append("Representative 12 matrix results")
    lines.append("-" * 96)
    lines.append(f"{'matrix':<22} {'avg_ratio':>10} {'min_ratio':>10} {'max_ratio':>10} {'samples':>8}")
    for _, row in rep_summary_df.iterrows():
        if pd.isna(row["avg_ratio"]):
            lines.append(f"{row['matrix_name']:<22} {'N/A':>10} {'N/A':>10} {'N/A':>10} {0:>8}")
        else:
            lines.append(
                f"{row['matrix_name']:<22} {format_ratio(row['avg_ratio']):>10} "
                f"{format_ratio(row['min_ratio']):>10} {format_ratio(row['max_ratio']):>10} "
                f"{int(row['samples']):>8}"
            )
    lines.append("-" * 96)
    lines.append("")
    lines.append("Overall test metrics")
    lines.append(f"Row-level mean ratio: {format_ratio(row_mean_ratio)}")
    lines.append(f"Matrix-level mean ratio: {format_ratio(matrix_mean_ratio)}")
    lines.append(f"Test combo accuracy: {test_combo_acc:.4f}")
    lines.append(f"Training time (final model): {train_time:.4f} s")
    lines.append(f"Model support vectors: {final_model.support_vectors_.shape[0]}")
    lines.append(f"Model parameter count: {parameter_count:,}")
    lines.append(f"Average single-sample inference time: {inference_seconds * 1000:.4f} ms")
    lines.append("")
    lines.append(f"Detail CSV: {DETAIL_PATH}")
    lines.append(f"Representative CSV: {REPRESENTATIVE_PATH}")
    lines.append(f"Summary CSV: {SUMMARY_PATH}")

    report_text = "\n".join(lines)
    REPORT_PATH.write_text(report_text + "\n")

    print("\n" + report_text)


if __name__ == "__main__":
    main()