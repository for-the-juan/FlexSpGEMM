#!/usr/bin/env python3
"""Run mat-142 LightGBM prediction on CPU with 32 threads."""

import csv
import os
import statistics
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import lightgbm as lgb
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = Path("/home/hangcheng.dong/PPoPP27/FlexSpGEMM")
MAT_DIR = Path("/home/hangcheng.dong/PPoPP27/mat-142/mat-142")
FEATURE_CSV = (
    REPO
    / "tmp_flex_grid_gpu1_20260705_090903"
    / "reports/lightgbm_mat142/mat142_lightgbm_features.csv"
)
MODEL_PATH = SCRIPT_DIR / "model_tuned.txt"

PRED_CSV = SCRIPT_DIR / "mat142_cpu32_lightgbm_predictions.csv"
MATRIX_SUMMARY_CSV = SCRIPT_DIR / "mat142_cpu32_lightgbm_predict_time_by_matrix.csv"
SUMMARY_CSV = SCRIPT_DIR / "mat142_cpu32_lightgbm_predict_time_summary.csv"
REPORT_MD = SCRIPT_DIR / "mat142_cpu32_lightgbm_predict_time_report.md"

THREADS = 32
WARMUP_REPEATS = 5
BATCH_REPEATS = 100
ROW_REPEATS = 20

TILES = ["8x8", "8x16", "8x32", "16x8", "16x16", "16x32", "32x8", "32x16", "32x32"]
TCS = ["0/8", "1/8", "2/8", "3/8", "4/8", "5/8", "6/8", "7/8", "8/8"]
COMBOS = [f"{tile}_{tc}" for tile in TILES for tc in TCS]
IDX_TO_COMBO = {idx: combo for idx, combo in enumerate(COMBOS)}


def fmt(value, ndigits=6):
    return f"{float(value):.{ndigits}f}"


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def timed_batch_predict(model, x_np, repeats):
    times = []
    last_prob = None
    last_idx = None
    for _ in range(repeats):
        start_s = time.perf_counter()
        prob = model.predict(x_np, num_threads=THREADS)
        pred_idx = prob.argmax(axis=1)
        times.append((time.perf_counter() - start_s) * 1000.0)
        last_prob = prob
        last_idx = pred_idx
    return times, last_prob, last_idx


def timed_row_predict(model, x_np):
    row_avg_times = []
    for row_idx in range(len(x_np)):
        sample = x_np[row_idx : row_idx + 1]
        for _ in range(2):
            prob = model.predict(sample, num_threads=THREADS)
            prob.argmax(axis=1)
        repeats = []
        for _ in range(ROW_REPEATS):
            start_s = time.perf_counter()
            prob = model.predict(sample, num_threads=THREADS)
            prob.argmax(axis=1)
            repeats.append((time.perf_counter() - start_s) * 1000.0)
        row_avg_times.append(statistics.mean(repeats))
    return row_avg_times


def main():
    if not FEATURE_CSV.exists():
        raise FileNotFoundError(FEATURE_CSV)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    load_start_s = time.perf_counter()
    model = lgb.Booster(model_file=str(MODEL_PATH))
    model_load_ms = (time.perf_counter() - load_start_s) * 1000.0
    feature_names = model.feature_name()

    df = pd.read_csv(FEATURE_CSV)
    mat_count = len(list(MAT_DIR.glob("*.mtx")))
    if df["matrix_name"].nunique() != mat_count:
        raise RuntimeError(
            f"feature CSV covers {df['matrix_name'].nunique()} matrices, expected {mat_count}"
        )

    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise RuntimeError(f"missing model feature columns: {missing[:10]}")

    x_df = df[feature_names].astype(float)
    x_np = np.ascontiguousarray(x_df.to_numpy(dtype=np.float64, copy=False))

    for _ in range(WARMUP_REPEATS):
        prob = model.predict(x_np, num_threads=THREADS)
        prob.argmax(axis=1)

    batch_times, prob, pred_idx = timed_batch_predict(model, x_np, BATCH_REPEATS)
    row_times = timed_row_predict(model, x_np)

    pred_rows = []
    for pos, row in df.iterrows():
        idx = int(pred_idx[pos])
        combo = IDX_TO_COMBO[idx]
        tile, tc = combo.rsplit("_", 1)
        tile_m, tile_n = tile.split("x")
        pred_rows.append(
            {
                "matrix_name": row["matrix_name"],
                "mode": row["mode"],
                "matrix_symmetry": row.get("matrix_symmetry", ""),
                "pred_combo": combo,
                "pred_tile_m": tile_m,
                "pred_tile_n": tile_n,
                "pred_tc_num": tc.split("/")[0],
                "pred_tc_den": tc.split("/")[1],
                "pred_class_idx": idx,
                "pred_probability": fmt(prob[pos][idx], 8),
                "single_row_predict_time_ms": fmt(row_times[pos]),
            }
        )

    with PRED_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    by_matrix = {}
    for row in pred_rows:
        matrix = row["matrix_name"]
        entry = by_matrix.setdefault(
            matrix,
            {
                "matrix_name": matrix,
                "matrix_symmetry": row["matrix_symmetry"],
                "modes_predicted": 0,
                "aa_pred_combo": "",
                "aa_single_row_predict_time_ms": "",
                "aat_pred_combo": "",
                "aat_single_row_predict_time_ms": "",
                "mean_single_row_predict_time_ms": "",
                "max_single_row_predict_time_ms": "",
            },
        )
        mode = row["mode"].lower()
        entry["modes_predicted"] += 1
        entry[f"{mode}_pred_combo"] = row["pred_combo"]
        entry[f"{mode}_single_row_predict_time_ms"] = row["single_row_predict_time_ms"]

    for entry in by_matrix.values():
        values = [
            float(entry[col])
            for col in ("aa_single_row_predict_time_ms", "aat_single_row_predict_time_ms")
            if str(entry.get(col, "")).strip()
        ]
        entry["mean_single_row_predict_time_ms"] = fmt(statistics.mean(values))
        entry["max_single_row_predict_time_ms"] = fmt(max(values))

    matrix_rows = [by_matrix[name] for name in sorted(by_matrix)]
    with MATRIX_SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)

    mode_counts = Counter(row["mode"] for row in pred_rows)
    combo_counts = Counter(row["pred_combo"] for row in pred_rows)
    batch_avg = statistics.mean(batch_times)
    batch_median = statistics.median(batch_times)
    row_avg = statistics.mean(row_times)
    row_median = statistics.median(row_times)

    summary_rows = [
        {"metric": "generated_utc", "value": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        {"metric": "device_mode", "value": "CPU"},
        {"metric": "cuda_visible_devices", "value": os.environ.get("CUDA_VISIBLE_DEVICES", "")},
        {"metric": "num_threads", "value": THREADS},
        {"metric": "matrix_count", "value": mat_count},
        {"metric": "prediction_rows", "value": len(pred_rows)},
        {"metric": "aa_rows", "value": mode_counts.get("AA", 0)},
        {"metric": "aat_rows", "value": mode_counts.get("AAT", 0)},
        {"metric": "model_load_ms", "value": fmt(model_load_ms, 3)},
        {"metric": "batch_repeats", "value": BATCH_REPEATS},
        {"metric": "batch_predict_avg_ms", "value": fmt(batch_avg)},
        {"metric": "batch_predict_median_ms", "value": fmt(batch_median)},
        {"metric": "batch_predict_min_ms", "value": fmt(min(batch_times))},
        {"metric": "batch_predict_p95_ms", "value": fmt(percentile(batch_times, 95))},
        {"metric": "batch_predict_max_ms", "value": fmt(max(batch_times))},
        {"metric": "batch_predict_avg_ms_per_row", "value": fmt(batch_avg / len(pred_rows))},
        {"metric": "single_row_predict_avg_ms", "value": fmt(row_avg)},
        {"metric": "single_row_predict_median_ms", "value": fmt(row_median)},
        {"metric": "single_row_predict_min_ms", "value": fmt(min(row_times))},
        {"metric": "single_row_predict_p95_ms", "value": fmt(percentile(row_times, 95))},
        {"metric": "single_row_predict_max_ms", "value": fmt(max(row_times))},
    ]
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# mat-142 CPU32 LightGBM Predict Time Report",
        "",
        f"- Generated: `{summary_rows[0]['value']}`",
        "- Device mode: `CPU`",
        f"- `CUDA_VISIBLE_DEVICES`: `{os.environ.get('CUDA_VISIBLE_DEVICES', '')}`",
        f"- `num_threads`: `{THREADS}`",
        f"- Model: `{MODEL_PATH}`",
        f"- Feature source: `{FEATURE_CSV}`",
        f"- Prediction CSV: `{PRED_CSV}`",
        f"- Matrix-level timing CSV: `{MATRIX_SUMMARY_CSV}`",
        f"- Timing summary CSV: `{SUMMARY_CSV}`",
        "",
        "## Scope",
        "",
        f"- Matrices: `{mat_count}`",
        f"- Prediction rows: `{len(pred_rows)}`",
        f"- AA rows: `{mode_counts.get('AA', 0)}`",
        f"- AAT rows: `{mode_counts.get('AAT', 0)}`",
        "",
        "## Predict Time",
        "",
        f"- Model load time: `{fmt(model_load_ms, 3)} ms`",
        f"- Batch predict repeats: `{BATCH_REPEATS}`",
        f"- Batch predict avg/median/min/p95/max: `{fmt(batch_avg)}` / `{fmt(batch_median)}` / `{fmt(min(batch_times))}` / `{fmt(percentile(batch_times, 95))}` / `{fmt(max(batch_times))}` ms",
        f"- Batch avg per prediction row: `{fmt(batch_avg / len(pred_rows))} ms`",
        f"- Single-row predict avg/median/min/p95/max: `{fmt(row_avg)}` / `{fmt(row_median)}` / `{fmt(min(row_times))}` / `{fmt(percentile(row_times, 95))}` / `{fmt(max(row_times))}` ms",
        "",
        "## Top Predicted Configs",
        "",
        "| pred_combo | count |",
        "|---|---:|",
    ]
    for combo, count in combo_counts.most_common(20):
        lines.append(f"| {combo} | {count} |")
    lines.append("")
    REPORT_MD.write_text("\n".join(lines))

    print(f"device=CPU num_threads={THREADS}")
    print(f"matrices={mat_count}, rows={len(pred_rows)}, AA={mode_counts.get('AA', 0)}, AAT={mode_counts.get('AAT', 0)}")
    print(f"batch_avg_ms={batch_avg:.6f}, per_row_batch_avg_ms={batch_avg / len(pred_rows):.6f}")
    print(f"single_row_avg_ms={row_avg:.6f}")
    print(f"wrote {PRED_CSV}")
    print(f"wrote {MATRIX_SUMMARY_CSV}")
    print(f"wrote {SUMMARY_CSV}")
    print(f"wrote {REPORT_MD}")


if __name__ == "__main__":
    main()
