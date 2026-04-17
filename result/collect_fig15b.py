#!/usr/bin/env python3
"""
Collect gflops_ratio from three methods (LightGBM, LLM, SVM) and generate fig15(b).csv
Also draw a bar chart comparing the three methods across 12 matrices.
"""

from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# File paths
LIGHTGBM_CSV = Path("../ML_method/LightGBM/predictResult_test12/test12_result.csv")
LLM_CSV = Path("../ML_method/LLM/LLM_12_results.csv")
SVM_CSV = Path("../ML_method/SVM/svm_12_results.csv")
OUTPUT_CSV = Path("data/fig15(b).csv")
OUTPUT_PLOT = Path("plots/fig15(b).png")

# Representative 12 matrices in order
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


def load_lightgbm_ratios(csv_path: Path) -> Dict[str, float]:
    """Load gflops_ratio from LightGBM test12_result.csv (AA mode only)."""
    df = pd.read_csv(csv_path)
    # Filter AA mode only
    df_aa = df[df["mode"] == "AA"].copy()
    ratios = {}
    for _, row in df_aa.iterrows():
        matrix_name = row["matrix_name"]
        gflops_ratio = row["gflops_ratio"]
        if pd.notna(gflops_ratio):
            ratios[matrix_name] = float(gflops_ratio)
    return ratios


def load_llm_ratios(csv_path: Path) -> Dict[str, float]:
    """Load gflops_ratio from LLM results."""
    df = pd.read_csv(csv_path)
    ratios = {}
    for _, row in df.iterrows():
        matrix_name = row["matrix_name"]
        gflops_ratio = row["gflops_ratio"]
        if pd.notna(gflops_ratio):
            ratios[matrix_name] = float(gflops_ratio)
    return ratios


def load_svm_ratios(csv_path: Path) -> Dict[str, float]:
    """Load gflops_ratio from SVM results."""
    df = pd.read_csv(csv_path)
    ratios = {}
    for _, row in df.iterrows():
        matrix_name = row["matrix_name"]
        gflops_ratio = row["gflops_ratio"]
        if pd.notna(gflops_ratio):
            ratios[matrix_name] = float(gflops_ratio)
    return ratios


def main():
    # Load data from three methods
    lightgbm_ratios = load_lightgbm_ratios(LIGHTGBM_CSV)
    llm_ratios = load_llm_ratios(LLM_CSV)
    svm_ratios = load_svm_ratios(SVM_CSV)

    # Build output dataframe
    output_data = []
    for matrix in REPRESENTATIVE_12:
        output_data.append({
            "matrix_name": matrix,
            "LightGBM": lightgbm_ratios.get(matrix, None),
            "LLM": llm_ratios.get(matrix, None),
            "SVM": svm_ratios.get(matrix, None),
        })

    output_df = pd.DataFrame(output_data)

    # Ensure output directory exists
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PLOT.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to: {OUTPUT_CSV}")

    # Draw bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(REPRESENTATIVE_12))
    width = 0.25

    lightgbm_values = [output_df.loc[output_df["matrix_name"] == m, "LightGBM"].values[0] for m in REPRESENTATIVE_12]
    llm_values = [output_df.loc[output_df["matrix_name"] == m, "LLM"].values[0] for m in REPRESENTATIVE_12]
    svm_values = [output_df.loc[output_df["matrix_name"] == m, "SVM"].values[0] for m in REPRESENTATIVE_12]

    # Replace None with 0 for plotting
    lightgbm_values = [v if v is not None else 0 for v in lightgbm_values]
    llm_values = [v if v is not None else 0 for v in llm_values]
    svm_values = [v if v is not None else 0 for v in svm_values]

    bars1 = ax.bar(x - width, lightgbm_values, width, label="LightGBM", color="#2196F3")
    bars2 = ax.bar(x, llm_values, width, label="LLM", color="#FF9800")
    bars3 = ax.bar(x + width, svm_values, width, label="SVM", color="#4CAF50")

    ax.set_xlabel("Matrix", fontsize=12)
    ax.set_ylabel("GFLOPS Ratio (pred/best)", fontsize=12)
    ax.set_title("Figure 15(b): GFLOPS Ratio Comparison Across Methods", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(REPRESENTATIVE_12, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"Plot saved to: {OUTPUT_PLOT}")

    # Print summary
    print("\nSummary:")
    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()