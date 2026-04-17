#!/bin/bash
# Task 2: Run AA and AAT multiplication on A100, append results to test100_result.csv
# Use FlexSpGEMM's 9 binaries + -tau parameter to implement 81 configurations

set -e

# Auto-activate conda environment
source /home/stu1/miniconda3/bin/activate FlexSpGEMM
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"

# Path configuration
BIN_DIR="../../../bin"
MATRIX_DIR="../../../data/test"
RESULT_CSV="./test100_result.csv"
LOG_DIR="./log"
PROBE_CSV="../../../data/data_prepare/data_get/probe.csv"

echo "================================================================================"
echo "  Task 2: Run AA and AAT multiplication on A100"
echo "================================================================================"
echo ""

# Check if test100_result.csv exists
if [ ! -f "$RESULT_CSV" ]; then
    echo "Error: $RESULT_CSV does not exist, please run python3 predict_test100.py first"
    exit 1
fi

# Clear old logs
echo "[Step 1/3] Preparation"
echo "  Clearing old log directory: $LOG_DIR"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

total_lines=$(tail -n +2 "$RESULT_CSV" | wc -l)
echo "  Entries to process: $total_lines"
echo ""

echo "[Step 2/3] Executing SpGEMM computation"
echo "  Binary directory: $BIN_DIR"
echo "  Matrix directory: $MATRIX_DIR"
echo ""

# Counters
current=0
success=0
failed=0
skipped=0

# Skip header, process line by line
while IFS=',' read -r matrix_name gpu mode pred_combo runtime_ms gflops csr2tile_ms; do

    current=$((current + 1))

    # Only process A100 data
    if [ "$gpu" != "A100" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    # Parse pred_combo (format: 16x16_0/8)
    tile_part=$(echo "$pred_combo" | cut -d'_' -f1)
    tc_part=$(echo "$pred_combo" | cut -d'_' -f2)
    tile_m=$(echo "$tile_part" | cut -d'x' -f1)
    tile_n=$(echo "$tile_part" | cut -d'x' -f2)
    tc_numerator=$(echo "$tc_part" | cut -d'/' -f1)

    # Determine aat parameter
    if [ "$mode" = "AA" ]; then
        aat_flag=0
    else
        aat_flag=1
    fi

    # Matrix file path
    matrix_file="$MATRIX_DIR/${matrix_name}.mtx"
    if [ ! -f "$matrix_file" ]; then
        echo "  [$current/$total_lines] ⚠ Skip ${matrix_name} ${mode} - Matrix file not found"
        failed=$((failed + 1))
        continue
    fi

    tau_value=$(awk -v tc="$tc_numerator" 'BEGIN { printf "%.3f", tc / 8.0 }')

    # Select corresponding FlexSpGEMM binary file
    binary="$BIN_DIR/test_m${tile_m}_n${tile_n}"
    if [ ! -f "$binary" ]; then
        echo "  [$current/$total_lines] ⚠ Skip ${matrix_name} ${mode} - Binary file not found: test_m${tile_m}_n${tile_n}"
        failed=$((failed + 1))
        continue
    fi

    # Log file
    log_file="$LOG_DIR/${matrix_name}_${mode}_${tile_m}x${tile_n}_tc${tc_numerator}.log"

    echo "  [$current/$total_lines] Running: ${matrix_name} | ${mode} | tile=${tile_m}x${tile_n} | TC=${tc_part} | tau=${tau_value}"

    # Run SpGEMM (allow non-zero exit code, check from log later)
    set +e
    (
        cd "$BIN_DIR" &&
        stdbuf -oL -eL "./$(basename "$binary")" -d 0 -aat "$aat_flag" -tau "$tau_value" "$matrix_file"
    ) > "$log_file" 2>&1
    run_status=$?
    set -e

    # Extract gflops and runtime from log
    gf=$(grep -oP '(?i)(?:Throughput\s*:|gflops\s*=)\s*\K[\d.]+' "$log_file" 2>/dev/null | head -1)
    rt=$(grep -oP '(?i)(?:Total Runtime\s*:|TileSpGEMM\s+runtime\s+is\s+)\s*\K[\d.]+' "$log_file" 2>/dev/null | head -1)

    if [ -n "$gf" ]; then
        echo "             ✓ Execution successful | runtime=${rt}ms | gflops=${gf}"
        success=$((success + 1))
    elif grep -qi "does not do symmetric matrix" "$log_file" 2>/dev/null; then
        echo "             ⊘ Skip symmetric matrix (AAT mode not supported)"
        skipped=$((skipped + 1))
    else
        echo "             ✗ Execution failed (exit=${run_status}), no data obtained"
        failed=$((failed + 1))
    fi

done < <(tail -n +2 "$RESULT_CSV")

echo ""
echo "  Execution statistics: success=$success, failed=$failed, skipped=$skipped"
echo ""

echo "[Step 3/3] Extracting performance data from logs and updating CSV"

python << 'PYTHON_SCRIPT'
import csv
import re
import os
import sys

RESULT_CSV = "./test100_result.csv"
LOG_DIR = "./log"
PROBE_CSV = "../../../data/data_prepare/data_get/probe.csv"

# Read existing CSV
rows = []
with open(RESULT_CSV, 'r') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(row)

# Regular expressions
runtime_pattern = re.compile(r'(?:Total Runtime\s*:|TileSpGEMM\s+runtime\s+is\s+)\s*([\d.]+)\s*ms', re.IGNORECASE)
gflops_pattern = re.compile(r'(?:Throughput\s*:|gflops\s*=)\s*([\d.]+)', re.IGNORECASE)
csr2tile_pattern = re.compile(r'(?:Format Conversion\s*:|csr2tile.*?)\s*([\d.]+)\s*ms', re.IGNORECASE)

# Update each row (clear old values then refill)
updated = 0
total = len(rows)
for i, row in enumerate(rows, 1):
    matrix_name = row['matrix_name']
    gpu = row['gpu']
    mode = row['mode']
    pred_combo = row['pred_combo']

    # Clear old results first
    row['runtime_ms'] = ''
    row['gflops'] = ''
    row['csr2tile_ms'] = ''

    # Task 2 only executes A100; H200 rows remain empty to avoid using A100 logs
    if gpu != 'A100':
        continue

    # Parse pred_combo
    parts = pred_combo.split('_')
    tile_part = parts[0]
    tc_part = parts[1] if len(parts) > 1 else '0/8'
    tile_m, tile_n = tile_part.split('x')
    tc_numerator = tc_part.split('/')[0]

    # Log file
    log_file = os.path.join(LOG_DIR, f"{matrix_name}_{mode}_{tile_m}x{tile_n}_tc{tc_numerator}.log")
    if not os.path.exists(log_file):
        continue

    with open(log_file, 'r', errors='ignore') as f:
        log_content = f.read()

    runtime_match = runtime_pattern.search(log_content)
    if runtime_match:
        row['runtime_ms'] = runtime_match.group(1)

    gflops_match = gflops_pattern.search(log_content)
    if gflops_match:
        row['gflops'] = gflops_match.group(1)

    csr2tile_match = csr2tile_pattern.search(log_content)
    if csr2tile_match:
        row['csr2tile_ms'] = csr2tile_match.group(1)

    if runtime_match:
        updated += 1

    if i % 20 == 0:
        print(f"  Progress: {i}/{total} logs parsed")

print(f"  ✓ Updated {updated}/{total} records")

# Write updated CSV (overwrite old file)
with open(RESULT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"  ✓ Results written to: {RESULT_CSV}")

# Update probe.csv
print()
print("  Updating probe.csv...")
try:
    import pandas as pd
    probe_df = pd.read_csv(PROBE_CSV)

    if 'csr2tile_ms' not in probe_df.columns:
        probe_df['csr2tile_ms'] = float('nan')
    if 'pipeline_overhead_ms' not in probe_df.columns:
        probe_df['pipeline_overhead_ms'] = float('nan')

    result_df = pd.read_csv(RESULT_CSV)
    updated_probe = 0

    for idx, probe_row in probe_df.iterrows():
        matrix_name = probe_row['matrix_name']
        mode = probe_row['mode']
        gpu = probe_row['gpu']

        if gpu != 'A100':
            continue

        match = result_df[(result_df['matrix_name'] == matrix_name) &
                          (result_df['mode'] == mode) &
                          (result_df['gpu'] == gpu)]

        if not match.empty:
            try:
                probe_df.at[idx, 'csr2tile_ms'] = float('nan')
                probe_df.at[idx, 'pipeline_overhead_ms'] = float('nan')

                csr2tile_value = match.iloc[0]['csr2tile_ms']
                if pd.notna(csr2tile_value) and str(csr2tile_value).strip() != '':
                    csr2tile_ms = float(csr2tile_value)
                    probe_df.at[idx, 'csr2tile_ms'] = round(csr2tile_ms, 2)

                    a_probe_ms = probe_row['A_probe_ms']
                    c_probe_total = probe_row['C_probe_total_ms']
                    lightgbm_decision = probe_row.get('lightgbm_decision_ms', 0)

                    if pd.notna(a_probe_ms) and pd.notna(c_probe_total) and pd.notna(lightgbm_decision):
                        a_probe_ms = float(a_probe_ms)
                        c_probe_total = float(c_probe_total)
                        lightgbm_decision = float(lightgbm_decision) if lightgbm_decision else 0
                        pipeline_overhead = a_probe_ms + c_probe_total + csr2tile_ms + lightgbm_decision
                        probe_df.at[idx, 'pipeline_overhead_ms'] = round(pipeline_overhead, 2)
                updated_probe += 1
            except:
                pass

    probe_df.to_csv(PROBE_CSV, index=False)
    print(f"  ✓ probe.csv updated ({updated_probe} entries)")
except Exception as e:
    print(f"  ⚠ Failed to update probe.csv: {e}")

PYTHON_SCRIPT

echo ""
echo "================================================================================"
echo "✓ Task 2 Completed!"
echo ""
echo "  Result file: $RESULT_CSV"
echo "  Log directory: $LOG_DIR"
echo "================================================================================"
