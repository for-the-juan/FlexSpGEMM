#!/bin/bash
# Task 2: 运行A100上的AA与AAT乘法，追加结果到test100_result.csv
# 使用FlexSpGEMM的9个二进制 + -tau 参数实现81种配置

set -e

# 自动激活conda环境
source /home/stu1/miniconda3/bin/activate FlexSpGEMM
export LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH:-}"

# 路径配置
BIN_DIR="/home/stu1/donghangcheng/code/FlexSpGEMM/bin"
MATRIX_DIR="/home/stu1/donghangcheng/code/FlexSpGEMM/data/test"
RESULT_CSV="/home/stu1/donghangcheng/code/FlexSpGEMM/ML_method/LightGBM/predictResult_test100/test100_result.csv"
LOG_DIR="/home/stu1/donghangcheng/code/FlexSpGEMM/ML_method/LightGBM/predictResult_test100/log"
PROBE_CSV="/home/stu1/donghangcheng/code/FlexSpGEMM/data/data_prepare/data_get/probe.csv"

echo "================================================================================"
echo "  Task 2: 运行A100上的AA与AAT乘法"
echo "================================================================================"
echo ""

# 检查test100_result.csv是否存在
if [ ! -f "$RESULT_CSV" ]; then
    echo "错误: $RESULT_CSV 不存在，请先运行 python3 predict_test100.py"
    exit 1
fi

# 清空旧日志
echo "[步骤 1/3] 准备工作"
echo "  清空旧日志目录: $LOG_DIR"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

total_lines=$(tail -n +2 "$RESULT_CSV" | wc -l)
echo "  待处理条目: $total_lines"
echo ""

echo "[步骤 2/3] 执行SpGEMM计算"
echo "  二进制目录: $BIN_DIR"
echo "  矩阵目录:   $MATRIX_DIR"
echo ""

# 计数器
current=0
success=0
failed=0
skipped=0

# 跳过表头，逐行处理
while IFS=',' read -r matrix_name gpu mode pred_combo runtime_ms gflops csr2tile_ms; do

    current=$((current + 1))

    # 只处理A100的数据
    if [ "$gpu" != "A100" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    # 解析pred_combo (格式: 16x16_0/8)
    tile_part=$(echo "$pred_combo" | cut -d'_' -f1)
    tc_part=$(echo "$pred_combo" | cut -d'_' -f2)
    tile_m=$(echo "$tile_part" | cut -d'x' -f1)
    tile_n=$(echo "$tile_part" | cut -d'x' -f2)
    tc_numerator=$(echo "$tc_part" | cut -d'/' -f1)

    # 确定aat参数
    if [ "$mode" = "AA" ]; then
        aat_flag=0
    else
        aat_flag=1
    fi

    # 矩阵文件路径
    matrix_file="$MATRIX_DIR/${matrix_name}.mtx"
    if [ ! -f "$matrix_file" ]; then
        echo "  [$current/$total_lines] ⚠ 跳过 ${matrix_name} ${mode} - 矩阵文件不存在"
        failed=$((failed + 1))
        continue
    fi

    tau_value=$(awk -v tc="$tc_numerator" 'BEGIN { printf "%.3f", tc / 8.0 }')

    # 选择对应的FlexSpGEMM二进制文件
    binary="$BIN_DIR/test_m${tile_m}_n${tile_n}"
    if [ ! -f "$binary" ]; then
        echo "  [$current/$total_lines] ⚠ 跳过 ${matrix_name} ${mode} - 二进制文件不存在: test_m${tile_m}_n${tile_n}"
        failed=$((failed + 1))
        continue
    fi

    # 日志文件
    log_file="$LOG_DIR/${matrix_name}_${mode}_${tile_m}x${tile_n}_tc${tc_numerator}.log"

    echo "  [$current/$total_lines] 运行: ${matrix_name} | ${mode} | tile=${tile_m}x${tile_n} | TC=${tc_part} | tau=${tau_value}"

    # 运行SpGEMM（允许非0退出码，后续从日志判断是否成功）
    set +e
    (
        cd "$BIN_DIR" &&
        stdbuf -oL -eL "./$(basename "$binary")" -d 0 -aat "$aat_flag" -tau "$tau_value" "$matrix_file"
    ) > "$log_file" 2>&1
    run_status=$?
    set -e

    # 从日志中提取gflops和runtime
    gf=$(grep -oP '(?i)(?:Throughput\s*:|gflops\s*=)\s*\K[\d.]+' "$log_file" 2>/dev/null | head -1)
    rt=$(grep -oP '(?i)(?:Total Runtime\s*:|TileSpGEMM\s+runtime\s+is\s+)\s*\K[\d.]+' "$log_file" 2>/dev/null | head -1)

    if [ -n "$gf" ]; then
        echo "             ✓ 执行成功 | runtime=${rt}ms | gflops=${gf}"
        success=$((success + 1))
    elif grep -qi "does not do symmetric matrix" "$log_file" 2>/dev/null; then
        echo "             ⊘ 跳过对称矩阵（AAT模式不支持）"
        skipped=$((skipped + 1))
    else
        echo "             ✗ 执行失败 (exit=${run_status})，未获取到数据"
        failed=$((failed + 1))
    fi

done < <(tail -n +2 "$RESULT_CSV")

echo ""
echo "  执行统计: 成功=$success, 失败=$failed, 跳过=$skipped"
echo ""

echo "[步骤 3/3] 从日志中提取性能数据并更新CSV"

python << 'PYTHON_SCRIPT'
import csv
import re
import os
import sys

RESULT_CSV = "/home/stu1/donghangcheng/code/FlexSpGEMM/ML_method/LightGBM/predictResult_test100/test100_result.csv"
LOG_DIR = "/home/stu1/donghangcheng/code/FlexSpGEMM/ML_method/LightGBM/predictResult_test100/log"
PROBE_CSV = "/home/stu1/donghangcheng/code/FlexSpGEMM/data/data_prepare/data_get/probe.csv"

# 读取现有的CSV
rows = []
with open(RESULT_CSV, 'r') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        rows.append(row)

# 正则表达式
runtime_pattern = re.compile(r'(?:Total Runtime\s*:|TileSpGEMM\s+runtime\s+is\s+)\s*([\d.]+)\s*ms', re.IGNORECASE)
gflops_pattern = re.compile(r'(?:Throughput\s*:|gflops\s*=)\s*([\d.]+)', re.IGNORECASE)
csr2tile_pattern = re.compile(r'(?:Format Conversion\s*:|csr2tile.*?)\s*([\d.]+)\s*ms', re.IGNORECASE)

# 更新每一行（清空旧值后重新填充）
updated = 0
total = len(rows)
for i, row in enumerate(rows, 1):
    matrix_name = row['matrix_name']
    gpu = row['gpu']
    mode = row['mode']
    pred_combo = row['pred_combo']

    # 先清空旧结果
    row['runtime_ms'] = ''
    row['gflops'] = ''
    row['csr2tile_ms'] = ''

    # Task 2 只执行 A100；H200 行保持为空，避免误用 A100 日志回填
    if gpu != 'A100':
        continue

    # 解析pred_combo
    parts = pred_combo.split('_')
    tile_part = parts[0]
    tc_part = parts[1] if len(parts) > 1 else '0/8'
    tile_m, tile_n = tile_part.split('x')
    tc_numerator = tc_part.split('/')[0]

    # 日志文件
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
        print(f"  进度: {i}/{total} 条日志已解析")

print(f"  ✓ 共更新 {updated}/{total} 条记录")

# 写入更新后的CSV（覆盖旧文件）
with open(RESULT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"  ✓ 结果已覆盖写入: {RESULT_CSV}")

# 更新probe.csv
print()
print("  更新probe.csv...")
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
    print(f"  ✓ probe.csv已更新 ({updated_probe}条)")
except Exception as e:
    print(f"  ⚠ 更新probe.csv失败: {e}")

PYTHON_SCRIPT

echo ""
echo "================================================================================"
echo "✓ Task 2 完成!"
echo ""
echo "  结果文件: $RESULT_CSV"
echo "  日志目录: $LOG_DIR"
echo "================================================================================"
