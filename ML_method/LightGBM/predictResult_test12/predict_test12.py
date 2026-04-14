#!/home/stu1/miniconda3/envs/FlexSpGEMM/bin/python
# -*- coding: utf-8 -*-
"""
predict_test12.py - Task 3: 对test12矩阵执行完整预测流程
包含预测、执行和基于81种实测组合的最佳gflops比较
"""

import csv
import os
import re
import subprocess
import time
import pandas as pd
import lightgbm as lgb

# 路径配置
MODEL_PATH = "/home/stu1/donghangcheng/code/FlexSpGEMM/ML_method/LightGBM/quick_predict_model/model_tuned.txt"
TEST_DATASET_CSV = "/home/stu1/donghangcheng/code/FlexSpGEMM/data/data_prepare/data_get/test.csv"
TEST12_FILE = "/home/stu1/donghangcheng/code/FlexSpGEMM/ML_method/LightGBM/predictResult_test12/test12mtx.txt"
OUTPUT_CSV = "/home/stu1/donghangcheng/code/FlexSpGEMM/ML_method/LightGBM/predictResult_test12/test12_result.csv"
LOG_DIR = "/home/stu1/donghangcheng/code/FlexSpGEMM/ML_method/LightGBM/predictResult_test12/log"
BIN_DIR = "/home/stu1/donghangcheng/code/FlexSpGEMM/bin"
MATRIX_DIR = "/home/stu1/donghangcheng/code/FlexSpGEMM/data/test"
PROBE_CSV = "/home/stu1/donghangcheng/code/FlexSpGEMM/data/data_prepare/data_get/probe.csv"

# 81个配置组合
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
    """读取test12矩阵列表"""
    with open(TEST12_FILE, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    matrices = [m.strip() for m in content.replace('、', ',').split(',')]
    return matrices

def parse_log_metrics(log_content):
    """从运行日志中提取runtime/gflops/csr2tile"""
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
    """运行SpGEMM"""
    matrix_file = os.path.join(MATRIX_DIR, f"{matrix_name}.mtx")

    if not os.path.exists(matrix_file):
        print(f"      ⚠ 矩阵文件不存在")
        return None, None, None

    binary = os.path.join(BIN_DIR, f"test_m{tile_m}_n{tile_n}")

    if not os.path.exists(binary):
        print(f"      ⚠ 二进制文件不存在")
        return None, None, None

    aat_flag = 0 if mode == "AA" else 1
    tau_value = f"{int(tc_numerator) / 8.0:.3f}"
    log_file = os.path.join(LOG_DIR, f"{matrix_name}_{mode}_{tile_m}x{tile_n}_tc{tc_numerator}.log")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/usr/local/cuda-11.8/lib64:{env.get('LD_LIBRARY_PATH', '')}"

    # 运行SpGEMM
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                ['stdbuf', '-oL', '-eL', binary, '-d', '0', '-aat', str(aat_flag), '-tau', tau_value, matrix_file],
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=300,
                env=env,
                cwd=BIN_DIR
            )

        # 检查是否成功（不使用check=True，因为某些配置会SIGSEGV）
        # 注意：即使exit code非0，程序可能已经输出了结果，继续尝试提取
    except subprocess.TimeoutExpired:
        print(f"      ✗ 执行超时 (>300秒)")
        return None, None, None
    except Exception as e:
        print(f"      ✗ 执行失败: {e}")
        return None, None, None

    with open(log_file, 'r', errors='ignore') as f:
        log_content = f.read()

    runtime, gflops, csr2tile = parse_log_metrics(log_content)

    # 根据提取结果判断是否成功
    if gflops is not None:
        print(f"      ✓ 执行成功 | runtime={runtime:.3f}ms | gflops={gflops:.3f}")
    elif "does not do symmetric matrix" in log_content.lower():
        print(f"      ⊘ 跳过对称矩阵（AAT模式不支持）")
    elif result.returncode != 0:
        print(f"      ✗ 程序崩溃 (exit code: {result.returncode})，未获取到数据")
    else:
        print(f"      ⚠ 执行完成但未找到性能数据")

    return runtime, gflops, csr2tile

def run_all_combos(matrix_name, mode):
    """Run all 81 combinations and return the measured metrics keyed by combo."""
    all_results = {}
    for combo in COMBOS:
        tile_part, tc_part = combo.split('_')
        tile_m, tile_n = tile_part.split('x')
        tc_numerator = tc_part.split('/')[0]
        print(f"      运行配置 {combo}")
        runtime, gflops, csr2tile = run_spgemm(matrix_name, mode, tile_m, tile_n, tc_numerator)
        all_results[combo] = {
            'runtime': runtime,
            'gflops': gflops,
            'csr2tile': csr2tile,
        }
    return all_results

def update_probe_csv(out_rows, decision_ms_per_row):
    """Overwrite probe.csv values for the test12 A100 rows."""
    print("  更新probe.csv...")
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
        print(f"  ✓ probe.csv已更新 ({updated}条)")
    except Exception as e:
        print(f"  ⚠ 更新probe.csv失败: {e}")

def main():
    print("=" * 80)
    print("  Task 3: test12矩阵完整预测流程")
    print("=" * 80)
    print()

    # 清空旧日志
    print("[步骤 1/6] 准备工作")
    print(f"  清空旧日志目录: {LOG_DIR}")
    if os.path.exists(LOG_DIR):
        import shutil
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)
    print()

    # 读取test12矩阵列表
    print("[步骤 2/6] 读取test12矩阵列表")
    test12_matrices = load_test12_matrices()
    print(f"  ✓ 共 {len(test12_matrices)} 个矩阵")
    for i, m in enumerate(test12_matrices, 1):
        print(f"    {i}. {m}")
    print()

    # 加载模型
    print("[步骤 3/6] 加载LightGBM模型")
    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"  ✓ 模型加载完成 (树数量: {model.num_trees()})")
    print()

    # 加载test数据
    print("[步骤 4/6] 加载测试数据并执行预测")
    test_df = pd.read_csv(TEST_DATASET_CSV)
    test_df = annotate_gpu_mode(test_df)
    test12_df = test_df[
        test_df['matrix_name'].isin(test12_matrices) &
        (test_df['gpu'] == 'A100')
    ].copy()
    print(f"  ✓ test12数据集样本数: {len(test12_df)}")

    feature_cols = [c for c in test_df.columns if c not in ['matrix_name', 'best_tile', 'best_tc', 'gpu', 'mode']]
    X_test = test12_df[feature_cols].astype(float)

    start_time = time.time()
    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)
    decision_time = time.time() - start_time
    decision_ms_per_row = decision_time * 1000 / len(test12_df) if len(test12_df) > 0 else 0

    print(f"  ✓ 预测完成 (耗时: {decision_time * 1000:.2f}ms)")
    print()

    # 执行SpGEMM
    print("[步骤 5/6] 执行SpGEMM计算，并对81种配置进行实测比较")
    out_rows = []
    total_tasks = len(test12_df)
    current_task = 0

    for pos, (_, row) in enumerate(test12_df.iterrows()):
        matrix_name = row['matrix_name']
        mode = row['mode']
        pred_idx = y_pred[pos]
        pred_combo = IDX_TO_COMBO[pred_idx]

        current_task += 1
        print(f"  [{current_task}/{total_tasks}] {matrix_name} | {mode} | 预测配置={pred_combo}")

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
            print(f"      → 决策配置实测gflops={gflops:.3f}")
        else:
            print(f"      ⚠ 决策配置未获得有效gflops")

        if best_gflops > 0:
            print(f"      ✓ 81种组合实测最佳gflops={best_gflops:.3f}")

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

    # 写入CSV（覆盖旧文件）
    print("[步骤 6/6] 生成输出文件")
    header = ['matrix_name', 'gpu', 'mode', 'pred_combo', 'runtime_ms', 'gflops',
              'csr2tile_ms', 'best_gflops', 'gflops_ratio']

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"  ✓ 结果已保存: {OUTPUT_CSV}")
    print(f"    总条目数: {len(out_rows)}")
    print()

    update_probe_csv(out_rows, decision_ms_per_row)
    print()

    # 统计
    ratios = [float(r['gflops_ratio']) for r in out_rows if r['gflops_ratio']]
    if ratios:
        print("  GFLOPS比率统计:")
        print(f"    平均值: {sum(ratios)/len(ratios):.4f} ({sum(ratios)/len(ratios)*100:.2f}%)")
        print(f"    最小值: {min(ratios):.4f} ({min(ratios)*100:.2f}%)")
        print(f"    最大值: {max(ratios):.4f} ({max(ratios)*100:.2f}%)")
        print(f"    ≥95%: {sum(1 for r in ratios if r >= 0.95)}/{len(ratios)}")
        print(f"    ≥90%: {sum(1 for r in ratios if r >= 0.90)}/{len(ratios)}")

    print()
    print("=" * 80)
    print("✓ Task 3 完成!")
    print("=" * 80)

if __name__ == '__main__':
    main()
