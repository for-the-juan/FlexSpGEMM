#!/home/stu1/miniconda3/envs/FlexSpGEMM/bin/python
# -*- coding: utf-8 -*-
"""
predict_test100.py - Task 1: 使用LightGBM模型对test矩阵进行预测
生成test100_result.csv，包含预测的tile尺寸和TC阈值
"""

import csv
import time
import pandas as pd
import lightgbm as lgb

# 路径配置
MODEL_PATH = "/home/stu1/donghangcheng/code/FlexSpGEMM/ML_method/LightGBM/quick_predict_model/model_tuned.txt"
TEST_DATASET_CSV = "/home/stu1/donghangcheng/code/FlexSpGEMM/data/data_prepare/data_get/test.csv"
PROBE_CSV = "/home/stu1/donghangcheng/code/FlexSpGEMM/data/data_prepare/data_get/probe.csv"
OUTPUT_CSV = "/home/stu1/donghangcheng/code/FlexSpGEMM/ML_method/LightGBM/predictResult_test100/test100_result.csv"

# 81个配置组合
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
    print("  Task 1: 使用LightGBM模型预测test矩阵的最佳配置")
    print("=" * 80)
    print()

    # 加载模型
    print("[步骤 1/5] 加载LightGBM模型")
    print(f"  模型路径: {MODEL_PATH}")
    start_time = time.time()
    model = lgb.Booster(model_file=MODEL_PATH)
    model_load_time = time.time() - start_time
    print(f"  ✓ 模型加载完成")
    print(f"    - 耗时: {model_load_time:.3f}秒")
    print(f"    - 树数量: {model.num_trees()}")
    print()

    # 加载test数据
    print("[步骤 2/5] 加载测试数据集")
    print(f"  数据路径: {TEST_DATASET_CSV}")
    test_df = pd.read_csv(TEST_DATASET_CSV)
    test_df = annotate_gpu_mode(test_df)
    print(f"  ✓ 数据加载完成")
    print(f"    - 样本总数: {len(test_df)}")
    print(f"    - 矩阵数量: {test_df['matrix_name'].nunique()}")
    print()

    # 获取特征列
    feature_cols = [c for c in test_df.columns if c not in ['matrix_name', 'best_tile', 'best_tc', 'gpu', 'mode']]
    print(f"  特征列数量: {len(feature_cols)}")
    print()

    # 预测
    print("[步骤 3/5] 执行预测")
    X_test = test_df[feature_cols].astype(float)

    start_time = time.time()
    print(f"  正在预测 {len(test_df)} 个样本...")
    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)
    decision_time = time.time() - start_time

    print(f"  ✓ 预测完成")
    print(f"    - 总耗时: {decision_time * 1000:.2f}ms")
    print(f"    - 平均决策时间: {decision_time * 1000 / len(test_df):.3f}ms/样本")
    print()

    # 生成输出
    print("[步骤 4/5] 生成预测结果")

    matrix_names = test_df['matrix_name'].unique()
    print(f"  处理 {len(test_df)} 个样本行...")

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
            print(f"    进度: {i}/{len(test_df)} 样本已处理")

    print(f"  ✓ 结果生成完成")
    print(f"    - 总条目数: {len(out_rows)}")
    print()

    # 写入CSV（覆盖旧文件）
    print(f"  写入结果文件: {OUTPUT_CSV}")
    header = ['matrix_name', 'gpu', 'mode', 'pred_combo', 'runtime_ms', 'gflops', 'csr2tile_ms']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"  ✓ 文件已保存（旧文件已覆盖）")
    print()

    # 更新probe.csv
    print("[步骤 5/5] 更新probe.csv")
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
        print(f"  ✓ probe.csv已更新")
        print(f"    - 更新条目数: {updated_count}")
        print(f"    - 平均决策时间: {decision_ms_per_row:.3f}ms/样本")
    except Exception as e:
        print(f"  ⚠ 警告: 更新probe.csv失败: {e}")

    print()
    print("=" * 80)
    print("✓ Task 1 完成!")
    print()
    print("下一步: 运行 ./run_test100.sh 执行实际的SpGEMM计算")
    print("=" * 80)

if __name__ == '__main__':
    main()
