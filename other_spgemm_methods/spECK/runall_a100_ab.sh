#!/bin/bash

DATASET_DIR_A="/home/stu1/Dataset/TileSpGEMMDataset/"
DATASET_DIR_B="/home/stu1/Dataset/TileSpGEMMDataset_t/"
LOG_DIR="/home/stu1/donghangcheng/spECK_log/TileSpGEMM_a100_aat"

mkdir -p "$LOG_DIR"

TIMEOUT_SECONDS=150

find "$DATASET_DIR_A" -name "*.mtx" -type f | sort | while read -r mtx_file_a; do
    base_name_=$(basename "$mtx_file_a" .mtx)
    mtx_file_b="${DATASET_DIR_B}${base_name_}.mtx"

    # 检查转置矩阵是否存在
    if [ ! -f "$mtx_file_b" ]; then
        echo "警告: 转置矩阵 ${base_name_}.mtx 不存在，跳过"
        continue
    fi

    echo "运行: ${base_name_} (A * A^T)"
    timeout $TIMEOUT_SECONDS ./speck "$mtx_file_a" "$mtx_file_b" > "$LOG_DIR/${base_name_}.log"

    exit_code=$?
    if [ $exit_code -eq 124 ]; then
        echo "警告: 运行超时（超过$(($TIMEOUT_SECONDS/60))分钟），已跳过"
        continue
    elif [ $exit_code -ne 0 ]; then
        echo "警告: 运行失败，退出码: $exit_code"
        continue
    fi
done

echo "所有测试完成!"