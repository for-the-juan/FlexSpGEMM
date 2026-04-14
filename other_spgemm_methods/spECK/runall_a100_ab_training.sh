#!/bin/bash

DATASET_DIR_A="/home/stu1/Dataset/training_dataset/"
DATASET_DIR_B="/home/stu1/Dataset/training_dataset_t/"
LOG_DIR="/home/stu1/donghangcheng/spECK_log/training_a100_aat"

mkdir -p "$LOG_DIR"

TIMEOUT_SECONDS=150

# 遍历training_dataset目录下的所有子目录中的.mtx文件
find "$DATASET_DIR_A" -name "*.mtx" -type f | sort | while read -r mtx_file_a; do
    base_name_=$(basename "$mtx_file_a" .mtx)
    
    # 在training_dataset_t目录下查找对应的转置矩阵（可能在不同的子目录中）
    mtx_file_b=$(find "$DATASET_DIR_B" -name "${base_name_}.mtx" -type f | head -n 1)

    # 检查转置矩阵是否存在
    if [ -z "$mtx_file_b" ] || [ ! -f "$mtx_file_b" ]; then
        echo "警告: 转置矩阵 ${base_name_}.mtx 不存在，跳过"
        continue
    fi

    echo "运行: ${base_name_} (A * A^T)"
    echo "  矩阵A: $mtx_file_a"
    echo "  矩阵B: $mtx_file_b"
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