#!/bin/bash

DATASET_DIR="/home/stu1/Dataset/training_dataset"
LOG_DIR="/home/stu1/donghangcheng/spECK_log/training_dataset_a100"

mkdir -p "$LOG_DIR"

TIMEOUT_SECONDS=150

find "$DATASET_DIR" -name "*.mtx" -type f | sort | while read -r mtx_file; do
    base_name_=$(basename "$mtx_file" .mtx)

    echo "运行: ${base_name_}"
    timeout $TIMEOUT_SECONDS ./speck "$mtx_file" > "$LOG_DIR/${base_name_}.log"

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