#!/bin/bash
# spECK 测试脚本：遍历两个数据集目录下的 .mtx 文件，每次运行间隔 5s，结果写入 md 文件
# 某个矩阵失败/崩溃时也会把输出写入 md 并继续测试其余矩阵

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPECK="${SCRIPT_DIR}/speck"
DIR1="/home/stu1/Dataset/TileSpGEMMDataset"
DIR2="/home/stu1/Dataset/HYTEDataset"
OUTPUT_MD="${SCRIPT_DIR}/speck_benchmark_results.md"
SLEEP_SEC=5

if [[ ! -x "$SPECK" ]]; then
    echo "错误: 可执行文件不存在或不可执行: $SPECK"
    exit 1
fi

for d in "$DIR1" "$DIR2"; do
    if [[ ! -d "$d" ]]; then
        echo "警告: 目录不存在，跳过: $d"
        continue
    fi
done

# 初始化 md 文件
{
    echo "# spECK 测试结果"
    echo ""
    echo "生成时间: $(date -Iseconds)"
    echo "可执行文件: $SPECK"
    echo "数据集目录: $DIR1, $DIR2"
    echo "每次运行间隔: ${SLEEP_SEC}s"
    echo ""
    echo "---"
    echo ""
} > "$OUTPUT_MD"

total=0
for dir in "$DIR1" "$DIR2"; do
    [[ ! -d "$dir" ]] && continue
    dir_name=$(basename "$dir")
    while IFS= read -r -d '' mtx; do
        total=$((total + 1))
        name=$(basename "$mtx")
        echo "[$total] $dir_name / $name"
        tmpout=$(mktemp)
        "$SPECK" "$mtx" > "$tmpout" 2>&1
        ret=$?
        {
            echo "## \`$dir_name/$name\`"
            if [[ $ret -ne 0 ]]; then
                echo ""
                echo "**（运行失败/崩溃，退出码: $ret）**"
            fi
            echo ""
            echo "\`\`\`"
            # 去掉终端 ANSI 颜色码，避免 md 里出现 [1;31m 等不可见字符
            sed 's/\x1b\[[0-9;]*m//g' "$tmpout"
            echo "\`\`\`"
            echo ""
        } >> "$OUTPUT_MD"
        rm -f "$tmpout"
        if [[ $ret -ne 0 ]]; then
            echo "  失败 (退出码 $ret)，已写入 md，继续下一个"
        fi
        echo "  休息 ${SLEEP_SEC}s ..."
        sleep "$SLEEP_SEC"
    done < <(find "$dir" -maxdepth 1 -name '*.mtx' -print0 | sort -z)
done

echo ""
echo "全部完成，共 $total 个矩阵，结果已写入: $OUTPUT_MD"
