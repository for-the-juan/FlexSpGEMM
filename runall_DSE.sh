#!/bin/bash

TILE_SIZE_M=16
TILE_SIZE_N=16
FRACTIONS1=(0 1 2 3 4 5 6 7 8 9 10)
FRACTIONS2=(1 2 4)

AA1="/home/stu1/Dataset/TileSpGEMMDataset/gupta3.mtx"

# mkdir -p ./log_DSE

base_name1_=$(basename "$AA1" .mtx)
mkdir -p "./log_DSE/${base_name1_}"

echo "处理矩阵: ${base_name1_}"
        
for frac1 in "${FRACTIONS1[@]}"; do
    for frac2 in "${FRACTIONS2[@]}"; do
        exec_name="test_m${TILE_SIZE_M}_n${TILE_SIZE_N}_dns${frac1}_slots${frac2}"
        if [ ! -f "./bin_DSE/${exec_name}" ]; then
            echo "警告: 可执行文件 ./bin_DSE/${exec_name} 不存在，跳过测试"
            continue
        fi
        ./bin_DSE/${exec_name} -d 0 -aat 0 "$AA1" > log_DSE/${base_name1_}/aat0_m${TILE_SIZE_M}_n${TILE_SIZE_N}_dns${frac1}_slots${frac2}.log
        echo "${base_name1_}_aat0_m${TILE_SIZE_M}_n${TILE_SIZE_N}_dns${frac1}_slots${frac2} 完成!"
    done
done

# for frac1 in "${FRACTIONS1[@]}"; do
#     for frac2 in "${FRACTIONS2[@]}"; do
#         exec_name="test_m${TILE_SIZE_M}_n${TILE_SIZE_N}_dns${frac1}_slots${frac2}"
#         if [ ! -f "./bin_DSE/${exec_name}" ]; then
#             echo "警告: 可执行文件 ./bin_DSE/${exec_name} 不存在，跳过测试"
#             continue
#         fi
#         ./bin_DSE/${exec_name} -d 0 -aat 0 "$AA2" > log_DSE/${base_name2_}/aat0_m${TILE_SIZE_M}_n${TILE_SIZE_N}_dns${frac1}_slots${frac2}.log
#         echo "${base_name2_}_aat0_m${TILE_SIZE_M}_n${TILE_SIZE_N}_dns${frac1}_slots${frac2} 完成!"
#     done
# done

echo "所有测试完成!"