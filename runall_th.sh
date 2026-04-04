#!/bin/bash

TILE_SIZE_M=(8 16 32 8 16 32 16 8 32)
TILE_SIZE_N=(8 16 32 16 8 16 32 32 8)
FRACTIONS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15)

AA="/home/stu1/Dataset/TileSpGEMMDataset"

mkdir -p ./log

for mtx_file in "$AA"/*.mtx; do
    base_name_=$(basename "$mtx_file" .mtx)
    mkdir -p "./log/${base_name_}"
    
    echo "处理矩阵: ${base_name_}"
    
    for ((i=0; i<${#TILE_SIZE_M[@]}; i++)); do
        m=${TILE_SIZE_M[i]}
        n=${TILE_SIZE_N[i]}
        
        for frac in "${FRACTIONS[@]}"; do
            exec_name="test_m${m}_n${n}_frac${frac}"
            
            if [ ! -f "./bin/${exec_name}" ]; then
                echo "警告: 可执行文件 ./bin/${exec_name} 不存在，跳过测试"
                continue
            fi
            
            echo "运行: ${base_name_}_m${m}_n${n}_frac${frac}"
            
            ./bin/${exec_name} -d 0 -aat 0 "$mtx_file" > log/${base_name_}/aat0_m${m}_n${n}_frac${frac}.log
            
            echo "${base_name_}_aat0_m${m}_n${n}_frac${frac} 完成!"
        done
    done
done

echo "所有测试完成!"