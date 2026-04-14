#!/bin/bash
# Compile_TC.sh - 编译81个二进制文件（9 tile × 9 TC阈值）

CC="nvcc"
VALUE_TYPE="double"
NVCC_FLAGS="-O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80 -std=c++17"
CUDA_INSTALL_PATH="/usr/local/cuda-11.8"
INCLUDES="-I${CUDA_INSTALL_PATH}/include"
CUDA_LIBS="-L${CUDA_INSTALL_PATH}/lib64 -lcudart -lcusparse"
LIBS=${CUDA_LIBS}

TILE_SIZE_M=(8 16 32)
TILE_SIZE_N=(8 16 32)
TC_FRACS=(0 1 2 3 4 5 6 7 8)

mkdir -p ../bin_tc
rm -rf ../bin_tc/*

echo "编译81个二进制文件..."
count=0
total=81

for i in ${TILE_SIZE_M[@]}; do
    for j in ${TILE_SIZE_N[@]}; do
        for frac in ${TC_FRACS[@]}; do
            smem_lrg_th=$(( (i * i * frac) / 8 ))
            if [ $smem_lrg_th -lt 1 ]; then
                smem_lrg_th=1
            fi

            OUTNAME="../bin_tc/test_m${i}_n${j}_tc${frac}"
            ${CC} ${NVCC_FLAGS} -Xcompiler -fopenmp -Xcompiler -mfma main.cu \
                -o ${OUTNAME} ${INCLUDES} ${LIBS} ${OPTIONS} \
                -D VALUE_TYPE=${VALUE_TYPE} \
                -D TILE_SIZE_M=${i} \
                -D TILE_SIZE_N=${j} \
                -D SMEM_LRG_TH=${smem_lrg_th} \
                -D CHECK_RESULT=0 &

            count=$((count + 1))
            # 每9个并行编译一批
            if [ $((count % 9)) -eq 0 ]; then
                wait
                echo "  进度: $count/$total"
            fi
        done
    done
done

wait
echo "  进度: $total/$total"

COMPILED=$(ls ../bin_tc/test_m*_n*_tc* 2>/dev/null | wc -l)
echo "✓ 编译完成! 共 ${COMPILED} 个二进制文件"
