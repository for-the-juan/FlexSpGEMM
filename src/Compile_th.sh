#!/bin/bash

#compilers
CC="nvcc"

#GLOBAL_PARAMETERS
MAT_VAL_TYPE="double"
VALUE_TYPE="double"

#CUDA_PARAMETERS
NVCC_FLAGS="-O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80 -std=c++17"

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH="/usr/local/cuda-11.8"

#includes
INCLUDES="-I${CUDA_INSTALL_PATH}/include"

#libs
CUDA_LIBS="-L${CUDA_INSTALL_PATH}/lib64 -lcudart -lcusparse"
LIBS=${CUDA_LIBS}

TILE_SIZE_M=(8 16 32)
TILE_SIZE_N=(8 16 32)

# SMEM_LRG_TH fractions from 2/16 to 15/16
FRACTIONS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15)

mkdir -p ../bin
rm -rf ../bin/*

for i in ${TILE_SIZE_M[@]}; do
    for j in ${TILE_SIZE_N[@]}; do
        for frac in ${FRACTIONS[@]}; do
            # Calculate SMEM_LRG_TH value
            smem_lrg_th=$(( (i * i * frac) / 16 ))
            
            # Generate executable with specific naming
            exec_name="test_m${i}_n${j}_frac${frac}"
            
            ${CC} ${NVCC_FLAGS} -Xcompiler -fopenmp -Xcompiler -mfma main.cu \
                -o "../bin/${exec_name}" \
                ${INCLUDES} ${LIBS} \
                -D VALUE_TYPE=${VALUE_TYPE} \
                -D TILE_SIZE_M=${i} \
                -D TILE_SIZE_N=${j} \
                -D SMEM_LRG_TH=${smem_lrg_th} &

        done
    done
done

# Wait for all background jobs to complete
wait

echo "Compilation complete. Executables are in ../bin/"