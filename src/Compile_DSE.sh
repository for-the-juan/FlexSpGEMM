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

TILE_SIZE_M=16
TILE_SIZE_N=16

# SMEM_LRG_TH fractions from 2/16 to 15/16
FRACTIONS_DNSTHRESHOLD=(0 1 2 3 4 5 6 7 8 9 10)
FRACTIONS_NUM_SLOTS=(1 2 4)

mkdir -p ../bin_DSE
rm -rf ../bin_DSE/*

for i in ${FRACTIONS_DNSTHRESHOLD[@]}; do
    for j in ${FRACTIONS_NUM_SLOTS[@]}; do
            # Generate executable with specific naming
            exec_name="test_m16_n16_dns${i}_slots${j}"
            
            ${CC} ${NVCC_FLAGS} -Xcompiler -fopenmp -Xcompiler -mfma main.cu \
                -o "../bin_DSE/${exec_name}" \
                ${INCLUDES} ${LIBS} \
                -D VALUE_TYPE=${VALUE_TYPE} \
                -D TILE_SIZE_M=${TILE_SIZE_M} \
                -D TILE_SIZE_N=${TILE_SIZE_N} \
                -D TILE_DENSE_THRESHOLD=${i} \
                -D NUM_SHARED_SLOTS=${j} &

    done
done

# Wait for all background jobs to complete
wait

echo "Compilation complete. Executables are in ../bin/"