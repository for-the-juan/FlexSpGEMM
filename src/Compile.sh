#!/bin/bash

set -e

#compilers
CC="nvcc"

#GLOBAL_PARAMETERS
MAT_VAL_TYPE="${MAT_VAL_TYPE:-double}"
VALUE_TYPE="${VALUE_TYPE:-${MAT_VAL_TYPE}}"

#CUDA_PARAMETERS
NVCC_FLAGS="-O3 -w -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80 -std=c++17"
#-gencode=arch=compute_61,code=sm_75
# -m64 -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61
#-Xcompiler -Wall -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH="/usr/local/cuda-11.8"

#includes
INCLUDES="-I${CUDA_INSTALL_PATH}/include"

#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
CUDA_LIBS="-L${CUDA_INSTALL_PATH}/lib64  -lcudart  -lcusparse"
LIBS=${CUDA_LIBS}

TILE_SIZE_M=(8 16 32)
TILE_SIZE_N=(8 16 32)

#options
#OPTIONS = -std=c99
TYPE_OPTIONS="-D MAT_VAL_TYPE=${MAT_VAL_TYPE} -D VALUE_TYPE=${VALUE_TYPE}"

mkdir -p ../bin
rm -rf ../bin/*
for i in ${TILE_SIZE_M[@]}; do
    for j in ${TILE_SIZE_N[@]}; do
        ${CC} ${NVCC_FLAGS} -Xcompiler -fopenmp -Xcompiler -mfma main.cu -o ../bin/test_m${i}_n${j} ${INCLUDES} ${LIBS} ${TYPE_OPTIONS} ${OPTIONS} -D TILE_SIZE_M=${i} -D TILE_SIZE_N=${j} &
    done
done

wait
