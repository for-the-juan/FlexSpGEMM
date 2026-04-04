#/bin/bash

#compilers
CC="nvcc"

#GLOBAL_PARAMETERS
MAT_VAL_TYPE="double"
VALUE_TYPE="double"

#CUDA_PARAMETERS
NVCC_FLAGS="-O3 -w -arch=compute_90 -code=sm_90 -gencode=arch=compute_90,code=sm_90 -std=c++17"
#-gencode=arch=compute_61,code=sm_75
# -m64 -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61
#-Xcompiler -Wall -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH="/data/apps/cuda/12.4/bin"

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

mkdir -p ../bin
rm -rf ../bin/*
for i in ${TILE_SIZE_M[@]}; do
    for j in ${TILE_SIZE_N[@]}; do
        ${CC} ${NVCC_FLAGS} -Xcompiler -fopenmp -Xcompiler -mfma main.cu -o ../bin/test_m${i}_n${j} ${INCLUDES} ${LIBS} ${OPTIONS} -D VALUE_TYPE=${VALUE_TYPE} -D TILE_SIZE_M=${i} -D TILE_SIZE_N=${j} &
    done
done