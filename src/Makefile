#compilers
CC=nvcc

#GLOBAL_PARAMETERS
MAT_VAL_TYPE ?= double
VALUE_TYPE ?= $(MAT_VAL_TYPE)

#CUDA_PARAMETERS
NVCC_ARCH ?= -arch=compute_80 -code=sm_80 -gencode=arch=compute_80,code=sm_80
NVCC_FLAGS ?= -O3 -w $(NVCC_ARCH) -std=c++17
#-gencode=arch=compute_61,code=sm_75
# -m64 -Xptxas -dlcm=cg -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61
#-Xcompiler -Wall -D_FORCE_INLINES -DVERBOSE --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr

#ENVIRONMENT_PARAMETERS
CUDA_INSTALL_PATH ?= /usr/local/cuda

#includes
INCLUDES ?= -I$(CUDA_INSTALL_PATH)/include

#libs
#CLANG_LIBS = -stdlib=libstdc++ -lstdc++
CUDA_LIBS ?= -L$(CUDA_INSTALL_PATH)/lib64  -lcudart  -lcusparse
LIBS ?= $(CUDA_LIBS)

#options
#OPTIONS = -std=c99
TYPE_OPTIONS := -D MAT_VAL_TYPE=$(MAT_VAL_TYPE) -D VALUE_TYPE=$(VALUE_TYPE)

make:
	$(CC) $(NVCC_FLAGS) -Xcompiler -fopenmp -Xcompiler -mfma main.cu -o test $(INCLUDES) $(LIBS) $(TYPE_OPTIONS) $(OPTIONS)
