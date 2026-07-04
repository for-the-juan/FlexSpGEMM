CXX ?= g++
NVCC ?= nvcc

BIN_DIR := bin

CXXFLAGS ?= -O3 -std=c++17 -fopenmp -Wall -Wextra
CSR2TILE_CXXFLAGS ?= $(CXXFLAGS) -I. -mfma
NVCC_ARCH ?= -arch=sm_120
NVCCFLAGS ?= -O3 -std=c++17 $(NVCC_ARCH) -Xcompiler -fopenmp -Xcompiler -Wall -Xcompiler -Wextra
CUDA_LIBS ?= -lcusparse -lcudart

CPP_TARGETS := $(BIN_DIR)/cpu_probe_timer $(BIN_DIR)/cpu_csr2tile_timer $(BIN_DIR)/tilespgemm_csr2tile_timer
GPU_TARGETS := $(BIN_DIR)/gpu_probe_timer $(BIN_DIR)/gpu_csr2tile_timer $(BIN_DIR)/double_wmma_benchmark $(BIN_DIR)/wmma_type_benchmark

.PHONY: all clean

all: $(CPP_TARGETS) $(GPU_TARGETS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/cpu_probe_timer: cpu_probe_timer.cpp probe_common.hpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) cpu_probe_timer.cpp -o $@

$(BIN_DIR)/cpu_csr2tile_timer: cpu_csr2tile_timer.cpp probe_common.hpp cuda_fp16.h ../csr2tile.h ../common.h ../utils.h | $(BIN_DIR)
	$(CXX) $(CSR2TILE_CXXFLAGS) cpu_csr2tile_timer.cpp -o $@

$(BIN_DIR)/tilespgemm_csr2tile_timer: tilespgemm_csr2tile_timer.cpp probe_common.hpp cuda_fp16.h ../csr2tile.bak.h ../common.h ../utils.h | $(BIN_DIR)
	$(CXX) $(CSR2TILE_CXXFLAGS) tilespgemm_csr2tile_timer.cpp -o $@

$(BIN_DIR)/gpu_probe_timer: gpu_probe_timer.cu probe_common.hpp | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) gpu_probe_timer.cu -o $@ $(CUDA_LIBS)

$(BIN_DIR)/gpu_csr2tile_timer: gpu_csr2tile_timer.cu probe_common.hpp ../gpu_csr2tile.h ../csr2tile.h ../common.h ../utils.h | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) gpu_csr2tile_timer.cu -o $@ $(CUDA_LIBS)

$(BIN_DIR)/double_wmma_benchmark: double_wmma_benchmark.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) double_wmma_benchmark.cu -o $@ -lcudart

$(BIN_DIR)/wmma_type_benchmark: wmma_type_benchmark.cu | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) wmma_type_benchmark.cu -o $@ -lcudart

clean:
	rm -rf $(BIN_DIR)
