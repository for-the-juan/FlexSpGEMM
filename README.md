# FlexSpGEMM
 

**FlexSpGEMM** is an open-source project that optimizes general sparse matrix-matrix multiplication (SpGEMM) on GPUs. It leverages machine learning methods to predict and adaptively select flexible tile sizes, enabling efficient computation across diverse sparse matrix workloads.


## Introduction

General sparse matrix-matrix multiplication(SpGEMM) executes C=AB, where A, B and C are all sparse matrices. FlexSpGEMM is a Flexible machine learning (ML) guided SpGEMM framework for modern GPUs. In the offline phase, we propose a ML-based predictor to identify both the tiling parameter and the associated hardware path based on the input matrices. In the online phase, a hybrid execution mechanism coordinates the cooperative use of CUDA cores and Tensor cores at tile granularity. To further reduce the format translation overhead on the Tensor core path, FlexSpGEMM introduces a speculative dense tile format that pre-constructs dense replicas only for a subset of input tiles with high potential benefit, and combines this design with lightweight runtime densification to enable efficient access. 

## Installation

<!-- To use this code, you need to modify the Makefile with correct g++ installation path and use make for automatic installation. -->
FlexSpGEMM evaluation requires the CUDA GPU driver, the nvcc CUDA compiler, and the cuSPARSE library, all of them are included with the CUDA v11.8. The artifacts have been tested on Ubuntu 18.04/22.04, and are expected to run correctly under other Linux distributions. In addition, the Python requirements is shown in environment.yml.

## Quick Start

### Step1: Complete the artifact setup
For the Python environment setup, you should run the command below (About 15min):
```bash
conda env create -f environment.yml
conda activate FlexSpGEMM
```
We need FlexSpGEMM environment for downloading.

Then, you can download the train, val and test matrices by the command below:
```bash
cd data
python download_test_matrices.py
python transpose_mtx.py
```
We only need test matrices for quick evaluation and reproduction.

### Step2: Predict tile shape and tau with LightGBM, LLM and SVM
For this step, we also need FlexSpGEMM environment.
Firstly, please get the data features by the command below:
```bash
cd data_prepare/data_get_sh
python run_pipeline.py
```
Then, you can run the command below to get the accuracy of LightGBM, LLM and SVM:
To quickly reproduce the results in our paper, we load checkpoints from previous trainning.
```bash
cd ../../..
cd ML_method
```

#### LightGBM
```bash
cd LightGBM
```

#### LLM
```bash
cd LLM
```

#### SVM
```bash
cd SVM
```

Finally, you should run the Python scripts below to collect all the metrics
```bash
cd 

```
### Step3: Run FlexSpGEMM and other four SpGEMM methods
For this step, we would compile and run the code to collect the results.
#### FlexSpGEMM

#### HSMU-SpGEMM

#### TileSpGEMM and cuSPARSE


### Step4: 

## Full Evaluation from Scratch
Data Download (3h, 19GB)
If you prefer to reproduce the complete evaluation pipeline from scratch, we have prepared the LightGBM dataset split in advance. The split configuration is located at: 

## Output information

The \[**Device**\] part prints the GPU device information, including the device ID, device name, and clock rate (in MHz).

The \[**Input Matrix**\] part prints the input matrix's information, including the path of the matrix file, the dimension (number of rows and columns), the number of nonzeros, and the matrix loading time (in seconds).

The [**Tiling Configuration**] part prints the tile size (TILE_SIZE_M × TILE_SIZE_N) used in our FlexSpGEMM algorithm.

The \[**Preprocessing**\] part prints the NNZ upper bound of the output matrix C (nnzCub), the runtime of transforming the input matrix from the CSR format to our tiled data structure (in milliseconds), and the memory consumption (in megabytes) of different data structures, including the CSR format, the dense format, the TileSpGEMM tiled format, and the FlexSpGEMM tiled format.

The [**Symbolic Stage**] part prints the runtime of the symbolic stage (in milliseconds), which determines the sparsity structure of the output matrix C, as well as the classification statistics of output tiles, including the tile type (Tiny, Small, Large, Dense, Full), the number of tiles in each category, the corresponding NNZ threshold, and the number of threads assigned to each category.

The [**Numeric Stage**] part prints the runtime of the numeric stage (in milliseconds), which computes the actual numerical values of the output matrix C.

The \[**Malloc**\] part prints the time spent on memory allocation (in milliseconds) during the computation.

The [**FlexSpGEMM Summary**] part prints the number of non-empty output tiles, the number of nonzeros of the resulting matrix C, the total FlexSpGEMM runtime (in milliseconds), and the throughput (in GFlops).

The [**Baseline: cuSPARSE SpGEMM**] part prints the cuSPARSE baseline results, including the total runtime (in milliseconds), throughput (in GFlops), the number of nonzeros of the output matrix C, the NNZ upper bound, and the compression rate.

The [**Correctness Validation**] part prints the correctness validation results by comparing our output with the reference solution generated by cuSPARSE, checking the NNZ count, the row pointer array, and the column index and value arrays.

The \[**Speedup**\] part prints the speedup of FlexSpGEMM over the cuSPARSE baseline, computed as the ratio of their total runtimes.

## Release version
Apr 6,2026 Version Alpha