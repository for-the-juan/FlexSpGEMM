// #pragma once

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>  // for uint8_t, uint16_t, uint32_t, uint64_t
#include <math.h>
#include <mm_malloc.h>
#include <x86intrin.h>
#include <immintrin.h>
#include <nmmintrin.h>


#include <omp.h>

#include <sys/time.h>
#include "cuda_fp16.h"

#include "utils.h"

#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE double
#endif

#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE int
#endif

// e.g., nvcc -DTILE_SIZE_M=64 ...
#ifndef TILE_SIZE_M
#define TILE_SIZE_M 32
#endif

#ifndef TILE_SIZE_N
#define TILE_SIZE_N 32
#endif

// WMMA fragment dimensions for Double Precision (FP64)
#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 4

#define QUADWARP_SIZE 8
#define HALFWARP_SIZE 16
#define WARP_SIZE 32

#define QUADWARP_PER_BLOCK 16
#define HALFWARP_PER_BLOCK 8
#define WARP_PER_BLOCK 4

#define USE_HALFWARP 1
#define USE_TENSORCORE 1
#define USE_COOPERATION 1

#define TILE_PER_WARP 16
#define TILE_PER_HALFWARP 8
#define TILE_PER_QUADWARP 4
#define TILE_PER_ADAPTIVE_WARP 8

#define VECTORIZE_NNZA_OR_NNZB_TH 8

#define SMEM_INTERSECTION_TH 16
#define SMEM_INTERSECTION_LEN 48

#define USE_GMEM_SPECULATIVE_INTERSECTION 0
#define GMEM_SPECULATIVE_INTERSECTION 1

#define SPECULATIVE_INTERSECTION 32

#define SPA_INT_PER_WARP 512
#define NUMCOLC_SPA_OR_HASH_TH     SPA_INT_PER_WARP * 32 // SPA_INT_PER_WARP int per warp

#define USE_DENSE 0

// e.g., INTERSECTION_SPARSE_OR_DNS_TH = 0.2 means when density is higher than 20%, use DNS for intersection
#define INTERSECTION_SPARSE_OR_DNS_TH 0.2
#define NNZTOTALA_FAST_TRACK_TH2 192

#define USE_DNS_THREAD 0

#define DEBUG 1

#ifndef DEBUG_PRINT_ENABLE
#define DEBUG_PRINT_ENABLE 0  // Temporarily enable for debugging
#endif

// CPU DEBUG
#if DEBUG_PRINT_ENABLE
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...) ((void)0)
#endif

// CUDA kernel DEBUG
#if DEBUG_PRINT_ENABLE
#define DEBUG_PRINT_CUDA(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT_CUDA(...) ((void)0)
#endif

#define REPEAT_NUM 1

#ifndef TIMING
#define TIMING 1
#endif

#ifndef SPACE
#define SPACE 1
#endif

#ifndef CHECK_RESULT
#define CHECK_RESULT 1
#endif

#define HASH_SCALE 107

#define SMEM_TNY_TH (TILE_SIZE_M * TILE_SIZE_M / 8)
#define SMEM_SML_TH (TILE_SIZE_M * TILE_SIZE_M / 8)

#ifndef SMEM_LRG_TH
#define SMEM_LRG_TH (TILE_SIZE_M * TILE_SIZE_M * 8 / 8)
#endif

#define SMEM_DNS_TH (TILE_SIZE_M * TILE_SIZE_M)

#define THREADS_USED_TNY_TH 16
#define THREADS_USED_SML_TH 16
#define THREADS_USED_LRG_TH 32
#define THREADS_USED_DNS_TH 32

#define STEP3_THREADS 128
#define STEP4_THREADS 128
#if TILE_SIZE_M == 32 && TILE_SIZE_N == 32
    #define STEP4_TC_THREADS 64
#else
    #define STEP4_TC_THREADS 128
#endif

#if TILE_SIZE_M * TILE_SIZE_N <= 256 && TILE_SIZE_M * TILE_SIZE_M <= 256
    typedef uint8_t TILE_CSR_PTR_TYPE;
#elif TILE_SIZE_M * TILE_SIZE_N <= 65536 && TILE_SIZE_M * TILE_SIZE_M <= 65536
    typedef uint16_t TILE_CSR_PTR_TYPE;
#else
    typedef uint32_t TILE_CSR_PTR_TYPE;
#endif

#if TILE_SIZE_M * TILE_SIZE_N <= 256
    typedef uint8_t TILE_CSR_COL_TYPE_A;
#elif TILE_SIZE_M * TILE_SIZE_N <= 65536
    typedef uint16_t TILE_CSR_COL_TYPE_A;
#else
    typedef uint32_t TILE_CSR_COL_TYPE_A;
#endif

#if TILE_SIZE_M <= 256
    typedef uint8_t TILE_CSR_COL_TYPE_B;
#else
    typedef uint16_t TILE_CSR_COL_TYPE_B;
#endif

typedef uint32_t INTERSEC_BITMASK_TYPE;

#if TILE_SIZE_N == 8
    typedef uint8_t TILE_MASK_TYPE_A;
#elif TILE_SIZE_N == 16
    typedef uint16_t TILE_MASK_TYPE_A;
#else  // TILE_SIZE_M >= 32
    typedef uint32_t TILE_MASK_TYPE_A;
#endif

// Matrix B and Matrix C share the same TILE_MASK_TYPE
#if TILE_SIZE_M == 8
    typedef uint8_t TILE_MASK_TYPE_B;
#elif TILE_SIZE_M == 16
    typedef uint16_t TILE_MASK_TYPE_B;
#else  // TILE_SIZE_M >= 32
    typedef uint32_t TILE_MASK_TYPE_B;
#endif

#define MaskBitsA (sizeof(TILE_MASK_TYPE_A) * 8)
#define MaskBitsB (sizeof(TILE_MASK_TYPE_B) * 8)
#define MaskBitsC (sizeof(TILE_MASK_TYPE_B) * 8)

#define MaskNumA (TILE_SIZE_N / MaskBitsA)
#define MaskNumB (TILE_SIZE_M / MaskBitsB)
#define MaskNumC (TILE_SIZE_M / MaskBitsC) // tile_size_k

#ifndef TILE_DENSE_THRESHOLD
#define TILE_DENSE_THRESHOLD 7
#endif

#ifndef ENABLE_MULTI_WARP_SHARED_SLOT
#define ENABLE_MULTI_WARP_SHARED_SLOT 1
#endif

#ifndef NUM_SHARED_SLOTS
    #if TILE_SIZE_M < 32 && TILE_SIZE_N < 32
        #define NUM_SHARED_SLOTS 2
    #else
        #define NUM_SHARED_SLOTS 1
    #endif
#endif

#ifndef SLOT_SPIN_LIMIT
#define SLOT_SPIN_LIMIT 0
#endif

#ifndef ENABLE_SLOT_TILE_CACHE
#define ENABLE_SLOT_TILE_CACHE 1
#endif

#ifndef SUBTILE_K_STEP
#define SUBTILE_K_STEP TILE_SIZE_N  // Default: process entire K dimension at once
#endif

#ifndef MAX_SMEM_PER_BLOCK
#define MAX_SMEM_PER_BLOCK (48 * 1024)
#endif

#ifndef SMATRIX
#define SMATRIX
typedef struct
{
    int m;
    int n;
    int nnz;
    int isSymmetric;
    MAT_VAL_TYPE *value;
    int *columnindex;
    MAT_PTR_TYPE *rowpointer;
    int tilem;
    int tilen;
    MAT_PTR_TYPE *tile_ptr;
    int *tile_columnidx;
    int *tile_rowidx;
    int *tile_nnz;
    int numtile;
    MAT_VAL_TYPE *tile_csr_Value;
    TILE_CSR_COL_TYPE_A *tile_csr_Col;
    TILE_CSR_PTR_TYPE *tile_csr_Ptr;
    TILE_MASK_TYPE_A *mask;
    int *csc_tile_ptr;
    int *csc_tile_rowidx;
    MAT_VAL_TYPE *dense_data;
    int *tile_dense_ready;   // Use dense or not
    int dense_tile_count;        // dense tile number
}SMatrixA;

// Matrix C also uses SMatrixB, since they share the same TILE_CSR_COL_TYPE and TILE_MASK_TYPE
typedef struct
{
    int m;
    int n;
    int nnz;
    int isSymmetric;
    MAT_VAL_TYPE *value;
    int *columnindex;
    MAT_PTR_TYPE *rowpointer;
    int tilem;
    int tilen;
    MAT_PTR_TYPE *tile_ptr;
    int *tile_columnidx;
    int *tile_rowidx;
    int *tile_nnz;
    int numtile;  //非零tile数
    MAT_VAL_TYPE *tile_csr_Value;
    TILE_CSR_COL_TYPE_B *tile_csr_Col;
    TILE_CSR_PTR_TYPE *tile_csr_Ptr;
    TILE_MASK_TYPE_B *mask;
    int *csc_tile_ptr;
    int *csc_tile_rowidx;
    MAT_VAL_TYPE *dense_data;
    int *tile_dense_ready;   // Use dense or not
    int dense_tile_count;        // dense tile number
}SMatrixB;
#endif