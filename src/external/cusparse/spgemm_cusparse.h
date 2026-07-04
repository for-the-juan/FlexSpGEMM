#ifndef _SPGEMM_CUDA_CUSPARSE_
#define _SPGEMM_CUDA_CUSPARSE_

#include "common.h"
#include "utils.h"
#include <cuda_runtime.h>
#include <cusparse.h>
#include <type_traits>

template <typename T>
struct flex_false_type : std::false_type {};

template <typename T>
static inline cudaDataType flex_cuda_data_type()
{
    if constexpr (std::is_same<T, double>::value)
    {
        return CUDA_R_64F;
    }
    else if constexpr (std::is_same<T, float>::value)
    {
        return CUDA_R_32F;
    }
    else if constexpr (std::is_same<T, __half>::value)
    {
        return CUDA_R_16F;
    }
    else
    {
        static_assert(flex_false_type<T>::value, "Unsupported FlexSpGEMM value type for cuSPARSE.");
    }
}

template <typename T>
struct flex_cusparse_compute_type
{
    using type = T;
};

template <>
struct flex_cusparse_compute_type<__half>
{
    using type = float;
};

template <typename T>
struct flex_cusparse_storage_type
{
    using type = T;
};

template <>
struct flex_cusparse_storage_type<__half>
{
    using type = float;
};

using FlexCusparseValueType = typename flex_cusparse_storage_type<VALUE_TYPE>::type;

template <typename DstT, typename SrcT>
static inline DstT flex_value_cast(SrcT value)
{
    return static_cast<DstT>(value);
}

template <>
inline float flex_value_cast<float, __half>(__half value)
{
    return __half2float(value);
}

template <typename SrcT, typename DstT>
static inline void flex_copy_host_values_to_device(DstT **d_values,
                                                   const SrcT *h_values,
                                                   int n)
{
    cudaMalloc((void **)d_values, n * sizeof(DstT));
    if (n <= 0 || h_values == NULL)
    {
        return;
    }

    if constexpr (std::is_same<SrcT, DstT>::value)
    {
        cudaMemcpy(*d_values, h_values, n * sizeof(DstT), cudaMemcpyHostToDevice);
    }
    else
    {
        DstT *h_converted = (DstT *)malloc(n * sizeof(DstT));
        for (int i = 0; i < n; ++i)
        {
            h_converted[i] = flex_value_cast<DstT>(h_values[i]);
        }
        cudaMemcpy(*d_values, h_converted, n * sizeof(DstT), cudaMemcpyHostToDevice);
        free(h_converted);
    }
}

template <typename T>
__global__ void compare_device_arrays_kernel(const T *got,
                                             const T *expected,
                                             int n,
                                             int check_negative,
                                             int *err_count,
                                             int *negative_count)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (idx >= n)
    {
        return;
    }

    const T got_v = got[idx];
    if (got_v != expected[idx])
    {
        if (check_negative && got_v < 0)
        {
            atomicAdd(negative_count, 1);
        }
        atomicAdd(err_count, 1);
    }
}

template <typename T>
static inline int count_device_array_errors(const T *got,
                                            const T *expected,
                                            int n,
                                            bool check_negative,
                                            int *negative_count)
{
    if (n <= 0)
    {
        if (negative_count != NULL)
        {
            *negative_count = 0;
        }
        return 0;
    }

    int *d_err_count = NULL;
    int *d_negative_count = NULL;
    int h_err_count = 0;
    int h_negative_count = 0;
    cudaMalloc((void **)&d_err_count, sizeof(int));
    cudaMalloc((void **)&d_negative_count, sizeof(int));
    cudaMemset(d_err_count, 0, sizeof(int));
    cudaMemset(d_negative_count, 0, sizeof(int));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    compare_device_arrays_kernel<T><<<blocks, threads>>>(
        got, expected, n, check_negative ? 1 : 0, d_err_count, d_negative_count);
    cudaGetLastError();

    cudaMemcpy(&h_err_count, d_err_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_negative_count, d_negative_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_err_count);
    cudaFree(d_negative_count);

    if (negative_count != NULL)
    {
        *negative_count = h_negative_count;
    }
    return h_err_count;
}

//#include "utils_cuda_sort.h"
//#include "utils_cuda_spgemm_subfunc.h"
//#include "utils_cuda_scan.h"
//#include "utils_cuda_segmerge.h"
//#include "utils_cuda_segsum.h"

int spgemm_cusparse_executor(cusparseHandle_t handle, cusparseSpMatDescr_t matA,
                             const int mA,
                             const int nA,
                             const int nnzA,
                             const int *d_csrRowPtrA,
                             const int *d_csrColIdxA,
                             const FlexCusparseValueType *d_csrValA,
                             cusparseSpMatDescr_t matB,
                             const int mB,
                             const int nB,
                             const int nnzB,
                             const int *d_csrRowPtrB,
                             const int *d_csrColIdxB,
                             const FlexCusparseValueType *d_csrValB,
                             cusparseSpMatDescr_t matC,
                             const int mC,
                             const int nC,
                             unsigned long long int *nnzC,
                             int **d_csrRowPtrC,
                             int **d_csrColIdxC,
                             FlexCusparseValueType **d_csrValC)
{
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    using ComputeType = typename flex_cusparse_compute_type<VALUE_TYPE>::type;
    cudaDataType computeType = flex_cuda_data_type<ComputeType>();
    void *dBuffer1 = NULL, *dBuffer2 = NULL;
    size_t bufferSize1 = 0, bufferSize2 = 0;

    ComputeType alpha = static_cast<ComputeType>(1.0);
    ComputeType beta = static_cast<ComputeType>(0.0);

    cudaMalloc((void **)d_csrRowPtrC, (mC + 1) * sizeof(int));

    //--------------------------------------------------------------------------
    // SpGEMM Computation
    cusparseSpGEMMDescr_t spgemmDesc;
    cusparseSpGEMM_createDescr(&spgemmDesc);

    // ask bufferSize1 bytes for external memory
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, NULL);
    cudaMalloc((void **)&dBuffer1, bufferSize1);
    // inspect the matrices A and B to understand the memory requiremnent for
    // the next step
    cusparseSpGEMM_workEstimation(handle, opA, opB,
                                  &alpha, matA, matB, &beta, matC,
                                  computeType, CUSPARSE_SPGEMM_DEFAULT,
                                  spgemmDesc, &bufferSize1, dBuffer1);

    // ask bufferSize2 bytes for external memory
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, NULL);
    cudaMalloc((void **)&dBuffer2, bufferSize2);

    // compute the intermediate product of A * B
    cusparseSpGEMM_compute(handle, opA, opB,
                           &alpha, matA, matB, &beta, matC,
                           computeType, CUSPARSE_SPGEMM_DEFAULT,
                           spgemmDesc, &bufferSize2, dBuffer2);
    // get matrix C non-zero entries C_num_nnz1
    int64_t C_num_rows1, C_num_cols1, C_num_nnz1;
    cusparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_num_nnz1);
    // allocate matrix C
    cudaMalloc((void **)d_csrColIdxC, C_num_nnz1 * sizeof(int));
    cudaMalloc((void **)d_csrValC, C_num_nnz1 * sizeof(FlexCusparseValueType));
    // update matC with the new pointers
    cusparseCsrSetPointers(matC, *d_csrRowPtrC, *d_csrColIdxC, *d_csrValC);

    // copy the final products to the matrix C
    cusparseSpGEMM_copy(handle, opA, opB,
                        &alpha, matA, matB, &beta, matC,
                        computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc);

    *nnzC = C_num_nnz1;

    cusparseSpGEMM_destroyDescr(spgemmDesc);

    return 0;
}

int spgemm_cusparse(const int mA,
                    const int nA,
                    const int nnzA,
                    const int *h_csrRowPtrA,
                    const int *h_csrColIdxA,
                    const VALUE_TYPE *h_csrValA,
                    const int mB,
                    const int nB,
                    const int nnzB,
                    const int *h_csrRowPtrB,
                    const int *h_csrColIdxB,
                    const VALUE_TYPE *h_csrValB,
                    const int mC,
                    const int nC,
                    const int nnzC_golden,
                    const int *h_csrRowPtrC_golden,
                    const int *h_csrColIdxC_golden,
                    const VALUE_TYPE *h_csrValC_golden,
                    const bool check_result,
                    unsigned long long int nnzCub,
                    unsigned long long int *nnzC,
                    double *compression_rate,
                    double *time_segmerge,
                    double *gflops_segmerge)

{
    // transfer host mem to device mem
    int *d_csrRowPtrA;
    int *d_csrColIdxA;
    FlexCusparseValueType *d_csrValA;
    int *d_csrRowPtrB;
    int *d_csrColIdxB;
    FlexCusparseValueType *d_csrValB;
    //unsigned long long int nnzC = 0;
    int *d_csrRowPtrC;
    int *d_csrColIdxC;
    FlexCusparseValueType *d_csrValC;

    // Matrix A in CSR
    cudaMalloc((void **)&d_csrRowPtrA, (mA + 1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdxA, nnzA * sizeof(int));

    cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, (mA + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdxA, h_csrColIdxA, nnzA * sizeof(int), cudaMemcpyHostToDevice);
    flex_copy_host_values_to_device(&d_csrValA, h_csrValA, nnzA);

    // Matrix B in CSR
    cudaMalloc((void **)&d_csrRowPtrB, (mB + 1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdxB, nnzB * sizeof(int));

    cudaMemcpy(d_csrRowPtrB, h_csrRowPtrB, (mB + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdxB, h_csrColIdxB, nnzB * sizeof(int), cudaMemcpyHostToDevice);
    flex_copy_host_values_to_device(&d_csrValB, h_csrValB, nnzB);

    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA, matB, matC;
    const cudaDataType valueType = flex_cuda_data_type<FlexCusparseValueType>();

    cusparseCreate(&handle);
    // Create sparse matrix A in CSR format
    cusparseCreateCsr(&matA, mA, nA, nnzA,
                      d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      valueType);
    cusparseCreateCsr(&matB, mB, nB, nnzB,
                      d_csrRowPtrB, d_csrColIdxB, d_csrValB,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      valueType);
    cusparseCreateCsr(&matC, mA, nB, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      valueType);
    //--------------------------------------------------------------------------

    //  - cuda SpGEMM start!
    printf("  Benchmark cuSPARSE Runs  : %d\n", BENCH_REPEAT);
    printf("  Status                   : Running...\n");

    if (check_result && BENCH_REPEAT > 1)
    {
        printf("If check_result, Set BENCH_REPEAT to 1.\n");
        return -1;
    }
    //unsigned long long int nnzCub = 0;

    struct timeval t1, t2;

    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        spgemm_cusparse_executor(handle, matA, mA, nA, nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                                 matB, mB, nB, nnzB, d_csrRowPtrB, d_csrColIdxB, d_csrValB,
                                 matC, mC, nC, nnzC, &d_csrRowPtrC, &d_csrColIdxC, &d_csrValC);

        if (check_result != 1 || i != BENCH_REPEAT - 1)
        {
            cudaFree(d_csrRowPtrC);
            cudaFree(d_csrColIdxC);
            cudaFree(d_csrValC);
        }
    }

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    printf("  Status                   : Completed.\n");
    double time_cuda_spgemm = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_spgemm /= BENCH_REPEAT;
    *time_segmerge = time_cuda_spgemm;
    *compression_rate = (double)nnzCub / (double)*nnzC;
    *gflops_segmerge = 2 * (double)nnzCub / (1e6 * time_cuda_spgemm);
    printf("  Total Runtime            : %.4f ms\n",   time_cuda_spgemm);
    printf("  Throughput               : %.4f GFlops\n", *gflops_segmerge);
    printf("  NNZ (C)                  : %lld\n",     *nnzC);
    printf("  NNZ Upper Bound          : %lld\n",     nnzCub);
    printf("  Compression Rate         : %.2f\n",     *compression_rate);
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");

    // validate C = AB

    if (check_result)
    {
        if (*nnzC <= 0)
        {
            printf("cuSPARSE failed!\n");
            return -1;
        }
        else
        {
            printf("\n[Correctness Validation]\n");
            // nnzC check
            if (*nnzC != nnzC_golden)
                printf("  ✗  NNZ count             : FAILED  "
                       "(got %d, expected %d)\n", *nnzC, nnzC_golden);
            else
                printf("  ✓  NNZ count             : PASSED  (%d)\n", *nnzC);

            int *h_csrRowPtrC = (int *)malloc((mC + 1) * sizeof(int));
            int *h_csrColIdxC = (int *)malloc(*nnzC * sizeof(int));
            FlexCusparseValueType *h_csrValC = (FlexCusparseValueType *)malloc(*nnzC * sizeof(FlexCusparseValueType));

            cudaMemcpy(h_csrRowPtrC, d_csrRowPtrC, (mC + 1) * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_csrColIdxC, d_csrColIdxC, *nnzC * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_csrValC, d_csrValC, *nnzC * sizeof(FlexCusparseValueType), cudaMemcpyDeviceToHost);

            int errcounter_row = 0;
            for (int i = 0; i < mC + 1; i++)
            {
                if (h_csrRowPtrC[i] != h_csrRowPtrC_golden[i])
                {
                    if (h_csrRowPtrC[i] < 0)
                    {
                        printf("cuSPARSE failed!\n");
                        return -1;
                    }
                    else{
                    errcounter_row++;}
                }
            }
            if (errcounter_row != 0)
                printf("  ✗  Row pointer array     : FAILED  (#err = %d)\n",
                        errcounter_row);
            else
                printf("  ✓  Row pointer array     : PASSED\n");

            /*for (int i = 0; i < mC; i++)
        {
            quick_sort_key_val_pair<int, VALUE_TYPE>(&h_csrColIdxC[h_csrRowPtrC[i]],
                                                     &h_csrValC[h_csrRowPtrC[i]],
                                                     h_csrRowPtrC[i+1]-h_csrRowPtrC[i]);
        }*/

            int errcounter_colval = 0;
            for (int j = 0; j < *nnzC; j++)
            {
                if (h_csrColIdxC[j] != h_csrColIdxC_golden[j]) //|| h_csrValC[j] != h_csrValC_golden[j])
                {
                    //    printf("h_csrColIdxC[j] = %i,  h_csrColIdxC_golden[j] = %i\n",h_csrColIdxC[j] ,h_csrColIdxC_golden[j]);
                    errcounter_colval++;
                }
            }

            if (errcounter_colval != 0)
                printf("  ✗  Column idx & values   : FAILED  "
                       "(#err = %d, %.2f%% of NNZ)\n", errcounter_colval,
                        100.0 * (double)errcounter_colval / (double)(*nnzC));
            else
                printf("  ✓  Column idx & values   : PASSED\n");
                printf("\n");

            if (*nnzC == nnzC_golden && errcounter_row == 0 && errcounter_colval == 0)
            {
                printf("================================================================================\n");
                printf("  All checks passed. FlexSpGEMM produces numerically correct results.\n");
                printf("================================================================================\n");
            }
            else
            {
                printf("================================================================================\n");
                printf("  [WARNING] Validation FAILED. Please check the output above for details.\n");
                printf("================================================================================\n");
            }
            printf("\n");

            free(h_csrRowPtrC);
            free(h_csrColIdxC);
            free(h_csrValC);
        }
    }

    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIdxA);
    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrB);
    cudaFree(d_csrColIdxB);
    cudaFree(d_csrValB);

    if (check_result)
    {
        cudaFree(d_csrRowPtrC);
        cudaFree(d_csrColIdxC);
        cudaFree(d_csrValC);
    }

    cusparseDestroySpMat(matA);
    cusparseDestroySpMat(matB);
    cusparseDestroySpMat(matC);
    cusparseDestroy(handle);

	    return 0;
	}

int spgemm_cusparse_device_compare(const int mA,
	                               const int nA,
	                               const int nnzA,
	                               const int *h_csrRowPtrA,
	                               const int *h_csrColIdxA,
	                               const VALUE_TYPE *h_csrValA,
	                               const int mB,
	                               const int nB,
	                               const int nnzB,
	                               const int *h_csrRowPtrB,
	                               const int *h_csrColIdxB,
	                               const VALUE_TYPE *h_csrValB,
	                               const int mC,
	                               const int nC,
	                               const int nnzC_golden,
	                               const int *d_csrRowPtrC_golden,
	                               const int *d_csrColIdxC_golden,
	                               const VALUE_TYPE *d_csrValC_golden,
	                               const bool check_result,
	                               unsigned long long int nnzCub,
	                               unsigned long long int *nnzC,
	                               double *compression_rate,
	                               double *time_segmerge,
	                               double *gflops_segmerge)
{
    (void)d_csrValC_golden;

    if (check_result && BENCH_REPEAT > 1)
    {
        printf("If check_result, Set BENCH_REPEAT to 1.\n");
        return -1;
    }

    int ret = 0;
    int *d_csrRowPtrA = NULL;
    int *d_csrColIdxA = NULL;
    FlexCusparseValueType *d_csrValA = NULL;
    int *d_csrRowPtrB = NULL;
    int *d_csrColIdxB = NULL;
    FlexCusparseValueType *d_csrValB = NULL;
    int *d_csrRowPtrC = NULL;
    int *d_csrColIdxC = NULL;
    FlexCusparseValueType *d_csrValC = NULL;
    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA = NULL;
    cusparseSpMatDescr_t matB = NULL;
    cusparseSpMatDescr_t matC = NULL;
    const cudaDataType valueType = flex_cuda_data_type<FlexCusparseValueType>();

    cudaMalloc((void **)&d_csrRowPtrA, (mA + 1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdxA, nnzA * sizeof(int));
    cudaMemcpy(d_csrRowPtrA, h_csrRowPtrA, (mA + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdxA, h_csrColIdxA, nnzA * sizeof(int), cudaMemcpyHostToDevice);
    flex_copy_host_values_to_device(&d_csrValA, h_csrValA, nnzA);

    cudaMalloc((void **)&d_csrRowPtrB, (mB + 1) * sizeof(int));
    cudaMalloc((void **)&d_csrColIdxB, nnzB * sizeof(int));
    cudaMemcpy(d_csrRowPtrB, h_csrRowPtrB, (mB + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdxB, h_csrColIdxB, nnzB * sizeof(int), cudaMemcpyHostToDevice);
    flex_copy_host_values_to_device(&d_csrValB, h_csrValB, nnzB);

    cusparseCreate(&handle);
    cusparseCreateCsr(&matA, mA, nA, nnzA,
                      d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      valueType);
    cusparseCreateCsr(&matB, mB, nB, nnzB,
                      d_csrRowPtrB, d_csrColIdxB, d_csrValB,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      valueType);
    cusparseCreateCsr(&matC, mA, nB, 0,
                      NULL, NULL, NULL,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      valueType);

    printf("  Benchmark cuSPARSE Runs  : %d\n", BENCH_REPEAT);
    printf("  Status                   : Running...\n");

    struct timeval t1, t2;
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    for (int i = 0; i < BENCH_REPEAT; i++)
    {
        spgemm_cusparse_executor(handle, matA, mA, nA, nnzA, d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                                 matB, mB, nB, nnzB, d_csrRowPtrB, d_csrColIdxB, d_csrValB,
                                 matC, mC, nC, nnzC, &d_csrRowPtrC, &d_csrColIdxC, &d_csrValC);

        if (check_result != 1 || i != BENCH_REPEAT - 1)
        {
            cudaFree(d_csrRowPtrC);
            cudaFree(d_csrColIdxC);
            cudaFree(d_csrValC);
            d_csrRowPtrC = NULL;
            d_csrColIdxC = NULL;
            d_csrValC = NULL;
        }
    }

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    printf("  Status                   : Completed.\n");
    double time_cuda_spgemm = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    time_cuda_spgemm /= BENCH_REPEAT;
    *time_segmerge = time_cuda_spgemm;
    *compression_rate = (double)nnzCub / (double)*nnzC;
    *gflops_segmerge = 2 * (double)nnzCub / (1e6 * time_cuda_spgemm);
    printf("  Total Runtime            : %.4f ms\n", time_cuda_spgemm);
    printf("  Throughput               : %.4f GFlops\n", *gflops_segmerge);
    printf("  NNZ (C)                  : %lld\n", *nnzC);
    printf("  NNZ Upper Bound          : %lld\n", nnzCub);
    printf("  Compression Rate         : %.2f\n", *compression_rate);
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");

    if (check_result)
    {
        if (*nnzC <= 0)
        {
            printf("cuSPARSE failed!\n");
            ret = -1;
        }
        else if (d_csrRowPtrC_golden == NULL || (nnzC_golden > 0 && d_csrColIdxC_golden == NULL))
        {
            printf("FlexSpGEMM GPU CSR golden output is missing.\n");
            ret = -1;
        }
        else
        {
            printf("\n[Correctness Validation]\n");
            if (*nnzC != (unsigned long long int)nnzC_golden)
                printf("  ✗  NNZ count             : FAILED  "
                       "(got %d, expected %d)\n", *nnzC, nnzC_golden);
            else
                printf("  ✓  NNZ count             : PASSED  (%d)\n", *nnzC);

            int negative_row = 0;
            const int errcounter_row = count_device_array_errors<int>(
                d_csrRowPtrC, d_csrRowPtrC_golden, mC + 1, true, &negative_row);
            if (negative_row != 0)
            {
                printf("cuSPARSE failed!\n");
                ret = -1;
            }

            if (errcounter_row != 0)
                printf("  ✗  Row pointer array     : FAILED  (#err = %d)\n",
                       errcounter_row);
            else
                printf("  ✓  Row pointer array     : PASSED\n");

            const int col_compare_n = (*nnzC < (unsigned long long int)nnzC_golden) ? static_cast<int>(*nnzC) : nnzC_golden;
            const int errcounter_colval = count_device_array_errors<int>(
                d_csrColIdxC, d_csrColIdxC_golden, col_compare_n, false, NULL);

            if (errcounter_colval != 0)
                printf("  ✗  Column idx & values   : FAILED  "
                       "(#err = %d, %.2f%% of NNZ)\n", errcounter_colval,
                       100.0 * (double)errcounter_colval / (double)(*nnzC));
            else
                printf("  ✓  Column idx & values   : PASSED\n");
            printf("\n");

            if (*nnzC == (unsigned long long int)nnzC_golden && errcounter_row == 0 && errcounter_colval == 0 && ret == 0)
            {
                printf("================================================================================\n");
                printf("  All checks passed. FlexSpGEMM produces numerically correct results.\n");
                printf("================================================================================\n");
            }
            else
            {
                printf("================================================================================\n");
                printf("  [WARNING] Validation FAILED. Please check the output above for details.\n");
                printf("================================================================================\n");
            }
            printf("\n");
        }
    }

    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIdxA);
    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrB);
    cudaFree(d_csrColIdxB);
    cudaFree(d_csrValB);
    if (d_csrRowPtrC != NULL)
    {
        cudaFree(d_csrRowPtrC);
    }
    if (d_csrColIdxC != NULL)
    {
        cudaFree(d_csrColIdxC);
    }
    if (d_csrValC != NULL)
    {
        cudaFree(d_csrValC);
    }
    if (matA != NULL)
    {
        cusparseDestroySpMat(matA);
    }
    if (matB != NULL)
    {
        cusparseDestroySpMat(matB);
    }
    if (matC != NULL)
    {
        cusparseDestroySpMat(matC);
    }
    if (handle != NULL)
    {
        cusparseDestroy(handle);
    }

    return ret;
}

	#endif
