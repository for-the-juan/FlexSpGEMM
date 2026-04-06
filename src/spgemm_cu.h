#include "common.h"
#include <cuda_runtime.h>
#include "external/cusparse/spgemm_cusparse.h"

int spgemm_cu (         const int             mA,
                        const int             nA,
                        const int             nnzA,
                        const MAT_PTR_TYPE   *csrRowPtrA,
                        const int            *csrColIdxA,
                        const MAT_VAL_TYPE   *csrValA,
                        const int             mB,
                        const int             nB,
                        const int             nnzB,
                        const MAT_PTR_TYPE   *csrRowPtrB,
                        const int            *csrColIdxB,
                        const MAT_VAL_TYPE   *csrValB,
                        const int             mC,
                        const int             nC,
                        const MAT_PTR_TYPE    nnzC_golden,
                        const MAT_PTR_TYPE   *csrRowPtrC_golden,
                        const int            *csrColIdxC_golden,
                        const MAT_VAL_TYPE   *csrValC_golden,
                        const bool           check_result,
                        unsigned long long int nnzCub,
                        unsigned long long int *nnzC,
                        double        *compression_rate,
                        double        *time_segmerge,
                        double        *gflops_segmerge )
{
    // run cuda SpGEMM (using cuSPARSE)
    printf("\n[Baseline: cuSPARSE SpGEMM]\n");
    int ret = spgemm_cusparse(mA, nA, nnzA, csrRowPtrA, csrColIdxA, csrValA,
                 mB, nB, nnzB, csrRowPtrB, csrColIdxB, csrValB,
                 mC, nC, nnzC_golden, csrRowPtrC_golden, csrColIdxC_golden, csrValC_golden,
                 check_result, nnzCub, nnzC, compression_rate, time_segmerge, gflops_segmerge);
    return ret;
}




