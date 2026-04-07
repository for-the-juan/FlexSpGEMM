#include"common.h"
#include"mmio_highlevel.h"
#include"utils.h"
#include"utils_cuda_scan.h"
#include "spgemm_nsparse_kernel.h"
#include "csr2tile.h"
#include "tilespgemm-cuda.h"
// #include "spgemm-cpu.h"
#include "tile2csr.h"
#include "spgemm_serialref_spa_new.h"
#include "spgemm_cu.h"

int main(int argc, char ** argv)
{
	if (argc < 6)
    {
        printf("Usage: ./test -d <device_id> -aat <0|1> -tau x <matrix.mtx>\n");
        printf("  -aat 0 : compute C = A * A\n");
        printf("  -aat 1 : compute C = A * A^T\n");
        return 0;
    }
    
    int device_id = 0;
    int aat = 0;

    int argi = 1;

    // load device id
    char *devstr;
    if(argc > argi)
    {
        devstr = argv[argi];
        argi++;
    }

    if (strcmp(devstr, "-d") != 0) return 0;

    if(argc > argi)
    {
        device_id = atoi(argv[argi]);
        argi++;
    }
    
    // set device
    cudaSetDevice(device_id);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    // Set aside 50% of L2 cache for persisting accesses 
    size_t size = min( int(deviceProp.l2CacheSize * 0.80) , deviceProp.persistingL2CacheMaxSize );
    cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size); 

    printf("\n");
    printf("================================================================================\n");
    printf("  FlexSpGEMM Performance Evaluation\n");
    printf("================================================================================\n");
    printf("\n");
    printf("[Device]\n");
    printf("  Device ID   : %d\n", device_id);
    printf("  Device Name : %s\n", deviceProp.name);
    printf("  Clock Rate  : %.2f MHz\n", deviceProp.clockRate * 1e-3f);
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");
           
    // load AAT flag
    char *aatstr;
    if(argc > argi)
    {
        aatstr = argv[argi];
        argi++;
    }

    if (strcmp(aatstr, "-aat") != 0) return 0;

    if(argc > argi)
    {
        aat = atoi(argv[argi]);
        argi++;
    }

 	struct timeval t1, t2;
	SMatrixA *matrixA = (SMatrixA *)malloc(sizeof(SMatrixA));
	SMatrixB *matrixB = (SMatrixB *)malloc(sizeof(SMatrixB));

	char  *filename;
    filename = argv[argi];
    argi++;

    // The tile of A is m×n, and the tile of B is n×m
    // load mtx A data to the csr format
    gettimeofday(&t1, NULL);
    mmio_allinone(&matrixA->m, &matrixA->n, &matrixA->nnz, &matrixA->isSymmetric, &matrixA->rowpointer, &matrixA->columnindex, &matrixA->value, filename);
    gettimeofday(&t2, NULL);
    double time_loadmat  = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    
    printf("\n[Input Matrix]\n");
    char *base = strrchr(filename, '/');
    char *fname = base ? base + 1 : filename;
    printf("  File        : %s\n", fname);
    printf("  Path        : %s\n", filename);
    printf("  Dimension   : %d x %d\n", matrixA->m, matrixA->n);
    printf("  NNZ (A)     : %d\n", matrixA->nnz);
    printf("  Load Time   : %.5f sec\n", time_loadmat / 1000.0);
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");

    printf("\n[Tiling Configuration]\n");
    printf("  Tile Size   : %d x %d  (TILE_SIZE_M x TILE_SIZE_N)\n", TILE_SIZE_M, TILE_SIZE_N);
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");

    if (!aat &&  matrixA->m != matrixA->n)
    {
        printf("[ERROR] Matrix squaring requires rowA == colA. Exiting.\n");
        return 0;
    }

	for (int i = 0; i < matrixA->nnz; i++)
	    matrixA->value[i] = i % 10;

    if (aat)
    {
        MAT_PTR_TYPE *cscColPtrA;
        int *cscRowIdxA;
        MAT_VAL_TYPE *cscValA ;
    
        if (matrixA->m == matrixA->n && matrixA->isSymmetric)
        {
           printf("Matrix AAT does not do symmetric matrix. Exit.\n");
           return 0;
        }

        matrixB->m = matrixA->n ;
        matrixB->n = matrixA->m ;
        matrixB->nnz = matrixA->nnz ;

        cscColPtrA = (MAT_PTR_TYPE *)malloc((matrixA->n + 1) * sizeof(MAT_PTR_TYPE));
        cscRowIdxA = (int *)malloc(matrixA->nnz   * sizeof(int));
        cscValA    = (MAT_VAL_TYPE *)malloc(matrixA->nnz  * sizeof(MAT_VAL_TYPE));

        // transpose A from csr to csc
        matrix_transposition(matrixA->m, matrixA->n, matrixA->nnz, matrixA->rowpointer, matrixA->columnindex, matrixA->value,cscRowIdxA, cscColPtrA, cscValA);

        matrixB->rowpointer = cscColPtrA;
        matrixB->columnindex = cscRowIdxA;
        matrixB->value    = cscValA;


    }
    else
    {
        matrixB->m = matrixA->m ;
        matrixB->n = matrixA->n ;
        matrixB->nnz = matrixA->nnz ;

        matrixB->rowpointer = matrixA->rowpointer;
        matrixB->columnindex = matrixA->columnindex;
        matrixB->value    = matrixA->value;
    }

        // calculate bytes and flops consumed
        unsigned long long int nnzCub = 0;
        for (int i = 0; i < matrixA->nnz; i++)
        {
            int rowidx = matrixA->columnindex[i];
            nnzCub += matrixB->rowpointer[rowidx + 1] - matrixB->rowpointer[rowidx];
        }
    
        printf("\n[Preprocessing]\n");
        printf("  NNZ Upper Bound (nnzCub) : %lld\n", nnzCub);

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        csr2tile_row_major(matrixA, TILE_SIZE_M, TILE_SIZE_N);

#if TIMING
        gettimeofday(&t2, NULL);
        double time_conversion = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        printf("  Format Conversion        : %.2f ms\n", time_conversion);
#endif

#if SPACE

double tile_bytes = (matrixA->tilem + 1) * sizeof(int) + matrixA->numtile * sizeof(int) + (matrixA->numtile + 1) *sizeof(int) +
                matrixA->nnz * sizeof(MAT_VAL_TYPE) + matrixA->nnz * sizeof(TILE_CSR_COL_TYPE_A) + matrixA->numtile * TILE_SIZE_M * sizeof(TILE_CSR_PTR_TYPE) +
                matrixA->numtile * TILE_SIZE_M * sizeof(TILE_MASK_TYPE_A);

double mem = tile_bytes/1024/1024;

double CSR_bytes = (matrixA->m +1) * sizeof(int) + (matrixA->nnz) * sizeof(int) + matrixA->nnz * sizeof(MAT_VAL_TYPE);
double csr_mem = CSR_bytes/1024/1024;

double dense_bytes = (long long int)matrixA->dense_tile_count * TILE_SIZE_M * TILE_SIZE_N * sizeof(MAT_VAL_TYPE) + 
                     (long long int)matrixA->dense_tile_count * sizeof(int);
double dense_mem = dense_bytes/1024/1024;

double all_dense_bytes = matrixA->numtile * TILE_SIZE_M * TILE_SIZE_N * sizeof(MAT_VAL_TYPE);
double all_dense_mem = all_dense_bytes/1024/1024;

printf("\n  Memory Overhead:\n");
printf("    CSR Memory Cost        : %.2f MB\n", csr_mem);
printf("    Dense Memory Cost      : %.2f MB\n", all_dense_mem);
printf("    TileSpGEMM Memory Cost : %.2f MB\n", mem);
printf("    FlexSpGEMM Memory Cost : %.2f MB  (DNS_THRESHOLD = %.2f)\n", dense_mem + mem, float(TILE_DENSE_THRESHOLD) / 10);
printf("\n");
printf("--------------------------------------------------------------------------------\n");

#endif

        csr2tile_col_major(matrixB, TILE_SIZE_M, TILE_SIZE_N);


        // how much unsigned int to store the row-wise tile bitmask
        int blk_intersec_bitmask_len = ceil((double)matrixA->tilen / 32.0);
        double densityA = (double)matrixA->numtile / ((double)matrixA->tilem*(double)matrixA->tilen);
        double densityB = (double)matrixB->numtile / ((double)matrixB->tilem*(double)matrixB->tilen);


        // the total unsigned int to store the whole tile bitmask of matrix A
        long long int lengthA = (long long int) (matrixA->tilem) * (long long int)( blk_intersec_bitmask_len) ;

    unsigned int *blk_intersec_bitmask_A = (unsigned int *)malloc(lengthA* sizeof(unsigned int));
    memset(blk_intersec_bitmask_A, 0, lengthA * sizeof(unsigned int));
    for (int i = 0; i < matrixA->tilem; i++)
    {
        for (int j = matrixA->tile_ptr[i]; j < matrixA->tile_ptr[i + 1]; j++)
        {
            int idx = matrixA->tile_columnidx[j];
            unsigned int bitmask = 1;
            bitmask <<=  (31- (idx % 32));
            long long int pos = (long long int)i * (long long int)blk_intersec_bitmask_len + idx / 32;
            blk_intersec_bitmask_A[pos] |= bitmask;
        }
    }

    // the calculation of tile bitmask of B is similar to A, but for CSC format
    long long int lengthB = (long long int) (matrixB->tilen) * (long long int)(blk_intersec_bitmask_len) ;

    unsigned int *blk_intersec_bitmask_B = (unsigned int *)malloc(lengthB * sizeof(unsigned int));
    memset(blk_intersec_bitmask_B, 0, lengthB * sizeof(unsigned int));
    for (int i = 0; i < matrixB->tilen; i++)
    {
        for (int j = matrixB->csc_tile_ptr[i]; j < matrixB->csc_tile_ptr[i+1]; j++)
        {
            int idx = matrixB->csc_tile_rowidx[j];
            unsigned int bitmask = 0x1;
            bitmask <<= (31 - (idx % 32));
            long long int pos = (long long int)i * (long long int )blk_intersec_bitmask_len + idx / 32;
            blk_intersec_bitmask_B[pos] |= bitmask;
        }
    }


    // generate rowidx of tiles in blockA
    int *tile_rowidx_A = (int *)malloc (matrixA->numtile * sizeof(int));
    for (int i = 0; i < matrixA->tilem; i++)
    {
        for (int j = matrixA->tile_ptr[i]; j < matrixA->tile_ptr[i+1]; j++)
        {
            tile_rowidx_A[j] = i;
        }
    }



#ifdef DEBUG
    // --------------------------------------------------------------------------------------------------------
    SMatrixB *matrixC = (SMatrixB *)malloc(sizeof(SMatrixB));
    
    struct timeval tv;
    unsigned long long int nnzC_computed;
    double compression_rate = 0;
    double time_tile = 0;
    double gflops_tile = 0;
    double time_symbolic = 0;
    double time_numeric = 0;
    double time_malloc = 0; 

    tilespgemm(matrixA,
               matrixB,
               matrixC,
               blk_intersec_bitmask_A,
               blk_intersec_bitmask_B,
               blk_intersec_bitmask_len,
               densityA,
               densityB,
               nnzCub,
               &nnzC_computed,
               &compression_rate,
               &time_tile,
               &gflops_tile,
               filename,
               &time_symbolic,&time_numeric,&time_malloc);

    // for (int i = 0; i < 10; i++){
    //     printf("[DEBUG] tile_ptr[%d]: %d\n", i, matrixC->tile_ptr[i]);
    // }
    
    // write results to text (csv) file
    FILE *fout = fopen("../result/results_tile.csv", "a");
    if (fout == NULL)
        printf("Writing results fails.\n");
    fprintf(fout, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f\n",
            filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, time_tile, gflops_tile);
    fclose(fout);

    // write runtime of each step to text (csv) file
    FILE *fout_time = fopen("../result/step_runtime.csv", "a");
    if (fout_time == NULL)
        printf("Writing results fails.\n");
    fprintf(fout_time, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f,%f,%f\n",
                filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, time_symbolic, time_numeric, time_malloc);
    fclose(fout_time);
    

#if SPACE
    // write memory space of CSR and tile format to text (csv) file
    FILE *fout_mem = fopen("../result/mem-cost.csv", "a");
    if (fout_mem == NULL)
        printf("Writing results fails.\n");
    fprintf(fout_mem, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f\n",
                filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, csr_mem,mem);
    fclose(fout_mem);

#endif

#if TIMING

    // write preprocessing overhead of CSR and tile format to text (scv) file
    FILE *fout_pre = fopen("../result/preprocessing.csv", "a");
    if (fout_pre == NULL)
        printf("Writing results fails.\n");
    fprintf(fout_pre, "%s,%i,%i,%i,%lld,%lld,%f,%f,%f\n",
                    filename, matrixA->m, matrixA->n, matrixA->nnz, nnzCub, nnzC_computed, compression_rate, time_conversion,time_tile);
    fclose(fout_pre);
    
#endif


#endif

#if CHECK_RESULT
tile2csr(matrixC, TILE_SIZE_M, TILE_SIZE_M);

    unsigned long long int nnzC = 0;
    double compression_rate1 = 0;
    double time_cusparse = 0;
    double gflops_cusparse = 0;
    int flag =0;
    int mC = matrixA->m;
    int nC = matrixB->n;
    int nnzC_golden = matrixC->nnz;
    bool check_result = CHECK_RESULT;

    MAT_PTR_TYPE *csrRowPtrC_golden = matrixC->rowpointer;
    int *csrColIdxC_golden = matrixC->columnindex;
    MAT_VAL_TYPE *csrValC_golden = matrixC->value;

    int cusparse_ret = spgemm_cu(matrixA->m, matrixA->n, matrixA->nnz, matrixA->rowpointer, matrixA->columnindex, matrixA->value,
              matrixB->m, matrixB->n, matrixB->nnz, matrixB->rowpointer, matrixB->columnindex, matrixB->value,
              mC, nC, nnzC_golden, csrRowPtrC_golden, csrColIdxC_golden, csrValC_golden,
              check_result, nnzCub, &nnzC, &compression_rate1, &time_cusparse, &gflops_cusparse);

    printf("[Speedup]\n");
    if (cusparse_ret != 0 || time_cusparse <= 0.0)
        {
            printf("  FlexSpGEMM vs. cuSPARSE  : Comparison failed "
                "(cuSPARSE did not produce valid results)\n");
        }
        else
        {
            printf("  FlexSpGEMM vs. cuSPARSE  : %.2fx\n",
                time_cusparse / time_tile);
        }
    printf("--------------------------------------------------------------------------------\n");

#endif
    matrix_destroy(matrixA);
    matrix_destroy_B(matrixB);

    free(matrixA->rowpointer);
    free(matrixA->columnindex);
    free(matrixA->value);

    return 0;

}