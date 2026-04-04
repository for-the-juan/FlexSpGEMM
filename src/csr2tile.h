#include "common.h"
#include "utils.h"

/*    STEP1: Calculate the number of non-empty tile of a sparse matrix   */
/*           Record the offset of tiles in each tile row                 */
void step1_kernel(SMatrixA *matrix, int tile_size_m, int tile_size_n)

{
    int *rowpointer = matrix->rowpointer;
    int m = matrix->m;
    int *columnidx = matrix->columnindex;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;

    unsigned thread = omp_get_max_threads();
    // unsigned thread = matrix->nthreads;

    char *flag_g = (char *)malloc(thread * tilen * sizeof(char));
#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        char *flag = flag_g + thread_id * tilen;
        memset(flag, 0, tilen * sizeof(char));
        int start = blki * tile_size_m;
        int end = blki == tilem - 1 ? m : (blki + 1) * tile_size_m;
        for (int j = rowpointer[start]; j < rowpointer[end]; j++)
        {
            int jc = columnidx[j] / tile_size_n;
            if (flag[jc] == 0)
            {
                flag[jc] = 1;
                tile_ptr[blki]++;
            }
        }
    }
    free(flag_g);
}

void step1_kernel_B(SMatrixB *matrix, int tile_size_m, int tile_size_n)

{
    int *rowpointer = matrix->rowpointer;
    int m = matrix->m;
    int *columnidx = matrix->columnindex;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;

    unsigned thread = omp_get_max_threads();
    // unsigned thread = matrix->nthreads;

    char *flag_g = (char *)malloc(thread * tilen * sizeof(char));
#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        char *flag = flag_g + thread_id * tilen;
        memset(flag, 0, tilen * sizeof(char));
        int start = blki * tile_size_m;
        int end = blki == tilem - 1 ? m : (blki + 1) * tile_size_m;
        for (int j = rowpointer[start]; j < rowpointer[end]; j++)
        {
            int jc = columnidx[j] / tile_size_n;
            if (flag[jc] == 0)
            {
                flag[jc] = 1;
                tile_ptr[blki]++;
            }
        }
    }
    free(flag_g);
}

/*   STEP2:  Calculate column and row index of each tile */
/*           Calculate the number of nonzeros of each tile*/
void step2_kernel(SMatrixA *matrix, TILE_CSR_PTR_TYPE *tile_csr_ptr, int tile_size_m, int tile_size_n)

{
    int m = matrix->m;
    int *rowpointer = matrix->rowpointer;
    int *columnidx = matrix->columnindex;

    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_rowidx = matrix->tile_rowidx;
    int *tile_nnz = matrix->tile_nnz;

    unsigned thread = omp_get_max_threads();
    // unsigned thread = matrix->nthreads;

    char *col_temp_g = (char *)malloc((thread * tilen) * sizeof(char));

    int *nnz_temp_g = (int *)malloc((thread * tilen) * sizeof(int));

    TILE_CSR_PTR_TYPE *ptr_per_tile_g = (TILE_CSR_PTR_TYPE *)malloc((thread * tilen * tile_size_m) * sizeof(TILE_CSR_PTR_TYPE));

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        char *col_temp = col_temp_g + thread_id * tilen;
        memset(col_temp, 0, tilen * sizeof(char));
        int *nnz_temp = nnz_temp_g + thread_id * tilen;
        memset(nnz_temp, 0, tilen * sizeof(int));
        TILE_CSR_PTR_TYPE *ptr_per_tile = ptr_per_tile_g + thread_id * tilen * tile_size_m;
        memset(ptr_per_tile, 0, tilen * tile_size_m * sizeof(TILE_CSR_PTR_TYPE));
        int pre_tile = tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? m - (tilem - 1) * tile_size_m : tile_size_m;
        int start = blki * tile_size_m;

        for (int ri = 0; ri < rowlen; ri++)
        {
            for (int j = rowpointer[start + ri]; j < rowpointer[start + ri + 1]; j++)
            {
                int jc = columnidx[j] / tile_size_n;
                col_temp[jc] = 1;
                nnz_temp[jc]++;
                ptr_per_tile[jc * tile_size_m + ri]++;  // FIXED: use tile_size_m for row pointers
            }
        }

        int count = 0;
        for (int blkj = 0; blkj < tilen; blkj++)
        {
            if (col_temp[blkj] == 1)
            {
                tile_columnidx[pre_tile + count] = blkj;
                tile_rowidx[pre_tile + count] = blki;
                tile_nnz[pre_tile + count] = nnz_temp[blkj];
                for (int ri = 0; ri < rowlen; ri++)
                {
                    tile_csr_ptr[(pre_tile + count) * tile_size_m + ri] = ptr_per_tile[blkj * tile_size_m + ri];
                }
                count++;
            }
        }
    }
    free(col_temp_g);
    free(nnz_temp_g);
    free(ptr_per_tile_g);
}

void step2_kernel_B(SMatrixB *matrix, TILE_CSR_PTR_TYPE *tile_csr_ptr, int tile_size_m, int tile_size_n)

{
    int m = matrix->m;
    int *rowpointer = matrix->rowpointer;
    int *columnidx = matrix->columnindex;

    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_rowidx = matrix->tile_rowidx;
    int *tile_nnz = matrix->tile_nnz;

    unsigned thread = omp_get_max_threads();
    // unsigned thread = matrix->nthreads;

    char *col_temp_g = (char *)malloc((thread * tilen) * sizeof(char));

    int *nnz_temp_g = (int *)malloc((thread * tilen) * sizeof(int));

    TILE_CSR_PTR_TYPE *ptr_per_tile_g = (TILE_CSR_PTR_TYPE *)malloc((thread * tilen * tile_size_m) * sizeof(TILE_CSR_PTR_TYPE));

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        char *col_temp = col_temp_g + thread_id * tilen;
        memset(col_temp, 0, tilen * sizeof(char));
        int *nnz_temp = nnz_temp_g + thread_id * tilen;
        memset(nnz_temp, 0, tilen * sizeof(int));
        TILE_CSR_PTR_TYPE *ptr_per_tile = ptr_per_tile_g + thread_id * tilen * tile_size_m;
        memset(ptr_per_tile, 0, tilen * tile_size_m * sizeof(TILE_CSR_PTR_TYPE));
        int pre_tile = tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? m - (tilem - 1) * tile_size_m : tile_size_m;
        int start = blki * tile_size_m;

        for (int ri = 0; ri < rowlen; ri++)
        {
            for (int j = rowpointer[start + ri]; j < rowpointer[start + ri + 1]; j++)
            {
                int jc = columnidx[j] / tile_size_n;
                col_temp[jc] = 1;
                nnz_temp[jc]++;
                ptr_per_tile[jc * tile_size_m + ri]++;  // FIXED: use tile_size_m for row pointers
            }
        }

        int count = 0;
        for (int blkj = 0; blkj < tilen; blkj++)
        {
            if (col_temp[blkj] == 1)
            {
                tile_columnidx[pre_tile + count] = blkj;
                tile_rowidx[pre_tile + count] = blki;
                tile_nnz[pre_tile + count] = nnz_temp[blkj];
                for (int ri = 0; ri < rowlen; ri++)
                {
                    tile_csr_ptr[(pre_tile + count) * tile_size_m + ri] = ptr_per_tile[blkj * tile_size_m + ri];
                }
                count++;
            }
        }
    }
    free(col_temp_g);
    free(nnz_temp_g);
    free(ptr_per_tile_g);
}

void step3_kernel(SMatrixA *matrix, int nnz_max, int tilecnt_max, int tile_size_m, int tile_size_n)
{
    int *rowpointer = matrix->rowpointer;
    int *columnidx = matrix->columnindex;
    MAT_VAL_TYPE *value = matrix->value;
    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    MAT_VAL_TYPE *tile_csr_Val = matrix->tile_csr_Value;
    TILE_CSR_COL_TYPE_A *tile_csr_Col = matrix->tile_csr_Col;
    TILE_CSR_PTR_TYPE *tile_csr_Ptr = matrix->tile_csr_Ptr;
    
    MAT_VAL_TYPE *dense_data = matrix->dense_data;
    int *tile_dense_ready = matrix->tile_dense_ready;

    TILE_MASK_TYPE_A *mask = matrix->mask;

    unsigned thread = omp_get_max_threads();

    TILE_CSR_COL_TYPE_A *csr_colidx_temp_g = (TILE_CSR_COL_TYPE_A *)malloc((thread * nnz_max) * sizeof(TILE_CSR_COL_TYPE_A));
    MAT_VAL_TYPE *csr_val_temp_g = (MAT_VAL_TYPE *)malloc((thread * nnz_max) * sizeof(MAT_VAL_TYPE));
    int *tile_count_g = (int *)malloc(thread * tilecnt_max * sizeof(int));

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        TILE_CSR_COL_TYPE_A *csr_colidx_temp = csr_colidx_temp_g + thread_id * nnz_max;
        MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + thread_id * nnz_max;
        int *tile_count = tile_count_g + thread_id * tilecnt_max;
        memset(csr_colidx_temp, 0, (nnz_max) * sizeof(TILE_CSR_COL_TYPE_A));
        memset(csr_val_temp, 0, (nnz_max) * sizeof(MAT_VAL_TYPE));
        memset(tile_count, 0, (tilecnt_max) * sizeof(int));
        int tilenum_per_row = tile_ptr[blki + 1] - tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? m - (tilem - 1) * tile_size_m : tile_size_m;
        int start = blki * tile_size_m;
        int end = blki == tilem - 1 ? m : (blki + 1) * tile_size_m;

        for (int ri = 0; ri < rowlen; ri++)
        {
            int current_row = start + ri;
            for (int blkj = rowpointer[current_row]; blkj < rowpointer[current_row + 1]; blkj++)
            {
                int jc_temp = columnidx[blkj] / tile_size_n;
                for (int bi = 0; bi < tilenum_per_row; bi++)
                {
                    int tile_id = tile_ptr[blki] + bi;
                    int jc = tile_columnidx[tile_id];
                    int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
                    if (jc == jc_temp)
                    {
                        csr_val_temp[pre_nnz + tile_count[bi]] = value[blkj];
                        csr_colidx_temp[pre_nnz + tile_count[bi]] = columnidx[blkj] - jc * tile_size_n;

                        // Only write to dense_data if this tile is dense (tile_dense_ready >= 0)
                        if (tile_dense_ready[tile_id] >= 0)
                        {
                            // ri is the local row index (0 to tile_size_m-1)
                            // col_in_tile is the local col index (0 to tile_size_n-1)
                            int col_in_tile = columnidx[blkj] - jc * tile_size_n;
                            
                            int dense_offset = tile_dense_ready[tile_id];
                            long long internal_offset = (long long)ri * tile_size_n + col_in_tile;
                            
                            dense_data[dense_offset + internal_offset] = value[blkj];
                        }
                        tile_count[bi]++;
                        break;
                    }
                }
            }
        }
        for (int bi = 0; bi < tilenum_per_row; bi++)
        {
            int tile_id = tile_ptr[blki] + bi;
            int tilennz = tile_nnz[tile_id + 1] - tile_nnz[tile_id];
            int offset = tile_nnz[tile_id];
            int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];

            TILE_CSR_PTR_TYPE *ptr_temp = tile_csr_Ptr + tile_id * tile_size_m;
            for (int ri = 0; ri < rowlen; ri++)
            {
                int start = ptr_temp[ri];
                int stop = ri == rowlen - 1 ? tilennz : ptr_temp[ri + 1];
                for (int k = start; k < stop; k++)
                {
                    TILE_CSR_COL_TYPE_A colidx = csr_colidx_temp[pre_nnz + k];
                    tile_csr_Val[offset + k] = csr_val_temp[pre_nnz + k];

                    tile_csr_Col[offset + k] = (ri * tile_size_n) + colidx;
                    assert((tile_size_n % MaskBitsA)==0);
                    int stride = colidx / MaskBitsA;
                    mask[tile_id * tile_size_m * tile_size_n / MaskBitsA + ri * tile_size_n / MaskBitsA + stride] |= (0x1ULL << (MaskBitsA - colidx % MaskBitsA - 1));
                }
            }
        }
    }
    free(csr_colidx_temp_g);
    free(csr_val_temp_g);
    free(tile_count_g);
}

void step3_kernel_B(SMatrixB *matrix, int nnz_max, int tilecnt_max, int tile_size_m, int tile_size_n)
{
    int *rowpointer = matrix->rowpointer;
    int *columnidx = matrix->columnindex;
    MAT_VAL_TYPE *value = matrix->value;
    int m = matrix->m;
    int n = matrix->n;
    int tilem = matrix->tilem;
    int tilen = matrix->tilen;
    MAT_PTR_TYPE *tile_ptr = matrix->tile_ptr;
    int *tile_columnidx = matrix->tile_columnidx;
    int *tile_nnz = matrix->tile_nnz;
    MAT_VAL_TYPE *tile_csr_Val = matrix->tile_csr_Value;
    TILE_CSR_COL_TYPE_B *tile_csr_Col = matrix->tile_csr_Col;
    TILE_CSR_PTR_TYPE *tile_csr_Ptr = matrix->tile_csr_Ptr;

    MAT_VAL_TYPE *dense_data = matrix->dense_data;
    int *tile_dense_ready = matrix->tile_dense_ready;

    TILE_MASK_TYPE_B *mask = matrix->mask;

    unsigned thread = omp_get_max_threads();
    // unsigned thread = matrix->nthreads;

    TILE_CSR_COL_TYPE_B *csr_colidx_temp_g = (TILE_CSR_COL_TYPE_B *)malloc((thread * nnz_max) * sizeof(TILE_CSR_COL_TYPE_B));
    MAT_VAL_TYPE *csr_val_temp_g = (MAT_VAL_TYPE *)malloc((thread * nnz_max) * sizeof(MAT_VAL_TYPE));
    int *tile_count_g = (int *)malloc(thread * tilecnt_max * sizeof(int));

#pragma omp parallel for
    for (int blki = 0; blki < tilem; blki++)
    {
        int thread_id = omp_get_thread_num();
        TILE_CSR_COL_TYPE_B *csr_colidx_temp = csr_colidx_temp_g + thread_id * nnz_max;
        MAT_VAL_TYPE *csr_val_temp = csr_val_temp_g + thread_id * nnz_max;
        int *tile_count = tile_count_g + thread_id * tilecnt_max;
        memset(csr_colidx_temp, 0, (nnz_max) * sizeof(TILE_CSR_COL_TYPE_B));
        memset(csr_val_temp, 0, (nnz_max) * sizeof(MAT_VAL_TYPE));
        memset(tile_count, 0, (tilecnt_max) * sizeof(int));
        int tilenum_per_row = tile_ptr[blki + 1] - tile_ptr[blki];
        int rowlen = blki == tilem - 1 ? m - (tilem - 1) * tile_size_m : tile_size_m;
        int start = blki * tile_size_m;
        int end = blki == tilem - 1 ? m : (blki + 1) * tile_size_m;

        for (int ri = 0; ri < rowlen; ri++)
        {
            for (int blkj = rowpointer[start + ri]; blkj < rowpointer[start + ri + 1]; blkj++)
            {
                int jc_temp = columnidx[blkj] / tile_size_n;
                for (int bi = 0; bi < tilenum_per_row; bi++)
                {
                    int tile_id = tile_ptr[blki] + bi;
                    int jc = tile_columnidx[tile_id];
                    int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];
                    if (jc == jc_temp)
                    {
                        csr_val_temp[pre_nnz + tile_count[bi]] = value[blkj];
                        csr_colidx_temp[pre_nnz + tile_count[bi]] = columnidx[blkj] - jc * tile_size_n;
                        
                        // Only write to dense_data if this tile is dense (tile_dense_ready >= 0)
                        if (tile_dense_ready[tile_id] >= 0)
                        {
                            int col_in_tile = columnidx[blkj] - jc * tile_size_n;
                            
                            int dense_offset = tile_dense_ready[tile_id];
                            long long internal_offset = (long long)ri * tile_size_n + col_in_tile;
                            
                            dense_data[dense_offset + internal_offset] = value[blkj];
                        }
                        
                        tile_count[bi]++;
                        break;
                    }
                }
            }
        }
        for (int bi = 0; bi < tilenum_per_row; bi++)
        {
            int tile_id = tile_ptr[blki] + bi;
            int tilennz = tile_nnz[tile_id + 1] - tile_nnz[tile_id];
            int offset = tile_nnz[tile_id];
            int pre_nnz = tile_nnz[tile_id] - tile_nnz[tile_ptr[blki]];

            TILE_CSR_PTR_TYPE *ptr_temp = tile_csr_Ptr + tile_id * tile_size_m;
            for (int ri = 0; ri < rowlen; ri++)
            {
                int start = ptr_temp[ri];
                int stop = ri == rowlen - 1 ? tilennz : ptr_temp[ri + 1];
                for (int k = start; k < stop; k++)
                {
                    TILE_CSR_COL_TYPE_B colidx = csr_colidx_temp[pre_nnz + k];
                    tile_csr_Val[offset + k] = csr_val_temp[pre_nnz + k];

                    tile_csr_Col[offset + k] = (ri * tile_size_n) + colidx;  // FIXED: use tile_size_n for columns
                    assert((tile_size_n % MaskBitsA)==0);
                    int stride = colidx / MaskBitsA;
                    mask[tile_id * tile_size_m * tile_size_n / MaskBitsA + ri * tile_size_n / MaskBitsA + stride] |= (0x1ULL << (MaskBitsA - colidx % MaskBitsA - 1));
                }
            }
        }
    }
    free(csr_colidx_temp_g);
    free(csr_val_temp_g);
    free(tile_count_g);
}

void csr2tile_row_major(SMatrixA *matrix, int tile_size_m, int tile_size_n)
{
    int nthreads = omp_get_max_threads();

    matrix->numtile = 0;
    matrix->tilem = matrix->m % tile_size_m == 0 ? matrix->m / tile_size_m : (matrix->m / tile_size_m) + 1;
    matrix->tilen = matrix->n % tile_size_n == 0 ? matrix->n / tile_size_n : (matrix->n / tile_size_n) + 1;

    matrix->tile_ptr = (MAT_PTR_TYPE *)malloc((matrix->tilem + 1) * sizeof(MAT_PTR_TYPE));
    memset(matrix->tile_ptr, 0, (matrix->tilem + 1) * sizeof(MAT_PTR_TYPE));

    step1_kernel(matrix, tile_size_m, tile_size_n);
    exclusive_scan(matrix->tile_ptr, matrix->tilem + 1);

    matrix->numtile = matrix->tile_ptr[matrix->tilem];

    matrix->tile_columnidx = (int *)malloc(matrix->numtile * sizeof(int));
    memset(matrix->tile_columnidx, 0, matrix->numtile * sizeof(int));

    matrix->tile_rowidx = (int *)malloc(matrix->numtile * sizeof(int));
    memset(matrix->tile_rowidx, 0, matrix->numtile * sizeof(int));

    matrix->tile_nnz = (int *)malloc((matrix->numtile + 1) * sizeof(int));
    memset(matrix->tile_nnz, 0, (matrix->numtile + 1) * sizeof(int));

    matrix->tile_csr_Ptr = (TILE_CSR_PTR_TYPE *)malloc((matrix->numtile * tile_size_m) * sizeof(TILE_CSR_PTR_TYPE));
    memset(matrix->tile_csr_Ptr, 0, (matrix->numtile * tile_size_m) * sizeof(TILE_CSR_PTR_TYPE));

    step2_kernel(matrix, matrix->tile_csr_Ptr, tile_size_m, tile_size_n);

#pragma omp parallel for
    for (int blki = 0; blki < matrix->tilem; blki++)
    {
        quick_sort_key(matrix->tile_columnidx + matrix->tile_ptr[blki], matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki]);
    }

    exclusive_scan(matrix->tile_nnz, matrix->numtile + 1);

    for (int i = 0; i < matrix->numtile; i++)
    {
        TILE_EXCLUSIVE_SCAN_FUNC(matrix->tile_csr_Ptr + i * tile_size_m, tile_size_m);
    }

    matrix->tile_csr_Col = (TILE_CSR_COL_TYPE_A *)malloc(matrix->nnz * sizeof(TILE_CSR_COL_TYPE_A));
    memset(matrix->tile_csr_Col, 0, matrix->nnz * sizeof(TILE_CSR_COL_TYPE_A));

    matrix->tile_csr_Value = (MAT_VAL_TYPE *)malloc(matrix->nnz * sizeof(MAT_VAL_TYPE));
    memset(matrix->tile_csr_Value, 0, matrix->nnz * sizeof(MAT_VAL_TYPE));

    // matrix->mask = (TILE_MASK_TYPE *)malloc(matrix->numtile * tile_size_m * tile_size_n / sizeof(TILE_MASK_TYPE) / BitsPerByte * sizeof(TILE_MASK_TYPE));
    matrix->mask = (TILE_MASK_TYPE_A *)malloc(matrix->numtile * tile_size_m * tile_size_n / MaskBitsA * sizeof(TILE_MASK_TYPE_A));
    memset(matrix->mask, 0, matrix->numtile * tile_size_m * tile_size_n / MaskBitsA * sizeof(TILE_MASK_TYPE_A));

    // Step: Count dense tiles and allocate dense_data space
    // Calculate threshold for dense tile (nnz >= threshold * tile_size_m * tile_size_n)
    int tile_total_elems = tile_size_m * tile_size_n;
    int dense_threshold = (int)((float)(TILE_DENSE_THRESHOLD) / 10 * (float)tile_total_elems);
    
    // Allocate tile_dense_ready array, initialize to -1 (not dense)
    matrix->tile_dense_ready = (int *)malloc(matrix->numtile * sizeof(int));
    for (int i = 0; i < matrix->numtile; i++)
    {
        matrix->tile_dense_ready[i] = -1;
    }
    
    // Count dense tiles and set tile_dense_ready
    matrix->dense_tile_count = 0;
    for (int i = 0; i < matrix->numtile; i++)
    {
        int tile_nnz = matrix->tile_nnz[i + 1] - matrix->tile_nnz[i];
        if (tile_nnz >= dense_threshold)
        {
            // This is a dense tile, record its offset in dense_data
            matrix->tile_dense_ready[i] = matrix->dense_tile_count * tile_total_elems;
            matrix->dense_tile_count++;
        }
    }
    
    // Allocate dense_data only for dense tiles
    if (matrix->dense_tile_count > 0)
    {
        matrix->dense_data = (MAT_VAL_TYPE *)malloc(matrix->dense_tile_count * tile_total_elems * sizeof(MAT_VAL_TYPE));
        memset(matrix->dense_data, 0, matrix->dense_tile_count * tile_total_elems * sizeof(MAT_VAL_TYPE));
    }
    else
    {
        matrix->dense_data = NULL;
    }

    int nnz_max = 0;
    int tilecnt_max = 0;
    for (int blki = 0; blki < matrix->tilem; blki++)
    {
        int start = blki * tile_size_m;
        int end = blki == matrix->tilem - 1 ? matrix->m : (blki + 1) * tile_size_m;
        nnz_max = nnz_max < matrix->rowpointer[end] - matrix->rowpointer[start] ? matrix->rowpointer[end] - matrix->rowpointer[start] : nnz_max;
        tilecnt_max = tilecnt_max < matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki] ? matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki] : tilecnt_max;
    }

    step3_kernel(matrix, nnz_max, tilecnt_max, tile_size_m, tile_size_n);
    int tile_id = 1;
    for (int ri = 0; ri < 4; ri++)
    {
        for (int stride = 0; stride < tile_size_n / MaskBitsA; stride++){
            DEBUG_PRINT("[CSR2TILE A MASK CHECK] %d row %d bitmask in %d tile: %04x\n", ri, stride, tile_id, matrix->mask[
                tile_id * tile_size_m * (tile_size_n / MaskBitsA) + ri * (tile_size_n / MaskBitsA) + stride]);
        }
    }
}

void csr2tile_col_major(SMatrixB *matrix, int tile_size_m, int tile_size_n)
{
    matrix->numtile = 0;
    matrix->tilem = matrix->m % tile_size_n == 0 ? matrix->m / tile_size_n : (matrix->m / tile_size_n) + 1;
    matrix->tilen = matrix->n % tile_size_m == 0 ? matrix->n / tile_size_m : (matrix->n / tile_size_m) + 1;

	SMatrixB *BT = (SMatrixB *)malloc(sizeof(SMatrixB));

    BT->m = matrix->n;
    BT->n = matrix->m;
    BT->nnz = matrix->nnz;
    BT->tilem = matrix->tilen;
    BT->tilen = matrix->tilem;

    BT->tile_ptr = (MAT_PTR_TYPE *)malloc((BT->tilem + 1) * sizeof(MAT_PTR_TYPE));
    memset(BT->tile_ptr, 0, (BT->tilem + 1) * sizeof(MAT_PTR_TYPE));

    MAT_PTR_TYPE *cscColPtrB = (MAT_PTR_TYPE *)malloc((matrix->n + 1) * sizeof(MAT_PTR_TYPE));
    int *cscRowIdxB = (int *)malloc(matrix->nnz * sizeof(int));
    MAT_VAL_TYPE *cscValB = (MAT_VAL_TYPE *)malloc(matrix->nnz * sizeof(MAT_VAL_TYPE));

    matrix_transposition(matrix->m, matrix->n, matrix->nnz, matrix->rowpointer, matrix->columnindex, matrix->value, cscRowIdxB, cscColPtrB, cscValB);


    BT->value = cscValB;
    BT->columnindex = cscRowIdxB;
    BT->rowpointer = cscColPtrB;


    step1_kernel_B(BT, tile_size_m, tile_size_n);
    exclusive_scan(BT->tile_ptr, BT->tilem + 1);

    BT->numtile = BT->tile_ptr[BT->tilem];
    BT->tile_columnidx = (int *)malloc(BT->numtile * sizeof(int));
    memset(BT->tile_columnidx, 0, BT->numtile * sizeof(int));
    BT->tile_rowidx = (int *)malloc(BT->numtile * sizeof(int));
    memset(BT->tile_rowidx, 0, BT->numtile * sizeof(int));
    BT->tile_nnz = (int *)malloc((BT->numtile + 1) * sizeof(int));
    memset(BT->tile_nnz, 0, (BT->numtile + 1) * sizeof(int));
    BT->tile_csr_Ptr = (TILE_CSR_PTR_TYPE *)malloc((BT->numtile * tile_size_m) * sizeof(TILE_CSR_PTR_TYPE));
    memset(BT->tile_csr_Ptr, 0, (BT->numtile * tile_size_m) * sizeof(TILE_CSR_PTR_TYPE));
    step2_kernel_B(BT, BT->tile_csr_Ptr, tile_size_m, tile_size_n);
    exclusive_scan(BT->tile_nnz, BT->numtile + 1);

    matrix->tile_ptr = (MAT_PTR_TYPE *)malloc((matrix->tilem + 1) * sizeof(MAT_PTR_TYPE));
    memset(matrix->tile_ptr, 0, (matrix->tilem + 1) * sizeof(MAT_PTR_TYPE));

    step1_kernel_B(matrix, tile_size_n, tile_size_m);
    exclusive_scan(matrix->tile_ptr, matrix->tilem + 1);
    matrix->numtile = matrix->tile_ptr[matrix->tilem];
    matrix->tile_columnidx = (int *)malloc(matrix->numtile * sizeof(int));
    memset(matrix->tile_columnidx, 0, matrix->numtile * sizeof(int));
    matrix->tile_rowidx = (int *)malloc(matrix->numtile * sizeof(int));
    memset(matrix->tile_rowidx, 0, matrix->numtile * sizeof(int));

    // Note: dense_data and tile_dense_ready will be allocated after counting dense tiles
    matrix->dense_data = NULL;
    matrix->tile_dense_ready = NULL;
    matrix->dense_tile_count = 0;

    matrix->csc_tile_ptr = BT->tile_ptr;
    matrix->csc_tile_rowidx = BT->tile_columnidx;
    matrix->tile_nnz = BT->tile_nnz;

    char *flag = (char *)malloc(matrix->tilen * sizeof(char));

    int colid = 0;
    for (int i = 0; i < matrix->tilem; i++)
    {
        memset(flag, 0, matrix->tilen * sizeof(char));
        int start = i * tile_size_n;
        int end = i == matrix->tilem - 1 ? matrix->m : (i + 1) * tile_size_n;
        for (int j = matrix->rowpointer[start]; j < matrix->rowpointer[end]; j++)
        {
            int jc = matrix->columnindex[j] / tile_size_m;
            if (flag[jc] == 0)
            {
                flag[jc] = 1;
                matrix->tile_columnidx[colid] = jc;
                colid++;
            }
        }
    }



    matrix->tile_csr_Ptr = (TILE_CSR_PTR_TYPE *)malloc((matrix->numtile * tile_size_n) * sizeof(TILE_CSR_PTR_TYPE));
    memset(matrix->tile_csr_Ptr, 0, (matrix->numtile * tile_size_n) * sizeof(TILE_CSR_PTR_TYPE));

    matrix->tile_csr_Col = (TILE_CSR_COL_TYPE_B *)malloc(matrix->nnz * sizeof(TILE_CSR_COL_TYPE_B));
    memset(matrix->tile_csr_Col, 0, matrix->nnz * sizeof(TILE_CSR_COL_TYPE_B));

    matrix->tile_csr_Value = (MAT_VAL_TYPE *)malloc(matrix->nnz * sizeof(MAT_VAL_TYPE));
    memset(matrix->tile_csr_Value, 0, matrix->nnz * sizeof(MAT_VAL_TYPE));

    matrix->mask = (TILE_MASK_TYPE_B *)malloc(matrix->numtile * tile_size_n * tile_size_m / MaskBitsB * sizeof(TILE_MASK_TYPE_B));
    memset(matrix->mask, 0, matrix->numtile * tile_size_n * tile_size_m / MaskBitsB * sizeof(TILE_MASK_TYPE_B));


#pragma omp parallel for
    for (int blki = 0; blki < matrix->tilem; blki++)
    {
        quick_sort_key(matrix->tile_columnidx + matrix->tile_ptr[blki], matrix->tile_ptr[blki + 1] - matrix->tile_ptr[blki]);
    }

    // Step: Count dense tiles and allocate dense_data space for Matrix B
    // Calculate threshold for dense tile (nnz >= threshold * tile_size_n * tile_size_m)
    int tile_total_elems_B = tile_size_n * tile_size_m;
    int dense_threshold_B = (int)((float)(TILE_DENSE_THRESHOLD) / 10 * (float)tile_total_elems_B);
    
    // Allocate tile_dense_ready array, initialize to -1 (not dense)
    matrix->tile_dense_ready = (int *)malloc(matrix->numtile * sizeof(int));
    for (int i = 0; i < matrix->numtile; i++)
    {
        matrix->tile_dense_ready[i] = -1;
    }
    
    // Count dense tiles and set tile_dense_ready
    matrix->dense_tile_count = 0;
    for (int i = 0; i < matrix->numtile; i++)
    {
        int tile_nnz = matrix->tile_nnz[i + 1] - matrix->tile_nnz[i];
        if (tile_nnz >= dense_threshold_B)
        {
            // This is a dense tile, record its offset in dense_data
            matrix->tile_dense_ready[i] = matrix->dense_tile_count * tile_total_elems_B;
            matrix->dense_tile_count++;
        }
    }
    
    // Allocate dense_data only for dense tiles
    if (matrix->dense_tile_count > 0)
    {
        matrix->dense_data = (MAT_VAL_TYPE *)malloc(matrix->dense_tile_count * tile_total_elems_B * sizeof(MAT_VAL_TYPE));
        memset(matrix->dense_data, 0, matrix->dense_tile_count * tile_total_elems_B * sizeof(MAT_VAL_TYPE));
    }
    else
    {
        matrix->dense_data = (MAT_VAL_TYPE *)malloc(1 * tile_total_elems_B * sizeof(MAT_VAL_TYPE));
        memset(matrix->dense_data, 0, 1 * tile_total_elems_B * sizeof(MAT_VAL_TYPE));
    }

    for (int blki = 0; blki < matrix->tilen; blki++)
    {
        int colbnum = matrix->csc_tile_ptr[blki + 1] - matrix->csc_tile_ptr[blki];
        SMatrixB *subrowmatrixB_trans = (SMatrixB *)malloc(colbnum * sizeof(SMatrixB));

        int rowlength = blki == matrix->tilen - 1 ? matrix->n - (matrix->tilen - 1) * tile_size_m : tile_size_m;

        int start = blki * tile_size_m;
        int end = blki == matrix->tilen - 1 ? matrix->n : (blki + 1) * tile_size_m;

        for (int bi = 0; bi < colbnum; bi++)
        {
            int tile_id = matrix->csc_tile_ptr[blki] + bi;
            int tilennz = matrix->tile_nnz[tile_id + 1] - matrix->tile_nnz[tile_id];
            subrowmatrixB_trans[bi].value = (MAT_VAL_TYPE *)malloc(tilennz * sizeof(MAT_VAL_TYPE));
            subrowmatrixB_trans[bi].columnindex = (int *)malloc(tilennz * sizeof(int));

            subrowmatrixB_trans[bi].rowpointer = (MAT_PTR_TYPE *)malloc((rowlength + 1) * sizeof(MAT_PTR_TYPE));
            memset(subrowmatrixB_trans[bi].rowpointer, 0, (rowlength + 1) * sizeof(MAT_PTR_TYPE));
        }

        int *num = (int *)malloc((colbnum) * sizeof(int));
        memset(num, 0, (colbnum) * sizeof(int));

        for (int ri = 0; ri < rowlength; ri++)
        {
            for (int j = cscColPtrB[start + ri]; j < cscColPtrB[start + ri + 1]; j++)
            {
                int ki;
                for (int k = matrix->csc_tile_ptr[blki], ki = 0; k < matrix->csc_tile_ptr[blki + 1], ki < colbnum; k++, ki++)
                {
                    int kcstart = matrix->csc_tile_rowidx[k] * tile_size_n;
                    int kcend = matrix->csc_tile_rowidx[k] == (matrix->m - 1) ? matrix->m : (matrix->csc_tile_rowidx[k] + 1) * tile_size_n;
                    if (cscRowIdxB[j] >= kcstart && cscRowIdxB[j] < kcend)
                    {
                        num[ki]++;
                        subrowmatrixB_trans[ki].value[num[ki] - 1] = cscValB[j];
                        subrowmatrixB_trans[ki].columnindex[num[ki] - 1] = cscRowIdxB[j] - matrix->csc_tile_rowidx[k] * tile_size_n;
                        break;
                    }
                }
            }
            for (int bi = 0; bi < colbnum; bi++)
            {
                subrowmatrixB_trans[bi].rowpointer[ri + 1] = num[bi];
            }
        }
        //transpose submatrix
        SMatrixB *subrowmatrixB = (SMatrixB *)malloc(colbnum * sizeof(SMatrixB));
        for (int bi = 0; bi < colbnum; bi++)
        {
            int tileid = matrix->csc_tile_ptr[blki] + bi;
            int tilennz = matrix->tile_nnz[tileid + 1] - matrix->tile_nnz[tileid];
            int collength = matrix->csc_tile_rowidx[tileid] == matrix->tilem - 1 ? matrix->m - (matrix->tilem - 1) * tile_size_n : tile_size_n;
            subrowmatrixB[bi].value = (MAT_VAL_TYPE *)malloc((tilennz) * sizeof(MAT_VAL_TYPE));
            subrowmatrixB[bi].columnindex = (int *)malloc((tilennz) * sizeof(int));

            subrowmatrixB[bi].rowpointer = (MAT_PTR_TYPE *)malloc((collength + 1) * sizeof(MAT_PTR_TYPE));
            memset(subrowmatrixB[bi].rowpointer, 0, (collength + 1) * sizeof(MAT_PTR_TYPE));
        }
        for (int bi = 0; bi < colbnum; bi++)
        {
            int tileid = matrix->csc_tile_ptr[blki] + bi;
            int tilennz = matrix->tile_nnz[tileid + 1] - matrix->tile_nnz[tileid];
            int collength = matrix->csc_tile_rowidx[tileid] == matrix->tilem - 1 ? matrix->m - (matrix->tilem - 1) * tile_size_n : tile_size_n;
            matrix_transposition(rowlength, collength, tilennz,
                                 subrowmatrixB_trans[bi].rowpointer, subrowmatrixB_trans[bi].columnindex, subrowmatrixB_trans[bi].value,
                                 subrowmatrixB[bi].columnindex, subrowmatrixB[bi].rowpointer, subrowmatrixB[bi].value);
        }
        for (int bi = 0; bi < colbnum; bi++)
        {
            int tileid = matrix->csc_tile_ptr[blki] + bi;
            int tilennz = matrix->tile_nnz[tileid + 1] - matrix->tile_nnz[tileid];
            int prennz = matrix->tile_nnz[tileid];
            int collength = matrix->csc_tile_rowidx[tileid] == matrix->tilem - 1 ? matrix->m - (matrix->tilem - 1) * tile_size_n : tile_size_n;
            //CSR val&col
            for (int bri = 0; bri < collength; bri++)
            {
                for (int k = subrowmatrixB[bi].rowpointer[bri]; k < subrowmatrixB[bi].rowpointer[bri + 1]; k++)
                {
                    int colidx = subrowmatrixB[bi].columnindex[k];
                    MAT_VAL_TYPE val = subrowmatrixB[bi].value[k];
                    matrix->tile_csr_Value[prennz + k] = val;
                    assert((tile_size_m % MaskBitsB) == 0);
                    int stride = colidx / MaskBitsB;
                    matrix->mask[tileid * tile_size_n * (tile_size_m / MaskBitsB) + bri * (tile_size_m / MaskBitsB) + stride] |= (0x1ULL << (MaskBitsB - colidx % MaskBitsB - 1));
                    matrix->tile_csr_Col[prennz + k] = colidx;

                    // Only write to dense_data if this tile is dense (tile_dense_ready >= 0)
                    if (matrix->tile_dense_ready[tileid] >= 0)
                    {
                        // bri is the row index in the tile
                        // colidx is the column index in the tile
                        int dense_offset = matrix->tile_dense_ready[tileid];
                        long long internal_offset = (long long)bri * tile_size_m + colidx;
                        
                        matrix->dense_data[dense_offset + internal_offset] = val;
                    }
                }
                matrix->tile_csr_Ptr[tileid * tile_size_n + bri] = subrowmatrixB[bi].rowpointer[bri];
            }

            for (int jid = collength; jid < tile_size_n; jid++)
            {
                matrix->tile_csr_Ptr[tileid * tile_size_n + jid] = subrowmatrixB[bi].rowpointer[collength];
            }
        }
        for (int bi = 0; bi < colbnum; bi++)
        {
            free(subrowmatrixB[bi].value);
            free(subrowmatrixB[bi].columnindex);
            free(subrowmatrixB[bi].rowpointer);
            free(subrowmatrixB_trans[bi].value);
            free(subrowmatrixB_trans[bi].columnindex);
            free(subrowmatrixB_trans[bi].rowpointer);
        }
        free(subrowmatrixB);
        free(subrowmatrixB_trans);
        free(num);
    }
    int tile_id = 3;
    for (int ci = 0; ci < 4; ci++)
    {
        for (int stride = 0; stride < tile_size_m / MaskBitsB; stride++){
            DEBUG_PRINT("[CSR2TILE B MASK CHECK] %d row %d bitmask in %d tile: %04x\n", ci, stride, tile_id, matrix->mask[
                tile_id * tile_size_n * (tile_size_m / MaskBitsB) + ci * (tile_size_m / MaskBitsB) + stride]);
        }
    }
}


void matrix_destroy(SMatrixA *matrix)
{
    free(matrix->tile_ptr);
    free(matrix->tile_columnidx);
    free(matrix->tile_nnz);
    free(matrix->tile_csr_Value);
    free(matrix->tile_csr_Col);
    free(matrix->tile_csr_Ptr);
    free(matrix->mask);
    if (matrix->dense_data != NULL)
    {
        free(matrix->dense_data);
        matrix->dense_data = NULL;
    }
    if (matrix->tile_dense_ready != NULL)
    {
        free(matrix->tile_dense_ready);
        matrix->tile_dense_ready = NULL;
    }
}

void matrix_destroy_B(SMatrixB *matrix)
{
    free(matrix->tile_ptr);
    free(matrix->tile_columnidx);
    free(matrix->tile_nnz);
    free(matrix->tile_csr_Value);
    free(matrix->tile_csr_Col);
    free(matrix->tile_csr_Ptr);
    free(matrix->mask);
    if (matrix->dense_data != NULL)
    {
        free(matrix->dense_data);
        matrix->dense_data = NULL;
    }
    if (matrix->tile_dense_ready != NULL)
    {
        free(matrix->tile_dense_ready);
        matrix->tile_dense_ready = NULL;
    }
}
