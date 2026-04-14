/**
 * =============================================================================
 * tile_probe.cpp - Multi-size Tile Lightweight Probe Program
 *
 * Function: Traverse the input matrix once to collect statistics for 7 tile sizes simultaneously
 * Purpose: Collect training features for LightGBM tile size selector
 *
 * Compile: make
 * Run: ./tile_probe <matrix.mtx>
 * =============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <sys/time.h>
#include <omp.h>

// ============================================================================
// Type definitions
// ============================================================================
typedef int MAT_PTR_TYPE;
typedef double MAT_VAL_TYPE;

// ============================================================================
// Matrix Market format reader (simplified version)
// ============================================================================
#include "mmio.h"

// ============================================================================
// Timing utilities
// ============================================================================
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

// ============================================================================
// Tile size configuration
// ============================================================================
#define NUM_TILE_SIZES 9

const int TILE_SIZES[NUM_TILE_SIZES][2] = {
    {8, 8},     // 0
    {16, 8},    // 1
    {8, 16},    // 2
    {16, 16},   // 3
    {16, 32},   // 4
    {32, 16},   // 5
    {32, 32},   // 6
    {8, 32},    // 7
    {32, 8}     // 8
};

// ============================================================================
// Tile probe result structure
// ============================================================================
typedef struct {
    int tile_m;
    int tile_n;
    int tile_cap;     // tile_m * tile_n
    int tilem;        // ceil(m / tile_m)
    int tilen;        // ceil(n / tile_n)

    // Basic statistics
    long long numtile;          // total non-empty tiles
    int *tiles_per_row;         // non-empty tiles per tile row
    int *tiles_per_col;         // [v2] non-empty tiles per tile column

    // nnz statistics within tiles
    long long sum_nnz;          // Σ(tile_nnz)
    long long sum_nnz_sq;       // Σ(tile_nnz²)
    int max_nnz;
    int min_nnz;

    // tile nnz histogram (8 bins)
    int hist_1;           // nnz == 1
    int hist_2_4;         // nnz ∈ [2, 4)
    int hist_4_8;         // nnz ∈ [4, 8)
    int hist_8_16;        // nnz ∈ [8, 16)
    int hist_16_32;       // nnz ∈ [16, 32)
    int hist_32_64;       // nnz ∈ [32, 64)
    int hist_64_128;      // nnz ∈ [64, 128)
    int hist_128_plus;    // nnz ≥ 128
} TileProbe;

// ============================================================================
// Initialize probe
// ============================================================================
void init_probe(TileProbe *probe, int tile_m, int tile_n, int m, int n) {
    probe->tile_m = tile_m;
    probe->tile_n = tile_n;
    probe->tile_cap = tile_m * tile_n;
    probe->tilem = (m + tile_m - 1) / tile_m;
    probe->tilen = (n + tile_n - 1) / tile_n;

    probe->numtile = 0;
    probe->tiles_per_row = (int *)calloc(probe->tilem, sizeof(int));
    probe->tiles_per_col = (int *)calloc(probe->tilen, sizeof(int));  // [v2]

    probe->sum_nnz = 0;
    probe->sum_nnz_sq = 0;
    probe->max_nnz = 0;
    probe->min_nnz = INT_MAX;

    probe->hist_1 = 0;
    probe->hist_2_4 = 0;
    probe->hist_4_8 = 0;
    probe->hist_8_16 = 0;
    probe->hist_16_32 = 0;
    probe->hist_32_64 = 0;
    probe->hist_64_128 = 0;
    probe->hist_128_plus = 0;
}

// ============================================================================
// Free probe resources
// ============================================================================
void free_probe(TileProbe *probe) {
    if (probe->tiles_per_row) {
        free(probe->tiles_per_row);
        probe->tiles_per_row = NULL;
    }
    if (probe->tiles_per_col) {
        free(probe->tiles_per_col);
        probe->tiles_per_col = NULL;
    }
}

// ============================================================================
// Core: Multi-size parallel probe (deeply optimized version)
//
// Optimization points:
//   1. Bit shift instead of division: tile dimensions are powers of 2, col>>3/4/5 instead of col/8/16/32
//   2. Fully unrolled inner loop: only compute 3 tile_col types (shift=3,4,5), avoid 7 iterations
//   3. Group by tile_m into 3 boundary checks (8/16/32), instead of 7 per-size checks
//   4. __builtin_clz replaces 7-level if-else for histogram binning
//   5. Pre-allocate thread-local memory to avoid malloc contention in parallel region
//   6. Auto-limit thread count to avoid memory/scheduling overhead from too many threads
//   7. 3 shared counters (tile_n=8/16/32) instead of 7 independent counters,
//      inner hot loop only does 3 updates, results distributed to 7 sizes via hierarchical settlement
// ============================================================================

// Settlement macro: statistics of a tile's nnz to corresponding size's statistics
// [v2] Added TCOL parameter to track tiles_per_col
#define SETTLE_TILE(NT, TROW, TCOL, C) do { \
    probes[NT].tiles_per_row[TROW]++; \
    l_tpc[NT][TCOL]++; \
    l_nt[NT]++; \
    l_sn[NT] += (C); \
    l_sq[NT] += (long long)(C) * (C); \
    if ((C) > l_mx[NT]) l_mx[NT] = (C); \
    if ((C) < l_mn[NT]) l_mn[NT] = (C); \
    int _b = 31 - __builtin_clz(C); \
    if (_b > 7) _b = 7; \
    l_h[NT][_b]++; \
} while (0)

void probe_all_tile_sizes(
    int m, int n, long long nnz,
    MAT_PTR_TYPE *rowptr, int *colidx,
    TileProbe probes[NUM_TILE_SIZES])
{
    // p0=8x8, p1=16x8, p2=8x16, p3=16x16, p4=16x32, p5=32x16, p6=32x32
    for (int p = 0; p < NUM_TILE_SIZES; p++) {
        init_probe(&probes[p], TILE_SIZES[p][0], TILE_SIZES[p][1], m, n);
    }

    const int BLOCK = 32;
    int nblk = (m + BLOCK - 1) / BLOCK;

    // tilen for the 3 unique tile_n values
    int tilen8  = probes[0].tilen;   // (n+7)/8
    int tilen16 = probes[3].tilen;   // (n+15)/16
    int tilen32 = probes[6].tilen;   // (n+31)/32

    // Adaptive thread count: too many threads cause memory overhead that slows down
    int max_threads = omp_get_max_threads();
    int nthreads = max_threads;
    if (nthreads > 16) nthreads = 16;
    // Ensure enough blocks for each thread to have work
    if (nblk < nthreads * 2) nthreads = (nblk + 1) / 2;
    if (nthreads < 1) nthreads = 1;

    // --- Pre-allocate all thread-local memory (outside parallel region to avoid malloc contention) ---
    // Each thread needs:
    //   3 shared counters cnt8/cnt16/cnt32 + corresponding touched lists
    //   4 accumulators acc1/acc3/acc5/acc6 + corresponding touched lists
    typedef struct {
        int *cnt8, *cnt16, *cnt32;           // shared counters (per tile_n)
        int *tch8, *tch16, *tch32;           // shared touched lists
        int *acc1, *acc3, *acc5, *acc6;      // hierarchical accumulators
        int *tch_a1, *tch_a3, *tch_a5, *tch_a6; // accumulator touched
        int *acc_p4, *tch_ap4;               // cnt32→p7(8x32) then accumulate→p4(16x32)
        int *acc_p8, *tch_ap8;               // acc1→p1(16x8) then accumulate→p8(32x8)
        int *l_tpc[NUM_TILE_SIZES];          // [v2] thread-local tiles_per_col
    } ThreadBuf;

    ThreadBuf *tbufs = (ThreadBuf *)malloc(nthreads * sizeof(ThreadBuf));
    for (int t = 0; t < nthreads; t++) {
        tbufs[t].cnt8  = (int *)calloc(tilen8,  sizeof(int));
        tbufs[t].cnt16 = (int *)calloc(tilen16, sizeof(int));
        tbufs[t].cnt32 = (int *)calloc(tilen32, sizeof(int));
        tbufs[t].tch8  = (int *)malloc(tilen8  * sizeof(int));
        tbufs[t].tch16 = (int *)malloc(tilen16 * sizeof(int));
        tbufs[t].tch32 = (int *)malloc(tilen32 * sizeof(int));
        tbufs[t].acc1  = (int *)calloc(tilen8,  sizeof(int));
        tbufs[t].acc3  = (int *)calloc(tilen16, sizeof(int));
        tbufs[t].acc5  = (int *)calloc(tilen16, sizeof(int));
        tbufs[t].acc6  = (int *)calloc(tilen32, sizeof(int));
        tbufs[t].tch_a1 = (int *)malloc(tilen8  * sizeof(int));
        tbufs[t].tch_a3 = (int *)malloc(tilen16 * sizeof(int));
        tbufs[t].tch_a5 = (int *)malloc(tilen16 * sizeof(int));
        tbufs[t].tch_a6 = (int *)malloc(tilen32 * sizeof(int));
        tbufs[t].acc_p4  = (int *)calloc(tilen32, sizeof(int));
        tbufs[t].tch_ap4 = (int *)malloc(tilen32 * sizeof(int));
        tbufs[t].acc_p8  = (int *)calloc(tilen8,  sizeof(int));
        tbufs[t].tch_ap8 = (int *)malloc(tilen8  * sizeof(int));
        // [v2] Allocate thread-local tiles_per_col
        for (int p = 0; p < NUM_TILE_SIZES; p++)
            tbufs[t].l_tpc[p] = (int *)calloc(probes[p].tilen, sizeof(int));
    }

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        ThreadBuf *tb = &tbufs[tid];

        // Local references
        int *cnt8 = tb->cnt8, *cnt16 = tb->cnt16, *cnt32 = tb->cnt32;
        int *tch8 = tb->tch8, *tch16 = tb->tch16, *tch32 = tb->tch32;
        int ntch8 = 0, ntch16 = 0, ntch32 = 0;

        int *acc1 = tb->acc1, *acc3 = tb->acc3, *acc5 = tb->acc5, *acc6 = tb->acc6;
        int *tch_a1 = tb->tch_a1, *tch_a3 = tb->tch_a3;
        int *tch_a5 = tb->tch_a5, *tch_a6 = tb->tch_a6;
        int ntch_a1 = 0, ntch_a3 = 0, ntch_a5 = 0, ntch_a6 = 0;

        int *acc_p4 = tb->acc_p4, *tch_ap4 = tb->tch_ap4;
        int *acc_p8 = tb->acc_p8, *tch_ap8 = tb->tch_ap8;
        int ntch_ap4 = 0, ntch_ap8 = 0;

        // [v2] Thread-local tiles_per_col references
        int **l_tpc = tb->l_tpc;

        // Thread-local statistics
        long long l_nt[NUM_TILE_SIZES] = {};
        long long l_sn[NUM_TILE_SIZES] = {};
        long long l_sq[NUM_TILE_SIZES] = {};
        int       l_mx[NUM_TILE_SIZES], l_mn[NUM_TILE_SIZES];
        int       l_h[NUM_TILE_SIZES][8] = {};
        for (int p = 0; p < NUM_TILE_SIZES; p++) {
            l_mx[p] = 0; l_mn[p] = INT_MAX;
        }

        #pragma omp for schedule(static)
        for (int blk = 0; blk < nblk; blk++) {
            int rs = blk << 5;
            int re = rs + BLOCK;
            if (re > m) re = m;

            for (int i = rs; i < re; i++) {
                // ====== Hot loop: only 3 counter updates ======
                MAT_PTR_TYPE je = rowptr[i + 1];
                for (MAT_PTR_TYPE j = rowptr[i]; j < je; j++) {
                    int col = colidx[j];
                    int c3 = col >> 3;
                    int c4 = col >> 4;
                    int c5 = col >> 5;

                    if (!cnt8[c3])  tch8[ntch8++]   = c3;
                    cnt8[c3]++;
                    if (!cnt16[c4]) tch16[ntch16++]  = c4;
                    cnt16[c4]++;
                    if (!cnt32[c5]) tch32[ntch32++]  = c5;
                    cnt32[c5]++;
                }

                // ====== Hierarchical settlement ======
                int il = i & 31;
                int last = (i + 1 >= re);

                // --- Every 8 rows: settle cnt8 → p0(8x8), accumulate → p1(16x8) ---
                //                    settle cnt16 → p2(8x16), accumulate → p3(16x16), p5(32x16) ---
                //                    settle cnt32 → p7(8x32), accumulate → acc_p4 for p4(16x32) ---
                if ((il & 7) == 7 || last) {
                    int tr8 = i >> 3;

                    // cnt8 → p0 direct settle, p1 accumulate
                    for (int t = 0; t < ntch8; t++) {
                        int tc = tch8[t];
                        int c  = cnt8[tc];
                        SETTLE_TILE(0, tr8, tc, c);              // p0: 8x8
                        if (!acc1[tc]) tch_a1[ntch_a1++] = tc;
                        acc1[tc] += c;                       // p1: 16x8 accumulate
                        cnt8[tc] = 0;
                    }
                    ntch8 = 0;

                    // cnt16 → p2 direct settle, p3/p5 accumulate
                    for (int t = 0; t < ntch16; t++) {
                        int tc = tch16[t];
                        int c  = cnt16[tc];
                        SETTLE_TILE(2, tr8, tc, c);              // p2: 8x16
                        if (!acc3[tc]) tch_a3[ntch_a3++] = tc;
                        acc3[tc] += c;                       // p3: 16x16 accumulate
                        if (!acc5[tc]) tch_a5[ntch_a5++] = tc;
                        acc5[tc] += c;                       // p5: 32x16 accumulate
                        cnt16[tc] = 0;
                    }
                    ntch16 = 0;

                    // cnt32 → p7 direct settle, acc_p4 accumulate (for p4: 16x32)
                    for (int t = 0; t < ntch32; t++) {
                        int tc = tch32[t];
                        int c  = cnt32[tc];
                        SETTLE_TILE(7, tr8, tc, c);              // p7: 8x32
                        if (!acc_p4[tc]) tch_ap4[ntch_ap4++] = tc;
                        acc_p4[tc] += c;                     // p4: 16x32 accumulate
                        cnt32[tc] = 0;
                    }
                    ntch32 = 0;
                }

                // --- Every 16 rows: settle acc1 → p1, accumulate → acc_p8 for p8(32x8) ---
                //                    settle acc3 → p3 ---
                //                    settle acc_p4 → p4(16x32), accumulate → p6(32x32) ---
                if ((il & 15) == 15 || last) {
                    int tr16 = i >> 4;

                    // acc1 → p1: 16x8, also accumulate → acc_p8 for p8(32x8)
                    for (int t = 0; t < ntch_a1; t++) {
                        int tc = tch_a1[t];
                        int c  = acc1[tc];
                        SETTLE_TILE(1, tr16, tc, c);
                        if (!acc_p8[tc]) tch_ap8[ntch_ap8++] = tc;
                        acc_p8[tc] += c;                     // p8: 32x8 accumulate
                        acc1[tc] = 0;
                    }
                    ntch_a1 = 0;

                    // acc3 → p3: 16x16
                    for (int t = 0; t < ntch_a3; t++) {
                        int tc = tch_a3[t];
                        int c  = acc3[tc];
                        SETTLE_TILE(3, tr16, tc, c);
                        acc3[tc] = 0;
                    }
                    ntch_a3 = 0;

                    // acc_p4 → p4: 16x32 direct settle, p6 accumulate
                    for (int t = 0; t < ntch_ap4; t++) {
                        int tc = tch_ap4[t];
                        int c  = acc_p4[tc];
                        SETTLE_TILE(4, tr16, tc, c);             // p4: 16x32
                        if (!acc6[tc]) tch_a6[ntch_a6++] = tc;
                        acc6[tc] += c;                       // p6: 32x32 accumulate
                        acc_p4[tc] = 0;
                    }
                    ntch_ap4 = 0;
                }

                // --- Every 32 rows: settle acc5 → p5, acc6 → p6, acc_p8 → p8 ---
                if ((il & 31) == 31 || last) {
                    int tr32 = i >> 5;

                    // acc5 → p5: 32x16
                    for (int t = 0; t < ntch_a5; t++) {
                        int tc = tch_a5[t];
                        int c  = acc5[tc];
                        SETTLE_TILE(5, tr32, tc, c);
                        acc5[tc] = 0;
                    }
                    ntch_a5 = 0;

                    // acc6 → p6: 32x32
                    for (int t = 0; t < ntch_a6; t++) {
                        int tc = tch_a6[t];
                        int c  = acc6[tc];
                        SETTLE_TILE(6, tr32, tc, c);
                        acc6[tc] = 0;
                    }
                    ntch_a6 = 0;

                    // acc_p8 → p8: 32x8
                    for (int t = 0; t < ntch_ap8; t++) {
                        int tc = tch_ap8[t];
                        int c  = acc_p8[tc];
                        SETTLE_TILE(8, tr32, tc, c);
                        acc_p8[tc] = 0;
                    }
                    ntch_ap8 = 0;
                }
            }
        } // end omp for

        // --- Merge to global ---
        #pragma omp critical
        {
            for (int p = 0; p < NUM_TILE_SIZES; p++) {
                probes[p].numtile    += l_nt[p];
                probes[p].sum_nnz    += l_sn[p];
                probes[p].sum_nnz_sq += l_sq[p];
                if (l_mx[p] > probes[p].max_nnz) probes[p].max_nnz = l_mx[p];
                if (l_mn[p] < probes[p].min_nnz) probes[p].min_nnz = l_mn[p];
                probes[p].hist_1        += l_h[p][0];
                probes[p].hist_2_4      += l_h[p][1];
                probes[p].hist_4_8      += l_h[p][2];
                probes[p].hist_8_16     += l_h[p][3];
                probes[p].hist_16_32    += l_h[p][4];
                probes[p].hist_32_64    += l_h[p][5];
                probes[p].hist_64_128   += l_h[p][6];
                probes[p].hist_128_plus += l_h[p][7];
            }
        }
    } // end omp parallel

    // [v2] Merge tiles_per_col (parallel merge, one thread per tile size)
    #pragma omp parallel for num_threads(NUM_TILE_SIZES < nthreads ? NUM_TILE_SIZES : nthreads)
    for (int p = 0; p < NUM_TILE_SIZES; p++) {
        int tl = probes[p].tilen;
        int *dst = probes[p].tiles_per_col;
        for (int t = 0; t < nthreads; t++) {
            int *src = tbufs[t].l_tpc[p];
            for (int c = 0; c < tl; c++) {
                dst[c] += src[c];
            }
        }
    }

    // Free pre-allocated memory
    for (int t = 0; t < nthreads; t++) {
        free(tbufs[t].cnt8);  free(tbufs[t].cnt16);  free(tbufs[t].cnt32);
        free(tbufs[t].tch8);  free(tbufs[t].tch16);  free(tbufs[t].tch32);
        free(tbufs[t].acc1);  free(tbufs[t].acc3);    free(tbufs[t].acc5);  free(tbufs[t].acc6);
        free(tbufs[t].tch_a1); free(tbufs[t].tch_a3); free(tbufs[t].tch_a5); free(tbufs[t].tch_a6);
        free(tbufs[t].acc_p4); free(tbufs[t].tch_ap4);
        free(tbufs[t].acc_p8); free(tbufs[t].tch_ap8);
        // [v2] Free thread-local tiles_per_col
        for (int p = 0; p < NUM_TILE_SIZES; p++)
            free(tbufs[t].l_tpc[p]);
    }
    free(tbufs);
}

#undef SETTLE_TILE

// ============================================================================
// 计算并打印统计特征
// ============================================================================
void print_probe_results(TileProbe *probe, int m, int n, long long nnz) {
    long long N = probe->numtile;
    if (N == 0) {
        printf("  [WARNING] No non-empty tiles!\n");
        return;
    }

    // 基础特征
    double tile_density = (double)N / ((double)probe->tilem * probe->tilen);
    double nnz_per_tile_avg = (double)probe->sum_nnz / N;
    double nnz_per_tile_var = (double)probe->sum_nnz_sq / N - nnz_per_tile_avg * nnz_per_tile_avg;
    double nnz_per_tile_std = sqrt(nnz_per_tile_var > 0 ? nnz_per_tile_var : 0);
    double nnz_per_tile_cv = (nnz_per_tile_avg > 0) ? nnz_per_tile_std / nnz_per_tile_avg : 0;
    double tile_fill_ratio_avg = nnz_per_tile_avg / probe->tile_cap;
    double tile_fill_ratio_max = (double)probe->max_nnz / probe->tile_cap;

    // tiles_per_row 统计
    double tiles_per_row_avg = (double)N / probe->tilem;
    int tiles_per_row_max = 0, tiles_per_row_min = INT_MAX;
    double tiles_per_row_sum_sq = 0;
    int empty_tile_rows = 0;

    for (int i = 0; i < probe->tilem; i++) {
        int tpr = probe->tiles_per_row[i];
        if (tpr > tiles_per_row_max) tiles_per_row_max = tpr;
        if (tpr < tiles_per_row_min) tiles_per_row_min = tpr;
        tiles_per_row_sum_sq += (double)tpr * tpr;
        if (tpr == 0) empty_tile_rows++;
    }

    double tiles_per_row_var = tiles_per_row_sum_sq / probe->tilem - tiles_per_row_avg * tiles_per_row_avg;
    double tiles_per_row_std = sqrt(tiles_per_row_var > 0 ? tiles_per_row_var : 0);
    double tiles_per_row_cv = (tiles_per_row_avg > 0) ? tiles_per_row_std / tiles_per_row_avg : 0;
    double empty_tile_row_ratio = (double)empty_tile_rows / probe->tilem;

    // [v2] tiles_per_col 统计
    double tiles_per_col_avg = (double)N / probe->tilen;
    int tiles_per_col_max = 0, tiles_per_col_min = INT_MAX;
    double tiles_per_col_sum_sq = 0;
    int empty_tile_cols = 0;

    for (int i = 0; i < probe->tilen; i++) {
        int tpc = probe->tiles_per_col[i];
        if (tpc > tiles_per_col_max) tiles_per_col_max = tpc;
        if (tpc < tiles_per_col_min) tiles_per_col_min = tpc;
        tiles_per_col_sum_sq += (double)tpc * tpc;
        if (tpc == 0) empty_tile_cols++;
    }

    double tiles_per_col_var = tiles_per_col_sum_sq / probe->tilen - tiles_per_col_avg * tiles_per_col_avg;
    double tiles_per_col_std = sqrt(tiles_per_col_var > 0 ? tiles_per_col_var : 0);
    double tiles_per_col_cv = (tiles_per_col_avg > 0) ? tiles_per_col_std / tiles_per_col_avg : 0;
    double empty_tile_col_ratio = (double)empty_tile_cols / probe->tilen;

    // 打印结果
    printf("  ├─ Grid: %d × %d tiles\n", probe->tilem, probe->tilen);
    printf("  ├─ numtile: %lld (density: %.6f%%)\n", N, tile_density * 100);
    printf("  ├─ nnz_per_tile: avg=%.2f, max=%d, min=%d, std=%.2f, cv=%.3f\n",
           nnz_per_tile_avg, probe->max_nnz, probe->min_nnz, nnz_per_tile_std, nnz_per_tile_cv);
    printf("  ├─ tile_fill_ratio: avg=%.4f, max=%.4f\n", tile_fill_ratio_avg, tile_fill_ratio_max);
    printf("  ├─ tiles_per_row: avg=%.2f, max=%d, min=%d, std=%.2f, cv=%.3f\n",
           tiles_per_row_avg, tiles_per_row_max, tiles_per_row_min, tiles_per_row_std, tiles_per_row_cv);
    printf("  ├─ empty_tile_row_ratio: %.4f\n", empty_tile_row_ratio);
    printf("  ├─ tiles_per_col: avg=%.2f, max=%d, min=%d, std=%.2f, cv=%.3f\n",
           tiles_per_col_avg, tiles_per_col_max, tiles_per_col_min, tiles_per_col_std, tiles_per_col_cv);
    printf("  ├─ empty_tile_col_ratio: %.4f\n", empty_tile_col_ratio);
    printf("  └─ nnz_histogram: [1]=%d, [2-4)=%d, [4-8)=%d, [8-16)=%d, [16-32)=%d, [32-64)=%d, [64-128)=%d, [128+]=%d\n",
           probe->hist_1, probe->hist_2_4, probe->hist_4_8, probe->hist_8_16,
           probe->hist_16_32, probe->hist_32_64, probe->hist_64_128, probe->hist_128_plus);
}

// ============================================================================
// 打印 CSV 格式的特征 (用于训练)
// ============================================================================
void print_csv_header() {
    printf("\n=== CSV Format Features ===\n");
    printf("tile_size,numtile,tile_density,nnz_per_tile_avg,nnz_per_tile_max,nnz_per_tile_min,");
    printf("nnz_per_tile_std,nnz_per_tile_cv,tile_fill_avg,tile_fill_max,");
    printf("tiles_per_row_avg,tiles_per_row_max,tiles_per_row_min,tiles_per_row_std,tiles_per_row_cv,");
    printf("empty_row_ratio,");
    printf("tiles_per_col_avg,tiles_per_col_max,tiles_per_col_min,tiles_per_col_std,tiles_per_col_cv,");
    printf("empty_col_ratio,");
    printf("hist_1,hist_2_4,hist_4_8,hist_8_16,hist_16_32,hist_32_64,hist_64_128,hist_128_plus,");
    printf("nnz_per_row_max,nnz_per_row_std,nnz_per_row_skewness,");
    printf("nnz_per_col_max,nnz_per_col_avg,nnz_per_col_std\n");
}

// [v2] 全局特征结构体
typedef struct {
    int    nnz_per_row_max;
    double nnz_per_row_std;
    double nnz_per_row_skewness;
    int    nnz_per_col_max;
    double nnz_per_col_avg;
    double nnz_per_col_std;
} GlobalFeatures;

void print_csv_row(TileProbe *probe, GlobalFeatures *gf) {
    long long N = probe->numtile;
    if (N == 0) return;

    double tile_density = (double)N / ((double)probe->tilem * probe->tilen);
    double nnz_per_tile_avg = (double)probe->sum_nnz / N;
    double nnz_per_tile_var = (double)probe->sum_nnz_sq / N - nnz_per_tile_avg * nnz_per_tile_avg;
    double nnz_per_tile_std = sqrt(nnz_per_tile_var > 0 ? nnz_per_tile_var : 0);
    double nnz_per_tile_cv = (nnz_per_tile_avg > 0) ? nnz_per_tile_std / nnz_per_tile_avg : 0;
    double tile_fill_avg = nnz_per_tile_avg / probe->tile_cap;
    double tile_fill_max = (double)probe->max_nnz / probe->tile_cap;

    double tiles_per_row_avg = (double)N / probe->tilem;
    int tiles_per_row_max = 0, tiles_per_row_min = INT_MAX;
    double tiles_per_row_sum_sq = 0;
    int empty_rows = 0;

    for (int i = 0; i < probe->tilem; i++) {
        int tpr = probe->tiles_per_row[i];
        if (tpr > tiles_per_row_max) tiles_per_row_max = tpr;
        if (tpr < tiles_per_row_min) tiles_per_row_min = tpr;
        tiles_per_row_sum_sq += (double)tpr * tpr;
        if (tpr == 0) empty_rows++;
    }

    double tiles_per_row_var = tiles_per_row_sum_sq / probe->tilem - tiles_per_row_avg * tiles_per_row_avg;
    double tiles_per_row_std = sqrt(tiles_per_row_var > 0 ? tiles_per_row_var : 0);
    double tiles_per_row_cv = (tiles_per_row_avg > 0) ? tiles_per_row_std / tiles_per_row_avg : 0;
    double empty_row_ratio = (double)empty_rows / probe->tilem;

    // [v2] tiles_per_col 统计
    double tiles_per_col_avg = (double)N / probe->tilen;
    int tiles_per_col_max = 0, tiles_per_col_min = INT_MAX;
    double tiles_per_col_sum_sq = 0;
    int empty_cols = 0;

    for (int i = 0; i < probe->tilen; i++) {
        int tpc = probe->tiles_per_col[i];
        if (tpc > tiles_per_col_max) tiles_per_col_max = tpc;
        if (tpc < tiles_per_col_min) tiles_per_col_min = tpc;
        tiles_per_col_sum_sq += (double)tpc * tpc;
        if (tpc == 0) empty_cols++;
    }

    double tiles_per_col_var = tiles_per_col_sum_sq / probe->tilen - tiles_per_col_avg * tiles_per_col_avg;
    double tiles_per_col_std = sqrt(tiles_per_col_var > 0 ? tiles_per_col_var : 0);
    double tiles_per_col_cv = (tiles_per_col_avg > 0) ? tiles_per_col_std / tiles_per_col_avg : 0;
    double empty_col_ratio = (double)empty_cols / probe->tilen;

    printf("%dx%d,%lld,%.6f,%.4f,%d,%d,%.4f,%.4f,%.6f,%.6f,",
           probe->tile_m, probe->tile_n, N, tile_density,
           nnz_per_tile_avg, probe->max_nnz, probe->min_nnz,
           nnz_per_tile_std, nnz_per_tile_cv, tile_fill_avg, tile_fill_max);
    printf("%.4f,%d,%d,%.4f,%.4f,%.6f,",
           tiles_per_row_avg, tiles_per_row_max, tiles_per_row_min,
           tiles_per_row_std, tiles_per_row_cv, empty_row_ratio);
    printf("%.4f,%d,%d,%.4f,%.4f,%.6f,",
           tiles_per_col_avg, tiles_per_col_max, tiles_per_col_min,
           tiles_per_col_std, tiles_per_col_cv, empty_col_ratio);
    printf("%d,%d,%d,%d,%d,%d,%d,%d,",
           probe->hist_1, probe->hist_2_4, probe->hist_4_8, probe->hist_8_16,
           probe->hist_16_32, probe->hist_32_64, probe->hist_64_128, probe->hist_128_plus);
    printf("%d,%.4f,%.4f,%d,%.4f,%.4f\n",
           gf->nnz_per_row_max, gf->nnz_per_row_std, gf->nnz_per_row_skewness,
           gf->nnz_per_col_max, gf->nnz_per_col_avg, gf->nnz_per_col_std);
}

// ============================================================================
// 打印跨尺寸比较特征
// ============================================================================
void print_cross_size_features(TileProbe probes[NUM_TILE_SIZES]) {
    printf("\n=== Cross-Size Comparison Features ===\n");

    // 找到 8x8, 16x16, 32x32 的索引
    int idx_8x8 = 0, idx_16x16 = 3, idx_32x32 = 6;

    if (probes[idx_16x16].numtile > 0 && probes[idx_8x8].numtile > 0) {
        double ratio_8_16 = (double)probes[idx_8x8].numtile / probes[idx_16x16].numtile;
        printf("numtile_8x8 / numtile_16x16 = %.4f (theoretical: 4.0)\n", ratio_8_16);
    }

    if (probes[idx_32x32].numtile > 0 && probes[idx_16x16].numtile > 0) {
        double ratio_16_32 = (double)probes[idx_16x16].numtile / probes[idx_32x32].numtile;
        printf("numtile_16x16 / numtile_32x32 = %.4f (theoretical: 4.0)\n", ratio_16_32);
    }

    if (probes[idx_32x32].numtile > 0 && probes[idx_8x8].numtile > 0) {
        double ratio_8_32 = (double)probes[idx_8x8].numtile / probes[idx_32x32].numtile;
        printf("numtile_8x8 / numtile_32x32 = %.4f (theoretical: 16.0)\n", ratio_8_32);
    }

    // 密度比值
    double density_8 = (double)probes[idx_8x8].numtile / (probes[idx_8x8].tilem * probes[idx_8x8].tilen);
    double density_32 = (double)probes[idx_32x32].numtile / (probes[idx_32x32].tilem * probes[idx_32x32].tilen);
    if (density_32 > 0) {
        printf("density_8x8 / density_32x32 = %.4f\n", density_8 / density_32);
    }
}

// ============================================================================
// 加载 MTX 矩阵文件
// ============================================================================
int load_mtx(const char *filename, int *m, int *n, long long *nnz, int *isSymmetric,
             MAT_PTR_TYPE **rowptr, int **colidx) {

    int m_tmp, n_tmp;
    int nnz_mtx_report;
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    if ((f = fopen(filename, "r")) == NULL) {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }

    if (mm_read_banner(f, &matcode) != 0) {
        printf("Error: Could not process Matrix Market banner.\n");
        return -2;
    }

    if (mm_is_pattern(matcode)) isPattern = 1;
    if (mm_is_real(matcode)) isReal = 1;
    if (mm_is_complex(matcode)) isComplex = 1;
    if (mm_is_integer(matcode)) isInteger = 1;

    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0) {
        printf("Error: Could not read matrix size.\n");
        return -4;
    }

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) {
        isSymmetric_tmp = 1;
    }

    // 临时存储
    MAT_PTR_TYPE *csrRowPtr_counter = (MAT_PTR_TYPE *)calloc(m_tmp + 1, sizeof(MAT_PTR_TYPE));
    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));

    // 读取数据
    for (int i = 0; i < nnz_mtx_report; i++) {
        int idxi, idxj;
        double fval;
        int ival;

        if (isReal) {
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        } else if (isComplex) {
            double fval_im;
            fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        } else if (isInteger) {
            fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
        } else if (isPattern) {
            fscanf(f, "%d %d\n", &idxi, &idxj);
        }

        idxi--;
        idxj--;

        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
    }

    fclose(f);

    // 对称矩阵处理
    if (isSymmetric_tmp) {
        for (int i = 0; i < nnz_mtx_report; i++) {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i]) {
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
        }
    }

    // exclusive scan
    int old_val = csrRowPtr_counter[0];
    csrRowPtr_counter[0] = 0;
    for (int i = 1; i <= m_tmp; i++) {
        int new_val = csrRowPtr_counter[i];
        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i - 1];
        old_val = new_val;
    }

    long long nnz_tmp = csrRowPtr_counter[m_tmp];

    // 分配最终数组
    MAT_PTR_TYPE *csrRowPtr_alias = (MAT_PTR_TYPE *)malloc((m_tmp + 1) * sizeof(MAT_PTR_TYPE));
    int *csrColIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));

    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));
    memset(csrRowPtr_counter, 0, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));

    // 填充 CSR
    if (isSymmetric_tmp) {
        for (int i = 0; i < nnz_mtx_report; i++) {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i]) {
                MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            } else {
                MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    } else {
        for (int i = 0; i < nnz_mtx_report; i++) {
            MAT_PTR_TYPE offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }

    // 返回结果
    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;
    *rowptr = csrRowPtr_alias;
    *colidx = csrColIdx_alias;

    // 清理
    free(csrRowPtr_counter);
    free(csrRowIdx_tmp);
    free(csrColIdx_tmp);

    return 0;
}

// ============================================================================
// 主函数
// ============================================================================
int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <matrix.mtx>\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];

    printf("============================================================\n");
    printf("  Multi-Size Tile Probe for AdaSpGEMM\n");
    printf("============================================================\n");
    printf("Matrix file: %s\n\n", filename);

    // ========================================================================
    // 加载矩阵
    // ========================================================================
    double t_start, t_end;

    int m, n, isSymmetric;
    long long nnz;
    MAT_PTR_TYPE *rowptr = NULL;
    int *colidx = NULL;

    t_start = get_time_ms();
    int ret = load_mtx(filename, &m, &n, &nnz, &isSymmetric, &rowptr, &colidx);
    t_end = get_time_ms();

    if (ret != 0) {
        printf("Error loading matrix!\n");
        return 1;
    }

    printf("=== Matrix Info ===\n");
    printf("  Rows (m): %d\n", m);
    printf("  Cols (n): %d\n", n);
    printf("  Nonzeros (nnz): %lld\n", nnz);
    printf("  Symmetric: %s\n", isSymmetric ? "Yes" : "No");
    printf("  Density: %.6e (%.6f%%)\n", (double)nnz / ((double)m * n), (double)nnz / ((double)m * n) * 100);
    printf("  Avg nnz/row: %.2f\n", (double)nnz / m);
    printf("  Load time: %.2f ms\n\n", t_end - t_start);

    // ========================================================================
    // 执行多尺寸探测
    // ========================================================================
    TileProbe probes[NUM_TILE_SIZES];

    printf("=== Running Multi-Size Tile Probe ===\n");
    t_start = get_time_ms();

    probe_all_tile_sizes(m, n, nnz, rowptr, colidx, probes);

    t_end = get_time_ms();
    printf("Probe time: %.2f ms\n\n", t_end - t_start);

    // ========================================================================
    // [v2] 计算矩阵级全局特征 (nnz_per_row / nnz_per_col 分布统计)
    // ========================================================================
    double nnz_per_row_avg = (double)nnz / m;
    int    nnz_per_row_max = 0;
    double nnz_per_row_sum_sq = 0;
    double nnz_per_row_sum_cb = 0;  // for skewness

    for (int i = 0; i < m; i++) {
        int rlen = rowptr[i + 1] - rowptr[i];
        if (rlen > nnz_per_row_max) nnz_per_row_max = rlen;
        double d = rlen - nnz_per_row_avg;
        nnz_per_row_sum_sq += d * d;
        nnz_per_row_sum_cb += d * d * d;
    }

    double nnz_per_row_std = sqrt(nnz_per_row_sum_sq / m);
    double nnz_per_row_skewness = 0;
    if (nnz_per_row_std > 0) {
        nnz_per_row_skewness = (nnz_per_row_sum_cb / m) / (nnz_per_row_std * nnz_per_row_std * nnz_per_row_std);
    }

    // nnz_per_col: 需要遍历 colidx 统计每列的 nnz
    int *col_nnz = (int *)calloc(n, sizeof(int));
    for (long long j = 0; j < nnz; j++) {
        col_nnz[colidx[j]]++;
    }

    double nnz_per_col_avg = (double)nnz / n;
    int    nnz_per_col_max = 0;
    double nnz_per_col_sum_sq = 0;

    for (int j = 0; j < n; j++) {
        if (col_nnz[j] > nnz_per_col_max) nnz_per_col_max = col_nnz[j];
        double d = col_nnz[j] - nnz_per_col_avg;
        nnz_per_col_sum_sq += d * d;
    }

    double nnz_per_col_std = sqrt(nnz_per_col_sum_sq / n);
    free(col_nnz);

    printf("\n=== [v2] Global Matrix Features ===\n");
    printf("  nnz_per_row: avg=%.2f, max=%d, std=%.2f, skewness=%.4f\n",
           nnz_per_row_avg, nnz_per_row_max, nnz_per_row_std, nnz_per_row_skewness);
    printf("  nnz_per_col: avg=%.2f, max=%d, std=%.2f\n",
           nnz_per_col_avg, nnz_per_col_max, nnz_per_col_std);
    printf("  RM (row max) / CM (col max) = %d / %d = %.4f\n",
           nnz_per_row_max, nnz_per_col_max,
           nnz_per_col_max > 0 ? (double)nnz_per_row_max / nnz_per_col_max : 0.0);

    // 填充全局特征结构体
    GlobalFeatures gf;
    gf.nnz_per_row_max     = nnz_per_row_max;
    gf.nnz_per_row_std     = nnz_per_row_std;
    gf.nnz_per_row_skewness = nnz_per_row_skewness;
    gf.nnz_per_col_max     = nnz_per_col_max;
    gf.nnz_per_col_avg     = nnz_per_col_avg;
    gf.nnz_per_col_std     = nnz_per_col_std;

    // ========================================================================
    // 打印结果
    // ========================================================================
    printf("=== Tile Statistics by Size ===\n");
    for (int p = 0; p < NUM_TILE_SIZES; p++) {
        printf("\n[Tile %d×%d]\n", probes[p].tile_m, probes[p].tile_n);
        print_probe_results(&probes[p], m, n, nnz);
    }

    // 打印跨尺寸比较
    print_cross_size_features(probes);

    // 打印 CSV 格式
    print_csv_header();
    for (int p = 0; p < NUM_TILE_SIZES; p++) {
        print_csv_row(&probes[p], &gf);
    }

    // ========================================================================
    // 清理
    // ========================================================================
    for (int p = 0; p < NUM_TILE_SIZES; p++) {
        free_probe(&probes[p]);
    }
    free(rowptr);
    free(colidx);

    printf("\n============================================================\n");
    printf("  Probe completed successfully!\n");
    printf("============================================================\n");

    return 0;
}
