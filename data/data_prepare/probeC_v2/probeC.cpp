/**
 * probeC.cpp - C tile density distribution estimator
 *
 * For C = A * B (or C = A * A), estimates the distribution of C tiles
 * into categories (Sml/Lrg/Dns/Ful) for 3 square tile sizes (8/16/32).
 * Tiny = Total_non_empty - Sml - Lrg - Dns - Ful (computed at runtime).
 *
 * Method: tile-level symbolic SpGEMM + ball-into-bins probabilistic model
 *
 * Compile: make
 * Run:     ./probeC <A.mtx>              (C = A*A)
 *          ./probeC <A.mtx> <B.mtx>      (C = A*B)
 *          ./probeC --aat <A.mtx>        (C = A*A^T)
 *          ./probeC --aat <A.mtx> <AT.mtx> (C = A*A^T, AT from file)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef int MAT_PTR_TYPE;

#include "mmio.h"

// ============================================================================
// 3 square tile size configurations (C tile depends only on tile_m)
// ============================================================================
#define NUM_CONFIGS 3

static const int TILE_SIZES[NUM_CONFIGS] = {8, 16, 32};

// ============================================================================
// Tile CSR structure
// ============================================================================
typedef struct {
    int row_div, col_div;   // tile row/col division sizes
    int tilem, tilen;       // number of tile rows/cols
    int numtile;
    int *tile_ptr;          // [tilem + 1]
    int *tile_colidx;       // [numtile]
    int *tile_nnz;          // [numtile] nnz per tile
} TileCSR;

// ============================================================================
// C tile distribution result
// Only predicts Sml/Lrg/Dns/Ful; Tiny = total_non_empty - others
// Also includes step 3/4 performance-relevant statistics
// ============================================================================
typedef struct {
    int tile_m;
    // Density classification counts
    int cnt_sml;
    int cnt_lrg;
    int cnt_dns;
    int cnt_ful;
    // Step 3/4 performance statistics (computed in the same traversal)
    int numblkC;                // total non-empty C tiles
    long long total_flops;      // Σ(nnz_a × nnz_b) over all tile pairs
    long long total_matchedcnt; // Σ matchedcnt over all C tiles (for avg)
    int max_matchedcnt;         // max matchedcnt across all C tiles
    float max_flops_per_tile;   // max per-tile flops (= raw * tile_m)
    // [v2] C tile column distribution
    int tilen_C;                // number of tile columns in C
    int *tiles_per_col_C;       // [tilen_C] non-empty C tiles per tile column
    // [v2] Estimated nnz statistics
    long long sum_est_nnz;      // Σ est_nnz
    long long sum_est_nnz_sq;   // Σ est_nnz²
    int max_est_nnz;            // max est_nnz
} CDist;

// ============================================================================
// Build tile CSR from matrix CSR (OpenMP parallelized)
// Pre-allocates markers to avoid heap contention; auto-fallback to sequential
// for small data where thread overhead exceeds parallel benefit.
// ============================================================================
void build_tile_csr(int m, int n, long long nnz, MAT_PTR_TYPE *rowptr,
                    int *colidx, int row_div, int col_div, TileCSR *tc)
{
    tc->row_div = row_div;
    tc->col_div = col_div;
    tc->tilem = (m + row_div - 1) / row_div;
    tc->tilen = (n + col_div - 1) / col_div;

    tc->tile_ptr = (int *)malloc((tc->tilem + 1) * sizeof(int));

    // Decide parallelism: skip for small matrices where overhead dominates
    int max_threads = omp_get_max_threads();
    int use_threads = (nnz > 500000 && tc->tilem >= 256) ?
                      (max_threads > 32 ? 32 : max_threads) : 1;

    // Pre-allocate all markers sequentially (avoid heap contention)
    int **markers = (int **)malloc(use_threads * sizeof(int *));
    for (int t = 0; t < use_threads; t++)
        markers[t] = (int *)calloc(tc->tilen, sizeof(int));

    // Pass 1: count tiles per row
    #pragma omp parallel num_threads(use_threads) if(use_threads > 1)
    {
        int *marker = markers[omp_get_thread_num()];

        #pragma omp for schedule(dynamic, 64)
        for (int ti = 0; ti < tc->tilem; ti++) {
            int rs = ti * row_div;
            int re = rs + row_div;
            if (re > m) re = m;
            int cnt = 0;
            for (int i = rs; i < re; i++) {
                for (MAT_PTR_TYPE j = rowptr[i]; j < rowptr[i + 1]; j++) {
                    int tj = colidx[j] / col_div;
                    if (!marker[tj]) {
                        marker[tj] = 1;
                        cnt++;
                    }
                }
            }
            tc->tile_ptr[ti + 1] = cnt;
            for (int i = rs; i < re; i++) {
                for (MAT_PTR_TYPE j = rowptr[i]; j < rowptr[i + 1]; j++) {
                    marker[colidx[j] / col_div] = 0;
                }
            }
        }
    }

    // Prefix sum
    tc->tile_ptr[0] = 0;
    for (int i = 0; i < tc->tilem; i++)
        tc->tile_ptr[i + 1] += tc->tile_ptr[i];
    tc->numtile = tc->tile_ptr[tc->tilem];

    tc->tile_colidx = (int *)malloc(tc->numtile * sizeof(int));
    tc->tile_nnz = (int *)calloc(tc->numtile, sizeof(int));

    // Pass 2: fill colidx and nnz (reuse same markers, already zeroed)
    #pragma omp parallel num_threads(use_threads) if(use_threads > 1)
    {
        int *marker = markers[omp_get_thread_num()];

        #pragma omp for schedule(dynamic, 64)
        for (int ti = 0; ti < tc->tilem; ti++) {
            int rs = ti * row_div;
            int re = rs + row_div;
            if (re > m) re = m;
            int base = tc->tile_ptr[ti];
            int cnt = 0;

            for (int i = rs; i < re; i++) {
                for (MAT_PTR_TYPE j = rowptr[i]; j < rowptr[i + 1]; j++) {
                    int tj = colidx[j] / col_div;
                    if (!marker[tj]) {
                        marker[tj] = cnt + 1;
                        tc->tile_colidx[base + cnt] = tj;
                        tc->tile_nnz[base + cnt] = 1;
                        cnt++;
                    } else {
                        tc->tile_nnz[base + marker[tj] - 1]++;
                    }
                }
            }
            for (int k = 0; k < cnt; k++)
                marker[tc->tile_colidx[base + k]] = 0;
        }
    }

    for (int t = 0; t < use_threads; t++)
        free(markers[t]);
    free(markers);
}

// ============================================================================
// Derive a 2x-coarser tile CSR from a finer one: O(numtile) instead of O(nnz)
// E.g., build tile_m=16 from tile_m=8 by merging every 2x2 block of tiles.
// ============================================================================
void derive_tile_csr(TileCSR *src, int m, int n, TileCSR *dst)
{
    dst->row_div = src->row_div * 2;
    dst->col_div = src->col_div * 2;
    dst->tilem = (m + dst->row_div - 1) / dst->row_div;
    dst->tilen = (n + dst->col_div - 1) / dst->col_div;
    dst->tile_ptr = (int *)malloc((dst->tilem + 1) * sizeof(int));

    int *marker = (int *)calloc(dst->tilen, sizeof(int));

    // Pass 1: count merged tiles per row
    for (int di = 0; di < dst->tilem; di++) {
        int cnt = 0;
        for (int r = 2 * di; r <= 2 * di + 1 && r < src->tilem; r++) {
            for (int s = src->tile_ptr[r]; s < src->tile_ptr[r + 1]; s++) {
                int dj = src->tile_colidx[s] / 2;
                if (!marker[dj]) { marker[dj] = 1; cnt++; }
            }
        }
        dst->tile_ptr[di + 1] = cnt;
        for (int r = 2 * di; r <= 2 * di + 1 && r < src->tilem; r++) {
            for (int s = src->tile_ptr[r]; s < src->tile_ptr[r + 1]; s++) {
                marker[src->tile_colidx[s] / 2] = 0;
            }
        }
    }

    // Prefix sum
    dst->tile_ptr[0] = 0;
    for (int i = 0; i < dst->tilem; i++)
        dst->tile_ptr[i + 1] += dst->tile_ptr[i];
    dst->numtile = dst->tile_ptr[dst->tilem];

    dst->tile_colidx = (int *)malloc(dst->numtile * sizeof(int));
    dst->tile_nnz = (int *)calloc(dst->numtile, sizeof(int));

    // Pass 2: fill colidx and nnz
    for (int di = 0; di < dst->tilem; di++) {
        int base = dst->tile_ptr[di];
        int cnt = 0;
        for (int r = 2 * di; r <= 2 * di + 1 && r < src->tilem; r++) {
            for (int s = src->tile_ptr[r]; s < src->tile_ptr[r + 1]; s++) {
                int dj = src->tile_colidx[s] / 2;
                if (!marker[dj]) {
                    marker[dj] = cnt + 1;
                    dst->tile_colidx[base + cnt] = dj;
                    dst->tile_nnz[base + cnt] = src->tile_nnz[s];
                    cnt++;
                } else {
                    dst->tile_nnz[base + marker[dj] - 1] += src->tile_nnz[s];
                }
            }
        }
        for (int k = 0; k < cnt; k++)
            marker[dst->tile_colidx[base + k]] = 0;
    }

    free(marker);
}

void free_tile_csr(TileCSR *tc) {
    free(tc->tile_ptr);
    free(tc->tile_colidx);
    free(tc->tile_nnz);
}

// ============================================================================
// Transpose a tile CSR: O(numtile) instead of O(nnz)
// Given tile CSR of A (tilem_A tile-rows × tilen_A tile-cols),
// produces tile CSR of A^T (tilen_A tile-rows × tilem_A tile-cols).
// The nnz per tile is preserved (transposed tile has same nnz).
// ============================================================================
void transpose_tile_csr(TileCSR *src, TileCSR *dst)
{
    dst->row_div = src->col_div;
    dst->col_div = src->row_div;
    dst->tilem = src->tilen;
    dst->tilen = src->tilem;

    int numtile = src->numtile;
    int dst_tilem = dst->tilem;

    dst->tile_ptr = (int *)calloc(dst_tilem + 1, sizeof(int));

    // Count tiles per tile-column of src (= tile-row of dst)
    for (int i = 0; i < src->tilem; i++) {
        for (int s = src->tile_ptr[i]; s < src->tile_ptr[i + 1]; s++) {
            dst->tile_ptr[src->tile_colidx[s] + 1]++;
        }
    }

    // Prefix sum
    for (int i = 1; i <= dst_tilem; i++)
        dst->tile_ptr[i] += dst->tile_ptr[i - 1];
    dst->numtile = numtile;

    dst->tile_colidx = (int *)malloc(numtile * sizeof(int));
    dst->tile_nnz = (int *)malloc(numtile * sizeof(int));

    // Fill: use a counter array
    int *cnt = (int *)calloc(dst_tilem, sizeof(int));
    for (int i = 0; i < src->tilem; i++) {
        for (int s = src->tile_ptr[i]; s < src->tile_ptr[i + 1]; s++) {
            int dj = src->tile_colidx[s];
            int pos = dst->tile_ptr[dj] + cnt[dj];
            dst->tile_colidx[pos] = i;       // row of src becomes col of dst
            dst->tile_nnz[pos] = src->tile_nnz[s];
            cnt[dj]++;
        }
    }
    free(cnt);
}

// ============================================================================
// Per-tile accumulator: merged est+cnt for cache locality (8 bytes per entry,
// both fields always accessed together in the same cache line)
// ============================================================================
typedef struct { float est; int cnt; } TileAcc;

// ============================================================================
// Estimate C tile distributions for ALL tile configs in a single parallel
// region.  This avoids repeated thread creation and per-thread allocation.
//
// C = A * B (general case; A==B degrades to C = A*A)
// tc_a[c] / tc_b[c]: tile CSR for config c, TILE_SIZES[c]
// ============================================================================
void estimate_all_dists(TileCSR *tc_a, TileCSR *tc_b, CDist *dists)
{
    // Find max tilen across all configs (for buffer sizing)
    int max_tilen = 0;
    for (int c = 0; c < NUM_CONFIGS; c++) {
        int tl = tc_b[c].tilen;
        if (tl > max_tilen) max_tilen = tl;
    }

    // Init all dists
    for (int c = 0; c < NUM_CONFIGS; c++) {
        dists[c].tile_m = TILE_SIZES[c];
        dists[c].cnt_sml = dists[c].cnt_lrg = dists[c].cnt_dns = dists[c].cnt_ful = 0;
        dists[c].numblkC = 0;
        dists[c].total_flops = 0;
        dists[c].total_matchedcnt = 0;
        dists[c].max_matchedcnt = 0;
        dists[c].max_flops_per_tile = 0.0f;
        // [v2] Initialize new fields
        dists[c].tilen_C = tc_b[c].tilen;
        dists[c].tiles_per_col_C = (int *)calloc(tc_b[c].tilen, sizeof(int));
        dists[c].sum_est_nnz = 0;
        dists[c].sum_est_nnz_sq = 0;
        dists[c].max_est_nnz = 0;
    }

    // Cap threads at physical core count (avoid hyperthreading cache thrashing)
    int nthreads = omp_get_max_threads();
    int use_threads = nthreads > 32 ? nthreads / 2 : nthreads;

    #pragma omp parallel num_threads(use_threads)
    {
        // Allocate once, reuse across all configs
        TileAcc *c_acc = (TileAcc *)calloc(max_tilen, sizeof(TileAcc));
        int *dirty = (int *)malloc(max_tilen * sizeof(int));
        // [v2] Thread-local tiles_per_col_C for each config
        int *l_tpc[NUM_CONFIGS];
        for (int c = 0; c < NUM_CONFIGS; c++)
            l_tpc[c] = (int *)calloc(tc_b[c].tilen, sizeof(int));

        for (int config = 0; config < NUM_CONFIGS; config++) {
            int tile_m = TILE_SIZES[config];
            int tilem_C = tc_a[config].tilem;
            int capacity = tile_m * tile_m;
            float inv_tile_m = 1.0f / tile_m;
            float inv_capacity = 1.0f / capacity;

            int tny_th = capacity / 8;
            int sml_th = capacity / 8;
            int lrg_th = capacity * 7 / 8;
            int dns_th = capacity;
            float ful_empty_th = (float)capacity / 16.0f;
            float ful_raw_th = capacity * 2.773f;

            const int *a_ptr = tc_a[config].tile_ptr;
            const int *a_col = tc_a[config].tile_colidx;
            const int *a_nnz = tc_a[config].tile_nnz;
            const int *b_ptr = tc_b[config].tile_ptr;
            const int *b_col = tc_b[config].tile_colidx;
            const int *b_nnz = tc_b[config].tile_nnz;

            int l_sml = 0, l_lrg = 0, l_dns = 0, l_ful = 0;
            int l_numblkC = 0;
            long long l_flops = 0, l_matchedcnt = 0;
            int l_max_matched = 0;
            float l_max_flops_tile = 0.0f;
            // [v2] thread-local est_nnz accumulators
            long long l_sum_est = 0, l_sum_est_sq = 0;
            int l_max_est = 0;

            #pragma omp for schedule(dynamic, 4)
            for (int ti = 0; ti < tilem_C; ti++) {
                int ndirty = 0;

                for (int ia = a_ptr[ti]; ia < a_ptr[ti + 1]; ia++) {
                    int k = a_col[ia];
                    int nnz_a_val = a_nnz[ia];
                    float fa = (float)nnz_a_val * inv_tile_m;
                    int sum_nnz_b = 0;

                    int ib_end = b_ptr[k + 1];
                    for (int ib = b_ptr[k]; ib < ib_end; ib++) {
                        int tj = b_col[ib];
                        int nnz_b_val = b_nnz[ib];

                        if (c_acc[tj].cnt == 0) {
                            dirty[ndirty++] = tj;
                        }
                        c_acc[tj].est += fa * nnz_b_val;
                        c_acc[tj].cnt++;
                        sum_nnz_b += nnz_b_val;
                    }
                    l_flops += (long long)nnz_a_val * sum_nnz_b;
                }

                l_numblkC += ndirty;

                for (int d = 0; d < ndirty; d++) {
                    int tj = dirty[d];
                    float raw = c_acc[tj].est;
                    int matched = c_acc[tj].cnt;

                    l_matchedcnt += matched;
                    if (matched > l_max_matched)
                        l_max_matched = matched;

                    float tile_flops = raw * tile_m;
                    if (tile_flops > l_max_flops_tile)
                        l_max_flops_tile = tile_flops;

                    // [v2] Count tiles per column
                    l_tpc[config][tj]++;

                    // [v2] Compute est_nnz for ALL tiles (for statistics)
                    int est_nnz;

                    // Fast path: skip expf for trivially small/large raw
                    if (raw <= tny_th) {
                        // Tiny (ball-into-bins concavity: est_nnz < raw)
                        est_nnz = (int)raw;
                        if (est_nnz < 1) est_nnz = 1;
                    } else if (raw >= ful_raw_th) {
                        // Full (expected_empty < capacity/16)
                        est_nnz = capacity;
                        l_ful++;
                    } else {
                        float expected_empty = capacity * expf(-raw * inv_capacity);
                        if (expected_empty < ful_empty_th) {
                            est_nnz = capacity;
                        } else {
                            est_nnz = capacity - (int)expected_empty;
                        }

                        if (est_nnz <= tny_th) {
                            // Tiny
                        } else if (est_nnz <= sml_th) {
                            l_sml++;
                        } else if (est_nnz <= lrg_th) {
                            l_lrg++;
                        } else if (est_nnz < dns_th) {
                            l_dns++;
                        } else {
                            l_ful++;
                        }
                    }

                    // [v2] Accumulate est_nnz stats
                    l_sum_est += est_nnz;
                    l_sum_est_sq += (long long)est_nnz * est_nnz;
                    if (est_nnz > l_max_est) l_max_est = est_nnz;

                    c_acc[tj].est = 0.0f;
                    c_acc[tj].cnt = 0;
                }
            }

            #pragma omp critical
            {
                dists[config].cnt_sml += l_sml;
                dists[config].cnt_lrg += l_lrg;
                dists[config].cnt_dns += l_dns;
                dists[config].cnt_ful += l_ful;
                dists[config].numblkC += l_numblkC;
                dists[config].total_flops += l_flops;
                dists[config].total_matchedcnt += l_matchedcnt;
                if (l_max_matched > dists[config].max_matchedcnt)
                    dists[config].max_matchedcnt = l_max_matched;
                if (l_max_flops_tile > dists[config].max_flops_per_tile)
                    dists[config].max_flops_per_tile = l_max_flops_tile;
                // [v2] Merge est_nnz stats
                dists[config].sum_est_nnz += l_sum_est;
                dists[config].sum_est_nnz_sq += l_sum_est_sq;
                if (l_max_est > dists[config].max_est_nnz)
                    dists[config].max_est_nnz = l_max_est;
                // [v2] Merge tiles_per_col_C
                for (int j = 0; j < dists[config].tilen_C; j++)
                    dists[config].tiles_per_col_C[j] += l_tpc[config][j];
            }
            // implicit barrier at end of omp for ensures all threads
            // finished this config before moving to the next
        }

        free(c_acc);
        free(dirty);
        // [v2] Free thread-local tiles_per_col
        for (int c = 0; c < NUM_CONFIGS; c++)
            free(l_tpc[c]);
    }
}

// ============================================================================
// Load MTX matrix (same as probe7)
// ============================================================================
int load_mtx(const char *filename, int *m, int *n, long long *nnz,
             int *isSymmetric, MAT_PTR_TYPE **rowptr, int **colidx)
{
    int m_tmp, n_tmp, nnz_mtx;
    MM_typecode matcode;
    FILE *f;

    int isInteger = 0, isReal = 0, isPattern = 0, isComplex = 0;

    if ((f = fopen(filename, "r")) == NULL) {
        printf("Error: Cannot open %s\n", filename);
        return -1;
    }
    if (mm_read_banner(f, &matcode) != 0) return -2;

    if (mm_is_pattern(matcode))  isPattern = 1;
    if (mm_is_real(matcode))     isReal = 1;
    if (mm_is_complex(matcode))  isComplex = 1;
    if (mm_is_integer(matcode))  isInteger = 1;

    if (mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx) != 0) return -3;

    int isSym = (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) ? 1 : 0;

    MAT_PTR_TYPE *cnt = (MAT_PTR_TYPE *)calloc(m_tmp + 1, sizeof(MAT_PTR_TYPE));
    int *ri = (int *)malloc(nnz_mtx * sizeof(int));
    int *ci = (int *)malloc(nnz_mtx * sizeof(int));

    for (int i = 0; i < nnz_mtx; i++) {
        int r, c; double v; int iv; double vim;
        if (isReal)         fscanf(f, "%d %d %lg\n", &r, &c, &v);
        else if (isComplex) fscanf(f, "%d %d %lg %lg\n", &r, &c, &v, &vim);
        else if (isInteger) fscanf(f, "%d %d %d\n", &r, &c, &iv);
        else                fscanf(f, "%d %d\n", &r, &c);
        r--; c--;
        cnt[r]++;
        ri[i] = r; ci[i] = c;
    }
    fclose(f);

    if (isSym) {
        for (int i = 0; i < nnz_mtx; i++)
            if (ri[i] != ci[i]) cnt[ci[i]]++;
    }

    // exclusive scan
    int old = cnt[0]; cnt[0] = 0;
    for (int i = 1; i <= m_tmp; i++) {
        int nv = cnt[i];
        cnt[i] = old + cnt[i - 1];
        old = nv;
    }
    long long total_nnz = cnt[m_tmp];

    MAT_PTR_TYPE *rp = (MAT_PTR_TYPE *)malloc((m_tmp + 1) * sizeof(MAT_PTR_TYPE));
    int *col = (int *)malloc(total_nnz * sizeof(int));
    memcpy(rp, cnt, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));
    memset(cnt, 0, (m_tmp + 1) * sizeof(MAT_PTR_TYPE));

    if (isSym) {
        for (int i = 0; i < nnz_mtx; i++) {
            MAT_PTR_TYPE off = rp[ri[i]] + cnt[ri[i]];
            col[off] = ci[i]; cnt[ri[i]]++;
            if (ri[i] != ci[i]) {
                off = rp[ci[i]] + cnt[ci[i]];
                col[off] = ri[i]; cnt[ci[i]]++;
            }
        }
    } else {
        for (int i = 0; i < nnz_mtx; i++) {
            MAT_PTR_TYPE off = rp[ri[i]] + cnt[ri[i]];
            col[off] = ci[i]; cnt[ri[i]]++;
        }
    }

    *m = m_tmp; *n = n_tmp; *nnz = total_nnz;
    *isSymmetric = isSym;
    *rowptr = rp; *colidx = col;
    free(cnt); free(ri); free(ci);
    return 0;
}

// ============================================================================
// Build CSC from CSR (i.e., transpose: A^T's CSR = A's CSC)
// Input:  A in CSR (m rows, n cols, nnz entries)
// Output: A^T in CSR (n rows, m cols, nnz entries)
// ============================================================================
void build_transpose(int m, int n, long long nnz,
                     MAT_PTR_TYPE *rowptr, int *colidx,
                     MAT_PTR_TYPE **t_rowptr, int **t_colidx)
{
    MAT_PTR_TYPE *trp = (MAT_PTR_TYPE *)calloc(n + 1, sizeof(MAT_PTR_TYPE));
    int *tcol = (int *)malloc(nnz * sizeof(int));

    // Count nnz per column of A (= nnz per row of A^T)
    for (long long i = 0; i < nnz; i++)
        trp[colidx[i] + 1]++;

    // Prefix sum
    for (int i = 1; i <= n; i++)
        trp[i] += trp[i - 1];

    // Fill column indices
    MAT_PTR_TYPE *cnt = (MAT_PTR_TYPE *)calloc(n, sizeof(MAT_PTR_TYPE));
    for (int i = 0; i < m; i++) {
        for (MAT_PTR_TYPE j = rowptr[i]; j < rowptr[i + 1]; j++) {
            int c = colidx[j];
            tcol[trp[c] + cnt[c]] = i;
            cnt[c]++;
        }
    }
    free(cnt);

    *t_rowptr = trp;
    *t_colidx = tcol;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("Usage: %s [--aat] <A.mtx> [B.mtx]\n", argv[0]);
        printf("  1 arg:         C = A * A\n");
        printf("  2 args:        C = A * B\n");
        printf("  --aat 1 arg:   C = A * A^T (transpose built internally)\n");
        printf("  --aat 2 args:  C = A * A^T (A^T loaded from file)\n");
        return 1;
    }

    // Parse --aat flag
    int aat_mode = 0;
    int arg_offset = 1;
    if (strcmp(argv[1], "--aat") == 0) {
        aat_mode = 1;
        arg_offset = 2;
        if (argc < 3) {
            printf("Error: --aat requires at least one matrix file\n");
            return 1;
        }
    }

    int remaining_args = argc - arg_offset;
    int two_matrices = (remaining_args >= 2);
    const char *file_a = argv[arg_offset];
    const char *file_b = two_matrices ? argv[arg_offset + 1] : NULL;

    // Load matrix A
    int m_a, n_a, isSym_a;
    long long nnz_a;
    MAT_PTR_TYPE *rowptr_a = NULL;
    int *colidx_a = NULL;

    double t0 = omp_get_wtime();
    if (load_mtx(file_a, &m_a, &n_a, &nnz_a, &isSym_a, &rowptr_a, &colidx_a) != 0) {
        printf("Error loading matrix A!\n");
        return 1;
    }
    double t1 = omp_get_wtime();

    printf("Matrix A: %s\n", file_a);
    printf("  Size: %d x %d, nnz: %lld, Symmetric: %s\n",
           m_a, n_a, nnz_a, isSym_a ? "Yes" : "No");
    printf("  Load time: %.2f ms\n", (t1 - t0) * 1000);

    // Load matrix B, build A^T, or reuse A
    int m_b, n_b, isSym_b;
    long long nnz_b;
    MAT_PTR_TYPE *rowptr_b = NULL;
    int *colidx_b = NULL;
    int free_b = 0;  // whether to free B separately

    if (aat_mode && two_matrices) {
        // --aat with AT file: load AT from file
        t0 = omp_get_wtime();
        if (load_mtx(file_b, &m_b, &n_b, &nnz_b, &isSym_b, &rowptr_b, &colidx_b) != 0) {
            printf("Error loading matrix A^T!\n");
            free(rowptr_a); free(colidx_a);
            return 1;
        }
        t1 = omp_get_wtime();
        printf("Matrix A^T (from file): %s\n", file_b);
        printf("  Size: %d x %d, nnz: %lld\n", m_b, n_b, nnz_b);
        printf("  Load time: %.2f ms\n", (t1 - t0) * 1000);
        free_b = 1;

        if (n_a != m_b) {
            printf("Error: A cols (%d) != A^T rows (%d)\n", n_a, m_b);
            free(rowptr_a); free(colidx_a);
            free(rowptr_b); free(colidx_b);
            return 1;
        }
    } else if (aat_mode && !two_matrices) {
        // --aat without AT file: build A^T tile CSR via tile-level transpose
        // (much faster than element-level transpose + tile CSR rebuild)
        m_b = n_a; n_b = m_a; nnz_b = nnz_a;
        rowptr_b = NULL; colidx_b = NULL;  // not needed
        free_b = 0;  // nothing to free at element level
        printf("A^T will be built at tile level (fast path)\n");
    } else if (two_matrices) {
        // A * B mode
        t0 = omp_get_wtime();
        if (load_mtx(file_b, &m_b, &n_b, &nnz_b, &isSym_b, &rowptr_b, &colidx_b) != 0) {
            printf("Error loading matrix B!\n");
            free(rowptr_a); free(colidx_a);
            return 1;
        }
        t1 = omp_get_wtime();
        printf("Matrix B: %s\n", file_b);
        printf("  Size: %d x %d, nnz: %lld, Symmetric: %s\n",
               m_b, n_b, nnz_b, isSym_b ? "Yes" : "No");
        printf("  Load time: %.2f ms\n", (t1 - t0) * 1000);
        free_b = 1;

        if (n_a != m_b) {
            printf("Error: A cols (%d) != B rows (%d), cannot multiply\n", n_a, m_b);
            free(rowptr_a); free(colidx_a);
            free(rowptr_b); free(colidx_b);
            return 1;
        }
    } else {
        // C = A * A: A must be square
        if (m_a != n_a) {
            printf("Error: matrix must be square for C = A*A\n");
            free(rowptr_a); free(colidx_a);
            return 1;
        }
        m_b = m_a; n_b = n_a; nnz_b = nnz_a;
        rowptr_b = rowptr_a; colidx_b = colidx_a;
        free_b = 0;
    }

    const char *op_str = aat_mode ? "A * A^T" : (two_matrices ? "A * B" : "A * A");
    printf("\nC = %s: %d x %d\n\n", op_str, m_a, n_b);

    // Build tile CSR for A and B
    printf("Building tile structures...\n");
    TileCSR tc_a[NUM_CONFIGS], tc_b[NUM_CONFIGS];
    int separate_b = free_b || two_matrices;
    int aat_fast = (aat_mode && !two_matrices);  // tile-level transpose path
    t0 = omp_get_wtime();

    build_tile_csr(m_a, n_a, nnz_a, rowptr_a, colidx_a,
                   TILE_SIZES[0], TILE_SIZES[0], &tc_a[0]);
    derive_tile_csr(&tc_a[0], m_a, n_a, &tc_a[1]);
    derive_tile_csr(&tc_a[1], m_a, n_a, &tc_a[2]);

    if (aat_fast) {
        // Build A^T's tile CSR by transposing A's tile CSR at tile level
        for (int c = 0; c < NUM_CONFIGS; c++)
            transpose_tile_csr(&tc_a[c], &tc_b[c]);
    } else if (separate_b) {
        build_tile_csr(m_b, n_b, nnz_b, rowptr_b, colidx_b,
                       TILE_SIZES[0], TILE_SIZES[0], &tc_b[0]);
        derive_tile_csr(&tc_b[0], m_b, n_b, &tc_b[1]);
        derive_tile_csr(&tc_b[1], m_b, n_b, &tc_b[2]);
    }

    t1 = omp_get_wtime();
    printf("Build time: %.2f ms\n\n", (t1 - t0) * 1000);

    // Estimate C distribution for all configs in one parallel region
    printf("Estimating C tile distributions...\n");
    CDist dists[NUM_CONFIGS];
    t0 = omp_get_wtime();

    {
        // Build pointer arrays for A and B tile CSRs
        TileCSR *pb_arr = (separate_b || aat_fast) ? tc_b : tc_a;
        estimate_all_dists(tc_a, pb_arr, dists);
    }

    t1 = omp_get_wtime();
    printf("Estimate time: %.2f ms\n\n", (t1 - t0) * 1000);

    // Print results
    printf("=== Estimated C Tile Distribution (non-Tiny) ===\n");
    printf("Note: Tiny = actual_total - Sml - Lrg - Dns - Ful\n\n");
    printf("%-10s %8s %8s %8s %8s\n",
           "TileSize", "Sml", "Lrg", "Dns", "Ful");
    for (int c = 0; c < NUM_CONFIGS; c++) {
        CDist *d = &dists[c];
        printf("%-4dx%-4d %8d %8d %8d %8d\n",
               d->tile_m, d->tile_m,
               d->cnt_sml, d->cnt_lrg, d->cnt_dns, d->cnt_ful);
    }

    // Print step 3/4 performance statistics
    printf("\n=== Step 3/4 Performance Statistics ===\n\n");
    printf("%-10s %10s %16s %12s %12s %16s\n",
           "TileSize", "numblkC", "total_flops", "avg_matched", "max_matched", "max_flops/tile");
    for (int c = 0; c < NUM_CONFIGS; c++) {
        CDist *d = &dists[c];
        float avg_matched = d->numblkC > 0 ?
            (float)d->total_matchedcnt / d->numblkC : 0.0f;
        printf("%-4dx%-4d %10d %16lld %12.2f %12d %16.0f\n",
               d->tile_m, d->tile_m,
               d->numblkC, d->total_flops,
               avg_matched, d->max_matchedcnt,
               d->max_flops_per_tile);
    }

    // [v2] Print C tile column distribution and est_nnz statistics
    printf("\n=== [v2] C Tile Column Distribution & Est NNZ Stats ===\n\n");
    for (int c = 0; c < NUM_CONFIGS; c++) {
        CDist *d = &dists[c];
        int tilen = d->tilen_C;

        // tiles_per_col_C statistics
        double tpc_avg = d->numblkC > 0 ? (double)d->numblkC / tilen : 0;
        int tpc_max = 0;
        double tpc_sum_sq = 0;
        int tpc_empty = 0;

        for (int j = 0; j < tilen; j++) {
            int v = d->tiles_per_col_C[j];
            if (v > tpc_max) tpc_max = v;
            double diff = v - tpc_avg;
            tpc_sum_sq += diff * diff;
            if (v == 0) tpc_empty++;
        }

        double tpc_std = sqrt(tpc_sum_sq / (tilen > 0 ? tilen : 1));
        double tpc_empty_ratio = (double)tpc_empty / (tilen > 0 ? tilen : 1);

        // est_nnz statistics
        double est_avg = d->numblkC > 0 ? (double)d->sum_est_nnz / d->numblkC : 0;
        double est_var = d->numblkC > 0 ?
            (double)d->sum_est_nnz_sq / d->numblkC - est_avg * est_avg : 0;
        double est_std = sqrt(est_var > 0 ? est_var : 0);

        printf("[%dx%d] tiles_per_col_C: avg=%.2f, max=%d (CM), std=%.2f, empty_ratio=%.4f\n",
               d->tile_m, d->tile_m, tpc_avg, tpc_max, tpc_std, tpc_empty_ratio);
        printf("        est_nnz: avg=%.2f, max=%d, std=%.2f\n",
               est_avg, d->max_est_nnz, est_std);
    }

    // Also print in parseable format
    printf("\n=== CSV ===\n");
    printf("tile_m,sml,lrg,dns,ful,numblkC,total_flops,avg_matchedcnt,max_matchedcnt,max_flops_per_tile,");
    printf("tpc_C_avg,tpc_C_max,tpc_C_std,tpc_C_empty_ratio,est_nnz_avg,est_nnz_max,est_nnz_std\n");
    for (int c = 0; c < NUM_CONFIGS; c++) {
        CDist *d = &dists[c];
        float avg_matched = d->numblkC > 0 ?
            (float)d->total_matchedcnt / d->numblkC : 0.0f;
        int tilen = d->tilen_C;

        // Recompute tpc stats for CSV
        double tpc_avg = d->numblkC > 0 ? (double)d->numblkC / tilen : 0;
        int tpc_max = 0;
        double tpc_sum_sq = 0;
        int tpc_empty = 0;
        for (int j = 0; j < tilen; j++) {
            int v = d->tiles_per_col_C[j];
            if (v > tpc_max) tpc_max = v;
            double diff = v - tpc_avg;
            tpc_sum_sq += diff * diff;
            if (v == 0) tpc_empty++;
        }
        double tpc_std = sqrt(tpc_sum_sq / (tilen > 0 ? tilen : 1));
        double tpc_empty_ratio = (double)tpc_empty / (tilen > 0 ? tilen : 1);

        double est_avg = d->numblkC > 0 ? (double)d->sum_est_nnz / d->numblkC : 0;
        double est_var = d->numblkC > 0 ?
            (double)d->sum_est_nnz_sq / d->numblkC - est_avg * est_avg : 0;
        double est_std = sqrt(est_var > 0 ? est_var : 0);

        printf("%d,%d,%d,%d,%d,%d,%lld,%.2f,%d,%.0f,%.4f,%d,%.4f,%.6f,%.4f,%d,%.4f\n",
               d->tile_m,
               d->cnt_sml, d->cnt_lrg, d->cnt_dns, d->cnt_ful,
               d->numblkC, d->total_flops,
               avg_matched, d->max_matchedcnt,
               d->max_flops_per_tile,
               tpc_avg, tpc_max, tpc_std, tpc_empty_ratio,
               est_avg, d->max_est_nnz, est_std);
    }

    // Cleanup
    for (int c = 0; c < NUM_CONFIGS; c++) {
        free_tile_csr(&tc_a[c]);
        if (separate_b || aat_fast) free_tile_csr(&tc_b[c]);
        // [v2] Free tiles_per_col_C
        free(dists[c].tiles_per_col_C);
    }
    free(rowptr_a); free(colidx_a);
    if (free_b) { free(rowptr_b); free(colidx_b); }

    return 0;
}
