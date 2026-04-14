#include "common.h"
#include "utils.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
namespace cg = cooperative_groups;

__constant__ int d_SMEM_LRG_TH;

__device__ __forceinline__
size_t align_up(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

__forceinline__ __device__ int sum_32_shfl(int sum)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}

__forceinline__ __device__ int sum_16_shfl(int sum)
{
#pragma unroll
    for (int mask = 1; mask < HALFWARP_SIZE; mask <<= 1)
        sum += __shfl_xor_sync(-1, sum, mask);

    return sum;
}

__forceinline__ __device__ int sum_16_shfl_cg(int sum)
{
    auto g = cg::tiled_partition<16>(cg::this_thread_block());

    sum = cg::reduce(g, sum, cg::plus<int>());
    
    return sum;
}

__forceinline__ __device__ int sum_8_shfl(int sum)
{
#pragma unroll
    for (int mask = 1; mask < QUADWARP_SIZE; mask <<= 1)
        sum += __shfl_xor_sync(-1, sum, mask);

    return sum;
}

template<int GROUP_SIZE>
__forceinline__ __device__ int sum_adaptive_shfl(int sum, int group_idx)
{
    static_assert(GROUP_SIZE == 8 || GROUP_SIZE == 16 || GROUP_SIZE == 32, 
                  "GROUP_SIZE must be 8, 16, or 32");

    const unsigned int MASK = (GROUP_SIZE == 32) ? 0xFFFFFFFF : (
                                (GROUP_SIZE == 16) ? 0x0000FFFF << (group_idx * GROUP_SIZE) : (
                                    (GROUP_SIZE == 8) ? 0x000000FF << (group_idx * GROUP_SIZE) : 0x00000000
                                )
                              );
    
#pragma unroll
    for (int mask = GROUP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(MASK, sum, mask);

    return sum;
}

__forceinline__ __device__ int binary_search_exact_kernel(const int *d_array, int l, int r, int key)
{
    while (l <= r)
    {
        int m = l + (r - l) / 2;
        int elem = d_array[m];
        // Check if x is present at mid
        if (elem == key)
            return m;

        // If x greater, ignore left half
        if (elem < key)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }
    return -1;
}

__forceinline__ __device__ int binary_search_exact_kernel_v2(const int *s_array, const int *d_array, int splitter,
                                                             int l, int r, int key)
{
    while (l <= r)
    {
        int m = l + (r - l) / 2;
        int elem = m < splitter ? s_array[m] : d_array[m];
        // Check if x is present at mid
        if (elem == key)
            return m;

        // If x greater, ignore left half
        if (elem < key)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }
    return -1;
}

__forceinline__ __device__ int binary_search_exact_uchar_kernel(const unsigned char *__restrict__ d_array, int l, int r, unsigned char key)
{
    while (l <= r)
    {
        int m = l + (r - l) / 2;
        unsigned char elem = d_array[m];
        // Check if x is present at mid
        if (elem == key)
            return m;

        // If x greater, ignore left half
        if (elem < key)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }
    return -1;
}

__forceinline__ __device__ int binary_search_exact_ushort_kernel(const uint16_t *__restrict__ d_array, int l, int r, uint16_t key)
{
    while (l <= r)
    {
        int m = l + (r - l) / 2;
        uint16_t elem = d_array[m];
        // Check if x is present at mid
        if (elem == key)
            return m;

        // If x greater, ignore left half
        if (elem < key)
            l = m + 1;

        // If x is smaller, ignore right half
        else
            r = m - 1;
    }
    return -1;
}

__forceinline__ __device__ int binary_search_right_boundary_kernel(const int *__restrict__ d_row_pointer,
                                                                   const int key_input,
                                                                   const int size)
{
    int start = 0;
    int stop = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = ld_gbl_auto(d_row_pointer + median);

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start - 1;
}

int binary_search_right_boundary_kernel_cpu(const int *d_row_pointer,
                                            const int key_input,
                                            const int size)
{
    int start = 0;
    int stop = size - 1;
    int median;
    int key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = d_row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start - 1;
}

__device__ __forceinline__ int intersection_binarysearch_kernel(const int *d_arraya, int abase, int astop, int lena,
                                                                const int *d_arrayb, int bbase, int bstop, int lenb,
                                                                int *d_posa, int *d_posb, int lenpos, int *d_cnt,
                                                                int lane_id, int warpsize)
{
    if (lena == 0 || lenb == 0)
    {
    }
    else if (lena < lenb)
    {
        for (int i = lane_id; i < lena; i += warpsize)
        {
            int idxa = d_arraya[abase + i];
            int res = binary_search_exact_kernel(d_arrayb + bbase, 0, lenb - 1, idxa);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = i;
                    d_posb[pos] = res;
                }
            }
        }
    }
    else
    {
        for (int i = lane_id; i < lenb; i += warpsize)
        {
            int idxb = d_arrayb[bbase + i];
            int res = binary_search_exact_kernel(d_arraya + abase, 0, lena - 1, idxb);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = res;
                    d_posb[pos] = i;
                }
            }
        }
    }

    return 0;
}

__device__ __forceinline__ int intersection_binarysearch_smem_kernel(const int *d_arraya, int abase, int astop, int lena,
                                                                     const int *d_arrayb, int bbase, int bstop, int lenb,
                                                                     int *s_intersection,
                                                                     int *d_posa, int *d_posb, int lenpos, int *d_cnt,
                                                                     int lane_id, int warpsize)
{
    if (lena == 0 || lenb == 0)
    {
    }
    else if (lena < lenb)
    {
        // optimize abase and lena, by search bstart and bstop in a
        const int bendidx = d_arrayb[bstop - 1];

        int use_smem = lenb <= SMEM_INTERSECTION_LEN && lena > SMEM_INTERSECTION_TH;
        if (use_smem)
        {
            for (int i = lane_id; i < lenb; i += warpsize)
                s_intersection[i] = d_arrayb[bbase + i];
        }

        for (int i = lane_id; i < lena; i += warpsize)
        {
            int idxa = d_arraya[abase + i];
            const int *searchspace = use_smem ? s_intersection : &d_arrayb[bbase];
            int res = binary_search_exact_kernel(searchspace, 0, lenb - 1, idxa);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = i;
                    d_posb[pos] = res;
                }
            }
        }
    }
    else
    {
        // optimize abase and lena, by search bstart and bstop in a
        int use_smem = lena <= SMEM_INTERSECTION_LEN && lenb > SMEM_INTERSECTION_TH;
        if (use_smem)
        {
            for (int i = lane_id; i < lena; i += warpsize)
                s_intersection[i] = d_arraya[abase + i];
        }

        for (int i = lane_id; i < lenb; i += warpsize)
        {
            int idxb = d_arrayb[bbase + i];
            const int *searchspace = use_smem ? s_intersection : &d_arraya[abase];
            int res = binary_search_exact_kernel(searchspace, 0, lena - 1, idxb);
            if (res != -1)
            {
                int pos = atomicAdd(d_cnt, 1);
                if (pos < lenpos)
                {
                    d_posa[pos] = res;
                    d_posb[pos] = i;
                }
            }
        }
    }
    return 0;
}

__global__ void tile_spgemm_step1_cuda_spa_kernel(int *d_blkrowptrA, int *d_blkcolidxA, int blkmA,
                                                  int *d_blkrowptrB, int *d_blkcolidxB, int blknB,
                                                  int *d_blkrowptrC)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;
    __shared__ unsigned int bitmask[WARP_PER_BLOCK * SPA_INT_PER_WARP];

    if (global_warp_id >= blkmA)
        return;

    const int nmasks = ceil((float)blknB / (float)32);
    const int local_warp_id = threadIdx.x >> 5; //global_id / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    unsigned int *bitmask_local = &bitmask[local_warp_id * SPA_INT_PER_WARP];

    for (int i = lane_id; i < nmasks; i += WARP_SIZE)
        bitmask_local[i] = 0;

    int astart = d_blkrowptrA[global_warp_id];
    int astop = d_blkrowptrA[global_warp_id + 1];
    for (int i = astart; i < astop; i++)
    {
        int rowidx = d_blkcolidxA[i];
        int bstart = d_blkrowptrB[rowidx];
        int bstop = d_blkrowptrB[rowidx + 1];
        for (int j = bstart + lane_id; j < bstop; j += WARP_SIZE)
        {
            int colidx = d_blkcolidxB[j];
            unsigned int mask = 1 << (31 - colidx % 32);
            atomicOr(&bitmask_local[colidx / 32], mask);
        }
    }
    //__syncthreads();

    int cnt = 0;
    for (int i = lane_id; i < nmasks; i += WARP_SIZE)
        cnt += __popc(bitmask_local[i]);
    cnt = sum_32_shfl(cnt);

    if (!lane_id){
        d_blkrowptrC[global_warp_id] = cnt;
    }
}

__global__ void tile_spgemm_step1_numeric_cuda_spa_kernel(int *d_blkrowptrA, int *d_blkcolidxA, int blkmA,
                                                          int *d_blkrowptrB, int *d_blkcolidxB, int blknB,
                                                          int *d_blkrowptrC, int *d_blkrowidxC, int *d_blkcolidxC,
                                                          int *d_spec_intersection_cnt, int *d_spec_intersection_posa, int *d_spec_intersection_posb)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;
    __shared__ unsigned int bitmask[WARP_PER_BLOCK * SPA_INT_PER_WARP];

    if (global_warp_id >= blkmA)
        return;

    const int nmasks = ceil((float)blknB / (float)32);
    const int nmasks_warpwise = ceil((float)nmasks / (float)WARP_SIZE) * WARP_SIZE; // make sure shfl func works
    const int local_warp_id = threadIdx.x >> 5;                                     //global_id / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    unsigned int *bitmask_local = &bitmask[local_warp_id * SPA_INT_PER_WARP];

    for (int i = lane_id; i < nmasks_warpwise; i += WARP_SIZE)
        bitmask_local[i] = 0;

    int cbase = d_blkrowptrC[global_warp_id];

    int astart = d_blkrowptrA[global_warp_id];
    int astop = d_blkrowptrA[global_warp_id + 1];
    for (int i = astart; i < astop; i++)
    {
        int rowidx = d_blkcolidxA[i];
        int bstart = d_blkrowptrB[rowidx];
        int bstop = d_blkrowptrB[rowidx + 1];
        for (int j = bstart + lane_id; j < bstop; j += WARP_SIZE)
        {
            int colidx = d_blkcolidxB[j];
            unsigned int mask = 1 << (31 - colidx % 32);
            atomicOr(&bitmask_local[colidx / 32], mask);
        }
    }

    int cnt = 0;
    int offset = 0;
    for (int i = lane_id; i < nmasks_warpwise; i += WARP_SIZE)
    {
        unsigned int maski = bitmask_local[i];
        int cnt = __popc(maski);

        // inclusive scan
        int cnt_scan = scan_32_shfl(cnt, lane_id);
        cnt_scan += offset;

        // sum
        offset = __shfl_sync(0xffffffff, cnt_scan, 31);

        // to exclusive scan
        cnt_scan -= cnt;

        // write to gmem
        int localoff = 0;
#pragma unroll
        for (int biti = 0; biti < 32; biti++)
        {
            if ((maski >> (31 - biti)) & 0x1)
            {
                d_blkrowidxC[cbase + cnt_scan + localoff] = global_warp_id;
                d_blkcolidxC[cbase + cnt_scan + localoff] = i * 32 + biti;
                localoff++;
            }
        }
    }
}

__global__ void tile_spgemm_step3_cuda_kernel_2level_quadwarp(const int *d_blkrowptrA,
                                                              const int *__restrict__ d_blkcolidxA,
                                                              const int *d_nnzb_A,
                                                              MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                              TILE_CSR_COL_TYPE_A *d_blkcsr_Col_A,
                                                              TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_A,
                                                              TILE_MASK_TYPE_A *d_blkmaskA,
                                                              int blkmA, int blknA, int numblkA, int nnzA,
                                                              const int *__restrict__ d_blkcolptrB,
                                                              const int *__restrict__ d_blkrowidxB,
                                                              const int *__restrict__ d_nnzb_B,
                                                              const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                              const TILE_CSR_COL_TYPE_B *__restrict__ d_blkcsr_Col_B,
                                                              const TILE_CSR_PTR_TYPE *__restrict__ d_blkcsr_Ptr_B,
                                                              const TILE_MASK_TYPE_B *__restrict__ d_blkmaskB,
                                                              int blkmB, int blknB, int numblkB, int nnzB,
                                                              unsigned int *d_blk_intersec_bitmask_A,
                                                              unsigned int *d_blk_intersec_bitmask_B,
                                                              int blk_intersec_bitmask_len,
                                                              int *d_blkrowidxC,
                                                              int *d_blkcolidxC,
                                                              TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_C,
                                                              int *d_nnzb_C,
                                                              TILE_MASK_TYPE_B *d_blkmaskC,
                                                              int *d_blksmem_tny_cnt,
                                                              int *d_blksmem_sml_cnt,
                                                              int *d_blksmem_lrg_cnt,
                                                              int *d_blksmem_dns_cnt,
                                                              int *d_blksmem_ful_cnt,
                                                              int *d_blkid_smem_tny,
                                                              int *d_blkid_smem_sml,
                                                              int *d_blkid_smem_lrg,
                                                              int *d_blkid_smem_dns,
                                                              int *d_blkid_smem_ful,
                                                              int *d_spec_intersection_cnt,
                                                              int *d_spec_intersection_posa,
                                                              int *d_spec_intersection_posb,
                                                              int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_quadwarp_id = global_id >> 3;

    // TODO: Optimize
    __shared__ unsigned int s_maskc[QUADWARP_PER_BLOCK * TILE_SIZE_M * MaskNumC];
    __shared__ TILE_MASK_TYPE_B s_blkmaskB[QUADWARP_PER_BLOCK * TILE_SIZE_N * MaskNumB];

    __shared__ int s_matched_posa[QUADWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[QUADWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[QUADWARP_PER_BLOCK];

    __shared__ int s_blksmem_tny_cnt[QUADWARP_PER_BLOCK];
    __shared__ int s_blksmem_sml_cnt[QUADWARP_PER_BLOCK];
    __shared__ int s_blksmem_lrg_cnt[QUADWARP_PER_BLOCK];
    __shared__ int s_blksmem_dns_cnt[QUADWARP_PER_BLOCK];
    __shared__ int s_blksmem_ful_cnt[QUADWARP_PER_BLOCK];

    __shared__ int s_blkid_smem_tny[QUADWARP_PER_BLOCK * TILE_PER_QUADWARP];
    __shared__ int s_blkid_smem_sml[QUADWARP_PER_BLOCK * TILE_PER_QUADWARP];
    __shared__ int s_blkid_smem_lrg[QUADWARP_PER_BLOCK * TILE_PER_QUADWARP];
    __shared__ int s_blkid_smem_dns[QUADWARP_PER_BLOCK * TILE_PER_QUADWARP];
    __shared__ int s_blkid_smem_ful[QUADWARP_PER_BLOCK * TILE_PER_QUADWARP];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int quadwarp_lane_id = (QUADWARP_SIZE - 1) & threadIdx.x;

    const int local_quadwarp_id = threadIdx.x >> 3;

    unsigned int *s_maskc_local = &s_maskc[local_quadwarp_id * TILE_SIZE_M * MaskNumC];
    TILE_MASK_TYPE_B *s_blkmaskB_local = &s_blkmaskB[local_quadwarp_id * TILE_SIZE_N * MaskNumB];

    int *s_matched_posa_local = &s_matched_posa[local_quadwarp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_quadwarp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_quadwarp_id];
    int *s_blksmem_tny_cnt_local = &s_blksmem_tny_cnt[local_quadwarp_id];
    int *s_blksmem_sml_cnt_local = &s_blksmem_sml_cnt[local_quadwarp_id];
    int *s_blksmem_lrg_cnt_local = &s_blksmem_lrg_cnt[local_quadwarp_id];
    int *s_blksmem_dns_cnt_local = &s_blksmem_dns_cnt[local_quadwarp_id];
    int *s_blksmem_ful_cnt_local = &s_blksmem_ful_cnt[local_quadwarp_id];

    int *s_blkid_smem_tny_local = &s_blkid_smem_tny[local_quadwarp_id * TILE_PER_QUADWARP];
    int *s_blkid_smem_sml_local = &s_blkid_smem_sml[local_quadwarp_id * TILE_PER_QUADWARP];
    int *s_blkid_smem_lrg_local = &s_blkid_smem_lrg[local_quadwarp_id * TILE_PER_QUADWARP];
    int *s_blkid_smem_dns_local = &s_blkid_smem_dns[local_quadwarp_id * TILE_PER_QUADWARP];
    int *s_blkid_smem_ful_local = &s_blkid_smem_ful[local_quadwarp_id * TILE_PER_QUADWARP];

    int tile_start = global_quadwarp_id * TILE_PER_QUADWARP;

    int tile_end = tile_start + TILE_PER_QUADWARP; //(global_warp_id + 1) * TPW;

    auto g8 = cg::tiled_partition<8>(cg::this_thread_block());

    if (!quadwarp_lane_id)
    {
        s_blksmem_tny_cnt_local[0] = 0;
        s_blksmem_sml_cnt_local[0] = 0;
        s_blksmem_lrg_cnt_local[0] = 0;
        s_blksmem_dns_cnt_local[0] = 0;
        s_blksmem_ful_cnt_local[0] = 0;
    }

    for (int tilei = tile_start; tilei < tile_end; tilei++)
    {
#pragma unroll
        for (int ri = 0; ri < MaskNumC; ri++){
            s_maskc_local[quadwarp_lane_id * MaskNumC + ri] = 0;
        }
        if (!quadwarp_lane_id)
        {
            s_matchedcnt_local[0] = 0;
        }

        int matchedcnt = 0;
        int lena = 0;
        int lenb = 0;
        int nnzcnt = 0;

        if (tilei < numblkC)
        {
            const int blki = d_blkrowidxC[tilei];
            const int blkj = d_blkcolidxC[tilei];

            const int abase = d_blkrowptrA[blki];
            const int astop = d_blkrowptrA[blki + 1];
            lena = astop - abase;

            const int bbase = ld_gbl_auto(d_blkcolptrB + blkj);
            const int bstop = ld_gbl_auto(d_blkcolptrB + blkj + 1);
            lenb = bstop - bbase;
            {
                int specres = 0;
                intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                 d_blkrowidxB, bbase, bstop, lenb,
                                                 s_matched_posa_local, s_matched_posb_local,
                                                 SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                 quadwarp_lane_id, QUADWARP_SIZE);
                g8.sync();
                matchedcnt = s_matchedcnt_local[0];

                if (matchedcnt == 0)
                {
                }
                else if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
                {
                    // save speculative posa and posb for step 4
                    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt <= GMEM_SPECULATIVE_INTERSECTION)
                    {
                        if (!quadwarp_lane_id)
                            d_spec_intersection_cnt[tilei] = matchedcnt;
                        for (int si = quadwarp_lane_id; si < matchedcnt; si += QUADWARP_SIZE)
                        {
                            d_spec_intersection_posa[tilei * GMEM_SPECULATIVE_INTERSECTION + si] = s_matched_posa_local[si];
                            d_spec_intersection_posb[tilei * GMEM_SPECULATIVE_INTERSECTION + si] = s_matched_posb_local[si];
                        }
                    }

                    for (int i = 0; i < matchedcnt; i++)
                    {
                        int posa = s_matched_posa_local[i];
                        int posb = s_matched_posb_local[i];

                        const int nnzastart = d_nnzb_A[(abase + posa)];
                        int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

#pragma unroll
                        for (int row = quadwarp_lane_id; row < TILE_SIZE_N; row += QUADWARP_SIZE)
                        {
#pragma unroll
                            for (int ri = 0; ri < MaskNumB; ri++)
                            {
                                s_blkmaskB_local[row * MaskNumB + ri] = 
                                    ld_gbl_auto(&d_blkmaskB[(bbase + posb) * TILE_SIZE_N * MaskNumB + row * MaskNumB + ri]);
                            }
                        }

                        for (int i = quadwarp_lane_id; i < nnztotala; i += QUADWARP_SIZE)
                        {
                            TILE_CSR_COL_TYPE_A rowcolidx = ld_gbl_auto(d_blkcsr_Col_A + nnzastart + i);
                            // rowcolidx：(ri * TILE_SIZE_N) + colidx
                            TILE_CSR_COL_TYPE_A row_in_B = rowcolidx % TILE_SIZE_N;
                            TILE_CSR_COL_TYPE_A row_in_C = rowcolidx / TILE_SIZE_N;
#pragma unroll
                            for (int ri = 0; ri < MaskNumC; ri++)
                            {
                                atomicOr(&s_maskc_local[row_in_C * MaskNumC + ri], s_blkmaskB_local[row_in_B * MaskNumB + ri]);
                            }
                        }
                    }
                }
                else
                {
                    const int astart = d_blkcolidxA[abase];
                    const int aend = d_blkcolidxA[astop - 1];
                    const int bstart = ld_gbl_auto(d_blkrowidxB + bbase);
                    const int bend = ld_gbl_auto(d_blkrowidxB + bstop - 1);

                    int posa_real = 0;
                    int posb_real = 0;
                    if (bstart > astart)
                    {
                        int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
                        posa_real = posa_real_new < 0 ? 0 : posa_real_new;
                    }
                    else if (bstart < astart)
                    {
                        int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
                        posb_real = posb_real_new < 0 ? 0 : posb_real_new;
                    }

                    if (bstop < astop)
                    {
                        int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
                        lena = lena_new > lena ? lena : lena_new;
                    }
                    else if (bstop > astop)
                    {
                        int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
                        lenb = lenb_new > lenb ? lenb : lenb_new;
                    }

                    int posa = posa_real;
                    int posb = posb_real;
                    int idxa = 0;
                    int idxb = 0;
                    int posa_updated = 1;
                    int posb_updated = 1;

                    while (posa < lena && posb < lenb)
                    {
                        idxa = posa_updated ? d_blkcolidxA[abase + posa] : idxa; //a[posa] : idxa;
                        idxb = posb_updated ? d_blkrowidxB[bbase + posb] : idxb; //b[posb] : idxb;

                        if (idxa == idxb)
                        {
                            const int nnzastart = d_nnzb_A[(abase + posa)];
                            int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

#pragma unroll
                            for (int row = quadwarp_lane_id; row < TILE_SIZE_N; row += QUADWARP_SIZE)
                            {
#pragma unroll
                                for (int ri = 0; ri < MaskNumB; ri++)
                                {
                                    s_blkmaskB_local[row * MaskNumB + ri] = 
                                        ld_gbl_auto(&d_blkmaskB[(bbase + posb) * TILE_SIZE_N * MaskNumB + row * MaskNumB + ri]);
                                }
                            }

                            for (int i = quadwarp_lane_id; i < nnztotala; i += QUADWARP_SIZE)
                            {
                                TILE_CSR_COL_TYPE_A rowcolidx = ld_gbl_auto(d_blkcsr_Col_A + nnzastart + i);
                                // rowcolidx：(ri * TILE_SIZE_N) + colidx
                                TILE_CSR_COL_TYPE_A row_in_B = rowcolidx % TILE_SIZE_N;
                                TILE_CSR_COL_TYPE_A row_in_C = rowcolidx / TILE_SIZE_N;
#pragma unroll
                                for (int ri = 0; ri < MaskNumC; ri++)
                                {
                                    atomicOr(&s_maskc_local[row_in_C * MaskNumC + ri], s_blkmaskB_local[row_in_B * MaskNumC + ri]);
                                }
                            }

                            posa++;
                            posa_updated = 1;
                            posb++;
                            posb_updated = 1;
                        }
                        else
                        {
                            // the smaller index goes forward
                            posa_updated = idxa < idxb ? 1 : 0;
                            posa += posa_updated;
                            posb_updated = idxa > idxb ? 1 : 0;
                            posb += posb_updated;
                        }
                    }
                }
            }
#pragma unroll
            for (int ri = 0; ri < MaskNumC; ri++)
            {
                nnzcnt += __popc(s_maskc_local[quadwarp_lane_id * MaskNumC + ri]);
            }
        }

        int nnzcnt_sum = sum_8_shfl(nnzcnt);

        int nnzcnt_scan = scan_32_shfl(nnzcnt, lane_id);

        nnzcnt_scan -= nnzcnt;
        nnzcnt_scan -= __shfl_sync(0xffffffff, nnzcnt_scan, (lane_id >> 3) << 3);

        if (tilei < numblkC && nnzcnt_sum)
        {
            long long int pos_c  = (long long int)(tilei) * TILE_SIZE_M + quadwarp_lane_id;
            d_blkcsr_Ptr_C[pos_c] = nnzcnt_scan;
#pragma unroll
            for (int maskid = 0; maskid < MaskNumC; maskid++)
            {
                d_blkmaskC[pos_c * MaskNumC + maskid] = (TILE_MASK_TYPE_B)s_maskc_local[quadwarp_lane_id * MaskNumC + maskid];
            }

            if (!quadwarp_lane_id)
            {
                d_nnzb_C[tilei] = nnzcnt_sum;

                if (nnzcnt_sum <= SMEM_TNY_TH && nnzcnt_sum != 0)
                {
                    int pos = atomicAdd(s_blksmem_tny_cnt_local, 1);
                    s_blkid_smem_tny_local[pos] = tilei;
                }
                else if (SMEM_TNY_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_SML_TH)
                {
                    int pos = atomicAdd(s_blksmem_sml_cnt_local, 1);
                    s_blkid_smem_sml_local[pos] = tilei;
                }
                else if (SMEM_SML_TH < nnzcnt_sum && nnzcnt_sum <= d_SMEM_LRG_TH)
                {
                    int pos = atomicAdd(s_blksmem_lrg_cnt_local, 1);
                    s_blkid_smem_lrg_local[pos] = tilei;
                }
                else if (d_SMEM_LRG_TH < nnzcnt_sum && nnzcnt_sum < SMEM_DNS_TH)
                {
                    int pos = atomicAdd(s_blksmem_dns_cnt_local, 1);
                    s_blkid_smem_dns_local[pos] = tilei;
                }
                else if (nnzcnt_sum >= SMEM_DNS_TH)
                {
                    int pos = atomicAdd(s_blksmem_ful_cnt_local, 1);
                    s_blkid_smem_ful_local[pos] = tilei;
                }
            }
        }
    }

    int len = s_blksmem_tny_cnt_local[0];
    int pos = 0;
    pos = quadwarp_lane_id == 0 ? atomicAdd(d_blksmem_tny_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 3) << 3);

    if (quadwarp_lane_id < len)
        d_blkid_smem_tny[pos + quadwarp_lane_id] = s_blkid_smem_tny_local[quadwarp_lane_id];

    len = s_blksmem_sml_cnt_local[0];
    pos = quadwarp_lane_id == 0 ? atomicAdd(d_blksmem_sml_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 3) << 3);
    if (quadwarp_lane_id < len)
        d_blkid_smem_sml[pos + quadwarp_lane_id] = s_blkid_smem_sml_local[quadwarp_lane_id];

    len = s_blksmem_lrg_cnt_local[0];
    pos = quadwarp_lane_id == 0 ? atomicAdd(d_blksmem_lrg_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 3) << 3);
    if (quadwarp_lane_id < len)
        d_blkid_smem_lrg[pos + quadwarp_lane_id] = s_blkid_smem_lrg_local[quadwarp_lane_id];

    len = s_blksmem_dns_cnt_local[0];
    pos = quadwarp_lane_id == 0 ? atomicAdd(d_blksmem_dns_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 3) << 3);

    if (quadwarp_lane_id < len)
        d_blkid_smem_dns[pos + quadwarp_lane_id] = s_blkid_smem_dns_local[quadwarp_lane_id];

    len = s_blksmem_ful_cnt_local[0];
    pos = quadwarp_lane_id == 0 ? atomicAdd(d_blksmem_ful_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 3) << 3);
    if (quadwarp_lane_id < len)
        d_blkid_smem_ful[pos + quadwarp_lane_id] = s_blkid_smem_ful_local[quadwarp_lane_id];
}

__global__ void tile_spgemm_step3_cuda_kernel_2level_halfwarp(const int *d_blkrowptrA,
                                                              const int *__restrict__ d_blkcolidxA,
                                                              const int *d_nnzb_A,
                                                              MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                              TILE_CSR_COL_TYPE_A *d_blkcsr_Col_A,
                                                              TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_A,
                                                              TILE_MASK_TYPE_A *d_blkmaskA,
                                                              int blkmA, int blknA, int numblkA, int nnzA,
                                                              const int *__restrict__ d_blkcolptrB,
                                                              const int *__restrict__ d_blkrowidxB,
                                                              const int *__restrict__ d_nnzb_B,
                                                              const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                              const TILE_CSR_COL_TYPE_B *__restrict__ d_blkcsr_Col_B,
                                                              const TILE_CSR_PTR_TYPE *__restrict__ d_blkcsr_Ptr_B,
                                                              const TILE_MASK_TYPE_B *__restrict__ d_blkmaskB,
                                                              int blkmB, int blknB, int numblkB, int nnzB,
                                                              unsigned int *d_blk_intersec_bitmask_A,
                                                              unsigned int *d_blk_intersec_bitmask_B,
                                                              int blk_intersec_bitmask_len,
                                                              int *d_blkrowidxC,
                                                              int *d_blkcolidxC,
                                                              TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_C,
                                                              int *d_nnzb_C,
                                                              TILE_MASK_TYPE_B *d_blkmaskC,
                                                              int *d_blksmem_tny_cnt,
                                                              int *d_blksmem_sml_cnt,
                                                              int *d_blksmem_lrg_cnt,
                                                              int *d_blksmem_dns_cnt,
                                                              int *d_blksmem_ful_cnt,
                                                              int *d_blkid_smem_tny,
                                                              int *d_blkid_smem_sml,
                                                              int *d_blkid_smem_lrg,
                                                              int *d_blkid_smem_dns,
                                                              int *d_blkid_smem_ful,
                                                              int *d_spec_intersection_cnt,
                                                              int *d_spec_intersection_posa,
                                                              int *d_spec_intersection_posb,
                                                              int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_halfwarp_id = global_id >> 4; //global_id / HALFWARP_SIZE;

    __shared__ unsigned int s_maskc[HALFWARP_PER_BLOCK * TILE_SIZE_M * MaskNumC];
    __shared__ TILE_MASK_TYPE_B s_blkmaskB[HALFWARP_PER_BLOCK * TILE_SIZE_N * MaskNumB];

    __shared__ int s_matched_posa[HALFWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[HALFWARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[HALFWARP_PER_BLOCK];

    __shared__ int s_blksmem_tny_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_sml_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_lrg_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_dns_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_ful_cnt[HALFWARP_PER_BLOCK];

    __shared__ int s_blkid_smem_tny[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_sml[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_lrg[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_dns[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];
    __shared__ int s_blkid_smem_ful[HALFWARP_PER_BLOCK * TILE_PER_HALFWARP];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int halfwarp_lane_id = (HALFWARP_SIZE - 1) & threadIdx.x;

    const int local_halfwarp_id = threadIdx.x >> 4; //threadIdx.x / HALFWARP_SIZE;

    unsigned int *s_maskc_local = &s_maskc[local_halfwarp_id * TILE_SIZE_M * MaskNumC];
    TILE_MASK_TYPE_B *s_blkmaskB_local = &s_blkmaskB[local_halfwarp_id * TILE_SIZE_N * MaskNumB];

    int *s_matched_posa_local = &s_matched_posa[local_halfwarp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_halfwarp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_halfwarp_id];
    int *s_blksmem_tny_cnt_local = &s_blksmem_tny_cnt[local_halfwarp_id];
    int *s_blksmem_sml_cnt_local = &s_blksmem_sml_cnt[local_halfwarp_id];
    int *s_blksmem_lrg_cnt_local = &s_blksmem_lrg_cnt[local_halfwarp_id];
    int *s_blksmem_dns_cnt_local = &s_blksmem_dns_cnt[local_halfwarp_id];
    int *s_blksmem_ful_cnt_local = &s_blksmem_ful_cnt[local_halfwarp_id];

    int *s_blkid_smem_tny_local = &s_blkid_smem_tny[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_sml_local = &s_blkid_smem_sml[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_lrg_local = &s_blkid_smem_lrg[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_dns_local = &s_blkid_smem_dns[local_halfwarp_id * TILE_PER_HALFWARP];
    int *s_blkid_smem_ful_local = &s_blkid_smem_ful[local_halfwarp_id * TILE_PER_HALFWARP];

    int tile_start = global_halfwarp_id * TILE_PER_HALFWARP;

    int tile_end = tile_start + TILE_PER_HALFWARP; //(global_warp_id + 1) * TPW;

    auto g16 = cg::tiled_partition<16>(cg::this_thread_block());

    if (!halfwarp_lane_id)
    {
        s_blksmem_tny_cnt_local[0] = 0;
        s_blksmem_sml_cnt_local[0] = 0;
        s_blksmem_lrg_cnt_local[0] = 0;
        s_blksmem_dns_cnt_local[0] = 0;
        s_blksmem_ful_cnt_local[0] = 0;
    }

    for (int tilei = tile_start; tilei < tile_end; tilei++)
    {
#pragma unroll
        for (int ri = 0; ri < MaskNumC; ri++){
            s_maskc_local[halfwarp_lane_id * MaskNumC + ri] = 0;
        }
        if (!halfwarp_lane_id)
        {
            s_matchedcnt_local[0] = 0;
        }

        int matchedcnt = 0;
        int lena = 0;
        int lenb = 0;
        int nnzcnt = 0;

        if (tilei < numblkC)
        {
            const int blki = d_blkrowidxC[tilei];
            const int blkj = d_blkcolidxC[tilei];

            const int abase = d_blkrowptrA[blki];
            const int astop = d_blkrowptrA[blki + 1];
            lena = astop - abase;

            const int bbase = ld_gbl_auto(d_blkcolptrB + blkj);
            const int bstop = ld_gbl_auto(d_blkcolptrB + blkj + 1);
            lenb = bstop - bbase;
            {
                int specres = 0;
                intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                 d_blkrowidxB, bbase, bstop, lenb,
                                                 s_matched_posa_local, s_matched_posb_local,
                                                 SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                 halfwarp_lane_id, HALFWARP_SIZE);
                g16.sync();
                matchedcnt = s_matchedcnt_local[0];

                if (matchedcnt == 0)
                {
                }
                else if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
                {
                    // save speculative posa and posb for step 4
                    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt <= GMEM_SPECULATIVE_INTERSECTION)
                    {
                        if (!halfwarp_lane_id)
                            d_spec_intersection_cnt[tilei] = matchedcnt;
                        for (int si = halfwarp_lane_id; si < matchedcnt; si += HALFWARP_SIZE)
                        {
                            d_spec_intersection_posa[tilei * GMEM_SPECULATIVE_INTERSECTION + si] = s_matched_posa_local[si];
                            d_spec_intersection_posb[tilei * GMEM_SPECULATIVE_INTERSECTION + si] = s_matched_posb_local[si];
                        }
                    }

                    for (int i = 0; i < matchedcnt; i++)
                    {
                        int posa = s_matched_posa_local[i];
                        int posb = s_matched_posb_local[i];

                        const int nnzastart = d_nnzb_A[(abase + posa)];
                        int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

#pragma unroll
                        for (int row = halfwarp_lane_id; row < TILE_SIZE_N; row += HALFWARP_SIZE)
                        {
#pragma unroll
                            for (int ri = 0; ri < MaskNumB; ri++)
                            {
                                s_blkmaskB_local[row * MaskNumB + ri] = 
                                    ld_gbl_auto(&d_blkmaskB[(bbase + posb) * TILE_SIZE_N * MaskNumB + row * MaskNumB + ri]);
                            }
                        }

                        for (int i = halfwarp_lane_id; i < nnztotala; i += HALFWARP_SIZE)
                        {
                            TILE_CSR_COL_TYPE_A rowcolidx = ld_gbl_auto(d_blkcsr_Col_A + nnzastart + i);
                            // rowcolidx：(ri * TILE_SIZE_N) + colidx
                            TILE_CSR_COL_TYPE_A row_in_B = rowcolidx % TILE_SIZE_N;
                            TILE_CSR_COL_TYPE_A row_in_C = rowcolidx / TILE_SIZE_N;
#pragma unroll
                            for (int ri = 0; ri < MaskNumC; ri++)
                            {
                                atomicOr(&s_maskc_local[row_in_C * MaskNumC + ri], s_blkmaskB_local[row_in_B * MaskNumB + ri]);
                            }
                        }
                    }
                }
                else
                {
                    const int astart = d_blkcolidxA[abase];
                    const int aend = d_blkcolidxA[astop - 1];
                    const int bstart = ld_gbl_auto(d_blkrowidxB + bbase);
                    const int bend = ld_gbl_auto(d_blkrowidxB + bstop - 1);

                    int posa_real = 0;
                    int posb_real = 0;
                    if (bstart > astart)
                    {
                        int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
                        posa_real = posa_real_new < 0 ? 0 : posa_real_new;
                    }
                    else if (bstart < astart)
                    {
                        int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
                        posb_real = posb_real_new < 0 ? 0 : posb_real_new;
                    }

                    if (bstop < astop)
                    {
                        int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
                        lena = lena_new > lena ? lena : lena_new;
                    }
                    else if (bstop > astop)
                    {
                        int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
                        lenb = lenb_new > lenb ? lenb : lenb_new;
                    }

                    int posa = posa_real;
                    int posb = posb_real;
                    int idxa = 0;
                    int idxb = 0;
                    int posa_updated = 1;
                    int posb_updated = 1;

                    while (posa < lena && posb < lenb)
                    {
                        idxa = posa_updated ? d_blkcolidxA[abase + posa] : idxa; //a[posa] : idxa;
                        idxb = posb_updated ? d_blkrowidxB[bbase + posb] : idxb; //b[posb] : idxb;

                        if (idxa == idxb)
                        {
                            const int nnzastart = d_nnzb_A[(abase + posa)];
                            int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

#pragma unroll
                            for (int row = halfwarp_lane_id; row < TILE_SIZE_N; row += HALFWARP_SIZE)
                            {
#pragma unroll
                                for (int ri = 0; ri < MaskNumB; ri++)
                                {
                                    s_blkmaskB_local[row * MaskNumB + ri] = 
                                        ld_gbl_auto(&d_blkmaskB[(bbase + posb) * TILE_SIZE_N * MaskNumB + row * MaskNumB + ri]);
                                }
                            }

                            for (int i = halfwarp_lane_id; i < nnztotala; i += HALFWARP_SIZE)
                            {
                                TILE_CSR_COL_TYPE_A rowcolidx = ld_gbl_auto(d_blkcsr_Col_A + nnzastart + i);
                                // rowcolidx：(ri * TILE_SIZE_N) + colidx
                                TILE_CSR_COL_TYPE_A row_in_B = rowcolidx % TILE_SIZE_N;
                                TILE_CSR_COL_TYPE_A row_in_C = rowcolidx / TILE_SIZE_N;
#pragma unroll
                                for (int ri = 0; ri < MaskNumC; ri++)
                                {
                                    atomicOr(&s_maskc_local[row_in_C * MaskNumC + ri], s_blkmaskB_local[row_in_B * MaskNumC + ri]);
                                }
                            }

                            posa++;
                            posa_updated = 1;
                            posb++;
                            posb_updated = 1;
                        }
                        else
                        {
                            // the smaller index goes forward
                            posa_updated = idxa < idxb ? 1 : 0;
                            posa += posa_updated;
                            posb_updated = idxa > idxb ? 1 : 0;
                            posb += posb_updated;
                        }
                    }
                }
            }
#pragma unroll
            for (int ri = 0; ri < MaskNumC; ri++)
            {
                nnzcnt += __popc(s_maskc_local[halfwarp_lane_id * MaskNumC + ri]);
            }
        }

        int nnzcnt_sum = sum_16_shfl_cg(nnzcnt);

        int nnzcnt_scan = scan_32_shfl(nnzcnt, lane_id);

        nnzcnt_scan -= nnzcnt;
        nnzcnt_scan -= __shfl_sync(0xffffffff, nnzcnt_scan, (lane_id >> 4) << 4);

        if (tilei < numblkC && nnzcnt_sum)
        {
            long long int pos_c  = (long long int)(tilei) * TILE_SIZE_M + halfwarp_lane_id;
            d_blkcsr_Ptr_C[pos_c] = nnzcnt_scan; // - nnzcnt;
#pragma unroll
            for (int maskid = 0; maskid < MaskNumC; maskid++)
            {
                d_blkmaskC[pos_c * MaskNumC + maskid] = (TILE_MASK_TYPE_B)s_maskc_local[halfwarp_lane_id * MaskNumC + maskid];
            }

            if (!halfwarp_lane_id)
            {
                d_nnzb_C[tilei] = nnzcnt_sum;

                if (nnzcnt_sum <= SMEM_TNY_TH && nnzcnt_sum != 0)
                {
                    int pos = atomicAdd(s_blksmem_tny_cnt_local, 1);
                    s_blkid_smem_tny_local[pos] = tilei;
                }
                else if (SMEM_TNY_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_SML_TH)
                {
                    int pos = atomicAdd(s_blksmem_sml_cnt_local, 1);
                    s_blkid_smem_sml_local[pos] = tilei;
                }
                else if (SMEM_SML_TH < nnzcnt_sum && nnzcnt_sum <= d_SMEM_LRG_TH)
                {
                    int pos = atomicAdd(s_blksmem_lrg_cnt_local, 1);
                    s_blkid_smem_lrg_local[pos] = tilei;
                }
                else if (d_SMEM_LRG_TH < nnzcnt_sum && nnzcnt_sum < SMEM_DNS_TH)
                {
                    int pos = atomicAdd(s_blksmem_dns_cnt_local, 1);
                    s_blkid_smem_dns_local[pos] = tilei;
                }
                else if (nnzcnt_sum >= SMEM_DNS_TH)
                {
                    int pos = atomicAdd(s_blksmem_ful_cnt_local, 1);
                    s_blkid_smem_ful_local[pos] = tilei;
                }
            }
        }
    }

    int len = s_blksmem_tny_cnt_local[0];
    int pos = 0;
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_tny_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);

    if (halfwarp_lane_id < len)
        d_blkid_smem_tny[pos + halfwarp_lane_id] = s_blkid_smem_tny_local[halfwarp_lane_id];

    len = s_blksmem_sml_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_sml_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_sml[pos + halfwarp_lane_id] = s_blkid_smem_sml_local[halfwarp_lane_id];

    len = s_blksmem_lrg_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_lrg_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_lrg[pos + halfwarp_lane_id] = s_blkid_smem_lrg_local[halfwarp_lane_id];

    len = s_blksmem_dns_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_dns_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);

    if (halfwarp_lane_id < len)
        d_blkid_smem_dns[pos + halfwarp_lane_id] = s_blkid_smem_dns_local[halfwarp_lane_id];

    len = s_blksmem_ful_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_ful_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_ful[pos + halfwarp_lane_id] = s_blkid_smem_ful_local[halfwarp_lane_id];
}

__global__ void tile_spgemm_step3_cuda_kernel_2level_warp(const int *d_blkrowptrA,
                                                              const int *__restrict__ d_blkcolidxA,
                                                              const int *d_nnzb_A,
                                                              MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                              TILE_CSR_COL_TYPE_A *d_blkcsr_Col_A,
                                                              TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_A,
                                                              TILE_MASK_TYPE_A *d_blkmaskA,
                                                              int blkmA, int blknA, int numblkA, int nnzA,
                                                              const int *__restrict__ d_blkcolptrB,
                                                              const int *__restrict__ d_blkrowidxB,
                                                              const int *__restrict__ d_nnzb_B,
                                                              const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                              const TILE_CSR_COL_TYPE_B *__restrict__ d_blkcsr_Col_B,
                                                              const TILE_CSR_PTR_TYPE *__restrict__ d_blkcsr_Ptr_B,
                                                              const TILE_MASK_TYPE_B *__restrict__ d_blkmaskB,
                                                              int blkmB, int blknB, int numblkB, int nnzB,
                                                              unsigned int *d_blk_intersec_bitmask_A,
                                                              unsigned int *d_blk_intersec_bitmask_B,
                                                              int blk_intersec_bitmask_len,
                                                              int *d_blkrowidxC,
                                                              int *d_blkcolidxC,
                                                              TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_C,
                                                              int *d_nnzb_C,
                                                              TILE_MASK_TYPE_B *d_blkmaskC,
                                                              int *d_blksmem_tny_cnt,
                                                              int *d_blksmem_sml_cnt,
                                                              int *d_blksmem_lrg_cnt,
                                                              int *d_blksmem_dns_cnt,
                                                              int *d_blksmem_ful_cnt,
                                                              int *d_blkid_smem_tny,
                                                              int *d_blkid_smem_sml,
                                                              int *d_blkid_smem_lrg,
                                                              int *d_blkid_smem_dns,
                                                              int *d_blkid_smem_ful,
                                                              int *d_spec_intersection_cnt,
                                                              int *d_spec_intersection_posa,
                                                              int *d_spec_intersection_posb,
                                                              int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_warp_id = global_id >> 5; //global_id / WARP_SIZE;

    // TODO: Optimize
    __shared__ unsigned int s_maskc[WARP_PER_BLOCK * TILE_SIZE_M * MaskNumC];
    __shared__ TILE_MASK_TYPE_B s_blkmaskB[WARP_PER_BLOCK * TILE_SIZE_N * MaskNumB];

    __shared__ int s_matched_posa[WARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[WARP_PER_BLOCK * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[WARP_PER_BLOCK];

    __shared__ int s_blksmem_tny_cnt[WARP_PER_BLOCK];
    __shared__ int s_blksmem_sml_cnt[WARP_PER_BLOCK];
    __shared__ int s_blksmem_lrg_cnt[WARP_PER_BLOCK];
    __shared__ int s_blksmem_dns_cnt[WARP_PER_BLOCK];
    __shared__ int s_blksmem_ful_cnt[WARP_PER_BLOCK];

    __shared__ int s_blkid_smem_tny[WARP_PER_BLOCK * TILE_PER_WARP];
    __shared__ int s_blkid_smem_sml[WARP_PER_BLOCK * TILE_PER_WARP];
    __shared__ int s_blkid_smem_lrg[WARP_PER_BLOCK * TILE_PER_WARP];
    __shared__ int s_blkid_smem_dns[WARP_PER_BLOCK * TILE_PER_WARP];
    __shared__ int s_blkid_smem_ful[WARP_PER_BLOCK * TILE_PER_WARP];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int warp_lane_id = (WARP_SIZE - 1) & threadIdx.x;

    const int local_warp_id = threadIdx.x >> 5; //threadIdx.x / WARP_SIZE;

    unsigned int *s_maskc_local = &s_maskc[local_warp_id * TILE_SIZE_M * MaskNumC];
    TILE_MASK_TYPE_B *s_blkmaskB_local = &s_blkmaskB[local_warp_id * TILE_SIZE_N * MaskNumB];

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];
    int *s_blksmem_tny_cnt_local = &s_blksmem_tny_cnt[local_warp_id];
    int *s_blksmem_sml_cnt_local = &s_blksmem_sml_cnt[local_warp_id];
    int *s_blksmem_lrg_cnt_local = &s_blksmem_lrg_cnt[local_warp_id];
    int *s_blksmem_dns_cnt_local = &s_blksmem_dns_cnt[local_warp_id];
    int *s_blksmem_ful_cnt_local = &s_blksmem_ful_cnt[local_warp_id];

    int *s_blkid_smem_tny_local = &s_blkid_smem_tny[local_warp_id * TILE_PER_WARP];
    int *s_blkid_smem_sml_local = &s_blkid_smem_sml[local_warp_id * TILE_PER_WARP];
    int *s_blkid_smem_lrg_local = &s_blkid_smem_lrg[local_warp_id * TILE_PER_WARP];
    int *s_blkid_smem_dns_local = &s_blkid_smem_dns[local_warp_id * TILE_PER_WARP];
    int *s_blkid_smem_ful_local = &s_blkid_smem_ful[local_warp_id * TILE_PER_WARP];

    int tile_start = global_warp_id * TILE_PER_WARP;

    int tile_end = tile_start + TILE_PER_WARP; //(global_warp_id + 1) * TPW;

    auto g32 = cg::tiled_partition<32>(cg::this_thread_block());

    if (!warp_lane_id)
    {
        s_blksmem_tny_cnt_local[0] = 0;
        s_blksmem_sml_cnt_local[0] = 0;
        s_blksmem_lrg_cnt_local[0] = 0;
        s_blksmem_dns_cnt_local[0] = 0;
        s_blksmem_ful_cnt_local[0] = 0;
    }

    for (int tilei = tile_start; tilei < tile_end; tilei++)
    {
#pragma unroll
        for (int C_warp_idx = warp_lane_id; C_warp_idx < TILE_SIZE_M; C_warp_idx += WARP_SIZE){
#pragma unroll
            for (int ri = 0; ri < MaskNumC; ri++){
                s_maskc_local[C_warp_idx * MaskNumC + ri] = 0;
            }
        }
        if (!warp_lane_id)
        {
            s_matchedcnt_local[0] = 0;
        }

        int matchedcnt = 0;
        int lena = 0;
        int lenb = 0;
        int nnzcnt[5] = {};

        if (tilei < numblkC)
        {
            const int blki = d_blkrowidxC[tilei];
            const int blkj = d_blkcolidxC[tilei];

            const int abase = d_blkrowptrA[blki];
            const int astop = d_blkrowptrA[blki + 1];
            lena = astop - abase;

            const int bbase = ld_gbl_auto(d_blkcolptrB + blkj);
            const int bstop = ld_gbl_auto(d_blkcolptrB + blkj + 1);
            lenb = bstop - bbase;
            {
                int specres = 0;
                intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                 d_blkrowidxB, bbase, bstop, lenb,
                                                 s_matched_posa_local, s_matched_posb_local,
                                                 SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                 warp_lane_id, WARP_SIZE);
                g32.sync();
                matchedcnt = s_matchedcnt_local[0];

                if (matchedcnt == 0)
                {
                }
                else if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
                {
                    // save speculative posa and posb for step 4
                    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt <= GMEM_SPECULATIVE_INTERSECTION)
                    {
                        if (!warp_lane_id)
                            d_spec_intersection_cnt[tilei] = matchedcnt;
                        for (int si = warp_lane_id; si < matchedcnt; si += WARP_SIZE)
                        {
                            d_spec_intersection_posa[tilei * GMEM_SPECULATIVE_INTERSECTION + si] = s_matched_posa_local[si];
                            d_spec_intersection_posb[tilei * GMEM_SPECULATIVE_INTERSECTION + si] = s_matched_posb_local[si];
                        }
                    }

                    for (int i = 0; i < matchedcnt; i++)
                    {
                        int posa = s_matched_posa_local[i];
                        int posb = s_matched_posb_local[i];

                        const int nnzastart = d_nnzb_A[(abase + posa)];
                        int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

#pragma unroll
                        for (int row = warp_lane_id; row < TILE_SIZE_N; row += WARP_SIZE)
                        {
#pragma unroll
                            for (int ri = 0; ri < MaskNumB; ri++)
                            {
                                s_blkmaskB_local[row * MaskNumB + ri] = 
                                    ld_gbl_auto(&d_blkmaskB[(bbase + posb) * TILE_SIZE_N * MaskNumB + row * MaskNumB + ri]);
                            }
                        }

                        for (int i = warp_lane_id; i < nnztotala; i += WARP_SIZE)
                        {
                            TILE_CSR_COL_TYPE_A rowcolidx = ld_gbl_auto(d_blkcsr_Col_A + nnzastart + i);
                            // rowcolidx：(ri * TILE_SIZE_N) + colidx
                            TILE_CSR_COL_TYPE_A row_in_B = rowcolidx % TILE_SIZE_N;
                            TILE_CSR_COL_TYPE_A row_in_C = rowcolidx / TILE_SIZE_N;
#pragma unroll
                            for (int ri = 0; ri < MaskNumC; ri++)
                            {
                                atomicOr(&s_maskc_local[row_in_C * MaskNumC + ri], s_blkmaskB_local[row_in_B * MaskNumB + ri]);
                            }
                        }
                    }
                }
                else
                {
                    const int astart = d_blkcolidxA[abase];
                    const int aend = d_blkcolidxA[astop - 1];
                    const int bstart = ld_gbl_auto(d_blkrowidxB + bbase);
                    const int bend = ld_gbl_auto(d_blkrowidxB + bstop - 1);

                    int posa_real = 0;
                    int posb_real = 0;
                    if (bstart > astart)
                    {
                        int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
                        posa_real = posa_real_new < 0 ? 0 : posa_real_new;
                    }
                    else if (bstart < astart)
                    {
                        int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
                        posb_real = posb_real_new < 0 ? 0 : posb_real_new;
                    }

                    if (bstop < astop)
                    {
                        int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
                        lena = lena_new > lena ? lena : lena_new;
                    }
                    else if (bstop > astop)
                    {
                        int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
                        lenb = lenb_new > lenb ? lenb : lenb_new;
                    }

                    int posa = posa_real;
                    int posb = posb_real;
                    int idxa = 0;
                    int idxb = 0;
                    int posa_updated = 1;
                    int posb_updated = 1;

                    while (posa < lena && posb < lenb)
                    {
                        idxa = posa_updated ? d_blkcolidxA[abase + posa] : idxa; //a[posa] : idxa;
                        idxb = posb_updated ? d_blkrowidxB[bbase + posb] : idxb; //b[posb] : idxb;

                        if (idxa == idxb)
                        {
                            const int nnzastart = d_nnzb_A[(abase + posa)];
                            int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

#pragma unroll
                            for (int row = warp_lane_id; row < TILE_SIZE_N; row += WARP_SIZE)
                            {
#pragma unroll
                                for (int ri = 0; ri < MaskNumB; ri++)
                                {
                                    s_blkmaskB_local[row * MaskNumB + ri] = 
                                        ld_gbl_auto(&d_blkmaskB[(bbase + posb) * TILE_SIZE_N * MaskNumB + row * MaskNumB + ri]);
                                }
                            }

                            for (int i = warp_lane_id; i < nnztotala; i += WARP_SIZE)
                            {
                                TILE_CSR_COL_TYPE_A rowcolidx = ld_gbl_auto(d_blkcsr_Col_A + nnzastart + i);
                                // rowcolidx：(ri * TILE_SIZE_N) + colidx
                                TILE_CSR_COL_TYPE_A row_in_B = rowcolidx % TILE_SIZE_N;
                                TILE_CSR_COL_TYPE_A row_in_C = rowcolidx / TILE_SIZE_N;
#pragma unroll
                                for (int ri = 0; ri < MaskNumC; ri++)
                                {
                                    atomicOr(&s_maskc_local[row_in_C * MaskNumC + ri], s_blkmaskB_local[row_in_B * MaskNumC + ri]);
                                }
                            }

                            posa++;
                            posa_updated = 1;
                            posb++;
                            posb_updated = 1;
                        }
                        else
                        {
                            // the smaller index goes forward
                            posa_updated = idxa < idxb ? 1 : 0;
                            posa += posa_updated;
                            posb_updated = idxa > idxb ? 1 : 0;
                            posb += posb_updated;
                        }
                    }
                }
            }

#pragma unroll
            for (int C_warp_idx = warp_lane_id; C_warp_idx < TILE_SIZE_M; C_warp_idx += WARP_SIZE){
#pragma unroll
                for (int ri = 0; ri < MaskNumC; ri++)
                {
                    nnzcnt[C_warp_idx / WARP_SIZE] += __popc(s_maskc_local[C_warp_idx * MaskNumC + ri]);
                }
            }
        }

        int nnzcnt_sum_[5] = {};
        int nnzcnt_scan[5] = {};

#pragma unroll
        for (int temp = 0; temp < (TILE_SIZE_M + WARP_SIZE - 1) / WARP_SIZE; temp++){
            nnzcnt_sum_[temp] = sum_32_shfl(nnzcnt[temp]);
            nnzcnt_scan[temp] = scan_32_shfl(nnzcnt[temp], lane_id);
            nnzcnt_scan[temp] -= nnzcnt[temp];
            nnzcnt_scan[temp] -= __shfl_sync(0xffffffff, nnzcnt_scan[temp], (lane_id >> 5) << 5);
        }

        int nnzcnt_sum = nnzcnt_sum_[0];

#pragma unroll
        for (int temp = 1; temp < (TILE_SIZE_M + WARP_SIZE - 1)  / WARP_SIZE; temp++){
            nnzcnt_scan[temp] += nnzcnt_sum;
            nnzcnt_sum += nnzcnt_sum_[temp];
        }

        if (tilei < numblkC && nnzcnt_sum)
        {
#pragma unroll
            for (int C_warp_idx = warp_lane_id; C_warp_idx < TILE_SIZE_M; C_warp_idx += WARP_SIZE){
                long long int pos_c  = (long long int)(tilei) * TILE_SIZE_M + C_warp_idx;
                d_blkcsr_Ptr_C[pos_c] = nnzcnt_scan[C_warp_idx / WARP_SIZE];
#pragma unroll
                for (int maskid = 0; maskid < MaskNumC; maskid++)
                {
                    d_blkmaskC[pos_c * MaskNumC + maskid] = (TILE_MASK_TYPE_B)s_maskc_local[C_warp_idx * MaskNumC + maskid];
                }

                if (!C_warp_idx)
                {
                    d_nnzb_C[tilei] = nnzcnt_sum;

                    if (nnzcnt_sum <= SMEM_TNY_TH && nnzcnt_sum != 0)
                    {
                        int pos = atomicAdd(s_blksmem_tny_cnt_local, 1);
                        s_blkid_smem_tny_local[pos] = tilei;
                    }
                    else if (SMEM_TNY_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_SML_TH)
                    {
                        int pos = atomicAdd(s_blksmem_sml_cnt_local, 1);
                        s_blkid_smem_sml_local[pos] = tilei;
                    }
                    else if (SMEM_SML_TH < nnzcnt_sum && nnzcnt_sum <= d_SMEM_LRG_TH)
                    {
                        int pos = atomicAdd(s_blksmem_lrg_cnt_local, 1);
                        s_blkid_smem_lrg_local[pos] = tilei;
                    }
                    else if (d_SMEM_LRG_TH < nnzcnt_sum && nnzcnt_sum < SMEM_DNS_TH)
                    {
                        int pos = atomicAdd(s_blksmem_dns_cnt_local, 1);
                        s_blkid_smem_dns_local[pos] = tilei;
                    }
                    else if (nnzcnt_sum >= SMEM_DNS_TH)
                    {
                        int pos = atomicAdd(s_blksmem_ful_cnt_local, 1);
                        s_blkid_smem_ful_local[pos] = tilei;
                    }
                }
            }
        }
    }

    int len = s_blksmem_tny_cnt_local[0];
    int pos = 0;
    pos = warp_lane_id == 0 ? atomicAdd(d_blksmem_tny_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 5) << 5);

    if (warp_lane_id < len)
        d_blkid_smem_tny[pos + warp_lane_id] = s_blkid_smem_tny_local[warp_lane_id];

    len = s_blksmem_sml_cnt_local[0];
    pos = warp_lane_id == 0 ? atomicAdd(d_blksmem_sml_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 5) << 5);
    if (warp_lane_id < len)
        d_blkid_smem_sml[pos + warp_lane_id] = s_blkid_smem_sml_local[warp_lane_id];

    len = s_blksmem_lrg_cnt_local[0];
    pos = warp_lane_id == 0 ? atomicAdd(d_blksmem_lrg_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 5) << 5);
    if (warp_lane_id < len)
        d_blkid_smem_lrg[pos + warp_lane_id] = s_blkid_smem_lrg_local[warp_lane_id];

    len = s_blksmem_dns_cnt_local[0];
    pos = warp_lane_id == 0 ? atomicAdd(d_blksmem_dns_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 5) << 5);

    if (warp_lane_id < len)
        d_blkid_smem_dns[pos + warp_lane_id] = s_blkid_smem_dns_local[warp_lane_id];

    len = s_blksmem_ful_cnt_local[0];
    pos = warp_lane_id == 0 ? atomicAdd(d_blksmem_ful_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 5) << 5);
    if (warp_lane_id < len)
        d_blkid_smem_ful[pos + warp_lane_id] = s_blkid_smem_ful_local[warp_lane_id];
}

__global__ void tile_spgemm_step3_cuda_kernel_dns_halfwarp(const int *d_blkrowptrA,
                                                           const int *__restrict__ d_blkcolidxA,
                                                           const int *__restrict__ d_nnzb_A,
                                                           MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                           TILE_CSR_COL_TYPE_A *__restrict__ d_blkcsr_Col_A,
                                                           TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_A,
                                                           TILE_MASK_TYPE_A *d_blkmaskA,
                                                           int blkmA, int blknA, int numblkA, int nnzA,
                                                           const int *__restrict__ d_blkcolptrB,
                                                           const int *__restrict__ d_blkrowidxB,
                                                           const int *__restrict__ d_nnzb_B,
                                                           const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                           const TILE_CSR_COL_TYPE_B *__restrict__ d_blkcsr_Col_B,
                                                           const TILE_CSR_PTR_TYPE *__restrict__ d_blkcsr_Ptr_B,
                                                           const TILE_MASK_TYPE_B *__restrict__ d_blkmaskB,
                                                           int blkmB, int blknB, int numblkB, int nnzB,
                                                           unsigned int *d_blk_intersec_bitmask_A,
                                                           unsigned int *d_blk_intersec_bitmask_B,
                                                           int blk_intersec_bitmask_len,
                                                           int *d_blkrowidxC,
                                                           int *d_blkcolidxC,
                                                           TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_C,
                                                           int *d_nnzb_C,
                                                           TILE_MASK_TYPE_B *d_blkmaskC,
                                                           int *d_blksmem_tny_cnt,
                                                           int *d_blksmem_sml_cnt,
                                                           int *d_blksmem_lrg_cnt,
                                                           int *d_blksmem_dns_cnt,
                                                           int *d_blksmem_ful_cnt,
                                                           int *d_blkid_smem_tny,
                                                           int *d_blkid_smem_sml,
                                                           int *d_blkid_smem_lrg,
                                                           int *d_blkid_smem_dns,
                                                           int *d_blkid_smem_ful,
                                                           int *d_spec_intersection_cnt,
                                                           int *d_spec_intersection_posa,
                                                           int *d_spec_intersection_posb,
                                                           int numblkC)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int global_halfwarp_id = global_id >> 4; // global_id / HALFWARP_SIZE;

    // TODO: Optimization
    __shared__ unsigned int s_maskc[HALFWARP_PER_BLOCK * TILE_SIZE_M * MaskNumC];
    __shared__ TILE_MASK_TYPE_B s_blkmaskB[HALFWARP_PER_BLOCK * TILE_SIZE_N * MaskNumB];

    __shared__ int s_blksmem_tny_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_sml_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_lrg_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_dns_cnt[HALFWARP_PER_BLOCK];
    __shared__ int s_blksmem_ful_cnt[HALFWARP_PER_BLOCK];

    // total tile with blkid
    __shared__ int s_blkid_smem_tny[HALFWARP_PER_BLOCK * TILE_PER_ADAPTIVE_WARP];
    __shared__ int s_blkid_smem_sml[HALFWARP_PER_BLOCK * TILE_PER_ADAPTIVE_WARP];
    __shared__ int s_blkid_smem_lrg[HALFWARP_PER_BLOCK * TILE_PER_ADAPTIVE_WARP];
    __shared__ int s_blkid_smem_dns[HALFWARP_PER_BLOCK * TILE_PER_ADAPTIVE_WARP];
    __shared__ int s_blkid_smem_ful[HALFWARP_PER_BLOCK * TILE_PER_ADAPTIVE_WARP];

    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int halfwarp_lane_id = (HALFWARP_SIZE - 1) & threadIdx.x;

    const int local_halfwarp_id = threadIdx.x >> 4; //threadIdx.x / HALFWARP_SIZE;

    unsigned int *s_maskc_local = &s_maskc[local_halfwarp_id * TILE_SIZE_M * MaskNumC];
    TILE_MASK_TYPE_B *s_blkmaskB_local = &s_blkmaskB[local_halfwarp_id * TILE_SIZE_N * MaskNumB];
    int *s_blksmem_tny_cnt_local = &s_blksmem_tny_cnt[local_halfwarp_id];
    int *s_blksmem_sml_cnt_local = &s_blksmem_sml_cnt[local_halfwarp_id];
    int *s_blksmem_lrg_cnt_local = &s_blksmem_lrg_cnt[local_halfwarp_id];
    int *s_blksmem_dns_cnt_local = &s_blksmem_dns_cnt[local_halfwarp_id];
    int *s_blksmem_ful_cnt_local = &s_blksmem_ful_cnt[local_halfwarp_id];

    int *s_blkid_smem_tny_local = &s_blkid_smem_tny[local_halfwarp_id * TILE_PER_ADAPTIVE_WARP];
    int *s_blkid_smem_sml_local = &s_blkid_smem_sml[local_halfwarp_id * TILE_PER_ADAPTIVE_WARP];
    int *s_blkid_smem_lrg_local = &s_blkid_smem_lrg[local_halfwarp_id * TILE_PER_ADAPTIVE_WARP];
    int *s_blkid_smem_dns_local = &s_blkid_smem_dns[local_halfwarp_id * TILE_PER_ADAPTIVE_WARP];
    int *s_blkid_smem_ful_local = &s_blkid_smem_ful[local_halfwarp_id * TILE_PER_ADAPTIVE_WARP];

    int tile_start = global_halfwarp_id * TILE_PER_ADAPTIVE_WARP;
    int tile_end = tile_start + TILE_PER_ADAPTIVE_WARP; // (global_warp_id + 1) * TPW;

    if (!halfwarp_lane_id)
    {
        s_blksmem_tny_cnt_local[0] = 0;
        s_blksmem_sml_cnt_local[0] = 0;
        s_blksmem_lrg_cnt_local[0] = 0;
        s_blksmem_dns_cnt_local[0] = 0;
        s_blksmem_ful_cnt_local[0] = 0;
    }

    for (int tilei = tile_start; tilei < tile_end; tilei++)
    {
#pragma unroll
        for (int C_halfwarp_idx = halfwarp_lane_id; C_halfwarp_idx < TILE_SIZE_M; C_halfwarp_idx += HALFWARP_SIZE){
#pragma unroll
            for (int ri = 0; ri < MaskNumC; ri++){
                s_maskc_local[C_halfwarp_idx * MaskNumC + ri] = 0;
            }
        }

        // TODO: support up to 16 * 24 = 384 rows, can be changed larger
        int nnzcnt[25] = {};
        int matchedcnt = 0;

        if (tilei < numblkC)
        {
            const int blki = d_blkrowidxC[tilei];
            const int blkj = d_blkcolidxC[tilei];

            const int abase = d_blkrowptrA[blki];

            const int bbase = ld_gbl_auto(d_blkcolptrB + blkj);

            int offseta = 0;
            int offsetb = 0;

            // A and B are searched by inner-product
            for (int di = 0; di < blk_intersec_bitmask_len; di++)
            {
                // blk_intersec_bitmask_len = ceil((double)matrixA->tilen / 32.0); Search A row and B column
                unsigned int bma = d_blk_intersec_bitmask_A[blki * blk_intersec_bitmask_len + di];
                unsigned int bmb = d_blk_intersec_bitmask_B[blkj * blk_intersec_bitmask_len + di];

                int posa = offseta;
                int posb = offsetb;

                // if the tile of matrix A can match the tile of matrix B
                if (__popc(bma & bmb))
                {
                    for (int ii = 31; ii >= 0; ii--)
                    {
                        unsigned int bita = (bma >> ii) & 0x1;
                        unsigned int bitb = (bmb >> ii) & 0x1;

                        if (bita && bitb)
                        {
                            if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt < GMEM_SPECULATIVE_INTERSECTION)
                            {
                                if (!halfwarp_lane_id)
                                {
                                    d_spec_intersection_cnt[tilei] = matchedcnt;
                                    d_spec_intersection_posa[tilei * GMEM_SPECULATIVE_INTERSECTION + matchedcnt] = posa;
                                    d_spec_intersection_posb[tilei * GMEM_SPECULATIVE_INTERSECTION + matchedcnt] = posb;
                                }
                            }

                            matchedcnt++;

                            const int nnzastart = ld_gbl_auto(d_nnzb_A + abase + posa);
                            const int nnztotala = ld_gbl_auto(d_nnzb_A + abase + posa + 1) - nnzastart;

#pragma unroll
                            for (int row = halfwarp_lane_id; row < TILE_SIZE_N; row += HALFWARP_SIZE)
                            {
#pragma unroll
                                for (int maskid = 0; maskid < MaskNumB; maskid++)
                                {
                                    s_blkmaskB_local[row * MaskNumB + maskid] = 
                                        ld_gbl_auto(&d_blkmaskB[(bbase + posb) * TILE_SIZE_N * MaskNumB + row * MaskNumB + maskid]);
                                }
                            }

                            // TODO: Threshold Optimization
                            if (nnztotala >= NNZTOTALA_FAST_TRACK_TH2)
                            {
#pragma unroll
                                for (int A_halfwarp_idx = halfwarp_lane_id; A_halfwarp_idx < TILE_SIZE_M; A_halfwarp_idx += HALFWARP_SIZE){
                                    int astart = d_blkcsr_Ptr_A[(abase + posa) * TILE_SIZE_M + A_halfwarp_idx];
                                    int astop = A_halfwarp_idx == TILE_SIZE_M - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase + posa) * TILE_SIZE_M + A_halfwarp_idx + 1];

                                    for (int aci = astart; aci < astop; aci++)
                                    {
                                        TILE_CSR_COL_TYPE_A rowcolidx = d_blkcsr_Col_A[nnzastart + aci];
                                        // rowcolidx：(ri * TILE_SIZE_N) + colidx
                                        TILE_CSR_COL_TYPE_A row_in_B = rowcolidx % TILE_SIZE_N;
#pragma unroll
                                        for (int ri = 0; ri < MaskNumC; ri++)
                                        {
                                            // A_halfwarp_idx == row_in_C
                                            s_maskc_local[A_halfwarp_idx * MaskNumC + ri] |= s_blkmaskB_local[row_in_B * MaskNumB + ri];
                                        }
                                    }
                                }
                            }
                            else
                            {
                                for (int i = halfwarp_lane_id; i < nnztotala; i += HALFWARP_SIZE)
                                {
                                    TILE_CSR_COL_TYPE_A rowcolidx = ld_gbl_auto(d_blkcsr_Col_A + nnzastart + i);
                                    // rowcolidx：(ri * TILE_SIZE_N) + colidx
                                    TILE_CSR_COL_TYPE_A row_in_B = rowcolidx % TILE_SIZE_N;
                                    TILE_CSR_COL_TYPE_A row_in_C = rowcolidx / TILE_SIZE_N;
#pragma unroll
                                    for (int ri = 0; ri < MaskNumC; ri++)
                                    {
                                        atomicOr(&s_maskc_local[row_in_C * MaskNumC + ri], s_blkmaskB_local[row_in_B * MaskNumB + ri]);
                                    }
                                }
                            }
                        }

                        posa += bita;
                        posb += bitb;
                    }
                }

                offseta += __popc(bma);
                offsetb += __popc(bmb);
            }

#pragma unroll
            for (int C_halfwarp_idx = halfwarp_lane_id; C_halfwarp_idx < TILE_SIZE_M; C_halfwarp_idx += HALFWARP_SIZE){
#pragma unroll
                for (int ri = 0; ri < MaskNumC; ri++)
                {
                    nnzcnt[C_halfwarp_idx / HALFWARP_SIZE] += __popc(s_maskc_local[C_halfwarp_idx * MaskNumC + ri]);
                }
            }
        }


        int nnzcnt_sum_[25] = {};
        int nnzcnt_scan[25] = {};
#pragma unroll
        for (int temp = 0; temp < (TILE_SIZE_M + HALFWARP_SIZE - 1) / HALFWARP_SIZE; temp++){
            nnzcnt_sum_[temp] = sum_16_shfl_cg(nnzcnt[temp]);
            nnzcnt_scan[temp] = scan_32_shfl(nnzcnt[temp], lane_id);
            nnzcnt_scan[temp] -= nnzcnt[temp];
            nnzcnt_scan[temp] -= __shfl_sync(0xffffffff, nnzcnt_scan[temp], (lane_id >> 4) << 4);
        }

        int nnzcnt_sum = nnzcnt_sum_[0];

#pragma unroll
        for (int temp = 1; temp < (TILE_SIZE_M + HALFWARP_SIZE - 1) / HALFWARP_SIZE; temp++){
            nnzcnt_scan[temp] += nnzcnt_sum;
            nnzcnt_sum += nnzcnt_sum_[temp];
        }
        // int nnzcnt_sum = sum_16_shfl_cg(nnzcnt);
        
        // int nnzcnt_scan = scan_32_shfl(nnzcnt, lane_id);
        // nnzcnt_scan -= nnzcnt;
        // nnzcnt_scan -= __shfl_sync(0xffffffff, nnzcnt_scan, (lane_id >> 4) << 4);

        if (tilei < numblkC && nnzcnt_sum)
        {
#pragma unroll
            for (int C_halfwarp_idx = halfwarp_lane_id; C_halfwarp_idx < TILE_SIZE_M; C_halfwarp_idx += HALFWARP_SIZE){
                long long int pos_c  = (long long int)(tilei) * TILE_SIZE_M + C_halfwarp_idx;
                d_blkcsr_Ptr_C[pos_c] = nnzcnt_scan[C_halfwarp_idx / HALFWARP_SIZE]; // - nnzcnt;
#pragma unroll
                for (int maskid = 0; maskid < MaskNumC; maskid++)
                {
                    d_blkmaskC[pos_c * MaskNumC + maskid] = (TILE_MASK_TYPE_B)s_maskc_local[C_halfwarp_idx * MaskNumC + maskid];
                }

                if (!C_halfwarp_idx)
                {
                    d_nnzb_C[tilei] = nnzcnt_sum;

                    if (nnzcnt_sum <= SMEM_TNY_TH && nnzcnt_sum != 0)
                    {
                        int pos = atomicAdd(s_blksmem_tny_cnt_local, 1);
                        s_blkid_smem_tny_local[pos] = tilei;
                    }
                    else if (SMEM_TNY_TH < nnzcnt_sum && nnzcnt_sum <= SMEM_SML_TH)
                    {
                        int pos = atomicAdd(s_blksmem_sml_cnt_local, 1);
                        s_blkid_smem_sml_local[pos] = tilei;
                    }
                    else if (SMEM_SML_TH < nnzcnt_sum && nnzcnt_sum <= d_SMEM_LRG_TH)
                    {
                        int pos = atomicAdd(s_blksmem_lrg_cnt_local, 1);
                        s_blkid_smem_lrg_local[pos] = tilei;
                    }
                    else if (d_SMEM_LRG_TH < nnzcnt_sum && nnzcnt_sum < SMEM_DNS_TH)
                    {
                        int pos = atomicAdd(s_blksmem_dns_cnt_local, 1);
                        s_blkid_smem_dns_local[pos] = tilei;
                    }
                    else if (nnzcnt_sum >= SMEM_DNS_TH)
                    {
                        int pos = atomicAdd(s_blksmem_ful_cnt_local, 1);
                        s_blkid_smem_ful_local[pos] = tilei;
                    }
                }
            }
        }
    }

    int len = s_blksmem_tny_cnt_local[0];
    int pos = 0;
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_tny_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_tny[pos + halfwarp_lane_id] = s_blkid_smem_tny_local[halfwarp_lane_id];

    len = s_blksmem_sml_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_sml_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_sml[pos + halfwarp_lane_id] = s_blkid_smem_sml_local[halfwarp_lane_id];

    len = s_blksmem_lrg_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_lrg_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_lrg[pos + halfwarp_lane_id] = s_blkid_smem_lrg_local[halfwarp_lane_id];

    len = s_blksmem_dns_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_dns_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_dns[pos + halfwarp_lane_id] = s_blkid_smem_dns_local[halfwarp_lane_id];

    len = s_blksmem_ful_cnt_local[0];
    pos = halfwarp_lane_id == 0 ? atomicAdd(d_blksmem_ful_cnt, len) : 0;
    pos = __shfl_sync(0xffffffff, pos, (lane_id >> 4) << 4);
    if (halfwarp_lane_id < len)
        d_blkid_smem_ful[pos + halfwarp_lane_id] = s_blkid_smem_ful_local[halfwarp_lane_id];
}

/*
    THREADS_USED: e.g., TNY -> 16, SML -> 16, LRG -> 32;
    !!! THREADS_USED should <= 32, otherwise intersection_binarysearch_kernel() will output wrong results !!!
    TODO: Support THREADS_USED > 32
*/
template <int SMEM_MATNNZ, int THREADS_USED = TILE_SIZE_M, int BUFFER_SIZE = 16>
__global__ void tile_spgemm_step4_cuda_sparse_kernel_adaptive_warp(int *d_blkrowptrA,
                                                                        const int *__restrict__ d_blkcolidxA,
                                                                        int *d_nnzb_A,
                                                                        MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                                        TILE_CSR_COL_TYPE_A *d_blkcsr_Col_A,
                                                                        TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_A,
                                                                        int blkmA, int blknA, int numblkA, int nnzA,
                                                                        const int *__restrict__ d_blkcolptrB,
                                                                        const int *__restrict__ d_blkrowidxB,
                                                                        const int *__restrict__ d_nnzb_B,
                                                                        const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                                        const TILE_CSR_COL_TYPE_B *__restrict__ d_blkcsr_Col_B,
                                                                        const TILE_CSR_PTR_TYPE *__restrict__ d_blkcsr_Ptr_B,
                                                                        int blkmB, int blknB, int numblkB, int nnzB,
                                                                        int *d_blkrowidxC,
                                                                        int *d_blkcolidxC,
                                                                        TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_C,
                                                                        TILE_CSR_COL_TYPE_B *d_blkcsr_Col_C,
                                                                        MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                                        int *d_nnzb_C,
                                                                        TILE_MASK_TYPE_B *d_blkmaskC,
                                                                        int numblkC,
                                                                        int *d_blkid,
                                                                        int *d_spec_intersection_cnt,
                                                                        int *d_spec_intersection_posa,
                                                                        int *d_spec_intersection_posb)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_id = global_id / THREADS_USED;

    // A warp for a tile
    if (global_warp_id >= numblkC)
        return;

    // Every AdaptiveWarp for a tile in C
    int tilei = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[tilei];
    const int blknnzctotal = d_nnzb_C[tilei + 1] - nnzcstart;
    if (!blknnzctotal)
        return;

    // auto TileGroup = createTileGroup<THREADS_USED>();

    // __shared__ uint8_t Buffer[BUFFER_SIZE * 1024]; 

    const int total_threads = STEP4_THREADS;
    const int local_warp_id = threadIdx.x / THREADS_USED;

    // __shared__ TILE_CSR_COL_TYPE_A s_blkcsr_Col_A[BUFFER_SIZE * 1024];
    // TILE_CSR_COL_TYPE_A *s_blkcsr_Col_A_local = &s_blkcsr_Col_A[local_warp_id * BUFFER_SIZE * 1024 / THREADS_USED];

    // __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[total_threads / THREADS_USED * SMEM_MATNNZ];
    __shared__ TILE_CSR_COL_TYPE_B s_blkcsr_Idx_C[total_threads / THREADS_USED * SMEM_MATNNZ];
    __shared__ TILE_CSR_PTR_TYPE s_blkcsr_Ptr_C[total_threads / THREADS_USED * TILE_SIZE_M];
    // MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * SMEM_MATNNZ];
    TILE_CSR_COL_TYPE_B *s_blkcsr_Idx_C_local = &s_blkcsr_Idx_C[local_warp_id * SMEM_MATNNZ];
    TILE_CSR_PTR_TYPE *s_blkcsr_Ptr_C_local = &s_blkcsr_Ptr_C[local_warp_id * TILE_SIZE_M];

    __shared__ TILE_CSR_PTR_TYPE s_csrRowPtrB[total_threads / THREADS_USED * TILE_SIZE_N];
    TILE_CSR_PTR_TYPE *s_csrRowPtrB_local = &s_csrRowPtrB[local_warp_id * TILE_SIZE_N];

    __shared__ int s_matched_posa[total_threads / THREADS_USED * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[total_threads / THREADS_USED * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[total_threads / THREADS_USED];

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];

    // Adaptive warp per tile
    // const int ADAPTWARP_PER_TILE = (TILE_SIZE_M + THREADS_USED - 1) / THREADS_USED;
    TILE_MASK_TYPE_B maskc[MaskNumC] = {};
    TILE_CSR_PTR_TYPE blknnzcstart;

    const int halfwarp_lane_id = (HALFWARP_SIZE - 1) & threadIdx.x;
    const int warp_lane_id = (WARP_SIZE - 1) & threadIdx.x;
    const int c_tile_lane_id = (THREADS_USED - 1) & threadIdx.x;

    auto g = cg::tiled_partition<THREADS_USED>(cg::this_thread_block());

    // Initial the value of matrix C, not to change
    for (int i = c_tile_lane_id; i < blknnzctotal; i += THREADS_USED){
        d_blkcsr_Val_C[nnzcstart + i] = 0.0;
        // s_blkcsr_Val_C_local[i] = 0.0;
    }

    if (!c_tile_lane_id)
        s_matchedcnt_local[0] = 0;

#pragma unroll
    for (int c_adaptwarp_idx = c_tile_lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += THREADS_USED){
        long long int pos_c = (long long int)(tilei) * TILE_SIZE_M + c_adaptwarp_idx;
        s_blkcsr_Ptr_C_local[c_adaptwarp_idx] = d_blkcsr_Ptr_C[pos_c];
        blknnzcstart = s_blkcsr_Ptr_C_local[c_adaptwarp_idx];
#pragma unroll
        for (int maskid = 0; maskid < MaskNumC; maskid++){
            maskc[maskid] = d_blkmaskC[pos_c * MaskNumC + maskid];
        }

        int cnt = 0;
#pragma unroll
        for (int maskid = 0; maskid < MaskNumC; maskid++){
#pragma unroll
            for (int i = 0; i < MaskBitsC; i++)
            {
                int idx = ((maskc[maskid] >> MaskBitsC - i - 1) & 0x1) == 1 ? (maskid * MaskBitsC) + i : -1;
                if (idx != -1)
                {
                    s_blkcsr_Idx_C_local[blknnzcstart + cnt] = idx;
                    cnt++;
                }
            }
        }
    }

    const int blki = d_blkrowidxC[tilei];
    const int blkj = d_blkcolidxC[tilei];

    const int abase = ld_gbl_auto(d_blkrowptrA + blki);
    const int astop = ld_gbl_auto(d_blkrowptrA + blki + 1);
    int lena = astop - abase;

    const int bbase = ld_gbl_auto(d_blkcolptrB + blkj);
    const int bstop = ld_gbl_auto(d_blkcolptrB + blkj + 1);
    int lenb = bstop - bbase;

    int matchedcnt = 0;
    int specres = 0;

    if (USE_GMEM_SPECULATIVE_INTERSECTION)
        matchedcnt = d_spec_intersection_cnt[tilei];

    // We will not access this branch, plan to optimize it in the future
    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt > 0)
    {}
    else
    {

        // TODO: to support THREADS_USED > 32, should we change the pointer of s_matchedcnt_local?
        specres = intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                   d_blkrowidxB, bbase, bstop, lenb,
                                                   s_matched_posa_local, s_matched_posb_local,
                                                   SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                   c_tile_lane_id, THREADS_USED);

        g.sync();

        // #if (THREADS_USED > 32)
        //     __syncthreads();
        // #endif
        // if (THREADS_USED > 32)
        //     __syncthreads();
        // TileGroup.sync();

        matchedcnt = s_matchedcnt_local[0];
    }

    if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
    {
        for (int posi = 0; posi < matchedcnt; posi++)
        {
            int posa = s_matched_posa_local[posi];
            int posb = s_matched_posb_local[posi];

            // atomic method
            const int nnzastart = d_nnzb_A[(abase + posa)];
            int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;

#pragma unroll
            for (int B_adaptive_warp_idx = c_tile_lane_id; B_adaptive_warp_idx < TILE_SIZE_N; B_adaptive_warp_idx += THREADS_USED){
                s_csrRowPtrB_local[B_adaptive_warp_idx] = ld_gbl_auto(d_blkcsr_Ptr_B + (bbase + posb) * TILE_SIZE_N + B_adaptive_warp_idx);
            }
            const int nnzbstart = ld_gbl_auto(d_nnzb_B + bbase + posb);
            int nnztotalb = ld_gbl_auto(d_nnzb_B + bbase + posb + 1) - nnzbstart;

            if (nnztotala > TILE_SIZE_M / 2)
            {
                // for (int i = c_tile_lane_id; i < nnztotala; i += THREADS_USED)
                //     s_blkcsr_Col_A_local[i] = d_blkcsr_Col_A[nnzastart + i];
                for (int i = c_tile_lane_id; i < nnztotala; i += THREADS_USED)
                {
                    // we use magic_index to reduce atomicadd
                    // int magic_index = (nnztotala & (MAGIC_NUMBER - 1)) ? ((i * MAGIC_NUMBER) % nnztotala) : i;
                    // TILE_CSR_COL_TYPE_A rowcolidx = s_blkcsr_Col_A_local[i];
                    TILE_CSR_COL_TYPE_A rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                    TILE_CSR_COL_TYPE_A rowidxa = rowcolidx / TILE_SIZE_N;
                    TILE_CSR_COL_TYPE_A rowidxb = rowcolidx % TILE_SIZE_N;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                    int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                    int blkoffseta_stop = rowidxa == TILE_SIZE_M - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                    const int startb = s_csrRowPtrB_local[rowidxb];                                            //d_csrRowPtrB[rowidxb];
                    const int stopb = rowidxb == TILE_SIZE_N - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1]; //d_csrRowPtrB[rowidxb+1];

                    for (int k = startb; k < stopb; k++)
                    {
                        TILE_CSR_COL_TYPE_B colidx = ld_gbl_auto(d_blkcsr_Col_B + nnzbstart + k);
                        MAT_VAL_TYPE valb = ld_gbl_auto(d_blkcsr_Val_B + nnzbstart + k);
                        int cnt = binary_search_exact_auto_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                        if (cnt != -1){
                            atomicAdd(&d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt], val * valb);
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < nnztotala; i++)
                {
                    TILE_CSR_COL_TYPE_A rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                    TILE_CSR_COL_TYPE_A rowidxa = rowcolidx / TILE_SIZE_N;
                    TILE_CSR_COL_TYPE_A rowidxb = rowcolidx % TILE_SIZE_N;
                    MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                    int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                    int blkoffseta_stop = rowidxa == TILE_SIZE_M - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                    const int startb = s_csrRowPtrB_local[rowidxb];
                    const int stopb = rowidxb == TILE_SIZE_N - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1];

#pragma unroll
                    for (int c_adaptwarp_idx = c_tile_lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += THREADS_USED){
                        int k = startb + c_adaptwarp_idx;
                        if (k < stopb)
                        {
                            TILE_CSR_COL_TYPE_B colidx = ld_gbl_auto(d_blkcsr_Col_B + nnzbstart + k);
                            MAT_VAL_TYPE valb = ld_gbl_auto(d_blkcsr_Val_B + nnzbstart + k);
                            int cnt = binary_search_exact_auto_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                            if (cnt != -1){
                                // atomicAdd(&d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt], val * valb);
                                d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt] += val * valb;
                                // s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop - 1];
        const int bstart = ld_gbl_auto(d_blkrowidxB + bbase);
        const int bend = ld_gbl_auto(d_blkrowidxB + bstop - 1);

        int posa_real = 0;
        int posb_real = 0;
        if (bstart > astart)
        {
            int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
            posa_real = posa_real_new < 0 ? 0 : posa_real_new;
        }
        else if (bstart < astart)
        {
            int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
            posb_real = posb_real_new < 0 ? 0 : posb_real_new;
        }

        if (bstop < astop)
        {
            int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
            lena = lena_new > lena ? lena : lena_new;
        }
        else if (bstop > astop)
        {
            int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
            lenb = lenb_new > lenb ? lenb : lenb_new;
        }

        int posa = posa_real;
        int posb = posb_real;
        int idxa = 0;
        int idxb = 0;
        int posa_updated = 1;
        int posb_updated = 1;

        while (posa < lena && posb < lenb)
        {
            idxa = posa_updated ? ld_gbl_auto(d_blkcolidxA + abase + posa) : idxa; //a[posa] : idxa;
            idxb = posb_updated ? ld_gbl_auto(d_blkrowidxB + bbase + posb) : idxb; //b[posb] : idxb;

            if (idxa == idxb)
            {
                const int nnzastart = d_nnzb_A[(abase + posa)];
                int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
                // if (lane_id < BLOCK_SIZE)
#pragma unroll
                for (int B_adaptive_warp_idx = c_tile_lane_id; B_adaptive_warp_idx < TILE_SIZE_N; B_adaptive_warp_idx += THREADS_USED){
                    s_csrRowPtrB_local[B_adaptive_warp_idx] = ld_gbl_auto(d_blkcsr_Ptr_B + (bbase + posb) * TILE_SIZE_N + B_adaptive_warp_idx);
                }
                const int nnzbstart = ld_gbl_auto(d_nnzb_B + bbase + posb);
                int nnztotalb = ld_gbl_auto(d_nnzb_B + bbase + posb + 1) - nnzbstart;

                if (nnztotala > TILE_SIZE_M / 2)
                {
                    for (int i = c_tile_lane_id; i < nnztotala; i += THREADS_USED)
                    {
                        // int magic_index = (nnztotala & (MAGIC_NUMBER - 1)) ? ((i * MAGIC_NUMBER) % nnztotala) : i;
                        TILE_CSR_COL_TYPE_A rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                        TILE_CSR_COL_TYPE_A rowidxa = rowcolidx / TILE_SIZE_N;
                        TILE_CSR_COL_TYPE_A rowidxb = rowcolidx % TILE_SIZE_N;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                        int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                        int blkoffseta_stop = rowidxa == TILE_SIZE_M - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                        const int startb = s_csrRowPtrB_local[rowidxb];                                            //d_csrRowPtrB[rowidxb];
                        const int stopb = rowidxb == TILE_SIZE_N - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1]; //d_csrRowPtrB[rowidxb+1];
                        for (int k = startb; k < stopb; k++)
                        {
                            TILE_CSR_COL_TYPE_B colidx = ld_gbl_auto(d_blkcsr_Col_B + nnzbstart + k);
                            MAT_VAL_TYPE valb = ld_gbl_auto(d_blkcsr_Val_B + nnzbstart + k);
                            int cnt = binary_search_exact_auto_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                            if (cnt != -1){
                                atomicAdd(&d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt], val * valb);
                                // atomicAdd(&s_blkcsr_Val_C_local[blkoffseta + cnt], val * valb);
                            }
                        }
                    }
                }
                else
                {
                    for (int i = 0; i < nnztotala; i++)
                    {
                        TILE_CSR_COL_TYPE_A rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                        TILE_CSR_COL_TYPE_A rowidxa = rowcolidx / TILE_SIZE_N;
                        TILE_CSR_COL_TYPE_A rowidxb = rowcolidx % TILE_SIZE_N;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];
                        int blkoffseta = s_blkcsr_Ptr_C_local[rowidxa];
                        int blkoffseta_stop = rowidxa == TILE_SIZE_M - 1 ? blknnzctotal : s_blkcsr_Ptr_C_local[rowidxa + 1];

                        const int startb = s_csrRowPtrB_local[rowidxb];
                        const int stopb = rowidxb == TILE_SIZE_N - 1 ? nnztotalb : s_csrRowPtrB_local[rowidxb + 1];

#pragma unroll
                        for (int c_adaptwarp_idx = c_tile_lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += THREADS_USED){
                            int k = startb + c_adaptwarp_idx;
                            if (k < stopb)
                            {
                                TILE_CSR_COL_TYPE_B colidx = ld_gbl_auto(d_blkcsr_Col_B + nnzbstart + k);
                                MAT_VAL_TYPE valb = ld_gbl_auto(d_blkcsr_Val_B + nnzbstart + k);
                                int cnt = binary_search_exact_auto_kernel(s_blkcsr_Idx_C_local + blkoffseta, 0, blkoffseta_stop - blkoffseta - 1, colidx);
                                if (cnt != -1){
                                    d_blkcsr_Val_C[nnzcstart + blkoffseta + cnt] += val * valb;
                                    // s_blkcsr_Val_C_local[blkoffseta + cnt] += val * valb;
                                }
                            }
                        }
                    }
                }

                // do spgemm of this pair
                posa++;
                posa_updated = 1;
                posb++;
                posb_updated = 1;
            }
            else
            {
                // the smaller index goes forward
                posa_updated = idxa < idxb ? 1 : 0;
                posa += posa_updated;
                posb_updated = idxa > idxb ? 1 : 0;
                posb += posb_updated;
            }
        }
    }

    for (int i = c_tile_lane_id; i < blknnzctotal; i += THREADS_USED)
    {
        d_blkcsr_Col_C[nnzcstart + i] = s_blkcsr_Idx_C_local[i];
        // d_blkcsr_Val_C[nnzcstart + i] = s_blkcsr_Val_C_local[i];
    }
}

template <int THREADS_USED = 32>
__global__ void tile_spgemm_step4_cuda_dns_kernel_adaptive_warp(int *d_blkrowptrA,
                                                                    const int *__restrict__ d_blkcolidxA,
                                                                    int *d_nnzb_A,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                                    TILE_CSR_COL_TYPE_A *d_blkcsr_Col_A,
                                                                    TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_A,
                                                                    int blkmA, int blknA, int numblkA, int nnzA,
                                                                    const int *__restrict__ d_blkcolptrB,
                                                                    const int *__restrict__ d_blkrowidxB,
                                                                    const int *__restrict__ d_nnzb_B,
                                                                    const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                                    const TILE_CSR_COL_TYPE_B *__restrict__ d_blkcsr_Col_B,
                                                                    const TILE_CSR_PTR_TYPE *__restrict__ d_blkcsr_Ptr_B,
                                                                    int blkmB, int blknB, int numblkB, int nnzB,
                                                                    int *d_blkrowidxC,
                                                                    int *d_blkcolidxC,
                                                                    TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_C,
                                                                    TILE_CSR_COL_TYPE_B *d_blkcsr_Col_C,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                                    int *d_nnzb_C,
                                                                    TILE_MASK_TYPE_B *d_blkmaskC,
                                                                    int numblkC,
                                                                    int *d_blkid,
                                                                    int *d_spec_intersection_cnt,
                                                                    int *d_spec_intersection_posa,
                                                                    int *d_spec_intersection_posb)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_id = global_id / THREADS_USED;

    if (global_warp_id >= numblkC)
        return;
    int tilei = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[tilei];
    const int blknnzctotal = d_nnzb_C[tilei + 1] - nnzcstart;
    if (!blknnzctotal)
        return;

    const int total_threads = STEP4_TC_THREADS;
    const int local_warp_id = threadIdx.x / THREADS_USED; //threadIdx.x / HALFWARP_SIZE;
    
    const int shared_val_c_size = total_threads / THREADS_USED * TILE_SIZE_M * TILE_SIZE_M * sizeof(MAT_VAL_TYPE);

    #if shared_val_c_size >= 32768
        MAT_VAL_TYPE s_blkcsr_Val_C[total_threads / THREADS_USED * TILE_SIZE_M * TILE_SIZE_M] = {};
    #else
        __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[total_threads / THREADS_USED * TILE_SIZE_M * TILE_SIZE_M];
    #endif

    MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * TILE_SIZE_M * TILE_SIZE_M];

    __shared__ int s_matched_posa[total_threads / THREADS_USED * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[total_threads / THREADS_USED * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[total_threads / THREADS_USED];

    const int c_tile_lane_id = (TILE_SIZE_M - 1) & threadIdx.x;
    const int warp_lane_id = (WARP_SIZE - 1) & threadIdx.x;

    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];

#pragma unroll
    for (int c_adaptwarp_idx = c_tile_lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += THREADS_USED){
#pragma unroll
        for (int i = 0; i < TILE_SIZE_M; i++){
            s_blkcsr_Val_C_local[i * TILE_SIZE_M + c_adaptwarp_idx] = 0.0;
        }
    }

    auto g = cg::tiled_partition<THREADS_USED>(cg::this_thread_block());

    if (!c_tile_lane_id)
        s_matchedcnt_local[0] = 0;

    const int blki = d_blkrowidxC[tilei];
    const int blkj = d_blkcolidxC[tilei];

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki + 1];
    int lena = astop - abase;

    const int bbase = ld_gbl_auto(d_blkcolptrB + blkj);
    const int bstop = ld_gbl_auto(d_blkcolptrB + blkj + 1);
    int lenb = bstop - bbase;

    int matchedcnt = 0;
    int specres = 0;

    if (USE_GMEM_SPECULATIVE_INTERSECTION)
        matchedcnt = d_spec_intersection_cnt[tilei];

    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt > 0)
    {}
    else
    {
        specres = intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                   d_blkrowidxB, bbase, bstop, lenb,
                                                   s_matched_posa_local, s_matched_posb_local,
                                                   SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                   c_tile_lane_id, THREADS_USED);

        g.sync();
        matchedcnt = s_matchedcnt_local[0];
    }

    if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
    {
        for (int i = 0; i < matchedcnt; i++)
        {
            int posa = s_matched_posa_local[i];
            int posb = s_matched_posb_local[i];

            const int nnzastart = d_nnzb_A[(abase + posa)];
            int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
            const TILE_CSR_PTR_TYPE *__restrict__ d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase + posb) * TILE_SIZE_N];
            const int nnzbstart = ld_gbl_auto(d_nnzb_B + bbase + posb);
            int nnztotalb = ld_gbl_auto(d_nnzb_B + bbase + posb + 1) - nnzbstart;

            for (int nnzi = 0; nnzi < nnztotala; nnzi++)
            {
                TILE_CSR_COL_TYPE_A rowcolidx = d_blkcsr_Col_A[nnzastart + nnzi];
                TILE_CSR_COL_TYPE_A rowidxa = rowcolidx / TILE_SIZE_N;
                TILE_CSR_COL_TYPE_A rowidxb = rowcolidx % TILE_SIZE_N;
                MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + nnzi];

                const int startb = ld_gbl_auto(d_csrRowPtrB + rowidxb);
                const int stopb = rowidxb == TILE_SIZE_N - 1 ? nnztotalb : ld_gbl_auto(d_csrRowPtrB + rowidxb + 1);
                
#pragma unroll
                for (int c_adaptwarp_idx = c_tile_lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += THREADS_USED){
                    int k = startb + c_adaptwarp_idx;
                    if (k < stopb)
                    {
                        TILE_CSR_COL_TYPE_B colidx = ld_gbl_auto(d_blkcsr_Col_B + nnzbstart + k);
                        MAT_VAL_TYPE valb = ld_gbl_auto(d_blkcsr_Val_B + nnzbstart + k);
                        s_blkcsr_Val_C_local[rowidxa * TILE_SIZE_M + colidx] += val * valb;
                    }
                }
            }
        }
    }
    else
    {
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop - 1];
        const int bstart = ld_gbl_auto(d_blkrowidxB + bbase);
        const int bend = ld_gbl_auto(d_blkrowidxB + bstop - 1);

        int posa_real = 0;
        int posb_real = 0;
        if (bstart > astart)
        {
            int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
            posa_real = posa_real_new < 0 ? 0 : posa_real_new;
        }
        else if (bstart < astart)
        {
            int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
            posb_real = posb_real_new < 0 ? 0 : posb_real_new;
        }

        if (bstop < astop)
        {
            int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
            lena = lena_new > lena ? lena : lena_new;
        }
        else if (bstop > astop)
        {
            int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
            lenb = lenb_new > lenb ? lenb : lenb_new;
        }

        int posa = posa_real;
        int posb = posb_real;
        int idxa = 0;
        int idxb = 0;
        int posa_updated = 1;
        int posb_updated = 1;

        while (posa < lena && posb < lenb)
        {
            idxa = posa_updated ? ld_gbl_auto(d_blkcolidxA + abase + posa) : idxa; //a[posa] : idxa;
            idxb = posb_updated ? ld_gbl_auto(d_blkrowidxB + bbase + posb) : idxb; //b[posb] : idxb;

            if (idxa == idxb)
            {
                const int nnzastart = d_nnzb_A[(abase + posa)];
                int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
                const TILE_CSR_PTR_TYPE *__restrict__ d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase + posb) * TILE_SIZE_N];
                const int nnzbstart = ld_gbl_auto(d_nnzb_B + bbase + posb);
                int nnztotalb = ld_gbl_auto(d_nnzb_B + bbase + posb + 1) - nnzbstart;
#pragma unroll
                for (int c_adaptwarp_idx = c_tile_lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += THREADS_USED){   
                    TILE_CSR_PTR_TYPE offseta_start = d_blkcsr_Ptr_A[(abase + posa) * TILE_SIZE_M + c_adaptwarp_idx];
                    TILE_CSR_PTR_TYPE offseta_end = c_adaptwarp_idx == TILE_SIZE_M - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase + posa) * TILE_SIZE_M + c_adaptwarp_idx + 1];

                    for (int i = offseta_start; i < offseta_end; i++)
                    {
                        TILE_CSR_COL_TYPE_A rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                        int rowidxa = rowcolidx / TILE_SIZE_N;
                        int rowidxb = rowcolidx % TILE_SIZE_N;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];

                        const int startb = ld_gbl_auto(d_csrRowPtrB + rowidxb);
                        const int stopb = rowidxb == TILE_SIZE_N - 1 ? nnztotalb : ld_gbl_auto(d_csrRowPtrB + rowidxb + 1);
                        for (int k = startb; k < stopb; k++)
                        {
                            TILE_CSR_COL_TYPE_B colidx = ld_gbl_auto(d_blkcsr_Col_B + nnzbstart + k);
                            MAT_VAL_TYPE valb = ld_gbl_auto(d_blkcsr_Val_B + nnzbstart + k);
                            s_blkcsr_Val_C_local[rowidxa * TILE_SIZE_M + colidx] += val * valb;
                        }
                    }
                }

                posa++;
                posa_updated = 1;
                posb++;
                posb_updated = 1;
            }
            else
            {
                // the smaller index goes forward
                posa_updated = idxa < idxb ? 1 : 0;
                posa += posa_updated;
                posb_updated = idxa > idxb ? 1 : 0;
                posb += posb_updated;
            }
        }
    }

    // Check
    if (blknnzctotal == TILE_SIZE_M * TILE_SIZE_M){
#pragma unroll
        for (int c_adaptwarp_idx = c_tile_lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += THREADS_USED){
#pragma unroll
            for (int i = 0; i < TILE_SIZE_M; i++)
            {
                int offset_local = i * TILE_SIZE_M + c_adaptwarp_idx;
                d_blkcsr_Col_C[nnzcstart + offset_local] = c_adaptwarp_idx;
                d_blkcsr_Val_C[nnzcstart + offset_local] = s_blkcsr_Val_C_local[offset_local];
            }
        }
    }
    else{
        const int ADAPTWARP_PER_TILE = (TILE_SIZE_M + THREADS_USED - 1) / THREADS_USED;
        TILE_MASK_TYPE_B maskc[MaskNumC] = {};
        TILE_CSR_PTR_TYPE blknnzcstart;

#pragma unroll
        for (int c_adaptwarp_idx = c_tile_lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += THREADS_USED){
            long long int pos_c = (long long int)(tilei) * TILE_SIZE_M + c_adaptwarp_idx;
            blknnzcstart = d_blkcsr_Ptr_C[pos_c];
#pragma unroll
            for (int maskid = 0; maskid < MaskNumC; maskid++){
                maskc[maskid] = d_blkmaskC[pos_c * MaskNumC + maskid];
            }

            int cnt = 0;
#pragma unroll
            for (int maskid = 0; maskid < MaskNumC; maskid++){
#pragma unroll
                for (int i = 0; i < MaskBitsC; i++)
                {
                    int idx = ((maskc[maskid] >> MaskBitsC - i - 1) & 0x1) == 1 ? (maskid * MaskBitsC) + i : -1;
                    if (idx != -1)
                    {
                        d_blkcsr_Col_C[nnzcstart + blknnzcstart + cnt] = idx;
                        d_blkcsr_Val_C[nnzcstart + blknnzcstart + cnt] = s_blkcsr_Val_C_local[c_adaptwarp_idx * TILE_SIZE_M + idx];
                        cnt++;
                    }
                }
            }
        }
    }
}

template <typename MaskType, typename RowPtrT, typename ValueT>
__device__ __forceinline__ void bitmask_tile_to_dense(
    const RowPtrT  *__restrict__ rowPtr,
    const MaskType *__restrict__ bitmasks,
    const ValueT   *__restrict__ values,
    const int rows,
    const int cols,
    ValueT         *__restrict__ dense)
{
    static_assert(sizeof(MaskType) <= 4, "MaskType must be 8, 16, or 32 bits");
    
    constexpr int BITS_PER_MASK = sizeof(MaskType) * 8;
    constexpr int ROWS_PER_STEP = 32 / BITS_PER_MASK;

    const int lane_id_ = threadIdx.x & (WARP_SIZE - 1);
    const int leading_dim = cols;
    const int local_row = lane_id_ / BITS_PER_MASK;
    const int bit_idx   = lane_id_ % BITS_PER_MASK;


#pragma unroll
    for (int idx = lane_id_; idx < rows * cols; idx += WARP_SIZE) {
        dense[idx] = ValueT(0);
    }
    __syncwarp();

#pragma unroll
    for (int r = 0; r < rows; r += ROWS_PER_STEP) {
        
        int target_row = r + local_row;

        if (target_row < rows && bit_idx < cols) {
            
            ValueT out_val = ValueT(0);
            MaskType mask = bitmasks[target_row];
            
            MaskType current_bit = static_cast<MaskType>(1) << bit_idx;
            
            if (mask & current_bit) {
                MaskType lower_mask = current_bit - 1;
                int local_offset = __popc(static_cast<unsigned int>(mask & lower_mask));
                int val_offset   = static_cast<int>(rowPtr[target_row]);
                
                out_val = values[val_offset + local_offset];
            }

            dense[target_row * leading_dim + bit_idx] = out_val;
        }
    }
    __syncwarp();
}

// Slot lock states
#define SLOT_FREE 0
#define SLOT_ACQUIRED 1

// ==================== Multi-Warp Shared Slot Helper Functions ====================

// Acquire a slot from the shared pool
// Returns slot index on success, -1 if no slot available
__device__ __forceinline__ int acquire_shared_slot(int *slot_locks, int num_slots) {
    for (int i = 0; i < num_slots; i++) {
        // Try to acquire lock atomically
        if (atomicCAS(&slot_locks[i], SLOT_FREE, SLOT_ACQUIRED) == SLOT_FREE) {
            return i;
        }
    }
    return -1;  // No slot available
}

// Release a slot back to the pool
__device__ __forceinline__ void release_shared_slot(int *slot_locks, int slot_id, int warp_id) {
    if (slot_id >= 0) {
        atomicExch(&slot_locks[slot_id], SLOT_FREE);
    }
}

// Spin-wait acquire with timeout
// Returns slot index on success, -1 on timeout
__device__ __forceinline__ int acquire_shared_slot_spin(int *slot_locks, int num_slots, int spin_limit, int lane_id, int warp_id) {
    // int spin_count = 0;
    int slot_id = -1;
    while (true) {
        slot_id = acquire_shared_slot(slot_locks, num_slots);
        if (slot_id >= 0) {
            return slot_id;
        }
    }
    return -1;  // Timeout
}

__global__ void tile_spgemm_step4_cuda_dns_kernel_shared_slot(int *d_blkrowptrA,
                                                                    const int *__restrict__ d_blkcolidxA,
                                                                    int *d_nnzb_A,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                                    TILE_CSR_COL_TYPE_A *d_blkcsr_Col_A,
                                                                    TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_A,
                                                                    TILE_MASK_TYPE_A *d_blkmaskA,
                                                                    MAT_VAL_TYPE *d_dense_val_A,
                                                                    int *d_tile_dense_ready_A,
                                                                    int blkmA, int blknA, int numblkA, int nnzA,
                                                                    const int *__restrict__ d_blkcolptrB,
                                                                    const int *__restrict__ d_blkrowidxB,
                                                                    const int *__restrict__ d_nnzb_B,
                                                                    const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                                    const TILE_CSR_COL_TYPE_B *__restrict__ d_blkcsr_Col_B,
                                                                    const TILE_CSR_PTR_TYPE *__restrict__ d_blkcsr_Ptr_B,
                                                                    const TILE_MASK_TYPE_B *__restrict__ d_blkmaskB,
                                                                    MAT_VAL_TYPE *d_dense_val_B,
                                                                    int *d_tile_dense_ready_B,
                                                                    int blkmB, int blknB, int numblkB, int nnzB,
                                                                    int *d_blkrowidxC,
                                                                    int *d_blkcolidxC,
                                                                    TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_C,
                                                                    TILE_CSR_COL_TYPE_B *d_blkcsr_Col_C,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                                    int *d_nnzb_C,
                                                                    TILE_MASK_TYPE_B *d_blkmaskC,
                                                                    int numblkC,
                                                                    int *d_blkid,
                                                                    int *d_spec_intersection_cnt,
                                                                    int *d_spec_intersection_posa,
                                                                    int *d_spec_intersection_posb,
                                                                    int *d_has_dense_calculated)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_id = global_id / WARP_SIZE;

    if (global_warp_id >= numblkC)
        return;
    int tilei = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[tilei];
    const int blknnzctotal = d_nnzb_C[tilei + 1] - nnzcstart;
    if (!blknnzctotal)
        return;

    const int total_threads = STEP4_TC_THREADS;
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps_per_block = total_threads / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    // ========== SHARED MEMORY ALLOCATION ==========
    __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[num_warps_per_block * TILE_SIZE_M * TILE_SIZE_M];
    __shared__ MAT_VAL_TYPE s_shared_slots_A[NUM_SHARED_SLOTS * TILE_SIZE_M * TILE_SIZE_N];
    __shared__ MAT_VAL_TYPE s_shared_slots_B[NUM_SHARED_SLOTS * TILE_SIZE_N * TILE_SIZE_M];
    __shared__ TILE_MASK_TYPE_A s_blkmaskA[num_warps_per_block * TILE_SIZE_M * MaskNumA];
    __shared__ TILE_MASK_TYPE_B s_blkmaskB[num_warps_per_block * TILE_SIZE_N * MaskNumB];
    __shared__ int s_slot_locks[NUM_SHARED_SLOTS];
    __shared__ int s_slot_tile_id_A[NUM_SHARED_SLOTS];
    __shared__ int s_slot_tile_id_B[NUM_SHARED_SLOTS];

    // Intersection results storage
    __shared__ int s_matched_posa[num_warps_per_block * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[num_warps_per_block * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[num_warps_per_block];
    __shared__ int acquired_slot[num_warps_per_block];

    // ========== INITIALIZATION ==========
    // Initialize slot locks and tile IDs (only first warp does this)
    if (threadIdx.x < NUM_SHARED_SLOTS) {
        s_slot_locks[threadIdx.x] = SLOT_FREE;
        s_slot_tile_id_A[threadIdx.x] = -1;
        s_slot_tile_id_B[threadIdx.x] = -1;
    }
    __syncthreads();

    MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * TILE_SIZE_M * TILE_SIZE_M];
    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];
    int *s_acquired_slot = &acquired_slot[local_warp_id];

    if (!lane_id){
#pragma unroll
        for (int i = 0; i < num_warps_per_block; i++){
            s_acquired_slot[i] = -1;
        }
    }

    __syncwarp();

    // Initialize C accumulator
#pragma unroll
    for (int c_adaptwarp_idx = lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += WARP_SIZE){
#pragma unroll
        for (int i = 0; i < TILE_SIZE_M; i++){
            s_blkcsr_Val_C_local[i * TILE_SIZE_M + c_adaptwarp_idx] = 0.0;
        }
    }

    if (!lane_id)
        s_matchedcnt_local[0] = 0;

    const int blki = d_blkrowidxC[tilei];
    const int blkj = d_blkcolidxC[tilei];

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki + 1];
    int lena = astop - abase;

    const int bbase = ld_gbl_auto(d_blkcolptrB + blkj);
    const int bstop = ld_gbl_auto(d_blkcolptrB + blkj + 1);
    int lenb = bstop - bbase;

    int matchedcnt = 0;
    int specres = 0;

    if (USE_GMEM_SPECULATIVE_INTERSECTION)
        matchedcnt = d_spec_intersection_cnt[tilei];

    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt > 0)
    {}
    else
    {
        specres = intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                   d_blkrowidxB, bbase, bstop, lenb,
                                                   s_matched_posa_local, s_matched_posb_local,
                                                   SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                   lane_id, WARP_SIZE);
        __syncwarp();
        matchedcnt = s_matchedcnt_local[0];
    }

    __syncwarp();

    if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
    {
        const int NUM_SUBTILES = TILE_SIZE_M / 8;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MAT_VAL_TYPE, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MAT_VAL_TYPE, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MAT_VAL_TYPE> c_frag[NUM_SUBTILES + 1][NUM_SUBTILES + 1];

#pragma unroll
        for(int i = 0; i < NUM_SUBTILES; i++)
#pragma unroll
            for(int j = 0; j < NUM_SUBTILES; j++){
                wmma::fill_fragment(c_frag[i][j], 0.0);
            }

        // ========== MAIN COMPUTATION LOOP WITH SHARED SLOTS ==========
        for (int k = 0; k < matchedcnt; k++)
        {
            int posa = s_matched_posa_local[k];
            int posb = s_matched_posb_local[k];

            bool a_dense_ready = d_tile_dense_ready_A[abase + posa] >= 0;
            bool b_dense_ready = d_tile_dense_ready_B[bbase + posb] >= 0;
            
            MAT_VAL_TYPE *tile_A_ptr;
            MAT_VAL_TYPE *tile_B_ptr;

            if (a_dense_ready && b_dense_ready) {
                // Both tiles have dense data ready in global memory
                int dense_idx_A = d_tile_dense_ready_A[abase + posa];
                int dense_idx_B = d_tile_dense_ready_B[bbase + posb];
                tile_A_ptr = d_dense_val_A + dense_idx_A;
                tile_B_ptr = d_dense_val_B + dense_idx_B;

#pragma unroll
                for (int k_step = 0; k_step < TILE_SIZE_N; k_step += WMMA_K){
#pragma unroll
                    for (int i = 0; i < NUM_SUBTILES; i++) {
#pragma unroll
                        for (int j = 0; j < NUM_SUBTILES; j++) {
                            const MAT_VAL_TYPE *ptr_a = tile_A_ptr + (i * WMMA_M * TILE_SIZE_N) + k_step;
                            const MAT_VAL_TYPE *ptr_b = tile_B_ptr + (k_step * TILE_SIZE_M) + (j * WMMA_N);
                            wmma::load_matrix_sync(a_frag, ptr_a, TILE_SIZE_N);
                            wmma::load_matrix_sync(b_frag, ptr_b, TILE_SIZE_M);
                            wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                        }
                    }
                }
            } 
            else {
                if(lane_id == 0){
                    acquired_slot[local_warp_id] = acquire_shared_slot_spin(s_slot_locks, NUM_SHARED_SLOTS, SLOT_SPIN_LIMIT, lane_id, local_warp_id);
                }

                __syncwarp();

                // ========== CONVERT TILES TO DENSE ==========
                tile_A_ptr = &s_shared_slots_A[acquired_slot[local_warp_id] * TILE_SIZE_M * TILE_SIZE_N];
                tile_B_ptr = &s_shared_slots_B[acquired_slot[local_warp_id] * TILE_SIZE_N * TILE_SIZE_M];

                if (!a_dense_ready) {
                    const int nnzastart = d_nnzb_A[(abase + posa)];
                    int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
                    bitmask_tile_to_dense(
                        &d_blkcsr_Ptr_A[(abase + posa) * TILE_SIZE_M],
                        &d_blkmaskA[(abase + posa) * TILE_SIZE_M * MaskNumA],
                        &d_blkcsr_Val_A[nnzastart],
                        TILE_SIZE_M,
                        TILE_SIZE_N,
                        tile_A_ptr
                    );
                } else {
                    int dense_idx_A = d_tile_dense_ready_A[abase + posa];
#pragma unroll
                    for (int i = lane_id; i < TILE_SIZE_M * TILE_SIZE_N; i += WARP_SIZE) {
                        tile_A_ptr[i] = d_dense_val_A[dense_idx_A + i];
                    }
                }

                if (!b_dense_ready) {
                    const int nnzbstart = ld_gbl_auto(d_nnzb_B + bbase + posb);
                    int nnztotalb = ld_gbl_auto(d_nnzb_B + bbase + posb + 1) - nnzbstart;
                    bitmask_tile_to_dense(
                        &d_blkcsr_Ptr_B[(bbase + posb) * TILE_SIZE_N],
                        &d_blkmaskB[(bbase + posb) * TILE_SIZE_N * MaskNumB],
                        &d_blkcsr_Val_B[nnzbstart],
                        TILE_SIZE_N,
                        TILE_SIZE_M,
                        tile_B_ptr
                    );
                } else {
                    int dense_idx_B = d_tile_dense_ready_B[bbase + posb];
#pragma unroll
                    for (int i = lane_id; i < TILE_SIZE_N * TILE_SIZE_M; i += WARP_SIZE) {
                        tile_B_ptr[i] = d_dense_val_B[dense_idx_B + i];
                    }
                }

                __syncwarp();  // Ensure all threads in warp see the data

#pragma unroll
                for (int k_step = 0; k_step < TILE_SIZE_N; k_step += WMMA_K){
#pragma unroll
                    for (int i = 0; i < NUM_SUBTILES; i++) {
#pragma unroll
                        for (int j = 0; j < NUM_SUBTILES; j++) {
                            const MAT_VAL_TYPE *ptr_a = tile_A_ptr + (i * WMMA_M * TILE_SIZE_N) + k_step;
                            const MAT_VAL_TYPE *ptr_b = tile_B_ptr + (k_step * TILE_SIZE_M) + (j * WMMA_N);

                            wmma::load_matrix_sync(a_frag, ptr_a, TILE_SIZE_N);
                            wmma::load_matrix_sync(b_frag, ptr_b, TILE_SIZE_M);
                            wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                        }
                    }
                }

                // ========== RELEASE SLOT ==========
                if (acquired_slot[local_warp_id] >= 0 && lane_id == 0) {
                    release_shared_slot(s_slot_locks, acquired_slot[local_warp_id], local_warp_id);
                }
            }
        }

#pragma unroll
        for(int i=0; i<NUM_SUBTILES; i++) {
#pragma unroll
            for(int j=0; j<NUM_SUBTILES; j++) {
                MAT_VAL_TYPE *dst_ptr = s_blkcsr_Val_C_local + (i * sizeof(MAT_VAL_TYPE) * TILE_SIZE_M) + (j * sizeof(MAT_VAL_TYPE));
                wmma::store_matrix_sync(dst_ptr, c_frag[i][j], TILE_SIZE_M, wmma::mem_row_major);
            }
        }
    }
    else
    {
        // Fallback path for large intersection counts
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop - 1];
        const int bstart = ld_gbl_auto(d_blkrowidxB + bbase);
        const int bend = ld_gbl_auto(d_blkrowidxB + bstop - 1);

        int posa_real = 0, posb_real = 0;
        if (bstart > astart) {
            int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
            posa_real = posa_real_new < 0 ? 0 : posa_real_new;
        } else if (bstart < astart) {
            int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
            posb_real = posb_real_new < 0 ? 0 : posb_real_new;
        }

        if (bstop < astop) {
            int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
            lena = lena_new > lena ? lena : lena_new;
        } else if (bstop > astop) {
            int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
            lenb = lenb_new > lenb ? lenb : lenb_new;
        }

        int posa = posa_real, posb = posb_real;
        int idxa = 0, idxb = 0;
        int posa_updated = 1, posb_updated = 1;

        while (posa < lena && posb < lenb)
        {
            idxa = posa_updated ? ld_gbl_auto(d_blkcolidxA + abase + posa) : idxa;
            idxb = posb_updated ? ld_gbl_auto(d_blkrowidxB + bbase + posb) : idxb;

            if (idxa == idxb)
            {
                const int nnzastart = d_nnzb_A[(abase + posa)];
                int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
                const TILE_CSR_PTR_TYPE *__restrict__ d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase + posb) * TILE_SIZE_N];
                const int nnzbstart = ld_gbl_auto(d_nnzb_B + bbase + posb);
                int nnztotalb = ld_gbl_auto(d_nnzb_B + bbase + posb + 1) - nnzbstart;
#pragma unroll
                for (int c_adaptwarp_idx = lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += WARP_SIZE){   
                    TILE_CSR_PTR_TYPE offseta_start = d_blkcsr_Ptr_A[(abase + posa) * TILE_SIZE_M + c_adaptwarp_idx];
                    TILE_CSR_PTR_TYPE offseta_end = c_adaptwarp_idx == TILE_SIZE_M - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase + posa) * TILE_SIZE_M + c_adaptwarp_idx + 1];

                    for (int i = offseta_start; i < offseta_end; i++)
                    {
                        TILE_CSR_COL_TYPE_A rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                        int rowidxa = rowcolidx / TILE_SIZE_N;
                        int rowidxb = rowcolidx % TILE_SIZE_N;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];

                        const int startb = ld_gbl_auto(d_csrRowPtrB + rowidxb);
                        const int stopb = rowidxb == TILE_SIZE_N - 1 ? nnztotalb : ld_gbl_auto(d_csrRowPtrB + rowidxb + 1);
                        for (int kb = startb; kb < stopb; kb++)
                        {
                            TILE_CSR_COL_TYPE_B colidx = ld_gbl_auto(d_blkcsr_Col_B + nnzbstart + kb);
                            MAT_VAL_TYPE valb = ld_gbl_auto(d_blkcsr_Val_B + nnzbstart + kb);
                            s_blkcsr_Val_C_local[rowidxa * TILE_SIZE_M + colidx] += val * valb;
                        }
                    }
                }
                posa++; posa_updated = 1; posb++; posb_updated = 1;
            }
            else
            {
                posa_updated = idxa < idxb ? 1 : 0;
                posa += posa_updated;
                posb_updated = idxa > idxb ? 1 : 0;
                posb += posb_updated;
            }
        }
    }

    __syncwarp();

    // Write back results
    if (blknnzctotal == TILE_SIZE_M * TILE_SIZE_M){
#pragma unroll
        for (int c_adaptwarp_idx = lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += WARP_SIZE){
#pragma unroll
            for (int i = 0; i < TILE_SIZE_M; i++)
            {
                int offset_local = i * TILE_SIZE_M + c_adaptwarp_idx;
                d_blkcsr_Col_C[nnzcstart + offset_local] = c_adaptwarp_idx;
                d_blkcsr_Val_C[nnzcstart + offset_local] = s_blkcsr_Val_C_local[offset_local];
            }
        }
    }
    else{
        TILE_MASK_TYPE_B maskc[MaskNumC] = {};
        TILE_CSR_PTR_TYPE blknnzcstart;

#pragma unroll
        for (int c_adaptwarp_idx = lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += WARP_SIZE){
            long long int pos_c = (long long int)(tilei) * TILE_SIZE_M + c_adaptwarp_idx;
            blknnzcstart = d_blkcsr_Ptr_C[pos_c];
#pragma unroll
            for (int maskid = 0; maskid < MaskNumC; maskid++){
                maskc[maskid] = d_blkmaskC[pos_c * MaskNumC + maskid];
            }

            int cnt = 0;
#pragma unroll
            for (int maskid = 0; maskid < MaskNumC; maskid++){
#pragma unroll
                for (int i = 0; i < MaskBitsC; i++)
                {
                    int idx = ((maskc[maskid] >> MaskBitsC - i - 1) & 0x1) == 1 ? (maskid * MaskBitsC) + i : -1;
                    if (idx != -1)
                    {
                        d_blkcsr_Col_C[nnzcstart + blknnzcstart + cnt] = idx;
                        d_blkcsr_Val_C[nnzcstart + blknnzcstart + cnt] = s_blkcsr_Val_C_local[c_adaptwarp_idx * TILE_SIZE_M + idx];
                        cnt++;
                    }
                }
            }
        }
    }
}

__global__ void tile_spgemm_step4_cuda_dns_kernel_tensor_core_no_slot(int *d_blkrowptrA,
                                                                    const int *__restrict__ d_blkcolidxA,
                                                                    int *d_nnzb_A,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_A,
                                                                    TILE_CSR_COL_TYPE_A *d_blkcsr_Col_A,
                                                                    TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_A,
                                                                    MAT_VAL_TYPE *d_dense_val_A,
                                                                    int blkmA, int blknA, int numblkA, int nnzA,
                                                                    const int *__restrict__ d_blkcolptrB,
                                                                    const int *__restrict__ d_blkrowidxB,
                                                                    const int *__restrict__ d_nnzb_B,
                                                                    const MAT_VAL_TYPE *__restrict__ d_blkcsr_Val_B,
                                                                    const TILE_CSR_COL_TYPE_B *__restrict__ d_blkcsr_Col_B,
                                                                    const TILE_CSR_PTR_TYPE *__restrict__ d_blkcsr_Ptr_B,
                                                                    MAT_VAL_TYPE *d_dense_val_B,
                                                                    int blkmB, int blknB, int numblkB, int nnzB,
                                                                    int *d_blkrowidxC,
                                                                    int *d_blkcolidxC,
                                                                    TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_C,
                                                                    TILE_CSR_COL_TYPE_B *d_blkcsr_Col_C,
                                                                    MAT_VAL_TYPE *d_blkcsr_Val_C,
                                                                    int *d_nnzb_C,
                                                                    TILE_MASK_TYPE_B *d_blkmaskC,
                                                                    int numblkC,
                                                                    int *d_blkid,
                                                                    int *d_spec_intersection_cnt,
                                                                    int *d_spec_intersection_posa,
                                                                    int *d_spec_intersection_posb)
{
    const int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int global_warp_id = global_id / WARP_SIZE;

    if (global_warp_id >= numblkC)
        return;
    int tilei = d_blkid[global_warp_id];

    const int nnzcstart = d_nnzb_C[tilei];
    const int blknnzctotal = d_nnzb_C[tilei + 1] - nnzcstart;
    if (!blknnzctotal)
        return;

    const int total_threads = STEP4_TC_THREADS;
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps_per_block = total_threads / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;

    // ========== SHARED MEMORY ALLOCATION ==========
    __shared__ MAT_VAL_TYPE s_blkcsr_Val_C[num_warps_per_block * TILE_SIZE_M * TILE_SIZE_M];
    __shared__ int s_matched_posa[num_warps_per_block * SPECULATIVE_INTERSECTION];
    __shared__ int s_matched_posb[num_warps_per_block * SPECULATIVE_INTERSECTION];
    __shared__ int s_matchedcnt[num_warps_per_block];

    MAT_VAL_TYPE *s_blkcsr_Val_C_local = &s_blkcsr_Val_C[local_warp_id * TILE_SIZE_M * TILE_SIZE_M];
    int *s_matched_posa_local = &s_matched_posa[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matched_posb_local = &s_matched_posb[local_warp_id * SPECULATIVE_INTERSECTION];
    int *s_matchedcnt_local = &s_matchedcnt[local_warp_id];

    // Initialize C accumulator
#pragma unroll
    for (int c_adaptwarp_idx = lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += WARP_SIZE){
#pragma unroll
        for (int i = 0; i < TILE_SIZE_M; i++){
            s_blkcsr_Val_C_local[i * TILE_SIZE_M + c_adaptwarp_idx] = 0.0;
        }
    }

    if (!lane_id)
        s_matchedcnt_local[0] = 0;

    const int blki = d_blkrowidxC[tilei];
    const int blkj = d_blkcolidxC[tilei];

    const int abase = d_blkrowptrA[blki];
    const int astop = d_blkrowptrA[blki + 1];
    int lena = astop - abase;

    const int bbase = ld_gbl_auto(d_blkcolptrB + blkj);
    const int bstop = ld_gbl_auto(d_blkcolptrB + blkj + 1);
    int lenb = bstop - bbase;

    int matchedcnt = 0;
    int specres = 0;

    if (USE_GMEM_SPECULATIVE_INTERSECTION)
        matchedcnt = d_spec_intersection_cnt[tilei];

    if (USE_GMEM_SPECULATIVE_INTERSECTION && matchedcnt > 0)
    {}
    else
    {
        specres = intersection_binarysearch_kernel(d_blkcolidxA, abase, astop, lena,
                                                   d_blkrowidxB, bbase, bstop, lenb,
                                                   s_matched_posa_local, s_matched_posb_local,
                                                   SPECULATIVE_INTERSECTION, s_matchedcnt_local,
                                                   lane_id, WARP_SIZE);
        __syncwarp();
        matchedcnt = s_matchedcnt_local[0];
    }

    __syncwarp();

    if (matchedcnt <= SPECULATIVE_INTERSECTION && specres == 0)
    {
        const int NUM_SUBTILES = TILE_SIZE_M / 8;
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, MAT_VAL_TYPE, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, MAT_VAL_TYPE, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, MAT_VAL_TYPE> c_frag[NUM_SUBTILES + 1][NUM_SUBTILES + 1];

#pragma unroll
        for(int i = 0; i < NUM_SUBTILES; i++)
#pragma unroll
            for(int j = 0; j < NUM_SUBTILES; j++){
                wmma::fill_fragment(c_frag[i][j], 0.0);
            }

        // ========== MAIN COMPUTATION LOOP WITH SHARED SLOTS ==========
        for (int k = 0; k < matchedcnt; k++)
        {
            int posa = s_matched_posa_local[k];
            int posb = s_matched_posb_local[k];

            long long offset_A = (long long)(abase + posa) * TILE_SIZE_M * TILE_SIZE_N; 
            long long offset_B = (long long)(bbase + posb) * TILE_SIZE_N * TILE_SIZE_M; 

            MAT_VAL_TYPE *tile_A_ptr = d_dense_val_A + offset_A;
            MAT_VAL_TYPE *tile_B_ptr = d_dense_val_B + offset_B;

#pragma unroll
            for (int k_step = 0; k_step < TILE_SIZE_N; k_step += WMMA_K){
#pragma unroll
                for (int i = 0; i < NUM_SUBTILES; i++) {
#pragma unroll
                    for (int j = 0; j < NUM_SUBTILES; j++) {
                        const MAT_VAL_TYPE *ptr_a = tile_A_ptr + (i * WMMA_M * TILE_SIZE_N) + k_step;
                        const MAT_VAL_TYPE *ptr_b = tile_B_ptr + (k_step * TILE_SIZE_M) + (j * WMMA_N);
                        wmma::load_matrix_sync(a_frag, ptr_a, TILE_SIZE_N);
                        wmma::load_matrix_sync(b_frag, ptr_b, TILE_SIZE_M);
                        wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
                    }
                }
            }
        }

#pragma unroll
        for(int i=0; i<NUM_SUBTILES; i++) {
#pragma unroll
            for(int j=0; j<NUM_SUBTILES; j++) {
                MAT_VAL_TYPE *dst_ptr = s_blkcsr_Val_C_local + (i * sizeof(MAT_VAL_TYPE) * TILE_SIZE_M) + (j * sizeof(MAT_VAL_TYPE));
                wmma::store_matrix_sync(dst_ptr, c_frag[i][j], TILE_SIZE_M, wmma::mem_row_major);
            }
        }
    }
    else
    {
        // Fallback path for large intersection counts
        const int astart = d_blkcolidxA[abase];
        const int aend = d_blkcolidxA[astop - 1];
        const int bstart = ld_gbl_auto(d_blkrowidxB + bbase);
        const int bend = ld_gbl_auto(d_blkrowidxB + bstop - 1);

        int posa_real = 0, posb_real = 0;
        if (bstart > astart) {
            int posa_real_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bstart, lena);
            posa_real = posa_real_new < 0 ? 0 : posa_real_new;
        } else if (bstart < astart) {
            int posb_real_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, astart, lenb);
            posb_real = posb_real_new < 0 ? 0 : posb_real_new;
        }

        if (bstop < astop) {
            int lena_new = binary_search_right_boundary_kernel(d_blkcolidxA + abase, bend, lena) + 1;
            lena = lena_new > lena ? lena : lena_new;
        } else if (bstop > astop) {
            int lenb_new = binary_search_right_boundary_kernel(d_blkrowidxB + bbase, aend, lenb) + 1;
            lenb = lenb_new > lenb ? lenb : lenb_new;
        }

        int posa = posa_real, posb = posb_real;
        int idxa = 0, idxb = 0;
        int posa_updated = 1, posb_updated = 1;

        while (posa < lena && posb < lenb)
        {
            idxa = posa_updated ? ld_gbl_auto(d_blkcolidxA + abase + posa) : idxa;
            idxb = posb_updated ? ld_gbl_auto(d_blkrowidxB + bbase + posb) : idxb;

            if (idxa == idxb)
            {
                const int nnzastart = d_nnzb_A[(abase + posa)];
                int nnztotala = d_nnzb_A[(abase + posa) + 1] - nnzastart;
                const TILE_CSR_PTR_TYPE *__restrict__ d_csrRowPtrB = &d_blkcsr_Ptr_B[(bbase + posb) * TILE_SIZE_N];
                const int nnzbstart = ld_gbl_auto(d_nnzb_B + bbase + posb);
                int nnztotalb = ld_gbl_auto(d_nnzb_B + bbase + posb + 1) - nnzbstart;
#pragma unroll
                for (int c_adaptwarp_idx = lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += WARP_SIZE){   
                    TILE_CSR_PTR_TYPE offseta_start = d_blkcsr_Ptr_A[(abase + posa) * TILE_SIZE_M + c_adaptwarp_idx];
                    TILE_CSR_PTR_TYPE offseta_end = c_adaptwarp_idx == TILE_SIZE_M - 1 ? nnztotala : d_blkcsr_Ptr_A[(abase + posa) * TILE_SIZE_M + c_adaptwarp_idx + 1];

                    for (int i = offseta_start; i < offseta_end; i++)
                    {
                        TILE_CSR_COL_TYPE_A rowcolidx = d_blkcsr_Col_A[nnzastart + i];
                        int rowidxa = rowcolidx / TILE_SIZE_N;
                        int rowidxb = rowcolidx % TILE_SIZE_N;
                        MAT_VAL_TYPE val = d_blkcsr_Val_A[nnzastart + i];

                        const int startb = ld_gbl_auto(d_csrRowPtrB + rowidxb);
                        const int stopb = rowidxb == TILE_SIZE_N - 1 ? nnztotalb : ld_gbl_auto(d_csrRowPtrB + rowidxb + 1);
                        for (int kb = startb; kb < stopb; kb++)
                        {
                            TILE_CSR_COL_TYPE_B colidx = ld_gbl_auto(d_blkcsr_Col_B + nnzbstart + kb);
                            MAT_VAL_TYPE valb = ld_gbl_auto(d_blkcsr_Val_B + nnzbstart + kb);
                            s_blkcsr_Val_C_local[rowidxa * TILE_SIZE_M + colidx] += val * valb;
                        }
                    }
                }
                posa++; posa_updated = 1; posb++; posb_updated = 1;
            }
            else
            {
                posa_updated = idxa < idxb ? 1 : 0;
                posa += posa_updated;
                posb_updated = idxa > idxb ? 1 : 0;
                posb += posb_updated;
            }
        }
    }

    __syncwarp();

    // Write back results
    if (blknnzctotal == TILE_SIZE_M * TILE_SIZE_M){
#pragma unroll
        for (int c_adaptwarp_idx = lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += WARP_SIZE){
#pragma unroll
            for (int i = 0; i < TILE_SIZE_M; i++)
            {
                int offset_local = i * TILE_SIZE_M + c_adaptwarp_idx;
                d_blkcsr_Col_C[nnzcstart + offset_local] = c_adaptwarp_idx;
                d_blkcsr_Val_C[nnzcstart + offset_local] = s_blkcsr_Val_C_local[offset_local];
            }
        }
    }
    else{
        TILE_MASK_TYPE_B maskc[MaskNumC] = {};
        TILE_CSR_PTR_TYPE blknnzcstart;

#pragma unroll
        for (int c_adaptwarp_idx = lane_id; c_adaptwarp_idx < TILE_SIZE_M; c_adaptwarp_idx += WARP_SIZE){
            long long int pos_c = (long long int)(tilei) * TILE_SIZE_M + c_adaptwarp_idx;
            blknnzcstart = d_blkcsr_Ptr_C[pos_c];
#pragma unroll
            for (int maskid = 0; maskid < MaskNumC; maskid++){
                maskc[maskid] = d_blkmaskC[pos_c * MaskNumC + maskid];
            }

            int cnt = 0;
#pragma unroll
            for (int maskid = 0; maskid < MaskNumC; maskid++){
#pragma unroll
                for (int i = 0; i < MaskBitsC; i++)
                {
                    int idx = ((maskc[maskid] >> MaskBitsC - i - 1) & 0x1) == 1 ? (maskid * MaskBitsC) + i : -1;
                    if (idx != -1)
                    {
                        d_blkcsr_Col_C[nnzcstart + blknnzcstart + cnt] = idx;
                        d_blkcsr_Val_C[nnzcstart + blknnzcstart + cnt] = s_blkcsr_Val_C_local[c_adaptwarp_idx * TILE_SIZE_M + idx];
                        cnt++;
                    }
                }
            }
        }
    }
}

void tilespgemm(SMatrixA *matrixA,
                SMatrixB *matrixB,
                SMatrixB *matrixC,
                unsigned int *blk_intersec_bitmask_A,
                unsigned int *blk_intersec_bitmask_B,
                int blk_intersec_bitmask_len,
                double densityA,
                double densityB,
                unsigned long long int nnzCub,
                unsigned long long int *nnzC_computed,
                double *compression_rate,
                double *time_tile,
                double *gflops_tile,
                char *filename,
                double *time_symbolic, double *time_numeric, double *time_malloc,
                int tc_threshold)
{
    int *d_blkrowptrA;
    int *d_blkcolidxA;
    int *d_nnzb_A;
    MAT_VAL_TYPE *d_blkcsr_Val_A;
    TILE_CSR_COL_TYPE_A *d_blkcsr_Col_A;
    TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_A;
    int blkmA = matrixA->tilem;
    int blknA = matrixA->tilen;
    int nnzA = matrixA->nnz;
    int numblkA = matrixA->numtile;
    int *blkrowptrA = matrixA->tile_ptr;
    int *blkcolidxA = matrixA->tile_columnidx;
    int *nnzb_A = matrixA->tile_nnz;
    MAT_VAL_TYPE *blkcsr_Val_A = matrixA->tile_csr_Value;
    TILE_CSR_COL_TYPE_A *blkcsr_Col_A = matrixA->tile_csr_Col;
    TILE_CSR_PTR_TYPE *blkcsr_Ptr_A = matrixA->tile_csr_Ptr;
    TILE_MASK_TYPE_A *blkmaskA = matrixA->mask;

    cudaMalloc((void **)&d_blkrowptrA, (blkmA + 1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxA, numblkA * sizeof(int));
    cudaMalloc((void **)&d_nnzb_A, (numblkA + 1) * sizeof(int));
    cudaMalloc((void **)&d_blkcsr_Val_A, nnzA * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_blkcsr_Col_A, nnzA * sizeof(TILE_CSR_COL_TYPE_A));
    cudaMalloc((void **)&d_blkcsr_Ptr_A, numblkA * TILE_SIZE_M * sizeof(TILE_CSR_PTR_TYPE));

    cudaMemcpy(d_blkrowptrA, blkrowptrA, (blkmA + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxA, blkcolidxA, numblkA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzb_A, nnzb_A, (numblkA + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Val_A, blkcsr_Val_A, nnzA * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Col_A, blkcsr_Col_A, nnzA * sizeof(TILE_CSR_COL_TYPE_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Ptr_A, blkcsr_Ptr_A, numblkA * TILE_SIZE_M * sizeof(TILE_CSR_PTR_TYPE), cudaMemcpyHostToDevice);

    int *d_blkcolptrB;
    int *d_blkrowidxB;
    int *d_blkrowptrB;
    int *d_blkcolidxB;
    int *d_nnzb_B;
    MAT_VAL_TYPE *d_blkcsr_Val_B;
    TILE_CSR_COL_TYPE_B *d_blkcsr_Col_B;
    TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_B;
    int blknB = matrixB->tilen;
    int blkmB = matrixB->tilem;
    int numblkB = matrixB->numtile;
    int nnzB = matrixB->nnz;
    int *blkcolptrB = matrixB->csc_tile_ptr;
    int *blkrowidxB = matrixB->csc_tile_rowidx;
    int *blkrowptrB = matrixB->tile_ptr;
    int *blkcolidxB = matrixB->tile_columnidx;
    int *nnzb_B = matrixB->tile_nnz;
    MAT_VAL_TYPE *blkcsr_Val_B = matrixB->tile_csr_Value;
    TILE_CSR_COL_TYPE_B *blkcsr_Col_B = matrixB->tile_csr_Col;
    TILE_CSR_PTR_TYPE *blkcsr_Ptr_B = matrixB->tile_csr_Ptr;
    TILE_MASK_TYPE_B *blkmaskB = matrixB->mask;

    cudaMalloc((void **)&d_blkcolptrB, (blknB + 1) * sizeof(int));
    cudaMalloc((void **)&d_blkrowidxB, numblkB * sizeof(int));
    cudaMalloc((void **)&d_blkrowptrB, (blkmB + 1) * sizeof(int));
    cudaMalloc((void **)&d_blkcolidxB, numblkB * sizeof(int));
    cudaMalloc((void **)&d_nnzb_B, (numblkB + 1) * sizeof(int));
    cudaMalloc((void **)&d_blkcsr_Val_B, nnzB * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void **)&d_blkcsr_Col_B, nnzB * sizeof(TILE_CSR_COL_TYPE_B));
    cudaMalloc((void **)&d_blkcsr_Ptr_B, numblkB * TILE_SIZE_N * sizeof(TILE_CSR_PTR_TYPE));

    cudaMemcpy(d_blkcolptrB, blkcolptrB, (blknB + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowidxB, blkrowidxB, numblkB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkrowptrB, blkrowptrB, (blkmB + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcolidxB, blkcolidxB, numblkB * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnzb_B, nnzb_B, (numblkB + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Val_B, blkcsr_Val_B, nnzB * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Col_B, blkcsr_Col_B, nnzB * sizeof(TILE_CSR_COL_TYPE_B), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blkcsr_Ptr_B, blkcsr_Ptr_B, numblkB * TILE_SIZE_N * sizeof(TILE_CSR_PTR_TYPE), cudaMemcpyHostToDevice);

    unsigned int *d_blk_intersec_bitmask_A;
    unsigned int *d_blk_intersec_bitmask_B;

    cudaMalloc((void **)&d_blk_intersec_bitmask_A, blkmA * blk_intersec_bitmask_len * sizeof(unsigned int));
    cudaMalloc((void **)&d_blk_intersec_bitmask_B, blknB * blk_intersec_bitmask_len * sizeof(unsigned int));

    cudaMemcpy(d_blk_intersec_bitmask_A, blk_intersec_bitmask_A, blkmA * blk_intersec_bitmask_len * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blk_intersec_bitmask_B, blk_intersec_bitmask_B, blknB * blk_intersec_bitmask_len * sizeof(unsigned int), cudaMemcpyHostToDevice);

    TILE_MASK_TYPE_B *d_blkmaskB;
    cudaMalloc((void **)&d_blkmaskB, numblkB * TILE_SIZE_N * MaskNumB * sizeof(TILE_MASK_TYPE_B));
    cudaMemcpy(d_blkmaskB, blkmaskB, numblkB * TILE_SIZE_N * MaskNumB * sizeof(TILE_MASK_TYPE_B), cudaMemcpyHostToDevice);

    TILE_MASK_TYPE_A *d_blkmaskA;
    cudaMalloc((void **)&d_blkmaskA, numblkA * TILE_SIZE_M * MaskNumA * sizeof(TILE_MASK_TYPE_A));
    cudaMemcpy(d_blkmaskA, blkmaskA, numblkA * TILE_SIZE_M * MaskNumA * sizeof(TILE_MASK_TYPE_A), cudaMemcpyHostToDevice);

    MAT_VAL_TYPE *d_dense_val_A;
    MAT_VAL_TYPE *d_dense_val_B;
    int *d_has_dense_calculated;
    int *d_tile_dense_ready_A;
    int *d_tile_dense_ready_B;
    
    // Allocate dense data with actual dense_tile_count size
    cudaMalloc((void**)&d_dense_val_A, matrixA->dense_tile_count * TILE_SIZE_M * TILE_SIZE_N * sizeof(MAT_VAL_TYPE));
    cudaMalloc((void**)&d_dense_val_B, matrixB->dense_tile_count * TILE_SIZE_N * TILE_SIZE_M * sizeof(MAT_VAL_TYPE));
    cudaMemcpy(d_dense_val_A, matrixA->dense_data, matrixA->dense_tile_count * TILE_SIZE_M * TILE_SIZE_N * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dense_val_B, matrixB->dense_data, matrixB->dense_tile_count * TILE_SIZE_N * TILE_SIZE_M * sizeof(MAT_VAL_TYPE), cudaMemcpyHostToDevice);
    
    // Allocate and copy tile_dense_ready arrays
    cudaMalloc((void**)&d_tile_dense_ready_A, numblkA * sizeof(int));
    cudaMalloc((void**)&d_tile_dense_ready_B, numblkB * sizeof(int));
    cudaMemcpy(d_tile_dense_ready_A, matrixA->tile_dense_ready, numblkA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile_dense_ready_B, matrixB->tile_dense_ready, numblkB * sizeof(int), cudaMemcpyHostToDevice);

    int numblkC = 0;
    int nnzC = 0;
    double tile_spgemm_time = 0;

    tc_threshold = (tc_threshold == 0) ? 1 : tc_threshold;
    cudaMemcpyToSymbol(d_SMEM_LRG_TH, &tc_threshold, sizeof(int));

    int nstreams = 5;

    cudaStream_t *streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nstreams);
    for (int i = 0; i < nstreams; i++)
    {
        cudaStreamCreate(&(streams[i]));
    }

    double time_all[REPEAT_NUM];

    int *d_blksmem_tny_cnt;
    int *d_blksmem_sml_cnt;
    int *d_blksmem_lrg_cnt;
    int *d_blksmem_dns_cnt;
    int *d_blksmem_ful_cnt;

    cudaMalloc((void **)&d_blksmem_tny_cnt, 1 * sizeof(int));
    cudaMalloc((void **)&d_blksmem_sml_cnt, 1 * sizeof(int));
    cudaMalloc((void **)&d_blksmem_lrg_cnt, 1 * sizeof(int));
    cudaMalloc((void **)&d_blksmem_dns_cnt, 1 * sizeof(int));
    cudaMalloc((void **)&d_blksmem_ful_cnt, 1 * sizeof(int));

    for (int ri = 0; ri < REPEAT_NUM; ri++)
    {
        // call cuda kernel
        struct timeval tstart, tend;
        struct timeval t1, t2;
        cudaDeviceSynchronize();
        gettimeofday(&tstart, NULL);

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        int *d_blkrowptrC;
        cudaMalloc((void **)&d_blkrowptrC, (blkmA + 1) * sizeof(int));
        int *f_h_tile_ptr_C = (int *)malloc((blkmA + 1) * sizeof(int));

#if TIMING
        *time_malloc = 0;
        *time_symbolic = 0;
        gettimeofday(&t2, NULL);
        *time_malloc += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

        numblkC = 0;
        sfBIN bin;
#if TIMING
        gettimeofday(&t1, NULL);
#endif
        if (blknB > NUMCOLC_SPA_OR_HASH_TH)
        {
            /* Initialize bin */
            init_bin(&bin, blkmA);

            /* Set max bin */
            set_max_bin(d_blkrowptrA, d_blkcolidxA, d_blkrowptrB, &bin, blkmA);
            /* Count nz of C */
            set_row_nnz(d_blkrowptrA, d_blkcolidxA,
                        d_blkrowptrB, d_blkcolidxB,
                        d_blkrowptrC,
                        &bin,
                        blkmA,
                        &numblkC);
            /* Set bin */
            set_min_bin(&bin, blkmA);
        }
        else
        {
            int num_threads = WARP_SIZE * WARP_PER_BLOCK;
            int num_blocks = ceil((double)blkmA / (double)(WARP_PER_BLOCK));
            tile_spgemm_step1_cuda_spa_kernel<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, blkmA,
                                                                           d_blkrowptrB, d_blkcolidxB, blknB,
                                                                           d_blkrowptrC);
            exclusive_scan_device_cuda_thrust<int>(d_blkrowptrC, blkmA + 1);
            cudaMemcpy(&numblkC, &d_blkrowptrC[blkmA], sizeof(int), cudaMemcpyDeviceToHost);
        }

#if TIMING
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        *time_symbolic += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif


#if TIMING
    gettimeofday(&t1, NULL);
#endif
        int *d_blkrowidxC;
        cudaMalloc((void **)&d_blkrowidxC, numblkC * sizeof(int));
        int *d_blkcolidxC;
        cudaMalloc((void **)&d_blkcolidxC, numblkC * sizeof(int));
        int *d_has_dense_calculated;
        cudaMalloc((void**)&d_has_dense_calculated, numblkC * sizeof(bool));
        cudaMemset(d_has_dense_calculated, 0, numblkC * sizeof(bool));

#if TIMING
        gettimeofday(&t2, NULL);
        *time_malloc += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        int *d_spec_intersection_cnt;
        int *d_spec_intersection_posa;
        int *d_spec_intersection_posb;

        if (USE_GMEM_SPECULATIVE_INTERSECTION)
        {
            cudaMalloc((void **)&d_spec_intersection_cnt, numblkC * sizeof(int));
            cudaMemset(d_spec_intersection_cnt, 0, numblkC * sizeof(int));
            cudaMalloc((void **)&d_spec_intersection_posa, numblkC * GMEM_SPECULATIVE_INTERSECTION * sizeof(int));
            cudaMalloc((void **)&d_spec_intersection_posb, numblkC * GMEM_SPECULATIVE_INTERSECTION * sizeof(int));
        }

        if (blknB > NUMCOLC_SPA_OR_HASH_TH)
        {
            calculate_value_col_bin(d_blkrowptrA, d_blkcolidxA, NULL,
                                    d_blkrowptrB, d_blkcolidxB, NULL,
                                    d_blkrowptrC, d_blkrowidxC, d_blkcolidxC, NULL,
                                    &bin, blkmA, blkmB);        
            release_bin(bin);
        }
        else
        {
            int num_threads = WARP_SIZE * WARP_PER_BLOCK;
            int num_blocks = ceil((double)blkmA / (double)(WARP_PER_BLOCK));
            tile_spgemm_step1_numeric_cuda_spa_kernel<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, blkmA,
                                                                                   d_blkrowptrB, d_blkcolidxB, blknB,
                                                                                   d_blkrowptrC, d_blkrowidxC, d_blkcolidxC,
                                                                                   d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        }

#if TIMING
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        *time_symbolic += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

#endif

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        long long int lengthC =  (long long int)numblkC * TILE_SIZE_M;

        TILE_CSR_PTR_TYPE *d_blkcsr_Ptr_C;
        cudaMalloc((void **)&d_blkcsr_Ptr_C, lengthC * sizeof(TILE_CSR_PTR_TYPE));
        
        if (d_blkcsr_Ptr_C == NULL)
        {
            printf("d_blkcsr_Ptr_C failed\n");
        }

        // tile_nnz_C
        int *d_nnzb_C;
        cudaMalloc((void **)&d_nnzb_C, (numblkC + 1) * sizeof(int));
        
        cudaMemset(d_nnzb_C, 0, (numblkC + 1) * sizeof(int));

        TILE_MASK_TYPE_B *d_blkmaskC;
        cudaMalloc((void **)&d_blkmaskC, lengthC * MaskNumC * sizeof(TILE_MASK_TYPE_B)); // tile_size_k

        int *d_blkid_smem_tny;
        int *d_blkid_smem_sml;
        int *d_blkid_smem_lrg;
        int *d_blkid_smem_dns;
        int *d_blkid_smem_ful;

        cudaMalloc((void **)&d_blkid_smem_tny, numblkC * sizeof(int));
        cudaMalloc((void **)&d_blkid_smem_sml, numblkC * sizeof(int));
        cudaMalloc((void **)&d_blkid_smem_lrg, numblkC * sizeof(int));
        cudaMalloc((void **)&d_blkid_smem_dns, numblkC * sizeof(int));
        cudaMalloc((void **)&d_blkid_smem_ful, numblkC * sizeof(int));

        cudaMemset(d_blksmem_tny_cnt, 0, 1 * sizeof(int));
        cudaMemset(d_blksmem_sml_cnt, 0, 1 * sizeof(int));
        cudaMemset(d_blksmem_lrg_cnt, 0, 1 * sizeof(int));
        cudaMemset(d_blksmem_dns_cnt, 0, 1 * sizeof(int));
        cudaMemset(d_blksmem_ful_cnt, 0, 1 * sizeof(int));

        int num_threads, num_blocks;

#if TIMING
        gettimeofday(&t2, NULL);
        *time_malloc += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        if (densityA > INTERSECTION_SPARSE_OR_DNS_TH && densityB > INTERSECTION_SPARSE_OR_DNS_TH && USE_DENSE)
        {
            if (USE_DNS_THREAD){}
            else
            {}
        }
    
        else
        {
            if (USE_HALFWARP)
            {
                num_threads = STEP3_THREADS;
                #if TILE_SIZE_M == 8
                    num_blocks = ceil((double)numblkC / (double)(num_threads / TILE_SIZE_M * TILE_PER_QUADWARP));
                #elif TILE_SIZE_M == 16
                    num_blocks = ceil((double)numblkC / (double)(num_threads / TILE_SIZE_M * TILE_PER_HALFWARP));
                #else
                    num_blocks = ceil((double)numblkC / (double)(num_threads / 32 * TILE_PER_WARP));
                #endif

                #if TILE_SIZE_M == 8
                    tile_spgemm_step3_cuda_kernel_2level_quadwarp<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                                                                                           blkmA, blknA, numblkA, nnzA,
                                                                                           d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                                                                                           blkmB, blknB, numblkB, nnzB,
                                                                                           d_blk_intersec_bitmask_A, d_blk_intersec_bitmask_B, blk_intersec_bitmask_len,
                                                                                           d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC,
                                                                                           d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, d_blksmem_ful_cnt,
                                                                                           d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, d_blkid_smem_ful,
                                                                                           d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb,
                                                                                           numblkC);
                #elif TILE_SIZE_M == 16
                    tile_spgemm_step3_cuda_kernel_2level_halfwarp<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                                                                                           blkmA, blknA, numblkA, nnzA,
                                                                                           d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                                                                                           blkmB, blknB, numblkB, nnzB,
                                                                                           d_blk_intersec_bitmask_A, d_blk_intersec_bitmask_B, blk_intersec_bitmask_len,
                                                                                           d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC,
                                                                                           d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, d_blksmem_ful_cnt,
                                                                                           d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, d_blkid_smem_ful,
                                                                                           d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb,
                                                                                           numblkC);
                #elif TILE_SIZE_M >= 32
                    tile_spgemm_step3_cuda_kernel_2level_warp<<<num_blocks, num_threads>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A, d_blkmaskA,
                                                                                           blkmA, blknA, numblkA, nnzA,
                                                                                           d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B, d_blkmaskB,
                                                                                           blkmB, blknB, numblkB, nnzB,
                                                                                           d_blk_intersec_bitmask_A, d_blk_intersec_bitmask_B, blk_intersec_bitmask_len,
                                                                                           d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C, d_nnzb_C, d_blkmaskC,
                                                                                           d_blksmem_tny_cnt, d_blksmem_sml_cnt, d_blksmem_lrg_cnt, d_blksmem_dns_cnt, d_blksmem_ful_cnt,
                                                                                           d_blkid_smem_tny, d_blkid_smem_sml, d_blkid_smem_lrg, d_blkid_smem_dns, d_blkid_smem_ful,
                                                                                           d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb,
                                                                                           numblkC);
                #endif
            }
            else{}
        }
        
        int *h_nnzb_C = (int *)malloc((numblkC + 1) * sizeof(int));
        memset(h_nnzb_C, 0, (numblkC + 1) * sizeof(int));
        cudaMemcpy(h_nnzb_C, d_nnzb_C, (numblkC + 1)* sizeof(int), cudaMemcpyDeviceToHost);

        exclusive_scan_device_cuda_thrust<int>(d_nnzb_C, numblkC + 1);
        nnzC = 0;
        cudaMemcpy(&nnzC, &d_nnzb_C[numblkC], sizeof(int), cudaMemcpyDeviceToHost);

#if TIMING
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        *time_symbolic += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        if (ri == 0)
        {
            printf("\n[Symbolic Stage]\n");
            printf("  Runtime                  : %.2f ms\n", *time_symbolic);
            printf("\n");
            printf("--------------------------------------------------------------------------------\n");
        }
#endif

#if TIMING
        gettimeofday(&t1, NULL);
#endif

        TILE_CSR_COL_TYPE_B *d_blkcsr_Col_C;
        cudaMalloc((void **)&d_blkcsr_Col_C, nnzC * sizeof(TILE_CSR_COL_TYPE_B));
        MAT_VAL_TYPE *d_blkcsr_Val_C;
        cudaMalloc((void **)&d_blkcsr_Val_C, nnzC * sizeof(MAT_VAL_TYPE));

        int blksmem_tny_cnt = 0;
        int blksmem_sml_cnt = 0;
        int blksmem_lrg_cnt = 0;
        int blksmem_dns_cnt = 0;
        int blksmem_ful_cnt = 0;

        cudaMemcpy(&blksmem_tny_cnt, d_blksmem_tny_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&blksmem_sml_cnt, d_blksmem_sml_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&blksmem_lrg_cnt, d_blksmem_lrg_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&blksmem_dns_cnt, d_blksmem_dns_cnt, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&blksmem_ful_cnt, d_blksmem_ful_cnt, sizeof(int), cudaMemcpyDeviceToHost);

#if TIMING
        gettimeofday(&t2, NULL);
        *time_malloc += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#endif

    // THREADS_USED may not exceed 32, otherwise intersection_binarysearch_kernel() will fail
    const int THREADS_USED_TNY = THREADS_USED_TNY_TH < 32 ? THREADS_USED_TNY_TH : 32;
    const int THREADS_USED_SML = THREADS_USED_SML_TH < 32 ? THREADS_USED_SML_TH : 32;
    const int THREADS_USED_LRG = THREADS_USED_LRG_TH < 32 ? THREADS_USED_LRG_TH : 32;
    const int THREADS_USED_DNS = THREADS_USED_DNS_TH < 32 ? THREADS_USED_DNS_TH : 32;
    
    printf("\n  Output Tile Classification:\n");
    printf("  %-10s  %10s  %12s  %10s\n",
           "Type", "Count", "Threshold", "Threads");
    printf("  %-10s  %10s  %12s  %10s\n",
           "----------", "----------", "------------", "----------");
    printf("  %-10s  %10d  %12d  %10d\n",
           "Tiny",  blksmem_tny_cnt, SMEM_TNY_TH, THREADS_USED_TNY);
    printf("  %-10s  %10d  %12d  %10d\n",
           "Small", blksmem_sml_cnt, SMEM_SML_TH, THREADS_USED_SML);
    printf("  %-10s  %10d  %12d  %10d\n",
           "Large", blksmem_lrg_cnt, tc_threshold, THREADS_USED_LRG);
    printf("  %-10s  %10d  %12d  %10d\n",
           "Dense", blksmem_dns_cnt, SMEM_DNS_TH, THREADS_USED_DNS);
    printf("  %-10s  %10d  %12d  %10d\n",
           "Full",  blksmem_ful_cnt, TILE_SIZE_M * TILE_SIZE_M, 32);
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");

#if TIMING
        gettimeofday(&t1, NULL);
#endif

    // tiny : 1 - 32
    if (blksmem_tny_cnt)
    {
        #if USE_COOPERATION
        num_threads = STEP4_THREADS;
        num_blocks = ceil((double)blksmem_tny_cnt / (double)(num_threads / THREADS_USED_TNY));
        tile_spgemm_step4_cuda_sparse_kernel_adaptive_warp<SMEM_TNY_TH, THREADS_USED_TNY><<<num_blocks, num_threads, 0, streams[0]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                                blkmA, blknA, numblkA, nnzA,
                                                                                                                d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                                blkmB, blknB, numblkB, nnzB,
                                                                                                                d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                                d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                                d_nnzb_C, d_blkmaskC, blksmem_tny_cnt, d_blkid_smem_tny,
                                                                                                                d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        #else
        num_threads = STEP4_THREADS;
        num_blocks = ceil((double)blksmem_tny_cnt / (double)(num_threads / 32));
        tile_spgemm_step4_cuda_sparse_kernel_adaptive_warp<SMEM_TNY_TH, 32><<<num_blocks, num_threads, 0, streams[0]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                                blkmA, blknA, numblkA, nnzA,
                                                                                                                d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                                blkmB, blknB, numblkB, nnzB,
                                                                                                                d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                                d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                                d_nnzb_C, d_blkmaskC, blksmem_tny_cnt, d_blkid_smem_tny,
                                                                                                                d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        #endif
    }

    // small : 33 - 64
    if (blksmem_sml_cnt)
    {
        #if USE_COOPERATION
        num_threads = STEP4_THREADS;
        num_blocks = ceil((double)blksmem_sml_cnt / (double)(num_threads / THREADS_USED_SML));
        tile_spgemm_step4_cuda_sparse_kernel_adaptive_warp<SMEM_SML_TH, THREADS_USED_SML><<<num_blocks, num_threads, 0, streams[1]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                        blkmA, blknA, numblkA, nnzA,
                                                                                                        d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                        blkmB, blknB, numblkB, nnzB,
                                                                                                        d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                        d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                        d_nnzb_C, d_blkmaskC, blksmem_sml_cnt, d_blkid_smem_sml,
                                                                                                        d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        #else
        num_threads = STEP4_THREADS;
        num_blocks = ceil((double)blksmem_sml_cnt / (double)(num_threads / 32));
        tile_spgemm_step4_cuda_sparse_kernel_adaptive_warp<SMEM_SML_TH, 32><<<num_blocks, num_threads, 0, streams[1]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                        blkmA, blknA, numblkA, nnzA,
                                                                                                        d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                        blkmB, blknB, numblkB, nnzB,
                                                                                                        d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                        d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                        d_nnzb_C, d_blkmaskC, blksmem_sml_cnt, d_blkid_smem_sml,
                                                                                                        d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        #endif
    }

    // large : 65 - 128
    if (blksmem_lrg_cnt)
    {
        num_threads = STEP4_THREADS;
        num_blocks = ceil((double)blksmem_lrg_cnt / (double)(num_threads / 32));
        tile_spgemm_step4_cuda_sparse_kernel_adaptive_warp<SMEM_LRG_TH_DEFAULT, THREADS_USED_LRG><<<num_blocks, num_threads, 0, streams[2]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                        blkmA, blknA, numblkA, nnzA,
                                                                                                        d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                        blkmB, blknB, numblkB, nnzB,
                                                                                                        d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                        d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                        d_nnzb_C, d_blkmaskC, blksmem_lrg_cnt, d_blkid_smem_lrg,
                                                                                                        d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
    }

    // dns : 129 - dns
    if (blksmem_dns_cnt)
    {
        #if ENABLE_MULTI_WARP_SHARED_SLOT && TILE_SIZE_M <= 32 && TILE_SIZE_N <=32
        // Use shared slot kernel for better memory efficiency
        num_threads = STEP4_TC_THREADS;
        num_blocks = ceil((double)blksmem_dns_cnt / (double)(num_threads / WARP_SIZE));
        tile_spgemm_step4_cuda_dns_kernel_shared_slot<<<num_blocks, num_threads, 0, streams[3]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                                    d_blkmaskA, d_dense_val_A,
                                                                                                                    d_tile_dense_ready_A,
                                                                                                                    blkmA, blknA, numblkA, nnzA,
                                                                                                                    d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                                    d_blkmaskB, d_dense_val_B,
                                                                                                                    d_tile_dense_ready_B,
                                                                                                                    blkmB, blknB, numblkB, nnzB,
                                                                                                                    d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                                    d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                                    d_nnzb_C, d_blkmaskC, blksmem_dns_cnt, d_blkid_smem_dns,
                                                                                                                    d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb,
                                                                                                                    d_has_dense_calculated);
        #elif USE_TENSORCORE && TILE_SIZE_M <= 32 && TILE_SIZE_N <=32
        num_threads = STEP4_TC_THREADS;
        num_blocks = ceil((double)blksmem_dns_cnt / (double)(num_threads / WARP_SIZE));
        tile_spgemm_step4_cuda_dns_kernel_tensor_core_no_slot<<<num_blocks, num_threads, 0, streams[3]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                                    d_dense_val_A,
                                                                                                                    blkmA, blknA, numblkA, nnzA,
                                                                                                                    d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                                    d_dense_val_B,
                                                                                                                    blkmB, blknB, numblkB, nnzB,
                                                                                                                    d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                                    d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                                    d_nnzb_C, d_blkmaskC, blksmem_dns_cnt, d_blkid_smem_dns,
                                                                                                                    d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        #else
        // Fallback to basic dense kernel
        num_threads = STEP4_TC_THREADS;
        num_blocks = ceil((double)blksmem_dns_cnt / (double)(num_threads / THREADS_USED_DNS));
        tile_spgemm_step4_cuda_dns_kernel_adaptive_warp<THREADS_USED_DNS><<<num_blocks, num_threads, 0, streams[3]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                        blkmA, blknA, numblkA, nnzA,
                                                                                                        d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                        blkmB, blknB, numblkB, nnzB,
                                                                                                        d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                        d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                        d_nnzb_C, d_blkmaskC, blksmem_dns_cnt, d_blkid_smem_dns,
                                                                                                        d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        #endif
    }

    // ful : 256
    if (blksmem_ful_cnt)
    {
        #if ENABLE_MULTI_WARP_SHARED_SLOT && TILE_SIZE_M <= 32 && TILE_SIZE_N <=32
        // Use shared slot kernel for better memory efficiency
        num_threads = STEP4_TC_THREADS;
        num_blocks = ceil((double)blksmem_ful_cnt / (double)(num_threads / WARP_SIZE));
        tile_spgemm_step4_cuda_dns_kernel_shared_slot<<<num_blocks, num_threads, 0, streams[4]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                                    d_blkmaskA, d_dense_val_A,
                                                                                                                    d_tile_dense_ready_A,
                                                                                                                    blkmA, blknA, numblkA, nnzA,
                                                                                                                    d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                                    d_blkmaskB, d_dense_val_B,
                                                                                                                    d_tile_dense_ready_B,
                                                                                                                    blkmB, blknB, numblkB, nnzB,
                                                                                                                    d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                                    d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                                    d_nnzb_C, d_blkmaskC, blksmem_ful_cnt, d_blkid_smem_ful,
                                                                                                                    d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb,
                                                                                                                    d_has_dense_calculated);
        #elif USE_TENSORCORE && TILE_SIZE_M <= 32 && TILE_SIZE_N <=32
        num_threads = STEP4_TC_THREADS;
        num_blocks = ceil((double)blksmem_ful_cnt / (double)(num_threads / WARP_SIZE));
        tile_spgemm_step4_cuda_dns_kernel_tensor_core_no_slot<<<num_blocks, num_threads, 0, streams[4]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                                    d_dense_val_A,
                                                                                                                    blkmA, blknA, numblkA, nnzA,
                                                                                                                    d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                                    d_dense_val_B,
                                                                                                                    blkmB, blknB, numblkB, nnzB,
                                                                                                                    d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                                    d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                                    d_nnzb_C, d_blkmaskC, blksmem_ful_cnt, d_blkid_smem_ful,
                                                                                                                    d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        #else
        // Fallback to basic dense kernel
        num_threads = STEP4_TC_THREADS;
        num_blocks = ceil((double)blksmem_ful_cnt / (double)(num_threads / THREADS_USED_DNS));
        tile_spgemm_step4_cuda_dns_kernel_adaptive_warp<THREADS_USED_DNS><<<num_blocks, num_threads, 0, streams[4]>>>(d_blkrowptrA, d_blkcolidxA, d_nnzb_A, d_blkcsr_Val_A, d_blkcsr_Col_A, d_blkcsr_Ptr_A,
                                                                                                        blkmA, blknA, numblkA, nnzA,
                                                                                                        d_blkcolptrB, d_blkrowidxB, d_nnzb_B, d_blkcsr_Val_B, d_blkcsr_Col_B, d_blkcsr_Ptr_B,
                                                                                                        blkmB, blknB, numblkB, nnzB,
                                                                                                        d_blkrowidxC, d_blkcolidxC, d_blkcsr_Ptr_C,
                                                                                                        d_blkcsr_Col_C, d_blkcsr_Val_C,
                                                                                                        d_nnzb_C, d_blkmaskC, blksmem_ful_cnt, d_blkid_smem_ful,
                                                                                                        d_spec_intersection_cnt, d_spec_intersection_posa, d_spec_intersection_posb);
        #endif
    }

#if TIMING
        cudaDeviceSynchronize();
        gettimeofday(&t2, NULL);
        *time_numeric = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
        if (ri == 0)
        {
            printf("\n[Numeric Stage]\n");
            printf("  Runtime                  : %.2f ms\n", *time_numeric);
            printf("\n");
            printf("--------------------------------------------------------------------------------\n");
            printf("\n[Malloc]\n");
            printf("  Memory Allocation        : %.2f ms\n", *time_malloc);
            printf("\n");
            printf("--------------------------------------------------------------------------------\n");
        }

#endif

        cudaDeviceSynchronize();
        gettimeofday(&tend, NULL);
        double time = (tend.tv_sec - tstart.tv_sec) * 1000.0 + (tend.tv_usec - tstart.tv_usec) / 1000.0;
        time_all[ri] = time;
        tile_spgemm_time += time;

#if CHECK_RESULT
        int *h_tile_nnz_C = (int *)malloc((numblkC + 1) * sizeof(int));
        int *h_tile_ptr_C = (int *)malloc((blkmA + 1) * sizeof(int));
        int *h_tile_columnidx_C = (int *)malloc(numblkC * sizeof(int));
        MAT_VAL_TYPE *h_tile_csr_Value_C = (MAT_VAL_TYPE *)malloc(nnzC * sizeof(MAT_VAL_TYPE));
        TILE_CSR_COL_TYPE_B *h_tile_csr_Col_C = (TILE_CSR_COL_TYPE_B *)malloc(nnzC * sizeof(TILE_CSR_COL_TYPE_B));
        TILE_CSR_PTR_TYPE *h_tile_csr_Ptr_C = (TILE_CSR_PTR_TYPE *)malloc(numblkC * TILE_SIZE_M * sizeof(TILE_CSR_PTR_TYPE));

        cudaMemcpy(h_tile_nnz_C, d_nnzb_C, (numblkC + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tile_ptr_C, d_blkrowptrC, (blkmA + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tile_columnidx_C, d_blkcolidxC, numblkC * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tile_csr_Value_C, d_blkcsr_Val_C, nnzC * sizeof(MAT_VAL_TYPE), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tile_csr_Col_C, d_blkcsr_Col_C, nnzC * sizeof(TILE_CSR_COL_TYPE_B), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tile_csr_Ptr_C, d_blkcsr_Ptr_C, numblkC * TILE_SIZE_M * sizeof(TILE_CSR_PTR_TYPE), cudaMemcpyDeviceToHost);

        matrixC->tile_ptr = h_tile_ptr_C;
        matrixC->tile_columnidx = h_tile_columnidx_C;
        matrixC->tile_nnz = h_tile_nnz_C;
        matrixC->numtile = numblkC;
        matrixC->nnz = nnzC;
        matrixC->m = matrixA->m;
        matrixC->n = matrixB->n;
        matrixC->tilem = matrixA->tilem;
        matrixC->tilen = matrixB->tilen;
        matrixC->tile_csr_Value = h_tile_csr_Value_C;
        matrixC->tile_csr_Col = h_tile_csr_Col_C;
        matrixC->tile_csr_Ptr = h_tile_csr_Ptr_C;

#endif

        cudaFree(d_blkrowptrC);
        cudaFree(d_blkrowidxC);
        cudaFree(d_blkcolidxC);
        cudaFree(d_blkmaskC);
        cudaFree(d_nnzb_C);
        cudaFree(d_blkcsr_Ptr_C);
        cudaFree(d_blkcsr_Col_C);
        cudaFree(d_blkcsr_Val_C);
        cudaFree(d_blkid_smem_tny);
        cudaFree(d_blkid_smem_sml);
        cudaFree(d_blkid_smem_lrg);
        cudaFree(d_blkid_smem_dns);
        cudaFree(d_blkid_smem_ful);
        if (USE_GMEM_SPECULATIVE_INTERSECTION)
        {
            cudaFree(d_spec_intersection_cnt);
            cudaFree(d_spec_intersection_posa);
            cudaFree(d_spec_intersection_posb);
        }
    }

    double time_min = time_all[0];
    for (int ri = 1; ri < REPEAT_NUM; ri++)
        time_min = time_min > time_all[ri] ? time_all[ri] : time_min;

    *nnzC_computed = nnzC;
    *compression_rate = (double)nnzCub / (double)(*nnzC_computed);
    tile_spgemm_time = time_min;
    *time_tile = tile_spgemm_time;
    *gflops_tile = 2.0 * (double)nnzCub / (tile_spgemm_time * 1e6);

    printf("\n[FlexSpGEMM Summary]\n");
    printf("  Non-empty Output Tiles   : %d\n",   numblkC);
    printf("  NNZ (C)                  : %d\n",   nnzC);
    printf("  Total Runtime            : %.2f ms\n", tile_spgemm_time);
    printf("  Throughput               : %.2f GFlops\n", *gflops_tile);
    printf("\n");
    printf("--------------------------------------------------------------------------------\n");

    cudaFree(d_blksmem_tny_cnt);
    cudaFree(d_blksmem_sml_cnt);
    cudaFree(d_blksmem_lrg_cnt);
    cudaFree(d_blksmem_dns_cnt);
    cudaFree(d_blksmem_ful_cnt);

    cudaFree(d_blkrowptrA);
    cudaFree(d_blkcolidxA);
    cudaFree(d_nnzb_A);
    cudaFree(d_blkcsr_Val_A);
    cudaFree(d_blkcsr_Col_A);
    cudaFree(d_blkcsr_Ptr_A);
    cudaFree(d_blkcolptrB);
    cudaFree(d_blkrowidxB);
    cudaFree(d_blkrowptrB);
    cudaFree(d_blkcolidxB);
    cudaFree(d_nnzb_B);
    cudaFree(d_blkcsr_Val_B);
    cudaFree(d_blkcsr_Col_B);
    cudaFree(d_blkcsr_Ptr_B);
    cudaFree(d_blkmaskB);
    cudaFree(d_blkmaskA);
    cudaFree(d_blk_intersec_bitmask_A);
    cudaFree(d_blk_intersec_bitmask_B);

    for (int i = 0; i < nstreams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
}