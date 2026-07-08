#pragma once

#include "common.h"

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/iterator/transform_iterator.h>

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu_csr2tile {

using Key = unsigned long long;

constexpr int kThreadsPerBlock = 256;
constexpr int kTileThreadsPerBlock = 64;

inline std::string cuda_error_string(cudaError_t err, const char *expr, const char *file, int line) {
    std::ostringstream os;
    os << file << ":" << line << ": CUDA call failed: " << expr << ": "
       << cudaGetErrorString(err);
    return os.str();
}

#define GPU_CSR2TILE_CHECK(expr)                                                            \
    do {                                                                                    \
        cudaError_t _err = (expr);                                                          \
        if (_err != cudaSuccess) {                                                          \
            throw std::runtime_error(gpu_csr2tile::cuda_error_string(_err, #expr, __FILE__, __LINE__)); \
        }                                                                                   \
    } while (0)

struct DeviceCsr {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    thrust::device_vector<MAT_PTR_TYPE> rowptr;
    thrust::device_vector<int> colidx;
    thrust::device_vector<MAT_VAL_TYPE> values;
};

struct TileStructure {
    int primary_count = 0;
    int secondary_count = 0;
    int num_tiles = 0;
    std::vector<int> ptr;
    std::vector<int> secondary_idx;
    std::vector<int> primary_idx;
};

struct DeviceTileStructure {
    int primary_count = 0;
    int secondary_count = 0;
    int num_tiles = 0;
    thrust::device_vector<int> d_ptr;
    thrust::device_vector<int> d_secondary_idx;
    thrust::device_vector<int> d_primary_idx;
};

template <typename ColT, typename MaskT>
struct FullTileBuild {
    int tile_rows = 0;
    int tile_cols = 0;
    int tile_row_size = 0;
    int tile_col_size = 0;
    int num_tiles = 0;
    int dense_tile_count = 0;
    std::vector<int> ptr;
    std::vector<int> secondary_idx;
    std::vector<int> primary_idx;
    std::vector<int> tile_nnz;
    std::vector<TILE_CSR_PTR_TYPE> tile_csr_ptr;
    std::vector<ColT> tile_csr_col;
    std::vector<MAT_VAL_TYPE> tile_csr_value;
    std::vector<MaskT> mask;
    std::vector<int> tile_dense_ready;
    std::vector<MAT_VAL_TYPE> dense_data;
};

template <typename ColT, typename MaskT>
struct DeviceFullTileBuild {
    int tile_rows = 0;
    int tile_cols = 0;
    int tile_row_size = 0;
    int tile_col_size = 0;
    int num_tiles = 0;
    int dense_tile_count = 0;
    thrust::device_vector<int> d_ptr;
    thrust::device_vector<int> d_secondary_idx;
    thrust::device_vector<int> d_primary_idx;
    thrust::device_vector<int> d_tile_nnz;
    thrust::device_vector<TILE_CSR_PTR_TYPE> d_tile_csr_ptr;
    thrust::device_vector<ColT> d_tile_csr_col;
    thrust::device_vector<MAT_VAL_TYPE> d_tile_csr_value;
    thrust::device_vector<MaskT> d_mask;
    thrust::device_vector<int> d_tile_dense_ready;
    thrust::device_vector<MAT_VAL_TYPE> d_dense_data;
};

struct Csr2TileStageTimes {
    double h2d_ms = 0.0;
    double row_structure_ms = 0.0;
    double fill_entry_keys_ms = 0.0;
    double sort_by_key_ms = 0.0;
    double reduce_by_key_ms = 0.0;
    double tile_index_ms = 0.0;
    double tile_csr_ptr_ms = 0.0;
    double write_entries_ms = 0.0;
    double dense_flags_ms = 0.0;
    double mask_dense_ms = 0.0;
    double narrow_mask_ms = 0.0;
    double d2d_output_copy_ms = 0.0;
    double build_total_ms = 0.0;
};

using StageClock = std::chrono::steady_clock;

inline StageClock::time_point stage_now() {
    return StageClock::now();
}

inline double stage_elapsed_ms(StageClock::time_point start, bool sync_device) {
    if (sync_device) {
        GPU_CSR2TILE_CHECK(cudaDeviceSynchronize());
    }
    const auto end = StageClock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct EntryToTileKey {
    int tile_area;

    __host__ __device__ Key operator()(Key entry_key) const {
        return entry_key / static_cast<Key>(tile_area);
    }
};

inline int ceil_div_int(int x, int y) {
    return (x + y - 1) / y;
}

template <typename T>
inline T *malloc_copy_exact(const std::vector<T> &v) {
    if (v.empty()) {
        return nullptr;
    }
    T *dst = static_cast<T *>(std::malloc(v.size() * sizeof(T)));
    if (!dst) {
        throw std::runtime_error("failed to allocate CSR2Tile host output");
    }
    std::memcpy(dst, v.data(), v.size() * sizeof(T));
    return dst;
}

template <typename T>
inline T *cuda_malloc_copy_device(const thrust::device_vector<T> &v) {
    if (v.empty()) {
        return nullptr;
    }
    T *dst = nullptr;
    GPU_CSR2TILE_CHECK(cudaMalloc(reinterpret_cast<void **>(&dst), v.size() * sizeof(T)));
    GPU_CSR2TILE_CHECK(cudaMemcpy(
        dst,
        thrust::raw_pointer_cast(v.data()),
        v.size() * sizeof(T),
        cudaMemcpyDeviceToDevice));
    return dst;
}

inline MAT_PTR_TYPE *malloc_copy_ptr(const std::vector<int> &v) {
    if (v.empty()) {
        return nullptr;
    }
    MAT_PTR_TYPE *dst = static_cast<MAT_PTR_TYPE *>(std::malloc(v.size() * sizeof(MAT_PTR_TYPE)));
    if (!dst) {
        throw std::runtime_error("failed to allocate CSR2Tile ptr output");
    }
    for (size_t i = 0; i < v.size(); ++i) {
        dst[i] = static_cast<MAT_PTR_TYPE>(v[i]);
    }
    return dst;
}

inline DeviceCsr copy_to_device(int rows,
                                int cols,
                                int nnz,
                                const MAT_PTR_TYPE *rowptr,
                                const int *colidx,
                                const MAT_VAL_TYPE *values) {
    DeviceCsr csr;
    csr.rows = rows;
    csr.cols = cols;
    csr.nnz = nnz;
    csr.rowptr.assign(rowptr, rowptr + rows + 1);
    csr.colidx.assign(colidx, colidx + nnz);
    csr.values.assign(values, values + nnz);
    return csr;
}

__global__ void fill_tile_keys_kernel(int rows,
                                      const MAT_PTR_TYPE *rowptr,
                                      const int *colidx,
                                      int tile_row_size,
                                      int tile_col_size,
                                      int tile_rows,
                                      int tile_cols,
                                      bool column_major_order,
                                      Key *tile_keys) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }
    const int tile_row = row / tile_row_size;
    for (int p = rowptr[row]; p < rowptr[row + 1]; ++p) {
        const int tile_col = colidx[p] / tile_col_size;
        tile_keys[p] = column_major_order
                           ? static_cast<Key>(tile_col) * static_cast<Key>(tile_rows) + static_cast<Key>(tile_row)
                           : static_cast<Key>(tile_row) * static_cast<Key>(tile_cols) + static_cast<Key>(tile_col);
    }
}

__global__ void fill_entry_keys_kernel(int rows,
                                       const MAT_PTR_TYPE *rowptr,
                                       const int *colidx,
                                       const MAT_VAL_TYPE *values,
                                       int tile_row_size,
                                       int tile_col_size,
                                       int tile_rows,
                                       int tile_cols,
                                       bool column_major_order,
                                       Key *entry_keys,
                                       MAT_VAL_TYPE *entry_values) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= rows) {
        return;
    }
    const int tile_row = row / tile_row_size;
    const int local_row = row - tile_row * tile_row_size;
    const int tile_area = tile_row_size * tile_col_size;
    for (int p = rowptr[row]; p < rowptr[row + 1]; ++p) {
        const int col = colidx[p];
        const int tile_col = col / tile_col_size;
        const int local_col = col - tile_col * tile_col_size;
        const Key tile_key = column_major_order
                                 ? static_cast<Key>(tile_col) * static_cast<Key>(tile_rows) + static_cast<Key>(tile_row)
                                 : static_cast<Key>(tile_row) * static_cast<Key>(tile_cols) + static_cast<Key>(tile_col);
        entry_keys[p] = tile_key * static_cast<Key>(tile_area) +
                        static_cast<Key>(local_row * tile_col_size + local_col);
        entry_values[p] = values[p];
    }
}

__global__ void count_primary_kernel(const Key *keys,
                                     int nkeys,
                                     int secondary_count,
                                     int *ptr_counts) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nkeys) {
        return;
    }
    const int primary = static_cast<int>(keys[idx] / static_cast<Key>(secondary_count));
    atomicAdd(ptr_counts + primary + 1, 1);
}

__global__ void keys_to_indices_kernel(const Key *keys,
                                       int nkeys,
                                       int secondary_count,
                                       int *primary_idx,
                                       int *secondary_idx) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nkeys) {
        return;
    }
    primary_idx[idx] = static_cast<int>(keys[idx] / static_cast<Key>(secondary_count));
    secondary_idx[idx] = static_cast<int>(keys[idx] % static_cast<Key>(secondary_count));
}

__global__ void build_tile_csr_ptr_from_entries_kernel(int num_tiles,
                                                       const int *tile_nnz_prefix,
                                                       const Key *entry_keys,
                                                       int tile_area,
                                                       int tile_col_size,
                                                       int tile_row_size,
                                                       TILE_CSR_PTR_TYPE *tile_csr_ptr) {
    const int tile_id = static_cast<int>(blockIdx.x);
    if (tile_id >= num_tiles) {
        return;
    }
    extern __shared__ int s_row_counts[];
    for (int row = static_cast<int>(threadIdx.x); row < tile_row_size; row += static_cast<int>(blockDim.x)) {
        s_row_counts[row] = 0;
    }
    __syncthreads();

    const int begin = tile_nnz_prefix[tile_id];
    const int end = tile_nnz_prefix[tile_id + 1];
    for (int p = begin + static_cast<int>(threadIdx.x); p < end; p += static_cast<int>(blockDim.x)) {
        const int local_offset = static_cast<int>(entry_keys[p] % static_cast<Key>(tile_area));
        const int local_row = local_offset / tile_col_size;
        atomicAdd(s_row_counts + local_row, 1);
    }
    __syncthreads();

    const int lane = static_cast<int>(threadIdx.x & 31);
    int sum = lane < tile_row_size ? s_row_counts[lane] : 0;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, sum, offset);
        if (lane >= offset) {
            sum += other;
        }
    }
    if (lane < tile_row_size) {
        const int count = s_row_counts[lane];
        tile_csr_ptr[tile_id * tile_row_size + lane] = static_cast<TILE_CSR_PTR_TYPE>(sum - count);
    }
}

__global__ void mark_dense_tiles_kernel(int num_tiles,
                                        const int *tile_counts,
                                        int dense_threshold,
                                        int *dense_flags) {
    const int tile_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tile_id >= num_tiles) {
        return;
    }
    dense_flags[tile_id] = tile_counts[tile_id] >= dense_threshold ? 1 : 0;
}

__global__ void build_dense_ready_kernel(int num_tiles,
                                         const int *dense_flags,
                                         const int *dense_offsets,
                                         int tile_area,
                                         int *tile_dense_ready) {
    const int tile_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tile_id >= num_tiles) {
        return;
    }
    tile_dense_ready[tile_id] = dense_flags[tile_id] ? dense_offsets[tile_id] * tile_area : -1;
}

template <typename ColT, bool StoreFlatLocalOffset>
__global__ void write_tile_entries_kernel(int nnz,
                                          const Key *entry_keys,
                                          const MAT_VAL_TYPE *entry_values,
                                          int tile_area,
                                          int tile_col_size,
                                          ColT *tile_csr_col,
                                          MAT_VAL_TYPE *tile_csr_value) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= nnz) {
        return;
    }
    const int local_offset = static_cast<int>(entry_keys[idx] % static_cast<Key>(tile_area));
    const int local_col = local_offset % tile_col_size;
    tile_csr_col[idx] = static_cast<ColT>(StoreFlatLocalOffset ? local_offset : local_col);
    tile_csr_value[idx] = entry_values[idx];
}

template <typename MaskT>
__global__ void build_mask_and_dense_by_tile_kernel(int num_tiles,
                                                    const int *tile_nnz_prefix,
                                                    const Key *entry_keys,
                                                    const MAT_VAL_TYPE *entry_values,
                                                    const int *tile_dense_ready,
                                                    int tile_area,
                                                    int tile_col_size,
                                                    int tile_row_size,
                                                    int mask_words_per_row,
                                                    int mask_bits,
                                                    unsigned int *mask_work,
                                                    MAT_VAL_TYPE *dense_data) {
    const int tile_id = static_cast<int>(blockIdx.x);
    if (tile_id >= num_tiles) {
        return;
    }
    extern __shared__ unsigned int s_mask[];
    const int mask_words_per_tile = tile_row_size * mask_words_per_row;
    for (int i = static_cast<int>(threadIdx.x); i < mask_words_per_tile; i += static_cast<int>(blockDim.x)) {
        s_mask[i] = 0;
    }
    __syncthreads();

    const int begin = tile_nnz_prefix[tile_id];
    const int end = tile_nnz_prefix[tile_id + 1];
    const int dense_offset = tile_dense_ready[tile_id];
    for (int p = begin + static_cast<int>(threadIdx.x); p < end; p += static_cast<int>(blockDim.x)) {
        const int local_offset = static_cast<int>(entry_keys[p] % static_cast<Key>(tile_area));
        const int local_row = local_offset / tile_col_size;
        const int local_col = local_offset - local_row * tile_col_size;
        const int stride = local_col / mask_bits;
        const unsigned int bit = 1u << (mask_bits - local_col % mask_bits - 1);
        atomicOr(s_mask + local_row * mask_words_per_row + stride, bit);

        if (dense_offset >= 0) {
            dense_data[dense_offset + local_offset] = entry_values[p];
        }
    }
    __syncthreads();

    const int mask_base = tile_id * mask_words_per_tile;
    for (int i = static_cast<int>(threadIdx.x); i < mask_words_per_tile; i += static_cast<int>(blockDim.x)) {
        mask_work[mask_base + i] = s_mask[i];
    }
}

template <typename MaskT>
__global__ void narrow_mask_kernel(int n, const unsigned int *mask_work, MaskT *mask) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) {
        return;
    }
    mask[idx] = static_cast<MaskT>(mask_work[idx]);
}

inline TileStructure build_tile_structure(const DeviceCsr &csr,
                                          int tile_row_size,
                                          int tile_col_size,
                                          bool column_major_order) {
    TileStructure out;
    const int tile_rows = ceil_div_int(csr.rows, tile_row_size);
    const int tile_cols = ceil_div_int(csr.cols, tile_col_size);
    out.primary_count = column_major_order ? tile_cols : tile_rows;
    out.secondary_count = column_major_order ? tile_rows : tile_cols;
    out.ptr.assign(static_cast<size_t>(out.primary_count + 1), 0);

    if (csr.nnz == 0) {
        return out;
    }

    const int threads = kThreadsPerBlock;
    const int row_blocks = ceil_div_int(csr.rows, threads);
    thrust::device_vector<Key> tile_keys(static_cast<size_t>(csr.nnz));
    fill_tile_keys_kernel<<<row_blocks, threads>>>(
        csr.rows,
        thrust::raw_pointer_cast(csr.rowptr.data()),
        thrust::raw_pointer_cast(csr.colidx.data()),
        tile_row_size,
        tile_col_size,
        tile_rows,
        tile_cols,
        column_major_order,
        thrust::raw_pointer_cast(tile_keys.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());

    thrust::sort(thrust::device, tile_keys.begin(), tile_keys.end());
    thrust::device_vector<Key> unique_keys(tile_keys.size());
    thrust::device_vector<int> counts(tile_keys.size());
    auto ends = thrust::reduce_by_key(
        thrust::device,
        tile_keys.begin(),
        tile_keys.end(),
        thrust::make_constant_iterator(1),
        unique_keys.begin(),
        counts.begin());
    out.num_tiles = static_cast<int>(ends.first - unique_keys.begin());
    unique_keys.resize(static_cast<size_t>(out.num_tiles));

    thrust::device_vector<int> d_ptr(static_cast<size_t>(out.primary_count + 1), 0);
    thrust::device_vector<int> d_primary(static_cast<size_t>(out.num_tiles));
    thrust::device_vector<int> d_secondary(static_cast<size_t>(out.num_tiles));
    const int key_blocks = ceil_div_int(out.num_tiles, threads);
    count_primary_kernel<<<key_blocks, threads>>>(
        thrust::raw_pointer_cast(unique_keys.data()),
        out.num_tiles,
        out.secondary_count,
        thrust::raw_pointer_cast(d_ptr.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    thrust::inclusive_scan(thrust::device, d_ptr.begin(), d_ptr.end(), d_ptr.begin());
    keys_to_indices_kernel<<<key_blocks, threads>>>(
        thrust::raw_pointer_cast(unique_keys.data()),
        out.num_tiles,
        out.secondary_count,
        thrust::raw_pointer_cast(d_primary.data()),
        thrust::raw_pointer_cast(d_secondary.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());

    out.ptr.assign(d_ptr.begin(), d_ptr.end());
    out.primary_idx.assign(d_primary.begin(), d_primary.end());
    out.secondary_idx.assign(d_secondary.begin(), d_secondary.end());
    return out;
}

inline DeviceTileStructure build_tile_structure_device(const DeviceCsr &csr,
                                                       int tile_row_size,
                                                       int tile_col_size,
                                                       bool column_major_order) {
    DeviceTileStructure out;
    const int tile_rows = ceil_div_int(csr.rows, tile_row_size);
    const int tile_cols = ceil_div_int(csr.cols, tile_col_size);
    out.primary_count = column_major_order ? tile_cols : tile_rows;
    out.secondary_count = column_major_order ? tile_rows : tile_cols;
    out.d_ptr.assign(static_cast<size_t>(out.primary_count + 1), 0);

    if (csr.nnz == 0) {
        return out;
    }

    const int threads = kThreadsPerBlock;
    const int row_blocks = ceil_div_int(csr.rows, threads);
    thrust::device_vector<Key> tile_keys(static_cast<size_t>(csr.nnz));
    fill_tile_keys_kernel<<<row_blocks, threads>>>(
        csr.rows,
        thrust::raw_pointer_cast(csr.rowptr.data()),
        thrust::raw_pointer_cast(csr.colidx.data()),
        tile_row_size,
        tile_col_size,
        tile_rows,
        tile_cols,
        column_major_order,
        thrust::raw_pointer_cast(tile_keys.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());

    thrust::sort(thrust::device, tile_keys.begin(), tile_keys.end());
    thrust::device_vector<Key> unique_keys(tile_keys.size());
    thrust::device_vector<int> counts(tile_keys.size());
    auto ends = thrust::reduce_by_key(
        thrust::device,
        tile_keys.begin(),
        tile_keys.end(),
        thrust::make_constant_iterator(1),
        unique_keys.begin(),
        counts.begin());
    out.num_tiles = static_cast<int>(ends.first - unique_keys.begin());
    unique_keys.resize(static_cast<size_t>(out.num_tiles));

    out.d_primary_idx.resize(static_cast<size_t>(out.num_tiles));
    out.d_secondary_idx.resize(static_cast<size_t>(out.num_tiles));
    const int key_blocks = ceil_div_int(out.num_tiles, threads);
    count_primary_kernel<<<key_blocks, threads>>>(
        thrust::raw_pointer_cast(unique_keys.data()),
        out.num_tiles,
        out.secondary_count,
        thrust::raw_pointer_cast(out.d_ptr.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    thrust::inclusive_scan(thrust::device, out.d_ptr.begin(), out.d_ptr.end(), out.d_ptr.begin());
    keys_to_indices_kernel<<<key_blocks, threads>>>(
        thrust::raw_pointer_cast(unique_keys.data()),
        out.num_tiles,
        out.secondary_count,
        thrust::raw_pointer_cast(out.d_primary_idx.data()),
        thrust::raw_pointer_cast(out.d_secondary_idx.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    return out;
}

template <typename ColT, typename MaskT, bool StoreFlatLocalOffset>
inline DeviceFullTileBuild<ColT, MaskT> build_full_tile_device(const DeviceCsr &csr,
                                                               int tile_row_size,
                                                               int tile_col_size,
                                                               bool column_major_order,
                                                               Csr2TileStageTimes *stage_times = nullptr) {
    const auto build_start = stage_now();
    DeviceFullTileBuild<ColT, MaskT> out;
    out.tile_rows = ceil_div_int(csr.rows, tile_row_size);
    out.tile_cols = ceil_div_int(csr.cols, tile_col_size);
    out.tile_row_size = tile_row_size;
    out.tile_col_size = tile_col_size;
    const int primary_count = column_major_order ? out.tile_cols : out.tile_rows;
    const int secondary_count = column_major_order ? out.tile_rows : out.tile_cols;
    out.d_ptr.assign(static_cast<size_t>(primary_count + 1), 0);

    if (csr.nnz == 0) {
        out.d_tile_nnz.assign(1, 0);
        return out;
    }

    const int tile_area = tile_row_size * tile_col_size;
    const int mask_bits = static_cast<int>(sizeof(MaskT) * 8);
    const int mask_words_per_row = tile_col_size / mask_bits;
    const int threads = kThreadsPerBlock;
    const int row_blocks = ceil_div_int(csr.rows, threads);

    thrust::device_vector<Key> entry_keys(static_cast<size_t>(csr.nnz));
    thrust::device_vector<MAT_VAL_TYPE> entry_values(static_cast<size_t>(csr.nnz));
    auto stage_start = stage_now();
    fill_entry_keys_kernel<<<row_blocks, threads>>>(
        csr.rows,
        thrust::raw_pointer_cast(csr.rowptr.data()),
        thrust::raw_pointer_cast(csr.colidx.data()),
        thrust::raw_pointer_cast(csr.values.data()),
        tile_row_size,
        tile_col_size,
        out.tile_rows,
        out.tile_cols,
        column_major_order,
        thrust::raw_pointer_cast(entry_keys.data()),
        thrust::raw_pointer_cast(entry_values.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    if (stage_times) {
        stage_times->fill_entry_keys_ms += stage_elapsed_ms(stage_start, true);
    }

    stage_start = stage_now();
    thrust::sort_by_key(thrust::device, entry_keys.begin(), entry_keys.end(), entry_values.begin());
    if (stage_times) {
        stage_times->sort_by_key_ms += stage_elapsed_ms(stage_start, false);
    }

    auto tile_key_begin = thrust::make_transform_iterator(entry_keys.begin(), EntryToTileKey{tile_area});
    auto tile_key_end = thrust::make_transform_iterator(entry_keys.end(), EntryToTileKey{tile_area});

    thrust::device_vector<Key> unique_keys(entry_keys.size());
    thrust::device_vector<int> counts(entry_keys.size());
    stage_start = stage_now();
    auto ends = thrust::reduce_by_key(
        thrust::device,
        tile_key_begin,
        tile_key_end,
        thrust::make_constant_iterator(1),
        unique_keys.begin(),
        counts.begin());
    if (stage_times) {
        stage_times->reduce_by_key_ms += stage_elapsed_ms(stage_start, false);
    }
    out.num_tiles = static_cast<int>(ends.first - unique_keys.begin());
    unique_keys.resize(static_cast<size_t>(out.num_tiles));
    counts.resize(static_cast<size_t>(out.num_tiles));

    out.d_primary_idx.resize(static_cast<size_t>(out.num_tiles));
    out.d_secondary_idx.resize(static_cast<size_t>(out.num_tiles));
    out.d_tile_nnz.assign(static_cast<size_t>(out.num_tiles + 1), 0);
    out.d_tile_csr_ptr.resize(static_cast<size_t>(out.num_tiles) * tile_row_size);
    out.d_tile_csr_col.resize(static_cast<size_t>(csr.nnz));
    out.d_tile_csr_value.resize(static_cast<size_t>(csr.nnz));
    thrust::device_vector<int> d_dense_flags(static_cast<size_t>(out.num_tiles));
    thrust::device_vector<int> d_dense_offsets(static_cast<size_t>(out.num_tiles));
    out.d_tile_dense_ready.resize(static_cast<size_t>(out.num_tiles));

    const int key_blocks = ceil_div_int(out.num_tiles, threads);
    stage_start = stage_now();
    count_primary_kernel<<<key_blocks, threads>>>(
        thrust::raw_pointer_cast(unique_keys.data()),
        out.num_tiles,
        secondary_count,
        thrust::raw_pointer_cast(out.d_ptr.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    thrust::inclusive_scan(thrust::device, out.d_ptr.begin(), out.d_ptr.end(), out.d_ptr.begin());
    keys_to_indices_kernel<<<key_blocks, threads>>>(
        thrust::raw_pointer_cast(unique_keys.data()),
        out.num_tiles,
        secondary_count,
        thrust::raw_pointer_cast(out.d_primary_idx.data()),
        thrust::raw_pointer_cast(out.d_secondary_idx.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    thrust::inclusive_scan(thrust::device, counts.begin(), counts.end(), out.d_tile_nnz.begin() + 1);
    if (stage_times) {
        stage_times->tile_index_ms += stage_elapsed_ms(stage_start, false);
    }

    stage_start = stage_now();
    build_tile_csr_ptr_from_entries_kernel<<<out.num_tiles, kTileThreadsPerBlock, tile_row_size * sizeof(int)>>>(
        out.num_tiles,
        thrust::raw_pointer_cast(out.d_tile_nnz.data()),
        thrust::raw_pointer_cast(entry_keys.data()),
        tile_area,
        tile_col_size,
        tile_row_size,
        thrust::raw_pointer_cast(out.d_tile_csr_ptr.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    if (stage_times) {
        stage_times->tile_csr_ptr_ms += stage_elapsed_ms(stage_start, true);
    }

    const int nnz_blocks = ceil_div_int(csr.nnz, threads);
    stage_start = stage_now();
    write_tile_entries_kernel<ColT, StoreFlatLocalOffset><<<nnz_blocks, threads>>>(
        csr.nnz,
        thrust::raw_pointer_cast(entry_keys.data()),
        thrust::raw_pointer_cast(entry_values.data()),
        tile_area,
        tile_col_size,
        thrust::raw_pointer_cast(out.d_tile_csr_col.data()),
        thrust::raw_pointer_cast(out.d_tile_csr_value.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    if (stage_times) {
        stage_times->write_entries_ms += stage_elapsed_ms(stage_start, true);
    }

    const int dense_threshold = static_cast<int>((static_cast<float>(TILE_DENSE_THRESHOLD) / 10.0f) *
                                                 static_cast<float>(tile_area));
    stage_start = stage_now();
    mark_dense_tiles_kernel<<<key_blocks, threads>>>(
        out.num_tiles,
        thrust::raw_pointer_cast(counts.data()),
        dense_threshold,
        thrust::raw_pointer_cast(d_dense_flags.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    thrust::exclusive_scan(thrust::device, d_dense_flags.begin(), d_dense_flags.end(), d_dense_offsets.begin());
    out.dense_tile_count = thrust::reduce(thrust::device, d_dense_flags.begin(), d_dense_flags.end(), 0);
    build_dense_ready_kernel<<<key_blocks, threads>>>(
        out.num_tiles,
        thrust::raw_pointer_cast(d_dense_flags.data()),
        thrust::raw_pointer_cast(d_dense_offsets.data()),
        tile_area,
        thrust::raw_pointer_cast(out.d_tile_dense_ready.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    if (stage_times) {
        stage_times->dense_flags_ms += stage_elapsed_ms(stage_start, true);
    }

    const int mask_len = out.num_tiles * tile_row_size * mask_words_per_row;
    const int mask_words_per_tile = tile_row_size * mask_words_per_row;
    thrust::device_vector<unsigned int> d_mask_work(static_cast<size_t>(mask_len), 0);
    out.d_mask.resize(static_cast<size_t>(mask_len));
    out.d_dense_data.assign(static_cast<size_t>(out.dense_tile_count) * tile_area, 0);

    stage_start = stage_now();
    build_mask_and_dense_by_tile_kernel<MaskT><<<out.num_tiles, kTileThreadsPerBlock, mask_words_per_tile * sizeof(unsigned int)>>>(
        out.num_tiles,
        thrust::raw_pointer_cast(out.d_tile_nnz.data()),
        thrust::raw_pointer_cast(entry_keys.data()),
        thrust::raw_pointer_cast(entry_values.data()),
        thrust::raw_pointer_cast(out.d_tile_dense_ready.data()),
        tile_area,
        tile_col_size,
        tile_row_size,
        mask_words_per_row,
        mask_bits,
        thrust::raw_pointer_cast(d_mask_work.data()),
        thrust::raw_pointer_cast(out.d_dense_data.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    if (stage_times) {
        stage_times->mask_dense_ms += stage_elapsed_ms(stage_start, true);
    }

    if (mask_len > 0) {
        const int mask_blocks = ceil_div_int(mask_len, threads);
        stage_start = stage_now();
        narrow_mask_kernel<MaskT><<<mask_blocks, threads>>>(
            mask_len,
            thrust::raw_pointer_cast(d_mask_work.data()),
            thrust::raw_pointer_cast(out.d_mask.data()));
        GPU_CSR2TILE_CHECK(cudaGetLastError());
        if (stage_times) {
            stage_times->narrow_mask_ms += stage_elapsed_ms(stage_start, true);
        }
    }

    if (stage_times) {
        stage_times->build_total_ms += stage_elapsed_ms(build_start, false);
    }
    return out;
}

template <typename ColT, typename MaskT, bool StoreFlatLocalOffset>
inline FullTileBuild<ColT, MaskT> build_full_tile(const DeviceCsr &csr,
                                                  int tile_row_size,
                                                  int tile_col_size,
                                                  bool column_major_order) {
    FullTileBuild<ColT, MaskT> out;
    out.tile_rows = ceil_div_int(csr.rows, tile_row_size);
    out.tile_cols = ceil_div_int(csr.cols, tile_col_size);
    out.tile_row_size = tile_row_size;
    out.tile_col_size = tile_col_size;
    const int primary_count = column_major_order ? out.tile_cols : out.tile_rows;
    const int secondary_count = column_major_order ? out.tile_rows : out.tile_cols;
    out.ptr.assign(static_cast<size_t>(primary_count + 1), 0);

    if (csr.nnz == 0) {
        out.tile_nnz.assign(1, 0);
        return out;
    }

    const int tile_area = tile_row_size * tile_col_size;
    const int mask_bits = static_cast<int>(sizeof(MaskT) * 8);
    const int mask_words_per_row = tile_col_size / mask_bits;
    const int threads = kThreadsPerBlock;
    const int row_blocks = ceil_div_int(csr.rows, threads);

    thrust::device_vector<Key> entry_keys(static_cast<size_t>(csr.nnz));
    thrust::device_vector<MAT_VAL_TYPE> entry_values(static_cast<size_t>(csr.nnz));
    fill_entry_keys_kernel<<<row_blocks, threads>>>(
        csr.rows,
        thrust::raw_pointer_cast(csr.rowptr.data()),
        thrust::raw_pointer_cast(csr.colidx.data()),
        thrust::raw_pointer_cast(csr.values.data()),
        tile_row_size,
        tile_col_size,
        out.tile_rows,
        out.tile_cols,
        column_major_order,
        thrust::raw_pointer_cast(entry_keys.data()),
        thrust::raw_pointer_cast(entry_values.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());

    thrust::sort_by_key(thrust::device, entry_keys.begin(), entry_keys.end(), entry_values.begin());

    auto tile_key_begin = thrust::make_transform_iterator(entry_keys.begin(), EntryToTileKey{tile_area});
    auto tile_key_end = thrust::make_transform_iterator(entry_keys.end(), EntryToTileKey{tile_area});

    thrust::device_vector<Key> unique_keys(entry_keys.size());
    thrust::device_vector<int> counts(entry_keys.size());
    auto ends = thrust::reduce_by_key(
        thrust::device,
        tile_key_begin,
        tile_key_end,
        thrust::make_constant_iterator(1),
        unique_keys.begin(),
        counts.begin());
    out.num_tiles = static_cast<int>(ends.first - unique_keys.begin());
    unique_keys.resize(static_cast<size_t>(out.num_tiles));
    counts.resize(static_cast<size_t>(out.num_tiles));

    thrust::device_vector<int> d_ptr(static_cast<size_t>(primary_count + 1), 0);
    thrust::device_vector<int> d_primary(static_cast<size_t>(out.num_tiles));
    thrust::device_vector<int> d_secondary(static_cast<size_t>(out.num_tiles));
    thrust::device_vector<int> d_tile_nnz(static_cast<size_t>(out.num_tiles + 1), 0);
    thrust::device_vector<TILE_CSR_PTR_TYPE> d_tile_csr_ptr(static_cast<size_t>(out.num_tiles) * tile_row_size);
    thrust::device_vector<ColT> d_tile_csr_col(static_cast<size_t>(csr.nnz));
    thrust::device_vector<MAT_VAL_TYPE> d_tile_csr_value(static_cast<size_t>(csr.nnz));
    thrust::device_vector<int> d_dense_flags(static_cast<size_t>(out.num_tiles));
    thrust::device_vector<int> d_dense_offsets(static_cast<size_t>(out.num_tiles));
    thrust::device_vector<int> d_tile_dense_ready(static_cast<size_t>(out.num_tiles));

    const int key_blocks = ceil_div_int(out.num_tiles, threads);
    count_primary_kernel<<<key_blocks, threads>>>(
        thrust::raw_pointer_cast(unique_keys.data()),
        out.num_tiles,
        secondary_count,
        thrust::raw_pointer_cast(d_ptr.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    thrust::inclusive_scan(thrust::device, d_ptr.begin(), d_ptr.end(), d_ptr.begin());
    keys_to_indices_kernel<<<key_blocks, threads>>>(
        thrust::raw_pointer_cast(unique_keys.data()),
        out.num_tiles,
        secondary_count,
        thrust::raw_pointer_cast(d_primary.data()),
        thrust::raw_pointer_cast(d_secondary.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    thrust::inclusive_scan(thrust::device, counts.begin(), counts.end(), d_tile_nnz.begin() + 1);

    build_tile_csr_ptr_from_entries_kernel<<<out.num_tiles, kTileThreadsPerBlock, tile_row_size * sizeof(int)>>>(
        out.num_tiles,
        thrust::raw_pointer_cast(d_tile_nnz.data()),
        thrust::raw_pointer_cast(entry_keys.data()),
        tile_area,
        tile_col_size,
        tile_row_size,
        thrust::raw_pointer_cast(d_tile_csr_ptr.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());

    const int nnz_blocks = ceil_div_int(csr.nnz, threads);
    write_tile_entries_kernel<ColT, StoreFlatLocalOffset><<<nnz_blocks, threads>>>(
        csr.nnz,
        thrust::raw_pointer_cast(entry_keys.data()),
        thrust::raw_pointer_cast(entry_values.data()),
        tile_area,
        tile_col_size,
        thrust::raw_pointer_cast(d_tile_csr_col.data()),
        thrust::raw_pointer_cast(d_tile_csr_value.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());

    const int dense_threshold = static_cast<int>((static_cast<float>(TILE_DENSE_THRESHOLD) / 10.0f) *
                                                 static_cast<float>(tile_area));
    mark_dense_tiles_kernel<<<key_blocks, threads>>>(
        out.num_tiles,
        thrust::raw_pointer_cast(counts.data()),
        dense_threshold,
        thrust::raw_pointer_cast(d_dense_flags.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());
    thrust::exclusive_scan(thrust::device, d_dense_flags.begin(), d_dense_flags.end(), d_dense_offsets.begin());
    out.dense_tile_count = thrust::reduce(thrust::device, d_dense_flags.begin(), d_dense_flags.end(), 0);
    build_dense_ready_kernel<<<key_blocks, threads>>>(
        out.num_tiles,
        thrust::raw_pointer_cast(d_dense_flags.data()),
        thrust::raw_pointer_cast(d_dense_offsets.data()),
        tile_area,
        thrust::raw_pointer_cast(d_tile_dense_ready.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());

    const int mask_len = out.num_tiles * tile_row_size * mask_words_per_row;
    const int mask_words_per_tile = tile_row_size * mask_words_per_row;
    thrust::device_vector<unsigned int> d_mask_work(static_cast<size_t>(mask_len), 0);
    thrust::device_vector<MaskT> d_mask(static_cast<size_t>(mask_len));
    thrust::device_vector<MAT_VAL_TYPE> d_dense_data(static_cast<size_t>(out.dense_tile_count) * tile_area, 0);

    build_mask_and_dense_by_tile_kernel<MaskT><<<out.num_tiles, kTileThreadsPerBlock, mask_words_per_tile * sizeof(unsigned int)>>>(
        out.num_tiles,
        thrust::raw_pointer_cast(d_tile_nnz.data()),
        thrust::raw_pointer_cast(entry_keys.data()),
        thrust::raw_pointer_cast(entry_values.data()),
        thrust::raw_pointer_cast(d_tile_dense_ready.data()),
        tile_area,
        tile_col_size,
        tile_row_size,
        mask_words_per_row,
        mask_bits,
        thrust::raw_pointer_cast(d_mask_work.data()),
        thrust::raw_pointer_cast(d_dense_data.data()));
    GPU_CSR2TILE_CHECK(cudaGetLastError());

    if (mask_len > 0) {
        const int mask_blocks = ceil_div_int(mask_len, threads);
        narrow_mask_kernel<MaskT><<<mask_blocks, threads>>>(
            mask_len,
            thrust::raw_pointer_cast(d_mask_work.data()),
            thrust::raw_pointer_cast(d_mask.data()));
        GPU_CSR2TILE_CHECK(cudaGetLastError());
    }

    out.ptr.assign(d_ptr.begin(), d_ptr.end());
    out.primary_idx.assign(d_primary.begin(), d_primary.end());
    out.secondary_idx.assign(d_secondary.begin(), d_secondary.end());
    out.tile_nnz.assign(d_tile_nnz.begin(), d_tile_nnz.end());
    out.tile_csr_ptr.assign(d_tile_csr_ptr.begin(), d_tile_csr_ptr.end());
    out.tile_csr_col.assign(d_tile_csr_col.begin(), d_tile_csr_col.end());
    out.tile_csr_value.assign(d_tile_csr_value.begin(), d_tile_csr_value.end());
    out.mask.assign(d_mask.begin(), d_mask.end());
    out.tile_dense_ready.assign(d_tile_dense_ready.begin(), d_tile_dense_ready.end());
    out.dense_data.assign(d_dense_data.begin(), d_dense_data.end());
    return out;
}

inline void gpu_csr2tile_row_major(SMatrixA *matrix, int tile_size_m, int tile_size_n, double *copy_ms = nullptr) {
    const auto copy_start = std::chrono::steady_clock::now();
    DeviceCsr csr = copy_to_device(
        matrix->m, matrix->n, matrix->nnz, matrix->rowpointer, matrix->columnindex, matrix->value);
    if (copy_ms) {
        const auto copy_end = std::chrono::steady_clock::now();
        *copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    }
    FullTileBuild<TILE_CSR_COL_TYPE_A, TILE_MASK_TYPE_A> build =
        build_full_tile<TILE_CSR_COL_TYPE_A, TILE_MASK_TYPE_A, true>(
            csr, tile_size_m, tile_size_n, false);

    matrix->tilem = build.tile_rows;
    matrix->tilen = build.tile_cols;
    matrix->numtile = build.num_tiles;
    matrix->dense_tile_count = build.dense_tile_count;
    matrix->tile_ptr = malloc_copy_ptr(build.ptr);
    matrix->tile_columnidx = malloc_copy_exact(build.secondary_idx);
    matrix->tile_rowidx = malloc_copy_exact(build.primary_idx);
    matrix->tile_nnz = malloc_copy_exact(build.tile_nnz);
    matrix->tile_csr_Ptr = malloc_copy_exact(build.tile_csr_ptr);
    matrix->tile_csr_Col = malloc_copy_exact(build.tile_csr_col);
    matrix->tile_csr_Value = malloc_copy_exact(build.tile_csr_value);
    matrix->mask = malloc_copy_exact(build.mask);
    matrix->tile_dense_ready = malloc_copy_exact(build.tile_dense_ready);
    matrix->dense_data = build.dense_tile_count > 0 ? malloc_copy_exact(build.dense_data) : nullptr;
    matrix->csc_tile_ptr = nullptr;
    matrix->csc_tile_rowidx = nullptr;
}

inline void gpu_csr2tile_row_major_device(SMatrixA *matrix,
                                          int tile_size_m,
                                          int tile_size_n,
                                          double *copy_ms = nullptr,
                                          Csr2TileStageTimes *stage_times = nullptr) {
    if (stage_times) {
        *stage_times = Csr2TileStageTimes{};
    }
    const auto copy_start = std::chrono::steady_clock::now();
    DeviceCsr csr = copy_to_device(
        matrix->m, matrix->n, matrix->nnz, matrix->rowpointer, matrix->columnindex, matrix->value);
    const auto copy_end = std::chrono::steady_clock::now();
    const double h2d_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    if (copy_ms) {
        *copy_ms = h2d_ms;
    }
    if (stage_times) {
        stage_times->h2d_ms = h2d_ms;
    }
    DeviceFullTileBuild<TILE_CSR_COL_TYPE_A, TILE_MASK_TYPE_A> build =
        build_full_tile_device<TILE_CSR_COL_TYPE_A, TILE_MASK_TYPE_A, true>(
            csr, tile_size_m, tile_size_n, false, stage_times);

    matrix->tilem = build.tile_rows;
    matrix->tilen = build.tile_cols;
    matrix->numtile = build.num_tiles;
    matrix->dense_tile_count = build.dense_tile_count;
    matrix->tile_ptr = nullptr;
    matrix->tile_columnidx = nullptr;
    matrix->tile_rowidx = nullptr;
    matrix->tile_nnz = nullptr;
    matrix->tile_csr_Ptr = nullptr;
    matrix->tile_csr_Col = nullptr;
    matrix->tile_csr_Value = nullptr;
    matrix->mask = nullptr;
    matrix->tile_dense_ready = nullptr;
    matrix->dense_data = nullptr;
    matrix->csc_tile_ptr = nullptr;
    matrix->csc_tile_rowidx = nullptr;

    const auto d2d_start = stage_now();
    matrix->d_tile_ptr = cuda_malloc_copy_device(build.d_ptr);
    matrix->d_tile_columnidx = cuda_malloc_copy_device(build.d_secondary_idx);
    matrix->d_tile_nnz = cuda_malloc_copy_device(build.d_tile_nnz);
    matrix->d_tile_csr_Ptr = cuda_malloc_copy_device(build.d_tile_csr_ptr);
    matrix->d_tile_csr_Col = cuda_malloc_copy_device(build.d_tile_csr_col);
    matrix->d_tile_csr_Value = cuda_malloc_copy_device(build.d_tile_csr_value);
    matrix->d_mask = cuda_malloc_copy_device(build.d_mask);
    matrix->d_tile_dense_ready = cuda_malloc_copy_device(build.d_tile_dense_ready);
    matrix->d_dense_data = build.dense_tile_count > 0 ? cuda_malloc_copy_device(build.d_dense_data) : nullptr;
    if (stage_times) {
        stage_times->d2d_output_copy_ms += stage_elapsed_ms(d2d_start, true);
    }
    matrix->device_tile_ready = 1;
}

inline void gpu_csr2tile_col_major(SMatrixB *matrix, int tile_size_m, int tile_size_n, double *copy_ms = nullptr) {
    const int tile_row_size = tile_size_n;
    const int tile_col_size = tile_size_m;
    const auto copy_start = std::chrono::steady_clock::now();
    DeviceCsr csr = copy_to_device(
        matrix->m, matrix->n, matrix->nnz, matrix->rowpointer, matrix->columnindex, matrix->value);
    if (copy_ms) {
        const auto copy_end = std::chrono::steady_clock::now();
        *copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    }

    TileStructure row_structure = build_tile_structure(csr, tile_row_size, tile_col_size, false);
    FullTileBuild<TILE_CSR_COL_TYPE_B, TILE_MASK_TYPE_B> csc_build =
        build_full_tile<TILE_CSR_COL_TYPE_B, TILE_MASK_TYPE_B, false>(
            csr, tile_row_size, tile_col_size, true);

    matrix->tilem = ceil_div_int(matrix->m, tile_row_size);
    matrix->tilen = ceil_div_int(matrix->n, tile_col_size);
    matrix->numtile = csc_build.num_tiles;
    matrix->dense_tile_count = csc_build.dense_tile_count;
    matrix->tile_ptr = malloc_copy_ptr(row_structure.ptr);
    matrix->tile_columnidx = malloc_copy_exact(row_structure.secondary_idx);
    matrix->tile_rowidx = malloc_copy_exact(row_structure.primary_idx);
    matrix->csc_tile_ptr = malloc_copy_exact(csc_build.ptr);
    matrix->csc_tile_rowidx = malloc_copy_exact(csc_build.secondary_idx);
    matrix->tile_nnz = malloc_copy_exact(csc_build.tile_nnz);
    matrix->tile_csr_Ptr = malloc_copy_exact(csc_build.tile_csr_ptr);
    matrix->tile_csr_Col = malloc_copy_exact(csc_build.tile_csr_col);
    matrix->tile_csr_Value = malloc_copy_exact(csc_build.tile_csr_value);
    matrix->mask = malloc_copy_exact(csc_build.mask);
    matrix->tile_dense_ready = malloc_copy_exact(csc_build.tile_dense_ready);
    if (csc_build.dense_tile_count > 0) {
        matrix->dense_data = malloc_copy_exact(csc_build.dense_data);
    } else {
        const int tile_area = tile_size_n * tile_size_m;
        matrix->dense_data = static_cast<MAT_VAL_TYPE *>(std::calloc(static_cast<size_t>(tile_area), sizeof(MAT_VAL_TYPE)));
        if (!matrix->dense_data) {
            throw std::runtime_error("failed to allocate B dense_data placeholder");
        }
    }
}

inline void gpu_csr2tile_col_major_device(SMatrixB *matrix,
                                          int tile_size_m,
                                          int tile_size_n,
                                          double *copy_ms = nullptr,
                                          Csr2TileStageTimes *stage_times = nullptr) {
    if (stage_times) {
        *stage_times = Csr2TileStageTimes{};
    }
    const int tile_row_size = tile_size_n;
    const int tile_col_size = tile_size_m;
    const auto copy_start = std::chrono::steady_clock::now();
    DeviceCsr csr = copy_to_device(
        matrix->m, matrix->n, matrix->nnz, matrix->rowpointer, matrix->columnindex, matrix->value);
    const auto copy_end = std::chrono::steady_clock::now();
    const double h2d_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    if (copy_ms) {
        *copy_ms = h2d_ms;
    }
    if (stage_times) {
        stage_times->h2d_ms = h2d_ms;
    }

    const auto row_structure_start = stage_now();
    DeviceTileStructure row_structure = build_tile_structure_device(csr, tile_row_size, tile_col_size, false);
    if (stage_times) {
        stage_times->row_structure_ms += stage_elapsed_ms(row_structure_start, false);
    }
    DeviceFullTileBuild<TILE_CSR_COL_TYPE_B, TILE_MASK_TYPE_B> csc_build =
        build_full_tile_device<TILE_CSR_COL_TYPE_B, TILE_MASK_TYPE_B, false>(
            csr, tile_row_size, tile_col_size, true, stage_times);

    matrix->tilem = ceil_div_int(matrix->m, tile_row_size);
    matrix->tilen = ceil_div_int(matrix->n, tile_col_size);
    matrix->numtile = csc_build.num_tiles;
    matrix->dense_tile_count = csc_build.dense_tile_count;
    matrix->tile_ptr = nullptr;
    matrix->tile_columnidx = nullptr;
    matrix->tile_rowidx = nullptr;
    matrix->csc_tile_ptr = nullptr;
    matrix->csc_tile_rowidx = nullptr;
    matrix->tile_nnz = nullptr;
    matrix->tile_csr_Ptr = nullptr;
    matrix->tile_csr_Col = nullptr;
    matrix->tile_csr_Value = nullptr;
    matrix->mask = nullptr;
    matrix->tile_dense_ready = nullptr;
    matrix->dense_data = nullptr;

    const auto d2d_start = stage_now();
    matrix->d_tile_ptr = cuda_malloc_copy_device(row_structure.d_ptr);
    matrix->d_tile_columnidx = cuda_malloc_copy_device(row_structure.d_secondary_idx);
    matrix->d_csc_tile_ptr = cuda_malloc_copy_device(csc_build.d_ptr);
    matrix->d_csc_tile_rowidx = cuda_malloc_copy_device(csc_build.d_secondary_idx);
    matrix->d_tile_nnz = cuda_malloc_copy_device(csc_build.d_tile_nnz);
    matrix->d_tile_csr_Ptr = cuda_malloc_copy_device(csc_build.d_tile_csr_ptr);
    matrix->d_tile_csr_Col = cuda_malloc_copy_device(csc_build.d_tile_csr_col);
    matrix->d_tile_csr_Value = cuda_malloc_copy_device(csc_build.d_tile_csr_value);
    matrix->d_mask = cuda_malloc_copy_device(csc_build.d_mask);
    matrix->d_tile_dense_ready = cuda_malloc_copy_device(csc_build.d_tile_dense_ready);
    matrix->d_dense_data = csc_build.dense_tile_count > 0 ? cuda_malloc_copy_device(csc_build.d_dense_data) : nullptr;
    if (stage_times) {
        stage_times->d2d_output_copy_ms += stage_elapsed_ms(d2d_start, true);
    }
    matrix->device_tile_ready = 1;
}

}  // namespace gpu_csr2tile

#undef GPU_CSR2TILE_CHECK
