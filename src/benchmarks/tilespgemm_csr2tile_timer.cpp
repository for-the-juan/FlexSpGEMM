#include "probe_common.hpp"

#include "../common.h"

typedef SMatrixA SMatrix;
typedef TILE_CSR_COL_TYPE_A TILE_CSR_COL_TYPE;
typedef TILE_MASK_TYPE_A TILE_MASK_TYPE;

#define step1_kernel tilespgemm_step1_kernel
#define step2_kernel tilespgemm_step2_kernel
#define step3_kernel tilespgemm_step3_kernel
#define csr2tile_row_major tilespgemm_csr2tile_row_major
#define csr2tile_col_major tilespgemm_csr2tile_col_major
#define matrix_destroy tilespgemm_matrix_destroy
#include "../csr2tile.bak.h"
#undef matrix_destroy
#undef csr2tile_col_major
#undef csr2tile_row_major
#undef step3_kernel
#undef step2_kernel
#undef step1_kernel

namespace {

void usage(const char *prog) {
    std::fprintf(stderr, "Usage: %s [-d <ignored_device_id>] -aat <0|1> -tau <ignored_ratio> <matrix.mtx>\n", prog);
}

double timeval_ms(const timeval &a, const timeval &b) {
    return (b.tv_sec - a.tv_sec) * 1000.0 + (b.tv_usec - a.tv_usec) / 1000.0;
}

void fill_smatrix(const bench::CsrMatrix &src, SMatrix *dst) {
    std::memset(dst, 0, sizeof(SMatrix));
    dst->m = src.rows;
    dst->n = src.cols;
    dst->nnz = static_cast<int>(src.nnz);
    dst->isSymmetric = src.symmetric ? 1 : 0;
    dst->rowpointer = static_cast<MAT_PTR_TYPE *>(std::malloc((src.rows + 1) * sizeof(MAT_PTR_TYPE)));
    dst->columnindex = static_cast<int *>(std::malloc(src.colidx.size() * sizeof(int)));
    dst->value = static_cast<MAT_VAL_TYPE *>(std::malloc(src.colidx.size() * sizeof(MAT_VAL_TYPE)));
    if (!dst->rowpointer || !dst->columnindex || !dst->value) {
        throw std::runtime_error("failed to allocate CSR arrays");
    }
    for (int i = 0; i <= src.rows; ++i) {
        dst->rowpointer[i] = static_cast<MAT_PTR_TYPE>(src.rowptr[static_cast<size_t>(i)]);
    }
    for (size_t i = 0; i < src.colidx.size(); ++i) {
        dst->columnindex[i] = src.colidx[i];
        dst->value[i] = static_cast<MAT_VAL_TYPE>(i % 10);
    }
}

void free_smatrix(SMatrix *matrix) {
    if (matrix->tile_ptr) {
        tilespgemm_matrix_destroy(matrix);
    }
    std::free(matrix->tile_rowidx);
    std::free(matrix->rowpointer);
    std::free(matrix->columnindex);
    std::free(matrix->value);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 6) {
        usage(argv[0]);
        return 1;
    }

    int argi = 1;
    int ignored_device_id = -1;
    if (std::strcmp(argv[argi], "-d") == 0) {
        if (argi + 1 >= argc) {
            usage(argv[0]);
            return 1;
        }
        ignored_device_id = std::atoi(argv[++argi]);
        ++argi;
    }

    if (std::strcmp(argv[argi], "-aat") != 0 || argi + 1 >= argc) {
        usage(argv[0]);
        return 1;
    }
    const int aat = std::atoi(argv[++argi]);
    ++argi;

    if (std::strcmp(argv[argi], "-tau") != 0 || argi + 1 >= argc) {
        usage(argv[0]);
        return 1;
    }
    const double ignored_tau = std::atof(argv[++argi]);
    ++argi;

    if (argi >= argc) {
        usage(argv[0]);
        return 1;
    }
    const std::string filename = argv[argi];

    try {
        std::printf("================================================================================\n");
        std::printf("  TileSpGEMM CSR2Tile Standalone Timer\n");
        std::printf("================================================================================\n\n");
        std::printf("[Execution]\n");
        std::printf("  Backend      : CPU/OpenMP\n");
        std::printf("  Source       : src/csr2tile.bak.h (TileSpGEMM-style CPU path)\n");
        if (ignored_device_id >= 0) {
            std::printf("  Ignored -d   : %d\n", ignored_device_id);
        }
        std::printf("\n--------------------------------------------------------------------------------\n");
        std::printf("  aat          : %d\n", aat);
        std::printf("  Ignored tau  : %.3f\n", ignored_tau);
        std::printf("\n--------------------------------------------------------------------------------\n");

        bench::Timer load_timer;
        bench::CsrMatrix loaded = bench::load_matrix_market(filename);
        const double load_ms = load_timer.elapsed_ms();

        if (!aat && loaded.rows != loaded.cols) {
            std::fprintf(stderr, "[ERROR] Matrix squaring requires rowA == colA. Exiting.\n");
            return 1;
        }

        SMatrix matrixA;
        fill_smatrix(loaded, &matrixA);

        std::printf("\n[Input Matrix]\n");
        std::printf("  File        : %s\n", bench::basename(filename).c_str());
        std::printf("  Path        : %s\n", filename.c_str());
        std::printf("  Dimension   : %d x %d\n", matrixA.m, matrixA.n);
        std::printf("  NNZ (A)     : %d\n", matrixA.nnz);
        std::printf("  Load Time   : %.5f sec\n", load_ms / 1000.0);
        std::printf("\n--------------------------------------------------------------------------------\n");

        std::printf("\n[Tiling Configuration]\n");
        std::printf("  Tile Size   : %d x %d  (TILE_SIZE_M x TILE_SIZE_N)\n", TILE_SIZE_M, TILE_SIZE_N);
        std::printf("\n--------------------------------------------------------------------------------\n");

        std::printf("\n[Preprocessing]\n");
        timeval t1;
        timeval t2;
        gettimeofday(&t1, NULL);
        tilespgemm_csr2tile_row_major(&matrixA, TILE_SIZE_M, TILE_SIZE_N);
        gettimeofday(&t2, NULL);

        std::printf("  TileSpGEMM Format Conversion : %.2f ms\n", timeval_ms(t1, t2));
        std::printf("  Mode                         : TileSpGEMM CSR2Tile-only standalone executable.\n");
        std::printf("  OpenMP Max Threads           : %d\n", omp_get_max_threads());
        std::printf("  Tiles (A)                    : %d\n", matrixA.numtile);
        std::printf("\n--------------------------------------------------------------------------------\n");

        free_smatrix(&matrixA);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
