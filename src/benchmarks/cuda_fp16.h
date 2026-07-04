#pragma once

// CPU-only shim for src/common.h. The CSR2Tile timer does not use CUDA half
// types, but common.h includes cuda_fp16.h unconditionally.
