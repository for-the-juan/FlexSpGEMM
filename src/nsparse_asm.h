#ifndef _SPGEMM_CUDA_NSPARSE_ASM_
#define _SPGEMM_CUDA_NSPARSE_ASM_
//#define FLOAT 1

#define real double

/*
 * Inline PTX
 */
__device__ __inline__ real ld_gbl_val(const real *val)
{
    real return_value;

// #ifndef HALF
//        asm("ld.global.cv.f16 %0, [%1];" : "=h"(return_value) : "l"(val));
#ifdef FLOAT
    asm("ld.global.cv.f32 %0, [%1];" : "=f"(return_value) : "l"(val));
#else
    asm("ld.global.cv.f64 %0, [%1];" : "=d"(return_value) : "l"(val));
#endif
  
    return return_value;
}

__device__ __inline__ float2 ld_gbl_float2(const float2 *val)
{
    float2 return_value;

    asm("ld.global.cv.v2.f32 {%0, %1}, [%2];" : "=f"(return_value.x), "=f"(return_value.y) : "l"(val));
    return return_value;
}

__device__ __inline__ float4 ld_gbl_float4(const float4 *val)
{
    float4 return_value;

    asm("ld.global.cv.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(return_value.x), "=f"(return_value.y), "=f"(return_value.z), "=f"(return_value.w) : "l"(val));
    return return_value;
}

__device__ __inline__ short ld_gbl_row(const short *row)
{
    short return_value;
    asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(row));
    return return_value;
}

__device__ __inline__ int ld_gbl_col(const int *col)
{
    int return_value;
    asm("ld.global.cv.s32 %0, [%1];" : "=r"(return_value) : "l"(col));
    return return_value;
}

__device__ __inline__ short ld_gbl_short(const short *col)
{
    short return_value;
    asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(col));
    return return_value;
}

__device__ __inline__ unsigned short ld_gbl_ushort(const unsigned short *col)
{
    unsigned short return_value;
    asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(col));
    return return_value;
}

template <typename T>
__device__ __inline__ T ld_gbl_auto(const T *col)
{
    T return_value;
    
    if constexpr (std::is_same_v<T, uint8_t>) {
        short return_value_;
        asm("ld.global.cv.u8 %0, [%1];" : "=h"(return_value_) : "l"(col));
        return_value = (uint8_t)return_value_;
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
        asm("ld.global.cv.u16 %0, [%1];" : "=h"(return_value) : "l"(col));
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
        asm("ld.global.cv.u32 %0, [%1];" : "=r"(return_value) : "l"(col));
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
        asm("ld.global.cv.s16 %0, [%1];" : "=h"(return_value) : "l"(col));
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
        asm("ld.global.cv.s32 %0, [%1];" : "=r"(return_value) : "l"(col));
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
        asm("ld.global.cv.s64 %0, [%1];" : "=l"(return_value) : "l"(col));
    }
    else if constexpr (std::is_same_v<T, float>) {
        asm("ld.global.cv.f32 %0, [%1];" : "=f"(return_value) : "l"(col));
    }
    else if constexpr (std::is_same_v<T, double>) {
        asm("ld.global.cv.f64 %0, [%1];" : "=d"(return_value) : "l"(col));
    }
    else {
        static_assert(sizeof(T) == 0, "Unsupported type for ld_gbl_cv");
    }

    return return_value;
}

template <typename T>
__forceinline__ __device__ int binary_search_exact_auto_kernel(const T *__restrict__ d_array, int l, int r, T key)
{
    while (l <= r)
    {
        int m = l + (r - l) / 2;
        T elem = d_array[m];
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

    // if we reach here, then element was
    // not present
    return -1;
}

__device__ __inline__ unsigned char ld_gbl_uchar(const unsigned char *row)
{
    short return_value;
    asm("ld.global.cv.u8 %0, [%1];" : "=h"(return_value) : "l"(row));
    return (unsigned char)return_value;
}

__device__ __inline__ void st_gbl_ans(const real *ans_gpu, real answer)
{

// #ifndef HALF
//   asm("ld.global.cv.fp16 %0, [%1];" : "=h"(return_value) : "l"(val));

#ifdef FLOAT
    asm("st.global.cs.f32 [%0], %1;" :: "l"(ans_gpu) , "f"(answer));
#else
    asm("st.global.cs.f64 [%0], %1;" :: "l"(ans_gpu) , "d"(answer));
#endif

}

__device__ __inline__ real ld_gbl_real(const real *val) {

  real return_value;

// #ifndef HALF
//   asm("ld.global.cv.fp16 %0, [%1];" : "=h"(return_value) : "l"(val));

#ifdef FLOAT
  asm("ld.global.cv.f32 %0, [%1];" : "=f"(return_value) : "l"(val));
#else
  asm("ld.global.cv.f64 %0, [%1];" : "=d"(return_value) : "l"(val));
#endif
  
  return return_value;
}

__device__ __inline__ int ld_gbl_int32(const int *col) {
  int return_value;
  asm("ld.global.cv.s32 %0, [%1];" : "=r"(return_value) : "l"(col));
  return return_value;
}

__device__ __inline__ void atomic_fadd(real *adr, real val)
{
#if __CUDA_ARCH__ >= 600
    atomicAdd(adr, val);
#else
#ifdef FLOAT
    atomicAdd(adr, val);
#elif defined DOUBLE
    unsigned long long int *address_ull = (unsigned long long int *)(adr);
    unsigned long long int old_val = *address_ull;
    unsigned long long int assumed;
    real input = val;
    do {
        assumed = old_val;
        old_val = atomicCAS(address_ull, assumed, __double_as_longlong(input + __longlong_as_double(assumed)));
    } while (assumed != old_val);
#endif
#endif
}

#endif
