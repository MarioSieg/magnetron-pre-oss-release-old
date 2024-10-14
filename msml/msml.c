/*
** (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
** MSML - Single file STB-style machine learning library in C99 with Python bindings.
** For license see LICENSE file.
*/

#define MSML_EXPORT_DLL
#include "msml.h"

#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <float.h>

#ifdef _MSC_VER
#   include <intrin.h>
#endif

#ifdef __aarch64__
#   include <arm_neon.h>
#   include <arm_acle.h>
#elif defined(__x86_64__) || defined(_M_X64)
#   include <nmmintrin.h>
#   include <wmmintrin.h>
#   ifndef _MSC_VER
#       include <cpuid.h>
#   endif
#endif

#ifdef _WIN32
#   error "MSML does not support Windows yet."
#elif defined(__APPLE__)
#   include <mach/mach.h>
#   include <mach/vm_statistics.h>
#   include <sys/sysctl.h>
#   include <sys/types.h>
#   include <unistd.h>
#else
#   include <unistd.h>
#endif

#include <ctype.h>
#include <time.h>
#include <errno.h>

msml_static_assert(sizeof(0u) == 4);
msml_static_assert(sizeof(0ull) == 8);

#define MSML_MAX_CPUS 8192
#define MSML_MAX_NUMA_NODES 64

#ifdef MSML_ENABLE_IMAGE_SUPPORT
#   define STBI_MALLOC(sz) msml_alloc(NULL, (sz))
#   define STBI_FREE(ptr) msml_alloc((ptr), 0)
#   define STBI_REALLOC(ptr, sz) msml_alloc((ptr), (sz))
#   define STBIR_MALLOC(sz, usr) msml_alloc(NULL, (sz))
#   define STBIR_FREE(ptr, usr) msml_alloc((ptr), 0)
#   define STBIR_REALLOC(ptr, sz, usr) msml_alloc((ptr), (sz))
#   define STBIW_MALLOC(sz) msml_alloc(NULL, (sz))
#   define STBIW_FREE(ptr) msml_alloc((ptr), 0)
#   define STBIW_REALLOC(ptr, sz) msml_alloc((ptr), (sz))
#   define STB_IMAGE_IMPLEMENTATION
#   include <stb_image.h>
#   define STB_IMAGE_RESIZE_IMPLEMENTATION
#   include <stb_image_resize2.h>
#   define STB_IMAGE_WRITE_IMPLEMENTATION
#   include <stb_image_write.h>
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#	define MSML_NORET __attribute__((noreturn))
#	define MSML_ALIGN(x) __attribute__((aligned(x)))
#	define MSML_AINLINE inline __attribute__((always_inline))
#	define MSML_NOINLINE __attribute__((noinline))
#   define MSML_HOTPROC __attribute__((hot))
#   define MSML_COLDPROC __attribute__((cold))
#   define MSML_PACKED __attribute__((packed))
#   define MSML_FALLTHROUGH __attribute__((fallthrough))
#   define MSML_UNUSED __attribute__((unused))
#	define msml_likely(x) __builtin_expect(!!(x), 1)
#	define msml_unlikely(x) __builtin_expect(!!(x), 0)
#   define msml_ffs(x) ((uint32_t)__builtin_ctz(x))
#   define msml_fls(x) ((uint32_t)(__builtin_clz(x)^31))
#   define msml_ffs64(x) ((uint32_t)__builtin_ctzll(x))
#   define msml_fls64(x) ((uint32_t)(__builtin_clzll(x)^63))
#else
    unsigned char _BitScanForward64(unsigned long*, unsigned __int64);
    unsigned char _BitScanReverse64(unsigned long*, unsigned __int64);
#   pragma intrinsic(_BitScanForward64)
#   pragma intrinsic(_BitScanReverse64)
#	define MSML_NORET __declspec(noreturn)
#	define MSML_ALIGN(x) __declspec(align(x))
#	define MSML_AINLINE inline __forceinline
#	define MSML_NOINLINE __declspec(noinline)
#   define MSML_HOTPROC
#   define MSML_COLDPROC
#   define MSML_PACKED __declspec(align(1))
#   define MSML_FALLTHROUGH
#   define MSML_UNUSED
#	define msml_likely(x) (x)
#	define msml_unlikely(x) (x)
    static __forceinline uint32_t msml_ffs(const uint32_t x) {
        unsigned long r; _BitScanForward(&r, x); return (uint32_t)r;
    }
    static __forceinline uint32_t msml_fls(const uint32_t x) {
        unsigned long r; _BitScanReverse(&r, x); return (uint32_t)r;
    }
    static __forceinline uint32_t msml_ffs64(const uint64_t x) {
      unsigned long r; _BitScanForward64(&r, x); return (uint32_t)r;
    }
    static __forceinline uint32_t msml_fls64(const uint64_t x) {
      unsigned long r; _BitScanReverse64(&r, x); return (uint32_t)r;
    }
#endif

#define msml_swap(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)
#define msml_max(x, y) (((x) > (y)) ? (x) : (y))
#define msml_min(x, y) (((x) < (y)) ? (x) : (y))
#define MSML_CCRED "\x1b[31m"
#define MSML_CCGREEN "\x1b[32m"
#define MSML_CCYELLOW "\x1b[33m"
#define MSML_CCBLUE "\x1b[34m"
#define MSML_CCMAGENTA "\x1b[35m"
#define MSML_CCCYAN "\x1b[36m"
#define MSML_CCRESET "\x1b[0m"
#define MSML_STRINGIZE2(x) #x
#define MSML_STRINGIZE(x) MSML_STRINGIZE2(x)
#ifdef __FILE_NAME__
#   define MSML_SRC_NAME __FILE_NAME__ ":" MSML_STRINGIZE(__LINE__)
#else
#   define MSML_SRC_NAME __FILE__ ":" MSML_STRINGIZE(__LINE__)
#endif
#define msml_log_info(msg, ...) fprintf(stdout,  "[MSML] " MSML_SRC_NAME " " msg "\n", ## __VA_ARGS__)
#define msml_log_warn(msg, ...) fprintf(stderr,  "[MSML] " MSML_SRC_NAME " " MSML_CCYELLOW msg MSML_CCRESET "\n", ## __VA_ARGS__)
#define msml_log_error(msg, ...) fprintf(stderr,  "[MSML] " MSML_SRC_NAME " " MSML_CCRED msg MSML_CCRESET "\n", ## __VA_ARGS__)

#if defined(__x86_64__) || defined(_M_X64)

    #define MSML__X86_64_CPUID_0H 0
    #define MSML__X86_64_CPUID_1H 1
    #define MSML__X86_64_CPUID_2H 2
    #define MSML__X86_64_CPUID_7H 3
    #define MSML__X86_64_CPUID_80000001H 4
    #define MSML__X86_64_CPUID_80000007H 5
    #define MSML__X86_64_CPUID_16H 6
    #define MSML__X86_64_CPUID_7H_1H 7

    #define MSML__X86_64_CPUID_EAX 0
    #define MSML__X86_64_CPUID_EBX 1
    #define MSML__X86_64_CPUID_ECX 2
    #define MSML__X86_64_CPUID_EDX 3

    #define msml_x86_64_feature_def(_, __) /* Enumerator | CPUDID Leaf | Register | Bit Index */\
        _(AVX                  ,    1H,        ECX,     28)__\
        _(AVX2                 ,    7H,        EBX,      5)__\
        _(AVXVNNI              ,    7H_1H,     EAX,      4)__\
        _(AVXVNNIINT8          ,    7H_1H,     EDX,      4)__\
        _(AVXVNNIINT16         ,    7H_1H,     EDX,     10)__\
        _(AVX512BW             ,    7H,        EBX,     30)__\
        _(AVX512CD             ,    7H,        EBX,     28)__\
        _(AVX512DQ             ,    7H,        EBX,     17)__\
        _(AVX512ER             ,    7H,        EBX,     27)__\
        _(AVX512F              ,    7H,        EBX,     16)__\
        _(AVX512IFMA           ,    7H,        EBX,     21)__\
        _(AVX512PF             ,    7H,        EBX,     26)__\
        _(AVX512VBMI           ,    7H,        ECX,      1)__\
        _(AVX512VL             ,    7H,        EBX,     31)__\
        _(AVX512_4FMAPS        ,    7H,        EDX,      3)__\
        _(AVX512_4VNNIW        ,    7H,        EDX,      2)__\
        _(AVX512_FP16          ,    7H,        EDX,     23)__\
        _(AVX512_BF16          ,    7H_1H,     EAX,      5)__\
        _(AVX512_BITALG        ,    7H,        ECX,     12)__\
        _(AVX512_VBMI2         ,    7H,        ECX,      6)__\
        _(AVX512_VNNI          ,    7H,        ECX,     11)__\
        _(AVX512_VP2INTERSECT  ,    7H,        EDX,      8)__\
        _(AVX512_VPOPCNTDQ     ,    7H,        ECX,     14)__\
        _(BMI                  ,    7H,        EBX,      3)__\
        _(BMI2                 ,    7H,        EBX,      8)__\
        _(F16C                 ,    1H,        ECX,     29)__\
        _(FMA                  ,    1H,        ECX,     12)__\
        _(FPU                  ,    1H,        EDX,      0)__\
        _(GFNI                 ,    7H,        ECX,      8)__\
        _(IA64                 ,    1H,        EDX,     30)__\
        _(MMX                  ,    1H,        EDX,     23)__\
        _(OSXSAVE              ,    1H,        ECX,     27)__\
        _(PCLMUL               ,    1H,        ECX,      1)__\
        _(RDRND                ,    1H,        ECX,     30)__\
        _(RDSEED               ,    7H,        EBX,     18)__\
        _(RDTSCP               ,    80000001H, EDX,     27)__\
        _(SHA                  ,    7H,        EBX,     29)__\
        _(SSE                  ,    1H,        EDX,     25)__\
        _(SSE2                 ,    1H,        EDX,     26)__\
        _(SSE3                 ,    1H,        ECX,      0)__\
        _(SSE4_1               ,    1H,        ECX,     19)__\
        _(SSE4_2               ,    1H,        ECX,     20)__\
        _(SSSE3                ,    1H,        ECX,      9)__\
        _(VAES                 ,    7H,        ECX,      9)__\
        _(VME                  ,    1H,        EDX,      1)__\
        _(VMX                  ,    1H,        ECX,      5)__\
        _(VPCLMULQDQ           ,    7H,        ECX,     10)__\
        _(XSAVE                ,    1H,        ECX,     26)__\
        _(HYBRID_CPU           ,    7H,        EDX,     15)__

    #define _(enumerator, leaf, reg, bit) MSML__X86_64_FEATURE_##enumerator
    typedef enum msml__x86_64_feature_t {
        msml_x86_64_feature_def(_, MSML_SEP)
        MSML__X86_64_FEATURE__COUNT
    } msml__x86_64_feature_t;
    #undef _
    #define _(enumerator, leaf, reg, bit) #enumerator
    static const char* const msml__x86_64_feature_names[MSML__X86_64_FEATURE__COUNT] = {
        msml_x86_64_feature_def(_, MSML_SEP)
    };
    #undef _
    #define _(enumerator, leaf, reg, bit) (0xff&MSML__X86_64_CPUID_##leaf)
    static const uint8_t msml__x86_64_feature_leaves[MSML__X86_64_FEATURE__COUNT] = {
        msml_x86_64_feature_def(_, MSML_SEP)
    };
    #undef _
    #define _(enumerator, leaf, reg, bit) (0xff&MSML__X86_64_CPUID_##reg)
    static const uint8_t msml__x86_64_feature_regs[MSML__X86_64_FEATURE__COUNT] = {
        msml_x86_64_feature_def(_, MSML_SEP)
    };
    #undef _
    #define _(enumerator, leaf, reg, bit) (1u<<(bit))
    static const uint32_t msml__x86_64_feature_masks[MSML__X86_64_FEATURE__COUNT] = {
        msml_x86_64_feature_def(_, MSML_SEP)
    };
    #undef _
    #undef msml_x86_64_feature_def

#endif

typedef struct msml__blas_compute_info_t msml__blas_compute_info_t; /* Forward declaration. */

struct msml_ctx_t {
    void* (*alloc_fn)(void* blk, size_t size); /* Memory allocator. */
    struct {
        char os_name[128]; /* OS name. */
        char cpu_name[128]; /* CPU name. */
        uint32_t cpu_virtual_cores; /* Virtual CPUs. */
        uint32_t cpu_physical_cores; /* Physical CPU cores. */
        uint32_t cpu_sockets; /* CPU sockets. */
        uint64_t phys_mem_total; /* Total physical memory in bytes. */
        uint64_t phys_mem_free; /* Free physical memory in bytes. */
        #if defined(__x86_64__) || defined(_M_X64)
            uint32_t x86_64_cpu_features[8][4]; /* x86-64 CPU features. */
        #endif
    } sys;
    struct {
        size_t chunk_size;
        size_t chunk_len;
        size_t chunk_cap;
        uint8_t** chunks;
        uint8_t* delta;
        bool warmup_chunks;
        size_t alloc_acc;
        size_t mapped_total;
        size_t alloc_total;
    } pool;
    union {
        struct {
            uint64_t state;
            uint64_t inc;
        } pcg;
        struct {
            uint32_t remaining;
            uint32_t next;
            uint32_t state[624];
        } mersenne;
    } prng_state;
    msml_prng_algorithm_t prng_algorithm;
    uintptr_t host_thread_id;
    void (*blas_dispatch[MSML_OP__COUNT])(const msml__blas_compute_info_t* bci, msml_tensor_t* r, const msml_tensor_t** inputs); /* BLAS dispatch table. Specialized for host CPU architecture. */
    void* ud; /* User data. */
};

struct msml_tensor_t {
    msml_ctx_t* ctx;
    int64_t rank;
    int64_t shape[MSML_MAX_DIMS];
    int64_t strides[MSML_MAX_DIMS];
    msml_dtype_t dtype;
    void* buf;
    int64_t buf_size;
    msml_op_t op;
    msml_tensor_t* inputs[MSML_MAX_INPUT_TENSORS];
    msml_tensor_t* view;
    size_t view_offs;
    char name[MSML_MAX_TENSOR_NAME_LEN];
    void* ud; /* User data. */
};

static MSML_NORET void msml_panic(const char* msg, ...) {
    fprintf(stderr, "%s", MSML_CCRED);
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
    fprintf(stderr, "%s", MSML_CCRESET);
    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

#define msml_assert(expr, msg, ...) \
    if (msml_unlikely(!(expr))) { \
        msml_panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
    }
#define msml_assert2(expr) msml_assert(expr, "")

void* msml_default_allocator_impl(void* blk, size_t size) {
    if (!size) {
        free(blk);
        return NULL;
    } else if(!blk) {
        blk = malloc(size);
        msml_assert(blk, "Failed to allocate %.03fKiB memory", (double)size/(double)(1<<10));
        return blk;
    } else {
        void* block = realloc(blk, size);
        msml_assert(blk, "Failed to reallocate %.03fKiB memory", (double)size/(double)(1<<10));
        return block;
    }
}

static void msml__humanize_memory_size(size_t n, double* out, const char** unit) {
    if (n < (1<<10)) {
        *out = (double)n;
        *unit = "B";
    } else if (n < (1<<20)) {
        *out = (double)n/(double)(1<<10);
        *unit = "KiB";
    } else if (n < (1<<30)) {
        *out = (double)n/(double)(1<<20);
        *unit = "MiB";
    } else {
        *out = (double)n/(double)(1<<30);
        *unit = "GiB";
    }
}

#ifdef _WIN32
#include <wchar.h>
extern __declspec(dllimport) int __stdcall MultiByteToWideChar(
    unsigned int cp,
    unsigned long flags,
    const char* str,
    int cbmb,
    wchar_t* widestr,
    int cchwide
);
extern __declspec(dllimport) int __stdcall WideCharToMultiByte(
    unsigned int cp,
    unsigned long flags,
    const wchar_t* widestr,
    int cchwide,
    char* str,
    int cbmb,
    const char* defchar,
    int* used_default
);
#endif

static FILE* msml__fopen(const char* file, const char* mode) {
    msml_assert(file && *file && mode && *mode, "Invalid file name or mode");
    FILE* f = NULL;
    #ifdef _WIN32
        wchar_t w_mode[64];
        wchar_t w_file[1024];
        if (MultiByteToWideChar(65001 /* UTF8 */, 0, file, -1, w_file, sizeof(w_file)/sizeof(*w_file)) == 0) return NULL;
        if (MultiByteToWideChar(65001 /* UTF8 */, 0, mode, -1, w_mode, sizeof(w_mode)/sizeof(*w_mode)) == 0) return NULL;
        #if defined(_MSC_VER) && _MSC_VER >= 1400
           if (_wfopen_s(&f, w_file, w_mode) != 0)
               return NULL;
        #else
           f = _wfopen(w_file, w_mode);
        #endif
    #elif defined(_MSC_VER) && _MSC_VER >= 1400
        if (fopen_s(&f, filename, mode) != 0) return NULL;
    #else
        f = fopen(file, mode);
    #endif
    return f;
}

static inline uintptr_t msml__thread_id(void) {
    uintptr_t tid;
    #if defined(_MSC_VER) && defined(_M_X64)
        tid = __readgsqword(48);
    #elif defined(_MSC_VER) && defined(_M_IX86)
        tid = __readfsdword(24);
    #elif defined(_MSC_VER) && defined(_M_ARM64)
        tid = __getReg(18);
    #elif defined(__i386__)
        __asm__ __volatile__("movl %%gs:0, %0" : "=r" (tid));  /* x86-32 WIN32 uses %GS */
    #elif defined(__MACH__) && defined(__x86_64__)
        __asm__ __volatile__("movq %%gs:0, %0" : "=r" (tid));  /* x86.64 OSX uses %GS */
    #elif defined(__x86_64__)
        __asm__ __volatile__("movq %%fs:0, %0" : "=r" (tid));  /* x86-64 Linux and BSD uses %FS */
    #elif defined(__arm__)
        __asm__ __volatile__("mrc p15, 0, %0, c13, c0, 3\nbic %0, %0, #3" : "=r" (tid));
    #elif defined(__aarch64__) && defined(__APPLE__)
        __asm__ __volatile__("mrs %0, tpidrro_el0" : "=r" (tid));
    #elif defined(__aarch64__)
        __asm__ __volatile__("mrs %0, tpidr_el0" : "=r" (tid));
    #elif defined(__powerpc64__)
    #   ifdef __clang__
        tid = (uintptr_t)__builtin_thread_pointer();
    #   else
        register uintptr_t tp __asm__ ("r13");
        __asm__ __volatile__("" : "=r" (tp));
        tid = tp;
    #   endif
    #elif defined(__powerpc__)
    #   ifdef __clang__
            tid = (uintptr_t)__builtin_thread_pointer();
    #   else
        register uintptr_t tp __asm__ ("r2");
        __asm__ __volatile__("" : "=r" (tp));
        tid = tp;
    #   endif
    #elif defined(__s390__) && defined(__GNUC__)
        tid = (uintptr_t)__builtin_thread_pointer();
    #elif defined(__riscv)
    #   ifdef __clang__
            tid = (uintptr_t)__builtin_thread_pointer();
    #   else
            __asm__ ("mv %0, tp" : "=r" (tid));
    #   endif
    #else
    #   error "Unsupported MSML platform"
    #endif
    return tid;
}

static int64_t msml_hpc_clock_us(void) { /* High precision clock in microseconds. */
    #ifdef _WIN32
    #error "MSML does not support Windows yet."
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
    #endif
}
static int64_t msml_hpc_clock_elapsed_us(int64_t start) { /* High precision clock elapsed time in microseconds. */
    return llabs(msml_hpc_clock_us() - start);
}
static double msml_hpc_clock_elapsed_ms(int64_t start) { /* High precision clock elapsed time in milliseconds. */
    return (double)msml_hpc_clock_elapsed_us(start) * 1.0e-3;
}

typedef uint32_t msml_bitset_t;
msml_static_assert(sizeof(msml_bitset_t) == 4);
#define MSML_BITSET_SIZE (sizeof(msml_bitset_t)*8)
#define MSML_BITSET_MASK (MSML_BITSET_SIZE-1)
#define msml_bitset_size(n) (((n)+MSML_BITSET_MASK)>>5)
#define msml_bitset_get(sets, i) (!!(sets[(i)>>5]&(1u<<((i)&MSML_BITSET_MASK))))
#define msml_bitset_set(sets, i) (sets[(i)>>5]|=(1u<<((i)&MSML_BITSET_MASK)))
#define msml_bitset_clear(sets, i) (sets[(i)>>5]&=~(1u<<((i)&MSML_BITSET_MASK)))
#define msml_bitset_toggle(sets, i) (sets[(i)>>5]^=(1u<<((i)&MSML_BITSET_MASK)))

static uint32_t MSML_AINLINE msml__bswap32(uint32_t x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    #   if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
            x = (uint32_t)__builtin_bswap32((int32_t)x);
    #   else
            x = (x & 0xff000000) >> 24 |
            (x & 0xff0000) >> 8 |
            (x & 0xff00) << 8 |
            (x & 0xff) << 24;
    #   endif
    #endif
    return x;
}

static uint64_t MSML_AINLINE msml__bswap64(uint64_t x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    #   if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
            x = (uint64_t)__builtin_bswap64((int64_t)x);
    #   else
            x = (x & 0xff00000000000000) >> 56 |
            (x & 0xff000000000000) >> 40 |
            (x & 0xff0000000000) >> 24 |
            (x & 0xff00000000) >> 8 |
            (x & 0xff000000) << 8 |
            (x & 0xff0000) << 24 |
            (x & 0xff00) << 40 |
            (x & 0xff) << 56;
    #   endif
    #endif
    return x;
}

static void* msml_advance_ptr(void** p, size_t sz, size_t align) {
    void* pp = (void*)((((uintptr_t)*p + align) - 1) & ~((align) - 1));
    *p = (void*)((uint8_t*)pp + sz);
    return pp;
}

#ifdef __aarch64__
    static uint64x2_t MSML_AINLINE msml__clmul_lo_e(uint64x2_t a, uint64x2_t b, uint64x2_t c) {
        uint64x2_t r;
        __asm__ __volatile__(
            "pmull %0.1q, %2.1d, %3.1d\n"
            "eor %0.16b, %0.16b, %1.16b\n"
            : "=w"(r), "+w"(c) : "w"(a), "w"(b)
        );
        return r;
    }
    static uint64x2_t MSML_AINLINE msml__clmul_hi_e(uint64x2_t a, uint64x2_t b, uint64x2_t c) {
        uint64x2_t r;
        __asm__ __volatile__(
            "pmull2 %0.1q, %2.2d, %3.2d\n"
            "eor %0.16b, %0.16b, %1.16b\n"
            : "=w"(r), "+w"(c) : "w"(a), "w"(b)
        );
        return r;
    }
#elif defined(__x86_64__) || defined(_M_X64)
    static uint32_t msml__xnmodp(uint64_t n) { /* x^n mod P, in log(n) time */
        uint64_t stack = ~(uint64_t)1;
        uint32_t acc, low;
        for (; n > 191; n = (n>>1) - 16) stack = (stack<<1) + (n & 1);
        stack = ~stack;
        acc = ((uint32_t)0x80000000) >> (n & 31);
        for (n >>= 5; n; --n) acc = _mm_crc32_u32(acc, 0);
        while ((low = stack & 1), stack >>= 1) {
            __m128i x = _mm_cvtsi32_si128(acc);
            uint64_t y = _mm_cvtsi128_si64(_mm_clmulepi64_si128(x, x, 0));
            acc = _mm_crc32_u64(0, y << low);
        }
        return acc;
    }
    static __m128i MSML_AINLINE msml__clmul_scalar(uint32_t a, uint32_t b) {
        return _mm_clmulepi64_si128(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b), 0);
    }
    static __m128i MSML_AINLINE msml__crc_shift(uint32_t crc, size_t sz) {
        return msml__clmul_scalar(crc, msml__xnmodp((sz<<3) - 33));
    }
#endif

static uint32_t msml__crc32c(const void* buffer, size_t size) { /* Compute CRC32 checksum with CRC32c polynomial. */
    if (msml_unlikely(!buffer || !size)) return 0;
    const uint8_t* buf = (const uint8_t*)buffer;
    #ifdef __aarch64__
        uint32_t crc = ~0;
        for (; size && ((uintptr_t)buf & 7); --size) crc = __crc32cb(crc, *buf++);
        if (((uintptr_t)buf & 8) && size >= 8) {
            crc = __crc32cd(crc, *(const uint64_t*)buf);
            buf += 8;
            size -= 8;
        }
        if (size >= 192) { /* First vector chunk. */
            uint64x2_t x0 = vld1q_u64((const uint64_t*)buf), y0;
            uint64x2_t x1 = vld1q_u64((const uint64_t*)(buf + 16)), y1;
            uint64x2_t x2 = vld1q_u64((const uint64_t*)(buf + 32)), y2;
            uint64x2_t x3 = vld1q_u64((const uint64_t*)(buf + 48)), y3;
            uint64x2_t x4 = vld1q_u64((const uint64_t*)(buf + 64)), y4;
            uint64x2_t x5 = vld1q_u64((const uint64_t*)(buf + 80)), y5;
            uint64x2_t x6 = vld1q_u64((const uint64_t*)(buf + 96)), y6;
            uint64x2_t x7 = vld1q_u64((const uint64_t*)(buf + 112)), y7;
            uint64x2_t x8 = vld1q_u64((const uint64_t*)(buf + 128)), y8;
            uint64x2_t x9 = vld1q_u64((const uint64_t*)(buf + 144)), y9;
            uint64x2_t x10 = vld1q_u64((const uint64_t*)(buf + 160)), y10;
            uint64x2_t x11 = vld1q_u64((const uint64_t*)(buf + 176)), y11;
            uint64x2_t k;
            { static const uint64_t MSML_ALIGN(16) k_[] = {0xa87ab8a8, 0xab7aff2a}; k = vld1q_u64(k_); }
            x0 = veorq_u64((uint64x2_t){crc, 0}, x0);
            buf += 192;
            size -= 192;
            while (size >= 192) { /* Work loop. */
                y0 = msml__clmul_lo_e(x0, k, vld1q_u64((const uint64_t*) buf)), x0 = msml__clmul_hi_e(x0, k, y0);
                y1 = msml__clmul_lo_e(x1, k, vld1q_u64((const uint64_t*) (buf + 16))), x1 = msml__clmul_hi_e(x1, k, y1);
                y2 = msml__clmul_lo_e(x2, k, vld1q_u64((const uint64_t*) (buf + 32))), x2 = msml__clmul_hi_e(x2, k, y2);
                y3 = msml__clmul_lo_e(x3, k, vld1q_u64((const uint64_t*) (buf + 48))), x3 = msml__clmul_hi_e(x3, k, y3);
                y4 = msml__clmul_lo_e(x4, k, vld1q_u64((const uint64_t*) (buf + 64))), x4 = msml__clmul_hi_e(x4, k, y4);
                y5 = msml__clmul_lo_e(x5, k, vld1q_u64((const uint64_t*) (buf + 80))), x5 = msml__clmul_hi_e(x5, k, y5);
                y6 = msml__clmul_lo_e(x6, k, vld1q_u64((const uint64_t*) (buf + 96))), x6 = msml__clmul_hi_e(x6, k, y6);
                y7 = msml__clmul_lo_e(x7, k, vld1q_u64((const uint64_t*) (buf + 112))), x7 = msml__clmul_hi_e(x7, k, y7);
                y8 = msml__clmul_lo_e(x8, k, vld1q_u64((const uint64_t*) (buf + 128))), x8 = msml__clmul_hi_e(x8, k, y8);
                y9 = msml__clmul_lo_e(x9, k, vld1q_u64((const uint64_t*) (buf + 144))), x9 = msml__clmul_hi_e(x9, k, y9);
                y10 = msml__clmul_lo_e(x10, k, vld1q_u64((const uint64_t*) (buf + 160))), x10 = msml__clmul_hi_e(x10, k, y10);
                y11 = msml__clmul_lo_e(x11, k, vld1q_u64((const uint64_t*) (buf + 176))), x11 = msml__clmul_hi_e(x11, k, y11);
                buf += 192;
                size -= 192;
            }
            /* Reduce x0 ... x11 to just x0. */
            { static const uint64_t MSML_ALIGN(16) k_[] = {0xf20c0dfe, 0x493c7d27}; k = vld1q_u64(k_); }
            y0 = msml__clmul_lo_e(x0, k, x1), x0 = msml__clmul_hi_e(x0, k, y0);
            y2 = msml__clmul_lo_e(x2, k, x3), x2 = msml__clmul_hi_e(x2, k, y2);
            y4 = msml__clmul_lo_e(x4, k, x5), x4 = msml__clmul_hi_e(x4, k, y4);
            y6 = msml__clmul_lo_e(x6, k, x7), x6 = msml__clmul_hi_e(x6, k, y6);
            y8 = msml__clmul_lo_e(x8, k, x9), x8 = msml__clmul_hi_e(x8, k, y8);
            y10 = msml__clmul_lo_e(x10, k, x11), x10 = msml__clmul_hi_e(x10, k, y10);
            { static const uint64_t MSML_ALIGN(16) k_[] = {0x3da6d0cb, 0xba4fc28e}; k = vld1q_u64(k_); }
            y0 = msml__clmul_lo_e(x0, k, x2), x0 = msml__clmul_hi_e(x0, k, y0);
            y4 = msml__clmul_lo_e(x4, k, x6), x4 = msml__clmul_hi_e(x4, k, y4);
            y8 = msml__clmul_lo_e(x8, k, x10), x8 = msml__clmul_hi_e(x8, k, y8);
            { static const uint64_t MSML_ALIGN(16) k_[] = {0x740eef02, 0x9e4addf8}; k = vld1q_u64(k_); }
            y0 = msml__clmul_lo_e(x0, k, x4), x0 = msml__clmul_hi_e(x0, k, y0);
            x4 = x8;
            y0 = msml__clmul_lo_e(x0, k, x4), x0 = msml__clmul_hi_e(x0, k, y0);
            /* Reduce 128 bits to 32 bits, and multiply by x^32. */
            crc = __crc32cd(0, vgetq_lane_u64(x0, 0));
            crc = __crc32cd(crc, vgetq_lane_u64(x0, 1));
        }
        for (; size >= 8; buf += 8, size -= 8) crc = __crc32cd(crc, *(const uint64_t*)buf);
        for (; size; --size) crc = __crc32cb(crc, *buf++);
        return ~crc;
    #elif defined(__x86_64__) || defined(_M_X64)
        uint32_t crc = ~0;
        for (; size && ((uintptr_t)buf & 7); --size) crc = _mm_crc32_u8(crc, *buf++);
        if (size >= 32) {
            size_t klen = ((size - 8) / 24)<<3;
            uint32_t crc1 = 0;
            uint32_t crc2 = 0;
            __m128i vc0;
            __m128i vc1;
            uint64_t vc;
            /* Main loop. */
            do {
                crc = _mm_crc32_u64(crc, *(const uint64_t*)buf);
                crc1 = _mm_crc32_u64(crc1, *(const uint64_t*)(buf + klen));
                crc2 = _mm_crc32_u64(crc2, *(const uint64_t*)(buf + (klen<<1)));
                buf += 8;
                size -= 24;
            } while (size >= 32);
            vc0 = msml__crc_shift(crc, (klen<<1) + 8);
            vc1 = msml__crc_shift(crc1, klen + 8);
            vc = _mm_extract_epi64(_mm_xor_si128(vc0, vc1), 0);
            /* Final 8 bytes. */
            buf += klen<<1;
            crc = crc2;
            crc = _mm_crc32_u64(crc, *(const uint64_t*)buf ^ vc), buf += 8;
            size -= 8;
        }
        for (; size >= 8; buf += 8, size -= 8) crc = _mm_crc32_u64(crc, *(const uint64_t*)buf);
        for (; size; --size) crc = _mm_crc32_u8(crc, *buf++);
        return ~crc;
    #else
        static const uint32_t crc_lut[256] = {
            0x00000000, 0xf26b8303, 0xe13b70f7, 0x1350f3f4, 0xc79a971f, 0x35f1141c,
            0x26a1e7e8, 0xd4ca64eb, 0x8ad958cf, 0x78b2dbcc, 0x6be22838, 0x9989ab3b,
            0x4d43cfd0, 0xbf284cd3, 0xac78bf27, 0x5e133c24, 0x105ec76f, 0xe235446c,
            0xf165b798, 0x030e349b, 0xd7c45070, 0x25afd373, 0x36ff2087, 0xc494a384,
            0x9a879fa0, 0x68ec1ca3, 0x7bbcef57, 0x89d76c54, 0x5d1d08bf, 0xaf768bbc,
            0xbc267848, 0x4e4dfb4b, 0x20bd8ede, 0xd2d60ddd, 0xc186fe29, 0x33ed7d2a,
            0xe72719c1, 0x154c9ac2, 0x061c6936, 0xf477ea35, 0xaa64d611, 0x580f5512,
            0x4b5fa6e6, 0xb93425e5, 0x6dfe410e, 0x9f95c20d, 0x8cc531f9, 0x7eaeb2fa,
            0x30e349b1, 0xc288cab2, 0xd1d83946, 0x23b3ba45, 0xf779deae, 0x05125dad,
            0x1642ae59, 0xe4292d5a, 0xba3a117e, 0x4851927d, 0x5b016189, 0xa96ae28a,
            0x7da08661, 0x8fcb0562, 0x9c9bf696, 0x6ef07595, 0x417b1dbc, 0xb3109ebf,
            0xa0406d4b, 0x522bee48, 0x86e18aa3, 0x748a09a0, 0x67dafa54, 0x95b17957,
            0xcba24573, 0x39c9c670, 0x2a993584, 0xd8f2b687, 0x0c38d26c, 0xfe53516f,
            0xed03a29b, 0x1f682198, 0x5125dad3, 0xa34e59d0, 0xb01eaa24, 0x42752927,
            0x96bf4dcc, 0x64d4cecf, 0x77843d3b, 0x85efbe38, 0xdbfc821c, 0x2997011f,
            0x3ac7f2eb, 0xc8ac71e8, 0x1c661503, 0xee0d9600, 0xfd5d65f4, 0x0f36e6f7,
            0x61c69362, 0x93ad1061, 0x80fde395, 0x72966096, 0xa65c047d, 0x5437877e,
            0x4767748a, 0xb50cf789, 0xeb1fcbad, 0x197448ae, 0x0a24bb5a, 0xf84f3859,
            0x2c855cb2, 0xdeeedfb1, 0xcdbe2c45, 0x3fd5af46, 0x7198540d, 0x83f3d70e,
            0x90a324fa, 0x62c8a7f9, 0xb602c312, 0x44694011, 0x5739b3e5, 0xa55230e6,
            0xfb410cc2, 0x092a8fc1, 0x1a7a7c35, 0xe811ff36, 0x3cdb9bdd, 0xceb018de,
            0xdde0eb2a, 0x2f8b6829, 0x82f63b78, 0x709db87b, 0x63cd4b8f, 0x91a6c88c,
            0x456cac67, 0xb7072f64, 0xa457dc90, 0x563c5f93, 0x082f63b7, 0xfa44e0b4,
            0xe9141340, 0x1b7f9043, 0xcfb5f4a8, 0x3dde77ab, 0x2e8e845f, 0xdce5075c,
            0x92a8fc17, 0x60c37f14, 0x73938ce0, 0x81f80fe3, 0x55326b08, 0xa759e80b,
            0xb4091bff, 0x466298fc, 0x1871a4d8, 0xea1a27db, 0xf94ad42f, 0x0b21572c,
            0xdfeb33c7, 0x2d80b0c4, 0x3ed04330, 0xccbbc033, 0xa24bb5a6, 0x502036a5,
            0x4370c551, 0xb11b4652, 0x65d122b9, 0x97baa1ba, 0x84ea524e, 0x7681d14d,
            0x2892ed69, 0xdaf96e6a, 0xc9a99d9e, 0x3bc21e9d, 0xef087a76, 0x1d63f975,
            0x0e330a81, 0xfc588982, 0xb21572c9, 0x407ef1ca, 0x532e023e, 0xa145813d,
            0x758fe5d6, 0x87e466d5, 0x94b49521, 0x66df1622, 0x38cc2a06, 0xcaa7a905,
            0xd9f75af1, 0x2b9cd9f2, 0xff56bd19, 0x0d3d3e1a, 0x1e6dcdee, 0xec064eed,
            0xc38d26c4, 0x31e6a5c7, 0x22b65633, 0xd0ddd530, 0x0417b1db, 0xf67c32d8,
            0xe52cc12c, 0x1747422f, 0x49547e0b, 0xbb3ffd08, 0xa86f0efc, 0x5a048dff,
            0x8ecee914, 0x7ca56a17, 0x6ff599e3, 0x9d9e1ae0, 0xd3d3e1ab, 0x21b862a8,
            0x32e8915c, 0xc083125f, 0x144976b4, 0xe622f5b7, 0xf5720643, 0x07198540,
            0x590ab964, 0xab613a67, 0xb831c993, 0x4a5a4a90, 0x9e902e7b, 0x6cfbad78,
            0x7fab5e8c, 0x8dc0dd8f, 0xe330a81a, 0x115b2b19, 0x020bd8ed, 0xf0605bee,
            0x24aa3f05, 0xd6c1bc06, 0xc5914ff2, 0x37faccf1, 0x69e9f0d5, 0x9b8273d6,
            0x88d28022, 0x7ab90321, 0xae7367ca, 0x5c18e4c9, 0x4f48173d, 0xbd23943e,
            0xf36e6f75, 0x0105ec76, 0x12551f82, 0xe03e9c81, 0x34f4f86a, 0xc69f7b69,
            0xd5cf889d, 0x27a40b9e, 0x79b737ba, 0x8bdcb4b9, 0x988c474d, 0x6ae7c44e,
            0xbe2da0a5, 0x4c4623a6, 0x5f16d052, 0xad7d5351
        };
        uint32_t crc = ~0u;
        for (size_t i=0; i < size; ++i)
            crc = (crc >> 8) ^ crc_lut[buf[i] ^ (crc & 0xff)];
        return ~crc;
    #endif
}

typedef struct msml_hashset_t {
    void* ud;
    size_t len;
    msml_bitset_t* used;
    const msml_tensor_t** keys;
} msml_hashset_t;
#define MSML_HASHSET_FULL ((size_t)-1)
#define MSML_HASHSET_DUPLICATE ((size_t)-2)
#define MSML_HASHSET_MAX ((size_t)-3) /* Must be last. */
#define msml_hashset_hash_fn(ptr) ((size_t)(uintptr_t)(ptr)>>3)

static size_t msml_hashset_compute_hash_size(size_t sz) {
    msml_assert2(sz > 0 && sz < MSML_HASHSET_MAX);
    static const size_t prime_lut[] = {
        2, 3, 5, 11, 17, 37, 67, 131, 257, 521, 1031,
        2053, 4099, 8209, 16411, 32771, 65537, 131101,
        262147, 524309, 1048583, 2097169, 4194319, 8388617,
        16777259, 33554467, 67108879, 134217757, 268435459,
        536870923, 1073741827, 2147483659
    };
    size_t l = 0;
    size_t r = sizeof(prime_lut)/sizeof(*prime_lut);
    while (l < r) { /* Binary search for the smallest prime > sz. */
        size_t mid = (l+r)>>1;
        if (prime_lut[mid] < sz) l = mid+1;
        else r = mid;
    }
    return l < sizeof(prime_lut)/sizeof(*prime_lut) ? prime_lut[l] : sz|1;
}
static msml_hashset_t msml_hashset_create(size_t size) {
    size = msml_hashset_compute_hash_size(size);
    msml_hashset_t set = {
        .ud = NULL,
        .len = size,
        .used = (msml_bitset_t*)msml_alloc(NULL, msml_bitset_size(size)*sizeof(*set.used)),
        .keys = (const msml_tensor_t**)msml_alloc(NULL, size*sizeof(*set.keys)),
    };
    memset(set.used, 0, msml_bitset_size(size)*sizeof(*set.used));
    return set;
}
static size_t msml_hashset_lookup(msml_hashset_t* set, const msml_tensor_t* key) {
    size_t k = msml_hashset_hash_fn(key) % set->len, i = k;
    while (msml_bitset_get(set->used, i) && set->keys[i] != key) { /* Linear probing. */
        i = (i+1) % set->len;
        if (i == k) return MSML_HASHSET_FULL;
    }
    return i;
}
static bool msml_hashset_contains_key(msml_hashset_t* set, const msml_tensor_t* key) {
    size_t i = msml_hashset_lookup(set, key);
    return msml_bitset_get(set->used, i) && i != MSML_HASHSET_FULL;
}
static size_t msml_hashset_insert(msml_hashset_t* set, const msml_tensor_t* key) {
    size_t k = msml_hashset_hash_fn(key) % set->len, i = k;
    do { /* Linear probing. */
        if (!msml_bitset_get(set->used, i)) { /* Insert key. */
            msml_bitset_set(set->used, i);
            set->keys[i] = key;
            return i;
        }
        if (set->keys[i] == key) return MSML_HASHSET_DUPLICATE; /* Key already exists. */
        i = (i+1) % set->len;
    } while (i != k);
    msml_panic("Insertion target not found");
}
static void msml_hashset_reset(msml_hashset_t* set) {
    memset(set->used, 0, msml_bitset_size(set->len)*sizeof(*set->used));
}
static void msml_hashset_destroy(msml_hashset_t* set) {
    msml_alloc(set->used, 0);
    msml_alloc(set->keys, 0);
}

static bool MSML_AINLINE msml__imull64_ov(int64_t a, int64_t b, int64_t* out) { /* Performs c = a*b with overflow checking. Returns true on overflow, else false. */
#ifdef _MSC_VER
    int64_t high;
    int64_t low = _mul128(a, b, &high);
    int64_t sign = low >> 63;
    *out = low;
    return high != sign;
#else
#if __SIZEOF_LONG_LONG__ == 8 && __SIZEOF_LONG__ == 8
    return __builtin_smulll_overflow(a, b, (long long*)out);
#else
    return __builtin_smull_overflow(a, b, out);
#endif
#endif
}

/* Generate n uniform random floats within [min, max]. */
static void msml__prng_generate_n(msml_ctx_t* ctx, float* out_gen, int64_t out_n, float min, float max) {
    float rescale_uniform = max - min;
    switch (ctx->prng_algorithm) {
        case MSML_PRNG_MERSENNE_TWISTER: {
            uint32_t* rem = &ctx->prng_state.mersenne.remaining;
            uint32_t* next = &ctx->prng_state.mersenne.next;
            uint32_t* state = ctx->prng_state.mersenne.state;
            for (int64_t ii=0; ii < out_n; ++ii) {
                if (--*rem <= 0) {
                    *rem = 624;
                    *next = 0;
                    uint32_t y, i;
                    for (i = 0; i < 624-397; ++i) {
                        y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                        state[i] = state[i+397] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                    }
                    for (; i < 624-1; ++i) {
                        y = (state[i] & 0x80000000u) | (state[i+1] & 0x7fffffffu);
                        state[i] = state[i + (397-624)] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                    }
                    y = (state[624-1] & 0x80000000u) | (*state & 0x7fffffffu);
                    state[624-1] = state[397-1] ^ (y>>1) ^ ((y&1) ? 0 : 0x9908b0dfu);
                }
                uint32_t y = state[(*next)++];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680;
                y ^= (y << 15) & 0xefc60000;
                y ^= y >> 18;
                out_gen[ii] = min + rescale_uniform * (1.f/(float)(1<<23)*((float)(y>>9) + 0.5f));
            }
        } break;
        case MSML_PRNG_PCG: {
            uint64_t* state = &ctx->prng_state.pcg.state;
            uint64_t* inc = &ctx->prng_state.pcg.inc;
            for (int64_t ii=0; ii < out_n; ++ii) {
                uint64_t prev = *state;
                *state = prev*6364136223846793005ull + *inc;
                uint32_t mixed = ((prev>>18u) ^ prev) >> 27u;
                uint32_t rot = prev >> 59u;
                uint32_t y = (mixed>>rot) | (mixed << ((-rot)&31));
                out_gen[ii] = min + rescale_uniform * (1.f/(float)(1<<23)*((float)(y>>9) + 0.5f));
            }
        } break;
        default:
            msml_panic("Unknown PRNG algorithm: %d", ctx->prng_algorithm);
    }
}

static void msml__prng_init(msml_ctx_t* ctx, uint64_t seed) {
    seed = seed ? seed : 0x853c49e6748fea9bull ^ (uintptr_t)ctx ^ (uintptr_t)&ctx; /* Default seed. */
    switch (ctx->prng_algorithm) {
        case MSML_PRNG_MERSENNE_TWISTER: {
            uint32_t* state = ctx->prng_state.mersenne.state;
            *state = (uint32_t)seed;
            for (size_t i=1; i < 624; ++i)
                state[i] = ((state[i-1] ^ (state[i-1] >> 30))*1812433253 + i) & ~0u;
            ctx->prng_state.mersenne.next = 0;
            ctx->prng_state.mersenne.remaining = 1;
        } break;
        case MSML_PRNG_PCG: {
            ctx->prng_state.pcg.state = seed ^ 0x853c49e6748fea9bull;
            ctx->prng_state.pcg.inc = 0xda3e39cb94b95bdbull;
        } break;
        default:
            msml_panic("Unknown PRNG algorithm: %d", ctx->prng_algorithm);
    }
}

static void msml__ctx_push_chunk(msml_ctx_t* ctx) {
    uint8_t* chunk = (uint8_t*)(*ctx->alloc_fn)(NULL, ctx->pool.chunk_size);
    if (ctx->pool.warmup_chunks) memset(chunk, 0, ctx->pool.chunk_size);
    ctx->pool.mapped_total += ctx->pool.chunk_size;
    ctx->pool.delta = chunk + ctx->pool.chunk_size;
    if (ctx->pool.chunk_len == ctx->pool.chunk_cap)
        ctx->pool.chunks = (uint8_t**)(*ctx->alloc_fn)(ctx->pool.chunks, (ctx->pool.chunk_cap<<=1) * sizeof(*ctx->pool.chunks));
    ctx->pool.chunks[ctx->pool.chunk_len++] = chunk;
}

static void msml__system_host_info_query(msml_ctx_t* ctx); /* Query host system information. */
static void msml__blas_compute_dispatch_table_install(msml_ctx_t* ctx); /* Install BLAS dispatch table. */

#if defined(__x86_64__) || defined(_M_X64)
    static bool msml__ctx_x86_64_cpu_has_feature(const msml_ctx_t* ctx, msml__x86_64_feature_t feature) {
        const uint8_t* leafs = msml__x86_64_feature_leaves, *regs = msml__x86_64_feature_regs;
        const uint32_t* masks = msml__x86_64_feature_masks;
        const uint32_t (*features)[8][4] = &ctx->sys.x86_64_cpu_features;
        return (*features)[leafs[feature]][regs[feature]] & masks[feature];
    }
#endif

msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info) {
    puts("Creating MSML context...");
    int64_t time_stamp_start = msml_hpc_clock_us();

    /* Print MSML version and compiler info. */
    const char* compiler_name = "Unknown";
    int compiler_version_major = 0, compiler_version_minor = 0;
    #ifdef __clang__
        compiler_name = "Clang";
        compiler_version_major = __clang_major__;
        compiler_version_minor = __clang_minor__;
    #elif defined(__GNUC__)
        compiler_name = "GCC";
        compiler_version_major = __GNUC__;
        compiler_version_minor = __GNUC_MINOR__;
    #elif defined(_MSC_VER)
        compiler_name = "MSVC";
        compiler_version_major = _MSC_VER / 100;
        compiler_version_minor = _MSC_VER % 100;
    #endif
    printf("MSML v.%d.%d - " __DATE__ " " __TIME__ " - %s %d.%d\n", msml_version_major(MSML_VERSION), msml_version_minor(MSML_VERSION), compiler_name, compiler_version_major, compiler_version_minor);

    /* Enable fast math optimizations for x86-64 platforms. */
    #if MSML_CFG_X86_64_FAST_MATH && (defined(__x86_64__) || defined(_M_X64))
        /*
        ** Enable non-IEEE hardware optimizations in MXCSR:
        ** 0x0040: DAZ (Denormals Are Zeros) -> Converts denormal inputs to zero.
        ** 0x8000: FTZ (Flush To Zero) -> Sets underflow results to zero.
        ** See Intel Manual Vol. 1 §10.2.3.3-4 for details.
        */
        unsigned mxcsr;
        __asm__ __volatile__("stmxcsr\t%0":"=m"(mxcsr)); /* Store MXCSR register to var. */
        mxcsr |= 0x8040; /* Enable DAZ and FTZ bits. */
        __asm__ __volatile__("ldmxcsr\t%0"::"m"(mxcsr)); /* Load MXCSR register from var. */
    #endif

    /* Initialize context with default values or from context info. */
    msml_ctx_info_t ctx_info = {0};
    if (info) ctx_info = *info;
    ctx_info.alloc_fn = ctx_info.alloc_fn ? ctx_info.alloc_fn : &msml_alloc; /* Use default allocator if not provided. */
    msml_ctx_t* ctx = (msml_ctx_t*)(*ctx_info.alloc_fn)(NULL, sizeof(*ctx)); /* Allocate context. */
    memset(ctx, 0, sizeof(*ctx));
    ctx->alloc_fn = ctx_info.alloc_fn;
    ctx->ud = ctx_info.user_data;
    ctx->pool.chunk_size = ctx_info.pool_chunk_size ? msml_max(ctx_info.pool_chunk_size, 8) : MSML_DEFAULT_CHUNK_SIZE;
    ctx->pool.chunk_cap = ctx_info.pool_chunks_cap ? msml_max(ctx_info.pool_chunks_cap, 1) : MSML_DEFAULT_CHUNK_CAP;
    ctx->pool.warmup_chunks = ctx_info.warmup_chunks;

    /* Query and print host system information. */
    msml__system_host_info_query(ctx);
    printf("OS/Kernel: %s\n", ctx->sys.os_name);
    printf("CPU: %s, Virtual Cores: %u, Physical Cores: %u, Sockets: %u\n", ctx->sys.cpu_name, ctx->sys.cpu_virtual_cores, ctx->sys.cpu_physical_cores, ctx->sys.cpu_sockets);
    #if defined(__x86_64__) || defined(_M_X64) /* Print CPU features for x86-64 platforms. */
        printf("CPU Features:");
        for (unsigned i=0, k=0; i < MSML__X86_64_FEATURE__COUNT; ++i) {
            if (msml__ctx_x86_64_cpu_has_feature(ctx, i)) {
                if (k++ % 8 == 0) printf("\n\t");
                printf("%s ", msml__x86_64_feature_names[i]);
            }
        }
        putchar('\n');
    #endif
    double mem_total, mem_free, mem_used;
    const char* mem_unit_total, *mem_unit_free, *mem_unit_used;
    msml__humanize_memory_size(ctx->sys.phys_mem_total, &mem_total, &mem_unit_total);
    msml__humanize_memory_size(ctx->sys.phys_mem_free, &mem_free, &mem_unit_free);
    msml__humanize_memory_size((size_t)llabs((int64_t)ctx->sys.phys_mem_total-(int64_t)ctx->sys.phys_mem_free), &mem_used, &mem_unit_used);
    double mem_used_percent = fabs((double)(ctx->sys.phys_mem_total-ctx->sys.phys_mem_free))/(double)ctx->sys.phys_mem_total*100.0;
    printf("Physical memory: %.03f %s, Free: %.03f %s, Used: %.03f %s (%.02f%%)\n", mem_total, mem_unit_total, mem_free, mem_unit_free, mem_used, mem_unit_used, mem_used_percent);

    /* Prepare memory pool. */
    ctx->pool.chunks = (uint8_t**)(*ctx->alloc_fn)(NULL, ctx->pool.chunk_cap * sizeof(*ctx->pool.chunks)); /* Allocate chunk pointers. */
    msml__ctx_push_chunk(ctx); /* Allocate the first chunk. */

    /* Initialize PRNG state. */
    uint64_t host_tid = msml__thread_id();
    ctx->prng_algorithm = ctx_info.prng_algorithm;
    msml__prng_init(ctx, ctx_info.prng_seed^host_tid^(uintptr_t)ctx^(uintptr_t)&ctx_info); /* Initialize PRNG state. */
    ctx->host_thread_id = host_tid;

    /* Install BLAS dispatch table, specialized for host CPU arch. */
    msml__blas_compute_dispatch_table_install(ctx);

    /* Print context initialization time. */
    printf("MSML context initialized in %.05f ms.\n", msml_hpc_clock_elapsed_ms(time_stamp_start));
    return ctx;
}

msml_ctx_t* msml_ctx_create2(size_t pool_chunk_size) {
    msml_ctx_info_t info = {0};
    info.pool_chunk_size = pool_chunk_size;
    return msml_ctx_create(&info);
}

void* msml_ctx_pool_alloc(msml_ctx_t* ctx, size_t size) {
    msml_assert(size > 0 && size < (size_t)PTRDIFF_MAX, "Allocation size must be within (0, %zu), but is: %zu", PTRDIFF_MAX, size);
    if (ctx->pool.delta - ctx->pool.chunks[ctx->pool.chunk_len-1] < (ptrdiff_t)size) {
        if (ctx->pool.chunk_size < size) { /* Increase the chunk size if it's too small to accommodate the requested length */
            size_t lim = (size_t)PTRDIFF_MAX >> 1;
            do ctx->pool.chunk_size <<= 1;
            while (ctx->pool.chunk_size < size && (ctx->pool.chunk_size <= lim));
        }
        msml__ctx_push_chunk(ctx);
        msml_log_info("Allocated pool chunk: %.03f MiB", (double)ctx->pool.chunk_size/(double)(1<<20));
    }
    ctx->pool.delta -= size;
    ++ctx->pool.alloc_acc;
    ctx->pool.alloc_total += size;
    return ctx->pool.delta;
}

void* msml_ctx_pool_alloc_aligned(msml_ctx_t* ctx, size_t size, size_t align) {
    msml_assert(align && !(align&(align-1)), "Alignment must be power of 2: %zu", align); /* Alignment must be a power of 2 */
    return (void*)(((uintptr_t)msml_ctx_pool_alloc(ctx, size+align-1)+align-1)&~(align-1));
}

size_t msml_ctx_total_allocated_pool_memory(const msml_ctx_t* ctx) {
    size_t mem = sizeof(*ctx);
    mem += sizeof(*ctx->pool.chunks) * ctx->pool.chunk_cap;
    mem += ctx->pool.alloc_total;
    return mem;
}

msml_prng_algorithm_t msml_ctx_get_prng_algorithm(const msml_ctx_t* ctx) { return ctx->prng_algorithm; }

void msml_ctx_set_prng_algorithm(msml_ctx_t* ctx, msml_prng_algorithm_t algorithm, uint64_t seed) {
    ctx->prng_algorithm = algorithm;
    msml__prng_init(ctx, seed);
}

const char* msml_ctx_get_os_name(const msml_ctx_t* ctx) { return ctx->sys.os_name; }
const char* msml_ctx_get_cpu_name(const msml_ctx_t* ctx) { return ctx->sys.cpu_name; }
uint32_t msml_ctx_get_cpu_virtual_cores(const msml_ctx_t* ctx) { return ctx->sys.cpu_virtual_cores; }
uint32_t msml_ctx_get_cpu_physical_cores(const msml_ctx_t* ctx) { return ctx->sys.cpu_physical_cores; }
uint32_t msml_ctx_get_cpu_sockets(const msml_ctx_t* ctx) { return ctx->sys.cpu_sockets; }
uint64_t msml_ctx_get_physical_memory_total(const msml_ctx_t* ctx) { return ctx->sys.phys_mem_total; }
uint64_t msml_ctx_get_physical_memory_free(const msml_ctx_t* ctx) { return ctx->sys.phys_mem_free; }
bool msml_ctx_is_numa_system(const msml_ctx_t* ctx) { return false; /* TODO */ }

void msml_ctx_destroy(msml_ctx_t* ctx) {
    size_t mem_total = msml_ctx_total_allocated_pool_memory(ctx);
    size_t mem_mapped = ctx->pool.mapped_total;
    void* (*alloc)(void* blk, size_t size) = ctx->alloc_fn;
    for (size_t i=0; i < ctx->pool.chunk_len; ++i) /* Free individual chunks */
        (*alloc)(ctx->pool.chunks[i], 0);
    (*alloc)(ctx->pool.chunks, 0);
    memset(ctx, (uintptr_t)ctx & 0xff, sizeof(*ctx));
    (*alloc)(ctx, 0);
    ctx = NULL;
    double alloc_total, mapped_total;
    const char* alloc_unit, *mapped_unit;
    msml__humanize_memory_size(mem_total, &alloc_total, &alloc_unit);
    msml__humanize_memory_size(mem_mapped, &mapped_total, &mapped_unit);
    printf("Allocated in pool: %.03f %s, Mapped memory: %.03f %s\n", alloc_total, alloc_unit, mapped_total, mapped_unit);
    puts("MSML context destroyed.");
}

#define msml__load_local_storage_group(xk, prefix, var) \
    const int64_t prefix##0 = (xk)->var[0]; \
    const int64_t prefix##1 = (xk)->var[1]; \
    const int64_t prefix##2 = (xk)->var[2]; \
    const int64_t prefix##3 = (xk)->var[3]; \
    (void)prefix##0; \
    (void)prefix##1; \
    (void)prefix##2; \
    (void)prefix##3;

#define msml__dot4_unrolled_var_arr(arr, x0, x1, x2, x3) \
    ( \
        arr[0]*(x0) \
        + arr[1]*(x1) \
        + arr[2]*(x2) \
        + arr[3]*(x3) \
    )

#define msml__resolve_physical_ptr(tensor, d0, d1, d2, d3) \
    (((uint8_t*)(tensor)->buf) + msml__dot4_unrolled_var_arr((tensor)->strides, d0, d1, d2, d3))

const msml_dtype_info_t* msml_get_dtype_info(msml_dtype_t type) {
    static const msml_dtype_info_t infos[MSML_DTYPE_COUNT_] = {
        [MSML_DTYPE_F32] = {
            sizeof(float),
            "f32"
        },
    };
    return &infos[type];
}

const char* msml_op_get_name(msml_op_t op) {
    #define _(enumerator, mnemonic, argcount) #enumerator
        static const char* const names[MSML_OP__COUNT] = {
            msml_op_def(_, MSML_SEP)
        };
    #undef _
    return names[op];
}

const char* msml_op_get_mnemonic(msml_op_t op) {
    #define _(enumerator, mnemonic, argcount) mnemonic
        static const char* const mnemonics[MSML_OP__COUNT] = {
            msml_op_def(_, MSML_SEP)
        };
    #undef _
    return mnemonics[op];
}

uint8_t msml_op_get_argcount(msml_op_t op) {
    #define _(enumerator, mnemonic, argcount) ((argcount)&0xff)
        static const uint8_t arg_counts[MSML_OP__COUNT] = {
            msml_op_def(_, MSML_SEP)
        };
    #undef _
    return arg_counts[op];
}

/* Generic validation expression. */
#define msml__validate_expr_gen(expr, message, ...) \
    if (msml_unlikely(!(expr))) { \
        if (print_error) { \
           msml_log_error(message, ## __VA_ARGS__); \
        } \
        return false; \
    }

#define msml__validate_shape_eq_r_x() msml__validate_expr_gen(msml_tensor_is_shape_eq(tensor->inputs[0], tensor), "Input tensor shape mismatch, both input tensors must have the same shape.");
#define msml__validate_shape_broadcastable_y_x() msml__validate_expr_gen(msml_tensor_can_broadcast(tensor->inputs[1], tensor->inputs[0]), "Input tensor shape mismatch, second input tensor must be broadcastable into first input tensor.")

static bool msml__validate_op_nop(const msml_tensor_t* tensor, bool print_error) {
    (void)tensor;
    (void)print_error;
    return true;
}

static bool msml__validate_op_transpose(const msml_tensor_t* tensor, bool print_error) {
    (void)print_error;
    return tensor->rank >= 2;
}

static bool msml__validate_op_clone(const msml_tensor_t* tensor, bool print_error) {
    (void)tensor;
    (void)print_error;
    return true;
}

static bool msml__validate_op_step(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_softmax(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_softmax_dv(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_sigmoid(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_sigmoid_dv(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_hard_sigmoid(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_hard_sigmoid_dv(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_silu(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_silu_dv(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_tanh(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_tanh_dv(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_relu(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_relu_dv(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_gelu(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_gelu_dv(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    return true;
}

static bool msml__validate_op_add(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    msml__validate_shape_broadcastable_y_x()
    msml__validate_expr_gen(tensor->strides[0] == sizeof(float), "Result must be contiguous.");
    msml__validate_expr_gen(tensor->inputs[0]->strides[0] == sizeof(float), "First tensor must be contiguous.");
    return true;
}

static bool msml__validate_op_sub(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    msml__validate_shape_broadcastable_y_x()
    msml__validate_expr_gen(tensor->strides[0] == sizeof(float), "Result must be contiguous.");
    msml__validate_expr_gen(tensor->inputs[0]->strides[0] == sizeof(float), "First tensor must be contiguous.");
    return true;
}

static bool msml__validate_op_mul(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    msml__validate_shape_broadcastable_y_x()
    msml__validate_expr_gen(tensor->strides[0] == sizeof(float), "Result must be contiguous.");
    msml__validate_expr_gen(tensor->inputs[0]->strides[0] == sizeof(float), "First tensor must be contiguous.");
    return true;
}

static bool msml__validate_op_div(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_shape_eq_r_x()
    msml__validate_shape_broadcastable_y_x()
    msml__validate_expr_gen(tensor->strides[0] == sizeof(float), "Result must be contiguous.");
    msml__validate_expr_gen(tensor->inputs[0]->strides[0] == sizeof(float), "First tensor must be contiguous.");
    return true;
}

static bool msml__validate_op_matmul(const msml_tensor_t* tensor, bool print_error) {
    msml__validate_expr_gen(
        tensor->shape[0] == tensor->inputs[0]->shape[1],
        "ERROR: Matmul operation failed due to shape mismatch.\n"
        "    - Result Tensor: '%s', Dimension [0] = %zu\n"
        "    - First Input Tensor: '%s', Dimension [1] = %zu\n"
        "    Hint: Ensure the second dimension of the first input tensor matches the first dimension of the result tensor.",
        tensor->name, (size_t)tensor->shape[0], tensor->inputs[0]->name, (size_t)tensor->inputs[0]->shape[1]
    );

    msml__validate_expr_gen(
        tensor->shape[1] == tensor->inputs[1]->shape[1],
        "ERROR: Matmul operation failed due to shape mismatch.\n"
        "    - Result Tensor: '%s', Dimension [1] = %zu\n"
        "    - Second Input Tensor: '%s', Dimension [1] = %zu\n"
        "    Hint: Ensure the dimensions match for a valid multiplication.",
        tensor->name, (size_t)tensor->shape[1], tensor->inputs[1]->name, (size_t)tensor->inputs[1]->shape[1]
    );

    msml__validate_expr_gen(
        tensor->shape[2] == tensor->inputs[1]->shape[2],
        "ERROR: Dimension mismatch.\n"
        "    - Result Tensor: '%s', Dimension [2] = %zu\n"
        "    - Second Input Tensor: '%s', Dimension [2] = %zu\n"
        "    Hint: The dimensions must match.",
        tensor->name, (size_t)tensor->shape[2], tensor->inputs[1]->name, (size_t)tensor->inputs[1]->shape[2]
    );

    msml__validate_expr_gen(
        tensor->shape[3] == tensor->inputs[1]->shape[3],
        "ERROR: Dimension mismatch.\n"
        "    - Result Tensor: '%s', Dimension [3] = %zu\n"
        "    - Second Input Tensor: '%s', Dimension [3] = %zu\n"
        "    Hint: The dimensions must match.",
        tensor->name, (size_t)tensor->shape[3], tensor->inputs[1]->name, (size_t)tensor->inputs[1]->shape[3]
    );
    msml__validate_expr_gen(tensor->inputs[0]->strides[0] == sizeof(float), "Both input tensors must be contiguous");
    msml__validate_expr_gen(tensor->inputs[1]->strides[0] == sizeof(float), "Both input tensors must be contiguous");
    msml__validate_expr_gen(tensor->strides[0] == sizeof(float), "Result tensor must be contiguous");
    msml__validate_expr_gen(tensor->strides[0] <= tensor->strides[1], "Result tensor cannot be permuted or transposed.");
    msml__validate_expr_gen(tensor->strides[1] <= tensor->strides[2], "Result tensor cannot be permuted or transposed.");
    msml__validate_expr_gen(tensor->strides[2] <= tensor->strides[03], "Result tensor cannot be permuted or transposed.");
    msml__validate_expr_gen(tensor->inputs[1]->shape[2] % tensor->inputs[0]->shape[2] == 0, "Second input tensor must be broadcastable into first input tensor.");
    msml__validate_expr_gen(tensor->inputs[1]->shape[3] % tensor->inputs[0]->shape[3] == 0, "Second input tensor must be broadcastable into first input tensor.");
    return true;
}

static bool (*msml__op_get_validator_routine(msml_op_t op))(const msml_tensor_t* tensor, bool print_error) {
    static bool (*const routines[MSML_OP__COUNT])(const msml_tensor_t* tensor, bool print_error) = {
        [MSML_OP_NOP] = &msml__validate_op_nop,
        [MSML_OP_TRANSPOSE] = &msml__validate_op_transpose,
        [MSML_OP_CLONE] = &msml__validate_op_clone,
        [MSML_OP_STEP] = &msml__validate_op_step,
        [MSML_OP_SOFTMAX] = &msml__validate_op_softmax,
        [MSML_OP_SOFTMAX_DV] = &msml__validate_op_softmax_dv,
        [MSML_OP_SIGMOID] = &msml__validate_op_sigmoid,
        [MSML_OP_SIGMOID_DV] = &msml__validate_op_sigmoid_dv,
        [MSML_OP_HARD_SIGMOID] = &msml__validate_op_hard_sigmoid,
        [MSML_OP_HARD_SIGMOID_DV] = &msml__validate_op_hard_sigmoid_dv,
        [MSML_OP_SILU] = &msml__validate_op_silu,
        [MSML_OP_SILU_DV] = &msml__validate_op_silu_dv,
        [MSML_OP_TANH] = &msml__validate_op_tanh,
        [MSML_OP_TANH_DV] = &msml__validate_op_tanh_dv,
        [MSML_OP_RELU] = &msml__validate_op_relu,
        [MSML_OP_RELU_DV] = &msml__validate_op_relu_dv,
        [MSML_OP_GELU] = &msml__validate_op_gelu,
        [MSML_OP_GELU_DV] = &msml__validate_op_gelu_dv,
        [MSML_OP_ADD] = &msml__validate_op_add,
        [MSML_OP_SUB] = &msml__validate_op_sub,
        [MSML_OP_MUL] = &msml__validate_op_mul,
        [MSML_OP_DIV] = &msml__validate_op_div,
        [MSML_OP_MATMUL] = &msml__validate_op_matmul,
    };
    return routines[op];
}

#undef msml__validate_inputs

/* Rescale factors to push the exponent of a number towards zero. */
#define rescale_exponents(P, N) \
  P(308), P(289), P(270), P(250), P(231), P(212), P(193), P(173), P(154), \
  P(135), P(115), P(96), P(77), P(58), P(38), P(0), P(0), P(0), N(39), N(58), \
  N(77), N(96), N(116), N(135), N(154), N(174), N(193), N(212), N(231), \
  N(251), N(270), N(289)
#define one_e_p(X) 1e+0 ## X
#define one_e_n(X) 1e-0 ## X
static const int16_t msml__rescale_e[] = { rescale_exponents(-, +) };
static const double msml__rescale_n[] = { rescale_exponents(one_e_p, one_e_n) };
#undef one_e_n
#undef one_e_p

/*
** For p in range -70 through 57, this table encodes pairs (m, e) such that
** 4*2^p <= (uint8_t)m*10^e, and is the smallest value for which this holds.
*/
static const int8_t msml__four_ulp_m_e[] = {
    34, -21, 68, -21, 14, -20, 28, -20, 55, -20, 2, -19, 3, -19, 5, -19, 9, -19,
    -82, -18, 35, -18, 7, -17, -117, -17, 28, -17, 56, -17, 112, -16, -33, -16,
    45, -16, 89, -16, -78, -15, 36, -15, 72, -15, -113, -14, 29, -14, 57, -14,
    114, -13, -28, -13, 46, -13, 91, -12, -74, -12, 37, -12, 73, -12, 15, -11, 3,
    -11, 59, -11, 2, -10, 3, -10, 5, -10, 1, -9, -69, -9, 38, -9, 75, -9, 15, -7,
    3, -7, 6, -7, 12, -6, -17, -7, 48, -7, 96, -7, -65, -6, 39, -6, 77, -6, -103,
    -5, 31, -5, 62, -5, 123, -4, -11, -4, 49, -4, 98, -4, -60, -3, 4, -2, 79, -3,
    16, -2, 32, -2, 63, -2, 2, -1, 25, 0, 5, 1, 1, 2, 2, 2, 4, 2, 8, 2, 16, 2,
    32, 2, 64, 2, -128, 2, 26, 2, 52, 2, 103, 3, -51, 3, 41, 4, 82, 4, -92, 4,
    33, 4, 66, 4, -124, 5, 27, 5, 53, 5, 105, 6, 21, 6, 42, 6, 84, 6, 17, 7, 34,
    7, 68, 7, 2, 8, 3, 8, 6, 8, 108, 9, -41, 9, 43, 10, 86, 9, -84, 10, 35, 10,
    69, 10, -118, 11, 28, 11, 55, 12, 11, 13, 22, 13, 44, 13, 88, 13, -80, 13,
    36, 13, 71, 13, -115, 14, 29, 14, 57, 14, 113, 15, -30, 15, 46, 15, 91, 15,
    19, 16, 37, 16, 73, 16, 2, 17, 3, 17, 6, 17
};

/* min(2^32-1, 10^e-1) for e in range 0 through 10 */
static const uint32_t msml__ndigits_dec_threshold[] = {
    0, 9U, 99U, 999U, 9999U, 99999U, 999999U,
    9999999U, 99999999U, 999999999U, 0xffffffffU
};

/* Compute the number of digits in the decimal representation of x. */
static size_t msml__ndigits_dec(uint32_t x) {
    size_t t = ((msml_fls(x | 1) * 77) >> 8) + 1; /* 2^8/77 is roughly log2(10) */
    return t + (x > msml__ndigits_dec_threshold[t]);
}

#define wint_r(x, sh, sc) { uint32_t d = (x*(((1<<sh)+sc-1)/sc))>>sh; x -= d*sc; *p++ = (char)('0'+d); }
static char* msml__wuint9(char* p, uint32_t u) {
    uint32_t v = u / 10000, w;
    u -= v * 10000;
    w = v / 10000;
    v -= w * 10000;
    *p++ = (char)('0'+w);
    wint_r(v, 23, 1000)
    wint_r(v, 12, 100)
    wint_r(v, 10, 10)
    *p++ = (char)('0'+v);
    wint_r(u, 23, 1000)
    wint_r(u, 12, 100)
    wint_r(u, 10, 10)
    *p++ = (char)('0'+u);
    return p;
}
#undef wint_r

#define wint_r(x, sh, sc) { uint32_t d = (x*(((1<<sh)+sc-1)/sc))>>sh; x -= d*sc; *p++ = (char)('0'+d); }
static char* msml__wint(char* p, int32_t k) {
    uint32_t u = (uint32_t)k;
    if (k < 0) { u = ~u+1u; *p++ = '-'; }
    if (u < 10000) {
        if (u < 10) goto dig1;
        if (u < 100) goto dig2;
        if (u < 1000) goto dig3;
    } else {
        uint32_t v = u / 10000; u -= v * 10000;
        if (v < 10000) {
            if (v < 10) goto dig5;
            if (v < 100) goto dig6;
            if (v < 1000) goto dig7;
        } else {
            uint32_t w = v / 10000; v -= w * 10000;
            if (w >= 10) wint_r(w, 10, 10)
                         *p++ = (char)('0'+w);
        }
        wint_r(v, 23, 1000)
        dig7: wint_r(v, 12, 100)
        dig6: wint_r(v, 10, 10)
        dig5: *p++ = (char)('0'+v);
    }
    wint_r(u, 23, 1000)
    dig3: wint_r(u, 12, 100)
    dig2: wint_r(u, 10, 10)
    dig1: *p++ = (char)('0'+u);
    return p;
}
#undef wint_r

/* -- Extended precision arithmetic --------------------------------------- */

/*
** The "nd" format is a fixed-precision decimal representation for numbers. It
** consists of up to 64 uint32_t values, with each uint32_t storing a value
** in the range [0, 1e9). A number in "nd" format consists of three variables:
**
**  uint32_t nd[64];
**  uint32_t ndlo;
**  uint32_t ndhi;
**
** The integral part of the number is stored in nd[0 ... ndhi], the value of
** which is sum{i in [0, ndhi] | nd[i] * 10^(9*i)}. If the fractional part of
** the number is zero, ndlo is zero. Otherwise, the fractional part is stored
** in nd[ndlo ... 63], the value of which is taken to be
** sum{i in [ndlo, 63] | nd[i] * 10^(9*(i-64))}.
**
** If the array part had 128 elements rather than 64, then every double would
** have an exact representation in "nd" format. With 64 elements, all integral
** doubles have an exact representation, and all non-integral doubles have
** enough digits to make both %.99e and %.99f do the right thing.
*/
#define MSML__ND_MUL2K_MAX_SHIFT 29
#define MSML__ND_MUL2K_DIV1E9(val) ((uint32_t)((val) / 1000000000))

/* Multiply nd by 2^k and add carry_in (ndlo is assumed to be zero). */
static uint32_t nd_mul2k(uint32_t* nd, uint32_t ndhi, uint32_t k, uint32_t carry_in, msml_format_flags sf) {
    uint32_t i, ndlo = 0, start = 1;
    /* Performance hacks. */
    if (k > MSML__ND_MUL2K_MAX_SHIFT*2 && MSML_FMT_FP(sf) != MSML_FMT_FP(MSML_FMT_T_FP_F)) {
        start = ndhi - (MSML_FMT_PREC(sf) + 17) / 8;
    }
    /* Real logic. */
    while (k >= MSML__ND_MUL2K_MAX_SHIFT) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << MSML__ND_MUL2K_MAX_SHIFT) | carry_in;
            carry_in = MSML__ND_MUL2K_DIV1E9(val);
            nd[i] = (uint32_t)val - carry_in * 1000000000;
        }
        if (carry_in) {
            nd[++ndhi] = carry_in; carry_in = 0;
            if (start++ == ndlo) ++ndlo;
        }
        k -= MSML__ND_MUL2K_MAX_SHIFT;
    }
    if (k) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << k) | carry_in;
            carry_in = MSML__ND_MUL2K_DIV1E9(val);
            nd[i] = (uint32_t)val - carry_in * 1000000000;
        }
        if (carry_in) nd[++ndhi] = carry_in;
    }
    return ndhi;
}

/* Divide nd by 2^k (ndlo is assumed to be zero). */
static uint32_t nd_div2k(uint32_t* nd, uint32_t ndhi, uint32_t k, msml_format_flags sf) {
    uint32_t ndlo = 0, stop1 = ~0, stop2 = ~0;
    /* Performance hacks. */
    if (!ndhi) {
        if (!nd[0]) {
            return 0;
        } else {
            uint32_t s = msml_ffs(nd[0]);
            if (s >= k) { nd[0] >>= k; return 0; }
            nd[0] >>= s; k -= s;
        }
    }
    if (k > 18) {
        if (MSML_FMT_FP(sf) == MSML_FMT_FP(MSML_FMT_T_FP_F)) {
            stop1 = 63 - (int32_t)MSML_FMT_PREC(sf) / 9;
        } else {
            int32_t floorlog2 = ndhi * 29 + msml_fls(nd[ndhi]) - k;
            int32_t floorlog10 = (int32_t)(floorlog2 * 0.30102999566398114);
            stop1 = 62 + (floorlog10 - (int32_t)MSML_FMT_PREC(sf)) / 9;
            stop2 = 61 + ndhi - (int32_t)MSML_FMT_PREC(sf) / 8;
        }
    }
    /* Real logic. */
    while (k >= 9) {
        uint32_t i = ndhi, carry = 0;
        for (;;) {
            uint32_t val = nd[i];
            nd[i] = (val >> 9) + carry;
            carry = (val & 0x1ff) * 1953125;
            if (i == ndlo) break;
            i = (i - 1) & 0x3f;
        }
        if (ndlo != stop1 && ndlo != stop2) {
            if (carry) { ndlo = (ndlo - 1) & 0x3f; nd[ndlo] = carry; }
            if (!nd[ndhi]) { ndhi = (ndhi - 1) & 0x3f; stop2--; }
        } else if (!nd[ndhi]) {
            if (ndhi != ndlo) { ndhi = (ndhi - 1) & 0x3f; stop2--; }
            else return ndlo;
        }
        k -= 9;
    }
    if (k) {
        uint32_t mask = (1U << k) - 1, mul = 1000000000 >> k, i = ndhi, carry = 0;
        for (;;) {
            uint32_t val = nd[i];
            nd[i] = (val >> k) + carry;
            carry = (val & mask) * mul;
            if (i == ndlo) break;
            i = (i - 1) & 0x3f;
        }
        if (carry) { ndlo = (ndlo - 1) & 0x3f; nd[ndlo] = carry; }
    }
    return ndlo;
}

/* Add m*10^e to nd (assumes ndlo <= e/9 <= ndhi and 0 <= m <= 9). */
static uint32_t nd_add_m10e(uint32_t* nd, uint32_t ndhi, uint8_t m, int32_t e) {
    uint32_t i, carry;
    if (e >= 0) {
        i = (uint32_t)e/9;
        carry = m * (msml__ndigits_dec_threshold[e - (int32_t)i*9] + 1);
    } else {
        int32_t f = (e-8)/9;
        i = (uint32_t)(64 + f);
        carry = m * (msml__ndigits_dec_threshold[e - f*9] + 1);
    }
    for (;;) {
        uint32_t val = nd[i] + carry;
        if (msml_unlikely(val >= 1000000000)) {
            val -= 1000000000;
            nd[i] = val;
            if (msml_unlikely(i == ndhi)) {
                ndhi = (ndhi + 1) & 0x3f;
                nd[ndhi] = 1;
                break;
            }
            carry = 1;
            i = (i + 1) & 0x3f;
        } else {
            nd[i] = val;
            break;
        }
    }
    return ndhi;
}

static bool nd_similar(uint32_t* nd, uint32_t ndhi, uint32_t* ref, size_t hilen, size_t prec) {
    char nd9[9], ref9[9];
    if (hilen <= prec) {
        if (msml_unlikely(nd[ndhi] != *ref)) return 0;
        prec -= hilen; ref--; ndhi = (ndhi - 1) & 0x3f;
        if (prec >= 9) {
            if (msml_unlikely(nd[ndhi] != *ref)) return 0;
            prec -= 9; ref--; ndhi = (ndhi - 1) & 0x3f;
        }
    } else {
        prec -= hilen - 9;
    }
    msml_assert(prec < 9, "bad precision %d", prec);
    msml__wuint9(nd9, nd[ndhi]);
    msml__wuint9(ref9, *ref);
    return !memcmp(nd9, ref9, prec) && (nd9[prec] < '5') == (ref9[prec] < '5');
}

/* Format f64 according to format flags. */
static char* msml__fmt_f64(msml_format_flags sf, double n, char* p) {
    size_t width = MSML_FMT_WIDTH(sf), prec = MSML_FMT_PREC(sf), len;
    union {
        uint64_t u64;
        double n;
        struct { /* TODO: make endian aware */
            uint32_t lo, hi;
        } u32;
    } t = {.n = n};
    if (msml_unlikely((t.u32.hi << 1) >= 0xffe00000)) {
        /* Handle non-finite values uniformly for %a, %e, %f, %g. */
        int prefix = 0, ch = (sf & MSML_FMT_F_UPPER) ? 0x202020 : 0;
        if (((t.u32.hi & 0x000fffff) | t.u32.lo) != 0) {
            ch ^= ('n' << 16) | ('a' << 8) | 'n';
            if ((sf & MSML_FMT_F_SPACE)) prefix = ' ';
        } else {
            ch ^= ('i' << 16) | ('n' << 8) | 'f';
            if ((t.u32.hi & 0x80000000)) prefix = '-';
            else if ((sf & MSML_FMT_F_PLUS)) prefix = '+';
            else if ((sf & MSML_FMT_F_SPACE)) prefix = ' ';
        }
        len = 3 + (prefix != 0);
        if (!(sf & MSML_FMT_F_LEFT)) while (width-- > len) *p++ = ' ';
        if (prefix) *p++ = prefix;
        *p++ = (char)(ch >> 16); *p++ = (char)(ch >> 8); *p++ = (char)ch;
    } else if (MSML_FMT_FP(sf) == MSML_FMT_FP(MSML_FMT_T_FP_A)) {
        /* %a */
        const char* hexdig = (sf & MSML_FMT_F_UPPER) ? "0123456789ABCDEFPX" : "0123456789abcdefpx";
        int32_t e = (t.u32.hi >> 20) & 0x7ff;
        char prefix = 0, eprefix = '+';
        if (t.u32.hi & 0x80000000) prefix = '-';
        else if ((sf & MSML_FMT_F_PLUS)) prefix = '+';
        else if ((sf & MSML_FMT_F_SPACE)) prefix = ' ';
        t.u32.hi &= 0xfffff;
        if (e) {
            t.u32.hi |= 0x100000;
            e -= 1023;
        } else if (t.u32.lo | t.u32.hi) {
            /* Non-zero denormal - normalise it. */
            uint32_t shift = t.u32.hi ? 20-msml_fls(t.u32.hi) : 52-msml_fls(t.u32.lo);
            e = -1022 - shift;
            t.u64 <<= shift;
        }
        /* abs(n) == t.u64 * 2^(e - 52) */
        /* If n != 0, bit 52 of t.u64 is set, and is the highest set bit. */
        if ((int32_t)prec < 0) {
            /* Default precision: use smallest precision giving exact result. */
            prec = t.u32.lo ? 13-msml_ffs(t.u32.lo)/4 : 5-msml_ffs(t.u32.hi|0x100000)/4;
        } else if (prec < 13) {
            /* Precision is sufficiently low as to maybe require rounding. */
            t.u64 += (((uint64_t)1) << (51 - prec*4));
        }
        if (e < 0) {
            eprefix = '-';
            e = -e;
        }
        len = 5 + msml__ndigits_dec((uint32_t)e) + prec + (prefix != 0)
              + ((prec | (sf & MSML_FMT_F_ALT)) != 0);
        if (!(sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO))) {
            while (width-- > len) *p++ = ' ';
        }
        if (prefix) *p++ = prefix;
        *p++ = '0';
        *p++ = hexdig[17]; /* x or X */
        if ((sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO)) == MSML_FMT_F_ZERO) {
            while (width-- > len) *p++ = '0';
        }
        *p++ = '0' + (t.u32.hi >> 20); /* Usually '1', sometimes '0' or '2'. */
        if ((prec | (sf & MSML_FMT_F_ALT))) {
            /* Emit fractional part. */
            char* q = p + 1 + prec;
            *p = '.';
            if (prec < 13) t.u64 >>= (52 - prec*4);
            else while (prec > 13) p[prec--] = '0';
            while (prec) { p[prec--] = hexdig[t.u64 & 15]; t.u64 >>= 4; }
            p = q;
        }
        *p++ = hexdig[16]; /* p or P */
        *p++ = eprefix; /* + or - */
        p = msml__wint(p, e);
    } else {
        /* %e or %f or %g - begin by converting n to "nd" format. */
        uint32_t nd[64];
        uint32_t ndhi = 0, ndlo, i;
        int32_t e = (int32_t)(t.u32.hi >> 20) & 0x7ff, ndebias = 0;
        char prefix = 0, *q;
        if (t.u32.hi & 0x80000000) prefix = '-';
        else if ((sf & MSML_FMT_F_PLUS)) prefix = '+';
        else if ((sf & MSML_FMT_F_SPACE)) prefix = ' ';
        prec += ((int32_t)prec >> 31) & 7; /* Default precision is 6. */
        if (MSML_FMT_FP(sf) == MSML_FMT_FP(MSML_FMT_T_FP_G)) {
            /* %g - decrement precision if non-zero (to make it like %e). */
            prec--;
            prec ^= (uint32_t)((int32_t)prec >> 31);
        }
        if ((sf & MSML_FMT_T_FP_E) && prec < 14 && n != 0) {
            /* Precision is sufficiently low that rescaling will probably work. */
            if ((ndebias = msml__rescale_e[e >> 6])) {
                t.n = n * msml__rescale_n[e >> 6];
                if (msml_unlikely(!e)) t.n *= 1e10, ndebias -= 10;
                t.u64 -= 2; /* Convert 2ulp below (later we convert 2ulp above). */
                nd[0] = 0x100000 | (t.u32.hi & 0xfffff);
                e = ((int32_t)(t.u32.hi >> 20) & 0x7ff) - 1075 - (MSML__ND_MUL2K_MAX_SHIFT < 29);
                goto load_t_lo; rescale_failed:
                t.n = n;
                e = (int32_t)(t.u32.hi >> 20) & 0x7ff;
                ndebias = 0;
                ndhi = 0;
            }
        }
        nd[0] = t.u32.hi & 0xfffff;
        if (e == 0) e++; else nd[0] |= 0x100000;
        e -= 1043;
        if (t.u32.lo) {
            e -= 32 + (MSML__ND_MUL2K_MAX_SHIFT < 29); load_t_lo:
            #if MSML__ND_MUL2K_MAX_SHIFT >= 29
                nd[0] = (nd[0] << 3) | (t.u32.lo >> 29);
                ndhi = nd_mul2k(nd, ndhi, 29, t.u32.lo & 0x1fffffff, sf);
            #elif MSML__ND_MUL2K_MAX_SHIFT >= 11
                ndhi = nd_mul2k(nd, ndhi, 11, t.u32.lo >> 21, sf);
                ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo >> 10) & 0x7ff, sf);
                ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo <<  1) & 0x7ff, sf);
            #else
            #   error "MSML__ND_MUL2K_MAX_SHIFT not big enough"
            #endif
        }
        if (e >= 0) {
            ndhi = nd_mul2k(nd, ndhi, (uint32_t)e, 0, sf);
            ndlo = 0;
        } else {
            ndlo = nd_div2k(nd, ndhi, (uint32_t)-e, sf);
            if (ndhi && !nd[ndhi]) ndhi--;
        }
        /* |n| == nd * 10^ndebias (for slightly loose interpretation of ==) */
        if ((sf & MSML_FMT_T_FP_E)) {
            /* %e or %g - assume %e and start by calculating nd's exponent (nde). */
            char eprefix = '+';
            int32_t nde = -1;
            size_t hilen;
            if (ndlo && !nd[ndhi]) {
                ndhi = 64; do {} while (!nd[--ndhi]);
                nde -= 64 * 9;
            }
            hilen = msml__ndigits_dec(nd[ndhi]);
            nde += (int32_t)(ndhi * 9 + hilen);
            if (ndebias) {
                /*
                ** Rescaling was performed, but this introduced some error, and might
                ** have pushed us across a rounding boundary. We check whether this
                ** error affected the result by introducing even more error (2ulp in
                ** either direction), and seeing whether a rounding boundary was
                ** crossed. Having already converted the -2ulp case, we save off its
                ** most significant digits, convert the +2ulp case, and compare them.
                */
                int32_t eidx = e + 70 + (MSML__ND_MUL2K_MAX_SHIFT < 29)
                               + (t.u32.lo >= 0xfffffffe && !(~t.u32.hi << 12));
                const int8_t *m_e = msml__four_ulp_m_e + eidx * 2;
                msml_assert(0 <= eidx && eidx < 128, "bad eidx %d", eidx);
                nd[33] = nd[ndhi];
                nd[32] = nd[(ndhi - 1) & 0x3f];
                nd[31] = nd[(ndhi - 2) & 0x3f];
                nd_add_m10e(nd, ndhi, (uint8_t)*m_e, m_e[1]);
                if (msml_unlikely(!nd_similar(nd, ndhi, nd + 33, hilen, prec + 1))) {
                    goto rescale_failed;
                }
            }
            if ((int32_t)(prec - nde) < (0x3f & -(int32_t)ndlo) * 9) {
                /* Precision is sufficiently low as to maybe require rounding. */
                ndhi = nd_add_m10e(nd, ndhi, 5, (int32_t)nde - prec - 1);
                nde += (hilen != msml__ndigits_dec(nd[ndhi]));
            }
            nde += ndebias;
            if ((sf & MSML_FMT_T_FP_F)) {
                /* %g */
                if ((int32_t)prec >= nde && nde >= -4) {
                    if (nde < 0) ndhi = 0;
                    prec -= nde;
                    goto g_format_like_f;
                } else if (!(sf & MSML_FMT_F_ALT) && prec && width > 5) {
                    /* Decrease precision in order to strip trailing zeroes. */
                    char tail[9];
                    uint32_t maxprec = hilen - 1 + ((ndhi - ndlo) & 0x3f) * 9;
                    if (prec >= maxprec) prec = maxprec;
                    else ndlo = (ndhi - (((int32_t)(prec - hilen) + 9) / 9)) & 0x3f;
                    i = prec - hilen - (((ndhi - ndlo) & 0x3f) * 9) + 10;
                    msml__wuint9(tail, nd[ndlo]);
                    while (prec && tail[--i] == '0') {
                        prec--;
                        if (!i) {
                            if (ndlo == ndhi) { prec = 0; break; }
                            ndlo = (ndlo + 1) & 0x3f;
                            msml__wuint9(tail, nd[ndlo]);
                            i = 9;
                        }
                    }
                }
            }
            if (nde < 0) {
                /* Make nde non-negative. */
                eprefix = '-';
                nde = -nde;
            }
            len = 3 + prec + (prefix != 0) + msml__ndigits_dec((uint32_t)nde) + (nde < 10)
                  + ((prec | (sf & MSML_FMT_F_ALT)) != 0);
            if (!(sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO))) {
                while (width-- > len) *p++ = ' ';
            }
            if (prefix) *p++ = prefix;
            if ((sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO)) == MSML_FMT_F_ZERO) {
                while (width-- > len) *p++ = '0';
            }
            q = msml__wint(p + 1, nd[ndhi]);
            p[0] = p[1]; /* Put leading digit in the correct place. */
            if ((prec | (sf & MSML_FMT_F_ALT))) {
                /* Emit fractional part. */
                p[1] = '.'; p += 2;
                prec -= (size_t)(q - p); p = q; /* Account for digits already emitted. */
                /* Then emit chunks of 9 digits (this may emit 8 digits too many). */
                for (i = ndhi; (int32_t)prec > 0 && i != ndlo; prec -= 9) {
                    i = (i - 1) & 0x3f;
                    p = msml__wuint9(p, nd[i]);
                }
                if ((sf & MSML_FMT_T_FP_F) && !(sf & MSML_FMT_F_ALT)) {
                    /* %g (and not %#g) - strip trailing zeroes. */
                    p += (int32_t)prec & ((int32_t)prec >> 31);
                    while (p[-1] == '0') p--;
                    if (p[-1] == '.') p--;
                } else {
                    /* %e (or %#g) - emit trailing zeroes. */
                    while ((int32_t)prec > 0) { *p++ = '0'; prec--; }
                    p += (int32_t)prec;
                }
            } else {
                p++;
            }
            *p++ = (sf & MSML_FMT_F_UPPER) ? 'E' : 'e';
            *p++ = eprefix; /* + or - */
            if (nde < 10) *p++ = '0'; /* Always at least two digits of exponent. */
            p = msml__wint(p, nde);
        } else {
            /* %f (or, shortly, %g in %f style) */
            if (prec < (size_t)(0x3f & -(int32_t)ndlo) * 9) {
                /* Precision is sufficiently low as to maybe require rounding. */
                ndhi = nd_add_m10e(nd, ndhi, 5, 0 - prec - 1);
            }
            g_format_like_f:
            if ((sf & MSML_FMT_T_FP_E) && !(sf & MSML_FMT_F_ALT) && prec && width) {
                /* Decrease precision in order to strip trailing zeroes. */
                if (ndlo) {
                    /* nd has a fractional part; we need to look at its digits. */
                    char tail[9];
                    uint32_t maxprec = (64 - ndlo) * 9;
                    if (prec >= maxprec) prec = maxprec;
                    else ndlo = 64 - (prec + 8) / 9;
                    i = prec - ((63 - ndlo) * 9);
                    msml__wuint9(tail, nd[ndlo]);
                    while (prec && tail[--i] == '0') {
                        prec--;
                        if (!i) {
                            if (ndlo == 63) { prec = 0; break; }
                            msml__wuint9(tail, nd[++ndlo]);
                            i = 9;
                        }
                    }
                } else {
                    /* nd has no fractional part, so precision goes straight to zero. */
                    prec = 0;
                }
            }
            len = ndhi * 9 + msml__ndigits_dec(nd[ndhi]) + prec + (prefix != 0)
                  + ((prec | (sf & MSML_FMT_F_ALT)) != 0);
            if (!(sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO))) {
                while (width-- > len) *p++ = ' ';
            }
            if (prefix) *p++ = prefix;
            if ((sf & (MSML_FMT_F_LEFT | MSML_FMT_F_ZERO)) == MSML_FMT_F_ZERO) {
                while (width-- > len) *p++ = '0';
            }
            /* Emit integer part. */
            p = msml__wint(p, nd[ndhi]);
            i = ndhi;
            while (i) p = msml__wuint9(p, nd[--i]);
            if ((prec | (sf & MSML_FMT_F_ALT))) {
                /* Emit fractional part. */
                *p++ = '.';
                /* Emit chunks of 9 digits (this may emit 8 digits too many). */
                while ((int32_t)prec > 0 && i != ndlo) {
                    i = (i - 1) & 0x3f;
                    p = msml__wuint9(p, nd[i]);
                    prec -= 9;
                }
                if ((sf & MSML_FMT_T_FP_E) && !(sf & MSML_FMT_F_ALT)) {
                    /* %g (and not %#g) - strip trailing zeroes. */
                    p += (int32_t)prec & ((int32_t)prec >> 31);
                    while (p[-1] == '0') p--;
                    if (p[-1] == '.') p--;
                } else {
                    /* %f (or %#g) - emit trailing zeroes. */
                    while ((int32_t)prec > 0) { *p++ = '0'; prec--; }
                    p += (int32_t)prec;
                }
            }
        }
    }
    if ((sf & MSML_FMT_F_LEFT)) while (width-- > len) *p++ = ' ';
    return p;
}

msml_ctx_t* msml_tensor_get_ctx(const msml_tensor_t* tensor) {
    return tensor->ctx;
}

msml_tensor_t* msml_tensor_create(msml_ctx_t* ctx, msml_dtype_t type, const int64_t* dims, int64_t rank, msml_tensor_t* view, size_t view_offs) {
    msml_assert(dims != NULL && rank > -1 && rank <= MSML_MAX_DIMS, "Rank must be within (0, %d]", MSML_MAX_DIMS);
    if (view && view->view) { /* Accumulate relative view offset. */
        view_offs += view->view_offs;
        view = view->view;
    }
    int64_t scalar_size = msml_get_dtype_info(type)->size;
    int64_t buf_size = scalar_size;
    for (int64_t i=0; i < rank; ++i) { /* Calculate buffer size and check for overflow. */
        msml_assert(dims[i] > 0, "Dimension must be > 0: %lld", dims[i]);
        msml_assert(!msml__imull64_ov(msml_max(1, dims[i]), buf_size, &buf_size), "Overflow in buffer size. Max: INT64_MAX. Reduce dimensions.");
    }
    msml_assert2(!view || !buf_size || buf_size + view_offs <= view->buf_size); /* Slice must be within viewed tensor data range. */
    msml_tensor_t* tensor = (msml_tensor_t*)msml_ctx_pool_alloc(ctx, sizeof(*tensor) + (view ? 0 : buf_size)); /* Allocate memory for tensor struct and data */
    memset(tensor, 0, sizeof(*tensor));
    tensor->ctx = ctx;
    tensor->rank = rank;
    tensor->dtype = type;
    tensor->buf_size = buf_size;
    tensor->view = view;
    tensor->view_offs = view_offs;
    for (int i=0; i < MSML_MAX_DIMS; ++i) /* Copy dimensions and set unused to identity. */
        tensor->shape[i] = i < rank ? msml_max(1, dims[i]) : 1;
    *tensor->strides = scalar_size;
    for (int i=1; i < MSML_MAX_DIMS; ++i) { /* Calculate strides and check for overflow. */
        msml_assert(!msml__imull64_ov(tensor->strides[i-1], tensor->shape[i-1], tensor->strides+i), "Overflow in stride calculation. Max: INT64_MAX. Reduce dimensions.");
    }
    tensor->buf = view ? (uint8_t*)view->buf + view_offs : (uint8_t*)(tensor + 1); /* Set buffer pointer to the end of the tensor struct, where data follows */
    return tensor;
}

msml_tensor_t* msml_tensor_create_1d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1}, 1, NULL, 0);
}

msml_tensor_t* msml_tensor_create_2d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2}, 2, NULL, 0);
}

msml_tensor_t* msml_tensor_create_3d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2, d3}, 3, NULL, 0);
}

msml_tensor_t* msml_tensor_create_4d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2, d3, d4}, 4, NULL, 0);
}

msml_tensor_t* msml_tensor_emit_op(msml_op_t op, msml_tensor_t** inputs, uint32_t n_inputs) {
    if (msml_unlikely(op == MSML_OP_NOP || n_inputs == 0 || n_inputs > MSML_MAX_INPUT_TENSORS)) {
        return NULL;
    }
    if (msml_unlikely(msml_op_get_argcount(op) != n_inputs)) {
        msml_log_error("Missing inputs");
        return NULL;
    }
    for (uint32_t i=0; i < n_inputs; ++i) { /* Make sure all required argument are not null. */
        if (!inputs[i]) {
            msml_log_error("Missing args");
            return NULL;
        }
    }
    msml_tensor_t* result;
    if (op == MSML_OP_MATMUL) {
        const int64_t shape[MSML_MAX_DIMS] = {inputs[0]->shape[1], inputs[1]->shape[1], inputs[1]->shape[2], inputs[1]->shape[3]};
        result = msml_tensor_create(inputs[0]->ctx, MSML_DTYPE_F32, shape, 4, NULL, 0);
    } else {
        result = msml_tensor_isomorphic(inputs[0]);
    }
    result->op = op;
    for (uint32_t i=0; i < n_inputs; ++i) { /* Make sure all required argument are not null. */
        result->inputs[i] = inputs[i];
    }
    return (*msml__op_get_validator_routine(op))(result, true) ? result : NULL;
}

msml_tensor_t* msml_tensor_isomorphic(msml_tensor_t* tensor) {
    msml_tensor_t* isomorph = msml_tensor_create(tensor->ctx, tensor->dtype, tensor->shape, tensor->rank, NULL, 0);
    msml_tensor_fmt_name(isomorph, "%s (isomorph)", tensor->name);
    return isomorph;
}

msml_tensor_t* msml_tensor_clone(msml_tensor_t* tensor) {
    msml_tensor_t* clone = msml_tensor_isomorphic(tensor);
    msml_tensor_set_op(clone, MSML_OP_CLONE);
    msml_tensor_set_arg(clone, 0, tensor);
    msml_tensor_fmt_name(clone, "%s (clone)", tensor->name);
    return clone;
}

msml_tensor_t* msml_tensor_view(msml_tensor_t* tensor) {
    msml_tensor_t* view = msml_tensor_create(tensor->ctx, tensor->dtype, tensor->shape, tensor->rank, tensor, 0);
    msml_tensor_fmt_name(view, "%s (view)", tensor->name);
    return view;
}

msml_tensor_t* msml_tensor_transpose(msml_tensor_t* tensor) {
    msml_tensor_t* transposed = msml_tensor_view(tensor);
    msml_tensor_set_op(transposed, MSML_OP_TRANSPOSE);
    msml_tensor_set_arg(transposed, 0, tensor);
    msml_swap(int64_t, transposed->shape[0], transposed->shape[1]);
    msml_swap(int64_t, transposed->strides[0], transposed->strides[1]);
    msml_tensor_fmt_name(transposed, "%s (transposed)", tensor->name);
    return transposed;
}

msml_tensor_t* msml_tensor_get_arg(const msml_tensor_t* tensor, size_t slot) {
    msml_assert(slot < MSML_MAX_INPUT_TENSORS, "Slot must be within [0, %d)", MSML_MAX_INPUT_TENSORS);
    return tensor->inputs[slot];
}

void msml_tensor_set_arg(msml_tensor_t* tensor, size_t slot, msml_tensor_t* arg) {
    msml_assert(slot < MSML_MAX_INPUT_TENSORS, "Slot must be within [0, %d)", MSML_MAX_INPUT_TENSORS);
    msml_assert(tensor->inputs[slot] == NULL, "Argument at slot #%zu already set", slot);
    tensor->inputs[slot] = arg;
}

msml_op_t msml_tensor_get_op(const msml_tensor_t* tensor) {
    return tensor->op;
}

void msml_tensor_set_op(msml_tensor_t* tensor, msml_op_t op) {
    tensor->op = op;
}

void msml_tensor_copy_buffer_from(msml_tensor_t* tensor, const void* data, size_t size) {
    msml_assert(size == (size_t)tensor->buf_size, "Buffer size mismatch: %zu != %lld", size, tensor->buf_size);
    memcpy(tensor->buf, data, size);
}

void msml_tensor_fill(msml_tensor_t* tensor, float x) {
    if (x == 0.0f) {
        memset(tensor->buf, 0, tensor->buf_size);
        return;
    }
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = (float*)tensor->buf;
            for (int64_t i=0; i < n; ++i) buf[i] = x;
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
}

void msml_tensor_fill_random(msml_tensor_t* tensor, float min, float max) {
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = (float*)tensor->buf;
            msml__prng_generate_n(tensor->ctx, buf, n, min, max);
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
}

size_t msml_tensor_get_memory_usage(const msml_tensor_t* tensor) {
    return sizeof(*tensor) + tensor->buf_size;
}

void msml_tensor_print(const msml_tensor_t* tensor, bool with_data) {
    msml_assert(tensor->dtype == MSML_DTYPE_F32, "Tensor must be F32");
    double buf_size_cvt = 0.0;
    const char* buf_size_unit = NULL;
    msml__humanize_memory_size(msml_tensor_get_memory_usage(tensor), &buf_size_cvt, &buf_size_unit);
    printf("Tensor '%s', DType: %s, Rank: %zu, Shape: [%zu, %zu, %zu, %zu], Strides: [%zu, %zu, %zu, %zu], Mem: %.03f %s \n",
       tensor->name,
       msml_get_dtype_info(tensor->dtype)->name,
       (size_t)tensor->rank,
       (size_t)tensor->shape[0],
       (size_t)tensor->shape[1],
       (size_t)tensor->shape[2],
       (size_t)tensor->shape[3],
       (size_t)tensor->strides[0],
       (size_t)tensor->strides[1],
       (size_t)tensor->strides[2],
       (size_t)tensor->strides[3],
       buf_size_cvt,
       buf_size_unit
    );
    if (with_data) {
        printf("[\n");
        const float* buf = (const float*)tensor->buf;
        for (int64_t i3=0; i3 < tensor->shape[2]; ++i3) { // TODO: d4
            printf("[\n");
            for (int64_t i2=0; i2 < tensor->shape[1]; ++i2) {
                putchar('\t');
                for (int64_t i1=0; i1 < tensor->shape[0]; ++i1) {
                    // TODO: dtype check
                    float x = buf[i3 * tensor->shape[1] * tensor->shape[0] + i2 * tensor->shape[0] + i1];
                    char fmt_buf[128];
                    *msml__fmt_f64(MSML_FMT_G14, x, fmt_buf) = '\0';
                    printf("%s ", fmt_buf);
                }
                putchar('\n');
            }
            printf("]\n");
        }
        printf("]\n");
    }
}

void msml_tensor_set_name(msml_tensor_t* tensor, const char* name) {
    strncpy(tensor->name, name, MSML_MAX_TENSOR_NAME_LEN);
    tensor->name[MSML_MAX_TENSOR_NAME_LEN-1] = '\0';
}

void msml_tensor_fmt_name(msml_tensor_t* tensor, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(tensor->name, sizeof(tensor->name), fmt, args);
    va_end(args);
}

const char* msml_tensor_get_name(const msml_tensor_t* tensor) {
    return tensor->name;
}

int64_t msml_tensor_rank(const msml_tensor_t* tensor) {
    return tensor->rank;
}

const int64_t* msml_tensor_shape(const msml_tensor_t* tensor) {
    return tensor->shape;
}

const int64_t* msml_tensor_strides(const msml_tensor_t* tensor) {
    return tensor->strides;
}

msml_dtype_t msml_tensor_dtype(const msml_tensor_t* tensor) {
    return tensor->dtype;
}

void* msml_tensor_buf(const msml_tensor_t* tensor) {
    return tensor->buf;
}

float* msml_tensor_buf_f32(const msml_tensor_t* tensor) {
    msml_assert(tensor->dtype == MSML_DTYPE_F32, "Tensor data type must be F32, not %s", msml_get_dtype_info(tensor->dtype)->name);
    return (float*)tensor->buf;
}

int64_t msml_tensor_buf_size(const msml_tensor_t* tensor) {
    return tensor->buf_size;
}

int64_t msml_tensor_buf_len(const msml_tensor_t* tensor) {
    return tensor->buf_size / msml_get_dtype_info(tensor->dtype)->size;
}

int64_t msml_tensor_num_rows(const msml_tensor_t* tensor) {
    int64_t rows=tensor->shape[1];
    for (int64_t i=2; i < MSML_MAX_DIMS; ++i)
        rows *= tensor->shape[i];
    return rows;
}

int64_t msml_tensor_num_cols(const msml_tensor_t* tensor) {
    return tensor->shape[0];
}

bool msml_tensor_is_scalar(const msml_tensor_t* tensor) {
    for (int i=0; i < MSML_MAX_DIMS; ++i)
        if (tensor->shape[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_vector(const msml_tensor_t* tensor) {
    for (int i=1; i < MSML_MAX_DIMS; ++i)
        if (tensor->shape[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_matrix(const msml_tensor_t* tensor) {
    for (int i=2; i < MSML_MAX_DIMS; ++i)
        if (tensor->shape[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_higher_order_3d(const msml_tensor_t* tensor) {
    for (int i=3; i < MSML_MAX_DIMS; ++i)
        if (tensor->shape[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_shape_eq(const msml_tensor_t* a, const msml_tensor_t* b) {
    return memcmp(a->shape, b->shape, sizeof(a->shape)) == 0;
}

bool msml_tensor_are_strides_eq(const msml_tensor_t* a, const msml_tensor_t* b) {
    return memcmp(a->strides, b->strides, sizeof(a->strides)) == 0;
}

bool msml_tensor_can_broadcast(const msml_tensor_t* a, const msml_tensor_t* b) {
    for (int i=0; i < MSML_MAX_DIMS; ++i)
        if ((b->shape[i] % a->shape[i]) != 0)
            return false;
    return true;
}

bool msml_tensor_is_transposed(const msml_tensor_t* tensor) {
    return tensor->strides[0] > tensor->strides[1];
}

void msml_tensor_virtual_to_physical_index(const msml_tensor_t* tensor, int64_t v_idx, int64_t(*p_idx)[MSML_MAX_DIMS]) {
    msml__load_local_storage_group(tensor, d, shape);
    (*p_idx)[3] = v_idx / (d2*d1*d0);
    (*p_idx)[2] = (v_idx - (*p_idx)[3]*d2*d1*d0) / (d1*d0);
    (*p_idx)[1] = (v_idx - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0) / d0;
    (*p_idx)[0] =  v_idx - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0 - (*p_idx)[1]*d0;
}

int64_t msml_tensor_physical_to_virtual_index(const msml_tensor_t* tensor, const int64_t (*p_idx)[MSML_MAX_DIMS]) {
    int64_t v_idx = 0;
    for (int i=0; i < MSML_MAX_DIMS; ++i)
        v_idx += (*p_idx)[i] * tensor->strides[i];
    return v_idx;
}

bool msml_tensor_is_contiguous(const msml_tensor_t* tensor) {
    return *tensor->strides == msml_get_dtype_info(tensor->dtype)->size;
}

float msml_tensor_get_scalar_physical_index(const msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    const uint8_t* dst = msml__resolve_physical_ptr(tensor, d0, d1, d2, d3);
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: return *(float*)dst;
        default: msml_panic("Unsupported data type: %s", msml_get_dtype_info(tensor->dtype)->name);
    }
}

void msml_tensor_set_scalar_physical_index(msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3, float x) {
    uint8_t* dst = msml__resolve_physical_ptr(tensor, d0, d1, d2, d3);
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: *(float*)dst = x; break;
        default: msml_panic("Unsupported data type: %s", msml_get_dtype_info(tensor->dtype)->name);
    }
}

float msml_tensor_get_scalar_virtual_index(const msml_tensor_t* tensor, int64_t v_idx) {
    if (!msml_tensor_is_contiguous(tensor)) {
        int64_t physical_idx[MSML_MAX_DIMS];
        msml_tensor_virtual_to_physical_index(tensor, v_idx, &physical_idx);
        return msml_tensor_get_scalar_physical_index(tensor, physical_idx[0], physical_idx[1], physical_idx[2], physical_idx[3]);
    }
    switch (tensor->dtype) {
        case MSML_DTYPE_F32:
            return ((const float*)tensor->buf)[v_idx];
        default:
            msml_panic("Unsupported data type: %s", msml_get_dtype_info(tensor->dtype)->name);
    }
}

void msml_tensor_set_scalar_virtual_index(msml_tensor_t* tensor, int64_t v_idx, float x) {
    if (!msml_tensor_is_contiguous(tensor)) {
        int64_t physical_idx[MSML_MAX_DIMS];
        msml_tensor_virtual_to_physical_index(tensor, v_idx, &physical_idx);
        msml_tensor_set_scalar_physical_index(tensor, physical_idx[0], physical_idx[1], physical_idx[2], physical_idx[3], x);
        return;
    }
    switch (tensor->dtype) {
        case MSML_DTYPE_F32:
            ((float*)tensor->buf)[v_idx] = x;
            break;
        default:
            msml_panic("Unsupported data type: %s", msml_get_dtype_info(tensor->dtype)->name);
    }
}

bool msml_tensor_eq(const msml_tensor_t* a, const msml_tensor_t* b) {
    if (a->dtype != b->dtype) return false;
    if (a->rank != b->rank) return false;
    if (memcmp(a->shape, b->shape, sizeof(a->shape)) != 0) return false;
    if (a->buf_size != b->buf_size) return false;
    int64_t n = msml_tensor_buf_len(a);
    switch (a->dtype) {
        case MSML_DTYPE_F32: {
            const float* buf_a = (const float*)a->buf;
            const float* buf_b = (const float*)b->buf;
            for (int64_t i = 0; i < n; ++i) {
                if (buf_a[i] != buf_b[i]) {
                    return false;
                }
            }
        } break;
        default:
            msml_panic("Unsupported data type: %s", msml_get_dtype_info(a->dtype)->name);
    }
    return true;
}

bool msml_tensor_isclose(const msml_tensor_t* a, const msml_tensor_t* b, float eps, double* percent_eq) {
    if (a->dtype != b->dtype) return false;
    if (a->rank != b->rank) return false;
    if (memcmp(a->shape, b->shape, sizeof(a->shape)) != 0) return false;
    if (a->buf_size != b->buf_size) return false;
    eps = eps < 0.0f ? FLT_EPSILON : eps;
    int64_t n = msml_tensor_buf_len(a);
    int64_t n_eq = 0;
    switch (a->dtype) {
        case MSML_DTYPE_F32: {
            const float* buf_a = (const float*)a->buf;
            const float* buf_b = (const float*)b->buf;
            for (int64_t i = 0; i < n; ++i)  /* |x - y| <= ε     ∀ x, y ∈ A, B */
                if (fabsf(buf_a[i] - buf_b[i]) <= eps) ++n_eq;
        } break;
        default:
            msml_panic("Unsupported data type: %s", msml_get_dtype_info(a->dtype)->name);
    }
    if (percent_eq) *percent_eq = (double)n_eq / (double)n * 100.0;
    return n_eq == n;
}

/* CPU BLAS impl */
#define MSML__GELU_COEFF 0.044715f

static void MSML_HOTPROC msml__vadd_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] + y[i];
    }
}

static void MSML_HOTPROC msml__vsub_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] - y[i];
    }
}

static void MSML_HOTPROC msml__vmul_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] * y[i];
    }
}

static void MSML_HOTPROC msml__vdiv_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] / y[i];
    }
}

static float MSML_UNUSED MSML_HOTPROC msml__vdot_f32(
    const int64_t n,
    const float* const x,
    const float* const y
) {
    #ifdef __ARM_NEON
        #define STEP 16ll
        const int64_t k = n & -STEP;
        float32x4_t acc[4] = {vdupq_n_f32(0)};
        float32x4_t vx[4];
        float32x4_t vy[4];
        for (int64_t i=0; i < k; i += STEP) { /* Process STEP elements at a time */
            #pragma GCC unroll 4
            for (int64_t j=0; j < 4; ++j) { /* Unrolled inner loop */
                vx[j] = vld1q_f32(x+i+(j<<2));
                vy[j] = vld1q_f32(y+i+(j<<2));
                acc[j] = vfmaq_f32(acc[j], vx[j], vy[j]); /* (FMA) Fused multiply-accumulate */
            }
        }
        acc[1] = vaddq_f32(acc[1], acc[3]); /* Fold acc[1] += acc[3] */
        *acc = vaddq_f32(*acc, acc[2]);     /* Fold acc[0] += acc[2] */
        *acc = vaddq_f32(*acc, acc[1]);     /* Fold acc[0] += acc[1] */
        float sum = vaddvq_f32(*acc);       /* Reduce to scalar with horizontal sum. */
        for (int64_t i=k; i < n; ++i) {     /* Process leftovers scalar-wise */
            sum += x[i]*y[i];
        }
        return sum;
        #undef STEP
    #else
        double r = 0.0;
        for (int64_t i=0; i < n; ++i) {
            r += x[i] * y[i];
        }
        return (float)r;
    #endif
}

static void MSML_HOTPROC msml__vstep_f32( /* Heaviside step function. */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] >= 0.0f ? 1.0f : 0.0f;
    }
}

static void MSML_HOTPROC msml__vsoftmax_f32( /* softmax : ℝ -> (0, ∞), x |-> e^x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = expf(x[i]); /* e^x */
    }
}

static void MSML_HOTPROC msml__vsoftmax_dv_f32( /* softmax' = softmax : ℝ -> (0, ∞), x |-> e^x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    return msml__vsoftmax_f32(n, o, x);
}

static void MSML_HOTPROC msml__vsigmoid_f32( /* σ : ℝ -> (0, 1), x |-> 1/(1 + e^(-x)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

static void MSML_HOTPROC msml__vsigmoid_dv_f32( /* σ' : ℝ -> (0, 1), x |-> -(e^x / ((e^x + 1)^2)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        const float e_x = expf(x[i]);
        const float e_x1 = e_x + 1.0f;
        o[i] = -(e_x / (e_x1*e_x1));
    }
}

static void MSML_HOTPROC msml__vhard_sigmoid_f32( /* σ^ : ℝ -> (0, 1), x |-> min(1, max(0, (x + 3)/6)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
    }
}

static void MSML_HOTPROC msml__vhard_sigmoid_dv_f32( /* σ^ : ℝ -> (0, 1), x |-> min(1, max(0, (x + 3)/6)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        msml_panic("NYI!");
    }
}

static void MSML_HOTPROC msml__vsilu_f32( /* silu : ℝ -> ℝ, x |-> x/(1 + e^(-x)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static void MSML_HOTPROC msml__vsilu_dv_f32( /* silu' : ℝ -> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        msml_panic("NYI!");
    }
}

static void MSML_HOTPROC msml__vtanh_f32( /* tanh : ℝ -> (-1, 1), x |-> tanh x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = tanhf(x[i]);
    }
}

static void MSML_HOTPROC msml__vtanh_dv_f32( /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        const float cx = coshf(x[i]);
        o[i] = 1.0f / (cx*cx);
    }
}

static void MSML_HOTPROC msml__vrelu_f32( /* relu : ℝ -> ℝ^+, x |-> max {x, 0} */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = msml_max(x[i], 0.0f);
    }
}

static void MSML_HOTPROC msml__vrelu_dv_f32( /* relu' : ℝ -> ℝ^+, x |-> { 0 if x < 0, UB if x = 0, else 1 */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] <= 0.0f ? 0.0f : 1.0f; /* relu' is mathematically undefined for x = 0, but we return 0 in this case. */
    }
}

static void MSML_HOTPROC msml__vgelu_f32( /* gelu : ℝ -> ℝ, x |-> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = 0.5f*x[i]*(1.0f + tanhf(0.79788456080286535587989211986876f*x[i]*(1.0f + MSML__GELU_COEFF*x[i]*x[i])));
    }
}

static void MSML_HOTPROC msml__vgelu_dv_f32( /* gelu' : ℝ -> ℝ, x |-> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        msml_panic("NYI"); /* TODO */
    }
}

struct msml__blas_compute_info_t {
    msml_ctx_t* ctx;
    int64_t n_threads;
    int64_t thread_idx;
};

static void msml__blas_nop(
    const msml__blas_compute_info_t* const bci,
    msml_tensor_t* const r,
    const msml_tensor_t** const inputs
) {
    (void)bci;
    (void)r;
    (void)inputs;
}

static void msml__blas_clone(
    const msml__blas_compute_info_t* const bci,
    msml_tensor_t* const r,
    const msml_tensor_t** const inputs
) {
    const msml_tensor_t* const x = inputs[0];
    msml_assert2(msml_tensor_is_shape_eq(x, r));
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    msml__load_local_storage_group(r, r_d, shape)
    msml__load_local_storage_group(r, r_s, strides)
    msml__load_local_storage_group(x, x_d, shape)
    msml__load_local_storage_group(x, x_s, strides)
    const int64_t ti = bci->thread_idx;
    const int64_t tc = bci->n_threads;
    if (msml_tensor_is_contiguous(r) && msml_tensor_is_contiguous(x)) { /* Fast path for contiguous input and output tensors. */
        const int64_t ne = msml_tensor_buf_len(x);
        const int64_t rc = (ne+tc-1) / tc;
        const int64_t rs = rc * ti;
        const int64_t re = msml_min(rs + rc, ne);
        if (msml_likely(rs < re)) memcpy(b_r + rs*sizeof(float), b_x + rs*sizeof(float), (re - rs)*sizeof(float));
        return;
    }
    const int64_t nr = x_d1;
    const int64_t dr = (nr+tc-1) / tc;
    const int64_t rs = dr * ti;
    const int64_t re = msml_min(rs + dr, nr);
    if (x_d0 == r_d0 && x_s0 == sizeof(float) && r_s0 == sizeof(float)) {
        const int64_t rw = x_d0 * (int64_t)sizeof(float);
        for (int64_t i3 = 0; i3 < x_d3; ++i3) {
            for (int64_t i2 = 0; i2 < x_d2; ++i2) {
                for (int64_t i1 = rs; i1 < re; ++i1) {
                    memcpy(
                        b_r + i1*r_s1 + i2*r_s2 + i3*r_s3,
                        b_x + i1*x_s1 + i2*x_s2 + i3*x_s3,
                        rw
                    );
                }
            }
        }
        return;
    }
    if (msml_tensor_is_contiguous(r)) { /* Fast path for contiguous output tensor. */
        int64_t id = 0;
        const int64_t rw = x_d0 * (int64_t)sizeof(float);
        if (x_d0 == sizeof(float)) {
            for (int64_t i3 = 0; i3 < x_d3; ++i3) {
                for (int64_t i2 = 0; i2 < x_d2; ++i2) {
                    id += rw * rs;
                    for (int64_t i1 = rs; i1 < re; ++i1) {
                        const uint8_t* const p_x = b_x + i1*x_s1 + i2*x_s2 + i3*x_s3;
                        memcpy(b_r + id, p_x, rw);
                        id += rw;
                    }
                    id += rw * (x_d1 - re);
                }
            }
        } else {
            for (int64_t i3 = 0; i3 < x_d3; ++i3) {
                for (int64_t i2 = 0; i2 < x_d2; ++i2) {
                    id += rw * rs;
                    for (int64_t i1 = rs; i1 < re; ++i1) {
                        for (int64_t i0 = 0; i0 < x_d0; i0++) {
                            const uint8_t* const p_x = b_x + i0*x_s0 + i1*x_s1 + i2*x_s2 + i3*x_d3;
                            memcpy(b_r + id, p_x, sizeof(float));
                            id += sizeof(float);
                        }
                    }
                    id += rw * (x_d1 - re);
                }
            }
        }
        return;
    }
    int64_t r_i0 = 0, r_i1 = 0, r_i2 = 0, r_i3 = 0;
    for (int64_t i3 = 0; i3 < x_d3; ++i3) {
        for (int64_t i2 = 0; i2 < x_d2; ++i2) {
            r_i0 += x_d0 * rs;
            while (r_i0 >= r_d0) {
                r_i0 -= r_d0;
                if (++r_i1 == r_d1) {
                    r_i1 = 0;
                    if (++r_i2 == r_d2) {
                        r_i2 = 0;
                        if (++r_i3 == r_d3) {
                            r_i3 = 0;
                        }
                    }
                }
            }
            for (int64_t i1 = rs; i1 < re; i1++) {
                for (int64_t i0 = 0; i0 < x_d0; i0++) {
                    *(float*)((b_x + i0*x_s0 + i1*x_s1 + i2*x_s2 + i3*x_s3)) = *(const float*)(b_r + r_i0*r_s0 + r_i1*r_s1 + r_i2*r_s2 + r_i3*r_s3);
                    if (++r_i0 == r_d0) {
                        r_i0 = 0;
                        if (++r_i1 == r_d1) {
                            r_i1 = 0;
                            if (++r_i2 == r_d2) {
                                r_i2 = 0;
                                if (++r_i3 == r_d3) {
                                    r_i3 = 0;
                                }
                            }
                        }
                    }
                }
            }
            r_i0 += x_d0 * (x_d1 - re);
            while (r_i0 >= r_d0) {
                r_i0 -= r_d0;
                if (++r_i1 == r_d1) {
                    r_i1 = 0;
                    if (++r_i2 == r_d2) {
                        r_i2 = 0;
                        if (++r_i3 == r_d3) {
                            r_i3 = 0;
                        }
                    }
                }
            }
        }
    }
}

static void msml__blas_transpose(
    const msml__blas_compute_info_t* const bci,
    msml_tensor_t* const r,
    const msml_tensor_t** const inputs
) {
    (void)bci;
    (void)r;
    (void)inputs;
}

#define msml__blas_impl_unary_op(name, T, vec_op) \
    static void MSML_HOTPROC msml__blas_##name( \
        const msml__blas_compute_info_t* const bci, \
        msml_tensor_t* const r, \
        const msml_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */ \
    ) { \
        (void)bci; \
        const msml_tensor_t* const x = inputs[0]; \
        uint8_t* const b_r = (uint8_t*)r->buf; \
        const uint8_t* const b_x = (const uint8_t*)x->buf; \
        msml__load_local_storage_group(r, r_s, strides) \
        msml__load_local_storage_group(x, x_s, strides) \
        const int64_t rc = msml_tensor_num_rows(x); \
        const int64_t cc = msml_tensor_num_cols(x); \
        for (int64_t ri=0; ri < rc; ++ri) { \
            vec_op(cc, (T*)(b_r + ri*r_s1), (const T*)(b_x + ri*x_s1)); \
        } \
    }

msml__blas_impl_unary_op(step_f32, float, msml__vstep_f32)
msml__blas_impl_unary_op(softmax_f32, float, msml__vsoftmax_f32)
msml__blas_impl_unary_op(softmax_dv_f32, float, msml__vsoftmax_dv_f32)
msml__blas_impl_unary_op(sigmoid_f32, float, msml__vsigmoid_f32)
msml__blas_impl_unary_op(sigmoid_dv_f32, float, msml__vsigmoid_dv_f32)
msml__blas_impl_unary_op(hard_sigmoid_f32, float, msml__vhard_sigmoid_f32)
msml__blas_impl_unary_op(hard_sigmoid_dv_f32, float, msml__vhard_sigmoid_dv_f32)
msml__blas_impl_unary_op(silu_f32, float, msml__vsilu_f32)
msml__blas_impl_unary_op(silu_dv_f32, float, msml__vsilu_dv_f32)
msml__blas_impl_unary_op(tanh_f32, float, msml__vtanh_f32)
msml__blas_impl_unary_op(tanh_dv_f32, float, msml__vtanh_dv_f32)
msml__blas_impl_unary_op(relu_f32, float, msml__vrelu_f32)
msml__blas_impl_unary_op(relu_dv_f32, float, msml__vrelu_dv_f32)
msml__blas_impl_unary_op(gelu_f32, float, msml__vgelu_f32)
msml__blas_impl_unary_op(gelu_dv_f32, float, msml__vgelu_dv_f32)

#undef msml__blas_impl_unary_op

/*
** const int64_t x_i3 = ri / (x_d2 * x_d1);
** const int64_t x_i2 = (ri / x_d1) % x_d2;
** const int64_t x_i1 = ri % x_d1;
*/

#define msml__blas_impl_binary_op(name, T, vec_op, scalar_op) \
    static void MSML_HOTPROC msml__blas_##name( \
        const msml__blas_compute_info_t* const bci, \
        msml_tensor_t* const r, \
        const msml_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */ \
    ) { \
        const msml_tensor_t* const x = inputs[0]; \
        const msml_tensor_t* const y = inputs[1]; \
        uint8_t* const b_r = (uint8_t*)r->buf; \
        const uint8_t* const b_x = (const uint8_t*)x->buf; \
        const uint8_t* const b_y = (const uint8_t*)y->buf; \
        msml__load_local_storage_group(r, r_d, shape) \
        msml__load_local_storage_group(r, r_s, strides) \
        msml__load_local_storage_group(x, x_d, shape) \
        msml__load_local_storage_group(x, x_s, strides) \
        msml__load_local_storage_group(y, y_d, shape) \
        msml__load_local_storage_group(y, y_s, strides) \
        const int64_t rc = msml_tensor_num_rows(x);  \
        const int64_t ti = bci->thread_idx;  \
        const int64_t tc = bci->n_threads;  \
        const int64_t rpt = (rc + tc - 1)/tc;  \
        const int64_t rs = rpt * ti;  \
        const int64_t re = msml_min(rs+rpt, rc); \
        if (y_s0 == sizeof(T)) { \
            for (int64_t ri=rs; ri < re; ++ri) { \
                const int64_t x_i3 = ri / (x_d2*x_d1); \
                const int64_t x_i2 = (ri - x_i3*x_d2*x_d1)/x_d1; \
                const int64_t x_i1 = ri - x_i3*x_d2*x_d1 - x_i2*x_d1; \
                const int64_t y_i3 = x_i3 % y_d3; \
                const int64_t y_i2 = x_i2 % y_d2; \
                const int64_t y_i1 = x_i1 % y_d1; \
                T* const p_r = (T*)(b_r + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1); \
                const T* const p_x = (const T*)(b_x + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1); \
                const T* const p_y = (const T*)(b_y + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1); \
                const int64_t pa = x_d0 / y_d0; \
                for (int64_t i=0; i < pa; ++i) { \
                    vec_op(y_d0, p_r + i*y_d0, p_x + i*y_d0, p_y); \
                } \
            } \
        } else { \
            for (int64_t ri=rs; ri < re; ++ri) { \
                const int64_t x_i3 = ri / (x_d2*x_d1); \
                const int64_t x_i2 = (ri - x_i3*x_d2*x_d1)/x_d1; \
                const int64_t x_i1 = ri - x_i3*x_d2*x_d1 - x_i2*x_d1; \
                const int64_t y_i3 = x_i3 % y_d3; \
                const int64_t y_i2 = x_i2 % y_d2; \
                const int64_t y_i1 = x_i1 % y_d1; \
                T* const p_r = (T*)(b_r + x_i3*r_s3 + x_i2*r_s2 + x_i1*r_s1); \
                const T* const p_x = (const T*)(b_x + x_i3*x_s3 + x_i2*x_s2 + x_i1*x_s1); \
                for (int64_t i=0; i < r_d0; ++i) { \
                    p_r[i] = p_x[i] scalar_op *(const T*)(b_y + y_i3*y_s3 + y_i2*y_s2 + y_i1*y_s1 + i%y_d0*y_s0); \
                } \
            } \
        } \
    }

msml__blas_impl_binary_op(add_f32, float, msml__vadd_f32, +)
msml__blas_impl_binary_op(sub_f32, float, msml__vsub_f32, -)
msml__blas_impl_binary_op(mul_f32, float, msml__vmul_f32, *)
msml__blas_impl_binary_op(div_f32, float, msml__vdiv_f32, /)

#undef msml__blas_impl_binary_op

#define MSML_MATMUL_BLK_X 16 /* Block size X for matrix multiplication */
#define MSML_MATMUL_BLK_Y 16 /* Block size Y for matrix multiplication */
#define MSML_MATMUL_USE_TMP_NON_SHARED_STORAGE 1 /* Use temporary storage for matrix multiplication to reduce false sharing. See: https://en.wikipedia.org/wiki/False_sharing */

/*
** Matrix multiplication.
** Mathematically, matmul is defined as R = A x B
** For performance reasons, we compute: Rᵀ = A x Bᵀ.
*/
static void MSML_HOTPROC msml__blas_matmul_f32(
    const msml__blas_compute_info_t* const bci,
    msml_tensor_t* const r,
    const msml_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const msml_tensor_t* const x = inputs[0];
    const msml_tensor_t* const y = inputs[1];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    const uint8_t* const b_y = (const uint8_t*)y->buf;
    msml__load_local_storage_group(r, r_d, shape)
    msml__load_local_storage_group(r, r_s, strides)
    msml__load_local_storage_group(x, x_d, shape)
    msml__load_local_storage_group(x, x_s, strides)
    msml__load_local_storage_group(y, y_d, shape)
    msml__load_local_storage_group(y, y_s, strides)
    const int64_t ti = bci->thread_idx;
    const int64_t tc = bci->n_threads;
    const bool y_cont = msml_tensor_is_contiguous(y);
    int64_t chunk = 0; /* TODO: Atomic sync */
    //if (ti == 0) chunk = tc;
    /* TODO: barrier */
    const int64_t nr0 = r_d0;
    const int64_t nr1 = r_d1*r_d2*r_d3;
    const int64_t chunk_size = nr0 == 1 || nr1 == 1 ? 64 : 16;
    int64_t nchunk0 = (nr0 + chunk_size-1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size-1) / chunk_size;
    if (nchunk0 * nchunk1 < (tc<<2) || msml_ctx_is_numa_system(x->ctx)) {
        nchunk0 = nr0 > nr1 ? tc : 1;
        nchunk1 = nr0 > nr1 ? 1 : tc;
    }
    const int64_t cr0 = (nr0 + nchunk0 - 1) / nchunk0;
    const int64_t cr1 = (nr1 + nchunk1 - 1) / nchunk1;
    const int64_t nchunks = nchunk0 * nchunk1;
    int64_t current_chunk = ti;
    while (current_chunk < nchunks)  /* TODO: Atomic sync */ {
        const int64_t ci0 = current_chunk % nchunk0;
        const int64_t ci1 = current_chunk / nchunk0;
        const int64_t r0s = cr0 * ci0;
        const int64_t r0e = msml_min(r0s + cr0, nr0);
        const int64_t r1s = cr1 * ci1;
        const int64_t r1e = msml_min(r1s + cr1, nr1);
        const int64_t r2 = y_d2 / x_d2;
        const int64_t r3 = y_d3 / x_d3;
        if (msml_unlikely(r0s >= r0e || r1s >= r1e)) return; /* No work in this chunk */
        const int64_t row_size = y_d0*sizeof(float);
        #if MSML_MATMUL_USE_TMP_NON_SHARED_STORAGE
            float tmp[32];
        #endif
        for (int64_t i1 = r1s; i1 < r1e; i1 += MSML_MATMUL_BLK_Y) {
            for (int64_t i0 = r0s; i0 < r0e; i0 += MSML_MATMUL_BLK_X) {
                for (int64_t ri = i1; ri < i1 + MSML_MATMUL_BLK_Y && ri < r1e; ++ri) {
                    const int64_t y_i3 = (ri/(y_d2*r_d1));
                    const int64_t y_i2 = (ri - y_i3*y_d2*r_d1)/r_d1;
                    const int64_t y_i1 = (ri - y_i3*y_d2*r_d1 - y_i2*r_d1);
                    const int64_t x_i3 = y_i3 / r3;
                    const int64_t x_i2 = y_i2 / r2;
                    const int64_t r_i3 = y_i3;
                    const int64_t r_i2 = y_i2;
                    const int64_t r_i1 = y_i1;
                    const uint8_t* const x_row = b_x + x_i2*x_s2 + x_i3*x_s3;
                    const float* const y_col =
                        (const float*)(b_y + (y_cont
                        ? (y_i1 + y_i2*y_d1 + y_i3*y_d2*y_d1) * row_size
                        : (y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3)));
                    float* r_col = (float*)(b_r + r_i1*r_s1 + r_i2*r_s2 + r_i3*r_s3);
                    #if MSML_MATMUL_USE_TMP_NON_SHARED_STORAGE /* Use temporary storage to reduce false sharing. */
                        for (int64_t i = i0; i < i0 + MSML_MATMUL_BLK_X && i < r0e; ++i)
                            tmp[i-i0] = msml__vdot_f32(x_d0, (const float*)(x_row + i*x_s1), y_col);
                        memcpy(r_col+i0, tmp, (msml_min(i0 + MSML_MATMUL_BLK_X, r0e) - i0)*sizeof(float)); /* Store to result buffer. */
                    #else /* Store directly to result buffer. */
                        for (int64_t i = i0; i < i0 + MSML_MATMUL_BLK_X && i < r0e; ++i) {
                            r_col[i] = msml__vdot_f32(x_d0, (const float*) (x_row + i * x_s1), y_col);
                        }
                    #endif
                }
            }
        }
        if (tc >= nchunks) break;
        current_chunk = ++chunk;
    }
}

#undef MSML_MATMUL_BLK_Y
#undef MSML_MATMUL_BLK_X

bool msml_tensor_is_op_possible(const msml_tensor_t* tensor, bool print_error) {
    return (*(msml__op_get_validator_routine(tensor->op)))(tensor, print_error);
}

/* Dispatch table for default CPU-implementation. */
static void msml__blas_compute_dispatch_table_default(void (*(*const dispatch_lut)[MSML_OP__COUNT])(const msml__blas_compute_info_t*, msml_tensor_t*, const msml_tensor_t**)) {
    (*dispatch_lut)[MSML_OP_NOP] = &msml__blas_nop;
    (*dispatch_lut)[MSML_OP_TRANSPOSE] = &msml__blas_transpose;
    (*dispatch_lut)[MSML_OP_CLONE] = &msml__blas_clone;
    (*dispatch_lut)[MSML_OP_STEP] = &msml__blas_step_f32;
    (*dispatch_lut)[MSML_OP_SOFTMAX] = &msml__blas_softmax_f32;
    (*dispatch_lut)[MSML_OP_SOFTMAX_DV] = &msml__blas_softmax_dv_f32;
    (*dispatch_lut)[MSML_OP_SIGMOID] = &msml__blas_sigmoid_f32;
    (*dispatch_lut)[MSML_OP_SIGMOID_DV] = &msml__blas_sigmoid_dv_f32;
    (*dispatch_lut)[MSML_OP_HARD_SIGMOID] = &msml__blas_hard_sigmoid_f32;
    (*dispatch_lut)[MSML_OP_HARD_SIGMOID_DV] = &msml__blas_hard_sigmoid_dv_f32;
    (*dispatch_lut)[MSML_OP_SILU] = &msml__blas_silu_f32;
    (*dispatch_lut)[MSML_OP_SILU_DV] = &msml__blas_silu_dv_f32;
    (*dispatch_lut)[MSML_OP_TANH] = &msml__blas_tanh_f32;
    (*dispatch_lut)[MSML_OP_TANH_DV] = &msml__blas_tanh_dv_f32;
    (*dispatch_lut)[MSML_OP_RELU] = &msml__blas_relu_f32;
    (*dispatch_lut)[MSML_OP_RELU_DV] = &msml__blas_relu_dv_f32;
    (*dispatch_lut)[MSML_OP_GELU] = &msml__blas_gelu_f32;
    (*dispatch_lut)[MSML_OP_GELU_DV] = &msml__blas_gelu_dv_f32;
    (*dispatch_lut)[MSML_OP_ADD] = &msml__blas_add_f32;
    (*dispatch_lut)[MSML_OP_SUB] = &msml__blas_sub_f32;
    (*dispatch_lut)[MSML_OP_MUL] = &msml__blas_mul_f32;
    (*dispatch_lut)[MSML_OP_DIV] = &msml__blas_div_f32;
    (*dispatch_lut)[MSML_OP_MATMUL] = &msml__blas_matmul_f32;
}

static void msml__blas_compute_dispatch_table_install(msml_ctx_t* const ctx) {
    msml__blas_compute_dispatch_table_default(&ctx->blas_dispatch);
    /* TODO: Add support for custom implementations for host CPU arch. */
    for (int i=MSML_OP_NOP; i < MSML_OP__COUNT; ++i) { /* Verify that all ops have a implementation, except NOP. */
        msml_assert(ctx->blas_dispatch[i] != NULL, "No default CPU implementation for op: %s", msml_op_get_name((msml_op_t)i));
    }
}

static void MSML_HOTPROC msml__compute_dag_eval(const msml__blas_compute_info_t* bci, msml_tensor_t* node, bool forward);

static unsigned MSML_HOTPROC msml__process_parent_inputs(
    const msml_tensor_t*** const out_inputs,
    const msml__blas_compute_info_t* const bci,
    msml_tensor_t* const node,
    const bool forward
) {
    msml_tensor_t** inputs = node->inputs;
    const uint32_t n_inputs = msml_op_get_argcount(node->op);
    for (uint32_t i=0; i < n_inputs; ++i) { /* Eval parents and verify arguments */
        uint32_t idx = forward ? i : n_inputs-i-1; /* Left-to-right or right-to-left */
        msml_assert(inputs[idx] != NULL, "Invalid argument %d for node %s", i, msml_op_get_name(node->op));
        msml__compute_dag_eval(bci, inputs[idx], forward); /* Eval parent node recursive */
    }
    *out_inputs = (const msml_tensor_t**)inputs;
    return n_inputs;
}

static void MSML_HOTPROC msml__compute_dag_eval(const msml__blas_compute_info_t* const bci, msml_tensor_t* const node, bool forward) {
    const msml_op_t op = node->op;
    if (op == MSML_OP_NOP || msml_unlikely(op >= MSML_OP__COUNT)) return; /* NOP */
    const msml_tensor_t** inputs;
    msml__process_parent_inputs(&inputs, bci, node, forward); /* Eval parents and verify arguments */
    void (**dispatch_lut)(const msml__blas_compute_info_t*, msml_tensor_t*, const msml_tensor_t**) = bci->ctx->blas_dispatch; /* Dispatch table */
    (*(*(dispatch_lut+op)))(bci, node, inputs); /* Dispatch to CPU implementation */
}

msml_tensor_t*  MSML_HOTPROC msml_tensor_evaluate(msml_tensor_t* tensor, msml_graph_eval_order_t order) {
    const msml__blas_compute_info_t info = {
        .ctx = tensor->ctx,
        .n_threads = 1,
        .thread_idx = 0
    };
    msml__compute_dag_eval(&info, tensor, order == MSML_GRAPH_EVAL_ORDER_FORWARD);
    return tensor;
}

#ifdef __APPLE__
    static bool msml__sysctl_mib01(uint8_t (*out)[256], size_t* o_len, int mib0, int mib1) { /* Get sysctl data */
        memset(out, 0, sizeof(*out));
        *o_len = 0;
        int name[2] = {mib0, mib1};
        size_t len = 0;
        if (msml_unlikely(sysctl(name, sizeof(name) / sizeof(*name), NULL, &len, NULL, 0))) return false; /* Get length */
        if (msml_unlikely(len >= sizeof(*out))) return false; /* Buffer too small */
        if (msml_unlikely(sysctl(name, sizeof(name) / sizeof(*name), *out, &len, NULL, 0))) return false; /* Get data */
        *o_len = len;
        return true;
    }
    static bool msml__sysctl_key(uint8_t (*out)[256], size_t* o_len, const char* key) { /* Get sysctl data */
        memset(out, 0, sizeof(*out));
        *o_len = 0;
        size_t len = 0;
        if (msml_unlikely(sysctlbyname(key, NULL, &len, NULL, 0))) return false; /* Get length */
        if (msml_unlikely(len >= sizeof(*out))) return false; /* Buffer too small */
        if (msml_unlikely(sysctlbyname(key, *out, &len, NULL, 0))) return false; /* Get data */
        *o_len = len;
        return true;
    }
    static uint64_t msml__sysctl_unpack_int(const uint8_t (*in)[256], size_t len) { /* Unpack sysctl data */
        switch (len) {
            case sizeof(uint16_t): { uint16_t r; memcpy(&r, *in, sizeof(r)); return r; }
            case sizeof(uint32_t): { uint32_t r; memcpy(&r, *in, sizeof(r)); return r; }
            case sizeof(uint64_t): { uint64_t r; memcpy(&r, *in, sizeof(r)); return r; }
            default: return 0;
        }
    }
#else
    static bool msml__cpuinfo_parse_value(const char* key, char (*out)[128]) {
        FILE* cpuinfo = msml__fopen("/proc/cpuinfo", "rt");
        if (msml_unlikely(!cpuinfo)) return false;
        size_t key_len = strlen(key);
        char line[128];
        while (fgets(line, sizeof(line), cpuinfo)) {
            size_t line_len = strlen(line);
            if (line_len > 0 && line[line_len-1] == '\n') line[line_len-1] = '\0';
            if (strncmp(line, key, key_len) == 0 && (isspace((unsigned char)line[key_len]) || line[key_len] == ':')) {
                char* colon = strchr(line, ':');
                if (!colon) continue;
                char* value = colon+1;
                while (isspace((unsigned char)*value)) ++value;
                char* end = value + strlen(value);
                for (; end > value && isspace((unsigned char)*(end-1)); --end);
                *end = '\0';
                size_t value_len = llabs(end-value);
                if (msml_unlikely(!value_len || value_len >= sizeof(*out))) {
                    fclose(cpuinfo);
                    return false;
                }
                snprintf(*out, sizeof(*out), "%s", value);
                fclose(cpuinfo);
                return true;
            }
        }
        fclose(cpuinfo);
        return false;
    }
    static uint64_t msml__parse_meminfo_value(const char* line) {
        const char *p = strchr(line, ':');
        if (msml_unlikely(!p)) return 0;
        ++p;
        p += strspn(p, " \t");
        errno = 0;
        char* end;
        uint64_t value = strtoull(p, &end, 10);
        if (msml_unlikely(errno != 0 || p == end)) return 0;
        return value<<10;
    }
#endif

static void msml_system_host_info_query_os_name(char (*out_os_name)[128]) { /* Get OS name */
    #ifdef _WIN32
    #error "Unsupported platform"
    #elif defined(__APPLE__)
        size_t len;
        uint8_t tmp[256];
        if (msml_likely(msml__sysctl_mib01(&tmp, &len, CTL_KERN, KERN_VERSION) && len && *tmp))
            snprintf(*out_os_name, sizeof(*out_os_name), "%s", (const char*)tmp);
    #else
        // TODO: Linux
    #endif
}

static void msml_system_host_info_query_cpu_name(char (*out_cpu_name)[128]) { /* Get CPU name */
    #ifdef _WIN32
    #error "Unsupported platform"
    #elif defined(__APPLE__)
        size_t len;
        uint8_t tmp[256];
        if (msml_likely(msml__sysctl_key(&tmp, &len, "machdep.cpu.brand_string") && len && *tmp))
            snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", (const char*)tmp);
    #else
        char cpu_name[128];
        if (msml_likely(msml__cpuinfo_parse_value("model name", &cpu_name) && *cpu_name))
            snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", cpu_name);
    #endif
}

static void msml_system_host_info_query_cpu_cores(uint32_t* out_virtual, uint32_t* out_physical, uint32_t* out_sockets) { /* Get CPU virtual (logical) cores. */
    #ifdef _WIN32
    #error "Unsupported platform"
    #elif defined(__APPLE__)
        uint8_t tmp[256];
        size_t len;
        if (msml_likely(msml__sysctl_key(&tmp, &len, "machdep.cpu.thread_count") && len))
            *out_virtual = msml__sysctl_unpack_int(&tmp, len);
        if (msml_likely(msml__sysctl_key(&tmp, &len, "machdep.cpu.core_count") && len))
            *out_physical = msml__sysctl_unpack_int(&tmp, len);
        if (msml_likely(msml__sysctl_key(&tmp, &len, "hw.packages") && len))
            *out_sockets = msml__sysctl_unpack_int(&tmp, len);
    #else
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        *out_virtual = nprocs > 0 ? (uint32_t)nprocs : 0;
        FILE* cpuinfo = msml__fopen("/proc/cpuinfo", "r");
        if (msml_unlikely(!cpuinfo)) return;
        uint32_t physical_ids[MSML_MAX_CPUS];
        uint32_t core_ids[MSML_MAX_CPUS];
        uint32_t package_ids[MSML_MAX_CPUS];
        uint32_t cpu_count = 0;
        uint32_t package_count = 0;
        uint32_t current_physical_id = 0;
        uint32_t current_core_id = 0;
        bool got_physical_id = false;
        bool got_core_id = false;
        char line[256];
        while (fgets(line, sizeof(line), cpuinfo) != NULL) {
            if (strncmp(line, "physical id", sizeof("physical id")-1) == 0) {
                char* ptr = strchr(line, ':');
                if (ptr) {
                    ++ptr;
                    for (; *ptr && !isdigit((unsigned char)*ptr); ++ptr);
                    if (*ptr) { current_physical_id = (uint32_t)strtoul(ptr, NULL, 10); got_physical_id = true; }
                }
            } else if (strncmp(line, "core id", sizeof("core id")-1) == 0) {
                char* ptr = strchr(line, ':');
                if (ptr) {
                    ++ptr;
                    for (; *ptr && !isdigit((unsigned char)*ptr); ++ptr);
                    if (*ptr) { current_core_id = (uint32_t)strtoul(ptr, NULL, 10); got_core_id = true; }
                }
            } else if (*line == '\n') {
                if (got_physical_id && got_core_id) {
                    bool is_unique = true;
                    for (int32_t i = 0; i < cpu_count; ++i) if (physical_ids[i] == current_physical_id && core_ids[i] == current_core_id) { is_unique = false; break; }
                    if (is_unique) {
                        if (cpu_count < MSML_MAX_CPUS) {
                            physical_ids[cpu_count] = current_physical_id;
                            core_ids[cpu_count] = current_core_id;
                            ++cpu_count;
                        } else break;
                    }
                    is_unique = true;
                    for (int32_t i = 0; i < package_count; ++i) if (package_ids[i] == current_physical_id) { is_unique = false; break; }
                    if (is_unique) {
                        if (package_count < MSML_MAX_CPUS) package_ids[package_count++] = current_physical_id;
                        else break;
                    }
                }
                got_physical_id = false;
                got_core_id = false;
            }
        }
        fclose(cpuinfo);
        *out_physical = cpu_count;
        *out_sockets = package_count;
    #endif
}

static void msml__system_host_info_query_memory(uint64_t* out_phys_mem_total, uint64_t* out_phys_mem_free) { /* Get physical memory */
    #ifdef _WIN32
    #error "Unsupported platform"
    #elif defined(__APPLE__)
        uint8_t tmp[256];
        size_t len;
        if (msml_likely(msml__sysctl_mib01(&tmp, &len, CTL_HW, HW_MEMSIZE) && len))
            *out_phys_mem_total = msml__sysctl_unpack_int(&tmp, len);
        struct vm_statistics64 stats;
        natural_t count = HOST_VM_INFO64_COUNT;
        if (msml_likely(host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)(&stats), &count) == KERN_SUCCESS))
            *out_phys_mem_free = stats.free_count * getpagesize();
    #else
        FILE* meminfo = msml__fopen("/proc/meminfo", "r");
        if (msml_unlikely(!meminfo)) return;
        char line[256];
        while (fgets(line, sizeof(line), meminfo)) {
            if (strncmp(line, "MemTotal:", sizeof("MemTotal:")-1) == 0)
                *out_phys_mem_total = msml__parse_meminfo_value(line);
            else if (strncmp(line, "MemAvailable:", sizeof("MemAvailable:")-1) == 0)
                *out_phys_mem_free = msml__parse_meminfo_value(line);
        }
        fclose(meminfo);
    #endif
}

#if defined(__x86_64__) || defined(_M_X64)
    static uint64_t MSML_AINLINE msml__xgetbv(void) { /* Query extended control register value. */
        #ifdef _MSC_VER
            return _xgetbv(0);
        #else
            uint32_t lo, hi;
            __asm__ __volatile__("xgetbv\n\t" : "=a" (lo), "=d" (hi) : "c" (0));
            return (uint64_t)lo | ((uint64_t)hi << 32);
        #endif
    }
    #define msml__cpy_regs(id) \
        (*features)[MSML__X86_64_CPUID_##id][MSML__X86_64_CPUID_EAX] = eax; \
        (*features)[MSML__X86_64_CPUID_##id][MSML__X86_64_CPUID_EBX] = ebx; \
        (*features)[MSML__X86_64_CPUID_##id][MSML__X86_64_CPUID_ECX] = ecx; \
        (*features)[MSML__X86_64_CPUID_##id][MSML__X86_64_CPUID_EDX] = edx
    static void msml__system_info_query_x86_64_cpu_features(uint32_t (*features)[8][4]) {
        uint32_t eax, ebx, ecx, edx;
        uint32_t max_basic_leaf, max_extended_leaf;

        __cpuid(0, eax, ebx, ecx, edx);
        msml__cpy_regs(0H);
        max_basic_leaf = eax;
        __cpuid(0x80000000u, eax, ebx, ecx, edx);
        max_extended_leaf = eax;
        if (max_basic_leaf >= 1u) {
            __cpuid(1, eax, ebx, ecx, edx);
            msml__cpy_regs(1H);
        }
        if (max_basic_leaf >= 2u) {
            __cpuid(2u, eax, ebx, ecx, edx);
            msml__cpy_regs(2H);
        }
        if (max_basic_leaf >= 7u) {
            __cpuid_count(7u, 0, eax, ebx, ecx, edx);
            msml__cpy_regs(7H);
        }
        if (max_basic_leaf >= 7u) {
            __cpuid_count(7u, 1, eax, ebx, ecx, edx);
            msml__cpy_regs(7H_1H);
        }
        if (max_basic_leaf >= 0x16u) {
            __cpuid(0x16u, eax, ebx, ecx, edx);
            msml__cpy_regs(16H);
        }
        if (max_extended_leaf >= 0x80000001u) {
            __cpuid(0x80000001u, eax, ebx, ecx, edx);
            msml__cpy_regs(80000001H);
        }
        if (max_extended_leaf >= 0x80000007u) {
            __cpuid(0x80000007u, eax, ebx, ecx, edx);
            msml__cpy_regs(80000007H);
        }
        bool cpu_avx_support = ((*features)[MSML__X86_64_CPUID_1H][MSML__X86_64_CPUID_ECX] & 0x10000000u) != 0;
        bool cpu_osxsave_support = ((*features)[MSML__X86_64_CPUID_1H][MSML__X86_64_CPUID_ECX] & 0x8000000u) != 0;
        if (cpu_avx_support && cpu_osxsave_support) {
            uint64_t xcr0 = msml__xgetbv();
            if ((xcr0 & 0x6) != 0x6u) {
                (*features)[MSML__X86_64_CPUID_1H][MSML__X86_64_CPUID_ECX] &= ~0x10000000u; /* Clear AVX */
                (*features)[MSML__X86_64_CPUID_7H][MSML__X86_64_CPUID_EBX] &= ~0x20u; /* Clear AVX2 */
            }
            if ((xcr0 & 0xe0) != 0xe0u) { /* OS does not support AVX-512, clear AVX512 */
                (*features)[MSML__X86_64_CPUID_7H][MSML__X86_64_CPUID_EBX] &= ~0xdc230000u;
                (*features)[MSML__X86_64_CPUID_7H][MSML__X86_64_CPUID_ECX] &= ~0x5842u;
                (*features)[MSML__X86_64_CPUID_7H][MSML__X86_64_CPUID_EDX] &= ~0x10cu;
                (*features)[MSML__X86_64_CPUID_7H_1H][MSML__X86_64_CPUID_EAX] &= ~0x20u;
            }
        } else {
            (*features)[MSML__X86_64_CPUID_1H][MSML__X86_64_CPUID_ECX] &= ~0x10000000u; /* Clear AVX */
            (*features)[MSML__X86_64_CPUID_7H][MSML__X86_64_CPUID_EBX] &= ~0x20u; /* Clear AVX2 */
            (*features)[MSML__X86_64_CPUID_7H][MSML__X86_64_CPUID_EBX] &= ~0xdc230000u; /* Clear AVX512 */
            (*features)[MSML__X86_64_CPUID_7H][MSML__X86_64_CPUID_ECX] &= ~0x5842u; /* Clear AVX512 */
            (*features)[MSML__X86_64_CPUID_7H][MSML__X86_64_CPUID_EDX] &= ~0x10cu; /* Clear AVX512 */
            (*features)[MSML__X86_64_CPUID_7H_1H][MSML__X86_64_CPUID_EAX] &= ~0x20u; /* Clear AVX512 */
        }
    }
    #undef msml__cpy_regs
#endif

static void msml__system_host_info_query(msml_ctx_t* ctx) {
    msml_system_host_info_query_os_name(&ctx->sys.os_name);
    msml_system_host_info_query_cpu_name(&ctx->sys.cpu_name);
    msml_system_host_info_query_cpu_cores(&ctx->sys.cpu_virtual_cores, &ctx->sys.cpu_physical_cores, &ctx->sys.cpu_sockets);
    msml__system_host_info_query_memory(&ctx->sys.phys_mem_total, &ctx->sys.phys_mem_free);
    #if defined(__x86_64__) || defined(_M_X64)
        msml__system_info_query_x86_64_cpu_features(&ctx->sys.x86_64_cpu_features);
    #endif
    if (msml_unlikely(!*ctx->sys.os_name)) snprintf(ctx->sys.os_name, sizeof(ctx->sys.os_name), "Unknown");
    if (msml_unlikely(!*ctx->sys.cpu_name)) snprintf(ctx->sys.cpu_name, sizeof(ctx->sys.cpu_name), "Unknown");
}

msml_tensor_t* msml_tensor_create_from_image(msml_ctx_t* ctx, const char* file_path, msml_desired_color_channels_t in_desired_channels, uint32_t resize_width, uint32_t resize_height) {
#ifdef MSML_ENABLE_IMAGE_SUPPORT
    int width, height, channels, desired_channels;
    switch (in_desired_channels) {
        default: desired_channels = STBI_default; break;
        case MSML_COLOR_CHANNELS_GRAY: desired_channels = STBI_grey; break;
        case MSML_COLOR_CHANNELS_GRAY_A: desired_channels = STBI_grey_alpha; break;
        case MSML_COLOR_CHANNELS_RGB: desired_channels = STBI_rgb; break;
        case MSML_COLOR_CHANNELS_RGBA: desired_channels = STBI_rgb_alpha; break;
    }
    uint8_t* image_data = stbi_load(file_path, &width, &height, &channels, desired_channels);
    if (!image_data || width == 0 || height == 0 || channels == 0) {
        msml_panic("Failed to load image from file: %s\n", file_path);
    }
    if (resize_width && resize_height) { /* Resize image if requested */
        uint8_t* resized_data = stbir_resize_uint8_srgb(
            image_data,
            width,
            height,
            0,
            NULL,
            (int)resize_width,
            (int)resize_height,
            0,
            (stbir_pixel_layout)desired_channels
        );
        if (resized_data) { /* Replace original image data with resized data */
            stbi_image_free(image_data);
            image_data = resized_data;
            width = (int)resize_width;
            height = (int)resize_height;
        }
    }
    msml_tensor_t* tensor = msml_tensor_create_3d(ctx, MSML_DTYPE_F32, width, height, channels);
    float* dst = (float*)tensor->buf;
    int64_t n = width*height*channels;
    msml_assert(n == msml_tensor_buf_len(tensor), "Buffer size mismatch: %zu != %lld", n, msml_tensor_buf_len(tensor));
    for (int64_t i=0; i < n; ++i)
        dst[i] = (float)image_data[i] / 255.0f; /* Normalize pixel values to [0, 1] */
    stbi_image_free(image_data);
    msml_log_info("Loaded tensor from image: %s, width: %d, height: %d, channels: %d", file_path, width, height, channels);
    return tensor;
#else
    msml_panic("Image support is disabled. MSML must be compiled with MSML_ENABLE_IMAGE_SUPPORT defined.");
#endif
}

void msml_tensor_save_to_image(const msml_tensor_t* tensor, const char* file_path) {
#ifdef MSML_ENABLE_IMAGE_SUPPORT
    const int64_t* dims = msml_tensor_shape(tensor);
    int64_t rank = msml_tensor_rank(tensor);
    msml_assert(rank == 3, "Tensor rank must be 3, but is: %" PRIi64, (size_t)rank);
    int64_t width = dims[0];
    int64_t height = dims[1];
    int64_t channels = dims[2];
    msml_assert(channels == 1 || channels == 3 || channels == 4, "Invalid number of channels: %" PRIi64, channels);
    size_t n = width*height*channels;
    msml_assert(n == msml_tensor_buf_len(tensor), "Buffer size mismatch: %zu != %lld", n, msml_tensor_buf_len(tensor));
    uint8_t* image_data = (*tensor->ctx->alloc_fn)(NULL, n); /* Allocate memory for image data */
    for (size_t i=0; i < n; ++i) { /* Clamp and denormalize pixel values to [0, 255] */
        image_data[i] = (uint8_t)(255.0f * msml_min(msml_max(((const float*)tensor->buf)[i], 0.0f), 1.0f));
    }
    int result = stbi_write_jpg(file_path, (int)width, (int)height, (int)channels, image_data, 100);
    msml_assert(result, "Failed to save tensor to image: %s", file_path);
    (*tensor->ctx->alloc_fn)(image_data, 0); /* Free image data */
    msml_log_info("Saved tensor to image: %s, width: %d, height: %d, channels: %d", file_path, (int)width, (int)height, (int)channels);
#else
    msml_panic("Image support is disabled. MSML must be compiled with MSML_ENABLE_IMAGE_SUPPORT defined.");
#endif
}

struct msml_compute_graph_t {
    msml_ctx_t* ctx;
    msml_tensor_t** nodes;
    msml_tensor_t** leafs;
    size_t num_nodes;
    size_t num_leafs;
    size_t size_total;
    msml_graph_eval_order_t order;
};
