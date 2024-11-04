/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

/*
**
**
** ### To add a new operation:
** 1. Add the operation to the wl_op_def macro, which defines all operations, with all information needed.
** 2. Write a validation routine, or use an existing one (e.g. 'wl__validate_op_binary').
** 3. Add the validation routine to the 'routines' table in 'wl__op_get_validator_routine', at the op index.
** 4. Write a result tensor constructor routine or use an existing one (e.g. 'wl__result_constructor_routine_isomorph').
** 5. Add the result tensor constructor routine to the 'routines' table in 'wl__op_get_result_constructor_routine', at the op index.
** 6. Write a BLAS computation routine.
** 7. Add the BLAS computation routine to the 'dispatch_lut' table in 'wl__blas_compute_dispatch_table_default', at the op index.
*/

#define WL_EXPORT_DLL
#include "wavelet.h"

#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <ctype.h>
#include <errno.h>

#ifdef _MSC_VER
#   include <intrin.h>
#endif

#ifdef __aarch64__
#   include <arm_neon.h>
#   include <arm_acle.h>
#elif defined(__x86_64__) || defined(_M_X64)
#   include <immintrin.h>
#   ifndef _MSC_VER
#       include <cpuid.h>
#   endif
#endif

#ifdef _WIN32
#   error "WAVELET does not support Windows yet."
#elif defined(__APPLE__)
#   include <mach/mach.h>
#   include <mach/vm_statistics.h>
#   include <sys/sysctl.h>
#   include <sys/types.h>
#   include <unistd.h>
#else
#   include <unistd.h>
#endif

wl_static_assert(sizeof(0u) == 4);
wl_static_assert(sizeof(0ull) == 8);

#define WL__MAX_CPUS 8192
#define WL__MAX_NUMA_NODES 64
#define WL__STORAGE_EXT ".wavelet"

#ifdef WL_ENABLE_IMAGE_SUPPORT
#define STBI_MALLOC(sz) wl_alloc(NULL, (sz))
#define STBI_FREE(ptr) wl_alloc((ptr), 0)
#define STBI_REALLOC(ptr, sz) wl_alloc((ptr), (sz))
#define STBIW_MALLOC(sz) wl_alloc(NULL, (sz))
#define STBIW_FREE(ptr) wl_alloc((ptr), 0)
#define STBIW_REALLOC(ptr, sz) wl_alloc((ptr), (sz))
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
#define WL__NORET __attribute__((noreturn))
#define WL__ALIGN(x) __attribute__((aligned(x)))
#define WL__AINLINE inline __attribute__((always_inline))
#define WL__NOINLINE __attribute__((noinline))
#define WL__HOTPROC __attribute__((hot))
#define WL__COLDPROC __attribute__((cold))
#define WL__PACKED __attribute__((packed))
#define WL__FALLTHROUGH __attribute__((fallthrough))
#define WL__UNUSED __attribute__((unused))
#define wl__likely(x) __builtin_expect(!!(x), 1)
#define wl__unlikely(x) __builtin_expect(!!(x), 0)
#define wl__ffs(x) ((uint32_t)__builtin_ctz(x))
#define wl__fls(x) ((uint32_t)(__builtin_clz(x)^31))
#define wl__ffs64(x) ((uint32_t)__builtin_ctzll(x))
#define wl__fls64(x) ((uint32_t)(__builtin_clzll(x)^63))
#else
unsigned char _BitScanForward64(unsigned long*, unsigned __int64);
unsigned char _BitScanReverse64(unsigned long*, unsigned __int64);
#pragma intrinsic(_BitScanForward64)
#pragma intrinsic(_BitScanReverse64)
#define WL__NORET __declspec(noreturn)
#define WL__ALIGN(x) __declspec(align(x))
#define WL__AINLINE inline __forceinline
#define WL__NOINLINE __declspec(noinline)
#define WL__HOTPROC
#define WL__COLDPROC
#define WL__PACKED __declspec(align(1))
#define WL__FALLTHROUGH
#define WL__UNUSED
#define wl__likely(x) (x)
#define wl__unlikely(x) (x)
static __forceinline uint32_t wl__ffs(const uint32_t x) {
    unsigned long r; _BitScanForward(&r, x); return (uint32_t)r;
}
static __forceinline uint32_t wl__fls(const uint32_t x) {
    unsigned long r; _BitScanReverse(&r, x); return (uint32_t)r;
}
static __forceinline uint32_t wl__ffs64(const uint64_t x) {
  unsigned long r; _BitScanForward64(&r, x); return (uint32_t)r;
}
static __forceinline uint32_t wl__fls64(const uint64_t x) {
  unsigned long r; _BitScanReverse64(&r, x); return (uint32_t)r;
}
#endif

#define wl__swap(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)
#define wl__max(x, y) (((x) > (y)) ? (x) : (y))
#define wl__min(x, y) (((x) < (y)) ? (x) : (y))
#define WL__CC_RED "\x1b[31m"
#define WL__CC_GREEN "\x1b[32m"
#define WL__CC_YELLOW "\x1b[33m"
#define WL__CC_BLUE "\x1b[34m"
#define WL__CC_MAGENTA "\x1b[35m"
#define WL__CC_CYAN "\x1b[36m"
#define WL__CC_RESET "\x1b[0m"
#define WL__STRINGIZE2(x) #x
#define WL__STRINGIZE(x) WL__STRINGIZE2(x)
#ifdef __FILE_NAME__
#   define WL__SRC_NAME __FILE_NAME__ ":" WL__STRINGIZE(__LINE__)
#else
#   define WL__SRC_NAME __FILE__ ":" WL__STRINGIZE(__LINE__)
#endif
#define wl__log_info(msg, ...) fprintf(stdout,   WL__CC_CYAN "[WAVELET] " WL__CC_RESET WL__SRC_NAME " " msg "\n", ## __VA_ARGS__)
#define wl__log_warn(msg, ...) fprintf(stderr,  WL__CC_CYAN "[WAVELET] " WL__CC_RESET WL__SRC_NAME " " WL__CC_YELLOW msg WL__CC_RESET "\n", ## __VA_ARGS__)
#define wl__log_error(msg, ...) fprintf(stderr,  WL__CC_CYAN "[WAVELET] " WL__CC_RESET WL__SRC_NAME " " WL__CC_RED msg WL__CC_RESET "\n", ## __VA_ARGS__)

static WL__NORET WL__COLDPROC void wl__panic(const char* msg, ...) {
    fprintf(stderr, "%s", WL__CC_RED);
    va_list args;
    va_start(args, msg);
    vfprintf(stderr, msg, args);
    va_end(args);
    fprintf(stderr, "%s", WL__CC_RESET);
    fputc('\n', stderr);
    fflush(stderr);
    fflush(stdout);
    abort();
}

#define wl__assert(expr, msg, ...) \
    if (wl__unlikely(!(expr))) { \
        wl__panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
    }
#define wl__assert2(expr) wl__assert(expr, "")

#if defined(__x86_64__) || defined(_M_X64)
#define WL__X86_64_CPUID_0H 0
#define WL__X86_64_CPUID_1H 1
#define WL__X86_64_CPUID_2H 2
#define WL__X86_64_CPUID_7H 3
#define WL__X86_64_CPUID_80000001H 4
#define WL__X86_64_CPUID_80000007H 5
#define WL__X86_64_CPUID_16H 6
#define WL__X86_64_CPUID_7H_1H 7
#define WL__X86_64_CPUID_EAX 0
#define WL__X86_64_CPUID_EBX 1
#define WL__X86_64_CPUID_ECX 2
#define WL__X86_64_CPUID_EDX 3

#define wl_x86_64_feature_def(_, __) /* Enumerator | CPUDID Leaf | Register | Bit Index */\
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

#define _(enumerator, leaf, reg, bit) WL__X86_64_FEATURE_##enumerator
typedef enum wl__x86_64_feature_t {
    wl_x86_64_feature_def(_, WL_SEP)
    WL__X86_64_FEATURE__COUNT
} wl__x86_64_feature_t;
#undef _
#define _(enumerator, leaf, reg, bit) #enumerator
static const char* const wl__x86_64_feature_names[WL__X86_64_FEATURE__COUNT] = {
    wl_x86_64_feature_def(_, WL_SEP)
};
#undef _
#define _(enumerator, leaf, reg, bit) (0xff&WL__X86_64_CPUID_##leaf)
static const uint8_t wl__x86_64_feature_leaves[WL__X86_64_FEATURE__COUNT] = {
    wl_x86_64_feature_def(_, WL_SEP)
};
#undef _
#define _(enumerator, leaf, reg, bit) (0xff&WL__X86_64_CPUID_##reg)
static const uint8_t wl__x86_64_feature_regs[WL__X86_64_FEATURE__COUNT] = {
    wl_x86_64_feature_def(_, WL_SEP)
};
#undef _
#undef wl_x86_64_feature_def
#endif

typedef struct wl__blas_compute_info_t wl__blas_compute_info_t; /* Forward declaration. */

/*
** Context contains all isolated state and data.
** Lifetimes of tensors and compute graphs are bound to the context - the context is the owner.
** Context itself is not thread-safe, use a thread-local context or synchronize access. (Multiple contexts can be used.)
*/
struct wl_ctx_t {
    void* (*alloc_fn)(void* blk, size_t size); /* Memory allocator. */
    struct {
        char os_name[128];                          /* OS name. */
        char cpu_name[128];                         /* CPU name. */
        uint32_t cpu_virtual_cores;                 /* Virtual CPUs. */
        uint32_t cpu_physical_cores;                /* Physical CPU cores. */
        uint32_t cpu_sockets;                       /* CPU sockets. */
        uint64_t phys_mem_total;                    /* Total physical memory in bytes. */
        uint64_t phys_mem_free;                     /* Free physical memory in bytes. */
        #if defined(__x86_64__) || defined(_M_X64)
            uint32_t x86_64_cpu_features[8][4];     /* x86-64 CPU features. */
        #endif
    } sys;
    struct {
        size_t chunk_size;                          /* Size of new allocated memory pool chunk. Can grow if needed. */
        size_t chunk_len;                           /* Length of each memory pool chunk. */
        size_t chunk_cap;                           /* Maximum number of memory pool chunks. */
        uint8_t** chunks;                           /* Stack of all allocated memory pool chunks. Active is top. */
        uint8_t* delta;                             /* Position in active memory pool chunk. Growing downwards. */
        size_t alloc_acc;                           /* Allocation counter. */
        size_t mapped_total;                        /* Total memory allocated from OS/allocator. */
        size_t alloc_total;                         /* Total memory allocated from memory pool. */
        bool warmup_chunks;                         /* If true, fresh pool chunks are zeroed to allocate kernel pages, can improve or decrease performance depending on scenario. */
    } pool;
    uint8_t* (*image_load_fn)(const char*, uint32_t(*)[3], wl_color_channels_t);
    void (*image_load_free_fn)(uint8_t*);
    bool (*image_save_fn)(const char*, const uint8_t*, const uint32_t(*)[3]);
    wl_exec_mode_t exec_mode;
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
    wl_prng_algorithm_t prng_algorithm;
    uintptr_t host_thread_id;
    void (*blas_dispatch[WL_OP__COUNT])(const wl__blas_compute_info_t*, wl_tensor_t*, const wl_tensor_t**); /* BLAS dispatch table. Specialized for host CPU architecture. */
    void* ud; /* User data. */
};

typedef enum wl__tensor_flags_t {
    WL__TFLAG_NONE = 0,
    WL__TFLAG_VIEW = 1<<0,         /* Tensor is a view. */
    WL__TFLAG_OP_INPUT = 1<<1,     /* Tensor is an operation input. */
    WL__TFLAG_OP_OUTPUT = 1<<2,    /* Tensor is an operation output. */
    WL__TFLAG_EXEC_EAGER = 1<<3,   /* Tensor is executed eagerly. */
    WL__TFLAG_IMAGE = 1<<4,        /* Tensor was loaded from an image. */
    WL__TFLAG_FROM_FS = 1<<5,      /* Tensor was loaded from the file system. Also true for WL__TFLAG_IMAGE. */
} wl__tensor_flags_t;
wl_static_assert(WL__TFLAG_FROM_FS <= 0xff); /* Must fit info 8-bits. */

/*
** Tensor with up to 6 Dimensions.
*/
struct wl_tensor_t {
    wl_ctx_t* ctx;                                  /* Host context. */
    int64_t rank;                                   /* Number of active dimensions. [1, MAX_DIMS] */
    int64_t shape[WL_MAX_DIMS];                     /* Shape of the tensor. */
    int64_t strides[WL_MAX_DIMS];                   /* Strides of the tensor. */
    wl_dtype_t dtype;                               /* Data type of the tensor. */
    void* buf;                                      /* Data buffer. */
    int64_t num_elems;                              /* Number of elements in the tensor. */
    wl__tensor_flags_t flags;                       /* Tensor flags. */
    wl_op_t op;                                     /* Opcode for operators. */
    wl_tensor_t* op_inputs[WL_MAX_INPUT_TENSORS];   /* Input tensors for operators. */
    wl_op_param_t op_params[WL_MAX_OP_PARAMS];      /* Operator parameters. */
    wl_tensor_t* view;                              /* View tensor. */
    size_t view_offs;                               /* Offset in view tensor. */
    char name[WL_MAX_TENSOR_NAME_LEN];              /* Tensor debug name. */
    void* ud;                                       /* User data. */
};

#if WL_BOUNDS_CHECK
#define wl__bnd_chk(ptr, base, n) \
    wl__assert((uintptr_t)(ptr) >= (uintptr_t)(base) && (uintptr_t)(ptr) < (uintptr_t)(base) + (n), \
        "\nBound check failed: %p not in [%p, %p), base+%zu, end+%zu", \
        (void*)(ptr), \
        (void*)(base), \
        (void*)((uintptr_t)(base)+(n)), \
        (size_t)llabs((long long)((int64_t)(ptr)-(int64_t)(base))), \
        (size_t)llabs((long long)(((int64_t)(base)+(n))-(int64_t)(ptr))) \
    )
#else
#define wl__bnd_chk(ptr, base, n)
#endif

void* wl_default_allocator_impl(void* blk, size_t size) {
    if (!size) {
        free(blk);
        return NULL;
    } else if(!blk) {
        blk = malloc(size);
        wl__assert(blk, "Failed to allocate %.03fKiB memory", (double)size/(double)(1<<10));
        return blk;
    } else {
        void* block = realloc(blk, size);
        wl__assert(blk, "Failed to reallocate %.03fKiB memory", (double)size/(double)(1<<10));
        return block;
    }
}

static void wl__humanize_memory_size(size_t n, double* out, const char** unit) {
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

#define WL__FMT_DIM_BUF_SIZE ((21+3)*WL_MAX_DIMS)
static void wl__fmt_dims(char (*buf)[WL__FMT_DIM_BUF_SIZE], const int64_t (*dims)[WL_MAX_DIMS], int64_t rank) {
    wl_static_assert(WL_MAX_DIMS == 6);
    memset(*buf, 0, sizeof(*buf));
    char* p = *buf;
    wl__assert2(p+rank*21+3 < *buf+WL__FMT_DIM_BUF_SIZE);
    *p++ = '(';
    for (int64_t i=0; i < rank; ++i) {
        p += snprintf(p, 21, "%" PRIi64, (*dims)[i]);
        if (i < rank-1) *p++ = ' ';
    }
    *p++ = ')';
    *p = '\0';
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

static FILE* wl__fopen(const char* file, const char* mode) {
    wl__assert(file && *file && mode && *mode, "Invalid file name or mode");
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

static inline uintptr_t wl__thread_id(void) {
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
    #   error "Unsupported WAVELET platform"
    #endif
    return tid;
}

static int64_t wl__hpc_clock_us(void) { /* High precision clock in microseconds. */
    #ifdef _WIN32
    #error "WAVELET does not support Windows yet."
    #else
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
    #endif
}
static int64_t wl__hpc_clock_elapsed_us(int64_t start) { /* High precision clock elapsed time in microseconds. */
    return llabs(wl__hpc_clock_us() - start);
}
static double wl__hpc_clock_elapsed_ms(int64_t start) { /* High precision clock elapsed time in milliseconds. */
    return (double)wl__hpc_clock_elapsed_us(start) * 1.0e-3;
}

typedef uint32_t wl__bitset_t;
wl_static_assert(sizeof(wl__bitset_t) == 4);
#define wl__bitset_size(n) (((n)+((4<<3)-1))>>5)
#define wl__bitset_get(sets, i) (!!(sets[(i)>>5]&(1u<<((i)&((4<<3)-1)))))
#define wl__bitset_set(sets, i) (sets[(i)>>5]|=(1u<<((i)&((4<<3)-1))))
#define wl__bitset_clear(sets, i) (sets[(i)>>5]&=~(1u<<((i)&((4<<3)-1))))
#define wl__bitset_toggle(sets, i) (sets[(i)>>5]^=(1u<<((i)&((4<<3)-1))))

static uint32_t WL__AINLINE wl__bswap32(uint32_t x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        #if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
            x = (uint32_t)__builtin_bswap32((int32_t)x);
        #else
            x = (x & 0xff000000) >> 24 |
            (x & 0xff0000) >> 8 |
            (x & 0xff00) << 8 |
            (x & 0xff) << 24;
        #endif
    #endif
    return x;
}

static uint64_t WL__AINLINE wl__bswap64(uint64_t x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
    #if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        #if (defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3))) || defined(__clang__)
            x = (uint64_t)__builtin_bswap64((int64_t)x);
        #else
            x = (x & 0xff00000000000000) >> 56 |
            (x & 0xff000000000000) >> 40 |
            (x & 0xff0000000000) >> 24 |
            (x & 0xff00000000) >> 8 |
            (x & 0xff000000) << 8 |
            (x & 0xff0000) << 24 |
            (x & 0xff00) << 40 |
            (x & 0xff) << 56;
        #endif
    #endif
    return x;
}

static WL__AINLINE void* wl__pincr(void** p, size_t sz, size_t align) {
    void* pp = (void*)(((uintptr_t)*p+align-1) & -align);
    *p = (void*)((uint8_t*)pp + sz);
    return pp;
}

#ifdef __aarch64__
static uint64x2_t WL__AINLINE wl__clmul_lo_e(uint64x2_t a, uint64x2_t b, uint64x2_t c) {
    register uint64x2_t r;
    __asm__ __volatile__(
        "pmull %0.1q, %2.1d, %3.1d\n"
        "eor %0.16b, %0.16b, %1.16b\n"
        : "=w"(r), "+w"(c) : "w"(a), "w"(b)
    );
    return r;
}
static uint64x2_t WL__AINLINE wl__clmul_hi_e(uint64x2_t a, uint64x2_t b, uint64x2_t c) {
    register uint64x2_t r;
    __asm__ __volatile__(
        "pmull2 %0.1q, %2.2d, %3.2d\n"
        "eor %0.16b, %0.16b, %1.16b\n"
        : "=w"(r), "+w"(c) : "w"(a), "w"(b)
    );
    return r;
}
#elif defined(__x86_64__) || defined(_M_X64)
static uint32_t wl__xnmodp(uint64_t n) { /* x^n mod P, in log(n) time */
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
static __m128i WL__AINLINE wl__clmul_scalar(uint32_t a, uint32_t b) {
    return _mm_clmulepi64_si128(_mm_cvtsi32_si128(a), _mm_cvtsi32_si128(b), 0);
}
static __m128i WL__AINLINE wl__crc_shift(uint32_t crc, size_t sz) {
    return wl__clmul_scalar(crc, wl__xnmodp((sz<<3) - 33));
}
#endif

static uint32_t wl__crc32c(const void* buffer, size_t size) { /* Compute CRC32 checksum with CRC32c polynomial. */
    if (wl__unlikely(!buffer || !size)) return 0;
    const uint8_t* buf = (const uint8_t*)buffer;
    #if WL_INTRIN && defined(__aarch64__)
        uint32_t crc = ~0;
        for (; size && ((uintptr_t)buf & 7); --size) crc = __crc32cb(crc, *buf++);
        if (((uintptr_t)buf & 8) && size >= 8) {
            crc = __crc32cd(crc, *(const uint64_t*)buf);
            buf += 8;
            size -= 8;
        }
        if (size >= 192) { /* First vector chunk. */
            uint64x2_t x0 = vld1q_u64((const uint64_t*)buf), y0;
            uint64x2_t x1 = vld1q_u64((const uint64_t*)(buf+16)), y1;
            uint64x2_t x2 = vld1q_u64((const uint64_t*)(buf+32)), y2;
            uint64x2_t x3 = vld1q_u64((const uint64_t*)(buf+48)), y3;
            uint64x2_t x4 = vld1q_u64((const uint64_t*)(buf+64)), y4;
            uint64x2_t x5 = vld1q_u64((const uint64_t*)(buf+80)), y5;
            uint64x2_t x6 = vld1q_u64((const uint64_t*)(buf+96)), y6;
            uint64x2_t x7 = vld1q_u64((const uint64_t*)(buf+112)), y7;
            uint64x2_t x8 = vld1q_u64((const uint64_t*)(buf+128)), y8;
            uint64x2_t x9 = vld1q_u64((const uint64_t*)(buf+144)), y9;
            uint64x2_t x10 = vld1q_u64((const uint64_t*)(buf+160)), y10;
            uint64x2_t x11 = vld1q_u64((const uint64_t*)(buf+176)), y11;
            uint64x2_t k;
            { static const uint64_t WL__ALIGN(16) k_[] = {0xa87ab8a8, 0xab7aff2a}; k = vld1q_u64(k_); }
            x0 = veorq_u64((uint64x2_t){crc, 0}, x0);
            buf += 192;
            size -= 192;
            while (size >= 192) { /* Work loop. */
                y0 = wl__clmul_lo_e(x0, k, vld1q_u64((const uint64_t*)buf)), x0 = wl__clmul_hi_e(x0, k, y0);
                y1 = wl__clmul_lo_e(x1, k, vld1q_u64((const uint64_t*)(buf+16))), x1 = wl__clmul_hi_e(x1, k, y1);
                y2 = wl__clmul_lo_e(x2, k, vld1q_u64((const uint64_t*)(buf+32))), x2 = wl__clmul_hi_e(x2, k, y2);
                y3 = wl__clmul_lo_e(x3, k, vld1q_u64((const uint64_t*)(buf+48))), x3 = wl__clmul_hi_e(x3, k, y3);
                y4 = wl__clmul_lo_e(x4, k, vld1q_u64((const uint64_t*)(buf+64))), x4 = wl__clmul_hi_e(x4, k, y4);
                y5 = wl__clmul_lo_e(x5, k, vld1q_u64((const uint64_t*)(buf+80))), x5 = wl__clmul_hi_e(x5, k, y5);
                y6 = wl__clmul_lo_e(x6, k, vld1q_u64((const uint64_t*)(buf+96))), x6 = wl__clmul_hi_e(x6, k, y6);
                y7 = wl__clmul_lo_e(x7, k, vld1q_u64((const uint64_t*)(buf+112))), x7 = wl__clmul_hi_e(x7, k, y7);
                y8 = wl__clmul_lo_e(x8, k, vld1q_u64((const uint64_t*)(buf+128))), x8 = wl__clmul_hi_e(x8, k, y8);
                y9 = wl__clmul_lo_e(x9, k, vld1q_u64((const uint64_t*)(buf+144))), x9 = wl__clmul_hi_e(x9, k, y9);
                y10 = wl__clmul_lo_e(x10, k, vld1q_u64((const uint64_t*)(buf+160))), x10 = wl__clmul_hi_e(x10, k, y10);
                y11 = wl__clmul_lo_e(x11, k, vld1q_u64((const uint64_t*)(buf+176))), x11 = wl__clmul_hi_e(x11, k, y11);
                buf += 192;
                size -= 192;
            }
            /* Reduce x0 ... x11 to just x0. */
            { static const uint64_t WL__ALIGN(16) k_[] = {0xf20c0dfe, 0x493c7d27}; k = vld1q_u64(k_); }
            y0 = wl__clmul_lo_e(x0, k, x1), x0 = wl__clmul_hi_e(x0, k, y0);
            y2 = wl__clmul_lo_e(x2, k, x3), x2 = wl__clmul_hi_e(x2, k, y2);
            y4 = wl__clmul_lo_e(x4, k, x5), x4 = wl__clmul_hi_e(x4, k, y4);
            y6 = wl__clmul_lo_e(x6, k, x7), x6 = wl__clmul_hi_e(x6, k, y6);
            y8 = wl__clmul_lo_e(x8, k, x9), x8 = wl__clmul_hi_e(x8, k, y8);
            y10 = wl__clmul_lo_e(x10, k, x11), x10 = wl__clmul_hi_e(x10, k, y10);
            { static const uint64_t WL__ALIGN(16) k_[] = {0x3da6d0cb, 0xba4fc28e}; k = vld1q_u64(k_); }
            y0 = wl__clmul_lo_e(x0, k, x2), x0 = wl__clmul_hi_e(x0, k, y0);
            y4 = wl__clmul_lo_e(x4, k, x6), x4 = wl__clmul_hi_e(x4, k, y4);
            y8 = wl__clmul_lo_e(x8, k, x10), x8 = wl__clmul_hi_e(x8, k, y8);
            { static const uint64_t WL__ALIGN(16) k_[] = {0x740eef02, 0x9e4addf8}; k = vld1q_u64(k_); }
            y0 = wl__clmul_lo_e(x0, k, x4), x0 = wl__clmul_hi_e(x0, k, y0);
            x4 = x8;
            y0 = wl__clmul_lo_e(x0, k, x4), x0 = wl__clmul_hi_e(x0, k, y0);
            /* Reduce 128 bits to 32 bits, and multiply by x^32. */
            crc = __crc32cd(0, vgetq_lane_u64(x0, 0));
            crc = __crc32cd(crc, vgetq_lane_u64(x0, 1));
        }
        for (; size >= 8; buf += 8, size -= 8) crc = __crc32cd(crc, *(const uint64_t*)buf);
        for (; size; --size) crc = __crc32cb(crc, *buf++);
        return ~crc;
    #elif WL_INTRIN && (defined(__x86_64__) || defined(_M_X64))
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
            vc0 = wl__crc_shift(crc, (klen<<1) + 8);
            vc1 = wl__crc_shift(crc1, klen + 8);
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

typedef enum wl__format_type {
    WL__FMT_EOF, WL__FMT_ERR, WL__FMT_LIT, WL__FMT_INT,
    WL__FMT_UINT, WL__FMT_NUM, WL__FMT_STR, WL__FMT_CHAR,
    WL__FMT_PTR
} wl__format_type; /* Format types for formatted output */

typedef uint32_t wl__format_flags; /* Flags for formatting output */

/* Format flags */
#define WL__FMT_F_LEFT  0x0100 /* Left-align the output */
#define WL__FMT_F_PLUS  0x0200 /* Prefix positive numbers with a plus sign */
#define WL__FMT_F_ZERO  0x0400 /* Pad with zeros instead of spaces */
#define WL__FMT_F_SPACE 0x0800 /* Prefix a space for positive numbers */
#define WL__FMT_F_ALT   0x1000 /* Alternate format flag */
#define WL__FMT_F_UPPER 0x2000 /* Use uppercase letters for hex output */

/* Format subtypes (bits reused) */
#define WL__FMT_T_HEX   0x0010 /* Hexadecimal format for unsigned integers */
#define WL__FMT_T_OCT   0x0020 /* Octal format for unsigned integers */
#define WL__FMT_T_FP_A  0x0000 /* 'a' format for floating-point numbers */
#define WL__FMT_T_FP_E  0x0010 /* 'e' format for floating-point numbers */
#define WL__FMT_T_FP_F  0x0020 /* 'f' format for floating-point numbers */
#define WL__FMT_T_FP_G  0x0030 /* 'g' format for floating-point numbers */
#define WL__FMT_T_QUOTED 0x0010 /* Quoted string format */

#define WL__FMT_SH_WIDTH 16    /* Shift width for formatting */
#define WL__FMT_SH_PREC  24    /* Shift precision for formatting */
#define WL__FMT_TYPE(sf) ((wl__format_type)((sf) & 15))  /* Extract format type */
#define WL__FMT_WIDTH(sf) (((sf) >> WL__FMT_SH_WIDTH) & 255u) /* Extract width */
#define WL__FMT_PREC(sf) ((((sf) >> WL__FMT_SH_PREC) & 255u) - 1u) /* Extract precision */
#define WL__FMT_FP(sf) (((sf) >> 4) & 3) /* Extract floating-point format */

/* Formats for conversion characters */
#define WL__FMT_A (WL__FMT_NUM|WL__FMT_T_FP_A) /* 'a' format */
#define WL__FMT_C (WL__FMT_CHAR) /* 'c' format */
#define WL__FMT_D (WL__FMT_INT)  /* 'd' format */
#define WL__FMT_E (WL__FMT_NUM|WL__FMT_T_FP_E) /* 'e' format */
#define WL__FMT_F (WL__FMT_NUM|WL__FMT_T_FP_F) /* 'f' format */
#define WL__FMT_G (WL__FMT_NUM|WL__FMT_T_FP_G) /* 'g' format */
#define WL__FMT_I WL__FMT_D /* 'i' format (same as 'd') */
#define WL__FMT_O (WL__FMT_UINT|WL__FMT_T_OCT) /* 'o' format */
#define WL__FMT_P (WL__FMT_PTR) /* 'p' format */
#define WL__FMT_Q (WL__FMT_STR|WL__FMT_T_QUOTED) /* Quoted string */
#define WL__FMT_S (WL__FMT_STR) /* 's' format */
#define WL__FMT_U (WL__FMT_UINT) /* 'u' format */
#define WL__FMT_X (WL__FMT_UINT|WL__FMT_T_HEX) /* 'x' format */
#define WL__FMT_G14 (WL__FMT_G | ((14+1) << WL__FMT_SH_PREC)) /* 'g' format with precision 14 */

static char* wl__fmt_f64(wl__format_flags sf, double n, char* p);

typedef struct wl__hashset_t {
    size_t len;
    wl__bitset_t* used;
    const wl_tensor_t** keys;
    bool is_pool;
} wl__hashset_t;
#define WL__HASHSET_FULL ((size_t)-1)
#define WL__HASHSET_DUPLICATE ((size_t)-2)
#define WL__HASHSET_MAX ((size_t)-3) /* Must be last. */
#define wl__hashset_hash_fn(ptr) ((size_t)(uintptr_t)(ptr)>>3)

static size_t wl__hashset_compute_hash_size(size_t sz) {
    wl__assert2(sz > 0 && sz < WL__HASHSET_MAX);
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

static wl__hashset_t wl__hashset_create(size_t size) {
    size = wl__hashset_compute_hash_size(size);
    wl__hashset_t set = {
        .len = size,
        .used = (wl__bitset_t*)wl_alloc(NULL, wl__bitset_size(size)*sizeof(*set.used)),
        .keys = (const wl_tensor_t**)wl_alloc(NULL, size*sizeof(*set.keys)),
        .is_pool = false
    };
    memset(set.used, 0, wl__bitset_size(size)*sizeof(*set.used));
    return set;
}

static wl__hashset_t wl__hashset_create_pooled(wl_ctx_t* ctx, size_t size) {
    size = wl__hashset_compute_hash_size(size);
    wl__hashset_t set = {
        .len = size,
        .used = (wl__bitset_t*)wl_ctx_pool_alloc_aligned(ctx, wl__bitset_size(size)*sizeof(*set.used), __alignof__(*set.used)),
        .keys = (const wl_tensor_t**)wl_ctx_pool_alloc_aligned(ctx, size*sizeof(*set.keys), __alignof__(*set.used)),
        .is_pool = true
    };
    memset(set.used, 0, wl__bitset_size(size)*sizeof(*set.used));
    return set;
}

static size_t wl__hashset_lookup(wl__hashset_t* set, const wl_tensor_t* key) {
    size_t k = wl__hashset_hash_fn(key) % set->len, i = k;
    while (wl__bitset_get(set->used, i) && set->keys[i] != key) { /* Linear probing. */
        i = (i+1) % set->len;
        if (i == k) return WL__HASHSET_FULL;
    }
    return i;
}

static bool wl__hashset_contains_key(wl__hashset_t* set, const wl_tensor_t* key) {
    size_t i = wl__hashset_lookup(set, key);
    return wl__bitset_get(set->used, i) && i != WL__HASHSET_FULL;
}

static size_t wl__hashset_insert(wl__hashset_t* set, const wl_tensor_t* key) {
    size_t k = wl__hashset_hash_fn(key) % set->len, i = k;
    do { /* Linear probing. */
        if (!wl__bitset_get(set->used, i)) { /* Insert key. */
            wl__bitset_set(set->used, i);
            set->keys[i] = key;
            return i;
        }
        if (set->keys[i] == key) return WL__HASHSET_DUPLICATE; /* Key already exists. */
        i = (i+1) % set->len;
    } while (i != k);
    wl__panic("Insertion target not found");
}

static void wl__hashset_reset(wl__hashset_t* set) {
    memset(set->used, 0, wl__bitset_size(set->len)*sizeof(*set->used));
}

static void wl_hashset_destroy(wl__hashset_t* set) {
    wl__assert2(!set->is_pool); /* Cannot destroy pooled hashset. */
    wl_alloc(set->used, 0);
    wl_alloc(set->keys, 0);
}

static bool WL__AINLINE wl__imull64_ov(int64_t a, int64_t b, int64_t* out) { /* Performs c = a*b with overflow checking. Returns true on overflow, else false. */
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
static void wl__prng_generate_n(wl_ctx_t* ctx, float* out_gen, int64_t out_n, float min, float max) {
    float rescale_uniform = max - min;
    switch (ctx->prng_algorithm) {
        case WL_PRNG_MERSENNE_TWISTER: {
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
        case WL_PRNG_PCG: {
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
            wl__panic("Unknown PRNG algorithm: %d", ctx->prng_algorithm);
    }
}

static void wl__prng_init(wl_ctx_t* ctx, uint64_t seed) {
    seed = seed ? seed : 0x853c49e6748fea9bull ^ (uintptr_t)ctx ^ (uintptr_t)&ctx; /* Default seed. */
    switch (ctx->prng_algorithm) {
        case WL_PRNG_MERSENNE_TWISTER: {
            uint32_t* state = ctx->prng_state.mersenne.state;
            *state = (uint32_t)seed;
            for (size_t i=1; i < 624; ++i)
                state[i] = ((state[i-1] ^ (state[i-1] >> 30))*1812433253 + i) & ~0u;
            ctx->prng_state.mersenne.next = 0;
            ctx->prng_state.mersenne.remaining = 1;
        } break;
        case WL_PRNG_PCG: {
            ctx->prng_state.pcg.state = seed ^ 0x853c49e6748fea9bull;
            ctx->prng_state.pcg.inc = 0xda3e39cb94b95bdbull;
        } break;
        default:
            wl__panic("Unknown PRNG algorithm: %d", ctx->prng_algorithm);
    }
}

static void wl__ctx_push_chunk(wl_ctx_t* ctx) {
    uint8_t* chunk = (uint8_t*)(*ctx->alloc_fn)(NULL, ctx->pool.chunk_size);
    if (ctx->pool.warmup_chunks) memset(chunk, 0, ctx->pool.chunk_size);
    ctx->pool.mapped_total += ctx->pool.chunk_size;
    ctx->pool.delta = chunk + ctx->pool.chunk_size;
    if (ctx->pool.chunk_len == ctx->pool.chunk_cap)
        ctx->pool.chunks = (uint8_t**)(*ctx->alloc_fn)(ctx->pool.chunks, (ctx->pool.chunk_cap<<=1) * sizeof(*ctx->pool.chunks));
    ctx->pool.chunks[ctx->pool.chunk_len++] = chunk;
}

static void wl__system_host_info_query(wl_ctx_t* ctx); /* Query host system information. */
static void wl__blas_compute_dispatch_table_install(wl_ctx_t* ctx); /* Install BLAS dispatch table. */

#if defined(__x86_64__) || defined(_M_X64)
static bool wl__ctx_x86_64_cpu_has_feature(const wl_ctx_t* ctx, wl__x86_64_feature_t feature) {
    const uint8_t* leafs = wl__x86_64_feature_leaves, *regs = wl__x86_64_feature_regs;
    const uint32_t (*features)[8][4] = &ctx->sys.x86_64_cpu_features;
    return (*features)[leafs[feature]][regs[feature]] & 1u<<(uint32_t)feature;
}
#endif

#if WL_ENABLE_IMAGE_SUPPORT
    static uint8_t* wl_default_image_load_impl(const char* file, uint32_t(*whc)[3], wl_color_channels_t channels);
    static void wl_default_image_load_free_fn_impl(uint8_t*);
    static bool wl_default_image_save_impl(const char* file, const uint8_t* buf, const uint32_t(*whc)[3]);
#endif

wl_ctx_t* wl_ctx_create(const wl_ctx_info_t* info) {
    wl__log_info("Creating WAVELET context...");
    int64_t time_stamp_start = wl__hpc_clock_us();

    /* Print WAVELET version and compiler info. */
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
    wl__log_info("WAVELET v.%d.%d - " __DATE__ " " __TIME__ " - %s %d.%d", wl_version_major(WL_VERSION), wl_version_minor(WL_VERSION), compiler_name, compiler_version_major, compiler_version_minor);

    /* Enable fast math optimizations for x86-64 platforms. */
    #if WL_CFG_X86_64_FAST_MATH && (defined(__x86_64__) || defined(_M_X64))
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
    wl_ctx_info_t ctx_info = {0};
    if (info) ctx_info = *info;
    ctx_info.alloc_fn = ctx_info.alloc_fn ? ctx_info.alloc_fn : &wl_alloc; /* Use default allocator if not provided. */
    wl_ctx_t* ctx = (wl_ctx_t*)(*ctx_info.alloc_fn)(NULL, sizeof(*ctx)); /* Allocate context. */
    memset(ctx, 0, sizeof(*ctx));
    ctx->alloc_fn = ctx_info.alloc_fn;
    ctx->ud = ctx_info.user_data;
    ctx->pool.chunk_size = ctx_info.pool_chunk_size ? wl__max(ctx_info.pool_chunk_size, 8) : WL_DEFAULT_CHUNK_SIZE;
    ctx->pool.chunk_cap = ctx_info.pool_chunks_cap ? wl__max(ctx_info.pool_chunks_cap, 1) : WL_DEFAULT_CHUNK_CAP;
    ctx->pool.warmup_chunks = ctx_info.warmup_chunks;

    /* Query and print host system information. */
    wl__system_host_info_query(ctx);
    wl__log_info("OS/Kernel: %s", ctx->sys.os_name);
    wl__log_info("CPU: %s, Virtual Cores: %u, Physical Cores: %u, Sockets: %u", ctx->sys.cpu_name, ctx->sys.cpu_virtual_cores, ctx->sys.cpu_physical_cores, ctx->sys.cpu_sockets);
    #if defined(__x86_64__) || defined(_M_X64) /* Print CPU features for x86-64 platforms. */
        printf("CPU Features:");
        for (unsigned i=0, k=0; i < WL__X86_64_FEATURE__COUNT; ++i) {
            if (wl__ctx_x86_64_cpu_has_feature(ctx, i)) {
                if (k++ % 8 == 0) printf("\n\t");
                printf("%s ", wl__x86_64_feature_names[i]);
            }
        }
        putchar('\n');
    #endif
    double mem_total, mem_free, mem_used;
    const char* mem_unit_total, *mem_unit_free, *mem_unit_used;
    wl__humanize_memory_size(ctx->sys.phys_mem_total, &mem_total, &mem_unit_total);
    wl__humanize_memory_size(ctx->sys.phys_mem_free, &mem_free, &mem_unit_free);
    wl__humanize_memory_size((size_t)llabs((int64_t)ctx->sys.phys_mem_total-(int64_t)ctx->sys.phys_mem_free), &mem_used, &mem_unit_used);
    double mem_used_percent = fabs((double)(ctx->sys.phys_mem_total-ctx->sys.phys_mem_free))/(double)ctx->sys.phys_mem_total*100.0;
    wl__log_info("Physical memory: %.03f %s, Free: %.03f %s, Used: %.03f %s (%.02f%%)", mem_total, mem_unit_total, mem_free, mem_unit_free, mem_used, mem_unit_used, mem_used_percent);

    /* Prepare memory pool. */
    ctx->pool.chunks = (uint8_t**)(*ctx->alloc_fn)(NULL, ctx->pool.chunk_cap * sizeof(*ctx->pool.chunks)); /* Allocate chunk pointers. */
    wl__ctx_push_chunk(ctx); /* Allocate the first chunk. */

    #if WL_ENABLE_IMAGE_SUPPORT
        ctx->image_load_fn = ctx_info.image_load_fn ? ctx_info.image_load_fn : &wl_default_image_load_impl;
        ctx->image_load_free_fn = ctx_info.image_load_free_fn ? ctx_info.image_load_free_fn : &wl_default_image_load_free_fn_impl;
        ctx->image_save_fn = ctx_info.image_save_fn ? ctx_info.image_save_fn : &wl_default_image_save_impl;
    #else
        ctx->image_load_fn = ctx_info.image_load_fn;
        ctx->image_save_fn = ctx_info.image_save_fn;
    #endif

    /* Initialize PRNG state. */
    uint64_t host_tid = wl__thread_id();
    ctx->prng_algorithm = ctx_info.prng_algorithm;
    wl__prng_init(ctx, ctx_info.prng_seed^host_tid^(uintptr_t)ctx^(uintptr_t)&ctx_info); /* Initialize PRNG state. */
    ctx->host_thread_id = host_tid;

    /* Install BLAS dispatch table, specialized for host CPU arch. */
    ctx->exec_mode = ctx_info.exec_mode;
    wl__blas_compute_dispatch_table_install(ctx);

    /* Print context initialization time. */
    wl__log_info("WAVELET context initialized in %.05f ms.", wl__hpc_clock_elapsed_ms(time_stamp_start));
    return ctx;
}

wl_ctx_t* wl_ctx_create2(size_t pool_chunk_size) {
    wl_ctx_info_t info = {0};
    info.pool_chunk_size = pool_chunk_size;
    return wl_ctx_create(&info);
}

void* wl_ctx_pool_alloc(wl_ctx_t* ctx, size_t size) {
    wl__assert(size > 0 && size < (size_t)PTRDIFF_MAX, "Allocation size must be within (0, %zu), but is: %zu", PTRDIFF_MAX, size);
    if (ctx->pool.delta - ctx->pool.chunks[ctx->pool.chunk_len-1] < (ptrdiff_t)size) {
        if (ctx->pool.chunk_size < size) { /* Increase the chunk size if it's too small to accommodate the requested length */
            size_t lim = (size_t)PTRDIFF_MAX >> 1;
            do ctx->pool.chunk_size <<= 1;
            while (ctx->pool.chunk_size < size && (ctx->pool.chunk_size <= lim));
        }
        wl__ctx_push_chunk(ctx);
        wl__log_info("Allocated pool chunk: %.03f MiB", (double)ctx->pool.chunk_size/(double)(1<<20));
    }
    ctx->pool.delta -= size;
    ++ctx->pool.alloc_acc;
    ctx->pool.alloc_total += size;
    return ctx->pool.delta;
}

void* wl_ctx_pool_alloc_aligned(wl_ctx_t* ctx, size_t size, size_t align) {
    wl__assert(align && !(align&(align-1)), "Alignment must be power of 2: %zu", align); /* Alignment must be a power of 2 */
    return (void*)(((uintptr_t)wl_ctx_pool_alloc(ctx, size+align-1)+align-1)&-align);
}

size_t wl_ctx_total_allocated_pool_memory(const wl_ctx_t* ctx) {
    size_t mem = sizeof(*ctx);
    mem += sizeof(*ctx->pool.chunks) * ctx->pool.chunk_cap;
    mem += ctx->pool.alloc_total;
    return mem;
}

wl_exec_mode_t wl_ctx_get_exec_mode(const wl_ctx_t* ctx) { return ctx->exec_mode; }

void wl_ctx_set_exec_mode(wl_ctx_t* ctx, wl_exec_mode_t mode) {
    ctx->exec_mode = mode;
    wl__log_info("Execution mode set to: %s", mode == WL_EXEC_MODE_EAGER ? "Eager" : "Deferred");
}

wl_prng_algorithm_t wl_ctx_get_prng_algorithm(const wl_ctx_t* ctx) { return ctx->prng_algorithm; }

void wl_ctx_set_prng_algorithm(wl_ctx_t* ctx, wl_prng_algorithm_t algorithm, uint64_t seed) {
    ctx->prng_algorithm = algorithm;
    wl__prng_init(ctx, seed); /* Reinitialize PRNG state with new seed. */
}

const char* wl_ctx_get_os_name(const wl_ctx_t* ctx) { return ctx->sys.os_name; }
const char* wl_ctx_get_cpu_name(const wl_ctx_t* ctx) { return ctx->sys.cpu_name; }
uint32_t wl_ctx_get_cpu_virtual_cores(const wl_ctx_t* ctx) { return ctx->sys.cpu_virtual_cores; }
uint32_t wl_ctx_get_cpu_physical_cores(const wl_ctx_t* ctx) { return ctx->sys.cpu_physical_cores; }
uint32_t wl_ctx_get_cpu_sockets(const wl_ctx_t* ctx) { return ctx->sys.cpu_sockets; }
uint64_t wl_ctx_get_physical_memory_total(const wl_ctx_t* ctx) { return ctx->sys.phys_mem_total; }
uint64_t wl_ctx_get_physical_memory_free(const wl_ctx_t* ctx) { return ctx->sys.phys_mem_free; }
bool wl_ctx_is_numa_system(const wl_ctx_t* ctx) { return false; /* TODO */ }

void wl_ctx_destroy(wl_ctx_t* ctx) {
    size_t mem_total = wl_ctx_total_allocated_pool_memory(ctx);
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
    wl__humanize_memory_size(mem_total, &alloc_total, &alloc_unit);
    wl__humanize_memory_size(mem_mapped, &mapped_total, &mapped_unit);
    wl__log_info("Allocated in pool: %.03f %s, Mapped memory: %.03f %s", alloc_total, alloc_unit, mapped_total, mapped_unit);
    wl__log_info("WAVELET context destroyed.");
}

#define wl__op_param_pack_u64(tag, x) ((wl_op_param_t)(((x)&((1ull<<(64-2))-1))|(((uint64_t)(tag)&3)<<(64-2))))
#define wl__op_param_is_tag(param, tag) ((((param)>>(64-2))&3) == (tag))
#define wl__op_param_unpack_u64(param) ((uint64_t)(param)&((1ull<<(64-2))-1))

wl_op_param_t wl_op_param_int(uint64_t x) {
    return wl__op_param_pack_u64(WL_OP_PARAM_INT, x);
}

bool wl_op_param_is_int(wl_op_param_t param) {
    return wl__op_param_is_tag(param, WL_OP_PARAM_INT);
}

uint64_t wl_op_param_unpack_int(wl_op_param_t param) {
    wl__assert2(wl_op_param_is_int(param));
    return wl__op_param_unpack_u64(param);
}

#undef wl__op_param_unpack_u64
#undef wl__op_param_is_tag
#undef wl__op_param_pack_u64

uint32_t wl_pack_color_u8(uint8_t r, uint8_t g, uint8_t b) {
    return ((uint32_t)b<<16)|((uint32_t)g<<8)|(uint32_t)r;
}

uint32_t wl_pack_color_f32(float r, float g, float b) {
    return (((uint32_t)(b*255.0f)&255)<<16)|(((uint32_t)(g*255.0f)&255)<<8)|((uint32_t)(r*255.0f)&255);
}

#define wl__load_local_storage_group_arr(arr, prefix) \
    const int64_t prefix##0 = (arr)[0]; \
    const int64_t prefix##1 = (arr)[1]; \
    const int64_t prefix##2 = (arr)[2]; \
    const int64_t prefix##3 = (arr)[3]; \
    const int64_t prefix##4 = (arr)[4]; \
    const int64_t prefix##5 = (arr)[5]; \
    (void)prefix##0; \
    (void)prefix##1; \
    (void)prefix##2; \
    (void)prefix##3; \
    (void)prefix##4; \
    (void)prefix##5

#define wl__load_local_storage_group(xk, prefix, var) wl__load_local_storage_group_arr((xk)->var, prefix)

const wl_dtype_info_t* wl_dtype_info_of(wl_dtype_t type) {
    static const wl_dtype_info_t infos[WL_DTYPE_COUNT_] = {
        [WL_DTYPE_F32] = {
            sizeof(float),
            "f32"
        },
    };
    return &infos[type];
}

const char* wl_op_get_name(wl_op_t op) {
    #define _(enumerator, mnemonic, argcount) #enumerator
    static const char* const names[WL_OP__COUNT] = {
        wl_op_def(_, WL_SEP)
    };
    #undef _
    return names[op];
}

const char* wl_op_get_mnemonic(wl_op_t op) {
    #define _(enumerator, mnemonic, argcount) mnemonic
    static const char* const mnemonics[WL_OP__COUNT] = {
        wl_op_def(_, WL_SEP)
    };
    #undef _
    return mnemonics[op];
}

uint8_t wl_op_get_argcount(wl_op_t op) {
    #define _(enumerator, mnemonic, argcount) ((argcount)&0xff)
    static const uint8_t arg_counts[WL_OP__COUNT] = {
        wl_op_def(_, WL_SEP)
    };
    #undef _
    return arg_counts[op];
}

/*
**  validation error print template
**
**
printf("SHORT ERROR DESCRIPTION"
        "ERROR: Failed to execute operation: %s.\n"
        "    - Input Tensor 1 '%s': MISMATCHES\n"
        "    - Input Tensor 2 '%s': MISMATCHES\n"
        "    Hint: ANY HINT FOR USR."
);
*/

static void WL__COLDPROC wl__validate_print_separator(void) {
    for (uint32_t i=0; i <= 128; ++i) fputc('=', stderr);
    fputc('\n', stderr);
}

static bool wl__validate_inputs(wl_op_t op, wl_tensor_t** inputs, uint32_t n_inputs) {
    if (wl__unlikely(n_inputs > WL_MAX_INPUT_TENSORS)) {
        wl__validate_print_separator();
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation requires at most %u input tensors, but %u were provided.\n"
            "    Hint: Ensure the correct number of input tensors are provided.\n",
            wl_op_get_name(op), WL_MAX_INPUT_TENSORS, n_inputs
        );
        wl__validate_print_separator();
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    if (wl__unlikely(wl_op_get_argcount(op) != n_inputs)) {
        wl__validate_print_separator();
        fprintf(stderr,
            "Failed to execute operation: %s.\n"
            "ERROR: Operation requires %u input tensors, but %u were provided.\n"
            "    Hint: Ensure the correct number of input tensors are provided.\n",
            wl_op_get_name(op), wl_op_get_argcount(op), n_inputs
        );
        wl__validate_print_separator();
        fputc('\n', stderr);
        fflush(stderr);
        return false;
    }
    for (uint32_t i=0; i < wl_op_get_argcount(op); ++i) {
        if (wl__unlikely(!inputs[i])) {
            wl__validate_print_separator();
            fprintf(stderr,
                "Failed to execute operation: %s.\n"
                "ERROR: Input tensor %u is NULL.\n"
                "    Hint: Ensure all input tensors are valid and non-NULL.\n",
                wl_op_get_name(op), i
            );
            wl__validate_print_separator();
            fputc('\n', stderr);
            fflush(stderr);
            return false;
        }
    }
    return true;
}

static bool wl__validate_shape_eq(wl_op_t op, const wl_tensor_t* a, const wl_tensor_t* b) {
    if (wl__likely(wl_tensor_is_shape_eq(a, b))) return true;
    wl__validate_print_separator();
    char shape_1[WL__FMT_DIM_BUF_SIZE];
    char shape_2[WL__FMT_DIM_BUF_SIZE];
    wl__fmt_dims(&shape_1, &a->shape, a->rank);
    wl__fmt_dims(&shape_2, &b->shape, b->rank);
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor shapes must be equal.\n"
        "    - Input Tensor 1 '%s' Shape: %s\n"
        "    - Input Tensor 2 '%s' Shape: %s\n"
        "    Hint: Adjust tensor shapes using transposition or permutation.\n",
        wl_op_get_name(op),
        a->name, shape_1,
        b->name, shape_2
    );
    wl__validate_print_separator();
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

static bool wl__validate_shape_broadcastable(wl_op_t op, const wl_tensor_t* a, const wl_tensor_t* b) { /* Check if tensor shapes are broadcast-able. (b into a) */
    if (wl__likely(wl_tensor_can_broadcast(b, a))) return true;
    wl__validate_print_separator();
    char shape_1[WL__FMT_DIM_BUF_SIZE];
    char shape_2[WL__FMT_DIM_BUF_SIZE];
    wl__fmt_dims(&shape_1, &a->shape, a->rank);
    wl__fmt_dims(&shape_2, &b->shape, b->rank);
    bool broadcast_able[WL_MAX_DIMS] = {0};
    for (uint32_t i=0; i < WL_MAX_DIMS; ++i)
        broadcast_able[i] = a->shape[i] % b->shape[i] == 0;
    char broadcast_able_str[WL_MAX_DIMS*2+4+1] = {0};
    char* p = broadcast_able_str;
    *p++ = '[';
    for (uint32_t i=0; i < WL_MAX_DIMS; ++i) {
        *p++ = broadcast_able[i] ? 'Y' : 'N';
        *p++ = i < WL_MAX_DIMS-1 ? ',' : ']';
    }
    *p = '\0';
    fprintf(stderr,
        "Failed to execute operation: %s.\n"
        "ERROR: Input tensor shapes must be broadcast-able.\n"
        "    - Input Tensor 1 '%s' Shape: %s\n"
        "    - Input Tensor 2 '%s' Shape: %s\n"
        "    Broadcast-able: %s\n"
        "    Hint: Adjust tensor shapes using transposition or permutation.\n",
        wl_op_get_name(op),
        a->name, shape_1,
        b->name, shape_2,
        broadcast_able_str
    );
    wl__validate_print_separator();
    fputc('\n', stderr);
    fflush(stderr);
    return false;
}

#define wl__validate_expr_gen(expr, message, ...) \
    if (wl__unlikely(!(expr))) { \
        if (1) { \
           wl__log_error(message, ## __VA_ARGS__); \
        } \
        return false; \
    }

static bool wl__validate_op_nop(wl_op_t op, wl_tensor_t* result, wl_tensor_t** inputs, uint32_t n_inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    (void)op, (void)result, (void)inputs, (void)n_inputs, (void)params;
    return true;
}

static bool wl__validate_op_unary(wl_op_t op, wl_tensor_t* result, wl_tensor_t** inputs, uint32_t n_inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    (void)params;
    if (wl__unlikely(!wl__validate_inputs(op, inputs, n_inputs))) return false;
    if (wl__unlikely(!wl__validate_shape_eq(op, result, inputs[0]))) return false;
    return true;
}

static bool wl__validate_op_binary(wl_op_t op, wl_tensor_t* result, wl_tensor_t** inputs, uint32_t n_inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    (void)params;
    if (wl__unlikely(!wl__validate_inputs(op, inputs, n_inputs))) return false;
    if (wl__unlikely(!wl__validate_shape_eq(op, result, inputs[0]))) return false;
    if (wl__unlikely(!wl__validate_shape_broadcastable(op, inputs[0], inputs[1]))) return false;
    wl__validate_expr_gen(result->strides[0] == wl_dtype_info_of(result->dtype)->size, "Result must be contiguous.");
    wl__validate_expr_gen(inputs[0]->strides[0] == wl_dtype_info_of(result->dtype)->size, "First tensor must be contiguous.");
    return true;
}

static bool wl__validate_op_transpose(wl_op_t op, wl_tensor_t* result, wl_tensor_t** inputs, uint32_t n_inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    (void)params;
    if (wl__unlikely(!wl__validate_inputs(op, inputs, n_inputs))) return false;
    return true;
}

static bool wl__validate_op_scalar(wl_op_t op, wl_tensor_t* result, wl_tensor_t** inputs, uint32_t n_inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    (void)params;
    if (wl__unlikely(!wl__validate_inputs(op, inputs, n_inputs))) return false;
    wl__validate_expr_gen(inputs[0]->strides[0] == sizeof(float), "Mean");
    wl__validate_expr_gen(result->shape[0] == 1, "Mean");
    wl__validate_expr_gen(result->shape[1] == inputs[0]->shape[1], "Mean");
    wl__validate_expr_gen(result->shape[2] == inputs[0]->shape[2], "Mean");
    wl__validate_expr_gen(result->shape[3] == inputs[0]->shape[3], "Mean");
    return true;
}

static bool wl__validate_op_matmul(wl_op_t op, wl_tensor_t* result, wl_tensor_t** inputs, uint32_t n_inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    if (wl__unlikely(!wl__validate_inputs(op, inputs, n_inputs))) return false;
    wl__validate_expr_gen(inputs[0]->shape[1] == inputs[1]->shape[0], "Input tensor shapes must be compatible for matrix multiplication.");
    wl__validate_expr_gen(wl_tensor_is_contiguous(inputs[0]), "First tensor must be contiguous.");
    wl__validate_expr_gen(wl_tensor_is_contiguous(inputs[1]), "Second tensor must be contiguous.");
    wl__validate_expr_gen(inputs[1]->shape[2] % inputs[0]->shape[2] == 0, "Result tensor shape mismatch.");
    wl__validate_expr_gen(inputs[1]->shape[3] % inputs[0]->shape[3] == 0, "Result tensor shape mismatch.");
    return true;
}

static bool (*wl__op_get_validator_routine(wl_op_t op))(wl_op_t, wl_tensor_t*, wl_tensor_t**, uint32_t, const wl_op_param_t(*)[WL_MAX_OP_PARAMS]) {
    static bool (*const routines[WL_OP__COUNT])(wl_op_t, wl_tensor_t*, wl_tensor_t**, uint32_t, const wl_op_param_t(*)[WL_MAX_OP_PARAMS]) = {
        [WL_OP_NOP] = &wl__validate_op_nop,
        [WL_OP_CLONE] = &wl__validate_op_unary,
        [WL_OP_VIEW] = &wl__validate_op_unary,
        [WL_OP_TRANSPOSE] = &wl__validate_op_transpose,
        [WL_OP_PERMUTE] = &wl__validate_op_transpose,
        [WL_OP_MEAN] = &wl__validate_op_scalar,
        [WL_OP_SUM] = &wl__validate_op_scalar,
        [WL_OP_ABS] = &wl__validate_op_unary,
        [WL_OP_NEG] = &wl__validate_op_unary,
        [WL_OP_LOG] = &wl__validate_op_unary,
        [WL_OP_SQR] = &wl__validate_op_unary,
        [WL_OP_SQRT] = &wl__validate_op_unary,
        [WL_OP_SIN] = &wl__validate_op_unary,
        [WL_OP_COS] = &wl__validate_op_unary,
        [WL_OP_STEP] = &wl__validate_op_unary,
        [WL_OP_SOFTMAX] = &wl__validate_op_unary,
        [WL_OP_SOFTMAX_DV] = &wl__validate_op_unary,
        [WL_OP_SIGMOID] = &wl__validate_op_unary,
        [WL_OP_SIGMOID_DV] = &wl__validate_op_unary,
        [WL_OP_HARD_SIGMOID] = &wl__validate_op_unary,
        [WL_OP_SILU] = &wl__validate_op_unary,
        [WL_OP_SILU_DV] = &wl__validate_op_unary,
        [WL_OP_TANH] = &wl__validate_op_unary,
        [WL_OP_TANH_DV] = &wl__validate_op_unary,
        [WL_OP_RELU] = &wl__validate_op_unary,
        [WL_OP_RELU_DV] = &wl__validate_op_unary,
        [WL_OP_GELU] = &wl__validate_op_unary,
        [WL_OP_GELU_DV] = &wl__validate_op_unary,
        [WL_OP_ADD] = &wl__validate_op_binary,
        [WL_OP_SUB] = &wl__validate_op_binary,
        [WL_OP_MUL] = &wl__validate_op_binary,
        [WL_OP_DIV] = &wl__validate_op_binary,
        [WL_OP_MATMUL] = &wl__validate_op_matmul,
    };
    return routines[op];
}

static wl_tensor_t* wl__tensor_create(wl_ctx_t* ctx, wl_dtype_t type, const int64_t* dims, int64_t rank, wl_tensor_t* view, size_t view_offs);

static wl_tensor_t* wl__result_constructor_routine_nop(wl_tensor_t** inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    (void)inputs, (void)params;
    return NULL;
}

static wl_tensor_t* wl__result_constructor_routine_isomorph_same_shape(wl_tensor_t** inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    (void)params;
    return wl__tensor_create(inputs[0]->ctx, inputs[0]->dtype, inputs[0]->shape, inputs[0]->rank, NULL, 0);
}

static wl_tensor_t* wl__result_constructor_routine_isomorph(wl_tensor_t** inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    (void)params;
    return wl__tensor_create(inputs[0]->ctx, inputs[0]->dtype, inputs[0]->shape, WL_MAX_DIMS, NULL, 0);
}

static wl_tensor_t* wl__result_constructor_routine_view(wl_tensor_t** inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    (void)params;
    return wl__tensor_create(inputs[0]->ctx, inputs[0]->dtype, inputs[0]->shape, WL_MAX_DIMS, inputs[0], 0);
}

static wl_tensor_t* wl__result_constructor_routine_scalar(wl_tensor_t** inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    int64_t shape[WL_MAX_DIMS];
    *shape = 1;
    #pragma GCC unroll 5
    for (uint32_t i=1; i < WL_MAX_DIMS; ++i)
        shape[i] = inputs[0]->shape[i];
    return wl__tensor_create(inputs[0]->ctx, inputs[0]->dtype, shape, WL_MAX_DIMS, NULL, 0);
}

static wl_tensor_t* wl__result_constructor_routine_transposed(wl_tensor_t** inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    wl_tensor_t* transposed = wl__result_constructor_routine_view(inputs, params);
    wl__swap(int64_t, transposed->shape[0], transposed->shape[1]);
    wl__swap(int64_t, transposed->strides[0], transposed->strides[1]);
    return transposed;
}

static wl_tensor_t* wl__result_constructor_routine_permuted(wl_tensor_t** inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    wl__assert2(params != NULL); /* TODO */
    wl_tensor_t* permuted = wl__result_constructor_routine_view(inputs, params);
    uint32_t axes[WL_MAX_DIMS];
    for (uint32_t i = 0; i < WL_MAX_DIMS; ++i) /* Unpack axes */
        axes[i] = wl_op_param_unpack_int((*params)[i]);
    for (uint32_t i = 0; i < WL_MAX_DIMS; ++i) { /* Check that all axes are unique */
        for (uint32_t j = i+1; j < WL_MAX_DIMS; ++j)
            wl__assert(axes[i] != axes[j], "Axes must be unique: %zu != %zu", axes[i], axes[j]);
    }
    for (uint32_t i=0; i < WL_MAX_DIMS; ++i) { /* Permute shape and strides */
        wl__assert2(axes[i] >= 0 && axes[i] < WL_MAX_DIMS);
        permuted->shape[axes[i]] = inputs[0]->shape[i];
        permuted->strides[axes[i]] = inputs[0]->strides[i];
    }
    return permuted;
}

static wl_tensor_t* wl__result_constructor_routine_matmul(wl_tensor_t** inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) { /* MxR = MxN * NxR */
    (void)params;
    int64_t shape[WL_MAX_DIMS];
    shape[0] = inputs[0]->shape[0]; /* M */
    shape[1] = inputs[1]->shape[1]; /* R */
    return wl__tensor_create(inputs[0]->ctx, WL_DTYPE_F32, shape, 2, NULL, 0);
}

static wl_tensor_t* (*wl__op_get_result_constructor_routine(wl_op_t op))(wl_tensor_t**, const wl_op_param_t(*)[WL_MAX_OP_PARAMS]) {
    static wl_tensor_t* (*const routines[])(wl_tensor_t**, const wl_op_param_t(*)[WL_MAX_OP_PARAMS]) = {
        [WL_OP_NOP] = &wl__result_constructor_routine_nop,
        [WL_OP_CLONE] = &wl__result_constructor_routine_isomorph_same_shape,
        [WL_OP_VIEW] = &wl__result_constructor_routine_view,
        [WL_OP_TRANSPOSE] = &wl__result_constructor_routine_transposed,
        [WL_OP_PERMUTE] = &wl__result_constructor_routine_permuted,
        [WL_OP_MEAN] = &wl__result_constructor_routine_scalar,
        [WL_OP_SUM] = &wl__result_constructor_routine_scalar,
        [WL_OP_ABS] = &wl__result_constructor_routine_isomorph,
        [WL_OP_NEG] = &wl__result_constructor_routine_isomorph,
        [WL_OP_LOG] = &wl__result_constructor_routine_isomorph,
        [WL_OP_SQR] = &wl__result_constructor_routine_isomorph,
        [WL_OP_SQRT] = &wl__result_constructor_routine_isomorph,
        [WL_OP_SIN] = &wl__result_constructor_routine_isomorph,
        [WL_OP_COS] = &wl__result_constructor_routine_isomorph,
        [WL_OP_STEP] = &wl__result_constructor_routine_isomorph,
        [WL_OP_SOFTMAX] = &wl__result_constructor_routine_isomorph,
        [WL_OP_SOFTMAX_DV] = &wl__result_constructor_routine_isomorph,
        [WL_OP_SIGMOID] = &wl__result_constructor_routine_isomorph,
        [WL_OP_SIGMOID_DV] = &wl__result_constructor_routine_isomorph,
        [WL_OP_HARD_SIGMOID] = &wl__result_constructor_routine_isomorph,
        [WL_OP_SILU] = &wl__result_constructor_routine_isomorph,
        [WL_OP_SILU_DV] = &wl__result_constructor_routine_isomorph,
        [WL_OP_TANH] = &wl__result_constructor_routine_isomorph,
        [WL_OP_TANH_DV] = &wl__result_constructor_routine_isomorph,
        [WL_OP_RELU] = &wl__result_constructor_routine_isomorph,
        [WL_OP_RELU_DV] = &wl__result_constructor_routine_isomorph,
        [WL_OP_GELU] = &wl__result_constructor_routine_isomorph,
        [WL_OP_GELU_DV] = &wl__result_constructor_routine_isomorph,
        [WL_OP_ADD] = &wl__result_constructor_routine_isomorph,
        [WL_OP_SUB] = &wl__result_constructor_routine_isomorph,
        [WL_OP_MUL] = &wl__result_constructor_routine_isomorph,
        [WL_OP_DIV] = &wl__result_constructor_routine_isomorph,
        [WL_OP_MATMUL] = &wl__result_constructor_routine_matmul,
    };
    wl_static_assert(WL_OP__COUNT == sizeof(routines)/sizeof(*routines));
    return routines[op];
}

#undef wl__validate_inputs

static WL__AINLINE int64_t wl__tensor_data_size(const wl_tensor_t* t) { return t->num_elems*wl_dtype_info_of(t->dtype)->size; }
int64_t wl_tensor_data_size(const wl_tensor_t* t) { return wl__tensor_data_size(t); }
static WL__AINLINE int64_t wl__tensor_num_elements(const wl_tensor_t* t) { return t->num_elems; }
int64_t wl_tensor_num_elements(const wl_tensor_t* t) { return t->num_elems; }
static WL__AINLINE int64_t wl__tensor_num_rows(const wl_tensor_t* t) {
    wl_static_assert(WL_MAX_DIMS == 6);
    wl__load_local_storage_group(t, d, shape);
    return d1*d2*d3*d4*d5;
}
int64_t wl_tensor_num_rows(const wl_tensor_t* t) { return wl__tensor_num_rows(t); }
static WL__AINLINE int64_t wl__tensor_num_cols(const wl_tensor_t* t) { return *t->shape; }
int64_t wl_tensor_num_cols(const wl_tensor_t* t) { return *t->shape; }

static wl_tensor_t* wl__tensor_create(wl_ctx_t* ctx, wl_dtype_t type, const int64_t* dims, int64_t rank, wl_tensor_t* view, size_t view_offs) {
    wl__assert(dims != NULL && rank >= 0 && rank <= WL_MAX_DIMS, "Rank must be within (0, %d]", WL_MAX_DIMS);
    if (view && view->view) { /* Accumulate relative view offset. */
        view_offs += view->view_offs;
        view = view->view;
    }
    int64_t scalar_size = wl_dtype_info_of(type)->size;
    int64_t elems_total = 1;
    for (int64_t i=0; i < rank; ++i) /* Calculate buffer size and check for overflow. */
        wl__assert2(dims[i] > 0 && !wl__imull64_ov(dims[i], elems_total, &elems_total)); /* Overflow in buffer size. Max: INT64_MAX. Reduce dimensions. */
    int64_t bytes_total = elems_total*scalar_size;
    wl__assert2(!view || !bytes_total || bytes_total + view_offs <= wl__tensor_data_size(view)); /* Slice must be within viewed tensor data range. */
    wl_tensor_t* t = (wl_tensor_t*)wl_ctx_pool_alloc(ctx, sizeof(*t) + (view ? 0 : bytes_total)); /* Allocate memory for tensor struct and data */
    memset(t, 0, sizeof(*t));
    *t = (wl_tensor_t) {
        .name = "tensor",
        .ctx = ctx,
        .rank = rank,
        .dtype = type,
        .num_elems = elems_total,
        .flags = view ? WL__TFLAG_VIEW : WL__TFLAG_NONE,
        .view = view,
        .view_offs = view_offs,
    };
    #pragma GCC unroll 6
    for (uint32_t i=0; i < WL_MAX_DIMS; ++i) /* Copy dimensions and set unused to identity. */
        t->shape[i] = i < rank ? dims[i] : 1;
    *t->strides = scalar_size;
    #pragma GCC unroll 5
    for (uint32_t i=1; i < WL_MAX_DIMS; ++i)    /* Calculate strides and check for overflow. */
        wl__assert2(!wl__imull64_ov(t->strides[i-1], t->shape[i-1], t->strides+i));
    t->buf = view ? (uint8_t*)view->buf + view_offs : (uint8_t*)(t + 1); /* Set buffer pointer to the end of the tensor struct, where data follows */
    return t;
}

wl_tensor_t* wl_tensor_create_1d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1) {
    return wl__tensor_create(ctx, type, (int64_t[]) {d1}, 1, NULL, 0);
}

wl_tensor_t* wl_tensor_create_2d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1, int64_t d2) {
    return wl__tensor_create(ctx, type, (int64_t[]) {d1, d2}, 2, NULL, 0);
}

wl_tensor_t* wl_tensor_create_3d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1, int64_t d2, int64_t d3) {
    return wl__tensor_create(ctx, type, (int64_t[]) {d1, d2, d3}, 3, NULL, 0);
}

wl_tensor_t* wl_tensor_create_4d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
    return wl__tensor_create(ctx, type, (int64_t[]) {d1, d2, d3, d4}, 4, NULL, 0);
}

wl_tensor_t* wl_tensor_create_5d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5) {
    return wl__tensor_create(ctx, type, (int64_t[]) {d1, d2, d3, d4, d5}, 5, NULL, 0);
}

wl_tensor_t* wl_tensor_create_6d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6) {
    return wl__tensor_create(ctx, type, (int64_t[]) {d1, d2, d3, d4, d5, d6}, 6, NULL, 0);
}

struct wl__blas_compute_info_t {
    wl_ctx_t* ctx;
    int64_t n_threads;
    int64_t thread_idx;
};

static void wl__blas_compute_info_sequential(wl_ctx_t* ctx, wl__blas_compute_info_t* bci) {
    *bci = (wl__blas_compute_info_t){
        .ctx = ctx,
        .n_threads = 1,
        .thread_idx = 0
    };
}

static void wl__blas_compute_info_parallel(wl_ctx_t* ctx, wl__blas_compute_info_t* bci, uint32_t n_threads) {
    *bci = (wl__blas_compute_info_t){
        .ctx = ctx,
        .n_threads = wl__max(1, n_threads),
        .thread_idx = 0
    };
}

static void WL__AINLINE wl__op_execute(wl_tensor_t* R, wl_op_t op, const wl_tensor_t** inputs, const wl__blas_compute_info_t* bci) {
    void (**dispatch_lut)(const wl__blas_compute_info_t*, wl_tensor_t*, const wl_tensor_t**) = bci->ctx->blas_dispatch; /* Dispatch table */
    (*(*(dispatch_lut+op)))(bci, R, inputs); /* Dispatch to operation. */
}

wl_tensor_t* wl_tensor_operator(wl_ctx_t* ctx, wl_op_t op, wl_tensor_t** inputs, uint32_t n_inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]) {
    wl__assert2(op != WL_OP_NOP && n_inputs <= WL_MAX_INPUT_TENSORS);
    wl_tensor_t* (*construct_result)(wl_tensor_t**, const wl_op_param_t(*)[WL_MAX_OP_PARAMS]) = wl__op_get_result_constructor_routine(op);
    bool (*validate_op)(wl_op_t, wl_tensor_t*, wl_tensor_t**, uint32_t, const wl_op_param_t(*)[WL_MAX_OP_PARAMS]) = wl__op_get_validator_routine(op);
    wl_tensor_t* R = (*construct_result)(inputs, params);
    if (wl__unlikely(!(*validate_op)(op, R, inputs, n_inputs, params))) return NULL;
    R->flags |= WL__TFLAG_OP_OUTPUT;
    wl__assert2(R->op == WL_OP_NOP);
    R->op = op; /* Set operation for deferred execution mode. */
    for (uint32_t i=0; i < n_inputs; ++i) { /* Set input tensors and flags. */
        inputs[i]->flags |= WL__TFLAG_OP_INPUT;
        R->op_inputs[i] = inputs[i];
    }
    if (params) memcpy(R->op_params, *params, sizeof(*params)); /* Copy operation parameters */
    if (ctx->exec_mode == WL_EXEC_MODE_EAGER) { /* In eager execution mode, we execute immediately. */
        wl__blas_compute_info_t bci;
        wl__blas_compute_info_sequential(ctx, &bci); /* Sequential eager execution. */
        wl__op_execute(R, op, (const wl_tensor_t**)inputs, &bci); /* Execute the operation immediately. */
    }
    return R;
}

static WL__AINLINE void wl_tensor_virtual_to_physical_index(const wl_tensor_t* t, int64_t v_idx, int64_t(*p_idx)[WL_MAX_DIMS]) {
    wl_static_assert(WL_MAX_DIMS == 6);
    wl__load_local_storage_group(t, d, shape);
    (*p_idx)[5] = v_idx / (d4*d3*d2*d1*d0);
    (*p_idx)[4] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0) / (d3*d2*d1*d0);
    (*p_idx)[3] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0) / (d2*d1*d0);
    (*p_idx)[2] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0) / (d1*d0);
    (*p_idx)[1] = (v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0) / d0;
    (*p_idx)[0] =  v_idx - (*p_idx)[5]*d4*d3*d2*d1*d0 - (*p_idx)[4]*d3*d2*d1*d0 - (*p_idx)[3]*d2*d1*d0 - (*p_idx)[2]*d1*d0 - (*p_idx)[1]*d0;
}

static WL__AINLINE int64_t wl_tensor_physical_to_virtual_index(const wl_tensor_t* t, const int64_t (*p_idx)[WL_MAX_DIMS]) {
    wl_static_assert(WL_MAX_DIMS == 6);
    wl__load_local_storage_group(t, s, strides);
    wl__load_local_storage_group_arr(*p_idx, i);
    return s0*i0 + s1*i1 + s2*i2 + s3*i3 + s4*i4 + s5*i5;
}

wl_tensor_t* wl_tensor_get_arg(const wl_tensor_t* t, size_t slot) {
    wl__assert(slot < WL_MAX_INPUT_TENSORS, "Slot must be within [0, %d)", WL_MAX_INPUT_TENSORS);
    return t->op_inputs[slot];
}

void wl_tensor_set_arg(wl_tensor_t* t, size_t slot, wl_tensor_t* arg) {
    wl__assert(slot < WL_MAX_INPUT_TENSORS, "Slot must be within [0, %d)", WL_MAX_INPUT_TENSORS);
    wl__assert(t->op_inputs[slot] == NULL, "Argument at slot #%zu already set", slot);
    t->op_inputs[slot] = arg;
}

void wl_tensor_copy_buffer_from(wl_tensor_t* t, const void* data, size_t size) {
    wl__assert(size == (size_t) wl__tensor_data_size(t), "Buffer size mismatch: %zu != %lld", size, wl__tensor_data_size(t));
    memcpy(t->buf, data, size);
}

void wl_tensor_fill(wl_tensor_t* t, float x) {
    if (x == 0.0f) {
        memset(t->buf, 0, wl__tensor_data_size(t));
        return;
    }
    switch (t->dtype) {
        case WL_DTYPE_F32: {
            int64_t n = wl__tensor_num_elements(t);
            float* buf = (float*)t->buf;
            for (int64_t i=0; i < n; ++i) buf[i] = x;
        } break;
        default: wl__panic("Unsupported DType: %d", t->dtype);
    }
}

void wl_tensor_fill_random(wl_tensor_t* t, float min, float max) {
    switch (t->dtype) {
        case WL_DTYPE_F32: {
            int64_t n = wl__tensor_num_elements(t);
            float* buf = (float*)t->buf;
            wl__prng_generate_n(t->ctx, buf, n, min, max);
        } break;
        default: wl__panic("Unsupported DType: %d", t->dtype);
    }
}

size_t wl_tensor_get_memory_usage(const wl_tensor_t* t) {
    return sizeof(*t) + wl__tensor_data_size(t);
}

static void wl__print_tensor_recursive(FILE* f, const wl_tensor_t* t, int64_t (*idx)[WL_MAX_DIMS], int64_t curr_dim, int64_t total_dims, int indent) {
    wl_static_assert(WL_MAX_DIMS == 6);
    wl__assert2(curr_dim >= 0 && curr_dim < WL_MAX_DIMS && total_dims >= 1 && total_dims <= WL_MAX_DIMS);
    int64_t dim_size = t->shape[curr_dim];
    if (curr_dim == total_dims - 1) {
        fprintf(f, "%*s[", indent, "");
        for (int64_t i = 0; i < dim_size; ++i) {
            (*idx)[curr_dim] = i;
            int64_t idx_rev[WL_MAX_DIMS] = {0};
            for (int64_t j = 0; j < total_dims; ++j) idx_rev[j] = (*idx)[total_dims - j - 1];
            float val = wl_tensor_get_scalar_physical_index(t, idx_rev[0], idx_rev[1], idx_rev[2], idx_rev[3], idx_rev[4], idx_rev[5]);
            char fmt_buf[128];
            *wl__fmt_f64(WL__FMT_G14, (double)val, fmt_buf) = '\0';
            fprintf(f, "%s", fmt_buf);
            if (i < dim_size - 1)fprintf(f, " ");
        }
        fprintf(f, "]");
    } else {
        fprintf(f, "%*s[\n", indent, "");
        for (int64_t i = 0; i < dim_size; ++i) {
            (*idx)[curr_dim] = i;
            wl__print_tensor_recursive(f, t, idx, curr_dim + 1, total_dims, indent + 1);
            if (i < dim_size - 1) fprintf(f, ",\n");
            else fprintf(f, "\n");
        }
        fprintf(f, "%*s]", indent, "");
    }
}

void wl_tensor_print(const wl_tensor_t* t, bool with_header, bool with_data) {
    wl__assert(t->dtype == WL_DTYPE_F32, "Tensor must be F32");
    wl__assert2(with_header || with_data);
    wl__load_local_storage_group(t, x_d, shape);
    FILE* f = stdout;
    if (with_header) {
        double buf_size_cvt = 0.0;
        const char* buf_size_unit = NULL;
        wl__humanize_memory_size(wl_tensor_get_memory_usage(t), &buf_size_cvt, &buf_size_unit);
        char shape[WL__FMT_DIM_BUF_SIZE];
        char strides[WL__FMT_DIM_BUF_SIZE];
        wl__fmt_dims(&shape, &t->shape, t->rank);
        wl__fmt_dims(&strides, &t->strides, t->rank);
        fprintf(f, "Tensor '%s', DType: %s, Rank: %" PRIi64 ", Elements: %" PRIi64 ", Shape: %s, Strides: %s, Mem: %.03f %s\n",
            t->name,
            wl_dtype_info_of(t->dtype)->name,
            t->rank,
            wl__tensor_num_elements(t),
            shape,
            strides,
            buf_size_cvt,
            buf_size_unit
        );
    }
    if (with_data) {
        int64_t idx[WL_MAX_DIMS] = {0};
        wl__print_tensor_recursive(f, t, &idx, 0, t->rank, 0);
        fprintf(f, "\n");
    }
}

void wl_tensor_set_name(wl_tensor_t* t, const char* name) {
    strncpy(t->name, name, WL_MAX_TENSOR_NAME_LEN);
    t->name[WL_MAX_TENSOR_NAME_LEN-1] = '\0';
}

void wl_tensor_fmt_name(wl_tensor_t* t, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(t->name, sizeof(t->name), fmt, args);
    va_end(args);
}

const char* wl_tensor_get_name(const wl_tensor_t* t) {
    return t->name;
}

int64_t wl_tensor_rank(const wl_tensor_t* t) { return t->rank; }
const int64_t* wl_tensor_shape(const wl_tensor_t* t) { return t->shape; }
const int64_t* wl_tensor_strides(const wl_tensor_t* t) { return t->strides; }
wl_dtype_t wl_tensor_dtype(const wl_tensor_t* t) { return t->dtype; }
void* wl_tensor_data(const wl_tensor_t* t) { return t->buf; }

float* wl_tensor_data_as_f32(const wl_tensor_t* t) {
    wl__assert(t->dtype == WL_DTYPE_F32, "Tensor data type must be F32, not %s", wl_dtype_info_of(t->dtype)->name);
    return (float*)t->buf;
}

bool wl_tensor_is_scalar(const wl_tensor_t* t) {
    #pragma GCC unroll 6
    for (uint32_t i=0; i < WL_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool wl_tensor_is_vector(const wl_tensor_t* t) {
    #pragma GCC unroll 5
    for (uint32_t i=1; i < WL_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool wl_tensor_is_matrix(const wl_tensor_t* t) {
    #pragma GCC unroll 4
    for (uint32_t i=2; i < WL_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool wl_tensor_is_volume(const wl_tensor_t* t) {
    #pragma GCC unroll 3
    for (uint32_t i=3; i < WL_MAX_DIMS; ++i)
        if (t->shape[i] != 1)
            return false;
    return true;
}

bool wl_tensor_is_shape_eq(const wl_tensor_t* a, const wl_tensor_t* b) {
    return memcmp(a->shape, b->shape, sizeof(a->shape)) == 0;
}

bool wl_tensor_are_strides_eq(const wl_tensor_t* a, const wl_tensor_t* b) {
    return memcmp(a->strides, b->strides, sizeof(a->strides)) == 0;
}

bool wl_tensor_can_broadcast(const wl_tensor_t* a, const wl_tensor_t* b) {
    #pragma GCC unroll 6
    for (uint32_t i=0; i < WL_MAX_DIMS; ++i)
        if ((b->shape[i] % a->shape[i]) != 0)
            return false;
    return true;
}

bool wl_tensor_is_transposed(const wl_tensor_t* t) { return t->strides[0] > t->strides[1]; }

bool wl_tensor_is_permuted(const wl_tensor_t* t) {
    #pragma GCC unroll 5
    for (uint32_t i=0; i < WL_MAX_DIMS-1; ++i)
        if (t->strides[i] > t->strides[i+1])
            return true;
    return false;
}

bool wl_tensor_is_contiguous(const wl_tensor_t* t) {
    return *t->strides == wl_dtype_info_of(t->dtype)->size;
}

float wl_tensor_get_scalar_physical_index(const wl_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5) {
    wl_static_assert(WL_MAX_DIMS == 6);
    wl__load_local_storage_group(t, s, strides);
    const uint8_t* dst = (const uint8_t*)t->buf + d0*s0 + d1*s1 + d2*s2 + d3*s3 + d4*s4 + d5*s5;
    switch (t->dtype) {
        case WL_DTYPE_F32: return *(const float*)dst;
        default: wl__panic("Unsupported data type: %s", wl_dtype_info_of(t->dtype)->name);
    }
}

void wl_tensor_set_scalar_physical_index(wl_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, float x) {
    wl_static_assert(WL_MAX_DIMS == 6);
    wl__load_local_storage_group(t, s, strides);
    uint8_t* dst = (uint8_t*)t->buf + d0*s0 + d1*s1 + d2*s2 + d3*s3 + d4*s4 + d5*s5;
    switch (t->dtype) {
        case WL_DTYPE_F32: *(float*)dst = x; break;
        default: wl__panic("Unsupported data type: %s", wl_dtype_info_of(t->dtype)->name);
    }
}

float wl_tensor_get_scalar_virtual_index(const wl_tensor_t* t, int64_t v_idx) {
    if (!wl_tensor_is_contiguous(t)) {
        int64_t pidx[WL_MAX_DIMS];
        wl_tensor_virtual_to_physical_index(t, v_idx, &pidx);
        return wl_tensor_get_scalar_physical_index(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5]);
    }
    switch (t->dtype) {
        case WL_DTYPE_F32:
            return ((const float*)t->buf)[v_idx];
        default:
            wl__panic("Unsupported data type: %s", wl_dtype_info_of(t->dtype)->name);
    }
}

void wl_tensor_set_scalar_virtual_index(wl_tensor_t* t, int64_t v_idx, float x) {
    if (!wl_tensor_is_contiguous(t)) {
        int64_t pidx[WL_MAX_DIMS];
        wl_tensor_virtual_to_physical_index(t, v_idx, &pidx);
        wl_tensor_set_scalar_physical_index(t, pidx[0], pidx[1], pidx[2], pidx[3], pidx[4], pidx[5], x);
        return;
    }
    switch (t->dtype) {
        case WL_DTYPE_F32:
            ((float*)t->buf)[v_idx] = x;
            break;
        default:
            wl__panic("Unsupported data type: %s", wl_dtype_info_of(t->dtype)->name);
    }
}

bool wl_tensor_eq(const wl_tensor_t* a, const wl_tensor_t* b) {
    if (a->dtype != b->dtype) return false;
    if (a->rank != b->rank) return false;
    if (memcmp(a->shape, b->shape, sizeof(a->shape)) != 0) return false;
    if (a->num_elems != b->num_elems) return false;
    int64_t n = wl__tensor_num_elements(a);
    switch (a->dtype) {
        case WL_DTYPE_F32: {
            const float* buf_a = (const float*)a->buf;
            const float* buf_b = (const float*)b->buf;
            for (int64_t i = 0; i < n; ++i) {
                if (buf_a[i] != buf_b[i]) {
                    return false;
                }
            }
        } break;
        default: wl__panic("Unsupported data type: %s", wl_dtype_info_of(a->dtype)->name);
    }
    return true;
}

bool wl_tensor_is_close(const wl_tensor_t* a, const wl_tensor_t* b, float eps, double* percent_eq) {
    if (a->dtype != b->dtype) return false;
    if (a->rank != b->rank) return false;
    if (memcmp(a->shape, b->shape, sizeof(a->shape)) != 0) return false;
    if (a->num_elems != b->num_elems) return false;
    eps = eps < 0.0f ? FLT_EPSILON : eps;
    int64_t n = wl__tensor_num_elements(a);
    int64_t n_eq = 0;
    switch (a->dtype) {
        case WL_DTYPE_F32: {
            const float* buf_a = (const float*)a->buf;
            const float* buf_b = (const float*)b->buf;
            for (int64_t i = 0; i < n; ++i)  /* |x - y| <= ε     ∀ x, y ∈ A, B */
                if (fabsf(buf_a[i] - buf_b[i]) <= eps) ++n_eq;
        } break;
        default: wl__panic("Unsupported data type: %s", wl_dtype_info_of(a->dtype)->name);
    }
    if (percent_eq) *percent_eq = (double)n_eq / (double)n * 100.0;
    return n_eq == n;
}

void wl_tensor_img_draw_box(wl_tensor_t* t, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t wi, uint32_t rgb) {
    wl__assert(t->rank == 3, "Tensor must be 3D image tensor");
    wl__assert2(x2 > x1 && y2 > y1);
    float* buf = wl_tensor_data_as_f32(t);
    uint32_t w = (uint32_t)wl_tensor_image_width(t);
    uint32_t h = (uint32_t)wl_tensor_image_height(t);
    uint32_t c = (uint32_t)wl_tensor_image_channels(t);
    wl__assert2(w && h && c);
    float r = (float)(rgb&255) / 255.0f;
    float g = (float)((rgb>>8)&255) / 255.0f;
    float b = (float)((rgb>>16)&255) / 255.0f;
    wi = wl__max(1, wi);
    for (uint32_t i = 0; i < wi; ++i) {
        uint32_t xx1 = x1+i;
        uint32_t yy1 = y1+i;
        uint32_t xx2 = x2-i;
        uint32_t yy2 = y2-i;
        if (xx1 >= w) xx1 = w-1;
        if (xx2 >= w) xx2 = w-1;
        if (yy1 >= h) yy1 = h-1;
        if (yy2 >= h) yy2 = h-1;
        for (uint32_t j = xx1; j <= xx2; ++j) {
            buf[j + yy1*w + 0*w*h] = r;
            buf[j + yy2*w + 0*w*h] = r;
            buf[j + yy1*w + 1*w*h] = g;
            buf[j + yy2*w + 1*w*h] = g;
            buf[j + yy1*w + 2*w*h] = b;
            buf[j + yy2*w + 2*w*h] = b;
        }
        for (uint32_t j = yy1; j <= yy2; ++j) {
            buf[xx1 + j*w + 0*w*h] = r;
            buf[xx2 + j*w + 0*w*h] = r;
            buf[xx1 + j*w + 1*w*h] = g;
            buf[xx2 + j*w + 1*w*h] = g;
            buf[xx1 + j*w + 2*w*h] = b;
            buf[xx2 + j*w + 2*w*h] = b;
        }
    }
}

wl_ctx_t* wl_tensor_get_ctx(const wl_tensor_t* t) { return t->ctx; }
void* wl_tensor_get_user_data(const wl_tensor_t* t) { return t->ud; }
void wl_tensor_set_user_data(wl_tensor_t* t, void* ud) { t->ud = ud; }

/* CPU BLAS impl */
#define WL__GELU_COEFF 0.044715f

#if WL_INTRIN && defined(__aarch64__) && defined(__ARM_NEON)
    static float32x4_t wl__simd_expf(float32x4_t x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0  */
        float32x4_t r = vdupq_n_f32(0x1.8p23f);
        float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
        float32x4_t n = vsubq_f32(z, r);
        float32x4_t b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
        uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
        float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
        uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
        float32x4_t u = vmulq_f32(b, b);
        float32x4_t j = vfmaq_f32(
            vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
            vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
            vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u), u);
        if (!vpaddd_u64(vreinterpretq_u64_u32(c))) return vfmaq_f32(k, j, k);
        uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
        float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
        float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
        return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
               vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)));
    }

    static float32x4_t wl__simd_tanh(float32x4_t x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t neg_one = vdupq_n_f32(-1.0f);
        float32x4_t two = vdupq_n_f32(2.0f);
        float32x4_t neg_two = vdupq_n_f32(-2.0f);
        float32x4_t a = vmulq_f32(neg_two, x);
        float32x4_t b = wl__simd_expf(a);
        float32x4_t c = vaddq_f32(one, b);
        float32x4_t inv = vrecpeq_f32(c);
        inv = vmulq_f32(vrecpsq_f32(c, inv), inv); /* Newton–Raphson method */
        inv = vmulq_f32(vrecpsq_f32(c, inv), inv); /* Newton–Raphson method */
        return vaddq_f32(neg_one, vmulq_f32(two, inv));
    }
#elif WL_INTRIN && defined(__AVX512F__) && defined(__AVX512DQ__)

    static __m512 wl__simd_expf(const __m512 x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0 */
        __m512 r = _mm512_set1_ps(0x1.8p23f);
        __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
        __m512 n = _mm512_sub_ps(z, r);
        __m512 b = _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f), _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
        __mmask16 d = _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
        __m512 u = _mm512_mul_ps(b, b);
        __m512 j = _mm512_fmadd_ps(
            _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b, _mm512_set1_ps(0x1.573e2ep-5f)), u,
            _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b, _mm512_set1_ps(0x1.fffdb6p-2f))), u, _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F))
        );
        __m512 res = _mm512_scalef_ps(j, n);
        if (_mm512_kortestz(d, d)) return res;
        __m512 zero = _mm512_setzero_ps();
        __m512 alt = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
        return _mm512_mask_blend_ps(d, res, alt);
    }

    static __m512 wl__simd_tanh(__m512 x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 neg_one = _mm512_set1_ps(-1.0f);
        __m512 two = _mm512_set1_ps(2.0f);
        __m512 neg_two = _mm512_set1_ps(-2.0f);
        __m512 a = _mm512_mul_ps(neg_two, x);
        __m512 b = wl__simd_expf(a);
        __m512 c = _mm512_add_ps(one, b);
        __m512 inv = _mm512_rcp14_ps(c);
        inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv); /* Newton–Raphson method */
        inv = _mm512_mul_ps(_mm512_rcp14_ps(_mm512_mul_ps(c, inv)), inv); /* Newton–Raphson method */
        return _mm512_fmadd_ps(two, inv, neg_one);
    }

#elif WL_INTRIN && defined(__AVX2__) && defined(__FMA__)
    static __m256 wl__simd_expf(const __m256 x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0 */
        __m256 r = _mm256_set1_ps(0x1.8p23f);
        __m256 z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
        __m256 n = _mm256_sub_ps(z, r);
        __m256 b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),_mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
        __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
        __m256 k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
        __m256i c = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
        __m256 u = _mm256_mul_ps(b, b);
        __m256 j = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b,_mm256_set1_ps(0x1.573e2ep-5f)), u,_mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b,_mm256_set1_ps(0x1.fffdb6p-2f))),u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
        if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) return _mm256_fmadd_ps(j, k, k);
        __m256i g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),_mm256_set1_epi32(0x82000000u));
        __m256 s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
        __m256 s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
        __m256i d = _mm256_castps_si256(_mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
        return _mm256_or_ps(
            _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
            _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
            _mm256_and_ps(_mm256_castsi256_ps(c),
            _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
            _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k))))
        );
    }

    static __m256 wl__simd_tanh(__m256 x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 neg_one = _mm256_set1_ps(-1.0f);
        __m256 two = _mm256_set1_ps(2.0f);
        __m256 neg_two = _mm256_set1_ps(-2.0f);
        __m256 a = _mm256_mul_ps(neg_two, x);
        __m256 b = wl__simd_expf(a);
        __m256 c = _mm256_add_ps(one, b);
        __m256 inv = _mm256_rcp_ps(c);
        inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv); /* Newton–Raphson method */
        inv = _mm256_mul_ps(_mm256_rcp_ps(_mm256_mul_ps(c, inv)), inv); /* Newton–Raphson method */
        return _mm256_fmadd_ps(two, inv, neg_one);
    }
#elif WL_INTRIN && defined(__SSE2__)
    static __m128 wl__simd_expf(const __m128 x) { /* exp(x) : ℝ -> (0, ∞), x |-> e^x. Error = 1.45358 + 0.5 ulps. x > 88.38 -> INF, x < -103.97 -> 0 */
        __m128 r = _mm_set1_ps(0x1.8p23f);
        __m128 z = _mm_add_ps(_mm_mul_ps(x, _mm_set1_ps(0x1.715476p+0f)), r);
        __m128 n = _mm_sub_ps(z, r);
        __m128 b = _mm_sub_ps(_mm_sub_ps(x, _mm_mul_ps(n, _mm_set1_ps(0x1.62e4p-1f))), _mm_mul_ps(n, _mm_set1_ps(0x1.7f7d1cp-20f)));
        __m128i e = _mm_slli_epi32(_mm_castps_si128(z), 23);
        __m128 k = _mm_castsi128_ps(_mm_add_epi32(e, _mm_castps_si128(_mm_set1_ps(1))));
        __m128i c = _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(126)));
        __m128 u = _mm_mul_ps(b, b);
        __m128 j = _mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_add_ps(_mm_mul_ps(_mm_set1_ps(0x1.0e4020p-7f), b), _mm_set1_ps(0x1.573e2ep-5f)),u),
        _mm_add_ps(_mm_mul_ps(_mm_set1_ps(0x1.555e66p-3f), b), _mm_set1_ps(0x1.fffdb6p-2f))), u),
        _mm_mul_ps(_mm_set1_ps(0x1.ffffecp-1f), b));
        if (!_mm_movemask_epi8(c)) return _mm_add_ps(_mm_mul_ps(j, k), k);
        __m128i g = _mm_and_si128(_mm_castps_si128(_mm_cmple_ps(n, _mm_setzero_ps())),_mm_set1_epi32(0x82000000u));
        __m128 s1 = _mm_castsi128_ps(_mm_add_epi32(g, _mm_set1_epi32(0x7f000000u)));
        __m128 s2 = _mm_castsi128_ps(_mm_sub_epi32(e, g));
        __m128i d = _mm_castps_si128(_mm_cmpgt_ps(_mm_andnot_ps(_mm_set1_ps(-0.f), n), _mm_set1_ps(192)));
        return _mm_or_ps(
            _mm_and_ps(_mm_castsi128_ps(d), _mm_mul_ps(s1, s1)),
            _mm_andnot_ps(_mm_castsi128_ps(d),
            _mm_or_ps(_mm_and_ps(_mm_castsi128_ps(c), _mm_mul_ps(_mm_add_ps(_mm_mul_ps(s2, j), s2), s1)),
            _mm_andnot_ps(_mm_castsi128_ps(c), _mm_add_ps(_mm_mul_ps(k, j), k))))
        );
    }

    static __m128 wl__simd_tanh(__m128 x) { /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
        __m128 one = _mm_set1_ps(1.0f);
        __m128 neg_one = _mm_set1_ps(-1.0f);
        __m128 two = _mm_set1_ps(2.0f);
        __m128 neg_two = _mm_set1_ps(-2.0f);
        __m128 a = _mm_mul_ps(neg_two, x);
        __m128 b = wl__simd_expf(a);
        __m128 c = _mm_add_ps(one, b);
        __m128 inv = _mm_rcp_ps(c);
        inv = _mm_mul_ps(_mm_rcp_ps(_mm_mul_ps(c, inv)), inv); /* Newton–Raphson method */
        inv = _mm_mul_ps(_mm_rcp_ps(_mm_mul_ps(c, inv)), inv); /* Newton–Raphson method */
        return _mm_add_ps(neg_one, _mm_mul_ps(two, inv));
    }
#endif

static void WL__HOTPROC wl__vadd_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] + y[i];
    }
}

static void WL__HOTPROC wl__vsub_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] - y[i];
    }
}

static void WL__HOTPROC wl__vmul_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] * y[i];
    }
}

static void WL__HOTPROC wl__vdiv_f32(
    const int64_t n,
    float* const o,
    const float* const x,
    const float* const y
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] / y[i];
    }
}

static float WL__UNUSED WL__HOTPROC wl__vdot_f32(
    const int64_t n,
    const float* const x,
    const float* const y
) {
    #if WL_INTRIN && defined(__ARM_NEON) && defined(__aarch64__)
        const int64_t k = n & -16;
        float32x4_t acc[4] = {vdupq_n_f32(0)};
        float32x4_t vx[4];
        float32x4_t vy[4];
        for (int64_t i=0; i < k; i += 16) { /* Process STEP elements at a time */
            vx[0] = vld1q_f32(x+i+(0<<2));
            vy[0] = vld1q_f32(y+i+(0<<2));
            acc[0] = vfmaq_f32(acc[0], vx[0], vy[0]);
            vx[1] = vld1q_f32(x+i+(1<<2));
            vy[1] = vld1q_f32(y+i+(1<<2));
            acc[1] = vfmaq_f32(acc[1], vx[1], vy[1]);
            vx[2] = vld1q_f32(x+i+(2<<2));
            vy[2] = vld1q_f32(y+i+(2<<2));
            acc[2] = vfmaq_f32(acc[2], vx[2], vy[2]);
            vx[3] = vld1q_f32(x+i+(3<<2));
            vy[3] = vld1q_f32(y+i+(3<<2));
            acc[3] = vfmaq_f32(acc[3], vx[3], vy[3]);
        }
        acc[1] = vaddq_f32(acc[1], acc[3]); /* Fold acc[1] += acc[3] */
        *acc = vaddq_f32(*acc, acc[2]);     /* Fold acc[0] += acc[2] */
        *acc = vaddq_f32(*acc, acc[1]);     /* Fold acc[0] += acc[1] */
        float sum = vaddvq_f32(*acc);       /* Reduce to scalar with horizontal sum. */
        for (int64_t i=k; i < n; ++i) {     /* Process leftovers scalar-wise */
            sum += x[i]*y[i];
        }
        return sum;
    #elif WL_INTRIN && defined(__AVX512F__) && defined(__FMA__)
        const int64_t k = n & -64;
        __m512 acc[4] = {_mm512_setzero_ps()};
        __m512 vx[4];
        __m512 vy[4];
        for (int64_t i=0; i < k; i += 64) {
            vx[0] = _mm512_loadu_ps(x+i+(0<<4));
            vy[0] = _mm512_loadu_ps(y+i+(0<<4));
            acc[0] = _mm512_fmadd_ps(vx[0], vy[0], acc[0]);
            vx[1] = _mm512_loadu_ps(x+i+(1<<4));
            vy[1] = _mm512_loadu_ps(y+i+(1<<4));
            acc[1] = _mm512_fmadd_ps(vx[1], vy[1], acc[1]);
            vx[2] = _mm512_loadu_ps(x+i+(2<<4));
            vy[2] = _mm512_loadu_ps(y+i+(2<<4));
            acc[2] = _mm512_fmadd_ps(vx[2], vy[2], acc[2]);
            vx[3] = _mm512_loadu_ps(x+i+(3<<4));
            vy[3] = _mm512_loadu_ps(y+i+(3<<4));
            acc[3] = _mm512_fmadd_ps(vx[3], vy[3], acc[3]);
        }
        acc[1] = _mm512_add_ps(acc[1], acc[3]);
        *acc = _mm512_add_ps(*acc, acc[2]);
        *acc = _mm512_add_ps(*acc, acc[1]);
        float sum = _mm512_reduce_add_ps(*acc);
        for (int64_t i=k; i < n; ++i) sum += x[i]*y[i]; /* Process leftovers scalar-wise */
        return sum;
    #elif WL_INTRIN && defined(__AVX__) && defined(__FMA__)
        const int64_t k = n & -32;
        __m256 acc[4] = {_mm256_setzero_ps()};
        __m256 vx[4];
        __m256 vy[4];
        for (int64_t i=0; i < k; i += 32) {
            vx[0] = _mm256_loadu_ps(x+i+(0<<3));
            vy[0] = _mm256_loadu_ps(y+i+(0<<3));
            acc[0] = _mm256_fmadd_ps(vx[0], vy[0], acc[0]);
            vx[1] = _mm256_loadu_ps(x+i+(1<<3));
            vy[1] = _mm256_loadu_ps(y+i+(1<<3));
            acc[1] = _mm256_fmadd_ps(vx[1], vy[1], acc[1]);
            vx[2] = _mm256_loadu_ps(x+i+(2<<3));
            vy[2] = _mm256_loadu_ps(y+i+(2<<3));
            acc[2] = _mm256_fmadd_ps(vx[2], vy[2], acc[2]);
            vx[3] = _mm256_loadu_ps(x+i+(3<<3));
            vy[3] = _mm256_loadu_ps(y+i+(3<<3));
            acc[3] = _mm256_fmadd_ps(vx[3], vy[3], acc[3]);
        }
        acc[1] = _mm256_add_ps(acc[1], acc[3]);
        *acc = _mm256_add_ps(*acc, acc[2]);
        *acc = _mm256_add_ps(*acc, acc[1]);
        __m128 v0 = _mm_add_ps(_mm256_castps256_ps128(*acc), _mm256_extractf128_ps(*acc, 1));
        v0 = _mm_hadd_ps(v0, v0);
        v0 = _mm_hadd_ps(v0, v0);
        float sum = _mm_cvtss_f32(v0);
        for (int64_t i=k; i < n; ++i) sum += x[i]*y[i]; /* Process leftovers scalar-wise */
        return sum;
    #elif WL_INTRIN && defined(__SSE2__)
        const int64_t k = n & -16;
        __m128 acc[4] = {_mm_setzero_ps()};
        __m128 vx[4];
        __m128 vy[4];
        for (int64_t i=0; i < k; i += 16) {
            vx[0] = _mm_loadu_ps(x+i+(0<<2));
            vy[0] = _mm_loadu_ps(y+i+(0<<2));
            acc[0] = _mm_add_ps(acc[0], _mm_mul_ps(vx[0], vy[0]));
            vx[1] = _mm_loadu_ps(x+i+(1<<2));
            vy[1] = _mm_loadu_ps(y+i+(1<<2));
            acc[1] = _mm_add_ps(acc[1], _mm_mul_ps(vx[1], vy[1]));
            vx[2] = _mm_loadu_ps(x+i+(2<<2));
            vy[2] = _mm_loadu_ps(y+i+(2<<2));
            acc[2] = _mm_add_ps(acc[2], _mm_mul_ps(vx[2], vy[2]));
            vx[3] = _mm_loadu_ps(x+i+(3<<2));
            vy[3] = _mm_loadu_ps(y+i+(3<<2));
            acc[3] = _mm_add_ps(acc[3], _mm_mul_ps(vx[3], vy[3]));
        }
        #ifdef __SSE3__
            acc[1] = _mm_add_ps(acc[1], acc[3]);
            *acc = _mm_add_ps(*acc, acc[2]);
            *acc = _mm_add_ps(*acc, acc[1]);
            *acc = _mm_hadd_ps(*acc, *acc);
            *acc = _mm_hadd_ps(*acc, *acc);
            float sum = _mm_cvtss_f32(*acc);
        #else
            __m128 shuf = _mm_shuffle_ps(*acc, *acc, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(*acc, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            float sum = _mm_cvtss_f32(sums);
        #endif
        for (int64_t i=k; i < n; ++i) sum += x[i]*y[i]; /* Process leftovers scalar-wise */
        return sum;
    #else
        double r = 0.0;
        for (int64_t i=0; i < n; ++i) r += (double)x[i] * (double)y[i];
        return (float)r;
    #endif
}

static double WL__HOTPROC wl__vsum_f64_f32( /* Σx. */
    const int64_t n,
    const float* const x
) {
    double sum = 0.0;
    for (int64_t i=0; i < n; ++i)
        sum += (double)x[i];
    return sum;
}

static float WL__HOTPROC wl__vsum_f32( /* Σx. */
    const int64_t n,
    const float* const x
) {
    return (float)wl__vsum_f64_f32(n, x);
}

static void WL__HOTPROC wl__vabs_f32( /* o = |x| */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i)
        o[i] = fabsf(x[i]);
}

static void WL__HOTPROC wl__vneg_f32( /* o = -x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i)
        o[i] = -x[i];
}

static void WL__HOTPROC wl__vlog_f32( /* o = log x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i)
        o[i] = logf(x[i]);
}

static void WL__HOTPROC wl__vsqr_f32( /* o = x² */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i)
        o[i] = x[i]*x[i];
}

static void WL__HOTPROC wl__vsqrt_f32( /* o = √x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i)
        o[i] = sqrtf(x[i]);
}

static void WL__HOTPROC wl__vsin_f32( /* o = sin x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i)
        o[i] = sinf(x[i]);
}

static void WL__HOTPROC wl__vcos_f32( /* o = cos x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i)
        o[i] = cosf(x[i]);
}

static void WL__HOTPROC wl__vstep_f32( /* Heaviside step function. */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i)
        o[i] = x[i] >= 0.0f ? 1.0f : 0.0f;
}

static void WL__HOTPROC wl__vsoftmax_f32( /* softmax : ℝ -> (0, ∞), x |-> e^x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
    #if WL_INTRIN && defined(__ARM_NEON) && defined(__aarch64__)
        for (; i+3 < n; i += 4) {
            vst1q_f32(o+i, wl__simd_expf(vld1q_f32(x+i)));
        }
    #elif WL_INTRIN && defined(__AVX512F__) && defined(__AVX512DQ__)
        for (; i+15 < n; i += 16) {
            _mm512_storeu_ps(o+i, wl__simd_expf(_mm512_loadu_ps(x+i)));
        }
    #elif WL_INTRIN && defined(__AVX2__) && defined(__FMA__)
        for (; i+7 < n; i += 8) {
            _mm256_storeu_ps(o+i, wl__simd_expf(_mm256_loadu_ps(x+i)));
        }
    #elif WL_INTRIN && defined(__SSE2__)
        for (; i+3 < n; i += 4) {
            _mm_storeu_ps(o+i, wl__simd_expf(_mm_loadu_ps(x+i)));
        }
    #endif
    for (; i < n; ++i) o[i] = expf(x[i]); /* Process leftovers scalar-wise */
}

static void WL__HOTPROC wl__vsoftmax_dv_f32( /* softmax' = softmax : ℝ -> (0, ∞), x |-> e^x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    return wl__vsoftmax_f32(n, o, x);
}

static void WL__HOTPROC wl__vsigmoid_f32( /* σ : ℝ -> (0, 1), x |-> 1/(1 + e^(-x)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
    #if WL_INTRIN && defined(__ARM_NEON) && defined(__aarch64__)
        const float32x4_t one = vdupq_n_f32(1.0f);
        const float32x4_t zero = vdupq_n_f32(0.0f);
        for (; i+3 < n; i += 4) {
            float32x4_t xx = vld1q_f32(x+i);
            float32x4_t neg_x = vsubq_f32(zero, xx);
            float32x4_t exp_neg_x = wl__simd_expf(neg_x);
            float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
            vst1q_f32(o+i, vdivq_f32(one, one_plus_exp_neg_x));
        }
    #elif WL_INTRIN && defined(__AVX512F__) && defined(__AVX512DQ__)
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 zero = _mm512_setzero_ps();
        for (; i+15 < n; i += 16) {
            __m512 xx = _mm512_loadu_ps(x+i);
            __m512 neg_x = _mm512_sub_ps(zero, xx);
            __m512 exp_neg_x = wl__simd_expf(neg_x);
            __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
            _mm512_storeu_ps(o+i, _mm512_div_ps(one, one_plus_exp_neg_x));
        }
    #elif WL_INTRIN && defined(__AVX2__) && defined(__FMA__)
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 zero = _mm256_setzero_ps();
        for (; i+7 < n; i += 8) {
            __m256 xx = _mm256_loadu_ps(x+i);
            __m256 neg_x = _mm256_sub_ps(zero, xx);
            __m256 exp_neg_x = wl__simd_expf(neg_x);
            __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
            _mm256_storeu_ps(o+i, _mm256_div_ps(one, one_plus_exp_neg_x));
        }
    #elif WL_INTRIN && defined(__SSE2__)
        __m128 one = _mm_set1_ps(1.0f);
        __m128 zero = _mm_setzero_ps();
        for (; i+3 < n; i += 4) {
            __m128 xx = _mm_loadu_ps(x+i);
            __m128 neg_x = _mm_sub_ps(zero, xx);
            __m128 exp_neg_x = wl__simd_expf(neg_x);
            __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
            _mm_storeu_ps(o+i, _mm_div_ps(one, one_plus_exp_neg_x));
        }
    #endif
    for (; i < n; ++i) o[i] = 1.0f / (1.0f + expf(-x[i])); /* Process leftovers scalar-wise */
}

static void WL__HOTPROC wl__vsigmoid_dv_f32( /* σ' : ℝ -> (0, 1), x |-> x * (1-x) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] * (1.0f - x[i]);
    }
}

static void WL__HOTPROC wl__vhard_sigmoid_f32( /* σ^ : ℝ -> (0, 1), x |-> min(1, max(0, (x + 3)/6)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = fminf(1.0f, fmaxf(0.0f, (x[i] + 3.0f) / 6.0f));
    }
}

static void WL__HOTPROC wl__vsilu_f32( /* silu : ℝ -> ℝ, x |-> x/(1 + e^(-x)) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
    #if WL_INTRIN && defined(__ARM_NEON) && defined(__aarch64__)
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t zero = vdupq_n_f32(0.0f);
        for (; i+3 < n; i += 4) {
            float32x4_t xx = vld1q_f32(x+i);
            float32x4_t neg_x = vsubq_f32(zero, xx);
            float32x4_t exp_neg_x = wl__simd_expf(neg_x);
            float32x4_t one_plus_exp_neg_x = vaddq_f32(one, exp_neg_x);
            vst1q_f32(o+i, vdivq_f32(xx, one_plus_exp_neg_x));
        }
    #elif WL_INTRIN && defined(__AVX512F__) && defined(__AVX512DQ__)
        __m512 one = _mm512_set1_ps(1);
        __m512 zero = _mm512_setzero_ps();
        for (; i+15 < n; i += 16) {
            __m512 xx = _mm512_loadu_ps(x+i);
            __m512 neg_x = _mm512_sub_ps(zero, xx);
            __m512 exp_neg_x = wl__simd_expf(neg_x);
            __m512 one_plus_exp_neg_x = _mm512_add_ps(one, exp_neg_x);
            _mm512_storeu_ps(o+i, _mm512_div_ps(xx, one_plus_exp_neg_x));
        }
    #elif WL_INTRIN && defined(__AVX2__) && defined(__FMA__)
        __m256 one = _mm256_set1_ps(1);
        __m256 zero = _mm256_setzero_ps();
        for (; i+7 < n; i += 8) {
            const __m256 xx = _mm256_loadu_ps(x+i);
            __m256 neg_x = _mm256_sub_ps(zero, xx);
            __m256 exp_neg_x = wl__simd_expf(neg_x);
            __m256 one_plus_exp_neg_x = _mm256_add_ps(one, exp_neg_x);
            _mm256_storeu_ps(o+i, _mm256_div_ps(xx, one_plus_exp_neg_x));
        }
    #elif WL_INTRIN && defined(__SSE2__)
        __m128 one = _mm_set1_ps(1);
        __m128 zero = _mm_setzero_ps();
        for (; i+3 < n; i += 4) {
            __m128 xx = _mm_loadu_ps(x+i);
            __m128 neg_x = _mm_sub_ps(zero, xx);
            __m128 exp_neg_x = wl__simd_expf(neg_x);
            __m128 one_plus_exp_neg_x = _mm_add_ps(one, exp_neg_x);
            _mm_storeu_ps(o+i, _mm_div_ps(xx, one_plus_exp_neg_x));
        }
    #endif
    for (; i < n; ++i) {
        o[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static void WL__HOTPROC wl__vsilu_dv_f32( /* silu' : ℝ -> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        wl__panic("NYI!");
    }
}

static void WL__HOTPROC wl__vtanh_f32( /* tanh : ℝ -> (-1, 1), x |-> tanh x */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
    #if WL_INTRIN && defined(__ARM_NEON) && defined(__aarch64__)
        for (; i+3 < n; i += 4) {
            vst1q_f32(o+i, wl__simd_tanh(vld1q_f32(x+i)));
        }
    #elif WL_INTRIN && defined(__AVX512F__) && defined(__AVX512DQ__)
        for (; i+15 < n; i += 16) {
            _mm512_storeu_ps(o+i, wl__simd_tanh(_mm512_loadu_ps(x+i)));
        }
    #elif WL_INTRIN && defined(__AVX2__) && defined(__FMA__)
        for (; i+7 < n; i += 8) {
            _mm256_storeu_ps(o+i, wl__simd_tanh(_mm256_loadu_ps(x+i)));
        }
    #elif WL_INTRIN && defined(__SSE2__)
        for (; i+3 < n; i += 4) {
            _mm_storeu_ps(o+i, wl__simd_tanh(_mm_loadu_ps(x+i)));
        }
    #endif
    for (; i < n; ++i) {
        o[i] = tanhf(x[i]);
    }
}

static void WL__HOTPROC wl__vtanh_dv_f32( /* tanh' : ℝ -> (-1, 1), x |-> 1 / ((cosh x)^2) */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        const float cx = coshf(x[i]);
        o[i] = 1.0f / (cx*cx);
    }
}

static void WL__HOTPROC wl__vrelu_f32( /* relu : ℝ -> ℝ^+, x |-> max {x, 0} */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = wl__max(x[i], 0.0f);
    }
}

static void WL__HOTPROC wl__vrelu_dv_f32( /* relu' : ℝ -> ℝ^+, x |-> { 0 if x < 0, UB if x = 0, else 1 */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        o[i] = x[i] <= 0.0f ? 0.0f : 1.0f; /* relu' is mathematically undefined for x = 0, but we return 0 in this case. */
    }
}

static void WL__HOTPROC wl__vgelu_f32( /* gelu : ℝ -> ℝ, x |-> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    int64_t i=0;
    #if WL_INTRIN && defined(__ARM_NEON) && defined(__aarch64__)
        float32x4_t half = vdupq_n_f32(0.5f);
        float32x4_t one = vdupq_n_f32(1.0f);
        float32x4_t coeff1 = vdupq_n_f32(0.79788456080286535587989211986876f);
        float32x4_t coeff2 = vdupq_n_f32(WL__GELU_COEFF);
        for (; i+3 < n; i += 4) {
            float32x4_t xx = vld1q_f32(x+i);
            float32x4_t a = vaddq_f32(one, vmulq_f32(coeff2, vmulq_f32(xx, xx)));
            float32x4_t b = vaddq_f32(one, wl__simd_tanh(vmulq_f32(coeff1, vmulq_f32(xx, a))));
            float32x4_t c = vmulq_f32(half, vmulq_f32(xx, b));
            vst1q_f32(o+i, c);
        }
    #elif WL_INTRIN && defined(__AVX512F__) && defined(__AVX512DQ__)
        __m512 half = _mm512_set1_ps(0.5f);
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 coeff1 = _mm512_set1_ps(0.79788456080286535587989211986876f);
        __m512 coeff2 = _mm512_set1_ps(WL__GELU_COEFF);
        for (; i+15 < n; i += 16) {
            __m512 xx = _mm512_loadu_ps(x+i);
            __m512 a = _mm512_fmadd_ps(coeff2, _mm512_mul_ps(xx, xx), one);
            __m512 b = _mm512_add_ps(one, wl__simd_tanh(_mm512_mul_ps(coeff1, _mm512_mul_ps(xx, a))));
            __m512 c = _mm512_mul_ps(half, _mm512_mul_ps(xx, b));
            _mm512_storeu_ps(o+i, c);
        }
    #elif WL_INTRIN && defined(__AVX2__) && defined(__FMA__)
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 coeff1 = _mm256_set1_ps(0.79788456080286535587989211986876f);
        __m256 coeff2 = _mm256_set1_ps(WL__GELU_COEFF);
        for (; i+7 < n; i += 8) {
            __m256 xx = _mm256_loadu_ps(x+i);
            __m256 a = _mm256_fmadd_ps(coeff2, _mm256_mul_ps(xx, xx), one);
            __m256 b = _mm256_add_ps(one, wl__simd_tanh(_mm256_mul_ps(coeff1, _mm256_mul_ps(xx, a))));
            __m256 c = _mm256_mul_ps(half, _mm256_mul_ps(xx, b));
            _mm256_storeu_ps(o+i, c);
        }
    #elif WL_INTRIN && defined(__SSE2__)
        __m128 half = _mm_set1_ps(0.5f);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 coeff1 = _mm_set1_ps(0.79788456080286535587989211986876f);
        __m128 coeff2 = _mm_set1_ps(WL__GELU_COEFF);
        for (; i+3 < n; i += 4) {
            __m128 xx = _mm_loadu_ps(x+i);
            __m128 a = _mm_add_ps(one, _mm_mul_ps(coeff2, _mm_mul_ps(xx, xx)));
            __m128 b = _mm_add_ps(one, wl__simd_tanh(_mm_mul_ps(coeff1, _mm_mul_ps(xx, a))));
            __m128 c = _mm_mul_ps(half, _mm_mul_ps(xx, b));
            _mm_storeu_ps(o+i, c);
        }
    #endif
    for (; i < n; ++i) {
        o[i] = 0.5f*x[i]*(1.0f + tanhf(0.79788456080286535587989211986876f*x[i]*(1.0f + WL__GELU_COEFF*x[i]*x[i])));
    }
}

static void WL__HOTPROC wl__vgelu_dv_f32( /* gelu' : ℝ -> ℝ, x |-> TODO */
    const int64_t n,
    float* const o,
    const float* const x
) {
    for (int64_t i=0; i < n; ++i) {
        wl__panic("NYI"); /* TODO */
    }
}

static void wl__blas_nop(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs
) {
    (void)bci;
    (void)r;
    (void)inputs;
}

static void wl__blas_clone(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs
) {
    const wl_tensor_t* const x = inputs[0];
    wl__assert2(wl_tensor_is_shape_eq(x, r));
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    memcpy(b_r, b_x, wl__tensor_data_size(r));
}

static void WL__HOTPROC wl__blas_mean_f32( /* Σx/n Arithmetic mean */
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_d, shape);
    wl__load_local_storage_group(x, x_s, strides);
    double sum = 0.0;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const float* const p_x = (const float*)(b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5);
                        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
                        sum += wl__vsum_f64_f32(
                            x_d0,
                            p_x
                        );
                    }
                }
            }
        }
    }
    sum /= (double)x->num_elems;
    *(float*)b_r = (float)sum;
}

static void WL__HOTPROC wl__blas_sum_f32( /* Σx/n Arithmetic mean */
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_d, shape);
    wl__load_local_storage_group(x, x_s, strides);
    double sum = 0.0;
    for (int64_t i5=0; i5 < x_d5; ++i5) {
        for (int64_t i4=0; i4 < x_d4; ++i4) {
            for (int64_t i3=0; i3 < x_d3; ++i3) {
                for (int64_t i2=0; i2 < x_d2; ++i2) {
                    for (int64_t i1=0; i1 < x_d1; ++i1) {
                        const float* const p_x = (const float*)(b_x + i1*x_s1 + i2*x_s2 + i3*x_s3 + i4*x_s4 + i5*x_s5);
                        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
                        sum += wl__vsum_f64_f32(
                            x_d0,
                            p_x
                        );
                    }
                }
            }
        }
    }
    *(float*)b_r = (float)sum;
}

static void WL__HOTPROC wl__blas_abs_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vabs_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_neg_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vneg_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_log_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vlog_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_sqr_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vsqr_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_sqrt_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vsqrt_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_sin_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vsin_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_cos_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vcos_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_step_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vstep_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_softmax_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vsoftmax_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_softmax_dv_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vsoftmax_dv_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_sigmoid_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vsigmoid_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_sigmoid_dv_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vsigmoid_dv_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_hard_sigmoid_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vhard_sigmoid_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_silu_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vsilu_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_silu_dv_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vsilu_dv_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_tanh_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vtanh_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_tanh_dv_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vtanh_dv_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_relu_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vrelu_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_relu_dv_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vrelu_dv_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_gelu_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vgelu_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_gelu_dv_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    (void)bci;
    const wl_tensor_t* const x = inputs[0];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    const int64_t cc = wl__tensor_num_cols(x);
    for (int64_t ri=0; ri < rc; ++ri) {
        float* const p_r = (float*)(b_r + ri*r_s1);
        const float* const p_x = (const float*)(b_x + ri*x_s1);
        wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
        wl__vgelu_dv_f32(cc, p_r, p_x);
    }
}

static void WL__HOTPROC wl__blas_add_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const wl_tensor_t* const x = inputs[0];
    const wl_tensor_t* const y = inputs[1];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    const uint8_t* const b_y = (const uint8_t*)y->buf;
    wl__load_local_storage_group(r, r_d, shape);
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_d, shape);
    wl__load_local_storage_group(x, x_s, strides);
    wl__load_local_storage_group(y, y_d, shape);
    wl__load_local_storage_group(y, y_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    int64_t x_i1 = 0;
    int64_t x_i2 = 0;
    int64_t x_i3 = 0;
    int64_t x_i4 = 0;
    int64_t x_i5 = 0;
    if (y_s0 == sizeof(float)) { /* Fast path for contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = (float*)(b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5);
            const float* const p_x = (const float*)(b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5);
            const float* const p_y = (const float*)(b_y + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5);
            wl__bnd_chk(p_y, b_y, wl__tensor_data_size(y));
            const int64_t pa = x_d0 / y_d0;
            for (int64_t i=0; i < pa; ++i) {  /* For each element in row */
                float* const pp_r = p_r + i*y_d0; /* Compute result ptr. */
                const float* const pp_x = p_x + i*y_d0; /* Compute x ptr. */
                wl__bnd_chk(pp_r, b_r, wl__tensor_data_size(r));
                wl__bnd_chk(pp_x, b_x, wl__tensor_data_size(x));
                wl__vadd_f32(y_d0, pp_r, pp_x, p_y);  /* Apply micro kernel vector op */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    } else { /* Slow path for non-contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = (float*)(b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5);
            const float* const p_x = (const float*)(b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5);
            for (int64_t i=0; i < r_d0; ++i) {  /* For each element in row */
                const float* const p_y = (const float*)(b_y + i%y_d0*y_s0 + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5); /* Compute result ptr. */
                wl__bnd_chk(p_r+i, b_r, wl__tensor_data_size(r));
                wl__bnd_chk(p_x+i, b_x, wl__tensor_data_size(x));
                wl__bnd_chk(p_y, b_y, wl__tensor_data_size(y));
                p_r[i] = p_x[i] + *p_y; /* Apply scalar op. */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    }
}

static void WL__HOTPROC wl__blas_sub_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const wl_tensor_t* const x = inputs[0];
    const wl_tensor_t* const y = inputs[1];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    const uint8_t* const b_y = (const uint8_t*)y->buf;
    wl__load_local_storage_group(r, r_d, shape);
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_d, shape);
    wl__load_local_storage_group(x, x_s, strides);
    wl__load_local_storage_group(y, y_d, shape);
    wl__load_local_storage_group(y, y_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    int64_t x_i1 = 0;
    int64_t x_i2 = 0;
    int64_t x_i3 = 0;
    int64_t x_i4 = 0;
    int64_t x_i5 = 0;
    if (y_s0 == sizeof(float)) { /* Fast path for contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = (float*)(b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5);
            const float* const p_x = (const float*)(b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5);
            const float* const p_y = (const float*)(b_y + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5);
            wl__bnd_chk(p_y, b_y, wl__tensor_data_size(y));
            const int64_t pa = x_d0 / y_d0;
            for (int64_t i=0; i < pa; ++i) {  /* For each element in row */
                float* const pp_r = p_r + i*y_d0; /* Compute result ptr. */
                const float* const pp_x = p_x + i*y_d0; /* Compute x ptr. */
                wl__bnd_chk(pp_r, b_r, wl__tensor_data_size(r));
                wl__bnd_chk(pp_x, b_x, wl__tensor_data_size(x));
                wl__vsub_f32(y_d0, pp_r, pp_x, p_y);  /* Apply micro kernel vector op */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    } else { /* Slow path for non-contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = (float*)(b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5);
            const float* const p_x = (const float*)(b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5);
            for (int64_t i=0; i < r_d0; ++i) {  /* For each element in row */
                const float* const p_y = (const float*)(b_y + i%y_d0*y_s0 + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5); /* Compute result ptr. */
                wl__bnd_chk(p_r+i, b_r, wl__tensor_data_size(r));
                wl__bnd_chk(p_x+i, b_x, wl__tensor_data_size(x));
                wl__bnd_chk(p_y, b_y, wl__tensor_data_size(y));
                p_r[i] = p_x[i] - *p_y; /* Apply scalar op. */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    }
}

static void WL__HOTPROC wl__blas_mul_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const wl_tensor_t* const x = inputs[0];
    const wl_tensor_t* const y = inputs[1];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    const uint8_t* const b_y = (const uint8_t*)y->buf;
    wl__load_local_storage_group(r, r_d, shape);
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_d, shape);
    wl__load_local_storage_group(x, x_s, strides);
    wl__load_local_storage_group(y, y_d, shape);
    wl__load_local_storage_group(y, y_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    int64_t x_i1 = 0;
    int64_t x_i2 = 0;
    int64_t x_i3 = 0;
    int64_t x_i4 = 0;
    int64_t x_i5 = 0;
    if (y_s0 == sizeof(float)) { /* Fast path for contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = (float*)(b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5);
            const float* const p_x = (const float*)(b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5);
            const float* const p_y = (const float*)(b_y + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5);
            wl__bnd_chk(p_y, b_y, wl__tensor_data_size(y));
            const int64_t pa = x_d0 / y_d0;
            for (int64_t i=0; i < pa; ++i) {  /* For each element in row */
                float* const pp_r = p_r + i*y_d0; /* Compute result ptr. */
                const float* const pp_x = p_x + i*y_d0; /* Compute x ptr. */
                wl__bnd_chk(pp_r, b_r, wl__tensor_data_size(r));
                wl__bnd_chk(pp_x, b_x, wl__tensor_data_size(x));
                wl__vmul_f32(y_d0, pp_r, pp_x, p_y);  /* Apply micro kernel vector op */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    } else { /* Slow path for non-contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = (float*)(b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5);
            const float* const p_x = (const float*)(b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5);
            for (int64_t i=0; i < r_d0; ++i) {  /* For each element in row */
                const float* const p_y = (const float*)(b_y + i%y_d0*y_s0 + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5); /* Compute result ptr. */
                wl__bnd_chk(p_r+i, b_r, wl__tensor_data_size(r));
                wl__bnd_chk(p_x+i, b_x, wl__tensor_data_size(x));
                wl__bnd_chk(p_y, b_y, wl__tensor_data_size(y));
                p_r[i] = p_x[i] * *p_y; /* Apply scalar op. */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    }
}

static void WL__HOTPROC wl__blas_div_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const wl_tensor_t* const x = inputs[0];
    const wl_tensor_t* const y = inputs[1];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    const uint8_t* const b_y = (const uint8_t*)y->buf;
    wl__load_local_storage_group(r, r_d, shape);
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_d, shape);
    wl__load_local_storage_group(x, x_s, strides);
    wl__load_local_storage_group(y, y_d, shape);
    wl__load_local_storage_group(y, y_s, strides);
    const int64_t rc = wl__tensor_num_rows(x);
    int64_t x_i1 = 0;
    int64_t x_i2 = 0;
    int64_t x_i3 = 0;
    int64_t x_i4 = 0;
    int64_t x_i5 = 0;
    if (y_s0 == sizeof(float)) { /* Fast path for contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = (float*)(b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5);
            const float* const p_x = (const float*)(b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5);
            const float* const p_y = (const float*)(b_y + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5);
            wl__bnd_chk(p_y, b_y, wl__tensor_data_size(y));
            const int64_t pa = x_d0 / y_d0;
            for (int64_t i=0; i < pa; ++i) {  /* For each element in row */
                float* const pp_r = p_r + i*y_d0; /* Compute result ptr. */
                const float* const pp_x = p_x + i*y_d0; /* Compute x ptr. */
                wl__bnd_chk(pp_r, b_r, wl__tensor_data_size(r));
                wl__bnd_chk(pp_x, b_x, wl__tensor_data_size(x));
                wl__vdiv_f32(y_d0, pp_r, pp_x, p_y);  /* Apply micro kernel vector op */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    } else { /* Slow path for non-contiguous input tensors. */
        for (int64_t ri=0; ri < rc; ++ri) { /* For each row */
            /* Compute broadcasted 5D indices for y and the resulting buffer ptrs for r,x,y. */
            const int64_t y_i5 = x_i5 % y_d5;
            const int64_t y_i4 = x_i4 % y_d4;
            const int64_t y_i3 = x_i3 % y_d3;
            const int64_t y_i2 = x_i2 % y_d2;
            const int64_t y_i1 = x_i1 % y_d1;
            float* const p_r = (float*)(b_r + x_i1*r_s1 + x_i2*r_s2 + x_i3*r_s3 + x_i4*r_s4 + x_i5*r_s5);
            const float* const p_x = (const float*)(b_x + x_i1*x_s1 + x_i2*x_s2 + x_i3*x_s3 + x_i4*x_s4 + x_i5*x_s5);
            for (int64_t i=0; i < r_d0; ++i) {  /* For each element in row */
                const float* const p_y = (const float*)(b_y + i%y_d0*y_s0 + y_i1*y_s1 + y_i2*y_s2 + y_i3*y_s3 + y_i4*y_s4 + y_i5*y_s5); /* Compute result ptr. */
                wl__bnd_chk(p_r+i, b_r, wl__tensor_data_size(r));
                wl__bnd_chk(p_x+i, b_x, wl__tensor_data_size(x));
                wl__bnd_chk(p_y, b_y, wl__tensor_data_size(y));
                p_r[i] = p_x[i] / *p_y; /* Apply scalar op. */
            }
            /* Incremental index computation. */
            ++x_i1;
            if (x_i1 >= x_d1) {
                x_i1 = 0;
                ++x_i2;
                if (x_i2 >= x_d2) {
                    x_i2 = 0;
                    ++x_i3;
                    if (x_i3 >= x_d3) {
                        x_i3 = 0;
                        ++x_i4;
                        if (x_i4 >= x_d4) {
                            x_i4 = 0;
                            ++x_i5;
                        }
                    }
                }
            }
        }
    }
}

#if 0 /* Naive matrix multiplication, but no broadcasting support. */
static void WL__HOTPROC wl__blas_matmul_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const wl_tensor_t* const x = inputs[0];
    const wl_tensor_t* const y = inputs[1];
    uint8_t* const b_r = (uint8_t*)r->buf;
    const uint8_t* const b_x = (const uint8_t*)x->buf;
    const uint8_t* const b_y = (const uint8_t*)y->buf;
    wl__load_local_storage_group(r, r_d, shape);
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_d, shape);
    wl__load_local_storage_group(x, x_s, strides);
    wl__load_local_storage_group(y, y_d, shape);
    wl__load_local_storage_group(y, y_s, strides);
    for (int64_t i3=0; i3 < r_d3; ++i3) {
        for (int64_t i2=0; i2 < r_d2; ++i2) {
            for (int64_t i1=0; i1 < r_d1; ++i1) {
                for (int64_t i0=0; i0 < r_d1; ++i0) {
                    double sum = 0.0;
                    for (int64_t k=0; k < x_d0; ++k) {
                        const float* const p_x = (const float*)(b_x + k*x_s0 + i0*x_s1 + i2*x_s2 + i3*x_s3);
                        const float* const p_y = (const float*)(b_y + i1*y_s0 + k*y_s1 + i2*y_s2 + i3*y_s3);
                        wl__bnd_chk(p_x, b_x, x->buf_size);
                        wl__bnd_chk(p_y, b_y, y->buf_size);
                        sum += (double)(*p_x**p_y);
                    }
                    float* const p_r = (float*)(b_r + i1*r_s0 + i0*r_s1 + i2*r_s2 + i3*r_s3);
                    wl__bnd_chk(p_r, b_r, r->buf_size);
                    *p_r = (float)sum;
                }
            }
        }
    }
}
#endif

/*
** Matrix multiplication.
** R = A x B
*/
static void WL__HOTPROC wl__blas_matmul_f32(
    const wl__blas_compute_info_t* const bci,
    wl_tensor_t* const r,
    const wl_tensor_t** const inputs /* Assumes correct inputs for op, all != NULL! */
) {
    const wl_tensor_t* const x = inputs[0];
    const wl_tensor_t* const y = inputs[1];
    float* const b_r = (float*)r->buf;
    const float* const b_x = (const float*)x->buf;
    const float* const b_y = (const float*)y->buf;
    wl__load_local_storage_group(r, r_d, shape);
    wl__load_local_storage_group(r, r_s, strides);
    wl__load_local_storage_group(x, x_d, shape);
    wl__load_local_storage_group(x, x_s, strides);
    wl__load_local_storage_group(y, y_d, shape);
    wl__load_local_storage_group(y, y_s, strides);
    wl__assert2(x_d2 == 1 && x_d3 == 1);
    wl__assert2(y_d2 == 1 && y_d3 == 1);
    const int64_t rows = x_d0;
    const int64_t cols = x_d1;
    const int64_t inners = y_d1;
#if 1 /* Reordering of loops for better cache locality. */
    for (int64_t i=0; i < rows; ++i) {
        for (int64_t k=0; k < cols; ++k) {
            const float* const p_x = b_x + x_d1*i + k;
            wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
            for (int64_t j = 0; j < inners; ++j) {
                float* const p_r = b_r + r_d1*i + j;
                const float* const p_y = b_y + y_d1*k + j;
                wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
                wl__bnd_chk(p_y, b_y, wl__tensor_data_size(y));
                *p_r += *p_x * *p_y;
            }
        }
    }
#elif 1 /* Tiled matrix multiplication. */
    const int64_t TILE_I = 256;
    const int64_t TILE_K = 256;
    const int64_t TILE_J = 256;
    for (int64_t ii = 0; ii < rows; ii += TILE_I) {
        int64_t i_end = (ii + TILE_I < rows) ? ii + TILE_I : rows;
        for (int64_t kk = 0; kk < cols; kk += TILE_K) {
            int64_t k_end = (kk + TILE_K < cols) ? kk + TILE_K : cols;
            for (int64_t jj = 0; jj < inners; jj += TILE_J) {
                int64_t j_end = (jj + TILE_J < inners) ? jj + TILE_J : inners;
                for (int64_t i = ii; i < i_end; ++i) {
                    for (int64_t k = kk; k < k_end; ++k) {
                        const float* const p_x = b_x + x_d1 * i + k;
                        wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
                        for (int64_t j = jj; j < j_end; ++j) {
                            float* const p_r = b_r + r_d1 * i + j;
                            const float* const p_y = b_y + y_d1 * k + j;
                            wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
                            wl__bnd_chk(p_y, b_y, wl__tensor_data_size(y));
                            *p_r += *p_x * *p_y;
                        }
                    }
                }
            }
        }
    }
#else /* Vector dot call */
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t k = 0; k < cols; ++k) {
            const float* const p_x = b_x + x_d1 * i + k;
            wl__bnd_chk(p_x, b_x, wl__tensor_data_size(x));
            float* const p_r = b_r + r_d1 * i;
            const float* const p_y = b_y + y_d1 * k;
            wl__bnd_chk(p_r, b_r, wl__tensor_data_size(r));
            wl__bnd_chk(p_y, b_y, wl__tensor_data_size(y));
            *p_r = wl__vdot_f32(inners, p_x, p_y);
        }
    }
#endif
}

/* Dispatch table for default CPU-implementation. */
static void wl__blas_compute_dispatch_table_default(void (*(*const dispatch_lut)[WL_OP__COUNT])(const wl__blas_compute_info_t*, wl_tensor_t*, const wl_tensor_t**)) {
    (*dispatch_lut)[WL_OP_NOP] = &wl__blas_nop; /* No operation */
    (*dispatch_lut)[WL_OP_CLONE] = &wl__blas_clone;
    (*dispatch_lut)[WL_OP_VIEW] = &wl__blas_nop; /* View is a no-op */
    (*dispatch_lut)[WL_OP_TRANSPOSE] = &wl__blas_nop; /* Transpose is a runtime no-op */
    (*dispatch_lut)[WL_OP_PERMUTE] = &wl__blas_nop; /* Transpose is a runtime no-op */
    (*dispatch_lut)[WL_OP_MEAN] = &wl__blas_mean_f32;
    (*dispatch_lut)[WL_OP_SUM] = &wl__blas_sum_f32;
    (*dispatch_lut)[WL_OP_ABS] = &wl__blas_abs_f32;
    (*dispatch_lut)[WL_OP_NEG] = &wl__blas_neg_f32;
    (*dispatch_lut)[WL_OP_LOG] = &wl__blas_log_f32;
    (*dispatch_lut)[WL_OP_SQR] = &wl__blas_sqr_f32;
    (*dispatch_lut)[WL_OP_SQRT] = &wl__blas_sqrt_f32;
    (*dispatch_lut)[WL_OP_SIN] = &wl__blas_sin_f32;
    (*dispatch_lut)[WL_OP_COS] = &wl__blas_cos_f32;
    (*dispatch_lut)[WL_OP_STEP] = &wl__blas_step_f32;
    (*dispatch_lut)[WL_OP_SOFTMAX] = &wl__blas_softmax_f32;
    (*dispatch_lut)[WL_OP_SOFTMAX_DV] = &wl__blas_softmax_dv_f32;
    (*dispatch_lut)[WL_OP_SIGMOID] = &wl__blas_sigmoid_f32;
    (*dispatch_lut)[WL_OP_SIGMOID_DV] = &wl__blas_sigmoid_dv_f32;
    (*dispatch_lut)[WL_OP_HARD_SIGMOID] = &wl__blas_hard_sigmoid_f32;
    (*dispatch_lut)[WL_OP_SILU] = &wl__blas_silu_f32;
    (*dispatch_lut)[WL_OP_SILU_DV] = &wl__blas_silu_dv_f32;
    (*dispatch_lut)[WL_OP_TANH] = &wl__blas_tanh_f32;
    (*dispatch_lut)[WL_OP_TANH_DV] = &wl__blas_tanh_dv_f32;
    (*dispatch_lut)[WL_OP_RELU] = &wl__blas_relu_f32;
    (*dispatch_lut)[WL_OP_RELU_DV] = &wl__blas_relu_dv_f32;
    (*dispatch_lut)[WL_OP_GELU] = &wl__blas_gelu_f32;
    (*dispatch_lut)[WL_OP_GELU_DV] = &wl__blas_gelu_dv_f32;
    (*dispatch_lut)[WL_OP_ADD] = &wl__blas_add_f32;
    (*dispatch_lut)[WL_OP_SUB] = &wl__blas_sub_f32;
    (*dispatch_lut)[WL_OP_MUL] = &wl__blas_mul_f32;
    (*dispatch_lut)[WL_OP_DIV] = &wl__blas_div_f32;
    (*dispatch_lut)[WL_OP_MATMUL] = &wl__blas_matmul_f32;
}

static void wl__blas_compute_dispatch_table_install(wl_ctx_t* const ctx) {
    wl__blas_compute_dispatch_table_default(&ctx->blas_dispatch);
    /* TODO: Add support for custom implementations for host CPU arch. */
    for (uint32_t i=WL_OP_NOP; i < WL_OP__COUNT; ++i) { /* Verify that all ops have a implementation, except NOP. */
        wl__assert(ctx->blas_dispatch[i] != NULL, "No default CPU implementation for op: %s", wl_op_get_name((wl_op_t)i));
    }
}

static void wl__tensor_graph_visit_node(wl_tensor_t* node, void (*visitor)(wl_tensor_t*, void*), bool forward, void* ud) {
    if (wl__unlikely(!node)) return;
    wl_tensor_t** parent_nodes = node->op_inputs;
    uint32_t n = wl_op_get_argcount(node->op);
    for (uint32_t i=0; i < n; ++i) {
        uint32_t j = forward ? i : n-i-1; /* Left-to-right or right-to-left */
        if (wl__likely(parent_nodes[j])) {
            wl__tensor_graph_visit_node(parent_nodes[j], visitor, forward, ud);
        }
    }
    (*visitor)(node, ud); /* Dispatch visitor hook */
}

struct wl_compute_graph_t {
    wl_ctx_t* ctx;
    wl_tensor_t** internal_nodes;
    wl_tensor_t** leaf_nodes;
    size_t num_nodes_total;
    size_t num_internal_nodes;
    size_t num_leaf_nodes;
    size_t mem_size_total;
    wl__hashset_t visited_hs;
    wl_graph_eval_order_t order;
    char name[WL_MAX_TENSOR_NAME_LEN];
};

static void WL__HOTPROC wl__compute_graph_accumulate_visitor(wl_tensor_t* node, void* ud) { /* Count number of tensors in graph. */
    ++*(size_t*)ud;
}

static void WL__HOTPROC wl__compute_graph_coalescence_nodes_visitor(wl_tensor_t* node, void* ud) { /* Count number of tensors in graph. */
    wl__assert(!(node->flags & WL__TFLAG_EXEC_EAGER), "Tensor must be in deferred execution mode.");
    wl_compute_graph_t* gra = (wl_compute_graph_t*)ud;
    size_t hz = wl__hashset_insert(&gra->visited_hs, node);
    wl__assert2(hz != WL__HASHSET_FULL);
    if (wl__unlikely(hz == WL__HASHSET_DUPLICATE)) return; /* Already visited */
    if (node->op == WL_OP_NOP) { /* Leaf node (constant, out of gradient flow) */
        gra->leaf_nodes[gra->num_leaf_nodes++] = node;
        for (uint32_t i=0; i < WL_MAX_INPUT_TENSORS; ++i) { /* All inputs must be NULL for NOP node. */
            wl__assert2(!node->op_inputs[i]);
        }
    } else { /* Non-leaf node */
        gra->internal_nodes[gra->num_internal_nodes++] = node;
        for (uint32_t i=0; i < wl_op_get_argcount(node->op); ++i) { /* All required inputs must be not NULL for operation node. */
            wl__assert2(i < sizeof(node->op_inputs)/sizeof(*node->op_inputs) && node->op_inputs[i]);
        }
    }
}

wl_compute_graph_t* wl_compute_graph_compile(wl_ctx_t* ctx, wl_tensor_t* root, wl_graph_eval_order_t order, const char* name) {
    wl__assert(root->ctx == ctx && !(root->flags & WL__TFLAG_EXEC_EAGER), "Tensor must be in deferred execution mode, to be used with static graphs.");
    size_t total_nodes = 0;
    wl__tensor_graph_visit_node(root, &wl__compute_graph_accumulate_visitor, order == WL_GRAPH_EVAL_ORDER_FORWARD, &total_nodes);
    wl__assert2(total_nodes > 0);
    uintptr_t mem_req = 0; /* Memory required for compute graph. */
    wl__pincr((void**)&mem_req, sizeof(wl_compute_graph_t), __alignof__(wl_compute_graph_t)); /* Graph struct itself */
    wl__pincr((void**)&mem_req, total_nodes*sizeof(*((wl_compute_graph_t*)0)->internal_nodes), __alignof__(*((wl_compute_graph_t*)0)->internal_nodes)); /* Nodes. */
    wl__pincr((void**)&mem_req, total_nodes*sizeof(*((wl_compute_graph_t*)0)->leaf_nodes), __alignof__(*((wl_compute_graph_t*)0)->leaf_nodes)); /* Leafs. */
    wl_compute_graph_t* gra = (wl_compute_graph_t*)wl_ctx_pool_alloc_aligned(ctx, mem_req, __alignof__(*gra));
    void* data = gra+1; /* Start of data, end of header */
    memset(gra, 0, mem_req);
    gra->ctx = ctx;
    gra->num_nodes_total = total_nodes;
    gra->internal_nodes = wl__pincr(&data, total_nodes*sizeof(*gra->internal_nodes), __alignof__(*gra->internal_nodes)); /* Fetch nodes. */
    gra->leaf_nodes = wl__pincr(&data, total_nodes*sizeof(*gra->leaf_nodes), __alignof__(*gra->leaf_nodes)); /* Fetch leafs. */
    gra->mem_size_total = mem_req;
    gra->order = order;
    gra->visited_hs = wl__hashset_create_pooled(ctx, total_nodes);
    wl__hashset_reset(&gra->visited_hs);
    size_t n_nodes = gra->num_internal_nodes;
    wl__tensor_graph_visit_node(root, &wl__compute_graph_coalescence_nodes_visitor, order == WL_GRAPH_EVAL_ORDER_FORWARD, gra);
    size_t new_nodes = gra->num_internal_nodes - n_nodes;
    if (new_nodes > 0) /* Latest node must be starting point. */
        wl__assert2(gra->internal_nodes[gra->num_internal_nodes - 1] == root);
    wl__assert2(gra->num_internal_nodes + gra->num_leaf_nodes <= total_nodes);
    if (name && *name) snprintf(gra->name, sizeof(gra->name), "%s", name);
    return gra;
}

wl_tensor_t* WL__HOTPROC wl_compute_graph_execute(wl_compute_graph_t* gra) {
    wl__assert2(gra->num_internal_nodes <= INT64_MAX);
    wl_tensor_t** nodes = gra->internal_nodes;
    size_t n = gra->num_internal_nodes;
    wl__assert2(gra->num_nodes_total && nodes[n-1]);
    wl_tensor_t* root = nodes[n-1]; /* Evaluation root node */
    wl__blas_compute_info_t bci;
    wl__blas_compute_info_sequential(gra->ctx, &bci);
    for (size_t i=0; i < n; ++i) { /* Execute all folded internal operation nodes in order. */
        wl_tensor_t* R = nodes[i];
        wl__op_execute(R, R->op, (const wl_tensor_t**)R->op_inputs, &bci);
    }
    return root;
}

bool wl_compute_graph_contains(const wl_compute_graph_t* gra, const wl_tensor_t* t) {
    for (size_t i = 0; i < gra->num_internal_nodes; ++i) /* Linear search for internal nodes. */
        if (gra->internal_nodes[i] == t) return true;
    for (size_t i = 0; i < gra->num_leaf_nodes; ++i) /* Linear search for leaf nodes. */
        if (gra->leaf_nodes[i] == t) return true;
    return false;
}

void WL__COLDPROC wl_compute_graph_dump_to_dot(const wl_compute_graph_t* gra, const char* file_name) {
    FILE* f = wl__fopen(file_name, "wt");
    if (wl__unlikely(!f)) {
        wl__log_error("Failed to open file for writing: %s", file_name);
        return;
    }
    fprintf(f, "digraph G {\n");
    fprintf(f, "\tnewrank = true;\n");
    fprintf(f, "\trankdir = LR;\n");
    char color[32];
    for (size_t i=0; i < gra->num_internal_nodes; ++i) {
        const wl_tensor_t* node = gra->internal_nodes[i];
        snprintf(color, sizeof(color), "paleturquoise1");
        fprintf(
            f,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = Mrecord; "
            "label=\"",
            (void*)node,
            color
        );
        if (*node->name) fprintf(f, "%s (%s)|", node->name, wl_dtype_info_of(node->dtype)->name);
        else fprintf(f, "(%s)|", wl_dtype_info_of(node->dtype)->name);
        if (wl_tensor_is_matrix(node)) fprintf(f, "OP #%zu [%zu, %zu] | <x>%s", i, (size_t)node->shape[0], (size_t)node->shape[1], wl_op_get_name(node->op));
        else fprintf(f, "OP #%zu [%zu, %zu, %zu] | <x>%s", i, (size_t)node->shape[0], (size_t)node->shape[1], (size_t)node->shape[2], wl_op_get_name(node->op));
        fprintf(f, "\"; ]\n");
    }
    for (size_t i=0; i < gra->num_leaf_nodes; ++i) {
        const wl_tensor_t* node = gra->leaf_nodes[i];
        snprintf(color, sizeof(color), "palegreen1");
        fprintf(
            f,
            "  \"%p\" [ "
            "style = filled; fillcolor = %s; shape = Mrecord; "
            "label=\"<x>",
            (void*)node,
            color
        );
        if (*node->name) fprintf(f, "%s (%s)|", node->name, wl_dtype_info_of(node->dtype)->name);
        else fprintf(f, "(%s)|", wl_dtype_info_of(node->dtype)->name);
        fprintf(f, "IN #%zu [%zu, %zu]", i, (size_t)node->shape[0], (size_t)node->shape[1]);
        size_t n = wl__tensor_num_elements(node);
        if (n < 4) {
            fprintf(f, " | (");
            for (size_t j=0; j < n; ++j) {
                switch (node->dtype) {
                    case WL_DTYPE_F32: {
                        char fmt_buf[128];
                        *wl__fmt_f64(WL__FMT_G14, (double)wl_tensor_get_scalar_virtual_index(node, (int64_t)j), fmt_buf) = '\0';
                        fprintf(f, "%s", fmt_buf);
                    } break;
                    default: wl__log_error("Invalid DType for format"); continue;
                }
                if (j < n-1) fprintf(f, ", ");
            }
            fprintf(f, ")");
        }
        fprintf(f, "\"; ]\n");
    }
    for (size_t i=0; i < gra->num_internal_nodes; ++i) {
        const wl_tensor_t* node = gra->internal_nodes[i];
        for (size_t j=0; j < WL_MAX_INPUT_TENSORS; ++j) {
            if (node->op_inputs[j]) {
                char label[16];
                snprintf(label, sizeof(label), "IN #%zu", j);
                fprintf(
                    f,
                    "  \"%p\":x -> \"%p\":x [ arrowhead = none; style = solid; label = \"%s\"; ]\n",
                    (void*)node->op_inputs[j],
                    (void*)node,
                    label
                );
            }
        }
    }
    for (size_t i=0; i < gra->num_leaf_nodes; ++i) {
        const wl_tensor_t* node = gra->leaf_nodes[i];
        for (size_t j=0; j < WL_MAX_INPUT_TENSORS; ++j) {
            if (node->op_inputs[j]) {
                char label[16];
                snprintf(label, sizeof(label), "IN #%zu", j);
                fprintf(
                        f,
                        "  \"%p\":%s -> \"%p\":%s [ label = \"%s\"; ]\n",
                        (void*)node->op_inputs[j], "x",
                        (void*)node, "x",
                        label
                );
            }
        }
    }
    fprintf(f, "}\n");
    wl__log_info("dot -Tpng %s -o %s.png && open %s.png\n", file_name, file_name, file_name);
    fclose(f);
}

wl_ctx_t* wl_compute_graph_get_ctx(const wl_compute_graph_t* gra) { return gra->ctx; }

const char* wl_compute_graph_get_name(const wl_compute_graph_t* gra) { return gra->name; }

const wl_tensor_t** wl_compute_graph_get_internal_nodes(const wl_compute_graph_t* gra, size_t* n_nodes) {
    if (n_nodes) *n_nodes = gra->num_internal_nodes;
    return (const wl_tensor_t**)gra->internal_nodes;
}

const wl_tensor_t** wl_compute_graph_get_leaf_nodes(const wl_compute_graph_t* gra, size_t* n_leaves) {
    if (n_leaves) *n_leaves = gra->num_leaf_nodes;
    return (const wl_tensor_t**)gra->leaf_nodes;
}

size_t wl_compute_graph_get_num_total_nodes(const wl_compute_graph_t* gra) { return gra->num_nodes_total; }

size_t wl_compute_graph_get_num_internal_nodes(const wl_compute_graph_t* graph) { return graph->num_internal_nodes; }

size_t wl_compute_graph_get_num_leaf_nodes(const wl_compute_graph_t* gra) { return gra->num_leaf_nodes; }

size_t wl_compute_graph_get_order(const wl_compute_graph_t* gra) { return gra->order; }

size_t wl_compute_graph_get_memory_usage(const wl_compute_graph_t* gra) { return gra->mem_size_total; }

#ifdef __APPLE__
    static bool wl__sysctl_mib01(uint8_t (*out)[256], size_t* o_len, int mib0, int mib1) { /* Get sysctl data */
        memset(out, 0, sizeof(*out));
        *o_len = 0;
        int name[2] = {mib0, mib1};
        size_t len = 0;
        if (wl__unlikely(sysctl(name, sizeof(name) / sizeof(*name), NULL, &len, NULL, 0))) return false; /* Get length */
        if (wl__unlikely(len >= sizeof(*out))) return false; /* Buffer too small */
        if (wl__unlikely(sysctl(name, sizeof(name) / sizeof(*name), *out, &len, NULL, 0))) return false; /* Get data */
        *o_len = len;
        return true;
    }
    static bool wl__sysctl_key(uint8_t (*out)[256], size_t* o_len, const char* key) { /* Get sysctl data */
        memset(out, 0, sizeof(*out));
        *o_len = 0;
        size_t len = 0;
        if (wl__unlikely(sysctlbyname(key, NULL, &len, NULL, 0))) return false; /* Get length */
        if (wl__unlikely(len >= sizeof(*out))) return false; /* Buffer too small */
        if (wl__unlikely(sysctlbyname(key, *out, &len, NULL, 0))) return false; /* Get data */
        *o_len = len;
        return true;
    }
    static uint64_t wl__sysctl_unpack_int(const uint8_t (*in)[256], size_t len) { /* Unpack sysctl data */
        switch (len) {
            case sizeof(uint16_t): { uint16_t r; memcpy(&r, *in, sizeof(r)); return r; }
            case sizeof(uint32_t): { uint32_t r; memcpy(&r, *in, sizeof(r)); return r; }
            case sizeof(uint64_t): { uint64_t r; memcpy(&r, *in, sizeof(r)); return r; }
            default: return 0;
        }
    }
#else
    static bool wl__cpuinfo_parse_value(const char* key, char (*out)[128]) {
        FILE* cpuinfo = wl__fopen("/proc/cpuinfo", "rt");
        if (wl__unlikely(!cpuinfo)) return false;
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
                if (wl__unlikely(!value_len || value_len >= sizeof(*out))) {
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
    static uint64_t wl__parse_meminfo_value(const char* line) {
        const char *p = strchr(line, ':');
        if (wl__unlikely(!p)) return 0;
        ++p;
        p += strspn(p, " \t");
        errno = 0;
        char* end;
        uint64_t value = strtoull(p, &end, 10);
        if (wl__unlikely(errno != 0 || p == end)) return 0;
        return value<<10;
    }
#endif

static void wl_system_host_info_query_os_name(char (*out_os_name)[128]) { /* Get OS name */
    #ifdef _WIN32
    #error "Unsupported platform"
    #elif defined(__APPLE__)
        size_t len;
        uint8_t tmp[256];
        if (wl__likely(wl__sysctl_mib01(&tmp, &len, CTL_KERN, KERN_VERSION) && len && *tmp))
            snprintf(*out_os_name, sizeof(*out_os_name), "%s", (const char*)tmp);
    #else
        // TODO: Linux
    #endif
}

static void wl_system_host_info_query_cpu_name(char (*out_cpu_name)[128]) { /* Get CPU name */
    #ifdef _WIN32
    #error "Unsupported platform"
    #elif defined(__APPLE__)
        size_t len;
        uint8_t tmp[256];
        if (wl__likely(wl__sysctl_key(&tmp, &len, "machdep.cpu.brand_string") && len && *tmp))
            snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", (const char*)tmp);
    #else
        char cpu_name[128];
        if (wl__likely(wl__cpuinfo_parse_value("model name", &cpu_name) && *cpu_name))
            snprintf(*out_cpu_name, sizeof(*out_cpu_name), "%s", cpu_name);
    #endif
}

static void wl_system_host_info_query_cpu_cores(uint32_t* out_virtual, uint32_t* out_physical, uint32_t* out_sockets) { /* Get CPU virtual (logical) cores. */
    #ifdef _WIN32
    #error "Unsupported platform"
    #elif defined(__APPLE__)
        uint8_t tmp[256];
        size_t len;
        if (wl__likely(wl__sysctl_key(&tmp, &len, "machdep.cpu.thread_count") && len))
            *out_virtual = wl__sysctl_unpack_int(&tmp, len);
        if (wl__likely(wl__sysctl_key(&tmp, &len, "machdep.cpu.core_count") && len))
            *out_physical = wl__sysctl_unpack_int(&tmp, len);
        if (wl__likely(wl__sysctl_key(&tmp, &len, "hw.packages") && len))
            *out_sockets = wl__sysctl_unpack_int(&tmp, len);
    #else
        long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
        *out_virtual = nprocs > 0 ? (uint32_t)nprocs : 0;
        FILE* cpuinfo = wl__fopen("/proc/cpuinfo", "r");
        if (wl__unlikely(!cpuinfo)) return;
        uint32_t physical_ids[WL__MAX_CPUS];
        uint32_t core_ids[WL__MAX_CPUS];
        uint32_t package_ids[WL__MAX_CPUS];
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
                        if (cpu_count < WL__MAX_CPUS) {
                            physical_ids[cpu_count] = current_physical_id;
                            core_ids[cpu_count] = current_core_id;
                            ++cpu_count;
                        } else break;
                    }
                    is_unique = true;
                    for (int32_t i = 0; i < package_count; ++i) if (package_ids[i] == current_physical_id) { is_unique = false; break; }
                    if (is_unique) {
                        if (package_count < WL__MAX_CPUS) package_ids[package_count++] = current_physical_id;
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

static void wl__system_host_info_query_memory(uint64_t* out_phys_mem_total, uint64_t* out_phys_mem_free) { /* Get physical memory */
    #ifdef _WIN32
    #error "Unsupported platform"
    #elif defined(__APPLE__)
        uint8_t tmp[256];
        size_t len;
        if (wl__likely(wl__sysctl_mib01(&tmp, &len, CTL_HW, HW_MEMSIZE) && len))
            *out_phys_mem_total = wl__sysctl_unpack_int(&tmp, len);
        struct vm_statistics64 stats;
        natural_t count = HOST_VM_INFO64_COUNT;
        if (wl__likely(host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)(&stats), &count) == KERN_SUCCESS))
            *out_phys_mem_free = stats.free_count * getpagesize();
    #else
        FILE* meminfo = wl__fopen("/proc/meminfo", "r");
        if (wl__unlikely(!meminfo)) return;
        char line[256];
        while (fgets(line, sizeof(line), meminfo)) {
            if (strncmp(line, "MemTotal:", sizeof("MemTotal:")-1) == 0)
                *out_phys_mem_total = wl__parse_meminfo_value(line);
            else if (strncmp(line, "MemAvailable:", sizeof("MemAvailable:")-1) == 0)
                *out_phys_mem_free = wl__parse_meminfo_value(line);
        }
        fclose(meminfo);
    #endif
}

#if defined(__x86_64__) || defined(_M_X64)
    static uint64_t WL__AINLINE wl__xgetbv(void) { /* Query extended control register value. */
        #ifdef _MSC_VER
            return _xgetbv(0);
        #else
            uint32_t lo, hi;
            __asm__ __volatile__("xgetbv\n\t" : "=a" (lo), "=d" (hi) : "c" (0));
            return (uint64_t)lo | ((uint64_t)hi << 32);
        #endif
    }
    #define wl__cpy_regs(id) \
        (*features)[WL__X86_64_CPUID_##id][WL__X86_64_CPUID_EAX] = eax; \
        (*features)[WL__X86_64_CPUID_##id][WL__X86_64_CPUID_EBX] = ebx; \
        (*features)[WL__X86_64_CPUID_##id][WL__X86_64_CPUID_ECX] = ecx; \
        (*features)[WL__X86_64_CPUID_##id][WL__X86_64_CPUID_EDX] = edx
    static void wl__system_info_query_x86_64_cpu_features(uint32_t (*features)[8][4]) {
        uint32_t eax, ebx, ecx, edx;
        uint32_t max_basic_leaf, max_extended_leaf;

        __cpuid(0, eax, ebx, ecx, edx);
        wl__cpy_regs(0H);
        max_basic_leaf = eax;
        __cpuid(0x80000000u, eax, ebx, ecx, edx);
        max_extended_leaf = eax;
        if (max_basic_leaf >= 1u) {
            __cpuid(1, eax, ebx, ecx, edx);
            wl__cpy_regs(1H);
        }
        if (max_basic_leaf >= 2u) {
            __cpuid(2u, eax, ebx, ecx, edx);
            wl__cpy_regs(2H);
        }
        if (max_basic_leaf >= 7u) {
            __cpuid_count(7u, 0, eax, ebx, ecx, edx);
            wl__cpy_regs(7H);
        }
        if (max_basic_leaf >= 7u) {
            __cpuid_count(7u, 1, eax, ebx, ecx, edx);
            wl__cpy_regs(7H_1H);
        }
        if (max_basic_leaf >= 0x16u) {
            __cpuid(0x16u, eax, ebx, ecx, edx);
            wl__cpy_regs(16H);
        }
        if (max_extended_leaf >= 0x80000001u) {
            __cpuid(0x80000001u, eax, ebx, ecx, edx);
            wl__cpy_regs(80000001H);
        }
        if (max_extended_leaf >= 0x80000007u) {
            __cpuid(0x80000007u, eax, ebx, ecx, edx);
            wl__cpy_regs(80000007H);
        }
        bool cpu_avx_support = ((*features)[WL__X86_64_CPUID_1H][WL__X86_64_CPUID_ECX] & 0x10000000u) != 0;
        bool cpu_osxsave_support = ((*features)[WL__X86_64_CPUID_1H][WL__X86_64_CPUID_ECX] & 0x8000000u) != 0;
        if (cpu_avx_support && cpu_osxsave_support) {
            uint64_t xcr0 = wl__xgetbv();
            if ((xcr0 & 0x6) != 0x6u) {
                (*features)[WL__X86_64_CPUID_1H][WL__X86_64_CPUID_ECX] &= ~0x10000000u; /* Clear AVX */
                (*features)[WL__X86_64_CPUID_7H][WL__X86_64_CPUID_EBX] &= ~0x20u; /* Clear AVX2 */
            }
            if ((xcr0 & 0xe0) != 0xe0u) { /* OS does not support AVX-512, clear AVX512 */
                (*features)[WL__X86_64_CPUID_7H][WL__X86_64_CPUID_EBX] &= ~0xdc230000u;
                (*features)[WL__X86_64_CPUID_7H][WL__X86_64_CPUID_ECX] &= ~0x5842u;
                (*features)[WL__X86_64_CPUID_7H][WL__X86_64_CPUID_EDX] &= ~0x10cu;
                (*features)[WL__X86_64_CPUID_7H_1H][WL__X86_64_CPUID_EAX] &= ~0x20u;
            }
        } else {
            (*features)[WL__X86_64_CPUID_1H][WL__X86_64_CPUID_ECX] &= ~0x10000000u; /* Clear AVX */
            (*features)[WL__X86_64_CPUID_7H][WL__X86_64_CPUID_EBX] &= ~0x20u; /* Clear AVX2 */
            (*features)[WL__X86_64_CPUID_7H][WL__X86_64_CPUID_EBX] &= ~0xdc230000u; /* Clear AVX512 */
            (*features)[WL__X86_64_CPUID_7H][WL__X86_64_CPUID_ECX] &= ~0x5842u; /* Clear AVX512 */
            (*features)[WL__X86_64_CPUID_7H][WL__X86_64_CPUID_EDX] &= ~0x10cu; /* Clear AVX512 */
            (*features)[WL__X86_64_CPUID_7H_1H][WL__X86_64_CPUID_EAX] &= ~0x20u; /* Clear AVX512 */
        }
    }
    #undef wl__cpy_regs
#endif

static void wl__system_host_info_query(wl_ctx_t* ctx) {
    wl_system_host_info_query_os_name(&ctx->sys.os_name);
    wl_system_host_info_query_cpu_name(&ctx->sys.cpu_name);
    wl_system_host_info_query_cpu_cores(&ctx->sys.cpu_virtual_cores, &ctx->sys.cpu_physical_cores, &ctx->sys.cpu_sockets);
    wl__system_host_info_query_memory(&ctx->sys.phys_mem_total, &ctx->sys.phys_mem_free);
    #if defined(__x86_64__) || defined(_M_X64)
        wl__system_info_query_x86_64_cpu_features(&ctx->sys.x86_64_cpu_features);
    #endif
    if (wl__unlikely(!*ctx->sys.os_name)) snprintf(ctx->sys.os_name, sizeof(ctx->sys.os_name), "Unknown");
    if (wl__unlikely(!*ctx->sys.cpu_name)) snprintf(ctx->sys.cpu_name, sizeof(ctx->sys.cpu_name), "Unknown");
}

static WL__AINLINE void wl__sto_write_u32_le(uint8_t** p, uint32_t x) {
    x = wl__bswap32(x);
    memcpy(*p, &x, sizeof(x));
    *p += sizeof(x);
}

static WL__AINLINE void wl__sto_write_u64_le(uint8_t** p, uint64_t x) {
    x = wl__bswap64(x);
    memcpy(*p, &x, sizeof(x));
    *p += sizeof(x);
}

static WL__AINLINE uint32_t wl__sto_read_u32_le(const uint8_t** p) {
    uint32_t x;
    memcpy(&x, *p, sizeof(x));
    x = wl__bswap32(x);
    *p += sizeof(x);
    return x;
}

static WL__AINLINE uint64_t wl__sto_read_u64_le(const uint8_t** p) {
    uint64_t x;
    memcpy(&x, *p, sizeof(x));
    x = wl__bswap64(x);
    *p += sizeof(x);
    return x;
}

#define WL__STO_MAGIC "WAVELET!"
wl_static_assert(sizeof(WL__STO_MAGIC)-1 == sizeof(uint64_t));
#define WL__STO_FILE_HEADER_SIZE ((sizeof(WL__STO_MAGIC)-1) + sizeof(uint32_t)*3)
#define WL__STO_TENSOR_HEADER_SIZE (WL_MAX_TENSOR_NAME_LEN + sizeof(uint32_t) + sizeof(int64_t)*WL_MAX_DIMS)
wl_static_assert(WL_MAX_TENSOR_NAME_LEN % 8 == 0);
wl_static_assert(WL_DTYPE_COUNT_ <= 0xff);
wl_static_assert(WL_MAX_DIMS <= 0xff);
#define wl__sto_sanitize(exp, ret) do { if (wl__unlikely(!(exp))) { wl__log_error("WAVELET storage sanitize error: " #exp); return (ret); } } while (0)

static bool wl__sto_write_file_header( /* Write file header -  file header must be same in every version. */
    uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    uint32_t num_tensors,
    uint32_t ud
) {
    wl__sto_sanitize(*p + WL__STO_FILE_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    uint64_t wl_magic;
    memcpy(&wl_magic, WL__STO_MAGIC, sizeof(wl_magic));
    wl__sto_write_u64_le(p, wl_magic);
    wl__sto_write_u32_le(p, version);
    wl__sto_write_u32_le(p, num_tensors);
    wl__sto_write_u32_le(p, ud);
    return wl__likely(*p - start == WL__STO_FILE_HEADER_SIZE);
}

static bool wl__sto_read_file_header(  /* Read file header - file header must be same in every version. */
    const uint8_t** p,
    const uint8_t* end,
    uint32_t* version,
    uint32_t* num_tensors,
    uint32_t* ud
) {
    wl__sto_sanitize(*p + WL__STO_FILE_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    uint64_t wl_magic = wl__sto_read_u64_le(p);
    wl__sto_sanitize(memcmp(&wl_magic, WL__STO_MAGIC, sizeof(wl_magic)) == 0, false);
    *version = wl__sto_read_u32_le(p);
    *num_tensors = wl__sto_read_u32_le(p);
    *ud = wl__sto_read_u32_le(p);
    return wl__likely(*p - start == WL__STO_FILE_HEADER_SIZE);
}

static bool wl__sto_write_tensor_header(
    uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    const char (*name)[WL_MAX_TENSOR_NAME_LEN],
    wl__tensor_flags_t flags,
    wl_dtype_t dtype,
    int64_t rank,
    const int64_t (*shape)[WL_MAX_DIMS]
) {
    wl__sto_sanitize(*p + WL__STO_TENSOR_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            uint64_t name_u64[sizeof(*name)/sizeof(uint64_t)];
            memcpy(name_u64, *name, sizeof(*name));
            for (size_t i=0; i < sizeof(name_u64)/sizeof(*name_u64); ++i)   /* Write name as multiple u64 */
                wl__sto_write_u64_le(p, name_u64[i]);
            uint32_t aux = 0;   /* Pack small fields into aux field */
            aux |= (flags & 0xff) << 16;
            aux |= (dtype & 0xff) << 8;
            aux |= (rank & 0xff);
            wl__sto_write_u32_le(p, aux);     /* Write aux field */
            for (size_t i=0; i < WL_MAX_DIMS; ++i) {      /* Write shape */
                wl__sto_sanitize((*shape)[i] >= 1 && (*shape)[i] < INT64_MAX, false);
                wl__sto_write_u64_le(p, (uint64_t)(*shape)[i]);
            }
        } break;
        default: return false;
    }
    return wl__likely(*p - start == WL__STO_TENSOR_HEADER_SIZE);
}

static bool wl__sto_read_tensor_header(
    const uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    char (*name)[WL_MAX_TENSOR_NAME_LEN],
    wl__tensor_flags_t* flags,
    wl_dtype_t* dtype,
    int64_t* rank,
    int64_t (*shape)[WL_MAX_DIMS]
) {
    wl__sto_sanitize(*p + WL__STO_TENSOR_HEADER_SIZE < end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            uint64_t name_u64[sizeof(*name)/sizeof(uint64_t)];
            for (size_t i=0; i < sizeof(name_u64)/sizeof(*name_u64); ++i) /* Read name as multiple u64 */
                name_u64[i] = wl__sto_read_u64_le(p);
            uint32_t aux = wl__sto_read_u32_le(p); /* Read aux field */
            *flags = (wl__tensor_flags_t)((aux >> 16) & 0xff);
            *dtype = (wl_dtype_t)((aux >> 8) & 0xff);
            *rank = (int64_t)(aux & 0xff);
            wl__sto_sanitize(*flags == 0 || (*flags <= 0xff && ((*flags & 1) == 0)), false); /* Check fields */
            wl__sto_sanitize(*dtype >= 0 && *dtype < WL_DTYPE_COUNT_, false);
            wl__sto_sanitize(*rank >= 1 && *rank <= WL_MAX_DIMS, false);
            for (size_t i=0; i < WL_MAX_DIMS; ++i) {  /* Read shape */
                uint64_t u64 = wl__sto_read_u64_le(p);
                wl__sto_sanitize(u64>= 1 && u64 <= (uint64_t)INT64_MAX, false);
                (*shape)[i] = (int64_t)u64;
            }
            wl__sto_sanitize(INT64_MAX/(*shape)[1] > (*shape)[0], false); /* Check for shape overflow */
            wl__sto_sanitize(INT64_MAX/(*shape)[2] > (*shape)[0]*(*shape)[1], false);
            wl__sto_sanitize(INT64_MAX/(*shape)[3] > (*shape)[0]*(*shape)[1]*(*shape)[2], false);
        } break;
        default: return false;
    }
    return wl__likely(*p - start == WL__STO_TENSOR_HEADER_SIZE);
}

static bool wl__sto_write_tensor_data(
    uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    wl_dtype_t dtype,
    const void* data,
    int64_t size
) {
    wl__sto_sanitize(size > 0, false);
    wl__sto_sanitize(*p + size <= end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            memcpy(*p, data, size);
            *p += size;
        } break;
        default: return false;
    }
    return wl__likely(*p - start == size);
}

static bool wl__sto_read_tensor_data(
    const uint8_t** p,
    const uint8_t* end,
    uint32_t version,
    wl_dtype_t dtype,
    void* data,
    int64_t size
) {
    wl__sto_sanitize(size > 0, false);
    wl__sto_sanitize(*p + size <= end, false);
    const uint8_t* start = *p;
    switch (version) {
        case 1: {
            memcpy(data, *p, size); /* TODO: endianess conversion */
            *p += size;
        } break;
        default: return false;
    }
    return wl__likely(*p - start == size);
}

static size_t wl__accumulate_data_size(wl_dtype_t dtype, const int64_t (*shape)[WL_MAX_DIMS]) {
    size_t size = wl_dtype_info_of(dtype)->size;
    #pragma GCC unroll 6
    for (size_t i=0; i < WL_MAX_DIMS; ++i) size *= (size_t)wl__max(1, (*shape)[i]);
    return size;
}

static size_t wl__sto_total_size(const wl_tensor_t** tensors, size_t n) {
    size_t total = WL__STO_FILE_HEADER_SIZE;
    for (size_t i=0; i < n; ++i) {
        total += WL__STO_TENSOR_HEADER_SIZE;
        total += wl__accumulate_data_size(tensors[i]->dtype, &tensors[i]->shape);
    }
    wl__assert((total & 3) == 0, "Unaligned storage size: %zu", total);
    return total;
}

static uint8_t* wl__sto_write_buffered(const wl_tensor_t** tensors, size_t n_tensors, size_t* out_size, uint32_t version) {
    if (wl__unlikely(!tensors || !n_tensors || n_tensors > UINT32_MAX || !out_size || !version || version > WL_STORAGE_VERSION)) return NULL;  /* Check input */
    *out_size = wl__sto_total_size(tensors, n_tensors);
    uint8_t* base = (uint8_t*)wl_alloc(NULL, *out_size );     /* Allocate buffer */
    uint8_t* needle = base;
    const uint8_t* end = base + *out_size ;
    if (wl__unlikely(!wl__sto_write_file_header(&needle, end, version, (uint32_t)n_tensors, 0))) goto error;     /* Write file header */
    for (size_t i=0; i < n_tensors; ++i) {   /* Write tensor headers */
        const wl_tensor_t* t = tensors[i];
        wl__assert2(t != NULL);
        if (wl__unlikely(!wl__sto_write_tensor_header(
            &needle,
            end,
            version,
            &t->name,
            t->flags,
            t->dtype,
            t->rank,
            &t->shape
        ))) goto error;
    }
    wl__assert2(needle - base == WL__STO_FILE_HEADER_SIZE + n_tensors*WL__STO_TENSOR_HEADER_SIZE);    /* Check written data size */
    for (size_t i=0; i < n_tensors; ++i) {  /* Write tensor data */
        const wl_tensor_t* t = tensors[i];
        if (wl__unlikely(!wl__sto_write_tensor_data(&needle, end, version, t->dtype, t->buf, wl__tensor_data_size(t)))) goto error;     /* Write data */
    }
    return base;
    error: /* Error handling */
        wl_alloc(base, 0);
        return NULL;
}

WL_EXPORT wl_tensor_t** wl__sto_read_buffered(wl_ctx_t* ctx, const uint8_t* buf, size_t size, size_t* out_n_tensors, uint32_t* out_version) { /* Load stored tensors from buffer. Function is exported for fuzzing test. */
    if (wl__unlikely(!ctx || !buf || !out_n_tensors || !out_version || size <= WL__STO_FILE_HEADER_SIZE + WL__STO_TENSOR_HEADER_SIZE + 1)) return NULL;    /* Check input */
    const uint8_t* needle = buf;
    const uint8_t* end = buf + size;
    uint32_t n_tensors;
    uint32_t ud;
    if (wl__unlikely(!wl__sto_read_file_header(&needle, end, out_version, &n_tensors, &ud))) return NULL;   /* Read file header */
    if (wl__unlikely(!*out_version || *out_version > WL_VERSION)) return NULL;
    if (wl__unlikely(!n_tensors)) return NULL;
    wl_tensor_t** tensors = (wl_tensor_t**)wl_alloc(NULL, n_tensors*sizeof(*tensors));   /* Allocate return tensor array */
    for (size_t i=0; i < n_tensors; ++i) {  /* Read tensor headers */
        char name[WL_MAX_TENSOR_NAME_LEN] = {0};
        wl__tensor_flags_t flags = 0;
        wl_dtype_t dtype = 0;
        int64_t rank = 0;
        int64_t shape[WL_MAX_DIMS] = {0};
        if (wl__unlikely(!wl__sto_read_tensor_header(&needle, end, *out_version, &name, &flags, &dtype, &rank, &shape))) goto error;   /* Read tensor header */
        wl_tensor_t* t = wl__tensor_create(ctx, dtype, shape, rank, NULL, 0);   /* Create placeholder tensor */
        wl_tensor_fmt_name(t, "%s", name);
        t->flags = flags | WL__TFLAG_FROM_FS;
        tensors[i] = t;
    }
    for (size_t i=0; i < n_tensors; ++i) {  /* Read tensor data */
        wl_tensor_t* t = tensors[i];
        size_t data_size = wl__accumulate_data_size(t->dtype, &t->shape);
        wl__assert2(needle + data_size <= end && data_size == wl__tensor_data_size(t));
        if (wl__unlikely(!wl__sto_read_tensor_data(&needle, end, *out_version, t->dtype, t->buf, data_size))) goto error;  /* Read data into tensor's buffer */
    }
    *out_n_tensors = n_tensors;
    return tensors;
    error:
        wl_alloc(tensors, 0);
        return NULL;
}

static bool wl__sto_has_wl_ext(const char* file) { /* Check if file path has WAVELET extension. */
    if (wl__unlikely(!file || strlen(file) < sizeof(WL__STORAGE_EXT))) return false;
    char* dot = strrchr(file, '.');
    return dot && !strcmp(dot, WL__STORAGE_EXT);
}

void wl_tensor_save(const wl_tensor_t* t, const char* file) {
    wl__assert(wl__sto_has_wl_ext(file), "Invalid file extension: %s", file);
    FILE* f = wl__fopen(file, "wb");  /* Open file */
    wl__assert(f, "Failed to open file stream: %s", file);
    uint32_t version = WL_STORAGE_VERSION;
    size_t n_tensors = 1;
    size_t n_bytes = 0;
    uint8_t* ser = wl__sto_write_buffered(&t, n_tensors, &n_bytes, version);   /* Serialize tensor */
    wl__assert(ser && n_bytes, "Failed to serialize tensor to file: %s", file);   /* Check serialization */
    wl__assert(fwrite(ser, 1, n_bytes, f) == n_bytes, "Failed to write %zu bytes to file: %s", n_bytes, file);    /* Write to file */
    wl_alloc(ser, 0);     /* Free buffer */
    fflush(f);
    fclose(f);
    double mem;
    const char* unit;
    wl__humanize_memory_size(n_bytes, &mem, &unit);
    wl__log_info("Saved %zu tensor%s to file: %s, %.03f %s written, storage v.%u", n_tensors, n_tensors > 1 ? "s" : "", file, mem, unit, version);
}

wl_tensor_t* wl_tensor_load(wl_ctx_t* ctx, const char* file) {
    wl__assert(wl__sto_has_wl_ext(file), "Invalid file extension: %s", file);
    FILE* f = wl__fopen(file, "rb");  /* Open file */
    wl__assert(f, "Failed to open file stream: %s", file);
    wl__assert2(fseek(f, 0, SEEK_END) == 0);  /* Seek to end */
    long n_bytes = ftell(f);    /* Get file size */
    wl__assert(n_bytes > WL__STO_FILE_HEADER_SIZE + WL__STO_TENSOR_HEADER_SIZE + 1, "Malformed file size");   /* Check file size */
    wl__assert2(fseek(f, 0, SEEK_SET) == 0); /* Seek to start */
    uint8_t* buf = (uint8_t*)wl_alloc(NULL, n_bytes);  /* Allocate buffer */
    wl__assert(fread(buf, 1, n_bytes, f) == n_bytes, "Failed to read %zu bytes from file: %s", n_bytes, file);    /* Read while file into buffer */
    fclose(f), f = NULL;    /* Close file */
    size_t n_tensors = 0;
    uint32_t version = 0;
    wl_tensor_t** tensors = wl__sto_read_buffered(ctx, buf, n_bytes, &n_tensors, &version);   /* Deserialize tensors */
    wl__assert(version > 0 && version <= WL_VERSION, "Unsupported storage version: %u", version);   /* Check version */
    wl__assert(tensors && n_tensors > 0, "Failed to load tensor from file: %s", file);
    wl_tensor_t* target = *tensors;
    wl_alloc(buf, 0);     /* Free buffer */
    wl_alloc(tensors, 0);     /* Free tensor array */
    double mem;
    const char* unit;
    wl__humanize_memory_size(n_bytes, &mem, &unit);
    wl__log_info("Loaded %zu tensor%s from file: %s, %.03f %s read, storage v.%u", n_tensors, n_tensors > 1 ? "s" : "", file, mem, unit, version);
    return target;
}

wl_tensor_t* wl_tensor_load_image(wl_ctx_t* ctx, const char* file, wl_color_channels_t channels, uint32_t rw, uint32_t rh) {
    uint8_t* (*loader)(const char*, uint32_t(*)[3], wl_color_channels_t) = ctx->image_load_fn;
    void (*load_free)(uint8_t*) = ctx->image_load_free_fn;
    wl__assert(loader && load_free, "Image loader not set");
    uint32_t whc[3] = {0};
    uint8_t* src = (*loader)(file, &whc, channels);
    wl__assert(src, "Failed to load tensor from image: '%s'", file);
    if (rw && rh) { /* Resize requested. */
        float* ori = (*ctx->alloc_fn)(NULL, whc[2]*whc[1]*whc[0]*sizeof(*ori));
        for (int64_t k=0; k < whc[2]; ++k) { /* Convert from interleaved to planar representation. */
            for (int64_t j=0; j < whc[1]; ++j) {
                for (int64_t i=0; i < whc[0]; ++i) {
                    ori[i + whc[0]*j + whc[0]*whc[1]*k] = (float)src[k + whc[2]*i + whc[2]*whc[0]*j] / 255.0f;  /* Normalize pixel values to [0, 1] */
                }
            }
        }
        wl_tensor_t* t = wl_tensor_create_3d(ctx, WL_DTYPE_F32, whc[2], rh, rw);
        float* dst = wl_tensor_data_as_f32(t);
        float* part = (*ctx->alloc_fn)(NULL, whc[2]*whc[1]*rw*sizeof(*part));
        float ws = (float)(whc[0] - 1)/(float)(rw - 1);
        float hs = (float)(whc[1] - 1)/(float)(rh - 1);
        for (uint32_t k = 0; k < whc[2]; ++k){
            for (uint32_t r = 0; r < whc[1]; ++r) {
                for (uint32_t c = 0; c < rw; ++c) {
                    float val = 0;
                    if (c == rw-1 || whc[0] == 1) {
                        val = ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + (whc[0] - 1)];
                    } else {
                        float sx = (float)c*ws;
                        uint32_t ix = (uint32_t)sx;
                        float dx = sx - (float)ix;
                        val = (1-dx) * (ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + ix]) + dx*(ori[k*(whc[0])*(whc[1]) + r*(whc[0]) + (ix + 1)]);
                    }
                    part[k*rw*(whc[1]) + r*rw + c] = val;
                }
            }
        }
        for (uint32_t k = 0; k < whc[2]; ++k) {
            for (uint32_t r = 0; r < rh; ++r) {
                float sy = (float)r*hs;
                uint32_t iy = (uint32_t)sy;
                float dy = sy - (float)iy;
                for (uint32_t c = 0; c < rw; ++c) {
                    float val = (1-dy)*(part[k*rw*whc[1] + iy*rw + c]);
                    dst[k*rw*rh + r*rw + c] = val;
                }
                if (r == rh-1 || whc[1] == 1) continue;
                for (uint32_t c = 0; c < rw; ++c) {
                    float val = dy*(part[k*rw*(whc[1]) + (iy + 1)*rw + c]);
                    dst[k*rw*rh + r*rw + c] += val;
                }
            }
        }
        (*ctx->alloc_fn)(ori, 0);
        (*ctx->alloc_fn)(part, 0);
        wl__assert(rw*rh*whc[2] == wl__tensor_num_elements(t), "Buffer size mismatch: %zu != %zu", rw*rh*whc[2], (size_t)wl__tensor_num_elements(t));
        (*load_free)(src);
        wl__log_info("Loaded and resized tensor from image: %s, %u x %u x %u", file, rw, rh, whc[2]);
        return t;
    } else {
        wl_tensor_t* t = wl_tensor_create_3d(ctx, WL_DTYPE_F32, whc[2], whc[1], whc[0]);
        t->flags |= WL__TFLAG_FROM_FS | WL__TFLAG_IMAGE;
        float* dst = wl_tensor_data_as_f32(t);
        for (int64_t k = 0; k < whc[2]; ++k) { /* Convert from interleaved to planar representation. */
            for (int64_t j = 0; j < whc[1]; ++j) {
                for (int64_t i = 0; i < whc[0]; ++i) {
                    dst[i + whc[0]*j + whc[0]*whc[1]*k] = (float)src[k + whc[2]*i + whc[2]*whc[0]*j] / 255.0f;  /* Normalize pixel values to [0, 1] */
                }
            }
        }
        wl__assert(whc[0]*whc[1]*whc[2] == wl__tensor_num_elements(t), "Buffer size mismatch: %zu != %zu", whc[0]*whc[1]*whc[2], (size_t)wl__tensor_num_elements(t));
        (*load_free)(src);
        wl__log_info("Loaded tensor from image: %s, %u x %u x %u", file, whc[0], whc[1], whc[2]);
        return t;
    }
}

void wl_tensor_save_image(const wl_tensor_t* t, const char* file) {
    bool (*saver)(const char*, const uint8_t*, const uint32_t(*)[3]) = t->ctx->image_save_fn;
    wl__assert(saver, "Image saver not set");
    int64_t rank = wl_tensor_rank(t);
    wl__assert(rank == 3, "Tensor rank must be 3, but is: %" PRIi64, (size_t)rank);
    int64_t w = wl_tensor_image_width(t);
    int64_t h = wl_tensor_image_height(t);
    int64_t c = wl_tensor_image_channels(t);
    wl__assert(c == 1 || c == 3 || c == 4, "Invalid number of channels: %zu", (size_t)c);
    wl__assert(w*h*c == wl__tensor_num_elements(t), "Buffer size mismatch: %zu != %zu", w*h*c, (size_t)wl__tensor_num_elements(t));
    uint8_t* dst = (*t->ctx->alloc_fn)(NULL, w*h*c); /* Allocate memory for image data */
    const float* src = wl_tensor_data_as_f32(t);
    for (int64_t k = 0; k < c; ++k) /* Convert from planar to interleaved format. */
        for (int64_t i = 0; i < w*h; ++i)
            dst[i*c + k] = (uint8_t)(src[i + k*w*h]*255.0f);
    const uint32_t whc[3] = {(uint32_t)w,(uint32_t)h,(uint32_t)c};
    wl__assert((*saver)(file, dst, &whc), "Failed to save tensor to image: %s", file);
    (*t->ctx->alloc_fn)(dst, 0); /* Free image data */
    wl__log_info("Saved tensor to image: %s, width: %d, height: %d, channels: %d", file, (int)w, (int)h, (int)c);
}

/* Rescale factors to push the exponent of a number towards zero. */
#define rescale_exponents(P, N) \
  P(308), P(289), P(270), P(250), P(231), P(212), P(193), P(173), P(154), \
  P(135), P(115), P(96), P(77), P(58), P(38), P(0), P(0), P(0), N(39), N(58), \
  N(77), N(96), N(116), N(135), N(154), N(174), N(193), N(212), N(231), \
  N(251), N(270), N(289)
#define one_e_p(X) 1e+0 ## X
#define one_e_n(X) 1e-0 ## X
static const int16_t wl__rescale_e[] = { rescale_exponents(-, +) };
static const double wl__rescale_n[] = { rescale_exponents(one_e_p, one_e_n) };
#undef one_e_n
#undef one_e_p

/*
** For p in range -70 through 57, this table encodes pairs (m, e) such that
** 4*2^p <= (uint8_t)m*10^e, and is the smallest value for which this holds.
*/
static const int8_t wl__four_ulp_m_e[] = {
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
static const uint32_t wl__ndigits_dec_threshold[] = {
    0, 9U, 99U, 999U, 9999U, 99999U, 999999U,
    9999999U, 99999999U, 999999999U, 0xffffffffU
};

/* Compute the number of digits in the decimal representation of x. */
static size_t wl__ndigits_dec(uint32_t x) {
    size_t t = ((wl__fls(x | 1) * 77) >> 8) + 1; /* 2^8/77 is roughly log2(10) */
    return t + (x > wl__ndigits_dec_threshold[t]);
}

#define wint_r(x, sh, sc) { uint32_t d = (x*(((1<<sh)+sc-1)/sc))>>sh; x -= d*sc; *p++ = (char)('0'+d); }
static char* wl__wuint9(char* p, uint32_t u) {
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
static char* wl__wint(char* p, int32_t k) {
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
#define WL__ND_MUL2K_MAX_SHIFT 29
#define WL__ND_MUL2K_DIV1E9(val) ((uint32_t)((val) / 1000000000))

/* Multiply nd by 2^k and add carry_in (ndlo is assumed to be zero). */
static uint32_t nd_mul2k(uint32_t* nd, uint32_t ndhi, uint32_t k, uint32_t carry_in, wl__format_flags sf) {
    uint32_t i, ndlo = 0, start = 1;
    /* Performance hacks. */
    if (k > WL__ND_MUL2K_MAX_SHIFT*2 && WL__FMT_FP(sf) != WL__FMT_FP(WL__FMT_T_FP_F)) {
        start = ndhi - (WL__FMT_PREC(sf) + 17) / 8;
    }
    /* Real logic. */
    while (k >= WL__ND_MUL2K_MAX_SHIFT) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << WL__ND_MUL2K_MAX_SHIFT) | carry_in;
            carry_in = WL__ND_MUL2K_DIV1E9(val);
            nd[i] = (uint32_t)val - carry_in * 1000000000;
        }
        if (carry_in) {
            nd[++ndhi] = carry_in; carry_in = 0;
            if (start++ == ndlo) ++ndlo;
        }
        k -= WL__ND_MUL2K_MAX_SHIFT;
    }
    if (k) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << k) | carry_in;
            carry_in = WL__ND_MUL2K_DIV1E9(val);
            nd[i] = (uint32_t)val - carry_in * 1000000000;
        }
        if (carry_in) nd[++ndhi] = carry_in;
    }
    return ndhi;
}

/* Divide nd by 2^k (ndlo is assumed to be zero). */
static uint32_t nd_div2k(uint32_t* nd, uint32_t ndhi, uint32_t k, wl__format_flags sf) {
    uint32_t ndlo = 0, stop1 = ~0, stop2 = ~0;
    /* Performance hacks. */
    if (!ndhi) {
        if (!nd[0]) {
            return 0;
        } else {
            uint32_t s = wl__ffs(nd[0]);
            if (s >= k) { nd[0] >>= k; return 0; }
            nd[0] >>= s; k -= s;
        }
    }
    if (k > 18) {
        if (WL__FMT_FP(sf) == WL__FMT_FP(WL__FMT_T_FP_F)) {
            stop1 = 63 - (int32_t)WL__FMT_PREC(sf) / 9;
        } else {
            int32_t floorlog2 = ndhi * 29 + wl__fls(nd[ndhi]) - k;
            int32_t floorlog10 = (int32_t)(floorlog2 * 0.30102999566398114);
            stop1 = 62 + (floorlog10 - (int32_t)WL__FMT_PREC(sf)) / 9;
            stop2 = 61 + ndhi - (int32_t)WL__FMT_PREC(sf) / 8;
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
        carry = m * (wl__ndigits_dec_threshold[e - (int32_t)i*9] + 1);
    } else {
        int32_t f = (e-8)/9;
        i = (uint32_t)(64 + f);
        carry = m * (wl__ndigits_dec_threshold[e - f*9] + 1);
    }
    for (;;) {
        uint32_t val = nd[i] + carry;
        if (wl__unlikely(val >= 1000000000)) {
            val -= 1000000000;
            nd[i] = val;
            if (wl__unlikely(i == ndhi)) {
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
        if (wl__unlikely(nd[ndhi] != *ref)) return 0;
        prec -= hilen; ref--; ndhi = (ndhi - 1) & 0x3f;
        if (prec >= 9) {
            if (wl__unlikely(nd[ndhi] != *ref)) return 0;
            prec -= 9; ref--; ndhi = (ndhi - 1) & 0x3f;
        }
    } else {
        prec -= hilen - 9;
    }
    wl__assert(prec < 9, "bad precision %d", prec);
    wl__wuint9(nd9, nd[ndhi]);
    wl__wuint9(ref9, *ref);
    return !memcmp(nd9, ref9, prec) && (nd9[prec] < '5') == (ref9[prec] < '5');
}

/* Format f64 according to format flags. */
static char* wl__fmt_f64(wl__format_flags sf, double n, char* p) {
    size_t width = WL__FMT_WIDTH(sf), prec = WL__FMT_PREC(sf), len;
    union {
        uint64_t u64;
        double n;
        struct { /* TODO: make endian aware */
            uint32_t lo, hi;
        } u32;
    } t = {.n = n};
    if (wl__unlikely((t.u32.hi << 1) >= 0xffe00000)) {
        /* Handle non-finite values uniformly for %a, %e, %f, %g. */
        int32_t prefix = 0, ch = (sf & WL__FMT_F_UPPER) ? 0x202020 : 0;
        if (((t.u32.hi & 0x000fffff) | t.u32.lo) != 0) {
            ch ^= ('n' << 16) | ('a' << 8) | 'n';
            if ((sf & WL__FMT_F_SPACE)) prefix = ' ';
        } else {
            ch ^= ('i' << 16) | ('n' << 8) | 'f';
            if ((t.u32.hi & 0x80000000)) prefix = '-';
            else if ((sf & WL__FMT_F_PLUS)) prefix = '+';
            else if ((sf & WL__FMT_F_SPACE)) prefix = ' ';
        }
        len = 3 + (prefix != 0);
        if (!(sf & WL__FMT_F_LEFT)) while (width-- > len) *p++ = ' ';
        if (prefix) *p++ = prefix;
        *p++ = (char)(ch >> 16); *p++ = (char)(ch >> 8); *p++ = (char)ch;
    } else if (WL__FMT_FP(sf) == WL__FMT_FP(WL__FMT_T_FP_A)) {
        /* %a */
        const char* hexdig = (sf & WL__FMT_F_UPPER) ? "0123456789ABCDEFPX" : "0123456789abcdefpx";
        int32_t e = (t.u32.hi >> 20) & 0x7ff;
        char prefix = 0, eprefix = '+';
        if (t.u32.hi & 0x80000000) prefix = '-';
        else if ((sf & WL__FMT_F_PLUS)) prefix = '+';
        else if ((sf & WL__FMT_F_SPACE)) prefix = ' ';
        t.u32.hi &= 0xfffff;
        if (e) {
            t.u32.hi |= 0x100000;
            e -= 1023;
        } else if (t.u32.lo | t.u32.hi) {
            /* Non-zero denormal - normalise it. */
            uint32_t shift = t.u32.hi ? 20-wl__fls(t.u32.hi) : 52-wl__fls(t.u32.lo);
            e = -1022 - shift;
            t.u64 <<= shift;
        }
        /* abs(n) == t.u64 * 2^(e - 52) */
        /* If n != 0, bit 52 of t.u64 is set, and is the highest set bit. */
        if ((int32_t)prec < 0) {
            /* Default precision: use smallest precision giving exact result. */
            prec = t.u32.lo ? 13-wl__ffs(t.u32.lo)/4 : 5-wl__ffs(t.u32.hi|0x100000)/4;
        } else if (prec < 13) {
            /* Precision is sufficiently low as to maybe require rounding. */
            t.u64 += (((uint64_t)1) << (51 - prec*4));
        }
        if (e < 0) {
            eprefix = '-';
            e = -e;
        }
        len = 5 + wl__ndigits_dec((uint32_t)e) + prec + (prefix != 0)
              + ((prec | (sf & WL__FMT_F_ALT)) != 0);
        if (!(sf & (WL__FMT_F_LEFT | WL__FMT_F_ZERO))) {
            while (width-- > len) *p++ = ' ';
        }
        if (prefix) *p++ = prefix;
        *p++ = '0';
        *p++ = hexdig[17]; /* x or X */
        if ((sf & (WL__FMT_F_LEFT | WL__FMT_F_ZERO)) == WL__FMT_F_ZERO) {
            while (width-- > len) *p++ = '0';
        }
        *p++ = '0' + (t.u32.hi >> 20); /* Usually '1', sometimes '0' or '2'. */
        if ((prec | (sf & WL__FMT_F_ALT))) {
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
        p = wl__wint(p, e);
    } else {
        /* %e or %f or %g - begin by converting n to "nd" format. */
        uint32_t nd[64];
        uint32_t ndhi = 0, ndlo, i;
        int32_t e = (int32_t)(t.u32.hi >> 20) & 0x7ff, ndebias = 0;
        char prefix = 0, *q;
        if (t.u32.hi & 0x80000000) prefix = '-';
        else if ((sf & WL__FMT_F_PLUS)) prefix = '+';
        else if ((sf & WL__FMT_F_SPACE)) prefix = ' ';
        prec += ((int32_t)prec >> 31) & 7; /* Default precision is 6. */
        if (WL__FMT_FP(sf) == WL__FMT_FP(WL__FMT_T_FP_G)) {
            /* %g - decrement precision if non-zero (to make it like %e). */
            prec--;
            prec ^= (uint32_t)((int32_t)prec >> 31);
        }
        if ((sf & WL__FMT_T_FP_E) && prec < 14 && n != 0) {
            /* Precision is sufficiently low that rescaling will probably work. */
            if ((ndebias = wl__rescale_e[e >> 6])) {
                t.n = n * wl__rescale_n[e >> 6];
                if (wl__unlikely(!e)) t.n *= 1e10, ndebias -= 10;
                t.u64 -= 2; /* Convert 2ulp below (later we convert 2ulp above). */
                nd[0] = 0x100000 | (t.u32.hi & 0xfffff);
                e = ((int32_t)(t.u32.hi >> 20) & 0x7ff) - 1075 - (WL__ND_MUL2K_MAX_SHIFT < 29);
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
            e -= 32 + (WL__ND_MUL2K_MAX_SHIFT < 29); load_t_lo:
#if WL__ND_MUL2K_MAX_SHIFT >= 29
            nd[0] = (nd[0]<<3) | (t.u32.lo>>29);
            ndhi = nd_mul2k(nd, ndhi, 29, t.u32.lo & 0x1fffffff, sf);
#elif WL__ND_MUL2K_MAX_SHIFT >= 11
            ndhi = nd_mul2k(nd, ndhi, 11, t.u32.lo>>21, sf);
            ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo>>10) & 0x7ff, sf);
            ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo<<1) & 0x7ff, sf);
#else
#error "WL__ND_MUL2K_MAX_SHIFT not big enough"
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
        if ((sf & WL__FMT_T_FP_E)) {
            /* %e or %g - assume %e and start by calculating nd's exponent (nde). */
            char eprefix = '+';
            int32_t nde = -1;
            size_t hilen;
            if (ndlo && !nd[ndhi]) {
                ndhi = 64; do {} while (!nd[--ndhi]);
                nde -= 64 * 9;
            }
            hilen = wl__ndigits_dec(nd[ndhi]);
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
                int32_t eidx = e + 70 + (WL__ND_MUL2K_MAX_SHIFT < 29)
                               + (t.u32.lo >= 0xfffffffe && !(~t.u32.hi << 12));
                const int8_t *m_e = wl__four_ulp_m_e + eidx * 2;
                wl__assert(0 <= eidx && eidx < 128, "bad eidx %d", eidx);
                nd[33] = nd[ndhi];
                nd[32] = nd[(ndhi - 1) & 0x3f];
                nd[31] = nd[(ndhi - 2) & 0x3f];
                nd_add_m10e(nd, ndhi, (uint8_t)*m_e, m_e[1]);
                if (wl__unlikely(!nd_similar(nd, ndhi, nd + 33, hilen, prec + 1))) {
                    goto rescale_failed;
                }
            }
            if ((int32_t)(prec - nde) < (0x3f & -(int32_t)ndlo) * 9) {
                /* Precision is sufficiently low as to maybe require rounding. */
                ndhi = nd_add_m10e(nd, ndhi, 5, (int32_t)nde - prec - 1);
                nde += (hilen != wl__ndigits_dec(nd[ndhi]));
            }
            nde += ndebias;
            if ((sf & WL__FMT_T_FP_F)) {
                /* %g */
                if ((int32_t)prec >= nde && nde >= -4) {
                    if (nde < 0) ndhi = 0;
                    prec -= nde;
                    goto g_format_like_f;
                } else if (!(sf & WL__FMT_F_ALT) && prec && width > 5) {
                    /* Decrease precision in order to strip trailing zeroes. */
                    char tail[9];
                    uint32_t maxprec = hilen - 1 + ((ndhi - ndlo) & 0x3f) * 9;
                    if (prec >= maxprec) prec = maxprec;
                    else ndlo = (ndhi - (((int32_t)(prec - hilen) + 9) / 9)) & 0x3f;
                    i = prec - hilen - (((ndhi - ndlo) & 0x3f) * 9) + 10;
                    wl__wuint9(tail, nd[ndlo]);
                    while (prec && tail[--i] == '0') {
                        prec--;
                        if (!i) {
                            if (ndlo == ndhi) { prec = 0; break; }
                            ndlo = (ndlo + 1) & 0x3f;
                            wl__wuint9(tail, nd[ndlo]);
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
            len = 3 + prec + (prefix != 0) + wl__ndigits_dec((uint32_t)nde) + (nde < 10)
                  + ((prec | (sf & WL__FMT_F_ALT)) != 0);
            if (!(sf & (WL__FMT_F_LEFT | WL__FMT_F_ZERO))) {
                while (width-- > len) *p++ = ' ';
            }
            if (prefix) *p++ = prefix;
            if ((sf & (WL__FMT_F_LEFT | WL__FMT_F_ZERO)) == WL__FMT_F_ZERO) {
                while (width-- > len) *p++ = '0';
            }
            q = wl__wint(p + 1, nd[ndhi]);
            p[0] = p[1]; /* Put leading digit in the correct place. */
            if ((prec | (sf & WL__FMT_F_ALT))) {
                /* Emit fractional part. */
                p[1] = '.'; p += 2;
                prec -= (size_t)(q - p); p = q; /* Account for digits already emitted. */
                /* Then emit chunks of 9 digits (this may emit 8 digits too many). */
                for (i = ndhi; (int32_t)prec > 0 && i != ndlo; prec -= 9) {
                    i = (i - 1) & 0x3f;
                    p = wl__wuint9(p, nd[i]);
                }
                if ((sf & WL__FMT_T_FP_F) && !(sf & WL__FMT_F_ALT)) {
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
            *p++ = (sf & WL__FMT_F_UPPER) ? 'E' : 'e';
            *p++ = eprefix; /* + or - */
            if (nde < 10) *p++ = '0'; /* Always at least two digits of exponent. */
            p = wl__wint(p, nde);
        } else {
            /* %f (or, shortly, %g in %f style) */
            if (prec < (size_t)(0x3f & -(int32_t)ndlo) * 9) {
                /* Precision is sufficiently low as to maybe require rounding. */
                ndhi = nd_add_m10e(nd, ndhi, 5, 0 - prec - 1);
            }
            g_format_like_f:
            if ((sf & WL__FMT_T_FP_E) && !(sf & WL__FMT_F_ALT) && prec && width) {
                /* Decrease precision in order to strip trailing zeroes. */
                if (ndlo) {
                    /* nd has a fractional part; we need to look at its digits. */
                    char tail[9];
                    uint32_t maxprec = (64 - ndlo) * 9;
                    if (prec >= maxprec) prec = maxprec;
                    else ndlo = 64 - (prec + 8) / 9;
                    i = prec - ((63 - ndlo) * 9);
                    wl__wuint9(tail, nd[ndlo]);
                    while (prec && tail[--i] == '0') {
                        prec--;
                        if (!i) {
                            if (ndlo == 63) { prec = 0; break; }
                            wl__wuint9(tail, nd[++ndlo]);
                            i = 9;
                        }
                    }
                } else {
                    /* nd has no fractional part, so precision goes straight to zero. */
                    prec = 0;
                }
            }
            len = ndhi * 9 + wl__ndigits_dec(nd[ndhi]) + prec + (prefix != 0)
                  + ((prec | (sf & WL__FMT_F_ALT)) != 0);
            if (!(sf & (WL__FMT_F_LEFT | WL__FMT_F_ZERO))) {
                while (width-- > len) *p++ = ' ';
            }
            if (prefix) *p++ = prefix;
            if ((sf & (WL__FMT_F_LEFT | WL__FMT_F_ZERO)) == WL__FMT_F_ZERO) {
                while (width-- > len) *p++ = '0';
            }
            /* Emit integer part. */
            p = wl__wint(p, nd[ndhi]);
            i = ndhi;
            while (i) p = wl__wuint9(p, nd[--i]);
            if ((prec | (sf & WL__FMT_F_ALT))) {
                /* Emit fractional part. */
                *p++ = '.';
                /* Emit chunks of 9 digits (this may emit 8 digits too many). */
                while ((int32_t)prec > 0 && i != ndlo) {
                    i = (i - 1) & 0x3f;
                    p = wl__wuint9(p, nd[i]);
                    prec -= 9;
                }
                if ((sf & WL__FMT_T_FP_E) && !(sf & WL__FMT_F_ALT)) {
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
    if ((sf & WL__FMT_F_LEFT)) while (width-- > len) *p++ = ' ';
    return p;
}

#if WL_ENABLE_IMAGE_SUPPORT
    static uint8_t* wl_default_image_load_impl(const char* file, uint32_t(*whc)[3], wl_color_channels_t channels) {
        wl__assert2(file && *file && whc);
        int w, h, c, dc;
        switch (channels) {
            default: dc = STBI_default; break;
            case WL_COLOR_CHANNELS_GRAY: dc = STBI_grey; break;
            case WL_COLOR_CHANNELS_GRAY_A: dc = STBI_grey_alpha; break;
            case WL_COLOR_CHANNELS_RGB: dc = STBI_rgb; break;
            case WL_COLOR_CHANNELS_RGBA: dc = STBI_rgb_alpha; break;
        }
        uint8_t* buf = stbi_load(file, &w, &h, &c, dc);
        if (wl__unlikely(!buf || !w || !h || !c || (c != 1 && c != 3 && c != 4))) return NULL;
        (*whc)[0] = (uint32_t)w;
        (*whc)[1] = (uint32_t)h;
        (*whc)[2] = (uint32_t)c;
        return buf;
    }

    static void wl_default_image_load_free_fn_impl(uint8_t* p) {
        stbi_image_free(p);
    }

    static bool wl_default_image_save_impl(const char* file, const uint8_t* buf, const uint32_t(*whc)[3]) {
        wl__assert2(file && *file && buf && whc);
        return stbi_write_jpg(file, (int)(*whc)[0], (int)(*whc)[1], (int)(*whc)[2], buf, 100) != 0;
    }
#endif
