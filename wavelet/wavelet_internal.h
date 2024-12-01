/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#ifndef WAVELET_INTERNAL_H
#define WAVELET_INTERNAL_H

#include "wavelet.h"

#ifdef _MSC_VER
#include <intrin.h>
#endif
#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#ifndef _MSC_VER
#include <cpuid.h>
#endif
#endif

#if defined(__GLIBC__) || defined(__GNU_LIBRARY__) || defined(__ANDROID__)
#include <endian.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <machine/endian.h>
#elif defined(BSD) || defined(_SYSTYPE_BSD)
#if defined(__OpenBSD__)
#include <machine/endian.h>
#else
#include <sys/endian.h>
#endif
#endif

#define WL__GELU_COEFF 0.044715f
#define WL__GRA_FWD WL_GRAPH_EVAL_ORDER_FORWARD
#define WL__GRA_BWD WL_GRAPH_EVAL_ORDER_REVERSE
#define WL__GRA_LEN 2

#define WL__MAX_CPUS 8192
#define WL__MAX_NUMA_NODES 64
#define WL__STORAGE_EXT ".wavelet"
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
#define __alignof__ __alignof
#endif
wl_static_assert(sizeof(0u) == 4);
wl_static_assert(sizeof(0ull) == 8);

extern WL__NORET WL__COLDPROC WL_EXPORT void wl__panic(const char* msg, ...);
extern WL_EXPORT bool wl__log_enabled;
extern WL_EXPORT void* (*wl__alloc)(void* blk, size_t size);

#define wl__swap(T, a, b) do { T tmp = (a); (a) = (b); (b) = tmp; } while (0)
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
#define wl__log_info(msg, ...) do { if (wl__unlikely(wl__log_enabled)) fprintf(stdout,   WL__CC_CYAN "[WAVELET] " WL__CC_RESET WL__SRC_NAME " " msg "\n", ## __VA_ARGS__); } while (0)
#define wl__log_warn(msg, ...) do { if (wl__unlikely(wl__log_enabled)) fprintf(stderr,  WL__CC_CYAN "[WAVELET] " WL__CC_RESET WL__SRC_NAME " " WL__CC_YELLOW msg WL__CC_RESET "\n", ## __VA_ARGS__); } while (0)
#define wl__log_error(msg, ...) do { if (wl__unlikely(wl__log_enabled)) fprintf(stderr,  WL__CC_CYAN "[WAVELET] " WL__CC_RESET WL__SRC_NAME " " WL__CC_RED msg WL__CC_RESET "\n", ## __VA_ARGS__); } while (0)

#define wl__assert(expr, msg, ...) \
    if (wl__unlikely(!(expr))) { \
        wl__panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
    }
#define wl__assert2(expr) wl__assert(expr, "")

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

/* Increment pointer or size with correct type alignment. */
static WL__AINLINE void* wl__pincr(void** p, size_t sz, size_t align) {
    void* pp = (void*)(((uintptr_t)*p+align-1)&-align);
    *p = (void*)((uint8_t*)pp+sz);
    return pp;
}

/* Abstract interface to any compute backend device (CPU, GPU, TPU etc..) */
typedef struct wl__compute_device_t {
    char name[128]; /* Device name. */
    void (*exec_forward)(struct wl__compute_device_t* dvc, wl_tensor_t* root); /* Execute a computation graph forward. */
    void (*exec_backward)(struct wl__compute_device_t* dvc, wl_tensor_t* root); /* Execute a computation graph backwards. */
    void* impl; /* Device specific implementation, if applicable. */
} wl__compute_device_t;

/* Profiling performance monitor per op. */
typedef struct wl__perf_mon_t {
    uint64_t elapsed_ns;
    uint64_t elapsed_ns_acc;
    uint64_t n_execs;
} wl__perf_mon_t;

/* Performance monitor for profiler session. */
typedef struct wl__op_perf_info_t {
    uint64_t elapsed_ns_acc;
    uint64_t n_execs;
} wl__op_perf_info_t;

/*
** Context contains all isolated state and data.
** Lifetimes of tensors and compute graphs are bound to the context - the context is the owner.
** Context itself is not thread-safe, use a thread-local context or synchronize access. (Multiple contexts can be used.)
*/
struct wl_ctx_t {
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
    size_t tensors_created;
    size_t tensors_alloced;
    wl_exec_mode_t exec_mode;
    bool profiler_enabled;
    wl__op_perf_info_t op_perf_mons_total[WL_OP__COUNT];
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
    wl_prng_algorithm_t prng_algorithm;     /* PRNG algorithm. */
    uintptr_t host_thread_id;               /* Host thread ID. */
    void (**sh_hooks)(wl_ctx_t*);           /* Shutdown hooks are invoked when context is destroyed. */
    size_t sh_len;                          /* Number of shutdown hooks. */
    size_t sh_cap;                          /* Maximum number of shutdown hooks. */
    wl_compute_device_type_t device_type; /* Active compute device. */
    wl__compute_device_t* device;   /* Active compute device. */
    uint8_t* (*image_load_fn)(const char*, uint32_t(*)[3], wl_color_channels_t);
    void (*image_load_free_fn)(uint8_t*);
    bool (*image_save_fn)(const char*, const uint8_t*, const uint32_t(*)[3]);
    void* ud; /* User data. */
};

typedef enum wl__tensor_flags_t {
    WL__TFLAG_NONE = 0,
    WL__TFLAG_VIEW = 1<<0,          /* Tensor is a view. */
    WL__TFLAG_OP_INPUT = 1<<1,      /* Tensor is an operation input. */
    WL__TFLAG_OP_OUTPUT = 1<<2,     /* Tensor is an operation output. */
    WL__TFLAG_EXEC_EAGER = 1<<3,    /* Tensor is executed eagerly. */
    WL__TFLAG_IMAGE = 1<<4,         /* Tensor was loaded from an image. */
    WL__TFLAG_FROM_FS = 1<<5,       /* Tensor was loaded from the file system. Also true for WL__TFLAG_IMAGE. */
    WL__TFLAG_RECORD_PERF = 1<<6,   /* Record performance data. */
    WL__FLAG_GRA = 1<<7,            /* Tensor is a gradient. */
} wl__tensor_flags_t;
wl_static_assert(WL__TFLAG_FROM_FS <= 0xff); /* Must fit info 8-bits. */

/*
** Tensor with up to 6 Dimensions.
*/
struct wl_tensor_t {
    wl_ctx_t* ctx;                                  /* Host context. */
    int64_t rank;                                   /* Number of active dimensions. [1, MAX_DIMS] */
    int64_t shape[WL_MAX_DIMS];                     /* Shape of the tensor. */
    int64_t strides[WL_MAX_DIMS];                   /* Strides of the tensor. We store the strides in element counts and NOT in bytes. */
    wl_dtype_t dtype;                               /* Data type of the tensor. */
    void* buf;                                      /* Data buffer. */
    int64_t num_elems;                              /* Number of elements in the tensor. */
    wl__tensor_flags_t flags;                       /* Tensor flags. */
    wl_op_t op;                                     /* Opcode for operators. */
    wl_tensor_t* op_inputs[WL_MAX_INPUT_TENSORS];   /* Input tensors for operators. */
    wl_op_param_t op_params[WL_MAX_OP_PARAMS];      /* Operator parameters. */
    wl_tensor_t* view;                              /* View tensor. */
    wl_tensor_t* grad;                              /* ∇f - Gradient tensor. */
    size_t view_offs;                               /* Offset in view tensor. */
    wl__perf_mon_t pmon;                            /* Performance monitor. */
    char name[WL_MAX_TENSOR_NAME_LEN];              /* Tensor debug name. */
    void* ud;                                       /* User data. */
};

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

#endif
