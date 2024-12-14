/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#ifndef WAVELET_INTERNAL_H
#define WAVELET_INTERNAL_H

#include "wavelet.h"

#ifdef _MSC_VER
#include <intrin.h>
#else
#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
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

#ifdef __cplusplus
extern "C" {
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

typedef int32_t wl__atomic_t;       /* Atomic integer type */
typedef enum wl__mo_t {             /* Atomic memory order */
    WL__MO_RELAXED = __ATOMIC_RELAXED,
    WL__MO_CONSUME = __ATOMIC_CONSUME,
    WL__MO_ACQUIRE = __ATOMIC_ACQUIRE,
    WL__MO_RELEASE = __ATOMIC_RELEASE,
    WL__MO_ACQ_REL = __ATOMIC_ACQ_REL,
    WL__MO_SEQ_CST = __ATOMIC_SEQ_CST
} wl__mo_t;

static WL__AINLINE void wl__atomic_store(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    __atomic_store_n(o, x, order);
}
static WL__AINLINE wl__atomic_t wl__atomic_load(volatile wl__atomic_t* o, wl__mo_t order) {
    return __atomic_load_n(o, order);
}
static WL__AINLINE wl__atomic_t wl__atomic_fetch_add(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    return __atomic_fetch_add(o, x, order);
}
static WL__AINLINE wl__atomic_t wl__atomic_fetch_sub(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    return __atomic_fetch_sub(o, x, order);
}
static WL__AINLINE wl__atomic_t wl__atomic_fetch_and(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    return __atomic_fetch_and(o, x, order);
}
static WL__AINLINE wl__atomic_t wl__atomic_fetch_or(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    return __atomic_fetch_or(o, x, order);
}
static WL__AINLINE wl__atomic_t wl__atomic_fetch_xor(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    return __atomic_fetch_xor(o, x, order);
}
static WL__AINLINE wl__atomic_t wl__atomic_exchange(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    return __atomic_exchange_n(o, x, order);
}
static WL__AINLINE bool wl__atomic_compare_exchange_weak(volatile wl__atomic_t* o, wl__atomic_t *exp, wl__atomic_t *des, wl__mo_t order_succ, wl__mo_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, true, order_succ, order_fail);
}
static WL__AINLINE bool wl__atomic_compare_exchange_strong(volatile wl__atomic_t* o, wl__atomic_t *exp, wl__atomic_t *des, wl__mo_t order_succ, wl__mo_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, false, order_succ, order_fail);
}

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
static WL__AINLINE uint32_t wl__ffs(const uint32_t x) {
    unsigned long r; _BitScanForward(&r, x); return (uint32_t)r;
}
static WL__AINLINE uint32_t wl__fls(const uint32_t x) {
    unsigned long r; _BitScanReverse(&r, x); return (uint32_t)r;
}
static WL__AINLINE uint32_t wl__ffs64(const uint64_t x) {
  unsigned long r; _BitScanForward64(&r, x); return (uint32_t)r;
}
static WL__AINLINE uint32_t wl__fls64(const uint64_t x) {
  unsigned long r; _BitScanReverse64(&r, x); return (uint32_t)r;
}
#define __alignof__ __alignof

typedef long wl__atomic_t;       /* Atomic integer type */
typedef enum wl__mo_t {             /* Atomic memory order */
    WL__MO_RELAXED,
    WL__MO_CONSUME,
    WL__MO_ACQUIRE,
    WL__MO_RELEASE,
    WL__MO_ACQ_REL,
    WL__MO_SEQ_CST
} wl__mo_t;

static WL__AINLINE void wl__atomic_store(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    (void)order; _InterlockedExchange(o, x);
}
static WL__AINLINE wl__atomic_t wl__atomic_load(volatile wl__atomic_t* o, wl__mo_t order) {
    (void)order;
    wl__atomic_t r;
    _InterlockedExchange(&r, *o);
    return r;
}
static WL__AINLINE wl__atomic_t wl__atomic_fetch_add(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    (void)order;
    return _InterlockedExchangeAdd(o, x);
}
static WL__AINLINE wl__atomic_t wl__atomic_fetch_sub(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    (void)order;
    return _InterlockedExchangeAdd(o, -x);
}
static WL__AINLINE wl__atomic_t wl__atomic_fetch_and(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    (void)order;
    return _InterlockedAnd(o, x);
}
static WL__AINLINE wl__atomic_t wl__atomic_fetch_or(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    (void)order;
    return _InterlockedOr(o, x);
}
static WL__AINLINE wl__atomic_t wl__atomic_fetch_xor(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    (void)order;
    return _InterlockedXor(o, x);
}
static WL__AINLINE wl__atomic_t wl__atomic_exchange(volatile wl__atomic_t* o, wl__atomic_t x, wl__mo_t order) {
    (void)order;
    return _InterlockedExchange(o, x);
}
static WL__AINLINE bool wl__atomic_compare_exchange_weak(volatile wl__atomic_t* o, wl__atomic_t *exp, wl__atomic_t *des, wl__mo_t order_succ, wl__mo_t order_fail) {
    (void)order_succ; (void)order_fail;
    return _InterlockedCompareExchange(o, *des, *exp) == *exp;
}
static WL__AINLINE bool wl__atomic_compare_exchange_strong(volatile wl__atomic_t* o, wl__atomic_t *exp, wl__atomic_t *des, wl__mo_t order_succ, wl__mo_t order_fail) {
    (void)order_succ; (void)order_fail;
    return _InterlockedCompareExchange(o, *des, *exp) == *exp;
}

#endif

wl_static_assert(sizeof(0u) == 4);
wl_static_assert(sizeof(0ull) == 8);

#ifdef __BYTE_ORDER
#if defined(__BIG_ENDIAN) && (__BYTE_ORDER == __BIG_ENDIAN)
#define WL__BE
#elif defined(__LITTLE_ENDIAN) && (__BYTE_ORDER == __LITTLE_ENDIAN)
#define WL__LE
#endif
#elif defined(_BYTE_ORDER)
#if defined(_BIG_ENDIAN) && (_BYTE_ORDER == _BIG_ENDIAN)
#define WL__BE
#elif defined(_LITTLE_ENDIAN) && (_BYTE_ORDER == _LITTLE_ENDIAN)
#define WL__LE
#endif
#elif defined(__BIG_ENDIAN__)
#define WL__BE
#elif defined(__LITTLE_ENDIAN__)
#define WL__LE
#else
#if defined(__ARMEL__) || defined(__THUMBEL__) || defined(__AARCH64EL__) || \
defined(_MIPSEL) || defined(__MIPSEL) || defined(__MIPSEL__) || \
defined(__ia64__) || defined(_IA64) || defined(__IA64__) || defined(__ia64) || \
defined(_M_IA64) || defined(__itanium__) || defined(i386) || defined(__i386__) || \
defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(__i386) || \
defined(_M_IX86) || defined(_X86_) || defined(__THW_INTEL__) || defined(__I86__) || \
defined(__INTEL__) || defined(__x86_64) || defined(__x86_64__) || \
defined(__amd64__) || defined(__amd64) || defined(_M_X64) || \
defined(__bfin__) || defined(__BFIN__) || defined(bfin) || defined(BFIN)
#define WL__LE
#elif defined(__m68k__) || defined(M68000) || defined(__hppa__) || defined(__hppa) || defined(__HPPA__) || \
defined(__sparc__) || defined(__sparc) || defined(__370__) || defined(__THW_370__) || \
defined(__s390__) || defined(__s390x__) || defined(__SYSC_ZARCH__)
#define WL__BE
#elif defined(__arm__) || defined(__arm64) || defined(__thumb__) || \
defined(__TARGET_ARCH_ARM) || defined(__TARGET_ARCH_THUMB) || defined(__ARM_ARCH) || \
defined(_M_ARM) || defined(_M_ARM64)
#if defined(_WIN32) || defined(_WIN64) || \
defined(__WIN32__) || defined(__TOS_WIN__) || defined(__WINDOWS__)
#define WL__LE
#else
#error "Unknown endianness"
#endif
#endif
#endif

static uint32_t WL__AINLINE wl__bswap32(uint32_t x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
    #ifdef WL__BE
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
    #ifdef WL__BE
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

extern WL__NORET WL__COLDPROC WL_EXPORT void wl__panic(const char* msg, ...);
extern WL_EXPORT bool wl__log_enabled;
extern WL_EXPORT void* (*wl__alloc)(void* blk, size_t size);
extern WL_EXPORT void* wl__alloc_aligned(size_t size, size_t align);
extern WL_EXPORT void wl__free_aligned(void* blk);

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
#define wl__log_info_force(msg, ...) do { fprintf(stdout,   WL__CC_CYAN "[WAVELET] " WL__CC_RESET WL__SRC_NAME " " msg "\n", ## __VA_ARGS__); } while (0)
#define wl__log_warn(msg, ...) do { fprintf(stdout,  WL__CC_CYAN "[WAVELET] " WL__CC_RESET WL__SRC_NAME " " WL__CC_YELLOW msg WL__CC_RESET "\n", ## __VA_ARGS__); fflush(stdout); } while (0)
#define wl__log_error(msg, ...) do { fprintf(stdout,  WL__CC_CYAN "[WAVELET] " WL__CC_RESET WL__SRC_NAME " " WL__CC_RED msg WL__CC_RESET "\n", ## __VA_ARGS__); fflush(stdout); } while (0)

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

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
typedef struct wl__icompute_device_t wl__icompute_device_t;

/* Buffer interface on a compute device */
typedef struct wl__itensor_storage_buffer wl__itensor_storage_buffer;
struct wl__itensor_storage_buffer {
    uintptr_t base;                                                                                     /* Pointer to buffer on device. Never access directly. */
    size_t size;                                                                                        /* Size of buffer in bytes. */
    size_t alignment;                                                                                   /* Alignment of buffer. */
    wl__icompute_device_t* host;                                                                        /* Host device. */
    void (*set)(wl__itensor_storage_buffer* sto, size_t offs, uint8_t x);                               /* Memset buffer. */
    void (*cpy_host_device)(wl__itensor_storage_buffer* sto, size_t offs, const void* src, size_t n);   /* Copy data from host to device. */
    void (*cpy_device_host)(wl__itensor_storage_buffer* sto, size_t offs, void* dst, size_t n);         /* Copy data from device to host. */
};

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
struct wl__icompute_device_t {
    char name[128];                                                         /* Device name. */
    void* impl;                                                             /* Device specific implementation, if applicable. */
    bool is_async;                                                          /* If device is async. */
    wl_compute_device_type_t type;                                          /* Device type enum. */
    void (*eager_exec_fwd)(wl__icompute_device_t* dvc, wl_tensor_t* root);  /* Execute a single op forward. */
    void (*eager_exec_bwd)(wl__icompute_device_t* dvc, wl_tensor_t* root);  /* Execute a single op backwards. */
    wl__itensor_storage_buffer* (*alloc_storage)(wl__icompute_device_t* dvc, size_t size, size_t align);
    void (*free_storage)(wl__icompute_device_t* dvc, wl__itensor_storage_buffer* buf);
};

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

#if WL__SANITIZE_RC
typedef struct wl__tensor_node_t wl__tensor_node_t;
struct wl__tensor_node_t {
    wl_tensor_t* tensor;
    wl__tensor_node_t* next;
};
#endif

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
        uint32_t x86_64_cpu_features[8][4];         /* x86-64 CPU features. */
#endif
    } sys;
#if WL__SANITIZE_RC
    wl__tensor_node_t* rc_tracked;                  /* Linked list of RC tensors for sanitize. */
#endif
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
    wl_prng_algorithm_t prng_algorithm;             /* PRNG algorithm. */
    uintptr_t host_thread_id;                       /* Host thread ID. */
    size_t sh_len;                                  /* Number of shutdown hooks. */
    size_t sh_cap;                                  /* Maximum number of shutdown hooks. */
    wl_compute_device_type_t device_type;           /* Active compute device. */
    wl__icompute_device_t* device;                  /* Active compute device. */
    uint8_t* (*image_load_fn)(const char*, uint32_t(*)[3], wl_color_channels_t);    /* Image loader. stb_image by default, you can plug-in your own. */
    void (*image_load_free_fn)(uint8_t*);                                           /* Image loader free function.  stb_image by default, you can plug-in your own. */
    bool (*image_save_fn)(const char*, const uint8_t*, const uint32_t(*)[3]);       /* Image saver. stb_image by default, you can plug-in your own. */
    void* ud; /* User data. */
};

typedef enum wl__tensor_flags_t {
    WL__TFLAG_NONE = 0,
    WL__TFLAG_OWNER = 1<<0,         /* Tensor is the owner of the buffer. */
    WL__TFLAG_VIEW = 1<<1,          /* Tensor is a view. */
    WL__FLAG_GRAD = 1<<2,           /* Tensor is a gradient. */
    WL__TFLAG_EXEC_EAGER = 1<<3,    /* Tensor is executed eagerly. */

    WL__TFLAG_LEN = 4
} wl__tensor_flags_t;
wl_static_assert(WL__TFLAG_LEN <= 0xff);

/*
** Tensor with up to 6 Dimensions.
*/
struct wl_tensor_t {
    struct {
        uint32_t rc_strong;                         /* Strong reference count. */
        uint32_t rc_weak;                           /* Weak reference count. */
#if WL__SANITIZE_RC
        void (*dtor)(wl_tensor_t*);                 /* Debug destructor. */
#endif
    } rcb;                                          /* Reference count control block. */
    wl_ctx_t* ctx;                                  /* Host context. */
    int64_t rank;                                   /* Number of active dimensions. [1, MAX_DIMS] */
    int64_t shape[WL_MAX_DIMS];                     /* Shape of the tensor. */
    int64_t strides[WL_MAX_DIMS];                   /* Strides of the tensor. We store the strides in element counts and NOT in bytes. */
    wl_dtype_t dtype;                               /* Data type of the tensor. */
    wl__itensor_storage_buffer* storage;            /* Storage buffer. */
    int64_t num_elems;                              /* Number of elements in the tensor. */
    wl__tensor_flags_t flags;                       /* Tensor flags. */
    wl_op_t op;                                     /* Opcode for operators. */
    wl_tensor_t* op_inputs[WL_MAX_INPUT_TENSORS];   /* Input tensors for operators. */
    wl_op_param_t op_params[WL_MAX_OP_PARAMS];      /* Operator parameters. */
    wl_tensor_t* view_uplink;                       /* View base tensor. */
    size_t view_offs;                               /* Offset in view tensor. */
    wl_tensor_t* grad;                              /* ∇f - Gradient tensor. */
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

#ifdef __cplusplus
}
#endif

#endif
