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

#define WL_GELU_COEFF 0.044715f
#define WL_GRA_FWD WL_GRAPH_EVAL_ORDER_FORWARD
#define WL_GRA_BWD WL_GRAPH_EVAL_ORDER_REVERSE
#define WL_GRA_LEN 2
#define WL_MAX_CPUS 8192
#define WL_MAX_NUMA_NODES 64
#define WL_STORAGE_EXT ".wavelet"

#if defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)

#define WL_NORET __attribute__((noreturn))
#define WL_ALIGN(x) __attribute__((aligned(x)))
#define WL_AINLINE inline __attribute__((always_inline))
#define WL_NOINLINE __attribute__((noinline))
#define WL_HOTPROC __attribute__((hot))
#define WL_COLDPROC __attribute__((cold))
#define WL_PACKED __attribute__((packed))
#define WL_FALLTHROUGH __attribute__((fallthrough))
#define WL_UNUSED __attribute__((unused))
#define wl_likely(x) __builtin_expect(!!(x), 1)
#define wl_unlikely(x) __builtin_expect(!!(x), 0)
#define wl_ffs(x) ((uint32_t)__builtin_ctz(x))
#define wl_fls(x) ((uint32_t)(__builtin_clz(x)^31))
#define wl_ffs64(x) ((uint32_t)__builtin_ctzll(x))
#define wl_fls64(x) ((uint32_t)(__builtin_clzll(x)^63))

typedef int32_t wl_atomic_t;       /* Atomic integer type */
typedef enum wl_mo_t {             /* Atomic memory order */
    WL_MO_RELAXED = __ATOMIC_RELAXED,
    WL_MO_CONSUME = __ATOMIC_CONSUME,
    WL_MO_ACQUIRE = __ATOMIC_ACQUIRE,
    WL_MO_RELEASE = __ATOMIC_RELEASE,
    WL_MO_ACQ_REL = __ATOMIC_ACQ_REL,
    WL_MO_SEQ_CST = __ATOMIC_SEQ_CST
} wl_mo_t;

static WL_AINLINE void wl_atomic_store(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    __atomic_store_n(o, x, order);
}
static WL_AINLINE wl_atomic_t wl_atomic_load(volatile wl_atomic_t* o, wl_mo_t order) {
    return __atomic_load_n(o, order);
}
static WL_AINLINE wl_atomic_t wl_atomic_fetch_add(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    return __atomic_fetch_add(o, x, order);
}
static WL_AINLINE wl_atomic_t wl_atomic_fetch_sub(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    return __atomic_fetch_sub(o, x, order);
}
static WL_AINLINE wl_atomic_t wl_atomic_fetch_and(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    return __atomic_fetch_and(o, x, order);
}
static WL_AINLINE wl_atomic_t wl_atomic_fetch_or(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    return __atomic_fetch_or(o, x, order);
}
static WL_AINLINE wl_atomic_t wl_atomic_fetch_xor(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    return __atomic_fetch_xor(o, x, order);
}
static WL_AINLINE wl_atomic_t wl_atomic_exchange(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    return __atomic_exchange_n(o, x, order);
}
static WL_AINLINE bool wl_atomic_compare_exchange_weak(volatile wl_atomic_t* o, wl_atomic_t *exp, wl_atomic_t *des, wl_mo_t order_succ, wl_mo_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, true, order_succ, order_fail);
}
static WL_AINLINE bool wl_atomic_compare_exchange_strong(volatile wl_atomic_t* o, wl_atomic_t *exp, wl_atomic_t *des, wl_mo_t order_succ, wl_mo_t order_fail) {
    return __atomic_compare_exchange(o, exp, des, false, order_succ, order_fail);
}

#else

unsigned char _BitScanForward64(unsigned long*, unsigned __int64);
unsigned char _BitScanReverse64(unsigned long*, unsigned __int64);
#pragma intrinsic(_BitScanForward64)
#pragma intrinsic(_BitScanReverse64)
#define WL_NORET __declspec(noreturn)
#define WL_ALIGN(x) __declspec(align(x))
#define WL_AINLINE inline __forceinline
#define WL_NOINLINE __declspec(noinline)
#define WL_HOTPROC
#define WL_COLDPROC
#define WL_PACKED __declspec(align(1))
#define WL_FALLTHROUGH
#define WL_UNUSED
#define wl_likely(x) (x)
#define wl_unlikely(x) (x)
static WL_AINLINE uint32_t wl_ffs(const uint32_t x) {
    unsigned long r; _BitScanForward(&r, x); return (uint32_t)r;
}
static WL_AINLINE uint32_t wl_fls(const uint32_t x) {
    unsigned long r; _BitScanReverse(&r, x); return (uint32_t)r;
}
static WL_AINLINE uint32_t wl_ffs64(const uint64_t x) {
  unsigned long r; _BitScanForward64(&r, x); return (uint32_t)r;
}
static WL_AINLINE uint32_t wl_fls64(const uint64_t x) {
  unsigned long r; _BitScanReverse64(&r, x); return (uint32_t)r;
}
#define __alignof__ __alignof

typedef long wl_atomic_t;       /* Atomic integer type */
typedef enum wl_mo_t {             /* Atomic memory order */
    WL_MO_RELAXED,
    WL_MO_CONSUME,
    WL_MO_ACQUIRE,
    WL_MO_RELEASE,
    WL_MO_ACQ_REL,
    WL_MO_SEQ_CST
} wl_mo_t;

static WL_AINLINE void wl_atomic_store(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    (void)order; _InterlockedExchange(o, x);
}
static WL_AINLINE wl_atomic_t wl_atomic_load(volatile wl_atomic_t* o, wl_mo_t order) {
    (void)order;
    wl_atomic_t r;
    _InterlockedExchange(&r, *o);
    return r;
}
static WL_AINLINE wl_atomic_t wl_atomic_fetch_add(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    (void)order;
    return _InterlockedExchangeAdd(o, x);
}
static WL_AINLINE wl_atomic_t wl_atomic_fetch_sub(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    (void)order;
    return _InterlockedExchangeAdd(o, -x);
}
static WL_AINLINE wl_atomic_t wl_atomic_fetch_and(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    (void)order;
    return _InterlockedAnd(o, x);
}
static WL_AINLINE wl_atomic_t wl_atomic_fetch_or(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    (void)order;
    return _InterlockedOr(o, x);
}
static WL_AINLINE wl_atomic_t wl_atomic_fetch_xor(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    (void)order;
    return _InterlockedXor(o, x);
}
static WL_AINLINE wl_atomic_t wl_atomic_exchange(volatile wl_atomic_t* o, wl_atomic_t x, wl_mo_t order) {
    (void)order;
    return _InterlockedExchange(o, x);
}
static WL_AINLINE bool wl_atomic_compare_exchange_weak(volatile wl_atomic_t* o, wl_atomic_t *exp, wl_atomic_t *des, wl_mo_t order_succ, wl_mo_t order_fail) {
    (void)order_succ; (void)order_fail;
    return _InterlockedCompareExchange(o, *des, *exp) == *exp;
}
static WL_AINLINE bool wl_atomic_compare_exchange_strong(volatile wl_atomic_t* o, wl_atomic_t *exp, wl_atomic_t *des, wl_mo_t order_succ, wl_mo_t order_fail) {
    (void)order_succ; (void)order_fail;
    return _InterlockedCompareExchange(o, *des, *exp) == *exp;
}

#endif

wl_static_assert(sizeof(0u) == 4);
wl_static_assert(sizeof(0ull) == 8);

#ifdef __BYTE_ORDER
#if defined(__BIG_ENDIAN) && (__BYTE_ORDER == __BIG_ENDIAN)
#define WL_BE
#elif defined(__LITTLE_ENDIAN) && (__BYTE_ORDER == __LITTLE_ENDIAN)
#define WL_LE
#endif
#elif defined(_BYTE_ORDER)
#if defined(_BIG_ENDIAN) && (_BYTE_ORDER == _BIG_ENDIAN)
#define WL_BE
#elif defined(_LITTLE_ENDIAN) && (_BYTE_ORDER == _LITTLE_ENDIAN)
#define WL_LE
#endif
#elif defined(__BIG_ENDIAN__)
#define WL_BE
#elif defined(__LITTLE_ENDIAN__)
#define WL_LE
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
#define WL_LE
#elif defined(__m68k__) || defined(M68000) || defined(__hppa__) || defined(__hppa) || defined(__HPPA__) || \
defined(__sparc__) || defined(__sparc) || defined(__370__) || defined(__THW_370__) || \
defined(__s390__) || defined(__s390x__) || defined(__SYSC_ZARCH__)
#define WL_BE
#elif defined(__arm__) || defined(__arm64) || defined(__thumb__) || \
defined(__TARGET_ARCH_ARM) || defined(__TARGET_ARCH_THUMB) || defined(__ARM_ARCH) || \
defined(_M_ARM) || defined(_M_ARM64)
#if defined(_WIN32) || defined(_WIN64) || \
defined(__WIN32__) || defined(__TOS_WIN__) || defined(__WINDOWS__)
#define WL_LE
#else
#error "Unknown endianness"
#endif
#endif
#endif

static uint32_t WL_AINLINE wl_bswap32(uint32_t x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
    #ifdef WL_BE
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

static uint64_t WL_AINLINE wl_bswap64(uint64_t x) { /* Swap bytes for endianess switch. Should be optimized to a (bswap/rev) instruction on modern compilers. */
    #ifdef WL_BE
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

extern WL_NORET WL_COLDPROC WL_EXPORT void wl_panic(const char* msg, ...);
extern WL_EXPORT bool wl_log_enabled;
extern WL_EXPORT void* (*wl_alloc)(void* blk, size_t size);
extern WL_EXPORT void* wl_alloc_aligned(size_t size, size_t align);
extern WL_EXPORT void wl_free_aligned(void* blk);

#define wl_swap(T, a, b) do { T tmp = (a); (a) = (b); (b) = tmp; } while (0)
#define wl_max(x, y) (((x) > (y)) ? (x) : (y))
#define wl_min(x, y) (((x) < (y)) ? (x) : (y))
#define WL_CC_RED "\x1b[31m"
#define WL_CC_GREEN "\x1b[32m"
#define WL_CC_YELLOW "\x1b[33m"
#define WL_CC_BLUE "\x1b[34m"
#define WL_CC_MAGENTA "\x1b[35m"
#define WL_CC_CYAN "\x1b[36m"
#define WL_CC_RESET "\x1b[0m"
#define WL_STRINGIZE2(x) #x
#define WL_STRINGIZE(x) WL_STRINGIZE2(x)
#ifdef __FILE_NAME__
#   define WL_SRC_NAME __FILE_NAME__ ":" WL_STRINGIZE(__LINE__)
#else
#   define WL_SRC_NAME __FILE__ ":" WL_STRINGIZE(__LINE__)
#endif
#define wl_log_info(msg, ...) do { if (wl_unlikely(wl_log_enabled)) fprintf(stdout,   WL_CC_CYAN "[WAVELET] " WL_CC_RESET WL_SRC_NAME " " msg "\n", ## __VA_ARGS__); } while (0)
#define wl_log_info_force(msg, ...) do { fprintf(stdout,   WL_CC_CYAN "[WAVELET] " WL_CC_RESET WL_SRC_NAME " " msg "\n", ## __VA_ARGS__); } while (0)
#define wl_log_warn(msg, ...) do { fprintf(stdout,  WL_CC_CYAN "[WAVELET] " WL_CC_RESET WL_SRC_NAME " " WL_CC_YELLOW msg WL_CC_RESET "\n", ## __VA_ARGS__); fflush(stdout); } while (0)
#define wl_log_error(msg, ...) do { fprintf(stdout,  WL_CC_CYAN "[WAVELET] " WL_CC_RESET WL_SRC_NAME " " WL_CC_RED msg WL_CC_RESET "\n", ## __VA_ARGS__); fflush(stdout); } while (0)

#define wl_assert(expr, msg, ...) \
    if (wl_unlikely(!(expr))) { \
        wl_panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
    }
#define wl_assert2(expr) wl_assert(expr, "")

#if WL_BOUNDS_CHECK
#define wl_bnd_chk(ptr, base, n) \
    wl_assert((uintptr_t)(ptr) >= (uintptr_t)(base) && (uintptr_t)(ptr) < (uintptr_t)(base) + (n), \
        "\nBound check failed: %p not in [%p, %p), base+%zu, end+%zu", \
        (void*)(ptr), \
        (void*)(base), \
        (void*)((uintptr_t)(base)+(n)), \
        (size_t)llabs((long long)((int64_t)(ptr)-(int64_t)(base))), \
        (size_t)llabs((long long)(((int64_t)(base)+(n))-(int64_t)(ptr))) \
    )
#else
#define wl_bnd_chk(ptr, base, n)
#endif

/* Increment pointer or size with correct type alignment. */
static WL_AINLINE void* wl_pincr(void** p, size_t sz, size_t align) {
    void* pp = (void*)(((uintptr_t)*p+align-1)&-align);
    *p = (void*)((uint8_t*)pp+sz);
    return pp;
}

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
typedef struct wl_compute_device_t wl_compute_device_t;

/* Buffer interface on a compute device */
typedef struct wl_storage_buffer wl_storage_buffer;
struct wl_storage_buffer {
    uintptr_t base;                                                                                     /* Pointer to buffer on device. Never access directly. */
    size_t size;                                                                                        /* Size of buffer in bytes. */
    size_t alignment;                                                                                   /* Alignment of buffer. */
    wl_compute_device_t* host;                                                                        /* Host device. */
    void (*set)(wl_storage_buffer* sto, size_t offs, uint8_t x);                               /* Memset buffer. */
    void (*cpy_host_device)(wl_storage_buffer* sto, size_t offs, const void* src, size_t n);   /* Copy data from host to device. */
    void (*cpy_device_host)(wl_storage_buffer* sto, size_t offs, void* dst, size_t n);         /* Copy data from device to host. */
};

/* Device interface to any compute backend device (CPU, GPU, TPU etc..) */
struct wl_compute_device_t {
    char name[128];                                                         /* Device name. */
    void* impl;                                                             /* Device specific implementation, if applicable. */
    bool is_async;                                                          /* If device is async. */
    wl_compute_device_type_t type;                                          /* Device type enum. */
    void (*eager_exec_fwd)(wl_compute_device_t* dvc, wl_tensor_t* root);  /* Execute a single op forward. */
    void (*eager_exec_bwd)(wl_compute_device_t* dvc, wl_tensor_t* root);  /* Execute a single op backwards. */
    wl_storage_buffer* (*alloc_storage)(wl_compute_device_t* dvc, size_t size, size_t align);
    void (*free_storage)(wl_compute_device_t* dvc, wl_storage_buffer* buf);
};

/* Profiling performance monitor per op. */
typedef struct wl_perf_mon_t {
    uint64_t elapsed_ns;
    uint64_t elapsed_ns_acc;
    uint64_t n_execs;
} wl_perf_mon_t;

/* Performance monitor for profiler session. */
typedef struct wl_op_perf_info_t {
    uint64_t elapsed_ns_acc;
    uint64_t n_execs;
} wl_op_perf_info_t;

#if WL_SANITIZE_RC
typedef struct wl_tensor_node_t wl_tensor_node_t;
struct wl_tensor_node_t {
    wl_tensor_t* tensor;
    wl_tensor_node_t* next;
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
#if WL_SANITIZE_RC
    wl_tensor_node_t* rc_tracked;                  /* Linked list of RC tensors for sanitize. */
#endif
    wl_exec_mode_t exec_mode;
    bool profiler_enabled;
    wl_op_perf_info_t op_perf_mons_total[WL_OP__COUNT];
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
    wl_compute_device_t* device;                  /* Active compute device. */
    uint8_t* (*image_load_fn)(const char*, uint32_t(*)[3], wl_color_channels_t);    /* Image loader. stb_image by default, you can plug-in your own. */
    void (*image_load_free_fn)(uint8_t*);                                           /* Image loader free function.  stb_image by default, you can plug-in your own. */
    bool (*image_save_fn)(const char*, const uint8_t*, const uint32_t(*)[3]);       /* Image saver. stb_image by default, you can plug-in your own. */
    void* ud; /* User data. */
};

typedef enum wl_tensor_flags_t {
    WL_TFLAG_NONE = 0,
    WL_TFLAG_OWNER = 1<<0,         /* Tensor is the owner of the buffer. */
    WL_TFLAG_VIEW = 1<<1,          /* Tensor is a view. */
    WL_FLAG_GRAD = 1<<2,           /* Tensor is a gradient. */
    WL_TFLAG_EXEC_EAGER = 1<<3,    /* Tensor is executed eagerly. */

    WL_TFLAG_LEN = 4
} wl_tensor_flags_t;
wl_static_assert(WL_TFLAG_LEN <= 0xff);

/*
** Tensor with up to 6 Dimensions.
*/
struct wl_tensor_t {
    struct {
        uint32_t rc_strong;                         /* Strong reference count. */
        uint32_t rc_weak;                           /* Weak reference count. */
#if WL_SANITIZE_RC
        void (*dtor)(wl_tensor_t*);                 /* Debug destructor. */
#endif
    } rcb;                                          /* Reference count control block. */
    wl_ctx_t* ctx;                                  /* Host context. */
    int64_t rank;                                   /* Number of active dimensions. [1, MAX_DIMS] */
    int64_t shape[WL_MAX_DIMS];                     /* Shape of the tensor. */
    int64_t strides[WL_MAX_DIMS];                   /* Strides of the tensor. We store the strides in element counts and NOT in bytes. */
    wl_dtype_t dtype;                               /* Data type of the tensor. */
    wl_storage_buffer* storage;            /* Storage buffer. */
    int64_t num_elems;                              /* Number of elements in the tensor. */
    wl_tensor_flags_t flags;                       /* Tensor flags. */
    wl_op_t op;                                     /* Opcode for operators. */
    wl_tensor_t* op_inputs[WL_MAX_INPUT_TENSORS];   /* Input tensors for operators. */
    wl_op_param_t op_params[WL_MAX_OP_PARAMS];      /* Operator parameters. */
    wl_tensor_t* view_uplink;                       /* View base tensor. */
    size_t view_offs;                               /* Offset in view tensor. */
    wl_tensor_t* grad;                              /* ∇f - Gradient tensor. */
    wl_perf_mon_t pmon;                            /* Performance monitor. */
    char name[WL_MAX_TENSOR_NAME_LEN];              /* Tensor debug name. */
    void* ud;                                       /* User data. */
};

#define wl_load_local_storage_group_arr(arr, prefix) \
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

#define wl_load_local_storage_group(xk, prefix, var) wl_load_local_storage_group_arr((xk)->var, prefix)

#ifdef __cplusplus
}
#endif

#endif
