/*
** (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
** MSML - Single file STB-style machine learning library in C99 with Python bindings.
** For license see LICENSE file.
*/

#define MSML_EXPORT_DLL
#include "msml.h"

#include <stdio.h>
#include <stdarg.h>
#include <time.h>

#ifdef MSML_ENABLE_IMAGE_SUPPORT
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
    unsigned char _BitScanForward64(unsigned long*, uint64_t);
    unsigned char _BitScanReverse64(unsigned long*, uint64_t);
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
#   define MSML_UNUSED __declspec(unused)
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

#define msml_max(x, y) (((x) > (y)) ? (x) : (y))
#define msml_min(x, y) (((x) < (y)) ? (x) : (y))
#define MSML_CCRED "\x1b[31m"
#define MSML_CCGREEN "\x1b[32m"
#define MSML_CCYELLOW "\x1b[33m"
#define MSML_CCBLUE "\x1b[34m"
#define MSML_CCMAGENTA "\x1b[35m"
#define MSML_CCCYAN "\x1b[36m"
#define MSML_CCRESET "\x1b[0m"
#define MSML_STRINGIZE(x) MSML_STRINGIZE2(x)
#define MSML_STRINGIZE2(x) #x
#define MSML_SRC_NAME __FILE_NAME__ ":" MSML_STRINGIZE(__LINE__)
#define msml_log_info(msg, ...) fprintf(stdout,  "[MSML] " MSML_SRC_NAME " " msg "\n", ## __VA_ARGS__)
#define msml_log_warn(msg, ...) fprintf(stderr,  "[MSML] " MSML_SRC_NAME " " MSML_CCYELLOW msg MSML_CCRESET "\n", ## __VA_ARGS__)

struct msml_ctx_t {
    void* (*alloc_fn)(void* blk, size_t size);
    size_t chunk_size;
    size_t chunk_len;
    size_t chunk_cap;
    uint8_t** chunks;
    uint8_t* delta;
    size_t alloc_acc;
    size_t mapped_total;
    size_t alloc_total;
    uint64_t prng_state[4];
    void* user_data;
};

struct msml_tensor_t {
    msml_ctx_t* ctx;
    int64_t rank;
    int64_t dims[MSML_MAX_DIMS];
    int64_t strides[MSML_MAX_DIMS];
    msml_dtype_t dtype;
    union {
        uint8_t* u8;
        float* f32;
    } buf;
    int64_t buf_size;
    char name[MSML_MAX_TENSOR_NAME_LEN];
    void* user_data;
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

void* msml_default_allocator(void* blk, size_t size) {
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

static void msml__prng_init_pre_seeded(uint64_t(*state)[4], uint64_t salt) {
    (*state)[0] = 0xa0d277570a345b8cull ^ salt;
    (*state)[1] = 0x764a296c5d4aa64full ^ salt;
    (*state)[2] = 0x51220704070adeaaull ^ salt;
    (*state)[3] = 0x2a2717b5a7b7b927ull ^ salt;
}

#define tausworthe223_gen(state, z, r, i, k, q, v) \
  z = state[i]; \
  z = (((z << q) ^ z) >> (k-v)) ^ ((z & ((uint64_t)(int64_t)-1 << (64-k))) << v); \
  r ^= z; \
  state[i] = z

#define tausworthe223_step(self, z, r) \
  tausworthe223_gen(self, z, r, 0, 63, 31, 18);\
  tausworthe223_gen(self, z, r, 1, 58, 19, 28);\
  tausworthe223_gen(self, z, r, 2, 55, 24,  7);\
  tausworthe223_gen(self, z, r, 3, 47, 21,  8)

static double msml__prng_next_f64(uint64_t(*state)[4]) {
    uint64_t z, r = 0;
    uint64_t* p_state = *state;
    tausworthe223_step(p_state, z, r);
    r = (r & 0x000fffffffffffffull) + 0x3ff0000000000000ull; /* IEEE-754 binary-64 pattern in the range 1.0 <= x < 2.0. */
    union { uint64_t u; double d; } u = { .u = r };
    return u.d - 1.0;
}
#define msml__prng_next_f64_interval(state, min, max) (msml__prng_next_f64(state)*((max)-(min))+(min)) /* Get next random float within [min, max]. */

static void msml__prng_init(uint64_t(*state)[4], double seed) {
    seed = seed != 0.0 ? seed : 5.249176108649e-01; /* Default seed. */
    uint32_t r = 0x11090601;  /* Four 8 bit-seeds merged into a scalar. */
    for (size_t i = 0; i < 4; ++i) {
        uint32_t m = 1u << (r & 0xff); /* Mask. */
        r >>= 8;
        double d = seed = seed * M_PI + M_E;
        union { double d; uint64_t u; } u = { .d = d };
        if (u.u < m) { u.u += m; }
        (*state)[i] = u.u;
    }
    for (int i = 0; i < (rand() % (64 + 1 - 16) + 16); ++i)
        (void)msml__prng_next_f64(state);
}

static void msml__ctx_push_chunk(msml_ctx_t* ctx) {
    uint8_t* chunk = (*ctx->alloc_fn)(NULL, ctx->chunk_size);
    ctx->mapped_total += ctx->chunk_size;
    ctx->delta = chunk + ctx->chunk_size;
    if (ctx->chunk_len == ctx->chunk_cap)
        ctx->chunks = (*ctx->alloc_fn)(ctx->chunks, (ctx->chunk_cap<<=1) * sizeof(*ctx->chunks));
    ctx->chunks[ctx->chunk_len++] = chunk;
}

msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info) {
    printf("Creating MSML context...\n");
    msml_ctx_info_t ctx_info;
    memset(&ctx_info, 0, sizeof(ctx_info));
    if (info) ctx_info = *info;
    ctx_info.alloc_fn = ctx_info.alloc_fn ? ctx_info.alloc_fn : &msml_default_allocator;
    msml_ctx_t* ctx = (*ctx_info.alloc_fn)(NULL, sizeof(*ctx));
    memset(ctx, 0, sizeof(*ctx));
    ctx->alloc_fn = ctx_info.alloc_fn;
    ctx->user_data = ctx_info.user_data;
    ctx->chunk_size = ctx_info.pool_chunk_size ? msml_max(ctx_info.pool_chunk_size, 8) : MSML_DEFAULT_CHUNK_SIZE;
    ctx->chunk_cap = ctx_info.pool_chunks_cap ? msml_max(ctx_info.pool_chunks_cap, 1) : MSML_DEFAULT_CHUNK_CAP;
    ctx->chunks = (uint8_t**)(*ctx->alloc_fn)(NULL, ctx->chunk_cap * sizeof(*ctx->chunks));
    msml__ctx_push_chunk(ctx);
    msml__prng_init_pre_seeded(&ctx->prng_state, (uintptr_t)ctx ^ (uintptr_t)ctx_info.alloc_fn);
    printf("MSML context created.\n");
    return ctx;
}

void* msml_ctx_pool_alloc(msml_ctx_t* ctx, size_t size) {
    msml_assert(size > 0 && size < (size_t)PTRDIFF_MAX, "Allocation size must be within (0, %zu), but is: %zu", PTRDIFF_MAX, size);
    if (ctx->delta - ctx->chunks[ctx->chunk_len-1] < (ptrdiff_t)size) {
        if (ctx->chunk_size < size) { /* Increase the chunk size if it's too small to accommodate the requested length */
            const size_t lim = (size_t)PTRDIFF_MAX>>1;
            do ctx->chunk_size <<= 1;
            while (ctx->chunk_size < size && (ctx->chunk_size <= lim));
        }
        msml__ctx_push_chunk(ctx);
        msml_log_info("Allocated pool chunk: %.03f MiB", (double)ctx->chunk_size/(double)(1<<20));
    }
    ctx->delta -= size;
    ++ctx->alloc_acc;
    ctx->alloc_total += size;
    return ctx->delta;
}

void* msml_ctx_pool_alloc_aligned(msml_ctx_t* ctx, size_t size, size_t align) {
    msml_assert(align && !(align & (align - 1)), "Alignment must be power of 2: %zu", align); /* Alignment must be a power of 2 */
    return (void*)(((uintptr_t)msml_ctx_pool_alloc(ctx, size + align-1) + align-1) & ~(align-1));
}

size_t msml_ctx_total_memory(const msml_ctx_t* ctx) {
    size_t mem = sizeof(*ctx);
    mem += sizeof(*ctx->chunks) * ctx->chunk_cap;
    mem += ctx->mapped_total;
    return mem;
}

void msml_ctx_destroy(msml_ctx_t* ctx) {
    size_t mem_total = msml_ctx_total_memory(ctx);
    for (size_t i=0; i < ctx->chunk_len; ++i) /* Free individual chunks */
        (*ctx->alloc_fn)(ctx->chunks[i], 0);
    (*ctx->alloc_fn)(ctx->chunks, 0);
    (*ctx->alloc_fn)(ctx, 0);
    double mem_size; const char* mem_unit;
    msml__humanize_memory_size(mem_total, &mem_size, &mem_unit);
    printf("Total memory allocated: %.03f %s\n", mem_size, mem_unit);
    printf("MSML context destroyed.\n");
}

#define load_local_storage_group(xk, prefix, var) \
    const int64_t prefix##0 = (xk)->var[0]; \
    const int64_t prefix##1 = (xk)->var[1]; \
    const int64_t prefix##2 = (xk)->var[2]; \
    const int64_t prefix##3 = (xk)->var[3]; \

#define resolve_physical_ptr(tensor, d0, d1, d2, d3) \
    ((tensor)->buf.u8 + d0*(tensor)->strides[0] + d1*(tensor)->strides[1] + d2*(tensor)->strides[2] + d3*(tensor)->strides[3])

const msml_dtype_info_t msml_dtype_info[MSML_DTYPE_COUNT_] = {
    [MSML_DTYPE_F32] = {
        sizeof(float),
        "f32"
    },
};

msml_tensor_t* msml_tensor_create(msml_ctx_t* ctx, msml_dtype_t type, const int64_t* dims, int64_t rank) {
    msml_assert(dims != NULL && rank > -1 && rank <= MSML_MAX_DIMS, "Rank must be within (0, %d]", MSML_MAX_DIMS);
    int64_t scalar_size = (int64_t)msml_dtype_info[type].size;
    int64_t buf_size = scalar_size;
    for (int64_t i=0; i < rank; ++i) {
        msml_assert(dims[i] > 0, "Dimension must be > 0: %lld", dims[i]);
        buf_size *= msml_max(1, dims[i]);
    }
    msml_tensor_t* tensor = msml_ctx_pool_alloc(ctx, sizeof(*tensor) + buf_size); /* Allocate memory for tensor struct and data */
    memset(tensor, 0, sizeof(*tensor));
    tensor->ctx = ctx;
    tensor->rank = rank;
    tensor->dtype = type;
    tensor->buf_size = buf_size;
    for (int64_t i=0; i < MSML_MAX_DIMS; ++i)
        tensor->dims[i] = i < rank ? msml_max(1, dims[i]) : 1;
    *tensor->strides = scalar_size;
    for (int i=1; i < MSML_MAX_DIMS; ++i)
        tensor->strides[i] = tensor->strides[i-1] * tensor->dims[i-1];
    tensor->buf.u8 = (uint8_t*)(tensor + 1); /* Set buffer pointer to the end of the tensor struct, where data follows */
    return tensor;
}

msml_tensor_t* msml_tensor_create_1d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1}, 1);
}

msml_tensor_t* msml_tensor_create_2d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2}, 2);
}

msml_tensor_t* msml_tensor_create_3d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2, d3}, 3);
}

msml_tensor_t* msml_tensor_create_4d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4) {
    return msml_tensor_create(ctx, type, (int64_t[]){d1, d2, d3, d4}, 4);
}

msml_tensor_t* msml_tensor_isomorphic_clone(msml_tensor_t* tensor) {
    msml_tensor_t* isomorph = msml_tensor_create(tensor->ctx, tensor->dtype, tensor->dims, tensor->rank);
    return isomorph;
}

msml_tensor_t* msml_tensor_deep_clone(msml_tensor_t* tensor) {
    msml_tensor_t* clone = msml_tensor_isomorphic_clone(tensor);
    memcpy(clone->buf.u8, tensor->buf.u8, tensor->buf_size);
    return clone;
}

void msml_tensor_copy_buffer_from(msml_tensor_t* tensor, const void* data, size_t size) {
    msml_assert(size == (size_t)tensor->buf_size, "Buffer size mismatch: %zu != %lld", size, tensor->buf_size);
    memcpy(tensor->buf.u8, data, size);
}

void msml_tensor_fill_zero(msml_tensor_t* tensor) {
    memset(tensor->buf.u8, 0, tensor->buf_size);
}

void msml_tensor_fill_one(msml_tensor_t* tensor) {
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = tensor->buf.f32;
            for (int64_t i=0; i < n; ++i) buf[i] = 1.0f;
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
}

void msml_tensor_fill(msml_tensor_t* tensor, float x) {
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = tensor->buf.f32;
            for (int64_t i=0; i < n; ++i) buf[i] = x;
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
}

void msml_tensor_fill_random(msml_tensor_t* tensor, float min, float max) {
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = tensor->buf.f32;
            for (int64_t i=0; i < n; ++i) buf[i] = (float)msml__prng_next_f64(&tensor->ctx->prng_state);
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
}

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
#define ND_MUL2K_MAX_SHIFT 29
#define ND_MUL2K_DIV1E9(val) ((uint32_t)((val) / 1000000000))

/* Multiply nd by 2^k and add carry_in (ndlo is assumed to be zero). */
static uint32_t nd_mul2k(uint32_t* nd, uint32_t ndhi, uint32_t k, uint32_t carry_in, msml_format_flags sf) {
    uint32_t i, ndlo = 0, start = 1;
    /* Performance hacks. */
    if (k > ND_MUL2K_MAX_SHIFT*2 && MSML_FMT_FP(sf) != MSML_FMT_FP(MSML_FMT_T_FP_F)) {
        start = ndhi - (MSML_FMT_PREC(sf) + 17) / 8;
    }
    /* Real logic. */
    while (k >= ND_MUL2K_MAX_SHIFT) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << ND_MUL2K_MAX_SHIFT) | carry_in;
            carry_in = ND_MUL2K_DIV1E9(val);
            nd[i] = (uint32_t)val - carry_in * 1000000000;
        }
        if (carry_in) {
            nd[++ndhi] = carry_in; carry_in = 0;
            if (start++ == ndlo) ++ndlo;
        }
        k -= ND_MUL2K_MAX_SHIFT;
    }
    if (k) {
        for (i = ndlo; i <= ndhi; i++) {
            uint64_t val = ((uint64_t)nd[i] << k) | carry_in;
            carry_in = ND_MUL2K_DIV1E9(val);
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
    } t;
    t.n = n;
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
        int32_t e = (t.u32.hi >> 20) & 0x7ff, ndebias = 0;
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
                e = ((t.u32.hi >> 20) & 0x7ff) - 1075 - (ND_MUL2K_MAX_SHIFT < 29);
                goto load_t_lo; rescale_failed:
                t.n = n;
                e = (t.u32.hi >> 20) & 0x7ff;
                ndebias = ndhi = 0;
            }
        }
        nd[0] = t.u32.hi & 0xfffff;
        if (e == 0) e++; else nd[0] |= 0x100000;
        e -= 1043;
        if (t.u32.lo) {
            e -= 32 + (ND_MUL2K_MAX_SHIFT < 29); load_t_lo:
            #if ND_MUL2K_MAX_SHIFT >= 29
                nd[0] = (nd[0] << 3) | (t.u32.lo >> 29);
                ndhi = nd_mul2k(nd, ndhi, 29, t.u32.lo & 0x1fffffff, sf);
            #elif ND_MUL2K_MAX_SHIFT >= 11
                ndhi = nd_mul2k(nd, ndhi, 11, t.u32.lo >> 21, sf);
                ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo >> 10) & 0x7ff, sf);
                ndhi = nd_mul2k(nd, ndhi, 11, (t.u32.lo <<  1) & 0x7ff, sf);
            #else
            #   error "ND_MUL2K_MAX_SHIFT not big enough"
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
            nde += ndhi * 9 + hilen;
            if (ndebias) {
                /*
                ** Rescaling was performed, but this introduced some error, and might
                ** have pushed us across a rounding boundary. We check whether this
                ** error affected the result by introducing even more error (2ulp in
                ** either direction), and seeing whether a rounding boundary was
                ** crossed. Having already converted the -2ulp case, we save off its
                ** most significant digits, convert the +2ulp case, and compare them.
                */
                int32_t eidx = e + 70 + (ND_MUL2K_MAX_SHIFT < 29)
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
                ndhi = nd_add_m10e(nd, ndhi, 5, nde - prec - 1);
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

void msml_tensor_print(const msml_tensor_t* tensor, bool with_data) {
    double buf_size_cvt = 0.0;
    const char* buf_size_unit = NULL;
    msml__humanize_memory_size(tensor->buf_size, &buf_size_cvt, &buf_size_unit);
    printf("Tensor '%s', DType: %s, Rank: %zu, Dims: [%zu, %zu, %zu, %zu], Strides: [%zu, %zu, %zu, %zu], Size: %.01f %s \n",
       tensor->name,
       msml_dtype_info[tensor->dtype].name,
       (size_t)tensor->rank,
       (size_t)tensor->dims[0],
       (size_t)tensor->dims[1],
       (size_t)tensor->dims[2],
       (size_t)tensor->dims[3],
       (size_t)tensor->strides[0],
       (size_t)tensor->strides[1],
       (size_t)tensor->strides[2],
       (size_t)tensor->strides[3],
       buf_size_cvt,
       buf_size_unit
    );
    if (with_data) {
        printf("[\n");
        const float* buf = tensor->buf.f32;
        for (int64_t i3=0; i3 < tensor->dims[2]; ++i3) { // TODO: d4
            for (int64_t i2=0; i2 < tensor->dims[1]; ++i2) {
                putchar('\t');
                for (int64_t i1=0; i1 < tensor->dims[0]; ++i1) {
                    // TODO: dtype check
                    float x = buf[i3*tensor->dims[1]*tensor->dims[0] + i2*tensor->dims[0] + i1];
                    char fmt_buf[128];
                    msml__fmt_f64(MSML_FMT_G14, x, fmt_buf);
                    printf("%s ", fmt_buf);
                }
                putchar('\n');
            }
        }
        printf("]\n");
    }
}

void msml_tensor_set_name(msml_tensor_t* tensor, const char* name) {
    strncpy(tensor->name, name, MSML_MAX_TENSOR_NAME_LEN);
    tensor->name[MSML_MAX_TENSOR_NAME_LEN-1] = '\0';
}

const char* msml_tensor_get_name(const msml_tensor_t* tensor) {
    return tensor->name;
}

int64_t msml_tensor_rank(const msml_tensor_t* tensor) {
    return tensor->rank;
}

const int64_t* msml_tensor_dims(const msml_tensor_t* tensor) {
    return tensor->dims;
}

const int64_t* msml_tensor_strides(const msml_tensor_t* tensor) {
    return tensor->strides;
}

msml_dtype_t msml_tensor_dtype(const msml_tensor_t* tensor) {
    return tensor->dtype;
}

void* msml_tensor_buf(const msml_tensor_t* tensor) {
    return tensor->buf.u8;
}

int64_t msml_tensor_buf_size(const msml_tensor_t* tensor) {
    return tensor->buf_size;
}

int64_t msml_tensor_buf_len(const msml_tensor_t* tensor) {
    return tensor->buf_size / (int64_t)msml_dtype_info[tensor->dtype].size;
}

int64_t msml_tensor_num_rows(const msml_tensor_t* tensor) {
    int64_t rows=tensor->dims[1];
    for (int64_t i=2; i < MSML_MAX_DIMS; ++i)
        rows *= tensor->dims[i];
    return rows;
}

int64_t msml_tensor_num_cols(const msml_tensor_t* tensor) {
    return tensor->dims[0];
}

bool msml_tensor_is_scalar(const msml_tensor_t* tensor) {
    for (int i=0; i < MSML_MAX_DIMS; ++i)
        if (tensor->dims[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_vector(const msml_tensor_t* tensor) {
    for (int i=1; i < MSML_MAX_DIMS; ++i)
        if (tensor->dims[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_matrix(const msml_tensor_t* tensor) {
    for (int i=2; i < MSML_MAX_DIMS; ++i)
        if (tensor->dims[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_higher_order_3d(const msml_tensor_t* tensor) {
    for (int i=3; i < MSML_MAX_DIMS; ++i)
        if (tensor->dims[i] != 1)
            return false;
    return true;
}

void msml_tensor_virtual_to_physical_index(const msml_tensor_t* tensor, int64_t v_idx, int64_t(*p_idx)[MSML_MAX_DIMS]) {
    load_local_storage_group(tensor, d, dims);
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
    return *tensor->strides == (int64_t)msml_dtype_info[tensor->dtype].size;
}

float msml_tensor_get_scalar_physical_index(const msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    const uint8_t* dst = resolve_physical_ptr(tensor, d0, d1, d2, d3);
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: return *(float*)dst;
        default: msml_panic("Unsupported data type: %s", msml_dtype_info[tensor->dtype].name);
    }
}

void msml_tensor_set_scalar_physical_index(msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3, float x) {
    uint8_t* dst = resolve_physical_ptr(tensor, d0, d1, d2, d3);
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: *(float*)dst = x; break;
        default: msml_panic("Unsupported data type: %s", msml_dtype_info[tensor->dtype].name);
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
            return tensor->buf.f32[v_idx];
        default:
            msml_panic("Unsupported data type: %s", msml_dtype_info[tensor->dtype].name);
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
            tensor->buf.f32[v_idx] = x;
            break;
        default:
            msml_panic("Unsupported data type: %s", msml_dtype_info[tensor->dtype].name);
    }
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
    float* dst = tensor->buf.f32;
    size_t n = width*height*channels;
    msml_assert(n == msml_tensor_buf_len(tensor), "Buffer size mismatch: %zu != %lld", n, msml_tensor_buf_len(tensor));
    for (size_t i=0; i < n; ++i)
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
    const int64_t* dims = msml_tensor_dims(tensor);
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
        float f32_u8 = msml_min(msml_max(tensor->buf.f32[i], 0.0f), 1.0f) * 255.0f;
        image_data[i] = (uint8_t)f32_u8;
    }
    int result = stbi_write_jpg(file_path, (int)width, (int)height, (int)channels, image_data, 100);
    msml_assert(result, "Failed to save tensor to image: %s", file_path);
    (*tensor->ctx->alloc_fn)(image_data, 0); /* Free image data */
    msml_log_info("Saved tensor to image: %s, width: %d, height: %d, channels: %d", file_path, (int)width, (int)height, (int)channels);
#else
    msml_panic("Image support is disabled. MSML must be compiled with MSML_ENABLE_IMAGE_SUPPORT defined.");
#endif
}
