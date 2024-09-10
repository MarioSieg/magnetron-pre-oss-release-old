/*
 * (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
 * MSML - Single header STB-style machine learning library in C99.
 * MIT licensed.
 * 
 * 
 * Do this:
 * #define MSML_IMPLEMENTATION
 * before you include this file into **ONE** C or C++ file to include the implementation.
 * 
 *  It should look like this:
 *  #include ...
 *  #include ...
 *  #define MSML_IMPLEMENTATION
 *  #include "msml.h"
 * 
 * You can override assert and memory allocation by defining msml_assert and msml_malloc, msml_realloc and msml_free before including this header.
 */

#ifndef MSML_INCLUDE_MSML_H
#define MSML_INCLUDE_MSML_H

#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifndef MSML_API
#define MSML_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MSML_DEFAULT_CHUNK_SIZE (1<<20)
#define MSML_DEFAULT_CHUNK_CAP (1<<3)

extern MSML_API void* msml_default_allocator(void* blk, size_t size);

typedef struct msml_ctx_info_t {
    void* (*alloc_fn)(void* blk, size_t size);
    size_t pool_chunk_size;
    size_t pool_chunks_cap;
    void* user_data;
} msml_ctx_info_t;

typedef struct msml_ctx_t msml_ctx_t;

extern MSML_API msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info);
extern MSML_API void* msml_ctx_pool_alloc(msml_ctx_t* ctx, size_t size);
extern MSML_API void* msml_ctx_pool_alloc_aligned(msml_ctx_t* ctx, size_t size, size_t align);
extern MSML_API void msml_ctx_destroy(msml_ctx_t* ctx);

typedef enum msml_dtype_t {
    MSML_DTYPE_F32,

    MSML_DTYPE_COUNT_
} msml_dtype_t;

typedef struct msml_dtype_info_t {
    size_t size;
    const char* name;
} msml_dtype_info_t;
extern MSML_API const msml_dtype_info_t msml_dtype_info[MSML_DTYPE_COUNT_];

#define MSML_MAX_DIMS 4
typedef struct msml_tensor_t {
    msml_ctx_t* ctx;
    int64_t dims[MSML_MAX_DIMS];
    int64_t strides[MSML_MAX_DIMS];
    int64_t rank;
    msml_dtype_t dtype;
    union {
        uint8_t* u8;
        float* f32;
    } buf;
    int64_t buf_size;
    void* user_data;
} msml_tensor_t;

extern MSML_API msml_tensor_t* msml_tensor_create(msml_ctx_t* ctx, msml_dtype_t type, const int64_t* dims, int64_t rank);
extern MSML_API int64_t msml_tensor_num_rows(const msml_tensor_t* tensor);
extern MSML_API int64_t msml_tensor_num_cols(const msml_tensor_t* tensor);
extern MSML_API bool msml_tensor_is_scalar(const msml_tensor_t* tensor);
extern MSML_API bool msml_tensor_is_vector(const msml_tensor_t* tensor);
extern MSML_API bool msml_tensor_is_matrix(const msml_tensor_t* tensor);
extern MSML_API bool msml_tensor_is_higher_order_3d(const msml_tensor_t* tensor);
extern MSML_API void msml_tensor_virtual_to_physical_index(const msml_tensor_t* tensor, int64_t v_idx, int64_t(*p_idx)[MSML_MAX_DIMS]);
extern MSML_API int64_t msml_tensor_physical_to_virtual_index(const msml_tensor_t* tensor, const int64_t (*p_idx)[MSML_MAX_DIMS]);
extern MSML_API bool msml_tensor_is_contiguous(const msml_tensor_t* tensor);

#ifdef __cplusplus
}
#endif
#endif

#ifdef MSML_IMPLEMENTATION

#include <stdio.h>
#include <stdarg.h>

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
#else
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

static void msml_panic(const char* msg, ...) {
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
    void* user_data;
};

static void msml__ctx_push_chunk(msml_ctx_t* ctx) {
    uint8_t* chunk = (uint8_t*)(*ctx->alloc_fn)(NULL, ctx->chunk_size);
    ctx->mapped_total += ctx->chunk_size;
    ctx->delta = chunk + ctx->chunk_size;
    if (ctx->chunk_len == ctx->chunk_cap)
        ctx->chunks = (uint8_t**)(*ctx->alloc_fn)(ctx->chunks, (ctx->chunk_cap <<= 1) * sizeof(*ctx->chunks));
    ctx->chunks[ctx->chunk_len++] = chunk;
}

msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info) {
    msml_ctx_info_t ctx_info;
    memset(&ctx_info, 0, sizeof(ctx_info));
    if (info) ctx_info = *info;
    ctx_info.alloc_fn = ctx_info.alloc_fn ? ctx_info.alloc_fn : &msml_default_allocator;
    msml_ctx_t* ctx = (msml_ctx_t*)(*ctx_info.alloc_fn)(NULL, sizeof(*ctx));
    memset(ctx, 0, sizeof(*ctx));
    ctx->alloc_fn = ctx_info.alloc_fn;
    ctx->user_data = ctx_info.user_data;
    ctx->chunk_size = ctx_info.pool_chunk_size ? msml_max(ctx_info.pool_chunk_size, 8) : MSML_DEFAULT_CHUNK_SIZE;
    ctx->chunk_cap = ctx_info.pool_chunks_cap ? msml_max(ctx_info.pool_chunks_cap, 1) : MSML_DEFAULT_CHUNK_CAP;
    ctx->chunks = (uint8_t**)(*ctx->alloc_fn)(NULL, ctx->chunk_cap * sizeof(*ctx->chunks));
    msml__ctx_push_chunk(ctx);
    return ctx;
}

void* msml_ctx_pool_alloc(msml_ctx_t* ctx, size_t size) {
    msml_assert(size > 0 && size < (size_t)PTRDIFF_MAX, "Allocation size must be within (0, %zu), but is: %zu", PTRDIFF_MAX, size);
    if (ctx->delta - ctx->chunks[ctx->chunk_len-1] < (ptrdiff_t)size) {
        if (ctx->chunk_size < size) { /* Increase the chunk size if it's too small to accommodate the requested length */
            while (((ctx->chunk_size <<= 1) < size)
                && (ctx->chunk_size <= (size_t)(PTRDIFF_MAX >> 1)));
        }
        msml__ctx_push_chunk(ctx);
    }
    ctx->delta -= size;
    ++ctx->alloc_acc;
    ctx->alloc_total += size;
    return ctx->delta;
}

void* msml_ctx_pool_alloc_aligned(msml_ctx_t* ctx, size_t size, size_t align) {
    msml_assert(align && !(align & (align - 1)), "Alignment must be power of 2: %zu", align); /* Alignment must be a power of 2 */
    const size_t mask = align - 1;
    return (void*)(((uintptr_t)msml_ctx_pool_alloc(ctx, size + mask) + mask) & ~mask);
}

void msml_ctx_destroy(msml_ctx_t* ctx) {
    for (size_t i=0; i < ctx->chunk_len; ++i) /* Free individual chunks */
        (*ctx->alloc_fn)(ctx->chunks[i], 0);
    (*ctx->alloc_fn)(ctx->chunks, 0);
    (*ctx->alloc_fn)(ctx, 0);
}

#define load_local_storage_group(xk, prefix, var) \
    const int64_t prefix##0 = (xk)->var[0]; \
    const int64_t prefix##1 = (xk)->var[1]; \
    const int64_t prefix##2 = (xk)->var[2]; \
    const int64_t prefix##3 = (xk)->var[3]; \

const msml_dtype_info_t msml_dtype_info[MSML_DTYPE_COUNT_] = {
    [MSML_DTYPE_F32] = {sizeof(float), "f32"},
};

msml_tensor_t* msml_tensor_create(msml_ctx_t* ctx, msml_dtype_t type, const int64_t* dims, int64_t rank) {
    msml_assert(rank > -1 && rank <= MSML_MAX_DIMS, "Rank must be within (0, %d]", MSML_MAX_DIMS);
    int64_t scalar_size = (int64_t)msml_dtype_info[type].size;
    int64_t buf_size = scalar_size;
    for (int64_t i=0; i < rank; ++i) {
        msml_assert(dims[i] > 0, "Dimension must be > 0: %lld", dims[i]);
        buf_size *= dims[i];
    }
    msml_tensor_t* tensor = (msml_tensor_t*)msml_ctx_pool_alloc(ctx, sizeof(*tensor) + buf_size); /* Allocate memory for tensor struct and data */
    memset(tensor, 0, sizeof(*tensor));
    tensor->ctx = ctx;
    tensor->rank = rank;
    tensor->dtype = type;
    tensor->buf_size = buf_size;
    for (int64_t i=0; i < MSML_MAX_DIMS; ++i) {
        tensor->dims[i] = i < rank ? dims[i] : 1;
    }
    *tensor->strides = scalar_size;
    for (int i=1; i < MSML_MAX_DIMS; ++i)
        tensor->strides[i] = tensor->strides[i-1] * tensor->dims[i-1];
    tensor->buf.u8 = (uint8_t*)(tensor + 1); /* Set buffer pointer to the end of the tensor struct, where data follows */
    return tensor;
}

int64_t msml_tensor_num_rows(const msml_tensor_t* tensor) {
    int64_t rows=tensor->dims[1];
    for (int64_t i=1; i < MSML_MAX_DIMS; ++i)
        rows *= tensor->dims[i];
    return rows;
}

int64_t msml_tensor_num_cols(const msml_tensor_t* tensor) {
    return tensor->dims[0];
}

bool msml_tensor_is_scalar(const msml_tensor_t* tensor) {
    for (int i=1; i < MSML_MAX_DIMS; ++i)
        if (tensor->dims[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_vector(const msml_tensor_t* tensor) {
    for (int i=2; i < MSML_MAX_DIMS; ++i)
        if (tensor->dims[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_matrix(const msml_tensor_t* tensor) {
    for (int i=3; i < MSML_MAX_DIMS; ++i)
        if (tensor->dims[i] != 1)
            return false;
    return true;
}

bool msml_tensor_is_higher_order_3d(const msml_tensor_t* tensor) {
    return tensor->dims[MSML_MAX_DIMS-1] == 1;
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

#endif

