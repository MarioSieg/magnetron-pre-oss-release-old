/*
 * (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
 * MSML - Single file STB-style machine learning library in C99 with Python bindings.
 * MIT licensed.
 */

#define MSML_EXPORT_DLL
#include "msml.h"

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

static void msml__ctx_push_chunk(msml_ctx_t* ctx) {
    uint8_t* chunk = (uint8_t*)(*ctx->alloc_fn)(NULL, ctx->chunk_size);
    ctx->mapped_total += ctx->chunk_size;
    ctx->delta = chunk + ctx->chunk_size;
    if (ctx->chunk_len == ctx->chunk_cap)
        ctx->chunks = (uint8_t**)(*ctx->alloc_fn)(ctx->chunks, (ctx->chunk_cap <<= 1) * sizeof(*ctx->chunks));
    ctx->chunks[ctx->chunk_len++] = chunk;
}

msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info) {
    printf("Creating MSML context...\n");
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
    printf("MSML context created.\n");
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
    printf("MSML context destroyed.\n");
}

#define load_local_storage_group(xk, prefix, var) \
    const int64_t prefix##0 = (xk)->var[0]; \
    const int64_t prefix##1 = (xk)->var[1]; \
    const int64_t prefix##2 = (xk)->var[2]; \
    const int64_t prefix##3 = (xk)->var[3]; \

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
        buf_size *= dims[i];
    }
    msml_tensor_t* tensor = (msml_tensor_t*)msml_ctx_pool_alloc(ctx, sizeof(*tensor) + buf_size); /* Allocate memory for tensor struct and data */
    memset(tensor, 0, sizeof(*tensor));
    tensor->ctx = ctx;
    tensor->rank = rank;
    tensor->dtype = type;
    tensor->buf_size = buf_size;
    for (int64_t i=0; i < MSML_MAX_DIMS; ++i)
        tensor->dims[i] = i < rank ? dims[i] : 1;
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

void msml_tensor_set_zero(msml_tensor_t* tensor) {
    memset(tensor->buf.u8, 0, tensor->buf_size);
}

void msml_tensor_set_one(msml_tensor_t* tensor) {
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = tensor->buf.f32;
            for (int64_t i=0; i < n; ++i) buf[i] = 1.0f;
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
}

void msml_tensor_set(msml_tensor_t* tensor, float x) {
    switch (tensor->dtype) {
        case MSML_DTYPE_F32: {
            int64_t n = msml_tensor_buf_len(tensor);
            float* buf = tensor->buf.f32;
            for (int64_t i=0; i < n; ++i) buf[i] = x;
        } break;
        default: msml_panic("Unsupported DType: %d", tensor->dtype);
    }
}

void msml_tensor_print(const msml_tensor_t* tensor, bool with_data) {
    printf("Tensor '%s', DType: %s, Rank: %zu, Dims: [%zu, %zu, %zu, %zu], Strides: [%zu, %zu, %zu, %zu], Size: %.03fKiB \n",
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
           (double)tensor->buf_size/(double)(1<<10)
    );
    if (with_data) {
        printf("[\n");
        const float* buf = tensor->buf.f32;
        for (int64_t i3=0; i3 < tensor->dims[2]; ++i3) { // TODO: d4
            for (int64_t i2=0; i2 < tensor->dims[1]; ++i2) {
                putchar('\t');
                for (int64_t i1=0; i1 < tensor->dims[0]; ++i1) {
                    // TODO: dtype check
                    printf("%f ", buf[i3*tensor->dims[1]*tensor->dims[0] + i2*tensor->dims[0] + i1]);
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
