/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#ifndef WL_INCLUDE_WL_H
#define WL_INCLUDE_WL_H

/* Compile time config macros */
#define WL_CFG_X86_64_FAST_MATH 1 /* Use fast math for x86_64 by setting mxcsr control register. */
#define WL_INTRIN 1 /* Use platform and compiler specific intrinsics for performance. */
#define WL_BOUNDS_CHECK 1 /* Enable bounds checking for BLAS routines. */

#if !defined(NDEBUG) && !WL_BOUNDS_CHECK
#undef WL_BOUNDS_CHECK
#define WL_BOUNDS_CHECK 1
#endif

#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>

#define WL_DEFAULT_CHUNK_SIZE (1ull<<30)  /* Default size of memory chunk in bytes. 1 GiB */
#define WL_DEFAULT_CHUNK_CAP 128          /* Default capacity of memory chunk */
#define WL_MAX_DIMS 6                     /* Maximum number of dimensions for a tensor */
#define WL_MAX_TENSOR_NAME_LEN 64         /* Maximum length for tensor name */
#define WL_MAX_INPUT_TENSORS 2            /* Maximum number of input tensors for an operation */
#define WL_MAX_OP_PARAMS 6                /* Maximum number of parameters for an operation */

#ifndef WL_EXPORT
#   ifdef WL_EXPORT_DLL
#       ifdef _MSC_VER
#           define WL_EXPORT __declspec(dllexport)
#       else
#           define WL_EXPORT __attribute__((visibility("default")))
#       endif
#   else
#       define WL_EXPORT
#   endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define wl_version_pack(major, minor) ((uint32_t)((((major)&0xff)<<8)+((minor)&0xff)))
#define wl_version_major(version) (((version)>>8)&0xff)
#define wl_version_minor(version) ((version)&0xff)
#define WL_VERSION wl_version_pack(0, 1) /* WAVELET library version. */
#define WL_STORAGE_VERSION 1 /* WAVELET storage format version. */

#define wl_assert_name2(name, line) name ## line
#define wl_assert_name(line) wl_assert_name2(_assert_, line)
#define wl_static_assert(expr) extern void wl_assert_name(__LINE__)(bool STATIC_ASSERTION_FAILED[((expr)?1:-1)])

extern WL_EXPORT void* wl_default_allocator_impl(void* blk, size_t size); /* Default memory allocator */
#ifndef wl_alloc /* Default allocator, can be overridden by defining it before.  */
#define wl_alloc wl_default_allocator_impl
#endif

typedef enum wl_exec_mode_t {
    WL_EXEC_MODE_EAGER = 0, /* Execute operations immediately. (Dynamic computation graph, like PyTorch). */
    WL_EXEC_MODE_DEFERRED = 1 /* Build computation graph and execute later. (Static computation graph, like TensorFlow 1.0). */
} wl_exec_mode_t;

typedef enum wl_prng_algorithm_t {
    WL_PRNG_MERSENNE_TWISTER = 0, /* Mersenne Twister PRNG */
    WL_PRNG_PCG = 1 /* Permuted Congruential Generator PRNG */
} wl_prng_algorithm_t;

typedef enum wl_color_channels_t {
    WL_COLOR_CHANNELS_AUTO,  /* Automatically detect number of color channels */
    WL_COLOR_CHANNELS_GRAY,  /* Grayscale F32 */
    WL_COLOR_CHANNELS_GRAY_A,/* Grayscale F32 + Alpha F32 */
    WL_COLOR_CHANNELS_RGB,   /* R32G32B32 */
    WL_COLOR_CHANNELS_RGBA   /* R32G32B32A32 */
} wl_color_channels_t;

typedef struct wl_ctx_info_t {
    void* (*alloc_fn)(void* blk, size_t size); /* Custom allocator function */
    size_t pool_chunk_size; /* Size of each memory pool chunk */
    size_t pool_chunks_cap; /* Maximum chunks in the pool */
    uint64_t prng_seed; /* Seed for PRNG if prng_init_seed == true */
    bool warmup_chunks; /* If true, fresh pool chunks are filled to allocate kernel pages, can improve performance depending on scenario. */
    wl_prng_algorithm_t prng_algorithm; /* PRNG algorithm */
    wl_exec_mode_t exec_mode; /* Default context execution mode */
    uint8_t* (*image_load_fn)(const char*, uint32_t(*)[3], wl_color_channels_t); /* Image raw data loader. */
    void (*image_load_free_fn)(uint8_t*); /* Free function for buffer returned by image_load_fn(). */
    bool (*image_save_fn)(const char*, const uint8_t*, const uint32_t(*)[3]); /* Image raw data saver. */
    void* user_data; /* User-defined data */
} wl_ctx_info_t;

typedef struct wl_ctx_t wl_ctx_t; /* Opaque context type for managing memory pools */

extern WL_EXPORT wl_ctx_t* wl_ctx_create(const wl_ctx_info_t* info); /* Create context with configuration data. */
extern WL_EXPORT wl_ctx_t* wl_ctx_create2(size_t pool_chunk_size); /* Create context with just pool chunk size. */
extern WL_EXPORT void* wl_ctx_pool_alloc(wl_ctx_t* ctx, size_t size); /* Allocate memory from pool */
extern WL_EXPORT void* wl_ctx_pool_alloc_aligned(wl_ctx_t* ctx, size_t size, size_t align); /* Aligned memory allocation */
extern WL_EXPORT size_t wl_ctx_total_allocated_pool_memory(const wl_ctx_t* ctx); /* Get total allocated pool memory */
extern WL_EXPORT wl_exec_mode_t wl_ctx_get_exec_mode(const wl_ctx_t* ctx); /* Get execution mode */
extern WL_EXPORT void wl_ctx_set_exec_mode(wl_ctx_t* ctx, wl_exec_mode_t mode); /* Set execution mode */
extern WL_EXPORT wl_prng_algorithm_t wl_ctx_get_prng_algorithm(const wl_ctx_t* ctx); /* Get PRNG algorithm */
extern WL_EXPORT void wl_ctx_set_prng_algorithm(wl_ctx_t* ctx, wl_prng_algorithm_t algorithm, uint64_t seed); /* Set PRNG algorithm */
extern WL_EXPORT const char* wl_ctx_get_os_name(const wl_ctx_t* ctx); /* Get the name of the operating system */
extern WL_EXPORT const char* wl_ctx_get_cpu_name(const wl_ctx_t* ctx); /* Get the name of the CPU */
extern WL_EXPORT uint32_t wl_ctx_get_cpu_virtual_cores(const wl_ctx_t* ctx); /* Get the number of virtual cores */
extern WL_EXPORT uint32_t wl_ctx_get_cpu_physical_cores(const wl_ctx_t* ctx); /* Get the number of physical cores */
extern WL_EXPORT uint32_t wl_ctx_get_cpu_sockets(const wl_ctx_t* ctx); /* Get the number of CPU sockets */
extern WL_EXPORT uint64_t wl_ctx_get_physical_memory_total(const wl_ctx_t* ctx); /* Get the total physical memory in bytes */
extern WL_EXPORT uint64_t wl_ctx_get_physical_memory_free(const wl_ctx_t* ctx); /* Get the free physical memory in bytes */
extern WL_EXPORT bool wl_ctx_is_numa_system(const wl_ctx_t* ctx); /* Check if the system is NUMA */
extern WL_EXPORT void wl_ctx_destroy(wl_ctx_t* ctx); /* Destroy context and free memory */

typedef enum wl_dtype_t {
    WL_DTYPE_F32,   /* 32-bit floating-point data type */
    WL_DTYPE_COUNT_ /* Total number of data types */
} wl_dtype_t;
wl_static_assert(WL_DTYPE_COUNT_ <= 0xff);

typedef struct wl_dtype_info_t {
    int64_t size;         /* Size of the data type in bytes */
    const char* name;    /* Name of the data type */
} wl_dtype_info_t;
extern WL_EXPORT const wl_dtype_info_t* wl_dtype_info_of(wl_dtype_t type);

#define WL_SEP ,
#define wl_op_def(_, __) /* Enumerator | Mnemonic | Argcount */\
    _(NOP,              "nop",              0)/* No Operation */__\
    _(CLONE,            "clone",            1)/* R = clone(X) */__\
    _(VIEW,             "view",             1)/* R = X[:] */__\
    _(TRANSPOSE,        "transpose",        1)/* R = Xᵀ */__\
    _(PERMUTE,          "permute",          1)/* R = permute(X, axes) */__\
    _(MEAN,             "mean",             1)/* R = ΣX/n */__\
    _(SUM,              "sum",              1)/* R = ΣX */__\
    _(ABS,              "abs",              1)/* R = |X| */__\
    _(NEG,              "neg",              1)/* R = -X */__\
    _(LOG,              "log",              1)/* R = log X */__\
    _(SQR,              "sqr",              1)/* R = X² */__\
    _(SQRT,             "sqrt",             1)/* R = √X */__\
    _(SIN,              "sin",              1)/* R = sin X */__\
    _(COS,              "cos",              1)/* R = cos X */__\
    _(STEP,             "step",             1)/* R = step(X) */__\
    _(SOFTMAX,          "softmax'",         1)/* R = softmax(X) */__\
    _(SOFTMAX_DV,       "softmax'",         1)/* R = softmax'(X) */__\
    _(SIGMOID,          "sigmoid",          1)/* R = sigmoid(X) */__\
    _(SIGMOID_DV,       "sigmoid'",         1)/* R = sigmoid'(X) */__\
    _(HARD_SIGMOID,     "hard_sigmoid",     1)/* R = hard_sigmoid(X) */__\
    _(SILU,             "SiLU",             1)/* R = silu(X) */__\
    _(SILU_DV,          "SiLU'",            1)/* R = silu'(X) */__\
    _(TANH,             "tanh",             1)/* R = tanh(X) */__\
    _(TANH_DV,          "tanh'",            1)/* R = tanh'(X) */__\
    _(RELU,             "ReLU",             1)/* R = relu(X) */__\
    _(RELU_DV,          "ReLU'",            1)/* R = relu'(X) */__\
    _(GELU,             "GeLU",             1)/* R = gelu(X) */__\
    _(GELU_DV,          "GeLU'",            1)/* R = gelu'(X) */__\
    _(ADD,              "+",                2)/* R = X+Y */__\
    _(SUB,              "-",                2)/* R = X-Y */__\
    _(MUL,              "*",                2)/* R = X*Y (Hadamard product) */__\
    _(DIV,              "/",                2)/* R = X/Y */__\
    _(MATMUL,           "@",                2)/* R = A@B */__

#define _(enumerator, mnemonic, argcount) WL_OP_##enumerator
typedef enum wl_op_t {
    wl_op_def(_, WL_SEP)
    WL_OP__COUNT
} wl_op_t;
#undef _
wl_static_assert(WL_OP_NOP == 0);
wl_static_assert(WL_OP_MATMUL+1 == WL_OP__COUNT);
wl_static_assert(WL_OP__COUNT <= 0xff);
extern WL_EXPORT const char* wl_op_get_name(wl_op_t op);
extern WL_EXPORT const char* wl_op_get_mnemonic(wl_op_t op);
extern WL_EXPORT uint8_t wl_op_get_argcount(wl_op_t op);
#define wl_op_is_unary(op) (wl_op_get_argcount(op) == 1)
#define wl_op_is_binary(op) (wl_op_get_argcount(op) == 2)

typedef enum wl_op_param_type_t {     /* 2-bit Parameter type tag for operation parameter. */
    WL_OP_PARAM_FLOAT = 0,            /* 32-bit floating-point value */
    WL_OP_PARAM_INT = 1,              /* 32-bit signed/unsigned integer */
} wl_op_param_type_t;

/*
** Operation parameter. Each operation CAN have up to WL_MAX_OP_PARAMS of those parameters.
** 2-bit discriminator/tag and 62-bit value. (Tag and value are packed into a single 64-bit integer and both truncated to their bit width.)
** Not to be confused with operation inputs which are tensors (e.g. A + B <- here are A and B input tensors). Instead, this is for operation-specific parameters.
*/
typedef uint64_t wl_op_param_t;
wl_static_assert(sizeof(wl_op_param_t) == 8);
extern WL_EXPORT wl_op_param_t wl_op_param_int(uint64_t x); /* Create an integer parameter */
extern WL_EXPORT bool wl_op_param_is_int(wl_op_param_t param); /* Check if parameter is integer */
extern WL_EXPORT uint64_t wl_op_param_unpack_int(wl_op_param_t param); /* Get integer value from parameter */

extern WL_EXPORT uint32_t wl_pack_color_u8(uint8_t r, uint8_t g, uint8_t b);
extern WL_EXPORT uint32_t wl_pack_color_f32(float r, float g, float b);

typedef enum wl_graph_eval_order_t {
    WL_GRAPH_EVAL_ORDER_FORWARD = 0, /* Evaluate graph from left to right */
    WL_GRAPH_EVAL_ORDER_REVERSE = 1 /* Evaluate graph from right to left */
} wl_graph_eval_order_t;

typedef struct wl_tensor_t wl_tensor_t; /* Opaque type representing a tensor */

extern WL_EXPORT wl_tensor_t* wl_tensor_create_1d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1); /* Create 1D tensor */
extern WL_EXPORT wl_tensor_t* wl_tensor_create_2d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1, int64_t d2); /* Create 2D tensor */
extern WL_EXPORT wl_tensor_t* wl_tensor_create_3d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1, int64_t d2, int64_t d3); /* Create 3D tensor */
extern WL_EXPORT wl_tensor_t* wl_tensor_create_4d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4); /* Create 4D tensor */
extern WL_EXPORT wl_tensor_t* wl_tensor_create_5d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5); /* Create 5D tensor */
extern WL_EXPORT wl_tensor_t* wl_tensor_create_6d(wl_ctx_t* ctx, wl_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6); /* Create 6D tensor */

extern WL_EXPORT wl_tensor_t* wl_tensor_operator(wl_ctx_t* ctx, wl_op_t op, wl_tensor_t** inputs, uint32_t n_inputs, const wl_op_param_t(*params)[WL_MAX_OP_PARAMS]); /* Set opcode and arguments for tensor, and return result computation node. Returns NULL on failure. */

extern WL_EXPORT void wl_tensor_copy_buffer_from(wl_tensor_t* t, const void* data, size_t size); /* Copy data into tensor buffer */
extern WL_EXPORT void wl_tensor_fill(wl_tensor_t* t, float x); /* Set all tensor elements to a specific value */
extern WL_EXPORT void wl_tensor_fill_random(wl_tensor_t* t, float min, float max); /* Fill tensor with random values within [min, max] */

extern WL_EXPORT size_t wl_tensor_get_memory_usage(const wl_tensor_t* t); /* Return memory used by this tensor in bytes. */
extern WL_EXPORT void wl_tensor_print(const wl_tensor_t* t, bool with_header, bool with_data); /* Print tensor info (with or without data) */
extern WL_EXPORT void wl_tensor_set_name(wl_tensor_t* t, const char* name); /* Set the name of the tensor */
extern WL_EXPORT void wl_tensor_fmt_name(wl_tensor_t* t, const char* fmt, ...); /* Format the name of the tensor */
extern WL_EXPORT const char* wl_tensor_get_name(const wl_tensor_t* t); /* Get the name of the tensor */
extern WL_EXPORT int64_t wl_tensor_rank(const wl_tensor_t* t); /* Get the rank (number of dimensions) of the tensor */
extern WL_EXPORT const int64_t* wl_tensor_shape(const wl_tensor_t* t); /* Get the dimensions of the tensor */
extern WL_EXPORT const int64_t* wl_tensor_strides(const wl_tensor_t* t); /* Get the strides of the tensor */
extern WL_EXPORT wl_dtype_t wl_tensor_dtype(const wl_tensor_t* t); /* Get the data type of the tensor */
extern WL_EXPORT void* wl_tensor_data(const wl_tensor_t* t); /* Get the tensor buffer pointer */
extern WL_EXPORT float* wl_tensor_data_as_f32(const wl_tensor_t* t); /* Get the tensor buffer pointer as float pointer. Only valid if tensor's dtype is f32, else panics. */
extern WL_EXPORT int64_t wl_tensor_data_size(const wl_tensor_t* t); /* Get the size of the tensor buffer in bytes. */
extern WL_EXPORT int64_t wl_tensor_num_elements(const wl_tensor_t* t); /* Get the total amount of elements in the tensor. */
extern WL_EXPORT int64_t wl_tensor_num_rows(const wl_tensor_t* t); /* Get the number of rows (for 2D tensors) */
extern WL_EXPORT int64_t wl_tensor_num_cols(const wl_tensor_t* t); /* Get the number of columns (for 2D tensors) */
extern WL_EXPORT bool wl_tensor_is_scalar(const wl_tensor_t* t); /* Check if the tensor is a scalar */
extern WL_EXPORT bool wl_tensor_is_vector(const wl_tensor_t* t); /* Check if the tensor is a vector */
extern WL_EXPORT bool wl_tensor_is_matrix(const wl_tensor_t* t); /* Check if the tensor is a matrix */
extern WL_EXPORT bool wl_tensor_is_volume(const wl_tensor_t* t); /* Check if the tensor is higher-order (3D or more) */
extern WL_EXPORT bool wl_tensor_is_shape_eq(const wl_tensor_t* a, const wl_tensor_t* b); /* Checks if a and b have the same shape. */
extern WL_EXPORT bool wl_tensor_are_strides_eq(const wl_tensor_t* a, const wl_tensor_t* b); /* Checks if a and b have the same strides. */
extern WL_EXPORT bool wl_tensor_can_broadcast(const wl_tensor_t* a, const wl_tensor_t* b); /* Checks if b can be broadcasted into a. */
extern WL_EXPORT bool wl_tensor_is_transposed(const wl_tensor_t* t); /* Check if the tensor is transposed */
extern WL_EXPORT bool wl_tensor_is_permuted(const wl_tensor_t* t); /* Check if the tensor is permuted */
extern WL_EXPORT bool wl_tensor_is_contiguous(const wl_tensor_t* t); /* Check if the tensor memory is contiguous */
extern WL_EXPORT float wl_tensor_get_scalar_physical_index(const wl_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5); /* Get scalar value at physical index */
extern WL_EXPORT void wl_tensor_set_scalar_physical_index(wl_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, float x); /* Set scalar value at physical index */
extern WL_EXPORT float wl_tensor_get_scalar_virtual_index(const wl_tensor_t* t, int64_t v_idx); /* Get scalar value at virtual index */
extern WL_EXPORT void wl_tensor_set_scalar_virtual_index(wl_tensor_t* t, int64_t v_idx, float x); /* Set scalar value at virtual index */
extern WL_EXPORT bool wl_tensor_eq(const wl_tensor_t* a, const wl_tensor_t* b); /* Check if two tensors are equal without epsilon. */
extern WL_EXPORT bool wl_tensor_is_close(const wl_tensor_t* a, const wl_tensor_t* b, float eps, double* percent_eq); /* Check if two tensors are equal with epsilon and percentage in equality. Set eps to < 0 to use machine epsilon. */
extern WL_EXPORT void wl_tensor_img_draw_box(wl_tensor_t* t, uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, uint32_t wi, uint32_t rgb);
extern WL_EXPORT wl_ctx_t* wl_tensor_get_ctx(const wl_tensor_t* t); /* Get the context of the tensor */
extern WL_EXPORT void* wl_tensor_get_user_data(const wl_tensor_t* t); /* Get the user data of the tensor */
extern WL_EXPORT void wl_tensor_set_user_data(wl_tensor_t* t, void* ud); /* Set the user data of the tensor */
extern WL_EXPORT void wl_tensor_save(const wl_tensor_t* t, const char* file); /* Save tensor to WAVELET binary file. */
extern WL_EXPORT wl_tensor_t* wl_tensor_load(wl_ctx_t* ctx, const char* file); /* Load tensor from WAVELET binary file. */
extern WL_EXPORT wl_tensor_t* wl_tensor_load_image(wl_ctx_t* ctx, const char* file, wl_color_channels_t channels, uint32_t resize_w, uint32_t resize_h); /* Create a tensor from an image file. */
extern WL_EXPORT void wl_tensor_save_image(const wl_tensor_t* t, const char* file); /* Save tensor data as an image */
#define wl_tensor_image_width(tensor) (wl_tensor_shape(tensor)[2]) /* Get image width from tensor */
#define wl_tensor_image_height(tensor) (wl_tensor_shape(tensor)[1]) /* Get image height from tensor */
#define wl_tensor_image_channels(tensor) (wl_tensor_shape(tensor)[0]) /* Get image channels from tensor */

typedef struct wl_compute_graph_t wl_compute_graph_t; /* Opaque type representing a compute graph */
extern WL_EXPORT wl_compute_graph_t* wl_compute_graph_compile(wl_ctx_t* ctx, wl_tensor_t* root, wl_graph_eval_order_t order, const char* name); /* Compile computation graph from root tensor. */
extern WL_EXPORT wl_tensor_t* wl_compute_graph_execute(wl_compute_graph_t* gra); /* Execute computation graph. */
extern WL_EXPORT bool wl_compute_graph_contains(const wl_compute_graph_t* gra, const wl_tensor_t* t); /* Check if the tensor is in the compute graph */
extern WL_EXPORT void wl_compute_graph_dump_to_dot(const wl_compute_graph_t* gra, const char* file_name); /* Dump computation graph to DOT file. */
extern WL_EXPORT wl_ctx_t* wl_compute_graph_get_ctx(const wl_compute_graph_t* gra); /* Get the context of the compute graph */
extern WL_EXPORT const char* wl_compute_graph_get_name(const wl_compute_graph_t* gra); /* Get the name of the compute graph */
extern WL_EXPORT const wl_tensor_t** wl_compute_graph_get_internal_nodes(const wl_compute_graph_t* gra, size_t* n_nodes); /* Get the nodes of the compute graph */
extern WL_EXPORT const wl_tensor_t** wl_compute_graph_get_leaf_nodes(const wl_compute_graph_t* gra, size_t* n_leaves); /* Get the leaves of the compute graph */
extern WL_EXPORT size_t wl_compute_graph_get_num_total_nodes(const wl_compute_graph_t* gra); /* Get the total number of nodes in the compute graph */
extern WL_EXPORT size_t wl_compute_graph_get_num_internal_nodes(const wl_compute_graph_t* graph); /* Get the number of internal nodes (non-leaf) in the compute graph */
extern WL_EXPORT size_t wl_compute_graph_get_num_leaf_nodes(const wl_compute_graph_t* gra); /* Get the total number of leaf in the compute graph */
extern WL_EXPORT size_t wl_compute_graph_get_order(const wl_compute_graph_t* gra); /* Get the evaluation order of the compute graph */
extern WL_EXPORT size_t wl_compute_graph_get_memory_usage(const wl_compute_graph_t* gra); /* Get the memory usage of the compute graph */

#ifdef __cplusplus
}
#endif
#endif
