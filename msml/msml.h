/*
** (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
*/

#ifndef MSML_INCLUDE_MSML_H
#define MSML_INCLUDE_MSML_H

/* Compile time config macros */
#define MSML_CFG_X86_64_FAST_MATH 1 /* Use fast math for x86_64 by setting mxcsr control register. */
#define MSML_INTRIN 1 /* Use platform and compiler specific intrinsics for performance. */
#define MSML_BOUNDS_CHECK 1 /* Enable bounds checking for BLAS routines. */

#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>

#define MSML_DEFAULT_CHUNK_SIZE (1ull<<30)  /* Default size of memory chunk in bytes. 1 GiB */
#define MSML_DEFAULT_CHUNK_CAP 128          /* Default capacity of memory chunk */
#define MSML_MAX_DIMS 4                     /* Maximum number of dimensions for a tensor */
#define MSML_MAX_TENSOR_NAME_LEN 64         /* Maximum length for tensor name */
#define MSML_MAX_INPUT_TENSORS 2            /* Maximum number of input tensors for an operation */
#define MSML_MAX_OP_PARAMS 4                /* Maximum number of parameters for an operation */

#ifndef MSML_EXPORT
#   ifdef MSML_EXPORT_DLL
#       ifdef _MSC_VER
#           define MSML_EXPORT __declspec(dllexport)
#       else
#           define MSML_EXPORT __attribute__((visibility("default")))
#       endif
#   else
#       define MSML_EXPORT
#   endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define msml_version_pack(major, minor) ((uint32_t)((((major)&0xff)<<8)+((minor)&0xff)))
#define msml_version_major(version) (((version)>>8)&0xff)
#define msml_version_minor(version) ((version)&0xff)
#define MSML_VERSION msml_version_pack(0, 1) /* MSML library version. */
#define MSML_STORAGE_VERSION 1 /* MSML tensor storage file format version. */

#define msml_assert_name2(name, line) name ## line
#define msml_assert_name(line) msml_assert_name2(_assert_, line)
#define msml_static_assert(expr) extern void msml_assert_name(__LINE__)(bool STATIC_ASSERTION_FAILED[((expr)?1:-1)])

extern MSML_EXPORT void* msml_default_allocator_impl(void* blk, size_t size); /* Default memory allocator */
#ifndef msml_alloc /* Default allocator, can be overridden by defining it before.  */
#define msml_alloc msml_default_allocator_impl
#endif

typedef enum msml_exec_mode_t {
    MSML_EXEC_MODE_EAGER = 0, /* Execute operations immediately. (Dynamic computation graph, like PyTorch). */
    MSML_EXEC_MODE_DEFERRED = 1 /* Build computation graph and execute later. (Static computation graph, like TensorFlow 1.0). */
} msml_exec_mode_t;

typedef enum msml_prng_algorithm_t {
    MSML_PRNG_MERSENNE_TWISTER = 0, /* Mersenne Twister PRNG */
    MSML_PRNG_PCG = 1 /* Permuted Congruential Generator PRNG */
} msml_prng_algorithm_t;

typedef struct msml_ctx_info_t {
    void* (*alloc_fn)(void* blk, size_t size);  /* Custom allocator function */
    size_t pool_chunk_size;                     /* Size of each memory pool chunk */
    size_t pool_chunks_cap;                     /* Maximum chunks in the pool */
    uint64_t prng_seed;                         /* Seed for PRNG if prng_init_seed == true */
    bool warmup_chunks;                         /* If true, fresh pool chunks are filled to allocate kernel pages, can improve performance depending on scenario. */
    msml_prng_algorithm_t prng_algorithm;       /* PRNG algorithm */
    msml_exec_mode_t exec_mode;                 /* Default context execution mode */
    void* user_data;                            /* User-defined data */
} msml_ctx_info_t;

typedef struct msml_ctx_t msml_ctx_t; /* Opaque context type for managing memory pools */

extern MSML_EXPORT msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info); /* Create context with configuration data. */
extern MSML_EXPORT msml_ctx_t* msml_ctx_create2(size_t pool_chunk_size); /* Create context with just pool chunk size. */
extern MSML_EXPORT void* msml_ctx_pool_alloc(msml_ctx_t* ctx, size_t size); /* Allocate memory from pool */
extern MSML_EXPORT void* msml_ctx_pool_alloc_aligned(msml_ctx_t* ctx, size_t size, size_t align); /* Aligned memory allocation */
extern MSML_EXPORT size_t msml_ctx_total_allocated_pool_memory(const msml_ctx_t* ctx); /* Get total allocated pool memory */
extern MSML_EXPORT msml_exec_mode_t msml_ctx_get_exec_mode(const msml_ctx_t* ctx); /* Get execution mode */
extern MSML_EXPORT void msml_ctx_set_exec_mode(msml_ctx_t* ctx, msml_exec_mode_t mode); /* Set execution mode */
extern MSML_EXPORT msml_prng_algorithm_t msml_ctx_get_prng_algorithm(const msml_ctx_t* ctx); /* Get PRNG algorithm */
extern MSML_EXPORT void msml_ctx_set_prng_algorithm(msml_ctx_t* ctx, msml_prng_algorithm_t algorithm, uint64_t seed); /* Set PRNG algorithm */
extern MSML_EXPORT const char* msml_ctx_get_os_name(const msml_ctx_t* ctx); /* Get the name of the operating system */
extern MSML_EXPORT const char* msml_ctx_get_cpu_name(const msml_ctx_t* ctx); /* Get the name of the CPU */
extern MSML_EXPORT uint32_t msml_ctx_get_cpu_virtual_cores(const msml_ctx_t* ctx); /* Get the number of virtual cores */
extern MSML_EXPORT uint32_t msml_ctx_get_cpu_physical_cores(const msml_ctx_t* ctx); /* Get the number of physical cores */
extern MSML_EXPORT uint32_t msml_ctx_get_cpu_sockets(const msml_ctx_t* ctx); /* Get the number of CPU sockets */
extern MSML_EXPORT uint64_t msml_ctx_get_physical_memory_total(const msml_ctx_t* ctx); /* Get the total physical memory in bytes */
extern MSML_EXPORT uint64_t msml_ctx_get_physical_memory_free(const msml_ctx_t* ctx); /* Get the free physical memory in bytes */
extern MSML_EXPORT bool msml_ctx_is_numa_system(const msml_ctx_t* ctx); /* Check if the system is NUMA */
extern MSML_EXPORT void msml_ctx_destroy(msml_ctx_t* ctx); /* Destroy context and free memory */

typedef enum msml_dtype_t {
    MSML_DTYPE_F32,   /* 32-bit floating-point data type */
    MSML_DTYPE_COUNT_ /* Total number of data types */
} msml_dtype_t;
msml_static_assert(MSML_DTYPE_COUNT_ <= 0xff);

typedef struct msml_dtype_info_t {
    int64_t size;         /* Size of the data type in bytes */
    const char* name;    /* Name of the data type */
} msml_dtype_info_t;
extern MSML_EXPORT const msml_dtype_info_t* msml_dtype_info_of(msml_dtype_t type);

typedef enum msml_desired_color_channels_t {
    MSML_COLOR_CHANNELS_AUTO,  /* Automatically detect number of color channels */
    MSML_COLOR_CHANNELS_GRAY,  /* Grayscale F32 */
    MSML_COLOR_CHANNELS_GRAY_A,/* Grayscale F32 + Alpha F32 */
    MSML_COLOR_CHANNELS_RGB,   /* R32G32B32 */
    MSML_COLOR_CHANNELS_RGBA   /* R32G32B32A32 */
} msml_desired_color_channels_t;

#define MSML_SEP ,
#define msml_op_def(_, __) /* Enumerator | Mnemonic | Argcount */\
    _(NOP,              "nop",              0)/* No Operation. */__\
    _(CLONE,            "clone",            1)/* R = clone(X). */__\
    _(VIEW,             "view",             1)/* R = X[:]. */__\
    _(TRANSPOSE,        "transpose",        1)/* R = Xᵀ. */__\
    _(PERMUTE,          "permute",          1)/* R = permute(X). */__\
    _(STEP,             "step",             1)/* R = 1 if x >= 0 else 0. Heaviside step function */__\
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
    _(ADD,              "+",                2)/* R = relu(X) */__\
    _(SUB,              "-",                2)/* R = X-Y */__\
    _(MUL,              "*",                2)/* R = X*Y (Hadamard product) */__\
    _(DIV,              "/",                2)/* R = X/Y. */__\
    _(MATMUL,           "@",                2)/* R = A x B.*/__

#define _(enumerator, mnemonic, argcount) MSML_OP_##enumerator
typedef enum msml_op_t {
    msml_op_def(_, MSML_SEP)
    MSML_OP__COUNT
} msml_op_t;
#undef _
msml_static_assert(MSML_OP_NOP == 0);
msml_static_assert(MSML_OP_MATMUL+1 == MSML_OP__COUNT);
msml_static_assert(MSML_OP__COUNT <= 0xff);
extern MSML_EXPORT const char* msml_op_get_name(msml_op_t op);
extern MSML_EXPORT const char* msml_op_get_mnemonic(msml_op_t op);
extern MSML_EXPORT uint8_t msml_op_get_argcount(msml_op_t op);
#define msml_op_is_unary(op) (msml_op_get_argcount(op) == 1)
#define msml_op_is_binary(op) (msml_op_get_argcount(op) == 2)

typedef enum msml_op_param_type_t {     /* 2-bit Parameter type tag for operation parameter. */
    MSML_OP_PARAM_FLOAT = 0,            /* 32-bit floating-point value */
    MSML_OP_PARAM_INT = 1,              /* 32-bit signed/unsigned integer */
} msml_op_param_type_t;

/*
** Operation parameter. Each operation CAN have up to MSML_MAX_OP_PARAMS of those parameters.
** 2-bit discriminator/tag and 62-bit value. (Tag and value are packed into a single 64-bit integer and both truncated to their bit width.)
** Not to be confused with operation inputs which are tensors (e.g. A + B <- here are A and B input tensors). Instead, this is for operation-specific parameters.
*/
typedef uint64_t msml_op_param_t;
msml_static_assert(sizeof(msml_op_param_t) == 8);
extern MSML_EXPORT msml_op_param_t msml_op_param_int(uint64_t x); /* Create an integer parameter */
extern MSML_EXPORT bool msml_op_param_is_int(msml_op_param_t param); /* Check if parameter is integer */
extern MSML_EXPORT uint64_t msml_op_param_unpack_int(msml_op_param_t param); /* Get integer value from parameter */

typedef enum msml_graph_eval_order_t {
    MSML_GRAPH_EVAL_ORDER_FORWARD = 0, /* Evaluate graph from left to right */
    MSML_GRAPH_EVAL_ORDER_REVERSE = 1 /* Evaluate graph from right to left */
} msml_graph_eval_order_t;

typedef struct msml_tensor_t msml_tensor_t; /* Opaque type representing a tensor */

extern MSML_EXPORT msml_tensor_t* msml_tensor_create_1d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1); /* Create 1D tensor */
extern MSML_EXPORT msml_tensor_t* msml_tensor_create_2d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2); /* Create 2D tensor */
extern MSML_EXPORT msml_tensor_t* msml_tensor_create_3d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3); /* Create 3D tensor */
extern MSML_EXPORT msml_tensor_t* msml_tensor_create_4d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4); /* Create 4D tensor */

extern MSML_EXPORT msml_tensor_t* msml_tensor_operator(msml_ctx_t* ctx, msml_op_t op, msml_tensor_t** inputs, uint32_t n_inputs, const msml_op_param_t(*params)[MSML_MAX_OP_PARAMS]); /* Set opcode and arguments for tensor, and return result computation node. Returns NULL on failure. */

extern MSML_EXPORT void msml_tensor_copy_buffer_from(msml_tensor_t* t, const void* data, size_t size); /* Copy data into tensor buffer */
extern MSML_EXPORT void msml_tensor_fill(msml_tensor_t* t, float x); /* Set all tensor elements to a specific value */
extern MSML_EXPORT void msml_tensor_fill_random(msml_tensor_t* t, float min, float max); /* Fill tensor with random values within [min, max] */

extern MSML_EXPORT size_t msml_tensor_get_memory_usage(const msml_tensor_t* t); /* Return memory used by this tensor in bytes. */
extern MSML_EXPORT void msml_tensor_print(const msml_tensor_t* t, bool with_header, bool with_data); /* Print tensor info (with or without data) */
extern MSML_EXPORT void msml_tensor_set_name(msml_tensor_t* t, const char* name); /* Set the name of the tensor */
extern MSML_EXPORT void msml_tensor_fmt_name(msml_tensor_t* t, const char* fmt, ...); /* Format the name of the tensor */
extern MSML_EXPORT const char* msml_tensor_get_name(const msml_tensor_t* t); /* Get the name of the tensor */
extern MSML_EXPORT int64_t msml_tensor_rank(const msml_tensor_t* t); /* Get the rank (number of dimensions) of the tensor */
extern MSML_EXPORT const int64_t* msml_tensor_shape(const msml_tensor_t* t); /* Get the dimensions of the tensor */
extern MSML_EXPORT const int64_t* msml_tensor_strides(const msml_tensor_t* t); /* Get the strides of the tensor */
extern MSML_EXPORT msml_dtype_t msml_tensor_dtype(const msml_tensor_t* t); /* Get the data type of the tensor */
extern MSML_EXPORT void* msml_tensor_data(const msml_tensor_t* t); /* Get the tensor buffer pointer */
extern MSML_EXPORT float* msml_tensor_data_as_f32(const msml_tensor_t* t); /* Get the tensor buffer pointer as float pointer. Only valid if tensor's dtype is f32, else panics. */
extern MSML_EXPORT int64_t msml_tensor_data_size(const msml_tensor_t* t); /* Get the size of the tensor buffer in bytes. */
extern MSML_EXPORT int64_t msml_tensor_num_elements(const msml_tensor_t* t); /* Get the total amount of elements in the tensor. */
extern MSML_EXPORT int64_t msml_tensor_num_rows(const msml_tensor_t* t); /* Get the number of rows (for 2D tensors) */
extern MSML_EXPORT int64_t msml_tensor_num_cols(const msml_tensor_t* t); /* Get the number of columns (for 2D tensors) */
extern MSML_EXPORT bool msml_tensor_is_scalar(const msml_tensor_t* t); /* Check if the tensor is a scalar */
extern MSML_EXPORT bool msml_tensor_is_vector(const msml_tensor_t* t); /* Check if the tensor is a vector */
extern MSML_EXPORT bool msml_tensor_is_matrix(const msml_tensor_t* t); /* Check if the tensor is a matrix */
extern MSML_EXPORT bool msml_tensor_is_volume(const msml_tensor_t* t); /* Check if the tensor is higher-order (3D or more) */
extern MSML_EXPORT bool msml_tensor_is_shape_eq(const msml_tensor_t* a, const msml_tensor_t* b); /* Checks if a and b have the same shape. */
extern MSML_EXPORT bool msml_tensor_are_strides_eq(const msml_tensor_t* a, const msml_tensor_t* b); /* Checks if a and b have the same strides. */
extern MSML_EXPORT bool msml_tensor_can_broadcast(const msml_tensor_t* a, const msml_tensor_t* b); /* Checks if b can be broadcasted into a. */
extern MSML_EXPORT bool msml_tensor_is_transposed(const msml_tensor_t* t); /* Check if the tensor is transposed */
extern MSML_EXPORT bool msml_tensor_is_permuted(const msml_tensor_t* t); /* Check if the tensor is permuted */
extern MSML_EXPORT bool msml_tensor_is_contiguous(const msml_tensor_t* t); /* Check if the tensor memory is contiguous */
extern MSML_EXPORT float msml_tensor_get_scalar_physical_index(const msml_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3); /* Get scalar value at physical index */
extern MSML_EXPORT void msml_tensor_set_scalar_physical_index(msml_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, float x); /* Set scalar value at physical index */
extern MSML_EXPORT float msml_tensor_get_scalar_virtual_index(const msml_tensor_t* t, int64_t v_idx); /* Get scalar value at virtual index */
extern MSML_EXPORT void msml_tensor_set_scalar_virtual_index(msml_tensor_t* t, int64_t v_idx, float x); /* Set scalar value at virtual index */
extern MSML_EXPORT bool msml_tensor_eq(const msml_tensor_t* a, const msml_tensor_t* b); /* Check if two tensors are equal without epsilon. */
extern MSML_EXPORT bool msml_tensor_is_close(const msml_tensor_t* a, const msml_tensor_t* b, float eps, double* percent_eq); /* Check if two tensors are equal with epsilon and percentage in equality. Set eps to < 0 to use machine epsilon. */
extern MSML_EXPORT msml_ctx_t* msml_tensor_get_ctx(const msml_tensor_t* t); /* Get the context of the tensor */
extern MSML_EXPORT void* msml_tensor_get_user_data(const msml_tensor_t* t); /* Get the user data of the tensor */
extern MSML_EXPORT void msml_tensor_set_user_data(msml_tensor_t* t, void* ud); /* Set the user data of the tensor */

extern MSML_EXPORT void msml_tensor_save(const msml_tensor_t* t, const char* file_name); /* Save tensor to MSML binary file. */
extern MSML_EXPORT msml_tensor_t* msml_tensor_load(msml_ctx_t* ctx, const char* file_name); /* Load tensor from MSML binary file. */
extern MSML_EXPORT msml_tensor_t* msml_tensor_create_from_image(msml_ctx_t* ctx, const char* file_path, msml_desired_color_channels_t channels, uint32_t resize_width, uint32_t resize_height); /* Create a tensor from an image file */
extern MSML_EXPORT void msml_tensor_save_to_image(const msml_tensor_t* t, const char* file_path); /* Save tensor data as an image */
#define msml_tensor_image_width(tensor) (msml_tensor_shape(tensor)[0]) /* Get image width from tensor */
#define msml_tensor_image_height(tensor) (msml_tensor_shape(tensor)[1]) /* Get image height from tensor */
#define msml_tensor_image_channels(tensor) (msml_tensor_shape(tensor)[2]) /* Get image channels from tensor */

typedef struct msml_compute_graph_t msml_compute_graph_t; /* Opaque type representing a compute graph */
extern MSML_EXPORT msml_compute_graph_t* msml_compute_graph_compile(msml_ctx_t* ctx, msml_tensor_t* root, msml_graph_eval_order_t order, const char* name); /* Compile computation graph from root tensor. */
extern MSML_EXPORT msml_tensor_t* msml_compute_graph_execute(msml_compute_graph_t* gra); /* Execute computation graph. */
extern MSML_EXPORT bool msml_compute_graph_contains(const msml_compute_graph_t* gra, const msml_tensor_t* t); /* Check if the tensor is in the compute graph */
extern MSML_EXPORT void msml_compute_graph_dump_to_dot(const msml_compute_graph_t* gra, const char* file_name); /* Dump computation graph to DOT file. */
extern MSML_EXPORT msml_ctx_t* msml_compute_graph_get_ctx(const msml_compute_graph_t* gra); /* Get the context of the compute graph */
extern MSML_EXPORT const char* msml_compute_graph_get_name(const msml_compute_graph_t* gra); /* Get the name of the compute graph */
extern MSML_EXPORT const msml_tensor_t** msml_compute_graph_get_internal_nodes(const msml_compute_graph_t* gra, size_t* n_nodes); /* Get the nodes of the compute graph */
extern MSML_EXPORT const msml_tensor_t** msml_compute_graph_get_leaf_nodes(const msml_compute_graph_t* gra, size_t* n_leaves); /* Get the leaves of the compute graph */
extern MSML_EXPORT size_t msml_compute_graph_get_num_total_nodes(const msml_compute_graph_t* gra); /* Get the total number of nodes in the compute graph */
extern MSML_EXPORT size_t msml_compute_graph_get_num_internal_nodes(const msml_compute_graph_t* graph); /* Get the number of internal nodes (non-leaf) in the compute graph */
extern MSML_EXPORT size_t msml_compute_graph_get_num_leaf_nodes(const msml_compute_graph_t* gra); /* Get the total number of leaf in the compute graph */
extern MSML_EXPORT size_t msml_compute_graph_get_order(const msml_compute_graph_t* gra); /* Get the evaluation order of the compute graph */
extern MSML_EXPORT size_t msml_compute_graph_get_memory_usage(const msml_compute_graph_t* gra); /* Get the memory usage of the compute graph */

#ifdef __cplusplus
}
#endif
#endif
