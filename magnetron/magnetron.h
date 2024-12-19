/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#ifndef MAGNETRON_H
#define MAGNETRON_H

/* Compile time config macros */
#define MAG_CFG_X86_64_FAST_MATH 1 /* Use fast math for x86_64 by setting mxcsr control register. */
#define MAG_INTRIN 1 /* Use platform and compiler specific intrinsics for performance. */
#define MAG_BOUNDS_CHECK 0 /* Enable bounds checking for BLAS routines. */
#define MAG_SANITIZE_RC 0 /* Enable runtime checks for debugging of reference counted tensors. */
#define MAG_EXPORT_DLL

#if !defined(NDEBUG) && !MAG_BOUNDS_CHECK
#undef MAG_BOUNDS_CHECK
#define MAG_BOUNDS_CHECK 1
#endif

#include <string.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAG_DEFAULT_CHUNK_SIZE (1ull<<30)  /* Default size of memory chunk in bytes. 1 GiB */
#define MAG_DEFAULT_CHUNK_CAP 128          /* Default capacity of memory chunk */
#define MAG_MAX_DIMS 6                     /* Maximum number of dimensions for a tensor */
#define MAG_MAX_TENSOR_NAME_LEN 64         /* Maximum length for tensor name */
#define MAG_MAX_INPUT_TENSORS 2            /* Maximum number of input tensors for an operation */
#define MAG_MAX_OP_PARAMS 6                /* Maximum number of parameters for an operation */

#ifndef MAG_EXPORT
#   ifdef MAG_EXPORT_DLL
#       ifdef _MSC_VER
#           define MAG_EXPORT __declspec(dllexport)
#       else
#           define MAG_EXPORT __attribute__((visibility("default")))
#       endif
#   else
#       define MAG_EXPORT
#   endif
#endif

#define mag_version_pack(major, minor) ((uint32_t)((((major)&0xff)<<8)+((minor)&0xff)))
#define mag_version_major(version) (((version)>>8)&0xff)
#define mag_version_minor(version) ((version)&0xff)
#define MAG_VERSION mag_version_pack(0, 1) /* magnetron library version. */
#define MAG_STORAGE_VERSION 1 /* magnetron storage format version. */

#define mag_assert_name2(name, line) name ## line
#define mag_assert_name(line) mag_assert_name2(_assert_, line)
#define mag_static_assert(expr) extern void mag_assert_name(__LINE__)(bool STATIC_ASSERTION_FAILED[((expr)?1:-1)])

typedef enum mag_compute_device_type_t {
    MAG_COMPUTE_DEVICE_TYPE_CPU = 0, /* CPU compute device */
    MAG_COMPUTE_DEVICE_TYPE_GPU_CUDA = 1,  /* CUDA GPU compute device */

    MAG_COMPUTE_DEVICE_TYPE__NUM
} mag_compute_device_type_t;
extern MAG_EXPORT const char* mag_device_type_get_name(mag_compute_device_type_t op);

typedef enum mag_exec_mode_t {
    MAG_EXEC_MODE_EAGER = 0, /* Execute operations immediately. (Dynamic computation graph, like PyTorch). */
    MAG_EXEC_MODE_DEFERRED = 1, /* Build computation graph and execute later. (Static computation graph, like TensorFlow 1.0). */

    MAG_EXEC_MODE__NUM
} mag_exec_mode_t;

typedef enum mag_prng_algorithm_t {
    MAG_PRNG_MERSENNE_TWISTER = 0, /* Mersenne Twister PRNG */
    MAG_PRNG_PCG = 1, /* Permuted Congruential Generator PRNG */

    MAG_PRNG__NUM
} mag_prng_algorithm_t;

typedef enum mag_color_channels_t {
    MAG_COLOR_CHANNELS_AUTO,  /* Automatically detect number of color channels */
    MAG_COLOR_CHANNELS_GRAY,  /* Grayscale F32 */
    MAG_COLOR_CHANNELS_GRAY_A,/* Grayscale F32 + Alpha F32 */
    MAG_COLOR_CHANNELS_RGB,   /* R32G32B32 */
    MAG_COLOR_CHANNELS_RGBA,   /* R32G32B32A32 */

    MAG_COLOR_CHANNELS__NUM
} mag_color_channels_t;

extern MAG_EXPORT void* (*mag_get_alloc_fn(void))(void* blk, size_t size); /* Get global allocator. */
extern MAG_EXPORT void mag_set_alloc_fn(void* (*alloc)(void* blk, size_t size)); /* Set global allocator. */
extern MAG_EXPORT void mag_set_set_log_mode(bool enabled); /* Enable/disable logging. */

typedef uint32_t mag_char32_t;

typedef struct mag_ctx_info_t {
    mag_compute_device_type_t device;        /* Compute device */
    size_t pool_chunk_size;                 /* Size of each memory pool chunk */
    size_t pool_chunks_cap;                 /* Maximum chunks in the pool */
    uint64_t prng_seed;                     /* Seed for PRNG if prng_init_seed == true */
    bool warmup_chunks;                     /* If true, fresh pool chunks are filled to allocate kernel pages, can improve performance depending on scenario. */
    mag_prng_algorithm_t prng_algorithm;     /* PRNG algorithm */
    mag_exec_mode_t exec_mode;               /* Default context execution mode */
    uint8_t* (*image_load_fn)(const char*, uint32_t(*)[3], mag_color_channels_t);    /* Image raw data loader. stb_image by default, you can plug-in your own. */
    void (*image_load_free_fn)(uint8_t*);                                           /* Image raw data loader free function. stb_image by default, you can plug-in your own. */
    bool (*image_save_fn)(const char*, const uint8_t*, const uint32_t(*)[3]);       /* Image raw data saver. stb_image by default, you can plug-in your own. */
    void* user_data; /* User-defined data */
} mag_ctx_info_t;

typedef struct mag_ctx_t mag_ctx_t; /* Opaque context type for managing memory pools */

extern MAG_EXPORT mag_ctx_t* mag_ctx_create(const mag_ctx_info_t* info); /* Create context with configuration data. */
extern MAG_EXPORT mag_ctx_t* mag_ctx_create2(mag_compute_device_type_t device); /* Create context with default config, and only specificy device. */
extern MAG_EXPORT mag_exec_mode_t mag_ctx_get_exec_mode(const mag_ctx_t* ctx); /* Get execution mode */
extern MAG_EXPORT void mag_ctx_set_exec_mode(mag_ctx_t* ctx, mag_exec_mode_t mode); /* Set execution mode */
extern MAG_EXPORT mag_prng_algorithm_t mag_ctx_get_prng_algorithm(const mag_ctx_t* ctx); /* Get PRNG algorithm */
extern MAG_EXPORT void mag_ctx_set_prng_algorithm(mag_ctx_t* ctx, mag_prng_algorithm_t algorithm, uint64_t seed); /* Set PRNG algorithm */
extern MAG_EXPORT mag_compute_device_type_t mag_ctx_get_compute_device_type(const mag_ctx_t* ctx); /* Get compute device type */
extern MAG_EXPORT const char* mag_ctx_get_compute_device_name(const mag_ctx_t* ctx); /* Get the name of the compute device */
extern MAG_EXPORT const char* mag_ctx_get_os_name(const mag_ctx_t* ctx); /* Get the name of the operating system */
extern MAG_EXPORT const char* mag_ctx_get_cpu_name(const mag_ctx_t* ctx); /* Get the name of the CPU */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_virtual_cores(const mag_ctx_t* ctx); /* Get the number of virtual cores */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_physical_cores(const mag_ctx_t* ctx); /* Get the number of physical cores */
extern MAG_EXPORT uint32_t mag_ctx_get_cpu_sockets(const mag_ctx_t* ctx); /* Get the number of CPU sockets */
extern MAG_EXPORT uint64_t mag_ctx_get_physical_memory_total(const mag_ctx_t* ctx); /* Get the total physical memory in bytes */
extern MAG_EXPORT uint64_t mag_ctx_get_physical_memory_free(const mag_ctx_t* ctx); /* Get the free physical memory in bytes */
extern MAG_EXPORT bool mag_ctx_is_numa_system(const mag_ctx_t* ctx); /* Check if the system is NUMA */
extern MAG_EXPORT size_t mag_ctx_get_total_tensors_created(const mag_ctx_t* ctx); /* Get total tensors created. (Including views) */
extern MAG_EXPORT void mag_ctx_profile_start_recording(mag_ctx_t* ctx); /* Start profiling */
extern MAG_EXPORT void mag_ctx_profile_stop_recording(mag_ctx_t* ctx, const char* export_csv_file); /* Reset profiling data */
extern MAG_EXPORT void mag_ctx_destroy(mag_ctx_t* ctx); /* Destroy context and free memory */

typedef enum mag_dtype_t {
    MAG_DTYPE_F32,   /* 32-bit floating-point data type */
    MAG_DTYPE__NUM /* Total number of data types */
} mag_dtype_t;
mag_static_assert(MAG_DTYPE__NUM <= 0xff);

typedef struct mag_dtype_info_t {
    int64_t size;         /* Size of the data type in bytes */
    const char* name;    /* Name of the data type */
} mag_dtype_info_t;
extern MAG_EXPORT const mag_dtype_info_t* mag_dtype_info_of(mag_dtype_t type);

#define MAG_SEP ,
#define mag_op_def(_, __) /* Enumerator | Mnemonic | Argcount | Paramcount, Inplace Support */\
    _(NOP,              "nop",              0, 0, false)/* No Operation */__\
    _(CLONE,            "clone",            1, 0, false)/* R = clone(X) */__\
    _(VIEW,             "view",             1, 0, false)/* R = X[:] */__\
    _(TRANSPOSE,        "transpose",        1, 0, false)/* R = Xᵀ */__\
    _(PERMUTE,          "permute",          1, 6, false)/* R = permute(X, axes) */__\
    _(MEAN,             "mean",             1, 0, false)/* R = ΣX/n */__\
    _(MIN,              "min",              1, 0, false)/* R = min x */__\
    _(MAX,              "max",              1, 0, false)/* R = max x */__\
    _(SUM,              "sum",              1, 0, false)/* R = ΣX */__\
    _(ABS,              "abs",              1, 0, true)/* R = |X| */__\
    _(NEG,              "neg",              1, 0, true)/* R = -X */__\
    _(LOG,              "log",              1, 0, true)/* R = log X */__\
    _(SQR,              "sqr",              1, 0, true)/* R = X² */__\
    _(SQRT,             "sqrt",             1, 0, true)/* R = √X */__\
    _(SIN,              "sin",              1, 0, true)/* R = sin X */__\
    _(COS,              "cos",              1, 0, true)/* R = cos X */__\
    _(STEP,             "step",             1, 0, true)/* R = step(X) */__\
    _(SOFTMAX,          "softmax'",         1, 0, true)/* R = softmax(X) */__\
    _(SOFTMAX_DV,       "softmax'",         1, 0, true)/* R = softmax'(X) */__\
    _(SIGMOID,          "sigmoid",          1, 0, true)/* R = sigmoid(X) */__\
    _(SIGMOID_DV,       "sigmoid'",         1, 0, true)/* R = sigmoid'(X) */__\
    _(HARD_SIGMOID,     "hard_sigmoid",     1, 0, true)/* R = hard_sigmoid(X) */__\
    _(SILU,             "SiLU",             1, 0, true)/* R = silu(X) */__\
    _(SILU_DV,          "SiLU'",            1, 0, true)/* R = silu'(X) */__\
    _(TANH,             "tanh",             1, 0, true)/* R = tanh(X) */__\
    _(TANH_DV,          "tanh'",            1, 0, true)/* R = tanh'(X) */__\
    _(RELU,             "ReLU",             1, 0, true)/* R = relu(X) */__\
    _(RELU_DV,          "ReLU'",            1, 0, true)/* R = relu'(X) */__\
    _(GELU,             "GeLU",             1, 0, true)/* R = gelu(X) */__\
    _(GELU_DV,          "GeLU'",            1, 0, true)/* R = gelu'(X) */__\
    _(ADD,              "+",                2, 0, true) /* R = X+Y */__\
    _(SUB,              "-",                2, 0, true) /* R = X-Y */__\
    _(MUL,              "*",                2, 0, true) /* R = X*Y (Hadamard product) */__\
    _(DIV,              "/",                2, 0, true) /* R = X/Y */__\
    _(ADDS,             "+ξ",               1, 1, true) /* R = X+ξ */__\
    _(SUBS,             "-ξ",               1, 1, true) /* R = X-ξ */__\
    _(MULS,             "*ξ",               1, 1, true) /* R = X*ξ (Hadamard product) */__\
    _(DIVS,             "/ξ",               1, 1, true) /* R = X/ξ */__\
    _(MATMUL,           "@",                2, 0, true)/* R = A@B */__

#define _(enumerator, mnemonic, argcount, paramcount, inplace) MAG_OP_##enumerator
typedef enum mag_op_t {
    mag_op_def(_, MAG_SEP)
    MAG_OP__NUM
} mag_op_t;
#undef _
mag_static_assert(MAG_OP_NOP == 0);
mag_static_assert(MAG_OP_MATMUL+1 == MAG_OP__NUM);
mag_static_assert(MAG_OP__NUM <= 0xff);
extern MAG_EXPORT const char* mag_op_get_name(mag_op_t op);
extern MAG_EXPORT const char* mag_op_get_mnemonic(mag_op_t op);
extern MAG_EXPORT uint8_t mag_op_get_argcount(mag_op_t op);
extern MAG_EXPORT uint8_t mag_op_get_paramcount(mag_op_t op);
extern MAG_EXPORT bool mag_op_supports_inplace(mag_op_t op);
#define mag_op_is_unary(op) (mag_op_get_argcount(op) == 1)
#define mag_op_is_binary(op) (mag_op_get_argcount(op) == 2)

typedef enum mag_op_param_type_t {     /* 2-bit Parameter type tag for operation parameter. */
    MAG_OP_PARAM_FLOAT = 0,            /* 32-bit floating-point value */
    MAG_OP_PARAM_INT = 1,              /* 32-bit signed/unsigned integer */
} mag_op_param_type_t;

/*
** Operation parameter. Each operation CAN have up to MAG_MAX_OP_PARAMS of those parameters.
** 2-bit discriminator/tag and 62-bit value. (Tag and value are packed into a single 64-bit integer and both truncated to their bit width.)
** Not to be confused with operation inputs which are tensors (e.g. A + B <- here are A and B input tensors). Instead, this is for operation-specific parameters.
*/
typedef uint64_t mag_op_param_t;
mag_static_assert(sizeof(mag_op_param_t) == 8);
extern MAG_EXPORT mag_op_param_t mag_op_param_int(uint32_t x); /* Create an integer parameter */
extern MAG_EXPORT bool mag_op_param_is_int(mag_op_param_t param); /* Check if parameter is integer */
extern MAG_EXPORT uint32_t mag_op_param_unpack_int(mag_op_param_t param); /* Get integer value from parameter */
extern MAG_EXPORT mag_op_param_t mag_op_param_float(float x); /* Create an integer parameter */
extern MAG_EXPORT bool mag_op_param_is_float(mag_op_param_t param); /* Check if parameter is integer */
extern MAG_EXPORT float mag_op_param_unpack_float(mag_op_param_t param); /* Get integer value from parameter */

extern MAG_EXPORT uint32_t mag_pack_color_u8(uint8_t r, uint8_t g, uint8_t b);
extern MAG_EXPORT uint32_t mag_pack_color_f32(float r, float g, float b);

typedef enum mag_graph_eval_order_t {
    MAG_GRAPH_EVAL_ORDER_FORWARD = 0, /* Evaluate graph from left to right */
    MAG_GRAPH_EVAL_ORDER_REVERSE = 1 /* Evaluate graph from right to left */
} mag_graph_eval_order_t;

/**
 * @brief Multidimensional tensor of arbitrary rank and data type.
 *      The tensor is reference counted and can be shared between multiple tensors.
 *      Rule of Thumb for Reference Counting:
 *          - If you only use the reference temporarily and do not store it, no need to adjust the reference count.
 *          - If you store the reference (e.g., in a data structure), increase the reference count when storing and decrease it when removing.
 *      The rank is > 0 and <= MAG_MAX_DIMS. The shape of the tensor is an array of dimensions of size MAG_MAX_DIMS.
 *      Is a node in a static or dynamic computation graph, depending on the context execution mode.
 */
typedef struct mag_tensor_t mag_tensor_t;

/**
 * @brief Create a new 1-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* mag_tensor_create_1d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1);

/**
 * @brief Create a new 2-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @param d2 Size of the second dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* mag_tensor_create_2d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2);

/**
 * @brief Create a new 3-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @param d2 Size of the second dimension. Must be > 0 and < INT64_MAX.
 * @param d3 Size of the third dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* mag_tensor_create_3d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3);

/**
 * @brief Create a new 4-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @param d2 Size of the second dimension. Must be > 0 and < INT64_MAX.
 * @param d3 Size of the third dimension. Must be > 0 and < INT64_MAX.
 * @param d4 Size of the fourth dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* mag_tensor_create_4d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4);

/**
 * @brief Create a new 5-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @param d2 Size of the second dimension. Must be > 0 and < INT64_MAX.
 * @param d3 Size of the third dimension. Must be > 0 and < INT64_MAX.
 * @param d4 Size of the fourth dimension. Must be > 0 and < INT64_MAX.
 * @param d5 Size of the fifth dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* mag_tensor_create_5d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5);

/**
 * @brief Create a new 6-dimensional tensor.
 *      Data is uninitialized, should be filled with values before using it.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param type Data type of the tensor. Must be a valid mag_dtype_t.
 * @param d1 Size of the first dimension. Must be > 0 and < INT64_MAX.
 * @param d2 Size of the second dimension. Must be > 0 and < INT64_MAX.
 * @param d3 Size of the third dimension. Must be > 0 and < INT64_MAX.
 * @param d4 Size of the fourth dimension. Must be > 0 and < INT64_MAX.
 * @param d5 Size of the fifth dimension. Must be > 0 and < INT64_MAX.
 * @param d6 Size of the sixth dimension. Must be > 0 and < INT64_MAX.
 * @returns New tensor. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* mag_tensor_create_6d(mag_ctx_t* ctx, mag_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, int64_t d6);


/**
 * @brief Emit operation with variable arguments.
 * Constructs and returns a result tensor R by applying an operation to input tensors. R = op(X, Y, ...).
 * If the execution mode is MAG_EXEC_MODE_EAGER, the operation is executed immediately, and R contains the computed result.
 * If the execution mode is MAG_EXEC_MODE_DEFERRED, the operation is added to the computation graph, and R is a placeholder tensor.
 * @param ctx Context to create the tensor in. Must not be NULL.
 * @param op Operation code (MAG_OP_*)
 * @param inplace If true, the operation is applied in-place, and the result is stored in the first input tensor. Otherwise, a new tensor is created.
 *          With inplace=true, R = op(X, Y) is equivalent to X = op(X, Y).
 *          Reduces memory usage and improves performance, because no new tensor is created.
 *          If the operation does not support in-place execution, the function creates a new tensor like with inplace=false.
 *          Example: R = X + Y turns into X += Y with inplace=true.
 * @param inputs Pointer to array of input tensors of size n_inputs. Must not be NULL.
 * @param n_inputs Number of input tensors, must be equal to the operation's argument count, if not, the function panics.
 * @param params Array of operation parameters. Can be NULL if the operation does not require any parameters.
 * @returns Result tensor R. Is never NULL.
 */
extern MAG_EXPORT mag_tensor_t* mag_tensor_operator(mag_ctx_t* ctx, mag_op_t op, bool inplace, mag_tensor_t** inputs, uint32_t n_inputs, const mag_op_param_t(*params)[MAG_MAX_OP_PARAMS]);

/**
 * @brief Increment reference count of tensor.
 *      Increment the strong reference count of the tensor. The tensor is not destroyed until the strong reference count reaches zero.
 *      Rule of Thumb for Reference Counting:
 *      - If you only use the reference temporarily and do not store it, no need to adjust the reference count.
 *      - If you store the reference (e.g., in a data structure), increase the reference count when storing and decrease it when removing.
 * @param t Tensor. Must not be NULL.
 */
extern MAG_EXPORT void mag_tensor_incref(mag_tensor_t* t);

/**
 * @brief Decrement reference count of tensor.
 *      Decrement the strong reference count of the tensor. The tensor is destroyed when the strong reference count reaches zero.
 *      Rule of Thumb for Reference Counting:
 *      - If you only use the reference temporarily and do not store it, no need to adjust the reference count.
 *      - If you store the reference (e.g., in a data structure), increase the reference count when storing and decrease it when removing.
 * @param t Tensor. Must not be NULL.
 * @returns True if the tensor was destroyed, false if the tensor is still alive.
 */
extern MAG_EXPORT bool mag_tensor_decref(mag_tensor_t* t);

extern MAG_EXPORT void mag_tensor_copy_buffer_from(mag_tensor_t* t, const void* data, size_t size); /* Copy data into tensor buffer */
extern MAG_EXPORT void mag_tensor_fill(mag_tensor_t* t, float x); /* Set all tensor elements to a specific value */
extern MAG_EXPORT void mag_tensor_fill_random_uniform(mag_tensor_t* t, float min, float max); /* Fill tensor with random values from uniform distribution within [min, max] */
extern MAG_EXPORT void mag_tensor_fill_random_normal(mag_tensor_t* t, float mean, float stddev); /* Fill tensor with random values from the normal distribution. */

extern MAG_EXPORT uint64_t mag_tensor_get_packed_refcounts(const mag_tensor_t* t); /* Return strong refcount is loword, weak refcount is hiword. */
extern MAG_EXPORT void mag_tensor_retain(mag_tensor_t* t); /* Increment refcount */
extern MAG_EXPORT size_t mag_tensor_get_memory_usage(const mag_tensor_t* t); /* Return memory used by this tensor in bytes. */
extern MAG_EXPORT void mag_tensor_print(const mag_tensor_t* t, bool with_header, bool with_data); /* Print tensor info (with or without data) */
extern MAG_EXPORT void mag_tensor_set_name(mag_tensor_t* t, const char* name); /* Set the name of the tensor */
extern MAG_EXPORT void mag_tensor_fmt_name(mag_tensor_t* t, const char* fmt, ...); /* Format the name of the tensor */
extern MAG_EXPORT const char* mag_tensor_get_name(const mag_tensor_t* t); /* Get the name of the tensor */
extern MAG_EXPORT int64_t mag_tensor_rank(const mag_tensor_t* t); /* Get the rank (number of dimensions) of the tensor */
extern MAG_EXPORT const int64_t* mag_tensor_shape(const mag_tensor_t* t); /* Get the dimensions of the tensor */
extern MAG_EXPORT const int64_t* mag_tensor_strides(const mag_tensor_t* t); /* Get the strides of the tensor */
extern MAG_EXPORT mag_dtype_t mag_tensor_dtype(const mag_tensor_t* t); /* Get the data type of the tensor */
extern MAG_EXPORT void* mag_tensor_data_ptr(const mag_tensor_t* t); /* Get the tensor raw buffer pointer. Might pointer to GPU or any other device memory. */
extern MAG_EXPORT int64_t mag_tensor_data_size(const mag_tensor_t* t); /* Get the size of the tensor buffer in bytes. */
extern MAG_EXPORT int64_t mag_tensor_numel(const mag_tensor_t* t); /* Get the total amount of elements in the tensor. */
extern MAG_EXPORT int64_t mag_tensor_num_rows(const mag_tensor_t* t); /* Get the number of rows (for 2D tensors) */
extern MAG_EXPORT int64_t mag_tensor_num_cols(const mag_tensor_t* t); /* Get the number of columns (for 2D tensors) */
extern MAG_EXPORT bool mag_tensor_is_scalar(const mag_tensor_t* t); /* Check if the tensor is a scalar */
extern MAG_EXPORT bool mag_tensor_is_vector(const mag_tensor_t* t); /* Check if the tensor is a vector */
extern MAG_EXPORT bool mag_tensor_is_matrix(const mag_tensor_t* t); /* Check if the tensor is a matrix */
extern MAG_EXPORT bool mag_tensor_is_volume(const mag_tensor_t* t); /* Check if the tensor is higher-order (3D or more) */
extern MAG_EXPORT bool mag_tensor_is_shape_eq(const mag_tensor_t* a, const mag_tensor_t* b); /* Checks if a and b have the same shape. */
extern MAG_EXPORT bool mag_tensor_are_strides_eq(const mag_tensor_t* a, const mag_tensor_t* b); /* Checks if a and b have the same strides. */
extern MAG_EXPORT bool mag_tensor_can_broadcast(const mag_tensor_t* a, const mag_tensor_t* b); /* Checks if b can be broadcasted into a. */
extern MAG_EXPORT bool mag_tensor_is_transposed(const mag_tensor_t* t); /* Check if the tensor is transposed */
extern MAG_EXPORT bool mag_tensor_is_permuted(const mag_tensor_t* t); /* Check if the tensor is permuted */
extern MAG_EXPORT bool mag_tensor_is_contiguous(const mag_tensor_t* t); /* Check if the tensor memory is contiguous */
extern MAG_EXPORT float mag_tensor_get_scalar_physical_index(mag_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5); /* Get scalar value at physical index */
extern MAG_EXPORT void mag_tensor_set_scalar_physical_index(mag_tensor_t* t, int64_t d0, int64_t d1, int64_t d2, int64_t d3, int64_t d4, int64_t d5, float x); /* Set scalar value at physical index */
extern MAG_EXPORT float mag_tensor_get_scalar_virtual_index(mag_tensor_t* t, int64_t v_idx); /* Get scalar value at virtual index */
extern MAG_EXPORT void mag_tensor_set_scalar_virtual_index(mag_tensor_t* t, int64_t v_idx, float x); /* Set scalar value at virtual index */
extern MAG_EXPORT bool mag_tensor_eq(const mag_tensor_t* a, const mag_tensor_t* b); /* Check if two tensors are equal without epsilon. */
extern MAG_EXPORT bool mag_tensor_is_close(const mag_tensor_t* a, const mag_tensor_t* b, float eps, double* percent_eq); /* Check if two tensors are equal with epsilon and percentage in equality. Set eps to < 0 to use machine epsilon. */
extern MAG_EXPORT void mag_tensor_img_draw_box(mag_tensor_t* t, int32_t x1, int32_t y1, int32_t x2, int32_t y2, int32_t wi, uint32_t rgb);
extern MAG_EXPORT void mag_tensor_img_draw_text(mag_tensor_t* t, int32_t x, int32_t y, int32_t size, uint32_t rgb, const char* txt); /* Draw text on image tensor */
extern MAG_EXPORT mag_ctx_t* mag_tensor_get_ctx(const mag_tensor_t* t); /* Get the context of the tensor */
extern MAG_EXPORT void* mag_tensor_get_user_data(const mag_tensor_t* t); /* Get the user data of the tensor */
extern MAG_EXPORT void mag_tensor_set_user_data(mag_tensor_t* t, void* ud); /* Set the user data of the tensor */
extern MAG_EXPORT void mag_tensor_save(const mag_tensor_t* t, const char* file); /* Save tensor to magnetron binary file. */
extern MAG_EXPORT mag_tensor_t* mag_tensor_load(mag_ctx_t* ctx, const char* file); /* Load tensor from magnetron binary file. */
extern MAG_EXPORT mag_tensor_t* mag_tensor_load_image(mag_ctx_t* ctx, const char* file, mag_color_channels_t channels, uint32_t resize_w, uint32_t resize_h); /* Create a tensor from an image file. */
extern MAG_EXPORT void mag_tensor_save_image(const mag_tensor_t* t, const char* file); /* Save tensor data as an image */
#define mag_tensor_image_width(tensor) (mag_tensor_shape(tensor)[2]) /* Get image width from tensor */
#define mag_tensor_image_height(tensor) (mag_tensor_shape(tensor)[1]) /* Get image height from tensor */
#define mag_tensor_image_channels(tensor) (mag_tensor_shape(tensor)[0]) /* Get image channels from tensor */

#ifdef __cplusplus
}
#endif
#endif
