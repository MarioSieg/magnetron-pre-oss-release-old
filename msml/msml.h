/*
** (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
** MSML - Single file STB-style machine learning library in C99 with Python bindings.
** For license see LICENSE file.
*/

#ifndef MSML_INCLUDE_MSML_H
#define MSML_INCLUDE_MSML_H

#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdbool.h>
#include <inttypes.h>

#ifndef MSML_API
#   ifdef MSML_EXPORT_DLL
#       ifdef _MSC_VER
#           define MSML_API __declspec(dllexport)
#       else
#           define MSML_API __attribute__((visibility("default")))
#       endif
#   else
#       define MSML_API
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
#define MSML_DEFAULT_CHUNK_SIZE (1ull<<30) /* Default size of memory chunk in bytes. 1 GiB */
#define MSML_DEFAULT_CHUNK_CAP (1ull<<3)   /* Default capacity of memory chunk */
#define MSML_MAX_DIMS 4                 /* Maximum number of dimensions for a tensor */
#define MSML_MAX_TENSOR_NAME_LEN 64     /* Maximum length for tensor name */
#define msml_assert_name2(name, line) name ## line
#define msml_assert_name(line) msml_assert_name2(_assert_, line)
#define msml_static_assert(expr) extern void msml_assert_name(__LINE__)(bool STATIC_ASSERTION_FAILED[((expr)?1:-1)])

extern MSML_API void* msml_default_allocator_impl(void* blk, size_t size); /* Default memory allocator */
#ifndef msml_allocator /* Default allocator, can be overridden by defining it before.  */
#define msml_allocator msml_default_allocator_impl
#endif

typedef enum msml_prng_algorithm_t {
    MSML_PRNG_MERSENNE_TWISTER, /* Mersenne Twister PRNG */
    MSML_PRNG_PCG, /* Permuted Congruential Generator PRNG */
    MSML_PRNG_COUNT_                /* Total number of PRNG algorithms */
} msml_prng_algorithm_t;

typedef struct msml_ctx_info_t {
    void* (*alloc_fn)(void* blk, size_t size);  /* Custom allocator function */
    size_t pool_chunk_size;                     /* Size of each memory pool chunk */
    size_t pool_chunks_cap;                     /* Maximum chunks in the pool */
    uint64_t prng_seed;                           /* Seed for PRNG if prng_init_seed == true */
    void* user_data;                            /* User-defined data */
    msml_prng_algorithm_t prng_algorithm;       /* PRNG algorithm */
} msml_ctx_info_t;

typedef struct msml_ctx_t msml_ctx_t; /* Opaque context type for managing memory pools */

extern MSML_API msml_ctx_t* msml_ctx_create(const msml_ctx_info_t* info); /* Create context with allocator and pool info */
extern MSML_API void* msml_ctx_pool_alloc(msml_ctx_t* ctx, size_t size); /* Allocate memory from pool */
extern MSML_API void* msml_ctx_pool_alloc_aligned(msml_ctx_t* ctx, size_t size, size_t align); /* Aligned memory allocation */
extern MSML_API size_t msml_ctx_total_memory(const msml_ctx_t* ctx); /* Get total allocated memory */
extern MSML_API msml_prng_algorithm_t msml_ctx_get_prng_algorithm(const msml_ctx_t* ctx); /* Get PRNG algorithm */
extern MSML_API void msml_ctx_set_prng_algorithm(msml_ctx_t* ctx, msml_prng_algorithm_t algorithm, uint64_t seed); /* Set PRNG algorithm */
extern MSML_API void msml_ctx_destroy(msml_ctx_t* ctx); /* Destroy context and free memory */

typedef enum msml_dtype_t {
    MSML_DTYPE_F32,   /* 32-bit floating-point data type */
    MSML_DTYPE_COUNT_ /* Total number of data types */
} msml_dtype_t;
msml_static_assert(MSML_DTYPE_COUNT_ <= 0xff);

typedef struct msml_dtype_info_t {
    size_t size;         /* Size of the data type in bytes */
    const char* name;    /* Name of the data type */
} msml_dtype_info_t;
extern MSML_API const msml_dtype_info_t* msml_get_dtype_info(msml_dtype_t type);

typedef enum msml_desired_color_channels_t {
    MSML_COLOR_CHANNELS_AUTO,  /* Automatically detect number of color channels */
    MSML_COLOR_CHANNELS_GRAY,  /* Grayscale F32 */
    MSML_COLOR_CHANNELS_GRAY_A,/* Grayscale F32 + Alpha F32 */
    MSML_COLOR_CHANNELS_RGB,   /* R32G32B32 */
    MSML_COLOR_CHANNELS_RGBA   /* R32G32B32A32 */
} msml_desired_color_channels_t;

typedef enum msml_format_type {
    MSML_FMT_EOF, MSML_FMT_ERR, MSML_FMT_LIT, MSML_FMT_INT,
    MSML_FMT_UINT, MSML_FMT_NUM, MSML_FMT_STR, MSML_FMT_CHAR,
    MSML_FMT_PTR
} msml_format_type; /* Format types for formatted output */

typedef uint32_t msml_format_flags; /* Flags for formatting output */

/* Format flags */
#define MSML_FMT_F_LEFT  0x0100 /* Left-align the output */
#define MSML_FMT_F_PLUS  0x0200 /* Prefix positive numbers with a plus sign */
#define MSML_FMT_F_ZERO  0x0400 /* Pad with zeros instead of spaces */
#define MSML_FMT_F_SPACE 0x0800 /* Prefix a space for positive numbers */
#define MSML_FMT_F_ALT   0x1000 /* Alternate format flag */
#define MSML_FMT_F_UPPER 0x2000 /* Use uppercase letters for hex output */

/* Format subtypes (bits reused) */
#define MSML_FMT_T_HEX   0x0010 /* Hexadecimal format for unsigned integers */
#define MSML_FMT_T_OCT   0x0020 /* Octal format for unsigned integers */
#define MSML_FMT_T_FP_A  0x0000 /* 'a' format for floating-point numbers */
#define MSML_FMT_T_FP_E  0x0010 /* 'e' format for floating-point numbers */
#define MSML_FMT_T_FP_F  0x0020 /* 'f' format for floating-point numbers */
#define MSML_FMT_T_FP_G  0x0030 /* 'g' format for floating-point numbers */
#define MSML_FMT_T_QUOTED 0x0010 /* Quoted string format */

#define MSML_FMT_SH_WIDTH 16    /* Shift width for formatting */
#define MSML_FMT_SH_PREC  24    /* Shift precision for formatting */
#define MSML_FMT_TYPE(sf) ((FormatType)((sf) & 15))  /* Extract format type */
#define MSML_FMT_WIDTH(sf) (((sf) >> MSML_FMT_SH_WIDTH) & 255u) /* Extract width */
#define MSML_FMT_PREC(sf) ((((sf) >> MSML_FMT_SH_PREC) & 255u) - 1u) /* Extract precision */
#define MSML_FMT_FP(sf) (((sf) >> 4) & 3) /* Extract floating-point format */

/* Formats for conversion characters */
#define MSML_FMT_A (MSML_FMT_NUM|MSML_FMT_T_FP_A) /* 'a' format */
#define MSML_FMT_C (MSML_FMT_CHAR) /* 'c' format */
#define MSML_FMT_D (MSML_FMT_INT)  /* 'd' format */
#define MSML_FMT_E (MSML_FMT_NUM|MSML_FMT_T_FP_E) /* 'e' format */
#define MSML_FMT_F (MSML_FMT_NUM|MSML_FMT_T_FP_F) /* 'f' format */
#define MSML_FMT_G (MSML_FMT_NUM|MSML_FMT_T_FP_G) /* 'g' format */
#define MSML_FMT_I MSML_FMT_D /* 'i' format (same as 'd') */
#define MSML_FMT_O (MSML_FMT_UINT|MSML_FMT_T_OCT) /* 'o' format */
#define MSML_FMT_P (MSML_FMT_PTR) /* 'p' format */
#define MSML_FMT_Q (MSML_FMT_STR|MSML_FMT_T_QUOTED) /* Quoted string */
#define MSML_FMT_S (MSML_FMT_STR) /* 's' format */
#define MSML_FMT_U (MSML_FMT_UINT) /* 'u' format */
#define MSML_FMT_X (MSML_FMT_UINT|MSML_FMT_T_HEX) /* 'x' format */
#define MSML_FMT_G14 (MSML_FMT_G | ((14+1) << MSML_FMT_SH_PREC)) /* 'g' format with precision 14 */

#define MSML_MAX_ARG_TENSORS 2
#define MSML_SEP ,
#define msml_op_def(_, __) /* Enumerator | Mnemonic | Argcount */\
    _(NOP, "nop", 0)/* No Operation. */__\
    _(ADD, "+", 2)/* R = A+B */__\
    _(SUB, "-", 2)/* R = A-B */__\
    _(MUL, "*", 2)/* R = A*B (Hadamard product). */__\
    _(DIV, "/", 2)/* R = A/B. */__\
    _(MATMUL, "@", 2)/* R = AB. (Matrix multiplication) */__

#define _(enumerator, mnemonic, argcount) MSML_OP_##enumerator
typedef enum msml_op_t {
    msml_op_def(_, MSML_SEP)
    MSML_OP__COUNT
} msml_op_t;
#undef _
msml_static_assert(MSML_OP_NOP == 0);
msml_static_assert(MSML_OP__COUNT <= 0xff);

extern MSML_API const char* msml_op_get_name(msml_op_t op);
extern MSML_API const char* msml_op_get_mnemonic(msml_op_t op);
extern MSML_API uint8_t msml_op_get_argcount(msml_op_t op);
#define msml_op_is_unary(op) (msml_op_get_argcount(op) == 1)
#define msml_op_is_binary(op) (msml_op_get_argcount(op) == 2)

typedef struct msml_tensor_t msml_tensor_t; /* Opaque type representing a tensor */

extern MSML_API msml_ctx_t* msml_tensor_get_ctx(const msml_tensor_t* tensor); /* Get the context of the tensor */
extern MSML_API msml_tensor_t* msml_tensor_create(msml_ctx_t* ctx, msml_dtype_t type, const int64_t* dims, int64_t rank); /* Create a tensor with specified dimensions */
extern MSML_API msml_tensor_t* msml_tensor_create_1d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1); /* Create 1D tensor */
extern MSML_API msml_tensor_t* msml_tensor_create_2d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2); /* Create 2D tensor */
extern MSML_API msml_tensor_t* msml_tensor_create_3d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3); /* Create 3D tensor */
extern MSML_API msml_tensor_t* msml_tensor_create_4d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4); /* Create 4D tensor */
extern MSML_API msml_tensor_t* msml_tensor_get_arg(const msml_tensor_t* tensor, size_t slot); /* Return arg at index or NULL if not set. */
extern MSML_API void msml_tensor_set_arg(msml_tensor_t* tensor, size_t slot, msml_tensor_t* arg); /* Return arg at index or NULL if not set. */
extern MSML_API msml_op_t msml_tensor_get_op(const msml_tensor_t* tensor); /* Get opcode for tensor. */
extern MSML_API void msml_tensor_set_op(msml_tensor_t* tensor, msml_op_t op); /* Set opcode for tensor. */
extern MSML_API msml_tensor_t* msml_tensor_isomorphic_clone(msml_tensor_t* tensor); /* Create new empty tensor with same shape as input, but without cloning data. */
extern MSML_API msml_tensor_t* msml_tensor_deep_clone(msml_tensor_t* tensor); /* Create new tensor with same shape and data as input. */
extern MSML_API void msml_tensor_copy_buffer_from(msml_tensor_t* tensor, const void* data, size_t size); /* Copy data into tensor buffer */
extern MSML_API void msml_tensor_fill_zero(msml_tensor_t* tensor); /* Set all tensor elements to zero */
extern MSML_API void msml_tensor_fill_one(msml_tensor_t* tensor); /* Set all tensor elements to one */
extern MSML_API void msml_tensor_fill(msml_tensor_t* tensor, float x); /* Set all tensor elements to a specific value */
extern MSML_API void msml_tensor_fill_random(msml_tensor_t* tensor, float min, float max); /* Fill tensor with random values within [min, max] */
extern MSML_API void msml_tensor_print(const msml_tensor_t* tensor, bool with_data); /* Print tensor info (with or without data) */
extern MSML_API void msml_tensor_set_name(msml_tensor_t* tensor, const char* name); /* Set the name of the tensor */
extern MSML_API const char* msml_tensor_get_name(const msml_tensor_t* tensor); /* Get the name of the tensor */
extern MSML_API int64_t msml_tensor_rank(const msml_tensor_t* tensor); /* Get the rank (number of dimensions) of the tensor */
extern MSML_API const int64_t* msml_tensor_dims(const msml_tensor_t* tensor); /* Get the dimensions of the tensor */
extern MSML_API const int64_t* msml_tensor_strides(const msml_tensor_t* tensor); /* Get the strides of the tensor */
extern MSML_API msml_dtype_t msml_tensor_dtype(const msml_tensor_t* tensor); /* Get the data type of the tensor */
extern MSML_API void* msml_tensor_buf(const msml_tensor_t* tensor); /* Get the tensor buffer pointer */
extern MSML_API int64_t msml_tensor_buf_size(const msml_tensor_t* tensor); /* Get the size of the tensor buffer */
extern MSML_API int64_t msml_tensor_buf_len(const msml_tensor_t* tensor); /* Get the length of the tensor buffer */
extern MSML_API int64_t msml_tensor_num_rows(const msml_tensor_t* tensor); /* Get the number of rows (for 2D tensors) */
extern MSML_API int64_t msml_tensor_num_cols(const msml_tensor_t* tensor); /* Get the number of columns (for 2D tensors) */
extern MSML_API bool msml_tensor_is_scalar(const msml_tensor_t* tensor); /* Check if the tensor is a scalar */
extern MSML_API bool msml_tensor_is_vector(const msml_tensor_t* tensor); /* Check if the tensor is a vector */
extern MSML_API bool msml_tensor_is_matrix(const msml_tensor_t* tensor); /* Check if the tensor is a matrix */
extern MSML_API bool msml_tensor_is_higher_order_3d(const msml_tensor_t* tensor); /* Check if the tensor is higher-order (3D or more) */
extern MSML_API void msml_tensor_virtual_to_physical_index(const msml_tensor_t* tensor, int64_t v_idx, int64_t(*p_idx)[MSML_MAX_DIMS]); /* Convert virtual index to physical index */
extern MSML_API int64_t msml_tensor_physical_to_virtual_index(const msml_tensor_t* tensor, const int64_t(*p_idx)[MSML_MAX_DIMS]); /* Convert physical index to virtual index */
extern MSML_API bool msml_tensor_is_contiguous(const msml_tensor_t* tensor); /* Check if the tensor memory is contiguous */
extern MSML_API float msml_tensor_get_scalar_physical_index(const msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3); /* Get scalar value at physical index */
extern MSML_API void msml_tensor_set_scalar_physical_index(msml_tensor_t* tensor, int64_t d0, int64_t d1, int64_t d2, int64_t d3, float x); /* Set scalar value at physical index */
extern MSML_API float msml_tensor_get_scalar_virtual_index(const msml_tensor_t* tensor, int64_t v_idx); /* Get scalar value at virtual index */
extern MSML_API void msml_tensor_set_scalar_virtual_index(msml_tensor_t* tensor, int64_t v_idx, float x); /* Set scalar value at virtual index */
extern MSML_API void msml_tensor_save(const msml_tensor_t* tensor, const char* file_name); /* Save tensor to MSML binary file. */
extern MSML_API msml_tensor_t* msml_tensor_load(msml_ctx_t* ctx, const char* file_name); /* Load tensor from MSML binary file. */

extern MSML_API msml_tensor_t* msml_tensor_create_from_image(msml_ctx_t* ctx, const char* file_path, msml_desired_color_channels_t channels, uint32_t resize_width, uint32_t resize_height); /* Create a tensor from an image file */
extern MSML_API void msml_tensor_save_to_image(const msml_tensor_t* tensor, const char* file_path); /* Save tensor data as an image */
#define msml_tensor_image_width(tensor) (msml_tensor_dims(tensor)[0]) /* Get image width from tensor */
#define msml_tensor_image_height(tensor) (msml_tensor_dims(tensor)[1]) /* Get image height from tensor */
#define msml_tensor_image_channels(tensor) (msml_tensor_dims(tensor)[2]) /* Get image channels from tensor */

#ifdef __cplusplus
}
#endif
#endif
