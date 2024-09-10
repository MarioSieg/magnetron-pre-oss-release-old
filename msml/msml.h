/*
** (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
** MSML - Single file STB-style machine learning library in C99 with Python bindings.
** For license see LICENSE file.
*/

#ifndef MSML_INCLUDE_MSML_H
#define MSML_INCLUDE_MSML_H

#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

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

#define MSML_DEFAULT_CHUNK_SIZE (1<<20)
#define MSML_DEFAULT_CHUNK_CAP (1<<3)
#define MSML_MAX_DIMS 4
#define MSML_MAX_TENSOR_NAME_LEN 64

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

typedef enum msml_desired_color_channels_t {
    MSML_COLOR_CHANNELS_AUTO,
    MSML_COLOR_CHANNELS_GRAY, /* Grayscale F32 */
    MSML_COLOR_CHANNELS_GRAY_A, /* Grayscale F32 + Alpha F32 */
    MSML_COLOR_CHANNELS_RGB, /* R32G32B32 */
    MSML_COLOR_CHANNELS_RGBA /* R32G32B32A32 */
} msml_desired_color_channels_t;

typedef enum msml_format_type {
    MSML_FMT_EOF, MSML_FMT_ERR, MSML_FMT_LIT,
    MSML_FMT_INT, MSML_FMT_UINT, MSML_FMT_NUM,
    MSML_FMT_STR, MSML_FMT_CHAR, MSML_FMT_PTR
} msml_format_type;

typedef uint32_t msml_format_flags;

/* Format flags. */
#define MSML_FMT_F_LEFT	0x0100
#define MSML_FMT_F_PLUS	0x0200
#define MSML_FMT_F_ZERO	0x0400
#define MSML_FMT_F_SPACE 0x0800
#define MSML_FMT_F_ALT 0x1000
#define MSML_FMT_F_UPPER 0x2000

/* Format subtypes (bits are reused). */
#define MSML_FMT_T_HEX 0x0010 /* MSML_FMT_UINT */
#define MSML_FMT_T_OCT 0x0020 /* MSML_FMT_UINT */
#define MSML_FMT_T_FP_A	0x0000 /* MSML_FMT_NUM */
#define MSML_FMT_T_FP_E	0x0010 /* MSML_FMT_NUM */
#define MSML_FMT_T_FP_F	0x0020 /* MSML_FMT_NUM */
#define MSML_FMT_T_FP_G	0x0030 /* MSML_FMT_NUM */
#define MSML_FMT_T_QUOTED 0x0010 /* MSML_FMT_STR */

#define MSML_FMT_SH_WIDTH 16
#define MSML_FMT_SH_PREC 24
#define MSML_FMT_TYPE(sf) ((FormatType)((sf) & 15))
#define MSML_FMT_WIDTH(sf) (((sf) >> MSML_FMT_SH_WIDTH) & 255u)
#define MSML_FMT_PREC(sf) ((((sf) >> MSML_FMT_SH_PREC) & 255u) - 1u)
#define MSML_FMT_FP(sf) (((sf) >> 4) & 3)

/* Formats for conversion characters. */
#define MSML_FMT_A (MSML_FMT_NUM|MSML_FMT_T_FP_A)
#define MSML_FMT_C (MSML_FMT_CHAR)
#define MSML_FMT_D (MSML_FMT_INT)
#define MSML_FMT_E (MSML_FMT_NUM|MSML_FMT_T_FP_E)
#define MSML_FMT_F (MSML_FMT_NUM|MSML_FMT_T_FP_F)
#define MSML_FMT_G (MSML_FMT_NUM|MSML_FMT_T_FP_G)
#define MSML_FMT_I MSML_FMT_D
#define MSML_FMT_O (MSML_FMT_UINT|MSML_FMT_T_OCT)
#define MSML_FMT_P (MSML_FMT_PTR)
#define MSML_FMT_Q (MSML_FMT_STR|MSML_FMT_T_QUOTED)
#define MSML_FMT_S (MSML_FMT_STR)
#define MSML_FMT_U (MSML_FMT_UINT)
#define MSML_FMT_X (MSML_FMT_UINT|MSML_FMT_T_HEX)
#define MSML_FMT_G14 (MSML_FMT_G | ((14+1) << MSML_FMT_SH_PREC))

typedef struct msml_tensor_t msml_tensor_t;

extern MSML_API msml_tensor_t* msml_tensor_create(msml_ctx_t* ctx, msml_dtype_t type, const int64_t* dims, int64_t rank);
extern MSML_API msml_tensor_t* msml_tensor_create_1d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1);
extern MSML_API msml_tensor_t* msml_tensor_create_2d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2);
extern MSML_API msml_tensor_t* msml_tensor_create_3d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3);
extern MSML_API msml_tensor_t* msml_tensor_create_4d(msml_ctx_t* ctx, msml_dtype_t type, int64_t d1, int64_t d2, int64_t d3, int64_t d4);
extern MSML_API msml_tensor_t* msml_tensor_create_from_image(msml_ctx_t* ctx, const char* file_path, msml_desired_color_channels_t channels, uint32_t resize_width, uint32_t resize_height);
extern MSML_API void msml_tensor_copy_buffer_from(msml_tensor_t* tensor, const void* data, size_t size);
extern MSML_API void msml_tensor_set_zero(msml_tensor_t* tensor);
extern MSML_API void msml_tensor_set_one(msml_tensor_t* tensor);
extern MSML_API void msml_tensor_set(msml_tensor_t* tensor, float x);
extern MSML_API void msml_tensor_print(const msml_tensor_t* tensor, bool with_data);
extern MSML_API void msml_tensor_set_name(msml_tensor_t* tensor, const char* name);
extern MSML_API const char* msml_tensor_get_name(const msml_tensor_t* tensor);
extern MSML_API int64_t msml_tensor_rank(const msml_tensor_t* tensor);
extern MSML_API const int64_t* msml_tensor_dims(const msml_tensor_t* tensor);
extern MSML_API const int64_t* msml_tensor_strides(const msml_tensor_t* tensor);
extern MSML_API msml_dtype_t msml_tensor_dtype(const msml_tensor_t* tensor);
extern MSML_API void* msml_tensor_buf(const msml_tensor_t* tensor);
extern MSML_API int64_t msml_tensor_buf_size(const msml_tensor_t* tensor);
extern MSML_API int64_t msml_tensor_buf_len(const msml_tensor_t* tensor);
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
