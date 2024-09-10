/*
 * (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>
 * MSML - Single file STB-style machine learning library in C99 with Python bindings.
 * MIT licensed.
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
    MSML_COLOR_CHANNELS_GRAY,
    MSML_COLOR_CHANNELS_GRAY_A,
    MSML_COLOR_CHANNELS_RGB,
    MSML_COLOR_CHANNELS_RGBA
} msml_desired_color_channels_t;

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
