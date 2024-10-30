// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"
#include <array>
#include <cstring>
#include <unordered_set>
#include <filesystem>

TEST(wl_tensor_t, init_1d) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* tensor = wl_tensor_create_1d(ctx, WL_DTYPE_F32, 10);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(wl_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(wl_tensor_dtype(tensor), WL_DTYPE_F32);
    ASSERT_EQ(wl_tensor_rank(tensor), 1);
    ASSERT_EQ(wl_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(wl_tensor_shape(tensor)[1], 1);
    ASSERT_EQ(wl_tensor_shape(tensor)[2], 1);
    ASSERT_EQ(wl_tensor_shape(tensor)[3], 1);
    ASSERT_EQ(wl_tensor_data_size(tensor), 10 * sizeof(float));
    ASSERT_EQ(wl_tensor_num_elements(tensor), 10);
    ASSERT_EQ(wl_tensor_num_cols(tensor), 10);
    ASSERT_EQ(wl_tensor_num_rows(tensor), 1);
    ASSERT_EQ(wl_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[2], 10*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[3], 10*sizeof(float));
    ASSERT_FALSE(wl_tensor_is_scalar(tensor));
    ASSERT_TRUE(wl_tensor_is_vector(tensor));
    ASSERT_TRUE(wl_tensor_is_matrix(tensor));
    ASSERT_TRUE(wl_tensor_is_volume(tensor));

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, init_2d) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* tensor = wl_tensor_create_2d(ctx, WL_DTYPE_F32, 10, 4);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(wl_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(wl_tensor_dtype(tensor), WL_DTYPE_F32);
    ASSERT_EQ(wl_tensor_rank(tensor), 2);
    ASSERT_EQ(wl_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(wl_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(wl_tensor_shape(tensor)[2], 1);
    ASSERT_EQ(wl_tensor_shape(tensor)[3], 1);
    ASSERT_EQ(wl_tensor_data_size(tensor), 10 * 4 * sizeof(float));
    ASSERT_EQ(wl_tensor_num_elements(tensor), 10 * 4);
    ASSERT_EQ(wl_tensor_num_cols(tensor), 10);
    ASSERT_EQ(wl_tensor_num_rows(tensor), 4);
    ASSERT_EQ(wl_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[3], 10*4*sizeof(float));
    ASSERT_FALSE(wl_tensor_is_scalar(tensor));
    ASSERT_FALSE(wl_tensor_is_vector(tensor));
    ASSERT_TRUE(wl_tensor_is_matrix(tensor));
    ASSERT_TRUE(wl_tensor_is_volume(tensor));

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, init_3d) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* tensor = wl_tensor_create_3d(ctx, WL_DTYPE_F32, 10, 4, 2);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(wl_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(wl_tensor_dtype(tensor), WL_DTYPE_F32);
    ASSERT_EQ(wl_tensor_rank(tensor), 3);
    ASSERT_EQ(wl_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(wl_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(wl_tensor_shape(tensor)[2], 2);
    ASSERT_EQ(wl_tensor_shape(tensor)[3], 1);
    ASSERT_EQ(wl_tensor_data_size(tensor), 10 * 4 * 2 * sizeof(float));
    ASSERT_EQ(wl_tensor_num_elements(tensor), 10 * 4 * 2);
    ASSERT_EQ(wl_tensor_num_cols(tensor), 10);
    ASSERT_EQ(wl_tensor_num_rows(tensor), 8);
    ASSERT_EQ(wl_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[3], 10*4*2*sizeof(float));
    ASSERT_FALSE(wl_tensor_is_scalar(tensor));
    ASSERT_FALSE(wl_tensor_is_vector(tensor));
    ASSERT_FALSE(wl_tensor_is_matrix(tensor));
    ASSERT_TRUE(wl_tensor_is_volume(tensor));

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, init_4d) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* tensor = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 10, 4, 2, 5);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(wl_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(wl_tensor_dtype(tensor), WL_DTYPE_F32);
    ASSERT_EQ(wl_tensor_rank(tensor), 4);
    ASSERT_EQ(wl_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(wl_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(wl_tensor_shape(tensor)[2], 2);
    ASSERT_EQ(wl_tensor_shape(tensor)[3], 5);
    ASSERT_EQ(wl_tensor_data_size(tensor), 10 * 4 * 2 * 5 * sizeof(float));
    ASSERT_EQ(wl_tensor_num_elements(tensor), 10 * 4 * 2 * 5);
    ASSERT_EQ(wl_tensor_num_cols(tensor), 10);
    ASSERT_EQ(wl_tensor_num_rows(tensor), 40);
    ASSERT_EQ(wl_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[3], 10*4*2*sizeof(float));
    ASSERT_FALSE(wl_tensor_is_scalar(tensor));
    ASSERT_FALSE(wl_tensor_is_vector(tensor));
    ASSERT_FALSE(wl_tensor_is_matrix(tensor));
    ASSERT_FALSE(wl_tensor_is_volume(tensor));

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, init_5d) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* tensor = wl_tensor_create_5d(ctx, WL_DTYPE_F32, 10, 4, 2, 5, 3);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(wl_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(wl_tensor_dtype(tensor), WL_DTYPE_F32);
    ASSERT_EQ(wl_tensor_rank(tensor), 5);
    ASSERT_EQ(wl_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(wl_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(wl_tensor_shape(tensor)[2], 2);
    ASSERT_EQ(wl_tensor_shape(tensor)[3], 5);
    ASSERT_EQ(wl_tensor_shape(tensor)[4], 3);
    ASSERT_EQ(wl_tensor_shape(tensor)[5], 1);
    ASSERT_EQ(wl_tensor_data_size(tensor), 10 * 4 * 2 * 5 * 3 * sizeof(float));
    ASSERT_EQ(wl_tensor_num_elements(tensor), 10 * 4 * 2 * 5 * 3);
    ASSERT_EQ(wl_tensor_num_cols(tensor), 10);
    ASSERT_EQ(wl_tensor_num_rows(tensor), 40*3);
    ASSERT_EQ(wl_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[3], 10*4*2*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[4], 10*4*2*5*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[5], 10*4*2*5*3*sizeof(float));

    ASSERT_FALSE(wl_tensor_is_scalar(tensor));
    ASSERT_FALSE(wl_tensor_is_vector(tensor));
    ASSERT_FALSE(wl_tensor_is_matrix(tensor));
    ASSERT_FALSE(wl_tensor_is_volume(tensor));

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, init_6d) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* tensor = wl_tensor_create_6d(ctx, WL_DTYPE_F32, 10, 4, 2, 5, 3, 2);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(wl_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(wl_tensor_dtype(tensor), WL_DTYPE_F32);
    ASSERT_EQ(wl_tensor_rank(tensor), 6);
    ASSERT_EQ(wl_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(wl_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(wl_tensor_shape(tensor)[2], 2);
    ASSERT_EQ(wl_tensor_shape(tensor)[3], 5);
    ASSERT_EQ(wl_tensor_shape(tensor)[4], 3);
    ASSERT_EQ(wl_tensor_shape(tensor)[5], 2);
    ASSERT_EQ(wl_tensor_data_size(tensor), 10 * 4 * 2 * 5 * 3 * 2 * sizeof(float));
    ASSERT_EQ(wl_tensor_num_elements(tensor), 10 * 4 * 2 * 5 * 3 * 2);
    ASSERT_EQ(wl_tensor_num_cols(tensor), 10);
    ASSERT_EQ(wl_tensor_num_rows(tensor), 40*3*2);
    ASSERT_EQ(wl_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[3], 10*4*2*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[4], 10*4*2*5*sizeof(float));
    ASSERT_EQ(wl_tensor_strides(tensor)[5], 10*4*2*5*3*sizeof(float));

    ASSERT_FALSE(wl_tensor_is_scalar(tensor));
    ASSERT_FALSE(wl_tensor_is_vector(tensor));
    ASSERT_FALSE(wl_tensor_is_matrix(tensor));
    ASSERT_FALSE(wl_tensor_is_volume(tensor));

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, print) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* tensor = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 2, 2, 2, 2);
    wl_tensor_fill_random(tensor, 0.0f, 1.0f);
    wl_tensor_print(tensor, false, true);

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, name) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* tensor = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 2, 2, 2, 2);
    wl_tensor_set_name(tensor, "Gradient Backup");
    ASSERT_STREQ(wl_tensor_get_name(tensor), "Gradient Backup");

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, deep_clone) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* origin = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 10, 4, 2, 5);
    wl_tensor_fill_random(origin, -1.0f, 1.0f);
    wl_tensor_t* clone = wl_tensor_emit_op_va(ctx, WL_OP_CLONE, origin);
    ASSERT_NE(origin, clone);
    ASSERT_EQ(wl_tensor_rank(origin), wl_tensor_rank(clone));
    ASSERT_EQ(wl_tensor_shape(origin)[0], wl_tensor_shape(clone)[0]);
    ASSERT_EQ(wl_tensor_shape(origin)[1], wl_tensor_shape(clone)[1]);
    ASSERT_EQ(wl_tensor_shape(origin)[2], wl_tensor_shape(clone)[2]);
    ASSERT_EQ(wl_tensor_shape(origin)[3], wl_tensor_shape(clone)[3]);
    ASSERT_EQ(wl_tensor_data_size(origin), wl_tensor_data_size(clone));
    ASSERT_EQ(wl_tensor_num_elements(origin), wl_tensor_num_elements(clone));
    ASSERT_EQ(wl_tensor_num_cols(origin), wl_tensor_num_cols(clone));
    ASSERT_EQ(wl_tensor_num_rows(origin), wl_tensor_num_rows(clone));
    ASSERT_EQ(wl_tensor_strides(origin)[0], wl_tensor_strides(clone)[0]);
    ASSERT_EQ(wl_tensor_strides(origin)[1], wl_tensor_strides(clone)[1]);
    ASSERT_EQ(wl_tensor_strides(origin)[2], wl_tensor_strides(clone)[2]);
    ASSERT_EQ(wl_tensor_strides(origin)[3], wl_tensor_strides(clone)[3]);
    ASSERT_TRUE(wl_tensor_is_shape_eq(origin, clone));
    ASSERT_TRUE(wl_tensor_are_strides_eq(origin, clone));

    const void* a = wl_tensor_data(origin);
    const void* b = wl_tensor_data(clone);
    ASSERT_NE(a, b);
    ASSERT_EQ(0, std::memcmp(a, b, wl_tensor_data_size(origin)));

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, equals) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* origin = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 10, 4, 2, 5);
    wl_tensor_fill_random(origin, -1.0f, 1.0f);
    wl_tensor_t* clone = wl_tensor_emit_op_va(ctx, WL_OP_CLONE, origin);
    wl_tensor_t* clone2 = wl_tensor_emit_op_va(ctx, WL_OP_CLONE, origin);
    wl_tensor_fill_random(clone2, 0.0f, 1.0f);
    ASSERT_TRUE(wl_tensor_eq(origin, clone));
    ASSERT_FALSE(wl_tensor_eq(origin, clone2));
    ASSERT_FALSE(wl_tensor_eq(clone, clone2));

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, buffer_linearly) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* origin = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 1, 2, 3, 4);
    wl_tensor_fill(origin, 0.0f);
    wl_tensor_data_as_f32(origin)[0] = 1.0f;
    wl_tensor_data_as_f32(origin)[wl_tensor_num_elements(origin) - 1] = -1.0f;
    for (int64_t i=0; i < wl_tensor_num_elements(origin); ++i) {
        std::cout << wl_tensor_data_as_f32(origin)[i] << " ";
    }
    std::cout << std::endl;

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, view) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* origin = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 10, 4, 2, 5);
    wl_tensor_fill(origin, 2.0f);
    int64_t slice_dims[] = {10, 4, 2, 5};
    wl_tensor_t* slice1 = wl_tensor_emit_op_va(ctx, WL_OP_VIEW, origin);
    ASSERT_EQ(wl_tensor_data(slice1), wl_tensor_data(origin));
    ASSERT_EQ(wl_tensor_data_size(slice1), wl_tensor_data_size(origin));
    ASSERT_EQ(wl_tensor_num_elements(slice1), wl_tensor_num_elements(origin));
    auto* buf = wl_tensor_data_as_f32(slice1);
    for (int64_t i=0; i < wl_tensor_num_elements(slice1); ++i) {
        ASSERT_FLOAT_EQ(buf[i], 2.0f);
    }
    wl_tensor_t* slice2 = wl_tensor_emit_op_va(ctx, WL_OP_VIEW, origin);
    ASSERT_EQ(wl_tensor_data(slice2), static_cast<std::uint8_t*>(wl_tensor_data(origin)));
    ASSERT_EQ(wl_tensor_data_size(slice2), 10 * 4 * 2 * 5 * sizeof(float));
    ASSERT_EQ(wl_tensor_num_elements(slice2), 10 * 4 * 2 * 5);
    auto* buf_slice2 = wl_tensor_data_as_f32(slice2);
    for (int64_t i = 0; i < wl_tensor_num_elements(slice2); ++i) {
        ASSERT_FLOAT_EQ(buf_slice2[i], 2.0f);
    }

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, transpose) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* origin = wl_tensor_create_2d(ctx, WL_DTYPE_F32, 4, 1);
    wl_tensor_fill_random(origin, -1.0f, 1.0f);
    wl_tensor_t* transposed = wl_tensor_emit_op_va(ctx, WL_OP_TRANSPOSE, origin);
    ASSERT_FALSE(wl_tensor_is_transposed(origin));
    ASSERT_TRUE(wl_tensor_is_transposed(transposed));
    ASSERT_EQ(wl_tensor_shape(origin)[0], wl_tensor_shape(transposed)[1]);
    ASSERT_EQ(wl_tensor_shape(origin)[1], wl_tensor_shape(transposed)[0]);
    ASSERT_EQ(wl_tensor_data_size(origin), wl_tensor_data_size(transposed));
    ASSERT_EQ(wl_tensor_num_elements(origin), wl_tensor_num_elements(transposed));
    ASSERT_EQ(wl_tensor_num_cols(origin), wl_tensor_num_rows(transposed));
    ASSERT_EQ(wl_tensor_num_rows(origin), wl_tensor_num_cols(transposed));
    ASSERT_TRUE(wl_tensor_is_contiguous(origin));
    ASSERT_FALSE(wl_tensor_is_contiguous(transposed));

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, permute) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    wl_tensor_t* origin = wl_tensor_create_2d(ctx, WL_DTYPE_F32, 4, 1);
    wl_tensor_fill_random(origin, -1.0f, 1.0f);
    wl_op_param_t params[WL_MAX_OP_PARAMS] {
        wl_op_param_int(5),
        wl_op_param_int(4),
        wl_op_param_int(3),
        wl_op_param_int(2),
        wl_op_param_int(1),
        wl_op_param_int(0)
    };
    wl_tensor_t* permuted = wl_tensor_operator(ctx, WL_OP_PERMUTE, &origin, 1, &params);
    ASSERT_FALSE(wl_tensor_is_transposed(origin));
    ASSERT_FALSE(wl_tensor_is_transposed(permuted));
    ASSERT_FALSE(wl_tensor_is_permuted(origin));
    ASSERT_TRUE(wl_tensor_is_permuted(permuted));
    ASSERT_EQ(wl_tensor_shape(origin)[0], wl_tensor_shape(permuted)[5]);
    ASSERT_EQ(wl_tensor_shape(origin)[1], wl_tensor_shape(permuted)[4]);
    ASSERT_EQ(wl_tensor_shape(origin)[2], wl_tensor_shape(permuted)[3]);
    ASSERT_EQ(wl_tensor_shape(origin)[3], wl_tensor_shape(permuted)[2]);
    ASSERT_EQ(wl_tensor_shape(origin)[4], wl_tensor_shape(permuted)[1]);
    ASSERT_EQ(wl_tensor_shape(origin)[5], wl_tensor_shape(permuted)[0]);
    ASSERT_EQ(wl_tensor_data_size(origin), wl_tensor_data_size(permuted));
    ASSERT_EQ(wl_tensor_num_elements(origin), wl_tensor_num_elements(permuted));
    ASSERT_EQ(wl_tensor_num_cols(origin), wl_tensor_num_rows(permuted));
    ASSERT_EQ(wl_tensor_num_rows(origin), wl_tensor_num_cols(permuted));
    ASSERT_TRUE(wl_tensor_is_contiguous(origin));
    ASSERT_FALSE(wl_tensor_is_contiguous(permuted));

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, isclose) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* origin = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 10, 4, 2, 5);
    wl_tensor_fill_random(origin, -1.0f, 1.0f);
    wl_tensor_t* clone = wl_tensor_emit_op_va(ctx, WL_OP_CLONE, origin);
    wl_tensor_t* clone2 = wl_tensor_emit_op_va(ctx, WL_OP_CLONE, origin);
    wl_tensor_fill_random(clone2, 0.0f, 1.0f);
    ASSERT_TRUE(wl_tensor_is_close(origin, clone, FLT_EPSILON, nullptr));
    ASSERT_FALSE(wl_tensor_is_close(origin, clone2, FLT_EPSILON, nullptr));
    ASSERT_FALSE(wl_tensor_is_close(clone, clone2, FLT_EPSILON, nullptr));
    wl_tensor_fill(clone, 0.0f);
    wl_tensor_fill(clone2, 0.0f);
    double percent = 0.0;
    ASSERT_TRUE(wl_tensor_is_close(clone, clone2, FLT_EPSILON, & percent));
    ASSERT_DOUBLE_EQ(percent, 100.0);
    wl_tensor_fill(clone2, 1.0f);
    ASSERT_FALSE(wl_tensor_is_close(clone, clone2, FLT_EPSILON, & percent));
    ASSERT_DOUBLE_EQ(percent, 0.0);

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, copy_buffer_from) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    std::array<float, 2*2*2*2> buf {};
    for (auto& x : buf) x = 2.5f;

    wl_tensor_t* tensor = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 2, 2, 2, 2);
    ASSERT_EQ(wl_tensor_data_size(tensor), sizeof(buf));
    ASSERT_EQ(wl_tensor_num_elements(tensor), buf.size());
    wl_tensor_copy_buffer_from(tensor, buf.data(), sizeof(buf));

    const void* a = wl_tensor_data(tensor);
    const void* b = buf.data();
    ASSERT_NE(a, b);
    ASSERT_EQ(0, std::memcmp(a, b, wl_tensor_data_size(tensor)));

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, fill) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* tensor = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 8, 10, 11, 2);
    float* buf = wl_tensor_data_as_f32(tensor);

    wl_tensor_fill(tensor, 0.0f);
    for (std::int64_t i=0; i < wl_tensor_data_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], 0.0f);
    }

    wl_tensor_fill(tensor, 2.5f);
    for (std::int64_t i=0; i < wl_tensor_data_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], 2.5f);
    }

    wl_tensor_fill(tensor, -1.0f);
    for (std::int64_t i=0; i < wl_tensor_data_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], -1.0f);
    }

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, random_pcg) {
    constexpr float rmin = 0.0;
    constexpr float rmax = 1.0;
    return; //TODO fix this test

    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    wl_ctx_set_prng_algorithm(ctx, WL_PRNG_PCG, std::bit_cast<std::uint64_t>(this));

    wl_tensor_t* tensor = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 128, 128, 128, 128);
    wl_tensor_fill_random(tensor, rmin, rmax);

    auto* buf = wl_tensor_data_as_f32(tensor);
    std::vector<float> set {};
    set.reserve(wl_tensor_num_elements(tensor));

    for (int64_t i = 0; i < wl_tensor_num_elements(tensor); ++i) {
        float x = buf[i];
        ASSERT_GT(x, rmin);
        ASSERT_LT(x, rmax);
        for (float y : set) {
            bool is_ne = std::abs(x - y) > std::numeric_limits<float>::epsilon(); // |x-y| > eps
            if (!is_ne) {
                std::cout << "i:" << i << " x: " << x << " y: " << y << std::endl;
            }
            ASSERT_TRUE(is_ne);
        }
        set.emplace_back(x);
    }

    wl_ctx_destroy(ctx);
}

TEST(wl_tensor_t, random_mersenne) {
    constexpr float rmin = 0.0;
    constexpr float rmax = 1.0;
    return; //TODO fix this test

    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    wl_ctx_set_prng_algorithm(ctx, WL_PRNG_MERSENNE_TWISTER, std::bit_cast<std::uint64_t>(this));

    wl_tensor_t* tensor = wl_tensor_create_4d(ctx, WL_DTYPE_F32, 128, 128, 128, 128);
    wl_tensor_fill_random(tensor, rmin, rmax);

    auto* buf = wl_tensor_data_as_f32(tensor);
    std::vector<float> set {};
    set.reserve(wl_tensor_num_elements(tensor));

    for (int64_t i = 0; i < wl_tensor_num_elements(tensor); ++i) {
        float x = buf[i];
        ASSERT_GT(x, rmin);
        ASSERT_LT(x, rmax);
        for (float y : set) {
            bool is_ne = std::abs(x - y) > std::numeric_limits<float>::epsilon(); // |x-y| > eps
            if (!is_ne) {
                std::cout << "i:" << i << " x: " << x << " y: " << y << std::endl;
            }
            ASSERT_TRUE(is_ne);
        }
        set.emplace_back(x);
    }

    wl_ctx_destroy(ctx);
}
