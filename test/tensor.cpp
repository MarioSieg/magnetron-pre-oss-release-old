// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"
#include <array>
#include <cstring>
#include <unordered_set>
#include <filesystem>

TEST(msml_tensor_t, init_1d) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_1d(ctx, MSML_DTYPE_F32, 10);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(msml_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(msml_tensor_dtype(tensor), MSML_DTYPE_F32);
    ASSERT_EQ(msml_tensor_rank(tensor), 1);
    ASSERT_EQ(msml_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(msml_tensor_shape(tensor)[1], 1);
    ASSERT_EQ(msml_tensor_shape(tensor)[2], 1);
    ASSERT_EQ(msml_tensor_shape(tensor)[3], 1);
    ASSERT_EQ(msml_tensor_data_size(tensor), 10 * sizeof(float));
    ASSERT_EQ(msml_tensor_num_elements(tensor), 10);
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 1);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*sizeof(float));
    ASSERT_FALSE(msml_tensor_is_scalar(tensor));
    ASSERT_TRUE(msml_tensor_is_vector(tensor));
    ASSERT_TRUE(msml_tensor_is_matrix(tensor));
    ASSERT_TRUE(msml_tensor_is_volume(tensor));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, init_2d) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 10, 4);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(msml_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(msml_tensor_dtype(tensor), MSML_DTYPE_F32);
    ASSERT_EQ(msml_tensor_rank(tensor), 2);
    ASSERT_EQ(msml_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(msml_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(msml_tensor_shape(tensor)[2], 1);
    ASSERT_EQ(msml_tensor_shape(tensor)[3], 1);
    ASSERT_EQ(msml_tensor_data_size(tensor), 10 * 4 * sizeof(float));
    ASSERT_EQ(msml_tensor_num_elements(tensor), 10 * 4);
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 4);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*4*sizeof(float));
    ASSERT_FALSE(msml_tensor_is_scalar(tensor));
    ASSERT_FALSE(msml_tensor_is_vector(tensor));
    ASSERT_TRUE(msml_tensor_is_matrix(tensor));
    ASSERT_TRUE(msml_tensor_is_volume(tensor));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, init_3d) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_3d(ctx, MSML_DTYPE_F32, 10, 4, 2);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(msml_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(msml_tensor_dtype(tensor), MSML_DTYPE_F32);
    ASSERT_EQ(msml_tensor_rank(tensor), 3);
    ASSERT_EQ(msml_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(msml_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(msml_tensor_shape(tensor)[2], 2);
    ASSERT_EQ(msml_tensor_shape(tensor)[3], 1);
    ASSERT_EQ(msml_tensor_data_size(tensor), 10 * 4 * 2 * sizeof(float));
    ASSERT_EQ(msml_tensor_num_elements(tensor), 10 * 4 * 2);
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 8);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*4*2*sizeof(float));
    ASSERT_FALSE(msml_tensor_is_scalar(tensor));
    ASSERT_FALSE(msml_tensor_is_vector(tensor));
    ASSERT_FALSE(msml_tensor_is_matrix(tensor));
    ASSERT_TRUE(msml_tensor_is_volume(tensor));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, init_4d) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 10, 4, 2, 5);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(msml_tensor_get_ctx(tensor), ctx);
    ASSERT_EQ(msml_tensor_dtype(tensor), MSML_DTYPE_F32);
    ASSERT_EQ(msml_tensor_rank(tensor), 4);
    ASSERT_EQ(msml_tensor_shape(tensor)[0], 10);
    ASSERT_EQ(msml_tensor_shape(tensor)[1], 4);
    ASSERT_EQ(msml_tensor_shape(tensor)[2], 2);
    ASSERT_EQ(msml_tensor_shape(tensor)[3], 5);
    ASSERT_EQ(msml_tensor_data_size(tensor), 10 * 4 * 2 * 5 * sizeof(float));
    ASSERT_EQ(msml_tensor_num_elements(tensor), 10 * 4 * 2 * 5);
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 40);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*4*2*sizeof(float));
    ASSERT_FALSE(msml_tensor_is_scalar(tensor));
    ASSERT_FALSE(msml_tensor_is_vector(tensor));
    ASSERT_FALSE(msml_tensor_is_matrix(tensor));
    ASSERT_FALSE(msml_tensor_is_volume(tensor));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, print) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 2, 2, 2, 2);
    msml_tensor_fill_random(tensor, 0.0f, 1.0f);
    msml_tensor_print(tensor, false, true);

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, name) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 2, 2, 2, 2);
    msml_tensor_set_name(tensor, "Gradient Backup");
    ASSERT_STREQ(msml_tensor_get_name(tensor), "Gradient Backup");

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, deep_clone) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* origin = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 10, 4, 2, 5);
    msml_tensor_fill_random(origin, -1.0f, 1.0f);
    msml_tensor_t* clone = msml_tensor_emit_op_va(ctx, MSML_OP_CLONE, origin);
    ASSERT_NE(origin, clone);
    ASSERT_EQ(msml_tensor_rank(origin), msml_tensor_rank(clone));
    ASSERT_EQ(msml_tensor_shape(origin)[0], msml_tensor_shape(clone)[0]);
    ASSERT_EQ(msml_tensor_shape(origin)[1], msml_tensor_shape(clone)[1]);
    ASSERT_EQ(msml_tensor_shape(origin)[2], msml_tensor_shape(clone)[2]);
    ASSERT_EQ(msml_tensor_shape(origin)[3], msml_tensor_shape(clone)[3]);
    ASSERT_EQ(msml_tensor_data_size(origin), msml_tensor_data_size(clone));
    ASSERT_EQ(msml_tensor_num_elements(origin), msml_tensor_num_elements(clone));
    ASSERT_EQ(msml_tensor_num_cols(origin), msml_tensor_num_cols(clone));
    ASSERT_EQ(msml_tensor_num_rows(origin), msml_tensor_num_rows(clone));
    ASSERT_EQ(msml_tensor_strides(origin)[0], msml_tensor_strides(clone)[0]);
    ASSERT_EQ(msml_tensor_strides(origin)[1], msml_tensor_strides(clone)[1]);
    ASSERT_EQ(msml_tensor_strides(origin)[2], msml_tensor_strides(clone)[2]);
    ASSERT_EQ(msml_tensor_strides(origin)[3], msml_tensor_strides(clone)[3]);
    ASSERT_TRUE(msml_tensor_is_shape_eq(origin, clone));
    ASSERT_TRUE(msml_tensor_are_strides_eq(origin, clone));

    const void* a = msml_tensor_data(origin);
    const void* b = msml_tensor_data(clone);
    ASSERT_NE(a, b);
    ASSERT_EQ(0, std::memcmp(a, b, msml_tensor_data_size(origin)));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, equals) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* origin = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 10, 4, 2, 5);
    msml_tensor_fill_random(origin, -1.0f, 1.0f);
    msml_tensor_t* clone = msml_tensor_emit_op_va(ctx, MSML_OP_CLONE, origin);
    msml_tensor_t* clone2 = msml_tensor_emit_op_va(ctx, MSML_OP_CLONE, origin);
    msml_tensor_fill_random(clone2, 0.0f, 1.0f);
    ASSERT_TRUE(msml_tensor_eq(origin, clone));
    ASSERT_FALSE(msml_tensor_eq(origin, clone2));
    ASSERT_FALSE(msml_tensor_eq(clone, clone2));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, buffer_linearly) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* origin = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 1, 2, 3, 4);
    msml_tensor_fill(origin, 0.0f);
    msml_tensor_data_as_f32(origin)[0] = 1.0f;
    msml_tensor_data_as_f32(origin)[msml_tensor_num_elements(origin) - 1] = -1.0f;
    for (int64_t i=0; i < msml_tensor_num_elements(origin); ++i) {
        std::cout << msml_tensor_data_as_f32(origin)[i] << " ";
    }
    std::cout << std::endl;

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, view) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* origin = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 10, 4, 2, 5);
    msml_tensor_fill(origin, 2.0f);
    int64_t slice_dims[] = {10, 4, 2, 5};
    msml_tensor_t* slice1 = msml_tensor_emit_op_va(ctx, MSML_OP_VIEW, origin);
    ASSERT_EQ(msml_tensor_data(slice1), msml_tensor_data(origin));
    ASSERT_EQ(msml_tensor_data_size(slice1), msml_tensor_data_size(origin));
    ASSERT_EQ(msml_tensor_num_elements(slice1), msml_tensor_num_elements(origin));
    auto* buf = msml_tensor_data_as_f32(slice1);
    for (int64_t i=0; i < msml_tensor_num_elements(slice1); ++i) {
        ASSERT_FLOAT_EQ(buf[i], 2.0f);
    }
    msml_tensor_t* slice2 = msml_tensor_emit_op_va(ctx, MSML_OP_VIEW, origin);
    ASSERT_EQ(msml_tensor_data(slice2), static_cast<std::uint8_t*>(msml_tensor_data(origin)));
    ASSERT_EQ(msml_tensor_data_size(slice2), 10 * 4 * 2 * 5 * sizeof(float));
    ASSERT_EQ(msml_tensor_num_elements(slice2), 10 * 4 * 2 * 5);
    auto* buf_slice2 = msml_tensor_data_as_f32(slice2);
    for (int64_t i = 0; i < msml_tensor_num_elements(slice2); ++i) {
        ASSERT_FLOAT_EQ(buf_slice2[i], 2.0f);
    }

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, transpose) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* origin = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 4, 1);
    msml_tensor_fill_random(origin, -1.0f, 1.0f);
    msml_tensor_t* transposed = msml_tensor_emit_op_va(ctx, MSML_OP_TRANSPOSE, origin);
    ASSERT_FALSE(msml_tensor_is_transposed(origin));
    ASSERT_TRUE(msml_tensor_is_transposed(transposed));
    ASSERT_EQ(msml_tensor_shape(origin)[0], msml_tensor_shape(transposed)[1]);
    ASSERT_EQ(msml_tensor_shape(origin)[1], msml_tensor_shape(transposed)[0]);
    ASSERT_EQ(msml_tensor_data_size(origin), msml_tensor_data_size(transposed));
    ASSERT_EQ(msml_tensor_num_elements(origin), msml_tensor_num_elements(transposed));
    ASSERT_EQ(msml_tensor_num_cols(origin), msml_tensor_num_rows(transposed));
    ASSERT_EQ(msml_tensor_num_rows(origin), msml_tensor_num_cols(transposed));
    ASSERT_TRUE(msml_tensor_is_contiguous(origin));
    ASSERT_FALSE(msml_tensor_is_contiguous(transposed));
}

TEST(msml_tensor_t, permute) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_tensor_t* origin = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 4, 1);
    msml_tensor_fill_random(origin, -1.0f, 1.0f);
    msml_op_param_t params[MSML_MAX_OP_PARAMS] {
        msml_op_param_int(3),
        msml_op_param_int(2),
        msml_op_param_int(1),
        msml_op_param_int(0)
    };
    msml_tensor_t* permuted = msml_tensor_operator(ctx, MSML_OP_PERMUTE, &origin, 1, &params);
    ASSERT_FALSE(msml_tensor_is_transposed(origin));
    ASSERT_FALSE(msml_tensor_is_transposed(permuted));
    ASSERT_FALSE(msml_tensor_is_permuted(origin));
    ASSERT_TRUE(msml_tensor_is_permuted(permuted));
    ASSERT_EQ(msml_tensor_shape(origin)[0], msml_tensor_shape(permuted)[3]);
    ASSERT_EQ(msml_tensor_shape(origin)[1], msml_tensor_shape(permuted)[2]);
    ASSERT_EQ(msml_tensor_shape(origin)[2], msml_tensor_shape(permuted)[1]);
    ASSERT_EQ(msml_tensor_shape(origin)[3], msml_tensor_shape(permuted)[0]);
    ASSERT_EQ(msml_tensor_data_size(origin), msml_tensor_data_size(permuted));
    ASSERT_EQ(msml_tensor_num_elements(origin), msml_tensor_num_elements(permuted));
    ASSERT_EQ(msml_tensor_num_cols(origin), msml_tensor_num_rows(permuted));
    ASSERT_EQ(msml_tensor_num_rows(origin), msml_tensor_num_cols(permuted));
    ASSERT_TRUE(msml_tensor_is_contiguous(origin));
    ASSERT_FALSE(msml_tensor_is_contiguous(permuted));
}

TEST(msml_tensor_t, isclose) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* origin = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 10, 4, 2, 5);
    msml_tensor_fill_random(origin, -1.0f, 1.0f);
    msml_tensor_t* clone = msml_tensor_emit_op_va(ctx, MSML_OP_CLONE, origin);
    msml_tensor_t* clone2 = msml_tensor_emit_op_va(ctx, MSML_OP_CLONE, origin);
    msml_tensor_fill_random(clone2, 0.0f, 1.0f);
    ASSERT_TRUE(msml_tensor_is_close(origin, clone, FLT_EPSILON, nullptr));
    ASSERT_FALSE(msml_tensor_is_close(origin, clone2, FLT_EPSILON, nullptr));
    ASSERT_FALSE(msml_tensor_is_close(clone, clone2, FLT_EPSILON, nullptr));
    msml_tensor_fill(clone, 0.0f);
    msml_tensor_fill(clone2, 0.0f);
    double percent = 0.0;
    ASSERT_TRUE(msml_tensor_is_close(clone, clone2, FLT_EPSILON, & percent));
    ASSERT_DOUBLE_EQ(percent, 100.0);
    msml_tensor_fill(clone2, 1.0f);
    ASSERT_FALSE(msml_tensor_is_close(clone, clone2, FLT_EPSILON, & percent));
    ASSERT_DOUBLE_EQ(percent, 0.0);

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, copy_buffer_from) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    std::array<float, 2*2*2*2> buf {};
    for (auto& x : buf) x = 2.5f;

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 2, 2, 2, 2);
    ASSERT_EQ(msml_tensor_data_size(tensor), sizeof(buf));
    ASSERT_EQ(msml_tensor_num_elements(tensor), buf.size());
    msml_tensor_copy_buffer_from(tensor, buf.data(), sizeof(buf));

    const void* a = msml_tensor_data(tensor);
    const void* b = buf.data();
    ASSERT_NE(a, b);
    ASSERT_EQ(0, std::memcmp(a, b, msml_tensor_data_size(tensor)));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, fill) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 8, 10, 11, 2);
    float* buf = msml_tensor_data_as_f32(tensor);

    msml_tensor_fill(tensor, 0.0f);
    for (std::int64_t i=0; i < msml_tensor_data_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], 0.0f);
    }

    msml_tensor_fill(tensor, 2.5f);
    for (std::int64_t i=0; i < msml_tensor_data_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], 2.5f);
    }

    msml_tensor_fill(tensor, -1.0f);
    for (std::int64_t i=0; i < msml_tensor_data_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], -1.0f);
    }

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, random_pcg) {
    constexpr float rmin = 0.0;
    constexpr float rmax = 1.0;
    return; //TODO fix this test

    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_ctx_set_prng_algorithm(ctx, MSML_PRNG_PCG, std::bit_cast<std::uint64_t>(this));

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 128, 128, 128, 128);
    msml_tensor_fill_random(tensor, rmin, rmax);

    auto* buf = msml_tensor_data_as_f32(tensor);
    std::vector<float> set {};
    set.reserve(msml_tensor_num_elements(tensor));

    for (int64_t i = 0; i < msml_tensor_num_elements(tensor); ++i) {
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

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, random_mersenne) {
    constexpr float rmin = 0.0;
    constexpr float rmax = 1.0;
    return; //TODO fix this test

    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_ctx_set_prng_algorithm(ctx, MSML_PRNG_MERSENNE_TWISTER, std::bit_cast<std::uint64_t>(this));

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 128, 128, 128, 128);
    msml_tensor_fill_random(tensor, rmin, rmax);

    auto* buf = msml_tensor_data_as_f32(tensor);
    std::vector<float> set {};
    set.reserve(msml_tensor_num_elements(tensor));

    for (int64_t i = 0; i < msml_tensor_num_elements(tensor); ++i) {
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

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, load_from_image) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    constexpr const char* image = "test_data/car.jpg";
    ASSERT_TRUE(std::filesystem::exists(image));

    msml_tensor_t* t = msml_tensor_create_from_image(ctx, image, MSML_COLOR_CHANNELS_RGB, 0, 0);
    ASSERT_EQ(msml_tensor_shape(t)[0], 1536);
    ASSERT_EQ(msml_tensor_shape(t)[1], 2048);
    ASSERT_EQ(msml_tensor_shape(t)[2], 3); // RGB
    ASSERT_EQ(msml_tensor_shape(t)[0], msml_tensor_image_width(t));
    ASSERT_EQ(msml_tensor_shape(t)[1], msml_tensor_image_height(t));
    ASSERT_EQ(msml_tensor_shape(t)[2], msml_tensor_image_channels(t)); // RGB

    auto* buf = msml_tensor_data_as_f32(t);
    for (int64_t i=0; i < msml_tensor_num_elements(t); ++i) {
        ASSERT_GE(buf[i], 0.0f);
        ASSERT_LE(buf[i], 1.0f);
    }

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, save_and_load_from_image) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    constexpr const char* image = "test_data/random.jpg";
    if (std::filesystem::exists(image)) {
        std::filesystem::remove(image);
    }

    msml_tensor_t* t = msml_tensor_create_3d(ctx, MSML_DTYPE_F32, 4, 4, 3);
    msml_tensor_fill_random(t, 0.0f, 1.0f);
    msml_tensor_save_to_image(t, image);

    ASSERT_TRUE(std::filesystem::exists(image));

    msml_tensor_t* t2 = msml_tensor_create_from_image(ctx, image, MSML_COLOR_CHANNELS_RGB, 0, 0);
    ASSERT_TRUE(msml_tensor_is_shape_eq(t, t2));
    ASSERT_TRUE(msml_tensor_are_strides_eq(t, t2));
    ASSERT_EQ(msml_tensor_data_size(t), msml_tensor_data_size(t2));
    const auto* t_b = msml_tensor_data_as_f32(t);
    const auto* t2_b = msml_tensor_data_as_f32(t);
    for (int64_t i=0; i < msml_tensor_num_elements(t); ++i) {
        bool is_ok = std::abs(t_b[i]-t2_b[i]) <= std::numeric_limits<float>::epsilon();
        if (!is_ok) {
            std::cout << "i: " << i << "x: " << t_b[i] << " y: " << t2_b[i] << " err: " << std::abs(t_b[i]-t2_b[i]) << std::endl;
        }
        ASSERT_TRUE(is_ok);
    }

    msml_ctx_destroy(ctx);

    std::filesystem::remove(image);
}
