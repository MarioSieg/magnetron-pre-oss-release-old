// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include <gtest/gtest.h>
#include <msml.h>
#include <unordered_set>

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
    ASSERT_EQ(msml_tensor_buf_size(tensor), 10*sizeof(float));
    ASSERT_EQ(msml_tensor_buf_len(tensor), 10);
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 1);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*sizeof(float));
    ASSERT_FALSE(msml_tensor_is_scalar(tensor));
    ASSERT_TRUE(msml_tensor_is_vector(tensor));
    ASSERT_TRUE(msml_tensor_is_matrix(tensor));
    ASSERT_TRUE(msml_tensor_is_higher_order_3d(tensor));

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
    ASSERT_EQ(msml_tensor_buf_size(tensor), 10*4*sizeof(float));
    ASSERT_EQ(msml_tensor_buf_len(tensor), 10*4);
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 4);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*4*sizeof(float));
    ASSERT_FALSE(msml_tensor_is_scalar(tensor));
    ASSERT_FALSE(msml_tensor_is_vector(tensor));
    ASSERT_TRUE(msml_tensor_is_matrix(tensor));
    ASSERT_TRUE(msml_tensor_is_higher_order_3d(tensor));

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
    ASSERT_EQ(msml_tensor_buf_size(tensor), 10*4*2*sizeof(float));
    ASSERT_EQ(msml_tensor_buf_len(tensor), 10*4*2);
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 8);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*4*2*sizeof(float));
    ASSERT_FALSE(msml_tensor_is_scalar(tensor));
    ASSERT_FALSE(msml_tensor_is_vector(tensor));
    ASSERT_FALSE(msml_tensor_is_matrix(tensor));
    ASSERT_TRUE(msml_tensor_is_higher_order_3d(tensor));

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
    ASSERT_EQ(msml_tensor_buf_size(tensor), 10*4*2*5*sizeof(float));
    ASSERT_EQ(msml_tensor_buf_len(tensor), 10*4*2*5);
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 40);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*4*2*sizeof(float));
    ASSERT_FALSE(msml_tensor_is_scalar(tensor));
    ASSERT_FALSE(msml_tensor_is_vector(tensor));
    ASSERT_FALSE(msml_tensor_is_matrix(tensor));
    ASSERT_FALSE(msml_tensor_is_higher_order_3d(tensor));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, print) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 2, 2, 2, 2);
    msml_tensor_fill_random(tensor, 0.0f, 1.0f);
    msml_tensor_print(tensor, true);

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, name) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 2, 2, 2, 2);
    msml_tensor_set_name(tensor, "Gradient Backup");
    ASSERT_STREQ(msml_tensor_get_name(tensor), "Gradient Backup");

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, op) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 2, 2, 2, 2);
    msml_tensor_set_op(tensor, MSML_OP_MATMUL);
    ASSERT_EQ(msml_tensor_get_op(tensor), MSML_OP_MATMUL);

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, arg_getset) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_tensor_t* arg1 = msml_tensor_create_1d(ctx, MSML_DTYPE_F32, 10);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 2, 2, 2, 2);
    msml_tensor_set_arg(tensor, 0, arg1);
    ASSERT_EQ(msml_tensor_get_arg(tensor, 0), arg1);

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, arg_invalid_slot) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_tensor_t* arg1 = msml_tensor_create_1d(ctx, MSML_DTYPE_F32, 10);
    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 2, 2, 2, 2);

    //ASSERT_DEATH_IF_SUPPORTED(msml_tensor_set_arg(tensor, 128, arg1), {});

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, arg_slot_used) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_tensor_t* arg1 = msml_tensor_create_1d(ctx, MSML_DTYPE_F32, 10);
    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 2, 2, 2, 2);

    msml_tensor_set_arg(tensor, 0, arg1);
    //ASSERT_DEATH_IF_SUPPORTED(msml_tensor_set_arg(tensor, 0, arg1), {});

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, deep_clone) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* origin = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 10, 4, 2, 5);
    msml_tensor_fill_random(origin, -1.0f, 1.0f);
    msml_tensor_t* clone = msml_tensor_deep_clone(origin);
    ASSERT_NE(origin, clone);
    ASSERT_EQ(msml_tensor_rank(origin), msml_tensor_rank(clone));
    ASSERT_EQ(msml_tensor_shape(origin)[0], msml_tensor_shape(clone)[0]);
    ASSERT_EQ(msml_tensor_shape(origin)[1], msml_tensor_shape(clone)[1]);
    ASSERT_EQ(msml_tensor_shape(origin)[2], msml_tensor_shape(clone)[2]);
    ASSERT_EQ(msml_tensor_shape(origin)[3], msml_tensor_shape(clone)[3]);
    ASSERT_EQ(msml_tensor_buf_size(origin), msml_tensor_buf_size(clone));
    ASSERT_EQ(msml_tensor_buf_len(origin), msml_tensor_buf_len(clone));
    ASSERT_EQ(msml_tensor_num_cols(origin), msml_tensor_num_cols(clone));
    ASSERT_EQ(msml_tensor_num_rows(origin), msml_tensor_num_rows(clone));
    ASSERT_EQ(msml_tensor_strides(origin)[0], msml_tensor_strides(clone)[0]);
    ASSERT_EQ(msml_tensor_strides(origin)[1], msml_tensor_strides(clone)[1]);
    ASSERT_EQ(msml_tensor_strides(origin)[2], msml_tensor_strides(clone)[2]);
    ASSERT_EQ(msml_tensor_strides(origin)[3], msml_tensor_strides(clone)[3]);
    ASSERT_TRUE(msml_tensor_is_shape_eq(origin, clone));
    ASSERT_TRUE(msml_tensor_are_strides_eq(origin, clone));

    const void* a = msml_tensor_buf(origin);
    const void* b = msml_tensor_buf(clone);
    ASSERT_NE(a, b);
    ASSERT_EQ(0, std::memcmp(a, b, msml_tensor_buf_size(origin)));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, isomorphic_clone) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* origin = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 10, 4, 2, 5);
    msml_tensor_fill_random(origin, -1.0f, 1.0f);
    msml_tensor_t* clone = msml_tensor_isomorphic_clone(origin);
    ASSERT_NE(origin, clone);
    ASSERT_EQ(msml_tensor_rank(origin), msml_tensor_rank(clone));
    ASSERT_EQ(msml_tensor_shape(origin)[0], msml_tensor_shape(clone)[0]);
    ASSERT_EQ(msml_tensor_shape(origin)[1], msml_tensor_shape(clone)[1]);
    ASSERT_EQ(msml_tensor_shape(origin)[2], msml_tensor_shape(clone)[2]);
    ASSERT_EQ(msml_tensor_shape(origin)[3], msml_tensor_shape(clone)[3]);
    ASSERT_EQ(msml_tensor_buf_size(origin), msml_tensor_buf_size(clone));
    ASSERT_EQ(msml_tensor_buf_len(origin), msml_tensor_buf_len(clone));
    ASSERT_EQ(msml_tensor_num_cols(origin), msml_tensor_num_cols(clone));
    ASSERT_EQ(msml_tensor_num_rows(origin), msml_tensor_num_rows(clone));
    ASSERT_EQ(msml_tensor_strides(origin)[0], msml_tensor_strides(clone)[0]);
    ASSERT_EQ(msml_tensor_strides(origin)[1], msml_tensor_strides(clone)[1]);
    ASSERT_EQ(msml_tensor_strides(origin)[2], msml_tensor_strides(clone)[2]);
    ASSERT_EQ(msml_tensor_strides(origin)[3], msml_tensor_strides(clone)[3]);
    ASSERT_TRUE(msml_tensor_is_shape_eq(origin, clone));
    ASSERT_TRUE(msml_tensor_are_strides_eq(origin, clone));

    const void* a = msml_tensor_buf(origin);
    const void* b = msml_tensor_buf(clone);
    ASSERT_NE(a, b);
    ASSERT_NE(0, std::memcmp(a, b, msml_tensor_buf_size(origin)));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, equals) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* origin = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 10, 4, 2, 5);
    msml_tensor_fill_random(origin, -1.0f, 1.0f);
    msml_tensor_t* clone = msml_tensor_deep_clone(origin);
    msml_tensor_t* clone2 = msml_tensor_isomorphic_clone(origin);
    msml_tensor_fill_random(clone2, 0.0f, 1.0f);
    ASSERT_TRUE(msml_tensor_eq(origin, clone));
    ASSERT_FALSE(msml_tensor_eq(origin, clone2));
    ASSERT_FALSE(msml_tensor_eq(clone, clone2));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, isclose) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* origin = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 10, 4, 2, 5);
    msml_tensor_fill_random(origin, -1.0f, 1.0f);
    msml_tensor_t* clone = msml_tensor_deep_clone(origin);
    msml_tensor_t* clone2 = msml_tensor_isomorphic_clone(origin);
    msml_tensor_fill_random(clone2, 0.0f, 1.0f);
    ASSERT_TRUE(msml_tensor_isclose(origin, clone, FLT_EPSILON, nullptr));
    ASSERT_FALSE(msml_tensor_isclose(origin, clone2, FLT_EPSILON, nullptr));
    ASSERT_FALSE(msml_tensor_isclose(clone, clone2, FLT_EPSILON, nullptr));
    msml_tensor_fill(clone, 0.0f);
    msml_tensor_fill(clone2, 0.0f);
    double percent = 0.0;
    ASSERT_TRUE(msml_tensor_isclose(clone, clone2, FLT_EPSILON, &percent));
    ASSERT_DOUBLE_EQ(percent, 100.0);
    msml_tensor_fill(clone2, 1.0f);
    ASSERT_FALSE(msml_tensor_isclose(clone, clone2, FLT_EPSILON, &percent));
    ASSERT_DOUBLE_EQ(percent, 0.0);

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, copy_buffer_from) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    std::array<float, 2*2*2*2> data {};
    std::fill(data.begin(), data.end(), 2.5);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 2, 2, 2, 2);
    ASSERT_EQ(msml_tensor_buf_size(tensor), sizeof(data));
    ASSERT_EQ(msml_tensor_buf_len(tensor), data.size());
    msml_tensor_copy_buffer_from(tensor, data.data(), sizeof(data));

    const void* a = msml_tensor_buf(tensor);
    const void* b = data.data();
    ASSERT_NE(a, b);
    ASSERT_EQ(0, std::memcmp(a, b, msml_tensor_buf_size(tensor)));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, fill) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 8, 10, 11, 2);
    float* buf = msml_tensor_buf_f32(tensor);

    msml_tensor_fill(tensor, 0.0f);
    for (std::int64_t i=0; i < msml_tensor_buf_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], 0.0f);
    }

    msml_tensor_fill(tensor, 2.5f);
    for (std::int64_t i=0; i < msml_tensor_buf_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], 2.5f);
    }

    msml_tensor_fill(tensor, -1.0f);
    for (std::int64_t i=0; i < msml_tensor_buf_size(tensor); ++i) {
        ASSERT_FLOAT_EQ(buf[0], -1.0f);
    }

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, random_pcg) {
    constexpr float rmin = 0.0;
    constexpr float rmax = 1.0;

    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_ctx_set_prng_algorithm(ctx, MSML_PRNG_PCG, std::bit_cast<std::uint64_t>(this));

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 128, 128, 128, 128);
    msml_tensor_fill_random(tensor, rmin, rmax);

    auto* buf = msml_tensor_buf_f32(tensor);
    std::vector<float> set {};
    set.reserve(msml_tensor_buf_len(tensor));

    for (size_t i = 0; i < msml_tensor_buf_len(tensor); ++i) {
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

    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_ctx_set_prng_algorithm(ctx, MSML_PRNG_MERSENNE_TWISTER, std::bit_cast<std::uint64_t>(this));

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 128, 128, 128, 128);
    msml_tensor_fill_random(tensor, rmin, rmax);

    auto* buf = msml_tensor_buf_f32(tensor);
    std::vector<float> set {};
    set.reserve(msml_tensor_buf_len(tensor));

    for (size_t i = 0; i < msml_tensor_buf_len(tensor); ++i) {
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
