#include <gtest/gtest.h>
#include <msml.h>
#include <unordered_set>

TEST(msml_tensor_t, init_1d) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_1d(ctx, MSML_DTYPE_F32, 10);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(msml_tensor_rank(tensor), 1);
    ASSERT_EQ(msml_tensor_dims(tensor)[0], 10);
    ASSERT_EQ(msml_tensor_dims(tensor)[1], 1);
    ASSERT_EQ(msml_tensor_dims(tensor)[2], 1);
    ASSERT_EQ(msml_tensor_dims(tensor)[3], 1);
    ASSERT_EQ(msml_tensor_buf_size(tensor), 10*sizeof(float));
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 1);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*sizeof(float));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, init_2d) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_2d(ctx, MSML_DTYPE_F32, 10, 4);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(msml_tensor_rank(tensor), 2);
    ASSERT_EQ(msml_tensor_dims(tensor)[0], 10);
    ASSERT_EQ(msml_tensor_dims(tensor)[1], 4);
    ASSERT_EQ(msml_tensor_dims(tensor)[2], 1);
    ASSERT_EQ(msml_tensor_dims(tensor)[3], 1);
    ASSERT_EQ(msml_tensor_buf_size(tensor), 10*4*sizeof(float));
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 4);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*4*sizeof(float));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, init_3d) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_3d(ctx, MSML_DTYPE_F32, 10, 4, 2);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(msml_tensor_rank(tensor), 3);
    ASSERT_EQ(msml_tensor_dims(tensor)[0], 10);
    ASSERT_EQ(msml_tensor_dims(tensor)[1], 4);
    ASSERT_EQ(msml_tensor_dims(tensor)[2], 2);
    ASSERT_EQ(msml_tensor_dims(tensor)[3], 1);
    ASSERT_EQ(msml_tensor_buf_size(tensor), 10*4*2*sizeof(float));
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 8);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*4*2*sizeof(float));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, init_4d) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 10, 4, 2, 5);
    ASSERT_NE(tensor, nullptr);
    ASSERT_EQ(msml_tensor_rank(tensor), 4);
    ASSERT_EQ(msml_tensor_dims(tensor)[0], 10);
    ASSERT_EQ(msml_tensor_dims(tensor)[1], 4);
    ASSERT_EQ(msml_tensor_dims(tensor)[2], 2);
    ASSERT_EQ(msml_tensor_dims(tensor)[3], 5);
    ASSERT_EQ(msml_tensor_buf_size(tensor), 10*4*2*5*sizeof(float));
    ASSERT_EQ(msml_tensor_num_cols(tensor), 10);
    ASSERT_EQ(msml_tensor_num_rows(tensor), 40);
    ASSERT_EQ(msml_tensor_strides(tensor)[0], sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[1], 10*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[2], 10*4*sizeof(float));
    ASSERT_EQ(msml_tensor_strides(tensor)[3], 10*4*2*sizeof(float));

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, random_tausworthe) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_ctx_set_prng_algorithm(ctx, MSML_PRNG_TAUSWORTHE, 0.0);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 128, 128, 128, 128);
    msml_tensor_fill_random(tensor, 0.0, 1.0);

    auto* buf = (float*)msml_tensor_buf(tensor);
    std::vector<float> set {};
    set.reserve(0xffff);

    for (size_t i = 0; i < 0xffff; ++i) {
        float x = buf[i];
        ASSERT_GE(x, 0.0);
        ASSERT_LE(x, 1.0);
        // todo: check for uniform distribution
        set.emplace_back(x);
    }

    msml_ctx_destroy(ctx);
}


TEST(msml_tensor_t, random_mersenne) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_ctx_set_prng_algorithm(ctx, MSML_PRNG_MERSENNE_TWISTER_64, 0.0);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 128, 128, 128, 128);
    msml_tensor_fill_random(tensor, 0.0, 1.0);

    auto* buf = (float*)msml_tensor_buf(tensor);
    std::vector<float> set {};
    set.reserve(0xffff);

    for (size_t i = 0; i < 0xffff; ++i) {
        float x = buf[i];
        ASSERT_GE(x, 0.0);
        ASSERT_LE(x, 1.0);
        // todo: check for uniform distribution
        set.emplace_back(x);
    }

    msml_ctx_destroy(ctx);
}

TEST(msml_tensor_t, random_switch_to_mersenne) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    msml_ctx_set_prng_algorithm(ctx, MSML_PRNG_TAUSWORTHE, 0.0);

    msml_tensor_t* tmp = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 128, 128, 128, 128);
    msml_tensor_fill_random(tmp, 0.0, 1.0);

    msml_ctx_set_prng_algorithm(ctx, MSML_PRNG_MERSENNE_TWISTER_64, 0.0);

    msml_tensor_t* tensor = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 128, 128, 128, 128);
    msml_tensor_fill_random(tensor, 0.0, 1.0);

    auto* buf = (float*)msml_tensor_buf(tensor);
    std::vector<float> set {};
    set.reserve(0xffff);

    for (size_t i = 0; i < 0xffff; ++i) {
        float x = buf[i];
        ASSERT_GE(x, 0.0);
        ASSERT_LE(x, 1.0);
        // todo: check for uniform distribution
        set.emplace_back(x);
    }

    msml_ctx_destroy(ctx);
}
