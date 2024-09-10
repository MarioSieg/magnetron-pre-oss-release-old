#include <gtest/gtest.h>
#include <msml.h>

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
