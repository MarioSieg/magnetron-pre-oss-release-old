// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

#include <filesystem>

TEST(storage, load) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* A = msml_tensor_create_4d(ctx, MSML_DTYPE_F32, 10, 4, 2, 5);
    msml_tensor_fill_random(A, -1.0f, 1.0f);

    if (std::filesystem::exists("test_data/test.msml"))
        std::filesystem::remove("test_data/test.msml");
    msml_tensor_save(A, "test_data/test.msml");
    ASSERT_TRUE(std::filesystem::exists("test_data/test.msml"));
    msml_tensor_t* B = msml_tensor_load(ctx, "test_data/test.msml");
    ASSERT_EQ(msml_tensor_dtype(B), MSML_DTYPE_F32);
    ASSERT_EQ(msml_tensor_rank(B), 4);
    ASSERT_EQ(msml_tensor_shape(B)[0], 10);
    ASSERT_EQ(msml_tensor_shape(B)[1], 4);
    ASSERT_EQ(msml_tensor_shape(B)[2], 2);
    ASSERT_EQ(msml_tensor_shape(B)[3], 5);
    ASSERT_EQ(msml_tensor_data_size(B), 10 * 4 * 2 * 5 * sizeof(float));
    ASSERT_TRUE(msml_tensor_eq(A, B));
    msml_ctx_destroy(ctx);
    ASSERT_TRUE(std::filesystem::remove("test_data/test.msml"));
}

TEST(storage, load_store_image) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* img = msml_tensor_load_image(ctx, "test_data/car.jpg", MSML_COLOR_CHANNELS_RGB, 0, 0);
    if (std::filesystem::exists("test_data/car.msml"))
        std::filesystem::remove("test_data/car.msml");
    msml_tensor_save(img, "test_data/car.msml");
    ASSERT_TRUE(std::filesystem::exists("test_data/car.msml"));
    msml_tensor_t* B = msml_tensor_load(ctx, "test_data/car.msml");
    ASSERT_EQ(msml_tensor_dtype(B), MSML_DTYPE_F32);
    ASSERT_EQ(msml_tensor_shape(B)[2], 1536);
    ASSERT_EQ(msml_tensor_shape(B)[1], 2048);
    ASSERT_EQ(msml_tensor_shape(B)[0], 3);
    ASSERT_TRUE(msml_tensor_eq(img, B));
    //msml_tensor_save_image(B, "test_data/car_from_msml.jpg");
    msml_ctx_destroy(ctx);
    ASSERT_TRUE(std::filesystem::remove("test_data/car.msml"));
}
