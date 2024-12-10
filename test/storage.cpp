// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

#include <filesystem>

TEST(storage, load_store) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* A = wl_tensor_create_6d(ctx, WL_DTYPE_F32, 10, 4, 2, 5, 2, 2);
    wl_tensor_fill_random_uniform(A, -1.0f, 1.0f);

    if (std::filesystem::exists("test_data/test.wavelet"))
        std::filesystem::remove("test_data/test.wavelet");
    wl_tensor_save(A, "test_data/test.wavelet");
    ASSERT_TRUE(std::filesystem::exists("test_data/test.wavelet"));
    wl_tensor_t* B = wl_tensor_load(ctx, "test_data/test.wavelet");
    ASSERT_EQ(wl_tensor_dtype(B), WL_DTYPE_F32);
    ASSERT_EQ(wl_tensor_rank(B), 6);
    ASSERT_EQ(wl_tensor_shape(B)[0], 10);
    ASSERT_EQ(wl_tensor_shape(B)[1], 4);
    ASSERT_EQ(wl_tensor_shape(B)[2], 2);
    ASSERT_EQ(wl_tensor_shape(B)[3], 5);
    ASSERT_EQ(wl_tensor_shape(B)[4], 2);
    ASSERT_EQ(wl_tensor_shape(B)[5], 2);
    ASSERT_EQ(wl_tensor_data_size(B), 10 * 4 * 2 * 5 * 2 * 2 * sizeof(float));
    ASSERT_TRUE(wl_tensor_eq(A, B));

    wl_tensor_decref(A);
    wl_tensor_decref(B);
    wl_ctx_destroy(ctx);
    ASSERT_TRUE(std::filesystem::remove("test_data/test.wavelet"));
}

TEST(storage, load_store_image) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* img = wl_tensor_load_image(ctx, "test_data/car.jpg", WL_COLOR_CHANNELS_RGB, 0, 0);
    if (std::filesystem::exists("test_data/car.wavelet"))
        std::filesystem::remove("test_data/car.wavelet");
    wl_tensor_save(img, "test_data/car.wavelet");
    ASSERT_TRUE(std::filesystem::exists("test_data/car.wavelet"));
    wl_tensor_t* B = wl_tensor_load(ctx, "test_data/car.wavelet");
    ASSERT_EQ(wl_tensor_dtype(B), WL_DTYPE_F32);
    ASSERT_EQ(wl_tensor_shape(B)[2], 1536);
    ASSERT_EQ(wl_tensor_shape(B)[1], 2048);
    ASSERT_EQ(wl_tensor_shape(B)[0], 3);
    ASSERT_TRUE(wl_tensor_eq(img, B));
    //wl_tensor_save_image(B, "test_data/car_from_wavelet.jpg");
    wl_tensor_decref(img);
    wl_tensor_decref(B);
    wl_ctx_destroy(ctx);
    ASSERT_TRUE(std::filesystem::remove("test_data/car.wavelet"));
}
