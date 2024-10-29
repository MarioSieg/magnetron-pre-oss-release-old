// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(image, load) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* img = msml_tensor_load_image(ctx, "test_data/test_img.png", MSML_COLOR_CHANNELS_RGB, 0, 0);
    msml_tensor_print(img, true, true);
    ASSERT_EQ(msml_tensor_image_channels(img), 3);
    ASSERT_EQ(msml_tensor_image_width(img), 2);
    ASSERT_EQ(msml_tensor_image_height(img), 4);
    ASSERT_EQ(msml_tensor_shape(img)[2], msml_tensor_image_width(img));
    ASSERT_EQ(msml_tensor_shape(img)[1], msml_tensor_image_height(img));
    ASSERT_EQ(msml_tensor_shape(img)[0], msml_tensor_image_channels(img)); // RGB

    auto* buf = msml_tensor_data_as_f32(img);
    for (int64_t i=0; i < msml_tensor_num_elements(img); ++i) {
        ASSERT_GE(buf[i], 0.0f);
        ASSERT_LE(buf[i], 1.0f);
    }

    // TODO: check data

    msml_ctx_destroy(ctx);
}

TEST(image, load_resize) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* img = msml_tensor_load_image(ctx, "test_data/car.jpg", MSML_COLOR_CHANNELS_RGB, 256, 211);
    ASSERT_EQ(msml_tensor_shape(img)[2], 256);
    ASSERT_EQ(msml_tensor_shape(img)[1], 211);
    ASSERT_EQ(msml_tensor_shape(img)[0], 3); // RGB
    ASSERT_EQ(msml_tensor_shape(img)[2], msml_tensor_image_width(img));
    ASSERT_EQ(msml_tensor_shape(img)[1], msml_tensor_image_height(img));
    ASSERT_EQ(msml_tensor_shape(img)[0], msml_tensor_image_channels(img)); // RGB

    msml_tensor_save_image(img, "test_data/car_resized.jpg");

    auto* buf = msml_tensor_data_as_f32(img);
    for (int64_t i=0; i < msml_tensor_num_elements(img); ++i) {
        ASSERT_GE(buf[i], 0.0f);
        ASSERT_LE(buf[i], 1.0f);
    }

    msml_ctx_destroy(ctx);
}

TEST(image, draw_box) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* img = msml_tensor_load_image(ctx, "test_data/car.jpg", MSML_COLOR_CHANNELS_RGB, 256, 256);
    msml_tensor_img_draw_box(img, 40, 40, 80, 80, 1, msml_pack_color_f32(1.0f, 0.0f, 0.0f));
    msml_tensor_img_draw_box(img, 120, 150, 160, 200, 4, msml_pack_color_f32(1.0f, 1.0f, 1.0f));
    msml_tensor_print(img, true, false);
    //msml_tensor_save_image(img, "test_data/car2.jpg");

    msml_ctx_destroy(ctx);
}
