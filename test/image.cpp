// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"


TEST(image, load) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* img = msml_tensor_load_image(ctx, "test_data/test_img.png", MSML_COLOR_CHANNELS_RGB, 0, 0);
    msml_tensor_print(img, true, true);
    ASSERT_EQ(msml_tensor_image_channels(img), 3);
    ASSERT_EQ(msml_tensor_image_width(img), 2);
    ASSERT_EQ(msml_tensor_image_height(img), 4);

    // TODO: check data

    msml_ctx_destroy(ctx);
}

TEST(image, draw_box) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);

    msml_tensor_t* img = msml_tensor_load_image(ctx, "test_data/car.jpg", MSML_COLOR_CHANNELS_RGB, 256, 256);
    msml_tensor_img_draw_box(img, 40, 40, 80, 80, 1, msml_pack_color_f32(1.0f, 0.0f, 0.0f));
    msml_tensor_img_draw_box(img, 120, 150, 160, 200, 4, msml_pack_color_f32(1.0f, 1.0f, 1.0f));
    msml_tensor_print(img, true, false);
    msml_tensor_save_image(img, "test_data/car2.jpg");

    msml_ctx_destroy(ctx);
}