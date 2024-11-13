// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(image, load) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* img = wl_tensor_load_image(ctx, "test_data/test_img.png", WL_COLOR_CHANNELS_RGB, 0, 0);
    wl_tensor_print(img, true, true);
    ASSERT_EQ(wl_tensor_image_channels(img), 3);
    ASSERT_EQ(wl_tensor_image_width(img), 2);
    ASSERT_EQ(wl_tensor_image_height(img), 4);
    ASSERT_EQ(wl_tensor_shape(img)[2], wl_tensor_image_width(img));
    ASSERT_EQ(wl_tensor_shape(img)[1], wl_tensor_image_height(img));
    ASSERT_EQ(wl_tensor_shape(img)[0], wl_tensor_image_channels(img)); // RGB

    auto* buf = wl_tensor_data_as_f32(img);
    for (int64_t i=0; i < wl_tensor_num_elements(img); ++i) {
        ASSERT_GE(buf[i], 0.0f);
        ASSERT_LE(buf[i], 1.0f);
    }

    // TODO: check data

    wl_ctx_destroy(ctx);
}

TEST(image, load_resize) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* img = wl_tensor_load_image(ctx, "test_data/car.jpg", WL_COLOR_CHANNELS_RGB, 256, 211);
    ASSERT_EQ(wl_tensor_shape(img)[2], 256);
    ASSERT_EQ(wl_tensor_shape(img)[1], 211);
    ASSERT_EQ(wl_tensor_shape(img)[0], 3); // RGB
    ASSERT_EQ(wl_tensor_shape(img)[2], wl_tensor_image_width(img));
    ASSERT_EQ(wl_tensor_shape(img)[1], wl_tensor_image_height(img));
    ASSERT_EQ(wl_tensor_shape(img)[0], wl_tensor_image_channels(img)); // RGB

    // wl_tensor_save_image(img, "test_data/car_resized.jpg");

    auto* buf = wl_tensor_data_as_f32(img);
    for (int64_t i=0; i < wl_tensor_num_elements(img); ++i) {
        ASSERT_GE(buf[i], 0.0f);
        ASSERT_LE(buf[i], 1.0f);
    }

    wl_ctx_destroy(ctx);
}

TEST(image, draw_box) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* img = wl_tensor_load_image(ctx, "test_data/car.jpg", WL_COLOR_CHANNELS_RGB, 256, 256);
    wl_tensor_img_draw_box(img, 40, 40, 80, 80, 1, wl_pack_color_f32(1.0f, 0.0f, 0.0f));
    wl_tensor_img_draw_box(img, 120, 150, 160, 200, 4, wl_pack_color_f32(1.0f, 1.0f, 1.0f));
    wl_tensor_print(img, true, false);
    //wl_tensor_save_image(img, "test_data/car2.jpg");

    wl_ctx_destroy(ctx);
}

TEST(image, draw_text) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);

    wl_tensor_t* img = wl_tensor_load_image(ctx, "test_data/car.jpg", WL_COLOR_CHANNELS_RGB, 256, 256);
    wl_tensor_img_draw_text(img, 100, 100, 10, 0xffffff, "Hallö!");
    wl_tensor_img_draw_text(img, 100, 200, 15, 0xffffff, "I want Pizza Salami! I want Pizza Salami! I want Pizza Salami! I want Pizza Salami!");
    wl_tensor_print(img, true, false);
    wl_tensor_save_image(img, "test_data/car3.jpg");

    wl_ctx_destroy(ctx);
}
