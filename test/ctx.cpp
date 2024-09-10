#include <gtest/gtest.h>
#include <msml.h>

TEST(msml_ctx_t, create_destroy) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    ASSERT_NE(ctx, nullptr);
    msml_ctx_destroy(ctx);
}

TEST(msml_ctx_t, alloc_pool_int) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    int* ptr = static_cast<int*>(msml_ctx_pool_alloc(ctx, sizeof(int)));
    ASSERT_NE(ptr, nullptr);
    *ptr = -1;
    ASSERT_EQ(*ptr, -1);
    msml_ctx_destroy(ctx);
}

TEST(msml_ctx_t, alloc_pool_int_aligned) {
    msml_ctx_t* ctx = msml_ctx_create(nullptr);
    int* ptr = static_cast<int*>(msml_ctx_pool_alloc_aligned(ctx, sizeof(int), 512));
    ASSERT_NE(ptr, nullptr);
    *ptr = -1;
    ASSERT_EQ(*ptr, -1);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 512, 0);
    msml_ctx_destroy(ctx);
}

TEST(msml_ctx_t, alloc_pool_huge) {
    msml_ctx_info_t info {};
    info.pool_chunk_size = 1;
    info.pool_chunks_cap = 1;
    msml_ctx_t* ctx = msml_ctx_create(&info);
    for (int i = 1; i <= 32; ++i) {
        int* ptr = static_cast<int*>(msml_ctx_pool_alloc(ctx, sizeof(int) * 0xffff * i));
        ASSERT_NE(ptr, nullptr);
        for (int j = 0; j < 0xffff * i; ++j) {
            ptr[j] = j;
            ASSERT_EQ(ptr[j], j);
        }
    }
    msml_ctx_destroy(ctx);
}
