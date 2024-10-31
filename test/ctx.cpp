// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#include "prelude.hpp"

TEST(ctx, create_destroy) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    ASSERT_NE(ctx, nullptr);
    ASSERT_NE(0, wl_ctx_total_allocated_pool_memory(ctx));
    wl_ctx_destroy(ctx);
}

TEST(ctx, alloc_pool_int) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    int* ptr = static_cast<int*>(wl_ctx_pool_alloc(ctx, sizeof(int)));
    ASSERT_NE(ptr, nullptr);
    *ptr = -1;
    ASSERT_EQ(*ptr, -1);
    wl_ctx_destroy(ctx);
}

TEST(ctx, alloc_pool_int_aligned) {
    wl_ctx_t* ctx = wl_ctx_create(nullptr);
    int* ptr = static_cast<int*>(wl_ctx_pool_alloc_aligned(ctx, sizeof(int), 512));
    ASSERT_NE(ptr, nullptr);
    *ptr = -1;
    ASSERT_EQ(*ptr, -1);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 512, 0);
    wl_ctx_destroy(ctx);
}

TEST(ctx, alloc_pool_huge) {
    wl_ctx_info_t info {};
    info.pool_chunk_size = 1;
    info.pool_chunks_cap = 1;
    wl_ctx_t* ctx = wl_ctx_create(&info);
    for (int i = 1; i <= 32; ++i) {
        int* ptr = static_cast<int*>(wl_ctx_pool_alloc(ctx, sizeof(int) * 0xffff * i));
        ASSERT_NE(ptr, nullptr);
        for (int j = 0; j < 0xffff * i; ++j) {
            ptr[j] = j;
            ASSERT_EQ(ptr[j], j);
        }
    }
    wl_ctx_destroy(ctx);
}
