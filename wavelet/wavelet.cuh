/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#pragma once

#include <array>
#include <cstdint>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>

namespace wl {
    constexpr std::size_t max_devices = 32;
    constexpr std::uint32_t warp_size = 32;
    constexpr std::uint32_t max_streams = 8;

    extern "C" [[noreturn]] auto wl__panic(const char* msg, ...) -> void;
    #define wl_cu_chk(expr) \
         do { \
            if (auto rrr {(expr)}; rrr != cudaSuccess) [[unlikely]] { \
                wl__panic(#expr, __func__, __FILE__, __LINE__, cudaGetErrorString(rrr)); \
            } \
        } while (0)

    [[nodiscard]] static __device__ __forceinline__ auto warp_hsum(float x) -> float {
        #pragma unroll
        for (int mask = 16; mask; mask >>= 1)
            x += __shfl_xor_sync(0xffffffff, x, mask, 32);
        return x;
    }

    [[nodiscard]] static __device__ __forceinline__ auto warp_hsum(float2 a) -> float2 {
        #pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            a.x += __shfl_xor_sync(0xffffffff, a.x, mask, 32);
            a.y += __shfl_xor_sync(0xffffffff, a.y, mask, 32);
        }
        return a;
    }

    [[nodiscard]] static __device__ __forceinline__ auto warp_hsum_max(float x) -> float {
        #pragma unroll
        for (int mask = 16; mask; mask >>= 1)
            x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, mask, 32));
        return x;
    }

    struct physical_device final {
        int id {}; // Device ID
        std::size_t vram_total {}; // Total video memory
        std::uint32_t cl {}; // Compute capability
        std::uint32_t nsm {}; // Number of SMs
        std::uint32_t ntpb {}; // Number of threads per block
        std::size_t smpb {}; // Shared memory per block
        bool has_vmm {}; // Has virtual memory management
        std::size_t vmm_granularity {}; // Virtual memory management granularity
    };

    struct physical_devices final {
        std::array<physical_device, max_devices> devices {};
        std::size_t num_devices {};
    };

    /* Memory pool interface. */
    class pool {
    public:
        pool(const pool&) = delete;
        pool(pool&&) = delete;
        auto operator=(const pool&) -> pool& = delete;
        auto operator=(pool&&) -> pool& = delete;
        virtual ~pool() = default;

        virtual auto alloc(std::size_t sz, std::size_t align, std::size_t& out_sz) -> void* = 0;
        virtual auto free(void* ptr, std::size_t sz) -> void = 0;

    protected:
        pool() = default;
    };

    /* Memory pool allocating virtual memory pages. */
    class vm_pool final : public pool {
    public:
        static constexpr std::size_t max_size = 64ull<<30; /* 64 GiB */

        explicit vm_pool(const physical_device& dvc);
        ~vm_pool() override;

        auto alloc(std::size_t sz, std::size_t align, std::size_t& out_sz) -> void* override;
        auto free(void* ptr, std::size_t sz) -> void override;

    private:
        CUdeviceptr m_dvc_address {};
        int m_dvc_id {};
        std::size_t m_needle {};
        std::size_t m_cap {};
        std::size_t m_granularity {};
    };
}
