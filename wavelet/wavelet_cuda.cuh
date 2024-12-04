/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#pragma once

#include "wavelet_internal.h"

#include <array>
#include <cstdint>
#include <span>

#include <cuda.h>

namespace wl::cuda {
    extern "C" auto wl__init_device_cuda(wl_ctx_t* ctx) -> wl__icompute_device_t*; /* Initialize GPU compute device. (Invoked from wavelet C core) */
    extern "C" auto wl__destroy_device_cuda(wl__icompute_device_t* dvc) -> void; /* Destroy GPU compute device. (Invoked from wavelet C core) */

    constexpr std::size_t max_devices = 32;
    constexpr std::uint32_t warp_size = 32;
    constexpr std::uint32_t max_streams = 8;

    struct physical_device final {
        std::int32_t id {};             /* Device ID */
        std::array<char, 256> name {};  /* Device name */
        std::size_t vram {};            /* Video memory in bytes */
        std::uint32_t cl {};            /* Compute capability */
        std::uint32_t nsm {};           /* Number of SMs */
        std::uint32_t ntpb {};          /* Number of threads per block */
        std::size_t smpb {};            /* Shared memory per block */
        std::size_t smpb_opt {};        /* Shared memory per block opt-in */
        bool has_vmm {};                /* Has virtual memory management */
        std::size_t vmm_granularity {}; /* Virtual memory management granularity */
    };

    /* Initialize CUDA runtime. Returns empty span if initialization failed or not devices are available. */
    [[nodiscard]] extern auto cuda_init() -> std::span<const physical_device>;

    extern auto set_active_device_by_id(std::int32_t id) -> void;
    extern auto get_active_device_id() -> std::int32_t;
    extern auto get_active_device() -> const physical_device&;

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
        std::int32_t m_dvc_id {};
        std::size_t m_needle {};
        std::size_t m_cap {};
        std::size_t m_granularity {};
    };
}
