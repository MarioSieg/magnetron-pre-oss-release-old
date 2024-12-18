// (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

#ifdef MAG_ENABLE_CUDA

#include "prelude.hpp"
#include "magnetron_cuda.cuh"

using namespace mag::cuda;

TEST(cuda, init) {
    for (auto&& dev : cuda_init()) {
        std::cout << "Device ID: " << dev.id << std::endl;
        std::cout << "Device Name: " << dev.name.data() << std::endl;
        std::cout << "Video Memory: " << static_cast<double>(dev.vram) / 1e9 << " GB" << std::endl;
        std::cout << "Compute Capability: " << dev.cl << std::endl;
        std::cout << "Number of SMs: " << dev.nsm << std::endl;
        std::cout << "Number of Threads per Block: " << dev.ntpb << std::endl;
        std::cout << "Shared Memory per Block: " << static_cast<double>(dev.smpb) / 1e3 << " KB" << std::endl;
        std::cout << "Shared Memory per Block Opt-In: " << static_cast<double>(dev.smpb_opt) / 1e3 << " KB" << std::endl;
        std::cout << "Has Virtual Memory Management: " << dev.has_vmm << std::endl;
        std::cout << "Virtual Memory Management Granularity: " << static_cast<double>(dev.vmm_granularity) / 1e3 << " KB" << std::endl;
    }
}

TEST(cuda, vm_pool) {
    vm_pool pool {cuda_init()[0]};
    std::size_t actual_sz = 0;
    auto* ptr = pool.alloc(1024, 32, actual_sz);
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(std::bit_cast<std::uintptr_t>(ptr) % 32, 0);
    std::cout << "Allocated: " << actual_sz << " bytes" << std::endl;
    pool.free(ptr, actual_sz);
}

#endif
