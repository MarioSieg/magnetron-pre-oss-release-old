/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include "wavelet_cuda.cuh"

#include <bit>
#include <cstdio>

namespace wl::cuda {
    extern "C" [[noreturn]] auto wl__panic(const char* msg, ...) -> void;

    /* Driver result check. */
    #define wl__cu_chk_rdv(expr) \
        do { \
        if (auto rrr {(expr)}; rrr != CUDA_SUCCESS) [[unlikely]] { \
            const char* err_str = "?"; \
            cuGetErrorString(rrr, &err_str); \
            wl__panic(#expr, __func__, __FILE__, __LINE__, err_str); \
        } \
    } while (0)

    /* Runtime result check. */
    #define wl__cu_chk_rt(expr) \
        do { \
            if (auto rrr {(expr)}; rrr != cudaSuccess) [[unlikely]] { \
                wl__panic(#expr, __func__, __FILE__, __LINE__, cudaGetErrorString(rrr)); \
            } \
        } while (0)

    #define wl__cu_assert(expr, msg, ...) \
        if (!(expr)) [[unlikely]] { \
            wl__panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
        }
    #define wl__cu_assert2(expr) wl__cu_assert(expr, "")

    auto wl__init_device_cuda(wl_ctx_t* ctx) -> wl__compute_device_t* {
        std::span<const physical_device> devices {cuda_init()};
        auto* dvc {static_cast<wl__compute_device_t*>((*wl__alloc)(nullptr, sizeof(wl__compute_device_t)))};
        const auto& active_dvc {devices[0]};
        std::snprintf(dvc->name, sizeof(dvc->name), "%s", active_dvc.name.data());
        return dvc;
    }

    void wl__destroy_device_cuda(wl__compute_device_t* dvc) {
        (*wl__alloc)(dvc, 0);
    }

    auto cuda_init() -> std::span<const physical_device> {
        static constinit bool is_init {};
        static constinit std::array<physical_device, max_devices> devices {};
        static constinit std::int32_t num_devices {};
        if (is_init) return {devices.data(), static_cast<std::size_t>(num_devices)};
        wl__cu_chk_rt(cudaGetDeviceCount(&num_devices));
        wl__cu_assert2(num_devices && num_devices <= max_devices);
        for (std::int32_t id {}; id < num_devices; ++id) { /* Iterate over devices */
            physical_device& dvc {devices[id]};
            CUdevice cu_dvc {};
            std::int32_t vmm_support {};
            wl__cu_chk_rdv(cuDeviceGet(&cu_dvc, id));
            wl__cu_chk_rdv(cuDeviceGetAttribute(&vmm_support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu_dvc));
            if (vmm_support) { /* Virtual memory management supported */
                CUmemAllocationProp alloc_props {};
                alloc_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                alloc_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                alloc_props.location.id = id;
                wl__cu_chk_rdv(cuMemGetAllocationGranularity(&dvc.vmm_granularity, &alloc_props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
            }
            dvc.has_vmm = !!vmm_support;
            cudaDeviceProp props {};
            wl__cu_chk_rt(cudaGetDeviceProperties(&props, id)); /* Get device properties */
            dvc.id = id;
            dvc.name = std::bit_cast<decltype(dvc.name)>(props.name);
            dvc.nsm = props.multiProcessorCount;
            dvc.smpb = props.sharedMemPerBlock;
            dvc.smpb_opt = props.sharedMemPerBlockOptin;
            dvc.cl = 100*props.major + 10*props.minor;
            dvc.ntpb = props.maxThreadsPerBlock;
            dvc.vram = props.totalGlobalMem;
        }
        is_init = true;
        return {devices.data(), static_cast<std::size_t>(num_devices)};
    }

    vm_pool::vm_pool(const physical_device& dvc) {
        m_granularity = dvc.vmm_granularity;
        m_dvc_id = dvc.id;
    }

    vm_pool::~vm_pool() {
        if (m_dvc_address) {
            wl__cu_chk_rdv(cuMemUnmap(m_dvc_address, m_cap));
            wl__cu_chk_rdv(cuMemAddressFree(m_dvc_address, max_size));
        }
    }

    auto vm_pool::alloc(std::size_t sz, std::size_t align, std::size_t& out_sz) -> void* {
        sz = (sz+align-1)&-align; /* Overallocate to ensure alignment. */
        if (std::size_t free {m_cap-m_needle}; sz > free) {
            std::size_t reserve {sz-free};
            reserve = (reserve+m_granularity-1)&-m_granularity; /* Align to granularity. */
            wl__cu_assert2(m_cap+reserve <= max_size);
            CUmemAllocationProp prop {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = m_dvc_id;
            CUmemGenericAllocationHandle handle {};
            wl__cu_chk_rdv(cuMemCreate(&handle, reserve, &prop, 0));
            if (!m_dvc_address) /* Reserve virtual memory */
                wl__cu_chk_rdv(cuMemAddressReserve(&m_dvc_address, max_size, 0, 0, 0));
            wl__cu_chk_rdv(cuMemMap(m_dvc_address+m_cap, reserve, 0, handle, 0));
            wl__cu_chk_rdv(cuMemRelease(handle)); /* Release handle */
            CUmemAccessDesc access {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = m_dvc_id;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            wl__cu_chk_rdv(cuMemSetAccess(m_dvc_address+m_cap, reserve, &access, 1)); /* Set access */
            m_cap += reserve;
        }
        wl__cu_assert2(m_dvc_address);
        auto* p {reinterpret_cast<void*>(m_dvc_address+m_needle)};
        out_sz = sz;
        m_needle += sz;
        return p;
    }

    auto vm_pool::free(void* ptr, std::size_t sz) -> void {
        m_needle -= sz;
        wl__cu_assert2(ptr == reinterpret_cast<void*>(m_dvc_address + m_needle));
    }
}
