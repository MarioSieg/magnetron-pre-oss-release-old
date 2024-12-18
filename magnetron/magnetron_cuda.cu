/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include "magnetron_cuda.cuh"

#include <bit>
#include <cstdio>

namespace mag::cuda {
    extern "C" [[noreturn]] auto mag_panic(const char* msg, ...) -> void;

    /* Driver result check. */
    #define mag_cu_chk_rdv(expr) \
        do { \
        if (auto rrr {(expr)}; rrr != CUDA_SUCCESS) [[unlikely]] { \
            const char* err_str = "?"; \
            cuGetErrorString(rrr, &err_str); \
            mag_panic(#expr, __func__, __FILE__, __LINE__, err_str); \
        } \
    } while (0)

    /* Runtime result check. */
    #define mag_cu_chk_rt(expr) \
        do { \
            if (auto rrr {(expr)}; rrr != cudaSuccess) [[unlikely]] { \
                mag_panic(#expr, __func__, __FILE__, __LINE__, cudaGetErrorString(rrr)); \
            } \
        } while (0)

    #define mag_cu_assert(expr, msg, ...) \
        if (!(expr)) [[unlikely]] { \
            mag_panic("%s:%d Assertion failed: " #expr " <- " msg, __FILE__, __LINE__, ## __VA_ARGS__);\
        }
    #define mag_cu_assert2(expr) mag_cu_assert(expr, "")

    static constinit bool s_is_init {};
    static constinit std::array<physical_device, max_devices> s_devices {};
    static constinit std::int32_t s_num_devices {};

    auto mag_init_device_cuda([[maybe_unused]] mag_ctx_t* ctx) -> mag_compute_device_t* {
        std::int32_t active_device_id {0}; // TODO: Implement device selection.
        std::span<const physical_device> devices {cuda_init()};
        if (devices.empty()) { /* No devices available or initialization failed, let runtime fallback to other compute device. */
            mag_log_error("No CUDA devices available, using CPU processing");
            return nullptr; /* Return null device. */
        }
        if (active_device_id < 0 || active_device_id >= devices.size()) {
            mag_log_error("Invalid device ID %d, using device 0", active_device_id);
            active_device_id = 0;
        }
        auto* dvc {static_cast<mag_compute_device_t*>((*mag_alloc)(nullptr, sizeof(mag_compute_device_t)))};
        set_active_device_by_id(active_device_id);
        const auto& active_dvc {get_active_device()};
        std::snprintf(dvc->name, sizeof(dvc->name), "%s", active_dvc.name.data());
        return dvc;
    }

    void mag_destroy_device_cuda(mag_compute_device_t* dvc) {
        (*mag_alloc)(dvc, 0);
    }

    /*
    ** Initialize CUDA runtime. Returns empty span if initialization failed or not devices are available.
    ** Normally, we panic when some CUDA runtime function fails, but in this case we just return an empty span,
    ** to allow the runtime to fall back to CPU processing, if no CUDA devices are available.
    */
    auto cuda_init() -> std::span<const physical_device> {
        if (s_is_init) return {s_devices.data(), static_cast<std::size_t>(s_num_devices)};
        if (cudaGetDeviceCount(&s_num_devices) != cudaSuccess) [[unlikely]] return {};
        if (!(s_num_devices && s_num_devices <= max_devices)) [[unlikely]] return {};
        for (std::int32_t id {}; id < s_num_devices; ++id) { /* Iterate over devices */
            physical_device& dvc {s_devices[id]};
            CUdevice cu_dvc {};
            std::int32_t vmm_support {};
            if (cuDeviceGet(&cu_dvc, id) != CUDA_SUCCESS) [[unlikely]] continue; /* Get device handle */
            if (cuDeviceGetAttribute(&vmm_support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu_dvc) != CUDA_SUCCESS) [[unlikely]] continue; /* Check VMM support */
            if (vmm_support) { /* Virtual memory management supported */
                CUmemAllocationProp alloc_props {};
                alloc_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                alloc_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                alloc_props.location.id = id;
                if (cuMemGetAllocationGranularity(&dvc.vmm_granularity, &alloc_props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) [[unlikely]] continue; /* Get VMM granularity */
            }
            dvc.has_vmm = !!vmm_support;
            cudaDeviceProp props {};
            if (cudaGetDeviceProperties(&props, id) != cudaSuccess) [[unlikely]] continue; /* Get device properties */
            dvc.id = id;
            dvc.name = std::bit_cast<decltype(dvc.name)>(props.name);
            dvc.nsm = props.multiProcessorCount;
            dvc.smpb = props.sharedMemPerBlock;
            dvc.smpb_opt = props.sharedMemPerBlockOptin;
            dvc.cl = 100*props.major + 10*props.minor;
            dvc.ntpb = props.maxThreadsPerBlock;
            dvc.vram = props.totalGlobalMem;
        }
        s_is_init = true;
        return {s_devices.data(), static_cast<std::size_t>(s_num_devices)};
    }

    auto set_active_device_by_id(std::int32_t id) -> void {
        std::int32_t curr;
        mag_cu_chk_rt(cudaGetDevice(&curr));
        if (curr == id) return; /* Already active */
        mag_cu_chk_rt(cudaSetDevice(id));
    }

    auto get_active_device_id() -> std::int32_t {
        std::int32_t curr;
        mag_cu_chk_rt(cudaGetDevice(&curr));
        return curr;
    }

    auto get_active_device() -> const physical_device& {
        const auto id {get_active_device_id()}; /* Ensure device is active. */
        mag_cu_assert2(id >= 0 && id < s_num_devices);
        return s_devices[id];
    }

    vm_pool::vm_pool(const physical_device& dvc) {
        m_granularity = dvc.vmm_granularity;
        m_dvc_id = dvc.id;
    }

    vm_pool::~vm_pool() {
        if (m_dvc_address) {
            mag_cu_chk_rdv(cuMemUnmap(m_dvc_address, m_cap));
            mag_cu_chk_rdv(cuMemAddressFree(m_dvc_address, max_size));
        }
    }

    auto vm_pool::alloc(std::size_t sz, std::size_t align, std::size_t& out_sz) -> void* {
        sz = (sz+align-1)&-align; /* Overallocate to ensure alignment. */
        if (std::size_t free {m_cap-m_needle}; sz > free) {
            std::size_t reserve {sz-free};
            reserve = (reserve+m_granularity-1)&-m_granularity; /* Align to granularity. */
            mag_cu_assert2(m_cap+reserve <= max_size);
            CUmemAllocationProp prop {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = m_dvc_id;
            CUmemGenericAllocationHandle handle {};
            mag_cu_chk_rdv(cuMemCreate(&handle, reserve, &prop, 0));
            if (!m_dvc_address) /* Reserve virtual memory */
                mag_cu_chk_rdv(cuMemAddressReserve(&m_dvc_address, max_size, 0, 0, 0));
            mag_cu_chk_rdv(cuMemMap(m_dvc_address+m_cap, reserve, 0, handle, 0));
            mag_cu_chk_rdv(cuMemRelease(handle)); /* Release handle */
            CUmemAccessDesc access {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = m_dvc_id;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            mag_cu_chk_rdv(cuMemSetAccess(m_dvc_address+m_cap, reserve, &access, 1)); /* Set access */
            m_cap += reserve;
        }
        mag_cu_assert2(m_dvc_address);
        auto* p {reinterpret_cast<void*>(m_dvc_address+m_needle)};
        out_sz = sz;
        m_needle += sz;
        return p;
    }

    auto vm_pool::free(void* ptr, std::size_t sz) -> void {
        m_needle -= sz;
        mag_cu_assert2(ptr == reinterpret_cast<void*>(m_dvc_address + m_needle));
    }
}
