/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include "wavelet.cuh"

#include <bit>
#include <cassert>

namespace wl::cuda {
    auto cuda_init() -> std::span<const physical_device> {
        static constinit bool is_init {};
        static constinit std::array<physical_device, max_devices> devices {};
        static constinit std::int32_t num_devices {};
        if (is_init) return {devices.data(), static_cast<std::size_t>(num_devices)};
        wl_cu_chk_rt(cudaGetDeviceCount(&num_devices));
        assert(num_devices && num_devices <= max_devices);
        for (std::int32_t id {}; id < num_devices; ++id) { /* Iterate over devices */
            physical_device& dvc {devices[id]};
            CUdevice cu_dvc {};
            std::int32_t vmm_support {};
            wl_cu_chk_rdv(cuDeviceGet(&cu_dvc, id));
            wl_cu_chk_rdv(cuDeviceGetAttribute(&vmm_support, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu_dvc));
            if (vmm_support) { /* Virtual memory management supported */
                CUmemAllocationProp alloc_props {};
                alloc_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                alloc_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                alloc_props.location.id = id;
                wl_cu_chk_rdv(cuMemGetAllocationGranularity(&dvc.vmm_granularity, &alloc_props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
            }
            dvc.has_vmm = !!vmm_support;
            cudaDeviceProp props {};
            wl_cu_chk_rt(cudaGetDeviceProperties(&props, id)); /* Get device properties */
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
            wl_cu_chk_rdv(cuMemUnmap(m_dvc_address, m_cap));
            wl_cu_chk_rdv(cuMemAddressFree(m_dvc_address, max_size));
        }
    }

    auto vm_pool::alloc(std::size_t sz, std::size_t align, std::size_t& out_sz) -> void* {
        sz = (sz+align-1)&-align; /* Overallocate to ensure alignment. */
        if (std::size_t free {m_cap-m_needle}; sz > free) {
            std::size_t reserve {sz-free};
            reserve = (reserve+m_granularity-1)&-m_granularity; /* Align to granularity. */
            assert(m_cap+reserve <= max_size);
            CUmemAllocationProp prop {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = m_dvc_id;
            CUmemGenericAllocationHandle handle {};
            wl_cu_chk_rdv(cuMemCreate(&handle, reserve, &prop, 0));
            if (!m_dvc_address) /* Reserve virtual memory */
                wl_cu_chk_rdv(cuMemAddressReserve(&m_dvc_address, max_size, 0, 0, 0));
            wl_cu_chk_rdv(cuMemMap(m_dvc_address+m_cap, reserve, 0, handle, 0));
            wl_cu_chk_rdv(cuMemRelease(handle)); /* Release handle */
            CUmemAccessDesc access {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = m_dvc_id;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            wl_cu_chk_rdv(cuMemSetAccess(m_dvc_address+m_cap, reserve, &access, 1)); /* Set access */
            m_cap += reserve;
        }
        assert(m_dvc_address);
        auto* p {reinterpret_cast<void*>(m_dvc_address+m_needle)};
        out_sz = sz;
        m_needle += sz;
        return p;
    }

    auto vm_pool::free(void* ptr, std::size_t sz) -> void {
        m_needle -= sz;
        assert(ptr == reinterpret_cast<void*>(m_dvc_address + m_needle));
    }
}
