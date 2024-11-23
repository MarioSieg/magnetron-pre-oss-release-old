/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#include "wavelet.cuh"

namespace wl {
    vm_pool::vm_pool(const physical_device& dvc) {
        m_granularity = dvc.vmm_granularity;
        m_dvc_id = dvc.id;
    }

    vm_pool::~vm_pool() {
        if (m_dvc_address) {
            wl_cu_chk(cuMemUnmap(m_dvc_address, m_cap));
            wl_cu_chk(cuMemAddressFree(m_dvc_address, max_size));
        }
    }
}
