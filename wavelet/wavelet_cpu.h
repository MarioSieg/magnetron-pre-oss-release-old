/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#ifndef WAVELET_CPU_H
#define WAVELET_CPU_H

#include "wavelet_internal.h"

extern wl__icompute_device_t* wl__init_device_cpu(wl_ctx_t* ctx, uint32_t num_threads); /* Initialize CPU compute device. num_threads = 0 = num of host CPUs */
extern void wl__destroy_device_cpu(wl__icompute_device_t* dvc); /* Destroy CPU compute device. */

#endif
