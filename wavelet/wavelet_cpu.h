/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#ifndef WAVELET_CPU_H
#define WAVELET_CPU_H

#include "wavelet_internal.h"

extern wl__compute_device_t* wl__cpu_init(wl_ctx_t* ctx, uint32_t num_threads); /* Initialize CPU compute device. num_threads = 0 = num of host CPUs */
extern void wl__cpu_destroy(wl__compute_device_t* dvc); /* Destroy CPU compute device. */

#endif
