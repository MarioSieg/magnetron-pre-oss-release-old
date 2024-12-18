/* (c) 2024 Mario "Neo" Sieg. <mario.sieg.64@gmail.com> */

#ifndef MAGNETRON_CPU_H
#define MAGNETRON_CPU_H

#include "magnetron_internal.h"

extern mag_compute_device_t* mag_init_device_cpu(mag_ctx_t* ctx, uint32_t num_threads); /* Initialize CPU compute device. num_threads = 0 = num of host CPUs */
extern void mag_destroy_device_cpu(mag_compute_device_t* dvc); /* Destroy CPU compute device. */

#endif
